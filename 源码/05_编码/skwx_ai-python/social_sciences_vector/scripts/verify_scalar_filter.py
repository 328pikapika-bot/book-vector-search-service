#!/usr/bin/env python
"""Verify one existing full-book partition and its scalar array filter."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import mysql.connector
from pymilvus import Collection, connections, utility


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-id", required=True)
    parser.add_argument("--mysql-host", default="127.0.0.1")
    parser.add_argument("--mysql-port", type=int, default=13306)
    parser.add_argument("--mysql-user", default="root")
    parser.add_argument("--mysql-password", required=True)
    parser.add_argument("--mysql-database", default="vector_benchmark")
    parser.add_argument("--milvus-host", default="127.0.0.1")
    parser.add_argument("--milvus-port", type=int, default=19531)
    parser.add_argument("--expected-code", default="51")
    parser.add_argument("--missing-code", default="999")
    parser.add_argument("--expected-rows", type=int, default=652)
    parser.add_argument("--output")
    args = parser.parse_args()

    connection = mysql.connector.connect(
        host=args.mysql_host,
        port=args.mysql_port,
        user=args.mysql_user,
        password=args.mysql_password,
        database=args.mysql_database,
        connection_timeout=10,
    )
    try:
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT vc.book_id, vc.name AS collection_name, vc.partition_name,
                   vlb.name AS milvus_database
            FROM vector_collection vc
            JOIN vector_library vl ON vc.library_id = vl.id
            JOIN vector_library_base vlb ON vl.library_id = vlb.id
            WHERE vc.partition_name=%s
            LIMIT 1
            """,
            (f"book_{args.book_id}",),
        )
        metadata = cursor.fetchone()
        if not metadata:
            raise RuntimeError(f"Missing MySQL metadata for book_{args.book_id}")
    finally:
        connection.close()

    alias = f"scalar_verify_{datetime.now().strftime('%H%M%S%f')}"
    connections.connect(
        alias=alias,
        host=args.milvus_host,
        port=args.milvus_port,
        db_name=metadata["milvus_database"],
    )
    try:
        if not utility.has_collection(metadata["collection_name"], using=alias):
            raise RuntimeError(f"Missing Milvus collection: {metadata['collection_name']}")
        collection = Collection(metadata["collection_name"], using=alias)
        collection.load()
        field_names = {field.name for field in collection.schema.fields}
        partition_names = {partition.name for partition in collection.partitions}
        count_rows = collection.query(
            expr=f'book_id == "{args.book_id}"',
            output_fields=["count(*)"],
            partition_names=[metadata["partition_name"]],
        )
        row_count = int(count_rows[0].get("count(*)", 0)) if count_rows else 0
        expected_hits = collection.query(
            expr=(
                f'book_id == "{args.book_id}" and '
                f'ARRAY_CONTAINS(publish_direct, "{args.expected_code}")'
            ),
            output_fields=["book_id", "publish_direct"],
            partition_names=[metadata["partition_name"]],
            limit=5,
        )
        missing_hits = collection.query(
            expr=(
                f'book_id == "{args.book_id}" and '
                f'ARRAY_CONTAINS(publish_direct, "{args.missing_code}")'
            ),
            output_fields=["book_id", "publish_direct"],
            partition_names=[metadata["partition_name"]],
            limit=5,
        )
        required_fields = {
            "publish_direct",
            "chunk_index",
            "chunk_total",
            "chunk_start",
            "chunk_end",
            "chunk_strategy",
        }
        checks = {
            "mysql_book_id_matches": metadata["book_id"] == args.book_id,
            "partition_exists": metadata["partition_name"] in partition_names,
            "field_count_is_58": len(field_names) == 58,
            "required_fields_present": required_fields.issubset(field_names),
            "row_count_matches": row_count == args.expected_rows,
            "expected_filter_hit": len(expected_hits) > 0,
            "missing_filter_empty": len(missing_hits) == 0,
        }
        result = {
            "success": all(checks.values()),
            "checked_at": datetime.now().isoformat(),
            "book_id": args.book_id,
            "database": metadata["milvus_database"],
            "collection": metadata["collection_name"],
            "partition": metadata["partition_name"],
            "field_count": len(field_names),
            "row_count": row_count,
            "expected_filter": f'ARRAY_CONTAINS(publish_direct, "{args.expected_code}")',
            "expected_filter_hits": len(expected_hits),
            "missing_filter": f'ARRAY_CONTAINS(publish_direct, "{args.missing_code}")',
            "missing_filter_hits": len(missing_hits),
            "checks": checks,
        }
    finally:
        connections.disconnect(alias)

    serialized = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
