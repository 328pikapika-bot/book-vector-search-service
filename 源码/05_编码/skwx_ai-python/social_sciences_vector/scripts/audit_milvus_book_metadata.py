#!/usr/bin/env python3
"""Read-only audit for book_id and title in Milvus collections.

Example:
  python3 audit_milvus_book_metadata.py --host 127.0.0.1 --db ssc_1 \
      --collection book_partition_1 --collection book_partition_2 --verify-values

The script never creates, loads, releases, deletes, or changes Milvus data.
Value verification runs only for collections that are already Loaded.
"""

import argparse
import json
import sys

from pymilvus import Collection, connections, utility


BOOK_FIELDS = ("book_id", "title")


def load_state_name(collection_name, alias):
    state = utility.load_state(collection_name, using=alias, timeout=10)
    return getattr(state, "name", str(state))


def audit_collection(collection_name, alias, verify_values, batch_size):
    collection = Collection(collection_name, using=alias)
    field_names = {field.name for field in collection.schema.fields}
    state = load_state_name(collection_name, alias)
    partitions = [
        partition
        for partition in collection.partitions
        if partition.name != "_default" and partition.num_entities > 0
    ]

    result = {
        "collection": collection_name,
        "load_state": state,
        "field_count": len(field_names),
        "has_book_id": "book_id" in field_names,
        "has_title": "title" in field_names,
        "nonempty_partition_count": len(partitions),
        "partition_entity_count": sum(partition.num_entities for partition in partitions),
    }

    if not verify_values:
        return result
    if not all(field in field_names for field in BOOK_FIELDS):
        result["value_check"] = "skipped: book_id or title does not exist in schema"
        return result
    if state != "Loaded":
        result["value_check"] = "skipped: collection is not Loaded"
        return result

    checked = missing_book_id = missing_title = partition_mismatch = 0
    examples = []
    for partition in partitions:
        expected_book_id = partition.name.removeprefix("book_")
        iterator = collection.query_iterator(
            batch_size=batch_size,
            limit=-1,
            expr="",
            output_fields=list(BOOK_FIELDS),
            partition_names=[partition.name],
            timeout=30,
        )
        try:
            while True:
                rows = iterator.next()
                if not rows:
                    break
                for row in rows:
                    checked += 1
                    book_id = str(row.get("book_id") or "").strip()
                    title = str(row.get("title") or "").strip()
                    invalid = False
                    if not book_id:
                        missing_book_id += 1
                        invalid = True
                    if not title:
                        missing_title += 1
                        invalid = True
                    if book_id != expected_book_id:
                        partition_mismatch += 1
                        invalid = True
                    if invalid and len(examples) < 20:
                        examples.append(
                            {
                                "partition": partition.name,
                                "book_id": book_id,
                                "title": title,
                            }
                        )
        finally:
            iterator.close()

    result.update(
        {
            "value_check": "completed",
            "rows_checked": checked,
            "missing_book_id": missing_book_id,
            "missing_title": missing_title,
            "partition_book_id_mismatch": partition_mismatch,
            "invalid_examples": examples,
        }
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Audit book metadata in Milvus.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default="19530")
    parser.add_argument("--db", required=True, help="Milvus database name, for example ssc_1")
    parser.add_argument("--collection", action="append", dest="collections")
    parser.add_argument("--verify-values", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1000)
    args = parser.parse_args()

    alias = "book_metadata_audit"
    connections.connect(alias=alias, host=args.host, port=args.port, db_name=args.db, timeout=10)
    collection_names = args.collections or sorted(utility.list_collections(using=alias))

    results = []
    for collection_name in collection_names:
        try:
            results.append(audit_collection(collection_name, alias, args.verify_values, args.batch_size))
        except Exception as exc:
            results.append(
                {
                    "collection": collection_name,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    print(json.dumps({"database": args.db, "collections": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
