#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run a realistic full-book vector import test.

Flow:
1. Read the raw decrypted book JSON.
2. Copy it into run-logs/mock-minio to simulate upstream object storage.
3. Convert the raw book JSON into the vector service payload.
4. POST the payload to /up_collection/ImportFromText.
5. Verify the new Milvus collection schema and publish_direct filtering.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
VECTOR_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[5] if len(SCRIPT_PATH.parents) > 5 else VECTOR_ROOT
RUN_LOGS = PROJECT_ROOT / "run-logs"

sys.path.insert(0, str(VECTOR_ROOT))


def value_or_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict) and "time" in value:
        try:
            return datetime.fromtimestamp(int(value["time"]) / 1000).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return str(value.get("time", ""))
    return str(value)


def value_or_zero(value: Any) -> int:
    try:
        if value is None or value == "":
            return 0
        return int(value)
    except Exception:
        return 0


def normalize_array(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, dict):
        parts = [value.get("CommonID") or value.get("Name") or value.get("name") or value]
    elif isinstance(value, (list, tuple, set)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                parts.append(item.get("CommonID") or item.get("Name") or item.get("name") or item.get("value") or item)
            else:
                parts.append(item)
    else:
        parts = [value]

    result: list[str] = []
    for item in parts:
        text = str(item).strip()
        if text and text not in result:
            result.append(text)
    return result


def collect_author_fields(raw_authors: Any) -> dict[str, list[Any]]:
    fields = {
        "author_name": [],
        "author_id": [],
        "college": [],
        "org_name": [],
        "country": [],
        "position": [],
        "academic_title": [],
        "ethnic": [],
        "age": [],
        "disciplines": [],
        "research": [],
        "study_branch_id": [],
        "unit_address": [],
    }
    if not isinstance(raw_authors, list):
        return fields

    for item in raw_authors:
        if not isinstance(item, dict):
            continue
        author = item.get("AuthorInfo") or item.get("authorInfo") or {}
        resource = item.get("ResourceAuthor") or item.get("resourceAuthor") or {}
        fields["author_name"].append(author.get("Name"))
        fields["author_id"].append(
            resource.get("AuthorID")
            or resource.get("aID")
            or author.get("CommonID")
        )
        fields["college"].append(author.get("College"))
        fields["org_name"].append(author.get("OrgName"))
        fields["country"].append(author.get("Country"))
        fields["position"].append(author.get("Position"))
        fields["academic_title"].append(author.get("AcademicTitle"))
        fields["ethnic"].append(author.get("Ethnic"))
        fields["age"].append(value_or_zero(author.get("Age")))
        fields["disciplines"].append(author.get("Disciplines"))
        fields["research"].append(author.get("Research"))
        fields["study_branch_id"].append(author.get("StudyBranchID"))
        fields["unit_address"].append(author.get("Address"))

    for key, values in fields.items():
        if key == "age":
            fields[key] = [value_or_zero(v) for v in values if value_or_zero(v) > 0]
        else:
            fields[key] = normalize_array(values)
    return fields


def to_paragraph(
    raw_book: dict[str, Any],
    item: dict[str, Any],
    book_id: str,
    publish_direct: list[str],
    fallback_authors: dict[str, list[Any]],
) -> dict[str, Any] | None:
    book_info = raw_book.get("bookInfo") or {}
    detail = item.get("directoryDetail") or {}
    literature = item.get("literatureInfo") or {}
    text = value_or_empty(literature.get("ContentText"))
    if not text.strip():
        return None

    author_fields = collect_author_fields(item.get("DYAuthorInfoDetail"))
    if not author_fields["author_name"]:
        author_fields = fallback_authors

    directory_id = (
        detail.get("CommonID")
        or detail.get("OldID")
        or literature.get("CommonID")
        or literature.get("DirectoryInnerCode")
    )

    paragraph = {
        "paragraph_id": value_or_empty(literature.get("CommonID") or directory_id),
        "book_id": book_id,
        "directory_id": value_or_empty(directory_id),
        "text": text,
        "literature_id": value_or_empty(literature.get("CommonID") or detail.get("LiteratureId")),
        "directory_type": value_or_empty(detail.get("Type")),
        "directory_code": value_or_empty(detail.get("Code")),
        "directory_name": value_or_empty(detail.get("Name") or literature.get("Title")),
        "directory_en_title": value_or_empty(detail.get("EnTitle") or literature.get("EnTitle")),
        "directory_inner_code": value_or_empty(detail.get("InnerCode") or literature.get("DirectoryInnerCode")),
        "directory_search_title": value_or_empty(detail.get("SearchTitle") or literature.get("SearchTitle")),
        "directory_en_search_title": value_or_empty(detail.get("EnSearchTitle") or literature.get("EnSearchTitle")),
        "directory_level": value_or_zero(detail.get("Level")),
        "directory_keywords": normalize_array(detail.get("KeyWords") or item.get("DYKeyWords")),
        "directory_e_keywords": normalize_array(detail.get("EKeyWords") or item.get("DYENKeyWords")),
        "directory_subject_words": normalize_array(detail.get("SubjectWords") or item.get("DYThemeWords")),
        "directory_issue_date": value_or_empty(detail.get("IssueDate") or detail.get("PubDate")),
        "directory_doc_type": value_or_empty(detail.get("DocType")),
        "directory_product_series": value_or_empty(detail.get("ProductSeries")),
        "directory_doi": value_or_empty(detail.get("Doi")),
        "directory_min_node_search": value_or_empty(detail.get("MinNodeSearch")),
        "directory_hit_count": value_or_zero(detail.get("HitCount")),
        "directory_down_count": value_or_zero(detail.get("DownCount")),
        "directory_book_publish_date": value_or_empty(detail.get("BookPublishDate") or book_info.get("PublishDate")),
        "theme": ",".join(normalize_array(item.get("DYThemeWords") or raw_book.get("Themewords"))),
        "en_theme": "",
        "search_title": value_or_empty(literature.get("SearchTitle") or detail.get("SearchTitle")),
        "directory_inter_code": value_or_empty(detail.get("InnerCode") or literature.get("DirectoryInnerCode")),
        "isbn": value_or_empty(literature.get("ISBN") or book_info.get("ISBN")),
        "language": value_or_empty(literature.get("Language") or book_info.get("Language")),
        "type": value_or_empty(literature.get("Type") or book_info.get("Type") or book_info.get("BookType")),
        "content_text": text,
        "publish_time": value_or_empty(literature.get("PublishTime") or book_info.get("PublishTime")),
        "en_search_title": value_or_empty(literature.get("EnSearchTitle")),
        "price": value_or_empty(literature.get("Price") or book_info.get("EbookPrice")),
        "research_region_values": [],
        "research_area_values": [],
        "publish_direct": publish_direct,
    }
    paragraph.update(author_fields)
    return paragraph


def build_payload(raw_book: dict[str, Any], book_id: str, chunk_size: int, force_new_collection: bool) -> dict[str, Any]:
    book_info = raw_book.get("bookInfo") or {}
    title = value_or_empty(book_info.get("Title") or raw_book.get("BookCommonID") or raw_book.get("BookID"))
    publish_direct = normalize_array(book_info.get("PublishDirect"))
    fallback_authors = collect_author_fields(raw_book.get("authorInfoDetail"))

    paragraphs = []
    for item in raw_book.get("directoryInfo") or []:
        paragraph = to_paragraph(raw_book, item, book_id, publish_direct, fallback_authors)
        if paragraph:
            paragraphs.append(paragraph)

    return {
        "title": f"{title}-full-chain-test-{book_id}",
        "book_id": book_id,
        "max_text_total": str(chunk_size),
        "vector_model": "m3e-base",
        "semantic_model": "",
        "force_new_collection": force_new_collection,
        "paragraphs": paragraphs,
    }


def post_json(url: str, payload: dict[str, Any], timeout: int) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = response.read().decode("utf-8")
    return json.loads(text)


def query_metadata(book_id: str) -> dict[str, Any]:
    import mysql.connector
    from config import mysql_database, mysql_host, mysql_password, mysql_user

    conn = mysql.connector.connect(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database,
    )
    try:
        cursor = conn.cursor(dictionary=True)
        cursor.execute(
            """
            SELECT vc.name AS collection_name, vc.partition_name, vc.library_id,
                   vl.library_id AS library_base_id, vl.id AS vector_library_id,
                   vl.name AS vector_library_name, vl.partition_count,
                   vlb.name AS milvus_db_name
            FROM vector_collection vc
            JOIN vector_library vl ON vc.library_id = vl.id
            JOIN vector_library_base vlb ON vl.library_id = vlb.id
            WHERE vc.partition_name = %s
            ORDER BY vc.create_time DESC
            LIMIT 1
            """,
            (f"book_{book_id}",),
        )
        row = cursor.fetchone()
        if not row:
            raise RuntimeError(f"Cannot find vector_collection row for book_{book_id}")
        return row
    finally:
        conn.close()


def verify_milvus(metadata: dict[str, Any], book_id: str, expect_code: str, miss_code: str) -> dict[str, Any]:
    from pymilvus import Collection, utility
    from national_development.milvus_connection import connect_milvus

    milvus_alias = connect_milvus(metadata["milvus_db_name"])
    collection_name = metadata["collection_name"]
    partition_name = metadata["partition_name"]
    if not utility.has_collection(collection_name, using=milvus_alias):
        raise RuntimeError(f"Milvus collection does not exist: {collection_name}")

    collection = Collection(collection_name, using=milvus_alias)
    collection.load()
    fields = [
        {
            "name": field.name,
            "type": str(field.dtype),
            "is_vector": field.name == "embedding",
            "params": field.params,
        }
        for field in collection.schema.fields
    ]
    field_names = {field["name"] for field in fields}
    chunk_fields = ["chunk_index", "chunk_total", "chunk_start", "chunk_end", "chunk_strategy"]
    has_publish_direct = "publish_direct" in field_names
    has_chunk_fields = all(field in field_names for field in chunk_fields)

    sample = collection.query(
        expr=f'book_id == "{book_id}"',
        output_fields=["book_id", "directory_id", "publish_direct", "text"] + chunk_fields,
        partition_names=[partition_name],
        limit=3,
    )
    count_rows = collection.query(
        expr=f'book_id == "{book_id}"',
        output_fields=["count(*)"],
        partition_names=[partition_name],
    )
    row_count = count_rows[0].get("count(*)") if count_rows else None
    hit_rows = collection.query(
        expr=f'book_id == "{book_id}" and ARRAY_CONTAINS(publish_direct, "{expect_code}")',
        output_fields=["book_id", "directory_id", "publish_direct"],
        partition_names=[partition_name],
        limit=5,
    )
    miss_rows = collection.query(
        expr=f'book_id == "{book_id}" and ARRAY_CONTAINS(publish_direct, "{miss_code}")',
        output_fields=["book_id", "directory_id", "publish_direct"],
        partition_names=[partition_name],
        limit=5,
    )
    analysis_rows = collection.query(
        expr=f'book_id == "{book_id}"',
        output_fields=["directory_id", "text"] + chunk_fields,
        partition_names=[partition_name],
        limit=10000,
    )

    strategy_distribution: dict[str, int] = {}
    html_remnant_count = 0
    total_text_length = 0
    directory_samples: dict[str, list[dict[str, Any]]] = {
        "DIR_73319731": [],
        "DIR_73319785": [],
    }
    for row in analysis_rows:
        strategy = row.get("chunk_strategy") or ""
        strategy_distribution[strategy] = strategy_distribution.get(strategy, 0) + 1
        text = row.get("text") or ""
        total_text_length += len(text)
        if "<" in text and ">" in text:
            html_remnant_count += 1
        directory_id = row.get("directory_id")
        if directory_id in directory_samples and len(directory_samples[directory_id]) < 5:
            directory_samples[directory_id].append(row)

    average_chunk_length = round(total_text_length / len(analysis_rows), 2) if analysis_rows else 0

    return {
        "milvus_db_name": metadata["milvus_db_name"],
        "collection_name": collection_name,
        "partition_name": partition_name,
        "field_count": len(fields),
        "has_publish_direct": has_publish_direct,
        "has_chunk_fields": has_chunk_fields,
        "chunk_fields": chunk_fields,
        "vector_fields": [field for field in fields if field["is_vector"]],
        "scalar_fields": [field["name"] for field in fields if not field["is_vector"]],
        "row_count": row_count,
        "sample_rows": sample,
        "array_contains_expected": {
            "expr": f'ARRAY_CONTAINS(publish_direct, "{expect_code}")',
            "returned": len(hit_rows),
            "rows": hit_rows,
        },
        "array_contains_missing": {
            "expr": f'ARRAY_CONTAINS(publish_direct, "{miss_code}")',
            "returned": len(miss_rows),
            "rows": miss_rows,
        },
        "chunk_quality": {
            "strategy_distribution": strategy_distribution,
            "average_chunk_length": average_chunk_length,
            "html_remnant_count": html_remnant_count,
            "directory_samples": directory_samples,
        },
    }


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full-book vector import chain test.")
    parser.add_argument("--book-json", default=str(PROJECT_ROOT / "decrypted_book.json"))
    parser.add_argument("--vector-url", default="http://127.0.0.1:8006")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--book-id", default="")
    parser.add_argument("--force-new-collection", action="store_true")
    parser.add_argument("--no-force-new-collection", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--expect-code", default="51")
    parser.add_argument("--miss-code", default="999")
    args = parser.parse_args()

    started = datetime.now()
    run_id = started.strftime("%Y%m%d%H%M%S")
    raw_path = Path(args.book_json).resolve()
    raw_book = json.loads(raw_path.read_text(encoding="utf-8"))
    original_book_id = value_or_empty(raw_book.get("BookID") or (raw_book.get("bookInfo") or {}).get("OldID"))
    book_id = args.book_id or f"{original_book_id}{run_id[-6:]}"

    mock_minio_dir = RUN_LOGS / "mock-minio" / "full-book-vector-chain" / run_id
    mock_minio_path = mock_minio_dir / f"{book_id}.json"
    mock_minio_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_path, mock_minio_path)

    payload = build_payload(
        raw_book=raw_book,
        book_id=book_id,
        chunk_size=args.chunk_size,
        force_new_collection=args.force_new_collection and not args.no_force_new_collection,
    )
    legacy_hard_chunk_estimate = sum(math.ceil(len(p["text"]) / args.chunk_size) for p in payload["paragraphs"])

    payload_path = RUN_LOGS / f"full-book-vector-payload-{book_id}.json"
    report_path = RUN_LOGS / f"full-book-vector-report-{book_id}.json"
    save_json(payload_path, payload)

    report: dict[str, Any] = {
        "started_at": started.isoformat(),
        "book_id": book_id,
        "source_json": str(raw_path),
        "mock_minio_json": str(mock_minio_path),
        "payload_json": str(payload_path),
        "vector_url": args.vector_url,
        "chunk_size": args.chunk_size,
        "paragraph_count": len(payload["paragraphs"]),
        "legacy_hard_chunk_estimate": legacy_hard_chunk_estimate,
        "publish_direct": payload["paragraphs"][0]["publish_direct"] if payload["paragraphs"] else [],
    }

    try:
        print(f"[1/5] copied source JSON to mock storage: {mock_minio_path}")
        print(f"[2/5] built payload: paragraphs={len(payload['paragraphs'])}, legacy_hard_chunk_estimate={legacy_hard_chunk_estimate}")
        print(f"[3/5] posting to vector service: {args.vector_url}/up_collection/ImportFromText")
        response = post_json(
            f"{args.vector_url.rstrip('/')}/up_collection/ImportFromText",
            payload,
            timeout=args.timeout,
        )
        report["import_response"] = response
        if response.get("stateCode") != 0:
            raise RuntimeError(f"Vector service returned failure: {response}")

        print("[4/5] querying MySQL metadata")
        metadata = query_metadata(book_id)
        report["metadata"] = metadata

        print("[5/5] verifying Milvus schema and array filter")
        verification = verify_milvus(metadata, book_id, args.expect_code, args.miss_code)
        report["verification"] = verification
        report["finished_at"] = datetime.now().isoformat()
        report["success"] = bool(
            verification["has_publish_direct"]
            and verification["has_chunk_fields"]
            and verification["array_contains_expected"]["returned"] > 0
            and verification["array_contains_missing"]["returned"] == 0
            and verification["row_count"] > 0
        )
        save_json(report_path, report)

        print("")
        print("SUCCESS" if report["success"] else "FINISHED_WITH_CHECK_FAILURE")
        print(f"book_id: {book_id}")
        print(f"collection: {verification['collection_name']}")
        print(f"partition: {verification['partition_name']}")
        print(f"row_count: {verification['row_count']}")
        print(f"strategy_distribution: {verification['chunk_quality']['strategy_distribution']}")
        print(f"average_chunk_length: {verification['chunk_quality']['average_chunk_length']}")
        print(f"html_remnant_count: {verification['chunk_quality']['html_remnant_count']}")
        print(f"publish_direct: {report['publish_direct']}")
        print(f"report: {report_path}")
        return 0 if report["success"] else 2
    except (urllib.error.URLError, urllib.error.HTTPError) as exc:
        report["success"] = False
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        save_json(report_path, report)
        print(f"FAILED: cannot call vector service: {exc}")
        print(f"report: {report_path}")
        return 1
    except Exception as exc:
        report["success"] = False
        report["error"] = repr(exc)
        report["traceback"] = traceback.format_exc()
        save_json(report_path, report)
        print(f"FAILED: {exc}")
        print(f"report: {report_path}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
