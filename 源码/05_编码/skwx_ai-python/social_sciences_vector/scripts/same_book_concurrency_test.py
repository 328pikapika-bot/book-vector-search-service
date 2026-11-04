#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import mysql.connector
from pymilvus import Collection, connections, utility


SCRIPT_PATH = Path(__file__).resolve()
VECTOR_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[5] if len(SCRIPT_PATH.parents) > 5 else VECTOR_ROOT

import sys

sys.path.insert(0, str(SCRIPT_PATH.parent))
from full_book_vector_chain_test import build_payload  # noqa: E402


def safe_json(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return {"raw": value[:2000]}


def request_json(
    base_url: str,
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    timeout: int = 1800,
    barrier: threading.Barrier | None = None,
) -> dict[str, Any]:
    if barrier:
        barrier.wait(timeout=60)
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}{path}",
        data=body,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method=method,
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = safe_json(response.read().decode("utf-8", errors="replace"))
            return {
                "http_status": response.status,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "body": response_body,
            }
    except urllib.error.HTTPError as exc:
        return {
            "http_status": exc.code,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "body": safe_json(exc.read().decode("utf-8", errors="replace")),
            "error": repr(exc),
        }
    except Exception as exc:
        return {
            "http_status": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "body": None,
            "error": repr(exc),
        }


class SameBookTest:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.alias = f"same_book_verify_{datetime.now().strftime('%H%M%S%f')}"
        connections.connect(
            alias=self.alias,
            host=args.milvus_host,
            port=args.milvus_port,
            db_name=args.milvus_database,
        )
        raw = json.loads(Path(args.book_json).resolve().read_text(encoding="utf-8"))
        self.full_template = build_payload(raw, "same-book-template", args.chunk_size, False)
        self.short_template = copy.deepcopy(self.full_template)
        paragraph = copy.deepcopy(
            next(
                item for item in self.short_template["paragraphs"]
                if len(str(item.get("text", ""))) >= 1500
            )
        )
        paragraph["text"] = str(paragraph["text"])[:1500]
        paragraph["content_text"] = paragraph["text"]
        self.short_template["paragraphs"] = [paragraph]
        self.short_template["max_text_total"] = str(args.chunk_size)
        self.results = {
            "started_at": datetime.now().isoformat(),
            "full_characters": sum(len(item.get("text", "")) for item in self.full_template["paragraphs"]),
            "full_paragraphs": len(self.full_template["paragraphs"]),
        }

    def close(self):
        connections.disconnect(self.alias)

    def payload(self, template: dict[str, Any], book_id: str) -> dict[str, Any]:
        payload = copy.deepcopy(template)
        payload["book_id"] = book_id
        payload["title"] = f"same-book-test-{book_id}"
        payload["force_new_collection"] = False
        for paragraph in payload["paragraphs"]:
            paragraph["book_id"] = book_id
        return payload

    def mysql_connection(self):
        return mysql.connector.connect(
            host=self.args.mysql_host,
            port=self.args.mysql_port,
            user=self.args.mysql_user,
            password=self.args.mysql_password,
            database=self.args.mysql_database,
            connection_timeout=10,
        )

    def mysql_state(self, book_id: str) -> dict[str, Any]:
        connection = self.mysql_connection()
        try:
            cursor = connection.cursor()
            cursor.execute(
                "SELECT book_id, partition_name FROM vector_collection WHERE partition_name=%s",
                (f"book_{book_id}",),
            )
            rows = cursor.fetchall()
            cursor.execute(
                "SELECT partition_count FROM vector_library WHERE name=%s LIMIT 1",
                (self.args.collection_name,),
            )
            library = cursor.fetchone()
            return {
                "rows": rows,
                "partition_count": int(library[0]) if library else None,
                "mapping_ok": all(row[1] == f"book_{row[0]}" for row in rows),
            }
        finally:
            connection.close()

    def wait_for_metadata(self, book_id: str, timeout: int = 60) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.mysql_state(book_id)["rows"]:
                return True
            time.sleep(0.1)
        return False

    def milvus_state(self, book_id: str) -> dict[str, Any]:
        if not utility.has_collection(self.args.collection_name, using=self.alias):
            return {"collection_exists": False, "partition_exists": False, "row_count": None}
        collection = Collection(self.args.collection_name, using=self.alias)
        collection.load()
        partition_name = f"book_{book_id}"
        partitions = {partition.name for partition in collection.partitions}
        if partition_name not in partitions:
            return {"collection_exists": True, "partition_exists": False, "row_count": None}
        rows = collection.query(
            expr=f'book_id == "{book_id}"',
            output_fields=["count(*)"],
            partition_names=[partition_name],
        )
        return {
            "collection_exists": True,
            "partition_exists": True,
            "row_count": int(rows[0].get("count(*)", 0)) if rows else 0,
        }

    def verify_present(self, book_id: str, expected_rows: int) -> dict[str, Any]:
        mysql_state = self.mysql_state(book_id)
        milvus_state = self.milvus_state(book_id)
        success = bool(
            len(mysql_state["rows"]) == 1
            and mysql_state["mapping_ok"]
            and milvus_state["partition_exists"]
            and milvus_state["row_count"] == expected_rows
        )
        return {"success": success, "mysql": mysql_state, "milvus": milvus_state}

    def verify_absent(self, book_id: str) -> dict[str, Any]:
        mysql_state = self.mysql_state(book_id)
        milvus_state = self.milvus_state(book_id)
        success = not mysql_state["rows"] and not milvus_state["partition_exists"]
        return {"success": success, "mysql": mysql_state, "milvus": milvus_state}

    def delete(self, book_id: str) -> dict[str, Any]:
        return request_json(self.args.vector_url, "DELETE", f"/book/{book_id}", timeout=1800)

    def duplicate_full_upload(self, book_id: str) -> dict[str, Any]:
        payload = self.payload(self.full_template, book_id)
        barrier = threading.Barrier(2)
        with ThreadPoolExecutor(max_workers=3) as executor:
            uploads = [
                executor.submit(
                    request_json,
                    self.args.vector_url,
                    "POST",
                    "/up_collection/ImportFromText",
                    payload,
                    self.args.full_timeout,
                    barrier,
                )
                for _ in range(2)
            ]
            metadata_seen = self.wait_for_metadata(book_id)
            search = request_json(
                self.args.vector_url,
                "POST",
                f"/Library/{book_id}/search",
                {
                    "question": "同书并发门禁测试",
                    "topN": 3,
                    "minRelevance": 0,
                    "vector_model": "m3e-base",
                    "filter": "0",
                    "condition": [],
                },
                timeout=30,
            )
            upload_results = [future.result() for future in as_completed(uploads)]

        messages = [str(item.get("body", {}).get("message")) for item in upload_results]
        uploads_ok = all(
            item.get("http_status") == 200 and item.get("body", {}).get("stateCode") == 0
            for item in upload_results
        )
        one_real_one_duplicate = messages.count("已存在，不重复入库") == 1
        search_blocked = search.get("http_status") == 409 and search.get("body", {}).get("stateCode") == 1
        verification = self.verify_present(book_id, expected_rows=652)
        cleanup = self.delete(book_id)
        cleanup_state = self.verify_absent(book_id)
        return {
            "success": all(
                [metadata_seen, uploads_ok, one_real_one_duplicate, search_blocked,
                 verification["success"], cleanup.get("body", {}).get("stateCode") == 0,
                 cleanup_state["success"]]
            ),
            "book_id": book_id,
            "metadata_seen_during_upload": metadata_seen,
            "uploads": upload_results,
            "messages": messages,
            "search_during_upload": search,
            "verification": verification,
            "cleanup": cleanup,
            "cleanup_state": cleanup_state,
        }

    def upload_delete_race(self, book_id: str) -> dict[str, Any]:
        payload = self.payload(self.short_template, book_id)
        with ThreadPoolExecutor(max_workers=2) as executor:
            upload_future = executor.submit(
                request_json,
                self.args.vector_url,
                "POST",
                "/up_collection/ImportFromText",
                payload,
                self.args.short_timeout,
            )
            metadata_seen = self.wait_for_metadata(book_id)
            delete_future = executor.submit(self.delete, book_id)
            upload = upload_future.result()
            delete = delete_future.result()
        final_state = self.verify_absent(book_id)
        return {
            "success": bool(
                metadata_seen
                and upload.get("http_status") == 200
                and upload.get("body", {}).get("stateCode") == 0
                and delete.get("http_status") == 200
                and delete.get("body", {}).get("stateCode") == 0
                and final_state["success"]
            ),
            "book_id": book_id,
            "metadata_seen_during_upload": metadata_seen,
            "upload": upload,
            "delete": delete,
            "final_state": final_state,
        }

    def failure_then_retry(self, book_id: str) -> dict[str, Any]:
        invalid_payload = self.payload(self.short_template, book_id)
        invalid_payload["vector_model"] = "invalid-local-test-model"
        failed = request_json(
            self.args.vector_url,
            "POST",
            "/up_collection/ImportFromText",
            invalid_payload,
            self.args.short_timeout,
        )
        after_failure = self.verify_absent(book_id)
        retry = request_json(
            self.args.vector_url,
            "POST",
            "/up_collection/ImportFromText",
            self.payload(self.short_template, book_id),
            self.args.short_timeout,
        )
        retry_state = self.verify_present(book_id, expected_rows=4)
        cleanup = self.delete(book_id)
        final_state = self.verify_absent(book_id)
        return {
            "success": bool(
                failed.get("http_status") == 500
                and failed.get("body", {}).get("stateCode") == 1
                and after_failure["success"]
                and retry.get("http_status") == 200
                and retry.get("body", {}).get("stateCode") == 0
                and retry_state["success"]
                and cleanup.get("body", {}).get("stateCode") == 0
                and final_state["success"]
            ),
            "book_id": book_id,
            "failed_request": failed,
            "after_failure": after_failure,
            "retry": retry,
            "retry_state": retry_state,
            "cleanup": cleanup,
            "final_state": final_state,
        }

    def run(self) -> int:
        marker = datetime.now().strftime("%y%m%d%H%M%S")
        try:
            self.results["duplicate_full_upload"] = self.duplicate_full_upload(f"97{marker}001")
            self.results["upload_delete_race"] = self.upload_delete_race(f"97{marker}002")
            self.results["failure_then_retry"] = self.failure_then_retry(f"97{marker}003")
        finally:
            self.results["finished_at"] = datetime.now().isoformat()
            self.results["success"] = all(
                self.results.get(name, {}).get("success")
                for name in ["duplicate_full_upload", "upload_delete_race", "failure_then_retry"]
            )
            output = Path(self.args.output).resolve()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(self.results, ensure_ascii=False, indent=2), encoding="utf-8")
            self.close()
        print(json.dumps({"success": self.results["success"], "output": str(output)}, ensure_ascii=False))
        return 0 if self.results["success"] else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--book-json", default=str(PROJECT_ROOT / "decrypted_book.json"))
    parser.add_argument("--vector-url", default="http://127.0.0.1:18006")
    parser.add_argument("--mysql-host", default="127.0.0.1")
    parser.add_argument("--mysql-port", type=int, default=13306)
    parser.add_argument("--mysql-user", default="root")
    parser.add_argument("--mysql-password", default="benchmark_root")
    parser.add_argument("--mysql-database", default="vector_benchmark")
    parser.add_argument("--milvus-host", default="127.0.0.1")
    parser.add_argument("--milvus-port", type=int, default=19531)
    parser.add_argument("--milvus-database", default="ssc_1")
    parser.add_argument("--collection-name", default="book_vector_v2")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--short-timeout", type=int, default=300)
    parser.add_argument("--full-timeout", type=int, default=1800)
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "run-logs" / "same-book-concurrency-result.json"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(SameBookTest(parse_args()).run())
