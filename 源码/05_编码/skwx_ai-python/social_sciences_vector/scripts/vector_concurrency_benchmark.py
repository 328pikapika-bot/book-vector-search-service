#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
import traceback
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
PROJECT_ROOT = SCRIPT_PATH.parents[5]
sys.path.insert(0, str(SCRIPT_PATH.parent))

from full_book_vector_chain_test import build_payload  # noqa: E402


REQUIRED_FIELDS = {
    "publish_direct",
    "chunk_index",
    "chunk_total",
    "chunk_start",
    "chunk_end",
    "chunk_strategy",
}


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, math.ceil(len(ordered) * ratio) - 1)
    return round(ordered[index], 3)


def safe_json_loads(value: str) -> Any:
    try:
        return json.loads(value)
    except Exception:
        return {"raw": value[:2000]}


class ResourceSampler:
    def __init__(self, interval: float = 2.0):
        self.interval = interval
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="benchmark-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=10)
        return self.samples

    def _run(self) -> None:
        try:
            import psutil
        except Exception:
            psutil = None

        while not self._stop.is_set():
            sample: dict[str, Any] = {"time": datetime.now().isoformat()}
            if psutil:
                sample["host_cpu_percent"] = psutil.cpu_percent(interval=None)
                memory = psutil.virtual_memory()
                sample["host_memory_percent"] = memory.percent
                sample["host_memory_used"] = memory.used
            try:
                result = subprocess.run(
                    ["docker", "stats", "--no-stream", "--format", "{{json .}}"],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=8,
                    check=False,
                )
                containers = []
                for line in result.stdout.splitlines():
                    item = safe_json_loads(line)
                    name = str(item.get("Name", ""))
                    if "social-sciences-vector-benchmark" in name:
                        containers.append(item)
                sample["containers"] = containers
            except Exception as exc:
                sample["docker_stats_error"] = repr(exc)
            self.samples.append(sample)
            self._stop.wait(self.interval)


class VectorBenchmark:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.run_id = datetime.now().strftime("%Y%m%d%H%M%S")
        self.output_dir = Path(args.output_dir).resolve() / self.run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._book_counter = 0
        self._book_lock = threading.Lock()
        self.milvus_alias: str | None = None
        raw_book = json.loads(Path(args.book_json).resolve().read_text(encoding="utf-8"))
        self.full_template = build_payload(raw_book, "benchmark-template", args.chunk_size, False)
        self.short_template = self._make_short_template(self.full_template)
        self.results: dict[str, Any] = {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "environment": self._environment(),
            "workload": {
                "full_paragraphs": len(self.full_template["paragraphs"]),
                "full_characters": sum(len(p.get("text", "")) for p in self.full_template["paragraphs"]),
                "short_paragraphs": len(self.short_template["paragraphs"]),
                "short_characters": sum(len(p.get("text", "")) for p in self.short_template["paragraphs"]),
                "chunk_size": args.chunk_size,
            },
            "functional": {},
            "short_stages": [],
            "full_stages": [],
            "mixed": {},
        }

    def _environment(self) -> dict[str, Any]:
        environment = {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "processor": platform.processor(),
        }
        try:
            import psutil

            environment.update(
                {
                    "logical_cpu": psutil.cpu_count(),
                    "physical_cpu": psutil.cpu_count(logical=False),
                    "memory_bytes": psutil.virtual_memory().total,
                }
            )
        except Exception:
            pass
        return environment

    def _make_short_template(self, full_template: dict[str, Any]) -> dict[str, Any]:
        short = copy.deepcopy(full_template)
        source_paragraph = next(
            (paragraph for paragraph in short["paragraphs"] if len(str(paragraph.get("text", ""))) >= 1500),
            max(short["paragraphs"], key=lambda paragraph: len(str(paragraph.get("text", "")))),
        )
        paragraph = copy.deepcopy(source_paragraph)
        text = str(paragraph.get("text", ""))[:1500]
        paragraph["text"] = text
        paragraph["content_text"] = text
        short["paragraphs"] = [paragraph]
        short["max_text_total"] = str(self.args.chunk_size)
        short["force_new_collection"] = False
        return short

    def _next_book_id(self, marker: str) -> str:
        with self._book_lock:
            self._book_counter += 1
            counter = self._book_counter
        digits = "".join(character for character in marker if character.isdigit())[-4:]
        return f"98{self.run_id[-10:]}{digits:0>4}{counter:05d}"

    def _payload(self, template: dict[str, Any], book_id: str) -> dict[str, Any]:
        payload = copy.deepcopy(template)
        payload["book_id"] = book_id
        payload["title"] = f"local-benchmark-{book_id}"
        payload["force_new_collection"] = False
        for paragraph in payload["paragraphs"]:
            paragraph["book_id"] = book_id
        return payload

    def _post(self, path: str, payload: dict[str, Any], timeout: int, barrier: threading.Barrier | None = None) -> dict[str, Any]:
        if barrier:
            barrier.wait(timeout=60)
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.args.vector_url.rstrip('/')}{path}",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
                status = response.status
            response_body = safe_json_loads(text)
            return {
                "http_status": status,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "response": response_body,
                "success": status == 200 and response_body.get("stateCode") == 0,
            }
        except urllib.error.HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            return {
                "http_status": exc.code,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "response": safe_json_loads(text),
                "success": False,
                "error": repr(exc),
            }
        except Exception as exc:
            return {
                "http_status": None,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "response": None,
                "success": False,
                "error": repr(exc),
            }

    def _delete(self, book_id: str, barrier: threading.Barrier | None = None) -> dict[str, Any]:
        if barrier:
            barrier.wait(timeout=60)
        request = urllib.request.Request(
            f"{self.args.vector_url.rstrip('/')}/book/{book_id}",
            method="DELETE",
        )
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                body = safe_json_loads(response.read().decode("utf-8", errors="replace"))
                status = response.status
            return {
                "operation": "delete",
                "book_id": book_id,
                "http_status": status,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "response": body,
                "success": status == 200 and body.get("stateCode") == 0,
            }
        except Exception as exc:
            return {
                "operation": "delete",
                "book_id": book_id,
                "duration_seconds": round(time.perf_counter() - started, 3),
                "success": False,
                "error": repr(exc),
            }

    def _search(self, book_id: str, barrier: threading.Barrier | None = None) -> dict[str, Any]:
        payload = {
            "question": "本地并发测试",
            "topN": 3,
            "minRelevance": 0,
            "vector_model": "m3e-base",
            "filter": "0",
            "condition": [],
        }
        result = self._post(f"/Library/{book_id}/search", payload, 300, barrier)
        result.update({"operation": "search", "book_id": book_id})
        return result

    def _mysql_connection(self):
        return mysql.connector.connect(
            host=self.args.mysql_host,
            port=self.args.mysql_port,
            user=self.args.mysql_user,
            password=self.args.mysql_password,
            database=self.args.mysql_database,
            connection_timeout=10,
        )

    def _milvus_using(self) -> str:
        if self.milvus_alias is None:
            self.milvus_alias = f"benchmark_verify_{self.run_id}"
            connections.connect(
                alias=self.milvus_alias,
                host=self.args.milvus_host,
                port=self.args.milvus_port,
                db_name=self.args.milvus_database,
            )
        return self.milvus_alias

    def verify_consistency(self, stage_book_ids: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "success": False,
            "stage_book_ids": stage_book_ids,
            "errors": [],
        }
        try:
            connection = self._mysql_connection()
            try:
                cursor = connection.cursor()
                cursor.execute(
                    "SELECT id, name, partition_count FROM vector_library "
                    "WHERE name=%s ORDER BY create_time DESC LIMIT 1",
                    (self.args.collection_name,),
                )
                library = cursor.fetchone()
                cursor.execute(
                    "SELECT book_id, partition_name FROM vector_collection "
                    "WHERE name=%s ORDER BY partition_name",
                    (self.args.collection_name,),
                )
                mysql_rows = [(str(row[0]), str(row[1])) for row in cursor.fetchall()]
                mysql_partitions = {row[1] for row in mysql_rows}
                book_id_mismatches = [
                    {"book_id": book_id, "partition_name": partition_name}
                    for book_id, partition_name in mysql_rows
                    if partition_name != f"book_{book_id}"
                ]
                cursor.execute(
                    "SELECT partition_name, COUNT(*) FROM vector_collection "
                    "GROUP BY partition_name HAVING COUNT(*) > 1"
                )
                duplicates = cursor.fetchall()
            finally:
                connection.close()

            milvus_using = self._milvus_using()
            collection_exists = utility.has_collection(self.args.collection_name, using=milvus_using)
            if collection_exists:
                collection = Collection(self.args.collection_name, using=milvus_using)
                collection.load()
                milvus_partitions = {p.name for p in collection.partitions if p.name != "_default"}
                fields = {field.name for field in collection.schema.fields}
                row_counts = {}
                for book_id in stage_book_ids:
                    partition_name = f"book_{book_id}"
                    if partition_name not in milvus_partitions:
                        row_counts[book_id] = None
                        continue
                    rows = collection.query(
                        expr=f'book_id == "{book_id}"',
                        output_fields=["count(*)"],
                        partition_names=[partition_name],
                    )
                    row_counts[book_id] = rows[0].get("count(*)") if rows else 0
            else:
                milvus_partitions = set()
                fields = set()
                row_counts = {book_id: None for book_id in stage_book_ids}

            partition_count = int(library[2]) if library else None
            expected_partition_count = len(milvus_partitions) + 1 if library else None
            stage_partition_names = {f"book_{book_id}" for book_id in stage_book_ids}
            missing_stage = sorted(stage_partition_names - mysql_partitions)
            row_count_values = [value for value in row_counts.values() if value is not None]

            if mysql_partitions != milvus_partitions:
                result["errors"].append("mysql_milvus_partition_mismatch")
            if duplicates:
                result["errors"].append("duplicate_vector_collection_rows")
            if book_id_mismatches:
                result["errors"].append("book_id_partition_mismatch")
            if partition_count != expected_partition_count:
                result["errors"].append("partition_count_mismatch")
            if stage_book_ids and missing_stage:
                result["errors"].append("missing_stage_partitions")
            if stage_book_ids and (not row_count_values or any((value or 0) <= 0 for value in row_count_values)):
                result["errors"].append("empty_stage_partition")
            if stage_book_ids and len(set(row_count_values)) != 1:
                result["errors"].append("inconsistent_row_counts")
            if collection_exists and not REQUIRED_FIELDS.issubset(fields):
                result["errors"].append("required_fields_missing")

            result.update(
                {
                    "library": library,
                    "mysql_partitions": sorted(mysql_partitions),
                    "milvus_partitions": sorted(milvus_partitions),
                    "partition_count": partition_count,
                    "expected_partition_count": expected_partition_count,
                    "duplicates": duplicates,
                    "book_id_mismatches": book_id_mismatches,
                    "row_counts": row_counts,
                    "field_count": len(fields),
                    "required_fields_present": REQUIRED_FIELDS.issubset(fields),
                }
            )
            result["success"] = not result["errors"]
        except Exception as exc:
            result["errors"].append(repr(exc))
            result["traceback"] = traceback.format_exc()
        return result

    def cleanup(self, book_ids: list[str]) -> dict[str, Any]:
        delete_results = [self._delete(book_id) for book_id in book_ids]
        consistency = self.verify_consistency([])
        remaining = set(consistency.get("mysql_partitions", [])) & {f"book_{book_id}" for book_id in book_ids}
        success = all(item.get("success") for item in delete_results) and not remaining and consistency.get("success")
        return {
            "success": success,
            "delete_results": delete_results,
            "remaining_stage_partitions": sorted(remaining),
            "consistency": consistency,
        }

    def run_upload_wave(
        self,
        stage_name: str,
        concurrency: int,
        template: dict[str, Any],
        timeout: int,
        cleanup: bool = True,
    ) -> dict[str, Any]:
        book_ids = [self._next_book_id(stage_name) for _ in range(concurrency)]
        barrier = threading.Barrier(concurrency)
        sampler = ResourceSampler()
        sampler.start()
        started = time.perf_counter()
        requests = []
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_map = {
                executor.submit(
                    self._post,
                    "/up_collection/ImportFromText",
                    self._payload(template, book_id),
                    timeout,
                    barrier,
                ): book_id
                for book_id in book_ids
            }
            for future in as_completed(future_map):
                item = future.result()
                item["book_id"] = future_map[future]
                requests.append(item)
        wall_seconds = time.perf_counter() - started
        resources = sampler.stop()
        consistency = self.verify_consistency(book_ids)
        durations = [item["duration_seconds"] for item in requests]
        success = all(item.get("success") for item in requests) and consistency.get("success")
        result = {
            "stage": stage_name,
            "concurrency": concurrency,
            "book_ids": book_ids,
            "request_count": len(requests),
            "success_count": sum(1 for item in requests if item.get("success")),
            "failure_count": sum(1 for item in requests if not item.get("success")),
            "wall_seconds": round(wall_seconds, 3),
            "throughput_books_per_minute": round(60 * len(requests) / wall_seconds, 4) if wall_seconds else None,
            "latency_seconds": {
                "min": round(min(durations), 3) if durations else None,
                "median": round(statistics.median(durations), 3) if durations else None,
                "p95": percentile(durations, 0.95),
                "max": round(max(durations), 3) if durations else None,
            },
            "requests": sorted(requests, key=lambda item: item["book_id"]),
            "resources": resources,
            "consistency": consistency,
            "success": success,
        }
        if cleanup:
            result["cleanup"] = self.cleanup(book_ids)
            result["success"] = result["success"] and result["cleanup"]["success"]
        return result

    def run_functional(self) -> str | None:
        short_id = self._next_book_id("functional-short")
        first = self._post(
            "/up_collection/ImportFromText",
            self._payload(self.short_template, short_id),
            self.args.short_timeout,
        )
        first_consistency = self.verify_consistency([short_id])
        search = self._search(short_id)
        deleted = self.cleanup([short_id])
        second = self._post(
            "/up_collection/ImportFromText",
            self._payload(self.short_template, short_id),
            self.args.short_timeout,
        )
        second_consistency = self.verify_consistency([short_id])
        second_cleanup = self.cleanup([short_id])

        full_id = self._next_book_id("functional-full")
        full = self._post(
            "/up_collection/ImportFromText",
            self._payload(self.full_template, full_id),
            self.args.full_timeout,
        )
        full_consistency = self.verify_consistency([full_id])
        success = all(
            [
                first.get("success"),
                first_consistency.get("success"),
                search.get("success"),
                deleted.get("success"),
                second.get("success"),
                second_consistency.get("success"),
                second_cleanup.get("success"),
                full.get("success"),
                full_consistency.get("success"),
            ]
        )
        self.results["functional"] = {
            "success": success,
            "short_first": first,
            "short_first_consistency": first_consistency,
            "search": search,
            "short_delete": deleted,
            "short_second": second,
            "short_second_consistency": second_consistency,
            "short_second_cleanup": second_cleanup,
            "full_book_id": full_id,
            "full_upload": full,
            "full_consistency": full_consistency,
        }
        return full_id if success else None

    def run_mixed(self, retained_full_book_id: str) -> dict[str, Any]:
        total_concurrency = max(3, int(self.args.mixed_concurrency))
        upload_count = max(1, total_concurrency // 3)
        delete_count = max(1, total_concurrency // 3)
        search_count = total_concurrency - upload_count - delete_count
        prepare = self.run_upload_wave(
            "mixed-prepare", delete_count, self.short_template, self.args.short_timeout, cleanup=False
        )
        if not prepare.get("success"):
            prepare["cleanup"] = self.cleanup(prepare.get("book_ids", []))
            return {"success": False, "prepare": prepare}

        upload_ids = [self._next_book_id("mixed-upload") for _ in range(upload_count)]
        delete_ids = prepare["book_ids"]
        barrier = threading.Barrier(total_concurrency)
        operations = []
        sampler = ResourceSampler()
        sampler.start()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=total_concurrency) as executor:
            futures = []
            for book_id in upload_ids:
                futures.append(
                    executor.submit(
                        self._post,
                        "/up_collection/ImportFromText",
                        self._payload(self.short_template, book_id),
                        self.args.short_timeout,
                        barrier,
                    )
                )
            for _ in range(search_count):
                futures.append(executor.submit(self._search, retained_full_book_id, barrier))
            for book_id in delete_ids:
                futures.append(executor.submit(self._delete, book_id, barrier))
            for future in as_completed(futures):
                operations.append(future.result())
        wall_seconds = time.perf_counter() - started
        resources = sampler.stop()
        # Only the simultaneously uploaded short books are expected to have
        # identical row counts.  The retained full book is intentionally much
        # larger; its presence is covered by the global partition comparison
        # and by the concurrent search operations.
        consistency = self.verify_consistency(upload_ids)
        cleanup = self.cleanup(upload_ids + [retained_full_book_id])
        success = all(item.get("success") for item in operations) and consistency.get("success") and cleanup.get("success")
        return {
            "success": success,
            "concurrency": total_concurrency,
            "operation_counts": {
                "upload": upload_count,
                "search": search_count,
                "delete": delete_count,
            },
            "prepare": prepare,
            "upload_ids": upload_ids,
            "delete_ids": delete_ids,
            "operations": operations,
            "wall_seconds": round(wall_seconds, 3),
            "resources": resources,
            "consistency": consistency,
            "cleanup": cleanup,
        }

    def _container_state(self) -> list[dict[str, Any]]:
        try:
            ps = subprocess.run(
                ["docker", "compose", "ps", "-q"],
                cwd=self.args.compose_dir,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=20,
                check=False,
            )
            states = []
            for container_id in ps.stdout.split():
                inspect = subprocess.run(
                    ["docker", "inspect", container_id],
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                    check=False,
                )
                data = safe_json_loads(inspect.stdout)
                if data:
                    state = data[0].get("State", {})
                    states.append(
                        {
                            "name": data[0].get("Name", "").lstrip("/"),
                            "status": state.get("Status"),
                            "restart_count": data[0].get("RestartCount"),
                            "oom_killed": state.get("OOMKilled"),
                        }
                    )
            return states
        except Exception as exc:
            return [{"error": repr(exc)}]

    def _service_logs(self) -> dict[str, Any]:
        result = subprocess.run(
            ["docker", "compose", "logs", "--no-color", "vector-service"],
            cwd=self.args.compose_dir,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=60,
            check=False,
        )
        text = (result.stdout or "") + (result.stderr or "")
        log_path = self.output_dir / "vector-service.log"
        log_path.write_text(text, encoding="utf-8")
        patterns = {
            "wrong_default_database": "database=default][collection=book_vector_v2",
            "http_500": "500 Internal Server Error",
            "collection_missing": "Collection 'book_vector_v2' not exist",
            "traceback": "Traceback (most recent call last)",
        }
        return {
            "path": str(log_path),
            "pattern_counts": {name: text.count(pattern) for name, pattern in patterns.items()},
        }

    def _summaries(self) -> dict[str, Any]:
        short_passed = []
        for stage in self.results["short_stages"]:
            if stage["success"]:
                short_passed.append(stage["concurrency"])

        full_by_concurrency: dict[int, list[dict[str, Any]]] = {}
        for stage in self.results["full_stages"]:
            full_by_concurrency.setdefault(int(stage["concurrency"]), []).append(stage)
        full_summary = []
        for concurrency, stages in sorted(full_by_concurrency.items()):
            wall = sum(stage["wall_seconds"] for stage in stages)
            count = sum(stage["request_count"] for stage in stages)
            passed = all(stage["success"] for stage in stages)
            throughput = 60 * count / wall if wall else 0
            full_summary.append(
                {
                    "concurrency": concurrency,
                    "passed": passed,
                    "requests": count,
                    "wall_seconds": round(wall, 3),
                    "throughput_books_per_minute": round(throughput, 4),
                }
            )
        passing_full = [item for item in full_summary if item["passed"]]
        max_throughput = max((item["throughput_books_per_minute"] for item in passing_full), default=0)
        recommended = next(
            (
                item["concurrency"]
                for item in passing_full
                if item["throughput_books_per_minute"] >= max_throughput * 0.95
            ),
            None,
        )
        return {
            "short_stable_limit": max(short_passed) if short_passed else None,
            "full_stable_limit": max((item["concurrency"] for item in passing_full), default=None),
            "recommended_full_concurrency": recommended,
            "full_concurrency_summary": full_summary,
        }

    def run(self) -> int:
        retained_full_book_id = None
        try:
            retained_full_book_id = self.run_functional()
            if not retained_full_book_id:
                raise RuntimeError("Functional baseline failed; load stages were not started.")

            if self.args.mode == "mixed":
                self.results["mixed"] = self.run_mixed(retained_full_book_id)
                retained_full_book_id = None
                return self._finish()

            if self.args.mode == "full10":
                stage = self.run_upload_wave(
                    "full-c10-w1",
                    10,
                    self.full_template,
                    self.args.full_timeout,
                )
                self.results["full_stages"].append(stage)
                self.results["retained_full_cleanup"] = self.cleanup([retained_full_book_id])
                retained_full_book_id = None
                return self._finish()

            for concurrency in [1, 5, 10, 20, 40]:
                stage_success = True
                for wave in range(1, 4):
                    stage = self.run_upload_wave(
                        f"short-c{concurrency}-w{wave}",
                        concurrency,
                        self.short_template,
                        self.args.short_timeout,
                    )
                    self.results["short_stages"].append(stage)
                    self._checkpoint()
                    stage_success = stage_success and stage["success"]
                    if not stage_success:
                        break
                if not stage_success:
                    break

            full_matrix = [(1, 3), (2, 2), (3, 2), (5, 1), (8, 1)]
            for concurrency, waves in full_matrix:
                stage_success = True
                for wave in range(1, waves + 1):
                    stage = self.run_upload_wave(
                        f"full-c{concurrency}-w{wave}",
                        concurrency,
                        self.full_template,
                        self.args.full_timeout,
                    )
                    self.results["full_stages"].append(stage)
                    self._checkpoint()
                    stage_success = stage_success and stage["success"]
                    if not stage_success:
                        break
                if not stage_success:
                    break

            self.results["mixed"] = self.run_mixed(retained_full_book_id)
            retained_full_book_id = None
        except Exception as exc:
            self.results["fatal_error"] = repr(exc)
            self.results["fatal_traceback"] = traceback.format_exc()
        finally:
            if retained_full_book_id:
                self.results["retained_full_cleanup"] = self.cleanup([retained_full_book_id])
            if "finished_at" not in self.results:
                return self._finish()

    def _finish(self) -> int:
        self.results["container_state"] = self._container_state()
        self.results["service_logs"] = self._service_logs()
        self.results["summary"] = self._summaries()
        self.results["finished_at"] = datetime.now().isoformat()
        load_stages_passed = True
        if self.args.mode == "full":
            load_stages_passed = bool(
                self.results.get("short_stages")
                and all(stage.get("success") for stage in self.results["short_stages"])
                and self.results.get("full_stages")
                and all(stage.get("success") for stage in self.results["full_stages"])
            )
        elif self.args.mode == "full10":
            load_stages_passed = bool(
                self.results.get("full_stages")
                and all(stage.get("success") for stage in self.results["full_stages"])
            )
        mixed_passed = self.args.mode == "full10" or self.results.get("mixed", {}).get("success")
        self.results["release_gate_passed"] = bool(
            self.results.get("functional", {}).get("success")
            and load_stages_passed
            and mixed_passed
            and not any(
                self.results["service_logs"]["pattern_counts"].get(name, 0)
                for name in ["wrong_default_database", "http_500", "collection_missing"]
            )
            and all(
                state.get("restart_count") == 0 and not state.get("oom_killed")
                for state in self.results["container_state"]
                if "error" not in state
            )
        )
        self._checkpoint()
        self._write_markdown()
        return 0 if self.results["release_gate_passed"] else 1

    def _checkpoint(self) -> None:
        (self.output_dir / "benchmark-results.json").write_text(
            json.dumps(self.results, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def _write_markdown(self) -> None:
        summary = self.results.get("summary", {})
        lines = [
            "# 向量服务本地并发压测报告",
            "",
            f"- 运行编号：`{self.run_id}`",
            f"- 开始时间：{self.results.get('started_at')}",
            f"- 结束时间：{self.results.get('finished_at')}",
            f"- 发布门禁：{'通过' if self.results.get('release_gate_passed') else '未通过'}",
            "",
            "## 环境与负载",
            "",
            f"- 主机：{self.results.get('environment')}",
            f"- 负载：{self.results.get('workload')}",
            "- 服务运行时：Python 3.10、PyMilvus 2.5.2、CPU 版 PyTorch 2.5.1",
            "- Milvus 服务端：2.5.14（与目标客户端同属 2.5 系列）",
            "",
            "## 容量结论",
            "",
            f"- 短请求稳定并发上限：{summary.get('short_stable_limit')}",
            f"- 完整图书稳定并发上限：{summary.get('full_stable_limit')}",
            f"- 完整图书推荐并发：{summary.get('recommended_full_concurrency')}",
            "",
            "## 短请求测试",
            "",
            "| 阶段 | 并发 | 成功/总数 | 总耗时(s) | P50(s) | P95(s) | 结果 |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
        for stage in self.results.get("short_stages", []):
            lines.append(
                f"| {stage['stage']} | {stage['concurrency']} | {stage['success_count']}/{stage['request_count']} | "
                f"{stage['wall_seconds']} | {stage['latency_seconds']['median']} | {stage['latency_seconds']['p95']} | "
                f"{'通过' if stage['success'] else '失败'} |"
            )
        lines.extend(
            [
                "",
                "## 完整图书测试",
                "",
                "| 阶段 | 并发 | 成功/总数 | 总耗时(s) | 图书/分钟 | P50(s) | P95(s) | 结果 |",
                "|---|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for stage in self.results.get("full_stages", []):
            lines.append(
                f"| {stage['stage']} | {stage['concurrency']} | {stage['success_count']}/{stage['request_count']} | "
                f"{stage['wall_seconds']} | {stage['throughput_books_per_minute']} | "
                f"{stage['latency_seconds']['median']} | {stage['latency_seconds']['p95']} | "
                f"{'通过' if stage['success'] else '失败'} |"
            )
        lines.extend(
            [
                "",
                "## 一致性与混合操作",
                "",
                f"- 功能基线：{'通过' if self.results.get('functional', {}).get('success') else '失败'}",
                f"- 混合操作：{'通过' if self.results.get('mixed', {}).get('success') else '失败'}",
                f"- 日志异常计数：{self.results.get('service_logs', {}).get('pattern_counts')}",
                f"- 容器状态：{self.results.get('container_state')}",
                "",
                "## 说明",
                "",
                "本报告结论仅代表本机隔离环境。生产服务器仍须从单本和低并发开始灰度。",
                "详细请求、资源采样和一致性结果见同目录 `benchmark-results.json`。",
            ]
        )
        (self.output_dir / "本地并发压测报告.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run isolated vector-service concurrency benchmarks.")
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
    parser.add_argument("--mode", choices=["full", "mixed", "full10"], default="full")
    parser.add_argument("--mixed-concurrency", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "run-logs" / "local-vector-benchmark"),
    )
    parser.add_argument(
        "--compose-dir",
        default=str(PROJECT_ROOT / "local-vector-benchmark"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(VectorBenchmark(parse_args()).run())
