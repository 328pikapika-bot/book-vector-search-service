#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path


VECTOR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VECTOR_ROOT))


def main() -> int:
    checks = []

    def record(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})

    try:
        import pymilvus

        record("python_version", sys.version_info[:2] == (3, 10), sys.version.split()[0])
        record("pymilvus_version", pymilvus.__version__ == "2.5.2", pymilvus.__version__)
    except Exception as exc:
        record("runtime_import", False, repr(exc))

    try:
        from constant import PROCESS_WORKERS

        record("single_process_worker", PROCESS_WORKERS == 1, f"PROCESS_WORKERS={PROCESS_WORKERS}")
    except Exception as exc:
        record("single_process_worker", False, repr(exc))

    try:
        import torch

        gpu_details = []
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                with torch.cuda.device(index):
                    free_bytes, total_bytes = torch.cuda.mem_get_info()
                gpu_details.append(
                    {
                        "device": f"cuda:{index}",
                        "name": torch.cuda.get_device_name(index),
                        "free_mib": round(free_bytes / 1024**2, 1),
                        "total_mib": round(total_bytes / 1024**2, 1),
                    }
                )
        record(
            "cuda_environment",
            True,
            json.dumps(
                {
                    "requested": os.environ.get("VECTOR_EMBEDDING_DEVICE", "auto"),
                    "torch": torch.__version__,
                    "torch_cuda": torch.version.cuda,
                    "available": torch.cuda.is_available(),
                    "gpus": gpu_details,
                    "fallback": "CPU" if not gpu_details else None,
                },
                ensure_ascii=False,
            ),
        )
    except Exception as exc:
        record("cuda_environment", True, f"GPU检查失败，将使用CPU: {exc!r}")

    try:
        import config

        prefix = str(getattr(config, "milvus_collection_name", "") or "").strip()
        record("collection_prefix", prefix == "book_vector_v2", prefix or "<empty>")
    except Exception as exc:
        record("config_import", False, repr(exc))

    try:
        from db_process.config.constant_config import M3E_BAS0E_EMBEDDING_MODEL_LOCAL_DICT

        model_dir = Path(M3E_BAS0E_EMBEDDING_MODEL_LOCAL_DICT["m3e-base"])
        model_file = model_dir / "model.safetensors"
        record("embedding_model", model_file.is_file() and model_file.stat().st_size > 100_000_000, model_file)
    except Exception as exc:
        record("embedding_model", False, repr(exc))

    try:
        import numpy as np
        from db_process.db_tool.embedding_model import (
            get_embedding_device_info,
            text2vec,
        )

        probe = text2vec(
            ["GPU向量计算发布前检查"],
            "m3e-base",
            batch_size=1,
            log_context="发布前检查",
        )
        device_info = get_embedding_device_info()
        probe_ok = probe.shape == (1, 768) and bool(np.isfinite(probe).all())
        record(
            "embedding_runtime",
            probe_ok,
            json.dumps(
                {"shape": list(probe.shape), "device": device_info},
                ensure_ascii=False,
            ),
        )
    except Exception as exc:
        record("embedding_runtime", False, repr(exc))

    database_names = []
    try:
        import mysql.connector
        from config import mysql_database, mysql_host, mysql_password, mysql_user

        connection = mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_database,
            connection_timeout=10,
        )
        try:
            cursor = connection.cursor()
            cursor.execute("SELECT name FROM vector_library_base ORDER BY create_time")
            database_names = [str(row[0]) for row in cursor.fetchall()]
            cursor.execute(
                "SELECT COUNT(*) FROM information_schema.tables "
                "WHERE table_schema=%s AND table_name IN "
                "('vector_library_base','vector_library','vector_collection')",
                (mysql_database,),
            )
            table_count = int(cursor.fetchone()[0])
            record("mysql_metadata_tables", table_count == 3, f"tables={table_count}, databases={database_names}")
            lock_name = f"vector_preflight_{uuid.uuid4().hex[:24]}"
            cursor.execute("SELECT GET_LOCK(%s, 0)", (lock_name,))
            lock_acquired = cursor.fetchone()[0] == 1
            lock_released = False
            if lock_acquired:
                cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                lock_released = cursor.fetchone()[0] == 1
            record(
                "mysql_named_lock",
                lock_acquired and lock_released,
                f"acquired={lock_acquired}, released={lock_released}",
            )
        finally:
            connection.close()
    except Exception as exc:
        record("mysql_connection", False, repr(exc))

    try:
        from pymilvus import db
        from national_development.milvus_connection import connect_milvus

        admin_alias = connect_milvus("default")
        milvus_databases = set(db.list_database(using=admin_alias))
        missing = sorted(set(database_names) - milvus_databases)
        record("milvus_databases", not missing, f"missing={missing}, available={sorted(milvus_databases)}")
    except Exception as exc:
        record("milvus_connection", False, repr(exc))

    success = all(check["ok"] for check in checks)
    print(json.dumps({"success": success, "checks": checks}, ensure_ascii=False, indent=2))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
