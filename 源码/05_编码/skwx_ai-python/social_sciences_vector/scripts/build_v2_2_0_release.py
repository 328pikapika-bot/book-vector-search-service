#!/usr/bin/env python
"""Build and verify the v2.2.0 incremental and full-code release archives."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import build_v2_1_0_release as release


VERSION = "v2.2.0"
release.INCREMENTAL_FILES = [
    "national_development/book_operation_lock.py",
    "national_development/milvus_connection.py",
    "national_development/create_library.py",
    "national_development/create_collection.py",
    "national_development/up_collection.py",
    "national_development/drop_collection.py",
    "national_development/drop_paragraph.py",
    "national_development/search_vector.py",
    "national_development/search_scalar.py",
    "national_development/field.py",
    "db_process/db_tool/embedding_model.py",
    "log/logger.py",
    "scripts/release_preflight.py",
]
release.PACKAGE_DOCS = [
    "版本说明_v2.2.0.md",
    "发布与回退说明_v2.2.0.md",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = {
        "incremental": output_dir / f"social_sciences_vector_{VERSION}_incremental.zip",
        "full_code": output_dir / f"social_sciences_vector_{VERSION}_full-code.zip",
    }
    verification = {}

    with tempfile.TemporaryDirectory(prefix="vector-v2.2-release-") as temporary:
        temporary_root = Path(temporary)
        incremental_root = temporary_root / "incremental"
        full_root = temporary_root / "full-code"
        incremental_root.mkdir()
        full_root.mkdir()

        incremental_expected = release.stage_package(
            incremental_root,
            [Path(name) for name in release.INCREMENTAL_FILES],
            docs_dir,
        )
        full_expected = release.stage_package(
            full_root,
            release.full_code_files(),
            docs_dir,
        )
        release.build_zip(incremental_root, archives["incremental"])
        release.build_zip(full_root, archives["full_code"])
        verification["incremental"] = release.verify_package(
            archives["incremental"], incremental_expected
        )
        verification["full_code"] = release.verify_package(
            archives["full_code"], full_expected
        )

    checksums = [
        f"{release.sha256(archive)}  {archive.name}"
        for archive in archives.values()
    ]
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    verification["success"] = all(item["ok"] for item in verification.values())
    verification["version"] = VERSION
    (output_dir / "release-verification.json").write_text(
        json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0 if verification["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
