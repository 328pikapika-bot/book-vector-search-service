#!/usr/bin/env python
"""Build and verify the v2.1.0 incremental and full-code release archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import py_compile
import re
import shutil
import tempfile
import zipfile
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
VECTOR_ROOT = SCRIPT_PATH.parents[1]

INCREMENTAL_FILES = [
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
    "log/logger.py",
    "scripts/release_preflight.py",
]

FULL_ROOT_FILES = [
    "constant.py",
    "download_model.py",
    "main_library_api.py",
    "main_library_run.sh",
    "requirements.txt",
]

FULL_DIRECTORIES = [
    "db_process",
    "llm_tool",
    "log",
    "national_development",
]

PACKAGE_DOCS = [
    "版本说明_v2.1.0.md",
    "发布与回退说明_v2.1.0.md",
    "本地并发压测报告_20260723.md",
]

EXCLUDED_NAMES = {
    "config.py",
    "__pycache__",
    ".idea",
    "log.txt",
    "test.log",
    "nohup.out",
    "decrypted_book.json",
}

EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".log", ".zip", ".xlsx", ".xls", ".ipynb", ".csv"}
EXCLUDED_PARTS = {"data", "input_data", "local-vector-benchmark", "run-logs"}
MODEL_SUFFIXES = {".bin", ".safetensors", ".onnx", ".pt", ".pth"}

SENSITIVE_PATTERNS = [
    re.compile(rb"benchmark_root", re.IGNORECASE),
    re.compile(rb"config\.local", re.IGNORECASE),
    re.compile(rb"mysql_password\s*=\s*['\"][^'\"]+['\"]", re.IGNORECASE),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def is_excluded(relative: Path) -> bool:
    if any(part in EXCLUDED_PARTS or part == "__pycache__" for part in relative.parts):
        return True
    if relative.name in EXCLUDED_NAMES:
        return True
    if relative.suffix.lower() in EXCLUDED_SUFFIXES | MODEL_SUFFIXES:
        return True
    if relative.name.startswith("test") and relative.suffix.lower() == ".py":
        return True
    if relative.name in {"migrate_data.py", "vector_concurrency_benchmark.py", "full_book_vector_chain_test.py"}:
        return True
    return False


def copy_file(source: Path, target_root: Path, relative: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    target = target_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def full_code_files() -> list[Path]:
    selected = [Path(name) for name in FULL_ROOT_FILES]
    for directory_name in FULL_DIRECTORIES:
        directory = VECTOR_ROOT / directory_name
        for source in directory.rglob("*"):
            if not source.is_file():
                continue
            relative = source.relative_to(VECTOR_ROOT)
            if not is_excluded(relative):
                selected.append(relative)
    selected.append(Path("scripts/release_preflight.py"))
    return sorted(set(selected), key=lambda item: item.as_posix())


def write_manifest(package_root: Path) -> None:
    files = sorted(
        (path for path in package_root.rglob("*") if path.is_file() and path.name not in {"FILELIST.txt", "SHA256SUMS.txt"}),
        key=lambda item: item.relative_to(package_root).as_posix(),
    )
    filelist = [path.relative_to(package_root).as_posix() for path in files]
    checksums = [f"{sha256(path)}  {path.relative_to(package_root).as_posix()}" for path in files]
    (package_root / "FILELIST.txt").write_text("\n".join(filelist) + "\n", encoding="utf-8")
    (package_root / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n", encoding="utf-8")


def build_zip(package_root: Path, archive: Path) -> None:
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as output:
        for path in sorted(package_root.rglob("*"), key=lambda item: item.relative_to(package_root).as_posix()):
            if path.is_file():
                output.write(path, path.relative_to(package_root).as_posix())


def verify_package(archive: Path, expected_files: set[str]) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="vector-release-verify-") as temporary:
        extract_root = Path(temporary)
        with zipfile.ZipFile(archive) as package:
            bad_member = package.testzip()
            package.extractall(extract_root)
            names = {name.rstrip("/") for name in package.namelist() if not name.endswith("/")}

        missing = sorted(expected_files - names)
        forbidden = sorted(
            name for name in names
            if is_excluded(Path(name)) or Path(name).name == "config.py"
        )
        compile_errors = []
        sensitive_hits = []
        for path in extract_root.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(extract_root).as_posix()
            data = path.read_bytes()
            for pattern in SENSITIVE_PATTERNS:
                if pattern.search(data):
                    sensitive_hits.append(relative)
                    break
            if path.suffix == ".py":
                try:
                    py_compile.compile(str(path), doraise=True)
                except Exception as exc:
                    compile_errors.append({"file": relative, "error": repr(exc)})

        ok = not any([bad_member, missing, forbidden, compile_errors, sensitive_hits])
        return {
            "ok": ok,
            "archive_sha256": sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "file_count": len(names),
            "bad_zip_member": bad_member,
            "missing": missing,
            "forbidden": forbidden,
            "compile_errors": compile_errors,
            "sensitive_hits": sorted(set(sensitive_hits)),
        }


def stage_package(root: Path, code_files: list[Path], docs_dir: Path) -> set[str]:
    for relative in code_files:
        copy_file(VECTOR_ROOT / relative, root, relative)
    for doc_name in PACKAGE_DOCS:
        copy_file(docs_dir / doc_name, root, Path(doc_name))
    write_manifest(root)
    return {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    docs_dir = Path(args.docs_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    archives = {
        "incremental": output_dir / "social_sciences_vector_v2.1.0_incremental.zip",
        "full_code": output_dir / "social_sciences_vector_v2.1.0_full-code.zip",
    }
    verification = {}

    with tempfile.TemporaryDirectory(prefix="vector-release-build-") as temporary:
        temporary_root = Path(temporary)
        incremental_root = temporary_root / "incremental"
        full_root = temporary_root / "full-code"
        incremental_root.mkdir()
        full_root.mkdir()

        incremental_expected = stage_package(
            incremental_root, [Path(name) for name in INCREMENTAL_FILES], docs_dir
        )
        full_expected = stage_package(full_root, full_code_files(), docs_dir)
        build_zip(incremental_root, archives["incremental"])
        build_zip(full_root, archives["full_code"])
        verification["incremental"] = verify_package(archives["incremental"], incremental_expected)
        verification["full_code"] = verify_package(archives["full_code"], full_expected)

    external_checksums = [
        f"{sha256(archive)}  {archive.name}" for archive in archives.values()
    ]
    (output_dir / "SHA256SUMS.txt").write_text("\n".join(external_checksums) + "\n", encoding="utf-8")
    verification["success"] = all(item["ok"] for item in verification.values())
    verification_path = output_dir / "release-verification.json"
    verification_path.write_text(json.dumps(verification, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(verification, ensure_ascii=False, indent=2))
    return 0 if verification["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
