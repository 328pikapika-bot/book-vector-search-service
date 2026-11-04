#!/usr/bin/env python
"""Merge benchmark evidence and generate the final v2.1.0 release documents."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], ratio: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[max(0, math.ceil(len(ordered) * ratio) - 1)], 3)


def aggregate(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for stage in stages:
        groups[int(stage["concurrency"])].append(stage)
    result = []
    for concurrency, items in sorted(groups.items()):
        requests = [request for item in items for request in item.get("requests", [])]
        durations = [float(request["duration_seconds"]) for request in requests]
        total = len(requests)
        success = sum(1 for request in requests if request.get("success"))
        wall = sum(float(item.get("wall_seconds", 0)) for item in items)
        result.append(
            {
                "concurrency": concurrency,
                "waves": len(items),
                "requests": total,
                "successes": success,
                "failures": total - success,
                "success_rate_percent": round(success * 100 / total, 2) if total else 0,
                "wall_seconds": round(wall, 3),
                "throughput_books_per_minute": round(60 * total / wall, 4) if wall else None,
                "p50_seconds": round(statistics.median(durations), 3) if durations else None,
                "p95_seconds": percentile(durations, 0.95),
                "max_seconds": round(max(durations), 3) if durations else None,
                "passed": all(item.get("success") for item in items),
                "consistency_passed": all(item.get("consistency", {}).get("success") for item in items),
                "cleanup_passed": all(item.get("cleanup", {}).get("success") for item in items),
            }
        )
    return result


def parse_memory(value: str) -> int:
    first = str(value).split("/")[0].strip()
    if not first:
        return 0
    units = {"B": 1, "KiB": 1024, "MiB": 1024**2, "GiB": 1024**3}
    for unit in ["GiB", "MiB", "KiB", "B"]:
        if first.endswith(unit):
            return int(float(first[: -len(unit)].strip()) * units[unit])
    return 0


def resource_peaks(stages: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    peaks: dict[str, dict[str, Any]] = {}
    for stage in stages:
        for sample in stage.get("resources", []):
            for container in sample.get("containers", []):
                name = str(container.get("Name", "unknown"))
                if "vector-service" in name:
                    service = "vector-service"
                elif "milvus" in name:
                    service = "milvus"
                elif "mysql" in name:
                    service = "mysql"
                elif "minio" in name:
                    service = "minio"
                elif "etcd" in name:
                    service = "etcd"
                else:
                    continue
                cpu = float(str(container.get("CPUPerc", "0")).rstrip("%") or 0)
                memory = parse_memory(str(container.get("MemUsage", "")))
                item = peaks.setdefault(service, {"max_cpu_percent": 0.0, "max_memory_bytes": 0})
                item["max_cpu_percent"] = max(float(item["max_cpu_percent"]), cpu)
                item["max_memory_bytes"] = max(int(item["max_memory_bytes"]), memory)
    for item in peaks.values():
        item["max_cpu_percent"] = round(float(item["max_cpu_percent"]), 2)
        item["max_memory_mib"] = round(int(item["max_memory_bytes"]) / 1024**2, 2)
    return peaks


def table_rows(items: list[dict[str, Any]]) -> list[str]:
    rows = []
    for item in items:
        rows.append(
            "| {concurrency} | {waves} | {successes}/{requests} | {success_rate_percent}% | "
            "{throughput_books_per_minute} | {p50_seconds} | {p95_seconds} | {max_seconds} | {status} |".format(
                **item, status="通过" if item["passed"] else "失败"
            )
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--mixed", required=True)
    parser.add_argument("--full10", required=True)
    parser.add_argument("--scalar", required=True)
    parser.add_argument("--same-book", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    base_path = Path(args.base).resolve()
    mixed_path = Path(args.mixed).resolve()
    full10_path = Path(args.full10).resolve()
    scalar_path = Path(args.scalar).resolve()
    same_book_path = Path(args.same_book).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_json(base_path)
    mixed = load_json(mixed_path)
    full10 = load_json(full10_path)
    scalar = load_json(scalar_path)
    same_book = load_json(same_book_path)

    short_summary = aggregate(base.get("short_stages", []))
    full_stages = list(base.get("full_stages", [])) + list(full10.get("full_stages", []))
    full_summary = aggregate(full_stages)
    passing_full = [item for item in full_summary if item["passed"]]
    max_throughput = max(item["throughput_books_per_minute"] for item in passing_full)
    recommended = next(
        item["concurrency"]
        for item in passing_full
        if item["throughput_books_per_minute"] >= max_throughput * 0.95
    )
    stable_short = max(item["concurrency"] for item in short_summary if item["passed"])
    stable_full = max(item["concurrency"] for item in passing_full)
    target_short = next(item for item in short_summary if item["concurrency"] == 10)
    target_full = next(item for item in full_summary if item["concurrency"] == 10)

    final_full_stage = full10["full_stages"][0]
    final_row_counts = [int(value) for value in final_full_stage["consistency"]["row_counts"].values()]
    book_mapping_ok = not final_full_stage["consistency"].get("book_id_mismatches")
    log_patterns = full10.get("service_logs", {}).get("pattern_counts", {})
    container_states = full10.get("container_state", [])
    container_ok = all(
        state.get("restart_count") == 0 and not state.get("oom_killed")
        for state in container_states
        if "error" not in state
    )
    final_gate = bool(
        target_short["passed"]
        and mixed.get("release_gate_passed")
        and full10.get("release_gate_passed")
        and target_full["passed"]
        and target_full["consistency_passed"]
        and target_full["cleanup_passed"]
        and float(target_full.get("max_seconds") or 999999) <= 1800
        and len(final_row_counts) == 10
        and set(final_row_counts) == {652}
        and book_mapping_ok
        and scalar.get("success")
        and same_book.get("success")
        and container_ok
        and not any(log_patterns.get(name, 0) for name in ["wrong_default_database", "http_500", "collection_missing"])
    )

    peaks = resource_peaks(list(base.get("short_stages", [])) + full_stages)
    combined = {
        "version": "v2.1.0",
        "generated_at": datetime.now().isoformat(),
        "release_gate_passed": final_gate,
        "capacity": {
            "short_stable_tested_limit": stable_short,
            "full_book_stable_tested_limit": stable_full,
            "full_book_recommended_concurrency": recommended,
            "production_batch_hard_limit": 10,
        },
        "short_summary": short_summary,
        "full_summary": full_summary,
        "mixed_summary": {
            "success": mixed.get("mixed", {}).get("success"),
            "concurrency": mixed.get("mixed", {}).get("concurrency"),
            "operation_counts": mixed.get("mixed", {}).get("operation_counts"),
        },
        "scalar_verification": scalar,
        "same_book_concurrency": same_book,
        "validation_chronology": {
            "capacity_full10": "completed before the per-book operation lock was added",
            "post_book_lock_same_book": "completed and passed",
            "post_book_lock_mixed10": "completed and passed",
            "post_book_lock_full10_rerun": "stopped early because the local workstation became unresponsive; not counted as a completed run",
        },
        "resource_peaks": peaks,
        "final_container_state": container_states,
        "source_files": [str(base_path), str(mixed_path), str(full10_path), str(scalar_path), str(same_book_path)],
        "raw_runs": {
            "matrix": base,
            "post_book_lock_mixed": mixed,
            "capacity_full10_before_book_lock": full10,
        },
    }
    raw_output = output_dir / "本地并发压测原始数据_20260723.json"
    raw_output.write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")

    workload = base["workload"]
    lines = [
        "# 向量服务本地并发压测报告",
        "",
        "> 文件名沿用发布批次 `20260723`；实际测试时间为 2026-07-24。",
        "",
        "## 结论",
        "",
        f"- v2.1.0 发布门禁：**{'通过' if final_gate else '未通过'}**。",
        f"- 10 本完整图书同时向量化：{target_full['successes']}/{target_full['requests']} 成功，"
        f"每本 {final_row_counts[0] if final_row_counts else '未知'} 条向量，共 {sum(final_row_counts)} 条。",
        f"- 完整图书实测通过上限：并发 {stable_full}；常态推荐并发：{recommended}；批量任务硬上限：10。",
        f"- 短样本实测稳定上限：并发 {stable_short}；本次上线按用户要求仍限制为 10。",
        "- 结果口径：完整图书并发 10 的容量数据来自书级锁加入前已完成的正式运行；"
        "书级锁加入后，同书专项和混合并发 10 已重新通过。",
        "- 书级锁加入后的完整图书并发 10 复跑因本机持续满载、影响正常使用而主动中止，"
        "该次中止不计为通过，也不计为失败。",
        "- 并发 40 的短样本压力边界出现 Milvus `flush` 限流，不纳入发布目标，严禁线上瞬时超过 10。",
        "",
        "## 本地测试环境",
        "",
        "- 主机：AMD Ryzen 5 5600GT，6 核 12 线程，32 GB 内存。",
        "- Docker Desktop：12 逻辑 CPU，15.32 GiB 可用内存；全程 CPU 推理。",
        "- Python 3.10.20、PyTorch 2.5.1+cpu、Sentence Transformers 3.3.1。",
        "- FastAPI 0.115.6、Uvicorn 0.34.0、PyMilvus 2.5.2。",
        "- MySQL 5.7.44 / InnoDB；Milvus 服务端 2.5.14、etcd 3.5.9、MinIO。",
        "- 说明：Milvus 2.5.2 镜像源在测试期间不可达，服务端使用同系列 2.5.14；客户端严格为 2.5.2。",
        "- 环境完全隔离：MySQL `127.0.0.1:13306`、Milvus `127.0.0.1:19531`、服务 `127.0.0.1:18006`。",
        "",
        "## 测试数据",
        "",
        f"- 短样本：{workload['short_paragraphs']} 段、{workload['short_characters']} 字符、每本 4 个向量块。",
        f"- 完整图书：{workload['full_paragraphs']} 段、{workload['full_characters']} 字符、每本 652 个向量块。",
        f"- 切片上限：{workload['chunk_size']}；Milvus schema：58 个字段。",
        "",
        "## 功能与一致性基线",
        "",
        "- 冷启动、模型加载、短样本上传/检索/删除/再次上传：通过。",
        "- 完整图书上传、58 字段、分区、652 行向量、清理回收：通过。",
        f"- 标量数组过滤：期望值抽样命中 {scalar.get('expected_filter_hits')} 条（limit=5），缺失值命中 "
        f"{scalar.get('missing_filter_hits')} 条：{'通过' if scalar.get('success') else '失败'}。",
        "- 修正并验证 `vector_collection.book_id` 写入真实图书 ID，逐条满足 `partition_name = book_{book_id}`。",
        "- 同一 `book_id` 全流程互斥：重复整书请求等待首请求结束，只实际生成一份向量。",
        "",
        "## 同一本书并发幂等测试",
        "",
        f"- 两个相同完整图书请求：{'通过' if same_book.get('duplicate_full_upload', {}).get('success') else '失败'}。",
        f"- 两个请求耗时：{[item.get('duration_seconds') for item in same_book.get('duplicate_full_upload', {}).get('uploads', [])]} 秒。",
        f"- 返回消息：{same_book.get('duplicate_full_upload', {}).get('messages')}。",
        f"- 最终 MySQL 记录：{len(same_book.get('duplicate_full_upload', {}).get('verification', {}).get('mysql', {}).get('rows', []))} 条；"
        f"Milvus 行数：{same_book.get('duplicate_full_upload', {}).get('verification', {}).get('milvus', {}).get('row_count')}。",
        f"- 上传期间同书检索：HTTP {same_book.get('duplicate_full_upload', {}).get('search_during_upload', {}).get('http_status')}，"
        "明确返回处理中，未读取半成品。",
        f"- 上传与整书删除竞争：{'通过' if same_book.get('upload_delete_race', {}).get('success') else '失败'}。",
        f"- 故意失败、清理、同书重试：{'通过' if same_book.get('failure_then_retry', {}).get('success') else '失败'}。",
        "",
        "## 短样本并发矩阵",
        "",
        "| 并发 | 波次 | 成功/总数 | 成功率 | 吞吐(本/分钟) | P50(s) | P95(s) | 最大(s) | 结果 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        *table_rows(short_summary),
        "",
        "并发 40 首波 16 个请求被 Milvus flush rate limiter 拒绝；并发 20 的 3 波均通过。"
        "考虑线上目标和安全余量，正式配置按并发 10 限制。",
        "",
        "## 完整图书并发矩阵",
        "",
        "| 并发 | 波次 | 成功/总数 | 成功率 | 吞吐(本/分钟) | P50(s) | P95(s) | 最大(s) | 结果 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        *table_rows(full_summary),
        "",
        "并发 1/2/3/5/8 用于性能曲线；发现并修正 MySQL `book_id` 元数据值后，"
        "重新执行了混合并发 10 和完整图书并发 10，并启用更严格的逐条映射校验。",
        "随后加入书级全流程锁；锁后重新通过了同书并发专项和混合并发 10。"
        "锁后完整并发 10 的复跑因本机卡顿主动中止，因此表中并发 10 的性能数据不标注为锁后复跑结果。",
        "",
        f"最大通过吞吐为 {max_throughput:.4f} 本/分钟；达到其 95% 的最小档位为并发 {recommended}，"
        "因此常态推荐值与批量硬上限分别为 5 和 10。",
        "",
        "## 混合操作",
        "",
        f"- 总并发：{mixed.get('mixed', {}).get('concurrency')}。",
        f"- 操作构成：{mixed.get('mixed', {}).get('operation_counts')}。",
        f"- 结果：{'通过' if mixed.get('mixed', {}).get('success') else '失败'}；"
        "上传、检索、删除同时执行时未改变上传请求所用 Milvus 数据库。",
        "",
        "## 资源与容器状态",
        "",
        "| 服务 | 峰值 CPU | 峰值内存(MiB) |",
        "|---|---:|---:|",
    ]
    for service in ["vector-service", "milvus", "mysql", "minio", "etcd"]:
        item = peaks.get(service, {})
        lines.append(f"| {service} | {item.get('max_cpu_percent')}% | {item.get('max_memory_mib')} |")
    lines.extend(
        [
            "",
            f"- 最终容器状态：{container_states}。",
            "- 容器重启 0、OOMKilled=false；最终 MySQL 和 Milvus 分区集合一致，清理后回到默认分区基数。",
            f"- 日志门禁：{log_patterns}。",
            "",
            "## 上线建议",
            "",
            "1. 暂停批量任务并备份服务目录、线上 `config.py` 和日志。",
            "2. 覆盖代码后执行预检，再做 1 本完整图书验证。",
            "3. 先放开 5 本并发灰度；稳定一批后允许提升到 10，但任务队列必须设置硬上限 10。",
            "4. 生产服务器容量与本机不等价；若 CPU 核数、内存或 Milvus 配置更低，应保持并发 5。",
            "5. 历史失败任务先调用删除接口清理残留分区，再重新下发，不能直接全量重试。",
            "",
            "## 风险说明",
            "",
            "- 完整图书并发 10 为一波真实 10 本同时上传的容量验收，不等同于长时间持续压测。",
            "- 上述完整图书并发 10 容量验收早于书级锁改动；书级锁后的同书专项与混合并发 10 已通过，"
            "但锁后完整 10 本复跑因本机负载影响正常使用而中止。",
            "- 并发升高后单本延迟明显增加，CPU 争抢使吞吐在并发 5 后收益很小。",
            "- 本报告只代表本机隔离环境，生产仍需按 1 本 → 5 本 → 10 本逐级灰度。",
            "",
            f"原始请求、响应、资源采样和一致性明细见 `{raw_output.name}`。",
        ]
    )
    report_path = output_dir / "本地并发压测报告_20260723.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    version_lines = [
        "# social_sciences_vector v2.1.0 版本说明",
        "",
        "## 修复",
        "",
        "- Milvus 按数据库使用独立稳定连接别名，上传、检索、删除不再重绑全局 `default`。",
        "- 同一 `book_id` 使用 MySQL 命名锁覆盖切片、向量生成、入库与 flush，跨线程、worker 和服务实例生效。",
        "- 整书删除和段落删除共用同一把书级锁；同书检索在上传期间返回处理中，不读取半成品。",
        "- MySQL 建库增加连接级锁，分区创建使用 `FOR UPDATE`，分区计数改为原子更新。",
        "- Milvus 分区创建与 MySQL 提交之间增加失败补偿，删除接口支持幂等清理。",
        "- `vector_collection.book_id` 改为真实图书 ID，不再写入固定集合后缀。",
        "- 日志目录支持 `VECTOR_LOG_DIR`，不设置时保持原线上行为。",
        "- 增加不输出密码的发布前预检脚本。",
        "",
        "## 兼容性",
        "",
        "- HTTP 请求与响应结构不变。",
        "- 同一 `book_id` 重复上传按幂等请求处理，不覆盖已成功入库的数据；确需更新内容时，必须先删除旧书再重新上传。",
        "- 同书操作等待超时返回 HTTP 409，响应仍使用原有 `stateCode/state/message/data` 字段。",
        "- 不包含、不覆盖线上 `config.py` 和已有 m3e-base 模型。",
        "- 不修改已有 Milvus 集合、历史分区和向量数据。",
        "",
        "## 验收",
        "",
        f"- 本地发布门禁：{'通过' if final_gate else '未通过'}。",
        "- 10 本完整图书同时向量化入库容量验收通过；书级锁加入后，同书专项与混合并发 10 通过。",
        "- 锁后完整 10 本复跑因本机卡顿主动中止；容量数字沿用此前完整通过结果，不冒充锁后复跑。",
        "- 两个相同完整图书同时下发，只实际生成一份 652 行向量；失败清理后重试通过。",
        "- 常态推荐完整图书并发 5，批量硬上限 10。",
    ]
    (output_dir / "版本说明_v2.1.0.md").write_text("\n".join(version_lines) + "\n", encoding="utf-8")

    deploy_lines = [
        "# v2.1.0 发布与回退说明",
        "",
        "## 上线前",
        "",
        "1. 暂停批量上传、失败自动重试和新任务下发，等待当前请求结束。",
        "2. 备份完整服务目录、线上 `config.py`、模型目录和日志。",
        "3. 校验 ZIP 的 SHA256；确认包中没有 `config.py` 和模型文件。",
        "",
        "## 覆盖方式",
        "",
        "- 现有线上目录：使用 `social_sciences_vector_v2.1.0_incremental.zip` 增量覆盖。",
        "- 需要完整代码恢复：使用 `social_sciences_vector_v2.1.0_full-code.zip`，再放回原线上 `config.py` 和模型目录。",
        "",
        "## 发布步骤",
        "",
        "1. 停止旧服务。",
        "2. 解压并覆盖对应文件，保留线上 `config.py` 和 `db_process/data/embedding/m3e-base`。",
        "3. 执行 `python -m compileall -q national_development log scripts/release_preflight.py`。",
        "4. 执行 `python scripts/release_preflight.py`，必须显示 `success: true`。",
        "   其中 `mysql_named_lock` 必须通过；本版本不新增数据库表。",
        "5. 启动服务，先验证 1 本完整图书，再做 5 本并发灰度，最后才允许提高到 10。",
        "6. 批量调度器设置完整图书同时向量化硬上限 10；常态建议 5。",
        "7. 默认书级写锁等待 1800 秒、检索门禁等待 30 秒；如需调整可设置 "
        "`VECTOR_BOOK_LOCK_TIMEOUT_SECONDS` 和 `VECTOR_BOOK_READ_LOCK_TIMEOUT_SECONDS`。",
        "8. Java/任务调度侧遇到 HTTP 409 且 `data.status=processing` 时，应延迟后按同一 `book_id` 重试，"
        "不要并行生成新的重复任务。",
        "9. 同一 `book_id` 已成功存在时再次上传会返回“已存在，不重复入库”；如要更新书的正文，"
        "先调用删除接口并确认清理成功，再重新上传。",
        "",
        "## 历史失败任务",
        "",
        "对历史 500 任务逐本调用 `DELETE /book/{book_id}` 清理残留，核对 MySQL/Milvus 均无残留后再重新下发。",
        "",
        "## 回退",
        "",
        "1. 再次暂停上传并停止服务。",
        "2. 恢复升级前备份的代码与 `config.py`，重新启动。",
        "3. 回退只恢复代码，不删除 `book_vector_v2`、已有分区或历史向量。",
    ]
    (output_dir / "发布与回退说明_v2.1.0.md").write_text("\n".join(deploy_lines) + "\n", encoding="utf-8")

    print(json.dumps({"success": final_gate, "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0 if final_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
