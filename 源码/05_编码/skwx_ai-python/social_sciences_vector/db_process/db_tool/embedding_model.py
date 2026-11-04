#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Shared embedding model with stable GPU selection and CPU fallback."""

import gc
import os
import threading

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from db_process.config.constant_config import M3E_BAS0E_EMBEDDING_MODEL_LOCAL_DICT
from log.logger import Logger


logger = Logger()
MODEL_PATH = M3E_BAS0E_EMBEDDING_MODEL_LOCAL_DICT["m3e-base"]
DEVICE_ENV_NAME = "VECTOR_EMBEDDING_DEVICE"
STARTUP_PROBE_TEXT = "GPU向量计算启动检查"

_MODEL_LOCK = threading.RLock()
_model_m3e_base = None
_active_device = "cpu"
_gpu_name = ""
_fallback_reason = ""


def _cuda_device_details(index):
    with torch.cuda.device(index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        "index": index,
        "device": f"cuda:{index}",
        "name": torch.cuda.get_device_name(index),
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
    }


def _select_device(requested_device):
    requested = str(requested_device or "auto").strip().lower()
    if requested == "cpu":
        return "cpu", None, "按配置使用CPU"

    if requested == "auto":
        if not torch.cuda.is_available() or torch.cuda.device_count() <= 0:
            return "cpu", None, "CUDA不可用"
        candidates = [_cuda_device_details(index) for index in range(torch.cuda.device_count())]
        selected = max(candidates, key=lambda item: item["free_bytes"])
        return selected["device"], selected, ""

    if requested == "cuda":
        requested = "cuda:0"
    if requested.startswith("cuda:"):
        try:
            index = int(requested.split(":", 1)[1])
        except ValueError:
            return "cpu", None, f"无效设备配置: {requested_device}"
        if not torch.cuda.is_available():
            return "cpu", None, "CUDA不可用"
        if index < 0 or index >= torch.cuda.device_count():
            return "cpu", None, f"GPU编号不存在: {requested}"
        details = _cuda_device_details(index)
        return details["device"], details, ""

    return "cpu", None, f"无效设备配置: {requested_device}"


def _load_model(device):
    return SentenceTransformer(MODEL_PATH, device=device)


def _encode(model, sentences, device, batch_size, normalize):
    return model.encode(
        sentences,
        batch_size=batch_size,
        device=device,
        normalize_embeddings=normalize,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def _validate_embeddings(embeddings):
    values = np.asarray(embeddings)
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise RuntimeError(f"向量维度异常: {values.shape}")
    if not np.isfinite(values).all():
        raise RuntimeError("向量包含非有限数值")
    return values


def _initialize_model():
    global _model_m3e_base, _active_device, _gpu_name, _fallback_reason

    requested = os.environ.get(DEVICE_ENV_NAME, "auto")
    try:
        selected_device, gpu_details, selection_reason = _select_device(requested)
    except Exception as exc:
        selected_device, gpu_details = "cpu", None
        selection_reason = f"GPU信息读取失败: {exc}"
    if selected_device.startswith("cuda"):
        try:
            candidate_model = _load_model(selected_device)
            _validate_embeddings(_encode(candidate_model, [STARTUP_PROBE_TEXT], selected_device, 1, False))
            _model_m3e_base = candidate_model
            _active_device = selected_device
            _gpu_name = gpu_details["name"] if gpu_details else ""
            _fallback_reason = ""
            logger.info(f"向量模型启动检查成功: 设备=GPU({_active_device}), 显卡={_gpu_name}")
            return
        except Exception as exc:
            _fallback_reason = f"GPU启动检查失败: {exc}"
            logger.warning(f"GPU启动检查失败，已使用CPU: {exc}")
            _model_m3e_base = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
    else:
        _fallback_reason = selection_reason
        if requested.strip().lower() == "cpu":
            logger.info("向量模型启动检查成功: 设备=CPU（按配置指定）")
        else:
            logger.warning(f"GPU启动检查失败，已使用CPU: {selection_reason}")

    _model_m3e_base = _load_model("cpu")
    _validate_embeddings(_encode(_model_m3e_base, [STARTUP_PROBE_TEXT], "cpu", 1, False))
    _active_device = "cpu"
    _gpu_name = ""


def _is_cuda_failure(exc):
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    if not isinstance(exc, RuntimeError):
        return False
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("cuda", "cublas", "cudnn", "device-side", "out of memory")
    )


def _fallback_to_cpu(exc, log_context):
    global _model_m3e_base, _active_device, _gpu_name, _fallback_reason

    previous_device = _active_device
    _model_m3e_base = None
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    _model_m3e_base = _load_model("cpu")
    _active_device = "cpu"
    _gpu_name = ""
    _fallback_reason = f"{previous_device}运行失败: {exc}"
    context = f"，任务={log_context}" if log_context else ""
    logger.warning(f"GPU向量化失败，已切换CPU继续运行{context}: {exc}")


def get_embedding_device_info():
    requested = os.environ.get(DEVICE_ENV_NAME, "auto")
    device_type = "GPU" if _active_device.startswith("cuda") else "CPU"
    display = f"GPU({_active_device})" if device_type == "GPU" else "CPU"
    return {
        "requested": requested,
        "device": _active_device,
        "device_type": device_type,
        "display": display,
        "gpu_name": _gpu_name,
        "fallback_reason": _fallback_reason,
    }


def text2vec(sentences, model_type, batch_size=64, normalize=False, log_context=None):
    if model_type != "m3e-base":
        raise ValueError(f"Unsupported embedding model: {model_type}")

    with _MODEL_LOCK:
        try:
            embeddings = _encode(
                _model_m3e_base, sentences, _active_device, batch_size, normalize
            )
        except Exception as exc:
            if not _active_device.startswith("cuda") or not _is_cuda_failure(exc):
                raise
            _fallback_to_cpu(exc, log_context)
            embeddings = _encode(
                _model_m3e_base, sentences, "cpu", batch_size, normalize
            )
        return _validate_embeddings(embeddings)


def semantic_model():
    pass


def is_normalized_vector(v, tolerance=1e-6):
    norm = np.linalg.norm(v)
    return np.abs(norm - 1) < tolerance


_initialize_model()


if __name__ == "__main__":
    samples = ["GPU向量计算测试", "社科图书向量发布"]
    vectors = text2vec(samples, "m3e-base", normalize=False)
    print(get_embedding_device_info())
    print(vectors.shape)
