#!/usr/bin/env python
"""Unit tests for GPU selection and the one-way CPU fallback."""

from __future__ import annotations

import contextlib
import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = Path(__file__).resolve().parents[1] / "db_process" / "db_tool" / "embedding_model.py"


class FakeCuda:
    class OutOfMemoryError(RuntimeError):
        pass

    def __init__(self, available=True, free_memory=None):
        self.available = available
        self.free_memory = free_memory or [1024]
        self.current_index = 0

    def is_available(self):
        return self.available

    def device_count(self):
        return len(self.free_memory) if self.available else 0

    @contextlib.contextmanager
    def device(self, index):
        previous = self.current_index
        self.current_index = index
        try:
            yield
        finally:
            self.current_index = previous

    def mem_get_info(self):
        free = self.free_memory[self.current_index]
        return free, free * 2

    def get_device_name(self, index):
        return f"Fake GPU {index}"

    def empty_cache(self):
        return None


class FakeModel:
    def __init__(self, device, fail_gpu_startup=False, fail_gpu_runtime=False):
        self.device = device
        self.calls = 0
        self.fail_gpu_startup = fail_gpu_startup
        self.fail_gpu_runtime = fail_gpu_runtime

    def encode(self, sentences, **kwargs):
        self.calls += 1
        device = kwargs["device"]
        if device.startswith("cuda") and self.fail_gpu_startup:
            raise RuntimeError("CUDA startup failure")
        if device.startswith("cuda") and self.fail_gpu_runtime and self.calls > 1:
            raise FakeCuda.OutOfMemoryError("CUDA out of memory")
        return np.ones((len(sentences), 768), dtype=np.float32)


def load_embedding_module(cuda, fail_gpu_startup=False, fail_gpu_runtime=False, requested="auto"):
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = cuda
    fake_torch.version = types.SimpleNamespace(cuda="12.1")
    fake_torch.__version__ = "test"

    sentence_module = types.ModuleType("sentence_transformers")
    sentence_module.SentenceTransformer = lambda path, device: FakeModel(
        device,
        fail_gpu_startup=fail_gpu_startup,
        fail_gpu_runtime=fail_gpu_runtime,
    )

    config_module = types.ModuleType("db_process.config.constant_config")
    config_module.M3E_BAS0E_EMBEDDING_MODEL_LOCAL_DICT = {"m3e-base": "/fake/model"}

    logger_module = types.ModuleType("log.logger")
    logger_module.Logger = lambda: types.SimpleNamespace(info=lambda message: None, warning=lambda message: None)

    module_name = f"embedding_model_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    modules = {
        "torch": fake_torch,
        "sentence_transformers": sentence_module,
        "db_process.config.constant_config": config_module,
        "log.logger": logger_module,
    }
    with mock.patch.dict(sys.modules, modules), mock.patch.dict(
        "os.environ", {"VECTOR_EMBEDDING_DEVICE": requested}, clear=False
    ):
        spec.loader.exec_module(module)
    return module


class EmbeddingDeviceTests(unittest.TestCase):
    def test_auto_selects_gpu_with_most_free_memory(self):
        module = load_embedding_module(FakeCuda(True, [100, 500]))
        self.assertEqual(module.get_embedding_device_info()["device"], "cuda:1")

    def test_no_cuda_uses_cpu(self):
        module = load_embedding_module(FakeCuda(False))
        self.assertEqual(module.get_embedding_device_info()["device"], "cpu")

    def test_explicit_cpu_uses_cpu(self):
        module = load_embedding_module(FakeCuda(True), requested="cpu")
        self.assertEqual(module.get_embedding_device_info()["device"], "cpu")

    def test_gpu_probe_failure_uses_cpu(self):
        module = load_embedding_module(FakeCuda(True), fail_gpu_startup=True)
        self.assertEqual(module.get_embedding_device_info()["device"], "cpu")

    def test_runtime_oom_permanently_falls_back_to_cpu(self):
        module = load_embedding_module(FakeCuda(True), fail_gpu_runtime=True)
        result = module.text2vec(["test"], "m3e-base", log_context="unit-test")
        self.assertEqual(result.shape, (1, 768))
        self.assertEqual(module.get_embedding_device_info()["device"], "cpu")
        module.text2vec(["second"], "m3e-base")
        self.assertEqual(module.get_embedding_device_info()["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
