#!/usr/bin/env python
"""Focused tests for Milvus partition capacity reconciliation."""

from __future__ import annotations

import importlib.util
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock


MODULE_PATH = Path(__file__).resolve().parents[1] / "national_development" / "create_collection.py"
CREATE_LIBRARY_PATH = Path(__file__).resolve().parents[1] / "national_development" / "create_library.py"


class FakeCursor:
    def __init__(self, partition_count, updates):
        self.partition_count = partition_count
        self.updates = updates
        self.sql = ""

    def execute(self, sql, params=None):
        self.sql = " ".join(sql.split())
        if self.sql.startswith("UPDATE vector_library SET partition_count"):
            self.updates.append(params)

    def fetchone(self):
        if "FROM vector_library WHERE id=" in self.sql:
            return "base-id", "book_vector_v2", 1024, self.partition_count
        if "FROM vector_library_base" in self.sql:
            return "base-id", "ssc_1"
        return None


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.commits = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.commits += 1


class FakeCollection:
    partition_total = 1024
    created = []

    def __init__(self, name=None, using=None, schema=None):
        self.name = name
        self.partitions = [object()] * self.partition_total

    def create_partition(self, name):
        self.created.append(name)

    def drop_partition(self, name):
        return None


def load_module(partition_count):
    updates = []
    cursor = FakeCursor(partition_count, updates)
    connection = FakeConnection(cursor)

    pymilvus = types.ModuleType("pymilvus")
    pymilvus.Collection = FakeCollection
    pymilvus.utility = types.SimpleNamespace(has_collection=lambda *args, **kwargs: True)
    pymilvus.FieldSchema = object
    pymilvus.DataType = object
    pymilvus.CollectionSchema = lambda fields, description=None: object()

    exceptions = types.ModuleType("pymilvus.exceptions")
    exceptions.PartitionAlreadyExistException = RuntimeError
    response = types.ModuleType("starlette.responses")
    response.JSONResponse = dict
    config = types.ModuleType("config")
    for name in ("mysql_host", "mysql_user", "mysql_password", "mysql_database", "milvus_host", "milvus_port"):
        setattr(config, name, "test")
    connector = types.ModuleType("mysql.connector")
    connector.connect = lambda **kwargs: connection
    mysql = types.ModuleType("mysql")
    mysql.connector = connector
    pypinyin = types.ModuleType("pypinyin")
    pypinyin.lazy_pinyin = lambda value: [value]
    create_library = types.ModuleType("national_development.create_library")
    create_library.create_library = lambda *args, **kwargs: None
    constants = types.ModuleType("db_process.config.constant_config")
    constants.CONSISTENCY_LEVEL = "Bounded"
    embedding = types.ModuleType("db_process.db_tool.embedding_model")
    embedding.text2vec = lambda *args, **kwargs: None
    fields_module = types.ModuleType("national_development.field")
    fields_module.fields = []
    milvus_connection = types.ModuleType("national_development.milvus_connection")
    milvus_connection.connect_milvus = lambda database: "test-alias"

    modules = {
        "pymilvus": pymilvus,
        "pymilvus.exceptions": exceptions,
        "starlette.responses": response,
        "config": config,
        "mysql": mysql,
        "mysql.connector": connector,
        "pypinyin": pypinyin,
        "national_development.create_library": create_library,
        "db_process.config.constant_config": constants,
        "db_process.db_tool.embedding_model": embedding,
        "national_development.field": fields_module,
        "national_development.milvus_connection": milvus_connection,
    }
    module_name = f"create_collection_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module, connection, updates


def load_create_library_module(milvus_names):
    pymilvus = types.ModuleType("pymilvus")
    pymilvus.db = object()
    pymilvus.utility = types.SimpleNamespace(list_collections=lambda using=None: milvus_names)
    pymilvus.Collection = object
    response = types.ModuleType("starlette.responses")
    response.JSONResponse = dict
    config = types.ModuleType("config")
    config.milvus_collection_name = "book_vector_v2"
    for name in ("mysql_host", "mysql_user", "mysql_password", "mysql_database", "milvus_host", "milvus_port"):
        setattr(config, name, "test")
    connector = types.ModuleType("mysql.connector")
    mysql = types.ModuleType("mysql")
    mysql.connector = connector
    pypinyin = types.ModuleType("pypinyin")
    pypinyin.lazy_pinyin = lambda value: [value]
    milvus_connection = types.ModuleType("national_development.milvus_connection")
    milvus_connection.connect_milvus = lambda database: "test-alias"

    modules = {
        "pymilvus": pymilvus,
        "starlette.responses": response,
        "config": config,
        "mysql": mysql,
        "mysql.connector": connector,
        "pypinyin": pypinyin,
        "national_development.milvus_connection": milvus_connection,
    }
    module_name = f"create_library_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, CREATE_LIBRARY_PATH)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class NameCursor:
    def __init__(self, names):
        self.names = names

    def execute(self, sql):
        return None

    def fetchall(self):
        return [(name,) for name in self.names]


class CollectionCapacityTests(unittest.TestCase):
    def setUp(self):
        FakeCollection.created = []

    def test_full_collection_is_reconciled_and_rejected(self):
        FakeCollection.partition_total = 1024
        module, connection, updates = load_module(partition_count=1000)
        with self.assertRaises(module.CollectionCapacityExceeded):
            module.create_collection("title", "80784591", "library-id", [])
        self.assertEqual(updates, [(1024, "library-id")])
        self.assertEqual(connection.commits, 1)
        self.assertEqual(FakeCollection.created, [])

    def test_last_available_partition_updates_to_actual_total(self):
        FakeCollection.partition_total = 1023
        module, connection, updates = load_module(partition_count=1000)
        module.create_collection("title", "80784591", "library-id", [])
        self.assertEqual(updates, [(1024, "library-id")])
        self.assertEqual(FakeCollection.created, ["book_80784591"])
        self.assertGreaterEqual(connection.commits, 1)

    def test_full_prefix_rolls_to_second_collection(self):
        module = load_create_library_module(["book_vector_v2"])
        name = module.next_configured_collection_name(
            NameCursor(["book_vector_v2"]), "book_vector_v2", "alias"
        )
        self.assertEqual(name, "book_vector_v2_2")

    def test_second_collection_rolls_to_third_collection(self):
        names = ["book_vector_v2", "book_vector_v2_2"]
        module = load_create_library_module(names)
        name = module.next_configured_collection_name(NameCursor(names), "book_vector_v2", "alias")
        self.assertEqual(name, "book_vector_v2_3")


if __name__ == "__main__":
    unittest.main()
