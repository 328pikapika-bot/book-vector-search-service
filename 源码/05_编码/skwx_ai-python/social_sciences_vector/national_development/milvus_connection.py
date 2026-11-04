import hashlib
import re
import threading

from pymilvus import connections

from config import milvus_host, milvus_port


_CONNECTION_LOCK = threading.RLock()


def _connection_alias(database_name, host, port):
    """Return a stable alias that is unique for one Milvus database."""
    database_name = str(database_name or "default").strip() or "default"
    safe_database_name = re.sub(r"[^0-9A-Za-z_]+", "_", database_name).strip("_") or "default"
    digest = hashlib.sha1(f"{host}:{port}/{database_name}".encode("utf-8")).hexdigest()[:10]
    return f"social_sciences_{safe_database_name[:24]}_{digest}"


def connect_milvus(database_name, host=None, port=None):
    """
    Connect to a Milvus database without rebinding PyMilvus' global
    ``default`` alias.

    Synchronous FastAPI endpoints run concurrently in a thread pool even when
    Uvicorn has only one worker.  A stable alias per database prevents one
    request from changing the database used by another request.
    """
    host = host or milvus_host
    port = port or milvus_port
    database_name = str(database_name or "default").strip() or "default"
    alias = _connection_alias(database_name, host, port)

    with _CONNECTION_LOCK:
        if not connections.has_connection(alias):
            connections.connect(
                alias=alias,
                host=host,
                port=port,
                db_name=database_name,
            )

    return alias
