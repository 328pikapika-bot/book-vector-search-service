import hashlib
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass

import mysql.connector

from config import mysql_database, mysql_host, mysql_password, mysql_user


DEFAULT_WRITE_TIMEOUT_SECONDS = 1800
DEFAULT_READ_TIMEOUT_SECONDS = 30


class BookOperationLockTimeout(TimeoutError):
    """Raised when one book is still being mutated by another request."""


class BookOperationLockLost(RuntimeError):
    """Raised when the MySQL session holding a book lock is lost."""


def _positive_int_from_env(name, default):
    try:
        value = int(os.getenv(name, str(default)))
        return max(0, value)
    except (TypeError, ValueError):
        return default


def write_lock_timeout_seconds():
    return _positive_int_from_env(
        "VECTOR_BOOK_LOCK_TIMEOUT_SECONDS",
        DEFAULT_WRITE_TIMEOUT_SECONDS,
    )


def read_lock_timeout_seconds():
    return _positive_int_from_env(
        "VECTOR_BOOK_READ_LOCK_TIMEOUT_SECONDS",
        DEFAULT_READ_TIMEOUT_SECONDS,
    )


def lock_name_for_book(book_id):
    book_id = str(book_id or "").strip()
    if not book_id:
        raise ValueError("book_id cannot be empty")
    digest = hashlib.sha256(book_id.encode("utf-8")).hexdigest()[:48]
    return f"vector_book_{digest}"


def _connection_mysql():
    return mysql.connector.connect(
        host=mysql_host,
        user=mysql_user,
        password=mysql_password,
        database=mysql_database,
        connection_timeout=10,
        autocommit=True,
    )


@dataclass
class BookLockHandle:
    book_id: str
    lock_name: str
    connection: object
    owner_connection_id: int
    wait_seconds: float

    def ensure_held(self):
        """Fail before a write if the connection-scoped MySQL lock was lost."""
        try:
            self.connection.ping(reconnect=False, attempts=1, delay=0)
            cursor = self.connection.cursor()
            try:
                cursor.execute("SELECT IS_USED_LOCK(%s)", (self.lock_name,))
                owner = cursor.fetchone()[0]
            finally:
                cursor.close()
        except Exception as exc:
            raise BookOperationLockLost(
                f"book operation lock connection lost for {self.book_id}"
            ) from exc
        if owner is None or int(owner) != int(self.owner_connection_id):
            raise BookOperationLockLost(
                f"book operation lock is no longer held for {self.book_id}"
            )


@contextmanager
def book_operation_lock(book_id, timeout_seconds=None):
    """
    Hold one MySQL named lock for a book across an entire mutation.

    Named locks are connection-scoped, so this works across FastAPI threads,
    Uvicorn workers and multiple algorithm-service instances that use the
    same MySQL server.
    """
    book_id = str(book_id or "").strip()
    lock_name = lock_name_for_book(book_id)
    timeout_seconds = (
        write_lock_timeout_seconds() if timeout_seconds is None else max(0, int(timeout_seconds))
    )
    connection = _connection_mysql()
    acquired = False
    cursor = connection.cursor()
    started = time.perf_counter()
    try:
        cursor.execute("SELECT GET_LOCK(%s, %s)", (lock_name, timeout_seconds))
        row = cursor.fetchone()
        acquired = bool(row and row[0] == 1)
        if not acquired:
            raise BookOperationLockTimeout(
                f"book {book_id} is still being processed; lock wait timed out after {timeout_seconds}s"
            )
        cursor.execute("SELECT CONNECTION_ID()")
        owner_connection_id = int(cursor.fetchone()[0])
        handle = BookLockHandle(
            book_id=book_id,
            lock_name=lock_name,
            connection=connection,
            owner_connection_id=owner_connection_id,
            wait_seconds=round(time.perf_counter() - started, 3),
        )
        yield handle
    finally:
        if acquired:
            try:
                cursor.execute("SELECT RELEASE_LOCK(%s)", (lock_name,))
                cursor.fetchone()
            except Exception:
                # Closing the dedicated connection also releases the named lock.
                pass
        try:
            cursor.close()
        except Exception:
            pass
        connection.close()


def wait_until_book_idle(book_id, timeout_seconds=None):
    """Wait for an active mutation and immediately release the gate."""
    timeout_seconds = (
        read_lock_timeout_seconds() if timeout_seconds is None else timeout_seconds
    )
    with book_operation_lock(book_id, timeout_seconds=timeout_seconds) as handle:
        return handle.wait_seconds
