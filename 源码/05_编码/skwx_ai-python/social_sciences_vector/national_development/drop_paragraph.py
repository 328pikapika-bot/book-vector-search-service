from datetime import datetime

import mysql.connector
import uvicorn
from fastapi import FastAPI, APIRouter
from pymilvus import Collection
from pymilvus.orm import utility
from starlette.responses import JSONResponse

from config import milvus_host, milvus_port, mysql_host, mysql_user, mysql_password, mysql_database
from national_development.book_operation_lock import (
    BookOperationLockTimeout,
    book_operation_lock,
)
from national_development.milvus_connection import connect_milvus

app_drop_paragraph = APIRouter()

def connection_mysql():
    db_connection = mysql.connector.connect(
        host=mysql_host,  # MySQL 服务器地址（例如：localhost、IP 地址等）
        user=mysql_user,  # 用户名
        password=mysql_password,  # 密码
        database=mysql_database  # 要连接的数据库名称
    )
    return db_connection

def _drop_paragraph_unlocked(book_id: str, paragraph_id: str):
    try:
        out_library_id = book_id
        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%dT%H:%M:%S.%f+00:00')

        # 连接到 MySQL 获取库名
        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            try:
                # 查询页面上的library_id
                sql = "SELECT book_id FROM vector_collection WHERE partition_name = %s;"
                cursor.execute(sql, (f"book_{out_library_id}",))
                library_book_id = cursor.fetchone()[0]

                sql_library_name = "SELECT name, name_zh FROM vector_library WHERE book_id = %s;"
                cursor.execute(sql_library_name, (library_book_id,))
                out_all =cursor.fetchone()
                out_library_name = out_all[0]  # 确保读取所有结果
                out_library_name_zh = out_all[1]  # 确保读取所有结果
            except Exception as e:
                return JSONResponse({"authorization": False, "stateCode": 1, "state": False, "message": str(e), "data": "请检查是否存在该paragraphId"})

        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            # 查询自己的 library_id
            library_id = "SELECT library_id FROM vector_library WHERE book_id = %s;"
            cursor.execute(library_id, (library_book_id,))
            in_library_id = cursor.fetchone()[0]  # 确保读取所有结果
            # 查询自己的 library_name
            library_time = "SELECT name FROM vector_library_base WHERE id = %s;"
            cursor.execute(library_time, (in_library_id,))
            library_name = cursor.fetchone()[0]  # 确保读取所有结果

        # 连接到 Milvus 的指定库
        milvus_alias = connect_milvus(library_name)
        collection = Collection(out_library_name, using=milvus_alias)

        collection.delete(
            expr=f"directory_id == '{paragraph_id}'"
        )

        result = {
            "authorization": "true",
            "stateCode": 0,
            "state": "true",
            "message": "null",
            "data": {
                "name": out_library_name_zh,
                "type": "Standard",
                "libraryId": out_library_id,
                "total": 0,
                "readonly": "false",
                "ready": "false",
                "createTime": formatted_time,
                "modifyTime": "null",
                "paragraph_id": paragraph_id
            }
        }
        return JSONResponse(result)
    except Exception as e:
        print(f"Error: {e}")
        return {"authorization": False, "stateCode": 1, "state": False, "message": str(e), "data": "请检查是否存在该paragraphId"}


@app_drop_paragraph.delete("/{book_id}/paragraph/{paragraph_id}")
def drop_paragraph(book_id: str, paragraph_id: str):
    try:
        with book_operation_lock(book_id) as lock_handle:
            lock_handle.ensure_held()
            return _drop_paragraph_unlocked(book_id, paragraph_id)
    except BookOperationLockTimeout as exc:
        return JSONResponse(
            {
                "authorization": False,
                "stateCode": 1,
                "state": False,
                "message": str(exc),
                "data": {"book_id": book_id, "status": "processing"},
            },
            status_code=409,
        )
