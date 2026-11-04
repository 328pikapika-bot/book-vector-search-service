from datetime import datetime
from config import mysql_user, mysql_host, mysql_password, mysql_database, milvus_host, milvus_port
import mysql.connector
import uvicorn
from fastapi import FastAPI, APIRouter
from pymilvus import Collection, Partition
from pymilvus.orm import utility
from starlette.responses import JSONResponse
from national_development.book_operation_lock import (
    BookOperationLockTimeout,
    book_operation_lock,
)
from national_development.milvus_connection import connect_milvus

app_drop_collection = APIRouter()

def connection_mysql():
    db_connection = mysql.connector.connect(
        host=mysql_host,  # MySQL 服务器地址（例如：localhost、IP 地址等）
        user=mysql_user,  # 用户名
        password=mysql_password,  # 密码
        database=mysql_database  # 要连接的数据库名称
    )
    return db_connection

def _drop_collection_unlocked(book_id: str):
    # print(book_id)
    try:
        # # 连接到 Milvus 服务器
        # connections.connect(host=milvus_host, port=milvus_port)

        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%dT%H:%M:%S.%f+00:00')

        # 连接到 MySQL 获取库名
        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            # 查询 collection_name
            sql_collection_name = "SELECT name, name_zh FROM vector_collection WHERE partition_name = %s;"
            cursor.execute(sql_collection_name, (f"book_{book_id}",))
            collection_name_out_name_zh = cursor.fetchone()
            if collection_name_out_name_zh is None:
                return {
                    "authorization": False,
                    "stateCode": 1,
                    "state": False,
                    "message": f"vector_collection not found for book_{book_id}",
                    "data": "请检查是否存在该bookId"
                }
            collection_name = collection_name_out_name_zh[0]  # 确保读取所有结果
            out_name_zh = collection_name_out_name_zh[1]
            db_connection.close()
        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            # 查询 create_time
            collection_time = "SELECT create_time FROM vector_collection WHERE partition_name = %s;"
            cursor.execute(collection_time, (f"book_{book_id}",))
            collection_create_time = cursor.fetchone()[0]  # 确保读取所有结果
            db_connection.close()
        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            # 查询 library_id
            sql_library_id = "SELECT library_id FROM vector_collection WHERE partition_name = %s;"
            cursor.execute(sql_library_id, (f"book_{book_id}",))
            library_id_result = cursor.fetchone()[0]  # 读取结果
            db_connection.close()

        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            # 传递查询到的 library_id 作为参数
            if library_id_result:  # 确保查询有结果
                out_library_id = library_id_result  # 假设返回的是一个元组
                sql_library_id = "SELECT library_id FROM vector_library WHERE id = %s;"
                cursor.execute(sql_library_id, (out_library_id,))
                in_library_id = cursor.fetchone()[0]
                print(in_library_id)
                sql_library_name = "SELECT name FROM vector_library_base WHERE id = %s;"
                cursor.execute(sql_library_name, (in_library_id,))
                library_name = cursor.fetchone()[0]  # 读取结果
            else:
                print("No library_id found")
            db_connection.close()

        print(f"collection name retrieved: {library_name}")

        # 连接到 Milvus 的指定库
        milvus_alias = connect_milvus(library_name)

        print(collection_name)
        # query_expr = f"title == '{collection_name}'"
        # collection_text = collection.query(query_expr, output_fields=["text"])

        # 删除指定的 collection
        print(f"Dropping partition: book_{book_id}")
        partition_name = f"book_{book_id}"
        if utility.has_collection(collection_name, using=milvus_alias):
            collection = Collection(collection_name, using=milvus_alias)
            if collection.has_partition(partition_name):
                partition = Partition(collection, partition_name)
                partition.release()
                collection.drop_partition(partition_name)
            else:
                print(f"Partition already missing: {partition_name}")
        else:
            print(f"Collection already missing: {collection_name}")

        # 使用上下文管理器处理数据库连接和游标
        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            sql = """
                DELETE FROM vector_collection WHERE partition_name = %s;
            """
            cursor.execute(sql, (f"book_{book_id}",))
            db_connection.commit()  # 确保提交操作

            sql = """
                UPDATE vector_library
                SET partition_count = GREATEST(partition_count - 1, 1)
                WHERE name=%s;
            """
            cursor.execute(sql, (collection_name,))
            db_connection.commit()
        # collection.load()


        result = {
            "authorization": "true",
            "stateCode": 0,
            "state": "true",
            "message": "null",
            "data": {
                "name": out_name_zh,
                "type": "Standard",
                "libraryId": out_library_id,
                "total": 0,
                "readonly": "false",
                "ready": "false",
                "createTime": collection_create_time,
                "modifyTime": formatted_time,
                "book_id": book_id
            }
        }

        return JSONResponse(result)
    except Exception as e:
        print(f"An error occurred: {e}")
        return {"authorization": False, "stateCode": 1, "state": False, "message": str(e), "data": "请检查是否存在该bookId"}


@app_drop_collection.delete("/book/{book_id}")
def drop_collection(book_id: str):
    try:
        with book_operation_lock(book_id) as lock_handle:
            lock_handle.ensure_held()
            return _drop_collection_unlocked(book_id)
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

