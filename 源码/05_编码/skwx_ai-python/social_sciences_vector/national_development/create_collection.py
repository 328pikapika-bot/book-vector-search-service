import json
import random
import uuid
from datetime import datetime

from pymilvus.exceptions import PartitionAlreadyExistException
from starlette.responses import JSONResponse

from config import mysql_host, mysql_user, mysql_password, mysql_database, milvus_host, milvus_port
import mysql.connector
import pypinyin
from national_development.create_library import create_library
from pymilvus import Collection, utility, FieldSchema, DataType, CollectionSchema
from db_process.config.constant_config import CONSISTENCY_LEVEL
from db_process.db_tool.embedding_model import text2vec
from national_development.field import fields
from national_development.milvus_connection import connect_milvus


EXPECTED_FIELD_NAMES = {field.name for field in fields}


class CollectionCapacityExceeded(RuntimeError):
    """The selected collection became full before its partition was created."""


def validate_existing_collection_schema(collection_name, using):
    collection = Collection(name=collection_name, using=using)
    existing_field_names = {field.name for field in collection.schema.fields}
    missing_fields = sorted(EXPECTED_FIELD_NAMES - existing_field_names)
    if missing_fields:
        raise RuntimeError(
            f"Milvus collection '{collection_name}' already exists with an old schema. "
            f"Missing fields: {', '.join(missing_fields)}. "
            "Create a new vector_library row/collection or migrate the existing collection schema."
        )
    return collection


def connection_mysql():
    # 创建数据库连接
    db_connection = mysql.connector.connect(
        host=mysql_host,  # MySQL 服务器地址（例如：localhost、IP 地址等）
        user=mysql_user,  # 用户名
        password=mysql_password,  # 密码
        database=mysql_database  # 要连接的数据库名称
    )
    return db_connection


def convert_to_pinyin(name):
    # 将中文字符串转换为拼音
    return ''.join(pypinyin.lazy_pinyin(name))


def split_text(text, chunk_size=1000):
    # 按照指定大小切割文本
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]


def generate_unique_random_integers(n):
    unique_numbers = set()
    while len(unique_numbers) < n:
        random_int = random.randint(10**17, 10**18 - 1)
        unique_numbers.add(random_int)  # set 自动去重
    return list(unique_numbers)


def create_collection(collection_name, book_id, library_id, paragraphs):
    collection = None
    partition_created = False
    partition_name = f"book_{book_id}"
    try:
        data = {"name": collection_name, "book_id": book_id, "paragraphs": paragraphs}

        id = "".join(str(num) for num in generate_unique_random_integers(1))
        data_collection_id = str(uuid.uuid4())
        now = datetime.now()
        formatted_time = now.strftime('%Y-%m-%dT%H:%M:%S.%f+00:00')
        name_zh = data['name']

        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            # 锁住目标 vector_library 行，串行化集合首建、分区创建和计数更新。
            # 这样多个并发上传不会读取到相同的 partition_count 后互相覆盖。
            sql = """SELECT library_id, name, partition_count_max, partition_count
                     FROM vector_library WHERE id=%s FOR UPDATE;
            """
            cursor.execute(sql, (library_id,))
            library_row = cursor.fetchone()
            if library_row is None:
                raise RuntimeError(f"vector_library not found: {library_id}")
            in_library_id, collection_name, partition_count_max, partition_count = library_row

            sql = "SELECT * FROM vector_library_base WHERE id = %s;"
            cursor.execute(sql, (in_library_id,))
            all_base = cursor.fetchone()
            if all_base is None:
                raise RuntimeError(f"vector_library_base not found: {in_library_id}")

            milvus_alias = connect_milvus(all_base[1])

            if partition_count == 1:
                # 创建collection
                schema = CollectionSchema(fields, description="所有图书合并的collection")
                if utility.has_collection(collection_name, using=milvus_alias):
                    collection = validate_existing_collection_schema(collection_name, milvus_alias)
                else:
                    # 创建集合
                    collection = Collection(name=collection_name, schema=schema, using=milvus_alias)

                    # 创建索引
                    temp_params = {
                        "index_type": "IVF_FLAT",
                        "metric_type": "COSINE",
                        "params": {"nlist": 1024}
                    }

                    index_name = collection_name + "_index"
                    collection.create_index(field_name="embedding", index_params=temp_params, index_name=index_name)
                    collection.load()

                    sql = """
                    UPDATE vector_library_base
                    SET
                        collection_count = collection_count + 1
                    WHERE id = %s
                    """
                    cursor.execute(sql, (all_base[0],))
            else:
                collection = Collection(name=collection_name, using=milvus_alias)

            # Capacity may have changed after create_library selected this row.
            # Milvus is authoritative because it enforces the hard partition limit.
            actual_partition_count = len(collection.partitions)
            if actual_partition_count >= partition_count_max:
                cursor.execute(
                    "UPDATE vector_library SET partition_count=%s WHERE id=%s",
                    (actual_partition_count, library_id),
                )
                db_connection.commit()
                raise CollectionCapacityExceeded(
                    f"Collection '{collection_name}' is full: "
                    f"{actual_partition_count}/{partition_count_max} partitions"
                )

            collection.create_partition(partition_name)
            partition_created = True
            sql = """
                UPDATE vector_library
                SET partition_count = %s
                WHERE id=%s
            """

            cursor.execute(sql, (actual_partition_count + 1, library_id))

            sql = """
            INSERT INTO vector_collection (
                id, name, name_zh, library_id, data_collection_id, book_id, create_time, modify_time, partition_name
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            values = (
                id, collection_name, "所有图书合并的collection", library_id, data_collection_id, book_id, formatted_time, None, partition_name
            )
            cursor.execute(sql, values)
            db_connection.commit()
            partition_created = False


        result = {
            "authorization": "true",
            "stateCode": 0,
            "state": "true",
            "message": "null",
            "data": {
                "name": name_zh,
                "is_exists": "false",
                "type": "Standard",
                "raw": data['paragraphs'],
                "libraryId": library_id,
                "total": 0,
                "readonly": "false",
                "ready": "false",
                "createTime": formatted_time,
                "modifyTime": "null",
                "partition_name": partition_name,
                "id": data_collection_id
            }
        }

        return result
    except Exception as err:
        # Milvus 与 MySQL 不是同一事务。若分区已创建但 MySQL 提交失败，
        # 立即补偿删除该分区，避免重试时遇到孤立分区。
        if partition_created and collection is not None:
            try:
                collection.drop_partition(partition_name)
            except Exception:
                pass
        raise err
        # print("错误", err)
        # return {"authorization": False, "stateCode": 1, "state": False, "message": str(err), "data": None}


