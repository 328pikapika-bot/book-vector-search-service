import json
import random
import uuid
from datetime import datetime
import config
from config import mysql_user, mysql_host, mysql_password, mysql_database, milvus_host, milvus_port
import numpy as np
import pypinyin
import re
from pymilvus import db, utility, Collection
from starlette.responses import JSONResponse
import mysql.connector
from national_development.milvus_connection import connect_milvus
# 连接到 Milvus 服务

REQUIRED_COLLECTION_FIELDS = {
    "publish_direct",
    "chunk_index",
    "chunk_total",
    "chunk_start",
    "chunk_end",
    "chunk_strategy",
}


def has_required_collection_schema(collection_name, using):
    collection = Collection(collection_name, using=using)
    field_names = {field.name for field in collection.schema.fields}
    return REQUIRED_COLLECTION_FIELDS.issubset(field_names)


def configured_collection_prefix():
    # 兼容一期配置文件尚未增加该可选项的情况。
    return str(getattr(config, "milvus_collection_name", "") or "").strip()


def ensure_configured_collection_usable(collection_name, using):
    if utility.has_collection(collection_name, using=using) and not has_required_collection_schema(collection_name, using):
        raise RuntimeError(
            f"Configured Milvus collection '{collection_name}' already exists with an old schema. "
            f"Required fields: {', '.join(sorted(REQUIRED_COLLECTION_FIELDS))}."
        )


def configured_collection_sequence(collection_name, prefix):
    if collection_name == prefix:
        return 1

    match = re.match(rf"^{re.escape(prefix)}_(\d+)$", str(collection_name))
    if match and int(match.group(1)) >= 2:
        return int(match.group(1))
    return None


def next_configured_collection_name(cursor, prefix, using):
    used_sequences = set()

    cursor.execute("SELECT name FROM vector_library")
    for (name,) in cursor.fetchall():
        sequence = configured_collection_sequence(name, prefix)
        if sequence is not None:
            used_sequences.add(sequence)

    for name in utility.list_collections(using=using):
        sequence = configured_collection_sequence(name, prefix)
        if sequence is not None:
            used_sequences.add(sequence)

    if not used_sequences:
        return prefix
    return f"{prefix}_{max(used_sequences) + 1}"


def next_book_partition_name(cursor, using):
    used_numbers = set()

    cursor.execute("SELECT name FROM vector_library WHERE name REGEXP '^book_partition_[0-9]+$'")
    for (name,) in cursor.fetchall():
        match = re.match(r"^book_partition_(\d+)$", str(name))
        if match:
            used_numbers.add(int(match.group(1)))

    for name in utility.list_collections(using=using):
        match = re.match(r"^book_partition_(\d+)$", str(name))
        if match:
            used_numbers.add(int(match.group(1)))

    next_number = max(used_numbers or {0}) + 1
    return f"book_partition_{next_number}"


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


def create_library(book_name, book_id, max_text_total, vector_model, semantic_model, force_new_collection=False):
    try:
        data = {"name": book_name, "book_id": book_id, "textModel": semantic_model, "embeddingModel": vector_model, "max_text_total": max_text_total}
        sql_id = str(uuid.uuid4())

        # 获取当前时间
        now = datetime.now()
        # 格式化为 ISO 8601 格式，带微秒和时区偏移
        formatted_time = now.strftime('%Y-%m-%dT%H:%M:%S.%f+00:00')
        name_zh = data['name']
        # 转换为拼音
        # name_pinyin = convert_to_pinyin(data['name'].replace('（', '').replace('）','').replace('(','').replace(')',''))
        name_pinyin = "book_" + data['book_id']
        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            # 同一时刻只允许一个请求选择或创建 vector_library，避免并发首建时
            # 多个请求同时插入同名记录。GET_LOCK 是连接级锁，连接关闭会自动释放。
            lock_name = f"social_sciences_vector:create_library:{mysql_database}"
            cursor.execute("SELECT GET_LOCK(%s, %s)", (lock_name, 120))
            lock_result = cursor.fetchone()
            if not lock_result or lock_result[0] != 1:
                raise TimeoutError("Timed out waiting for the vector_library creation lock.")

            # 查看是否存在多余容量
            sql_count = "select * from vector_library_base where collection_count_max-collection_count>=1"
            cursor.execute(sql_count)

            all_base = cursor.fetchone()

            if all_base is None:  # 如果不存在，则创建一个新数据库

                # 查询当前4000的库的数量
                select_count_sql ="""select count(id) from vector_library_base where collection_count=4000"""
                cursor.execute(select_count_sql)
                id_count = cursor.fetchone()[0]
                # 创建数据库
                sql_id_library_base = str(uuid.uuid4())
                sql_create = """
                INSERT INTO vector_library_base (
                    id, name, name_zh, collection_count_max, collection_count, create_time
                ) VALUES (%s, %s, %s, %s, %s, %s)
                """

                values = (
                    sql_id_library_base, f"ssc_{id_count + 1}", "社科", 4000, 0, formatted_time
                )
                cursor.execute(sql_create, values)
                db_connection.commit()  # 确保提交操作

                # 创建成功时再查一遍
                sql = "select * from vector_library_base where collection_count_max>collection_count"
                cursor.execute(sql)
                all_base = cursor.fetchone()



                name_ssc = f"ssc_{id_count + 1}"
                admin_alias = connect_milvus("default")
                if name_ssc not in db.list_database(using=admin_alias):
                    db.create_database(name_ssc, using=admin_alias)

            all_book = None
            milvus_alias = connect_milvus(all_base[1])
            configured_prefix = configured_collection_prefix()
            if configured_prefix and not force_new_collection:
                sql = """
                    SELECT * FROM vector_library
                    WHERE partition_count_max-partition_count>=1
                    ORDER BY create_time DESC;
                """
                cursor.execute(sql)
                configured_candidates = []
                for candidate_book in cursor.fetchall():
                    sequence = configured_collection_sequence(candidate_book[2], configured_prefix)
                    if sequence is not None:
                        configured_candidates.append((sequence, candidate_book))

                for _, candidate_book in sorted(configured_candidates, key=lambda item: item[0], reverse=True):
                    collection_name = candidate_book[2]
                    ensure_configured_collection_usable(collection_name, milvus_alias)
                    if not utility.has_collection(collection_name, using=milvus_alias) and candidate_book[18] != 1:
                        raise RuntimeError(
                            f"Configured vector_library '{collection_name}' exists in MySQL, "
                            "but the Milvus collection is missing and partition_count is not 1."
                        )
                    all_book = candidate_book
                    break
            elif not force_new_collection:
                sql = "SELECT * FROM vector_library WHERE partition_count_max-partition_count>=1 ORDER BY create_time DESC;"
                cursor.execute(sql)
                for candidate_book in cursor.fetchall():
                    try:
                        collection_name = candidate_book[2]
                        if not utility.has_collection(collection_name, using=milvus_alias):
                            all_book = candidate_book
                            break
                        if has_required_collection_schema(collection_name, milvus_alias):
                            all_book = candidate_book
                            break
                    except Exception:
                        continue
            if all_book is None:
                collection_name = (
                    next_configured_collection_name(cursor, configured_prefix, milvus_alias)
                    if configured_prefix
                    else next_book_partition_name(cursor, milvus_alias)
                )
                if configured_prefix:
                    ensure_configured_collection_usable(collection_name, milvus_alias)
                sql = """
                INSERT INTO vector_library (
                    id, book_id, name, name_zh, text_model, text_model_max_token_total, embedding_model_max_token_total,
                    embedding_model, max_ask_prompt_size, max_matches_count, empty_answer, answer_tokens,
                    description, create_time, modify_time, delete_time, library_id, partition_count_max, partition_count
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """

                values = (
                    sql_id, collection_name.replace("book_", ""), collection_name, "所有图书合并的collection", data.get("textModel", "AllenNLP"), 8000, data.get("max_text_total", "1000"), data.get("embeddingModel", "m3e-base"), 2048, 5,
                    "知识库内未包含相关内容", 2048, None, formatted_time, None, None, all_base[0], 1024, 1
                )
                cursor.execute(sql, values)
                db_connection.commit()  # 确保提交操作

                # 创建成功时再查一遍
                sql = "select * from vector_library where id=%s"
                cursor.execute(sql, (sql_id,))
                all_book = cursor.fetchone()

            sql = """
            SELECT * FROM vector_library WHERE book_id=%s;
            """
            cursor.execute(sql, (all_book[1],))
            all_data = cursor.fetchone()

            data = {"book_id": all_data[1], "name": all_data[2], "name_zh": all_data[3], "text_model": all_data[4],
                    "embeddingModel": all_data[5], "textModelMaxTokenTotal": all_data[6],"embeddingModelMaxTokenTotal": all_data[7],
                    "maxAskPromptSize": all_data[8], "maxMatchesCount": all_data[9],"emptyAnswer": all_data[10],
                    "answerTokens": all_data[11],"description":  all_data[12], "createTime": all_data[13],
                    "modifyTime": all_data[14], "deleteTime": all_data[15], "library_id": all_data[16]
                    }
            sql_id = all_data[0]

            result = {
                "authorization": "true",
                "stateCode": 0,
                "state": "true",
                "message": "null",
                "data": {
                    "name": data.get("name", "社科测试"),
                    "embeddingModel": data.get("embeddingModel", "text-embedding-v2"),
                    "textModel": data.get("textModel", "qwen-plus"),
                    "textModelMaxTokenTotal": data.get("textModelMaxTokenTotal", 8000),
                    "embeddingModelMaxTokenTotal": data.get("embeddingModelMaxTokenTotal", 1000),
                    "maxAskPromptSize": data.get("maxAskPromptSize", 2048),
                    "maxMatchesCount": data.get("maxMatchesCount", 5),
                    "emptyAnswer": data.get("emptyAnswer", "知识库内未包含相关内容"),
                    "answerTokens": data.get("answerTokens", 2048),
                    "description": data.get("description", "null"),
                    "createTime": data.get("createTime", formatted_time),
                    "modifyTime": data.get("modifyTime", "null"),
                    "deleteTime": data.get("deleteTime", "null"),
                    "id": sql_id
                }
            }

        return result
    except Exception as err:
        raise err
        # return {"authorization": False, "stateCode": 1, "state": False, "message": str(err), "data": None}
