import json
import uuid
from collections import defaultdict
from datetime import datetime
from itertools import groupby
from typing import Optional

import jieba
import numpy as np
import pypinyin
import uvicorn
from fastapi import APIRouter, FastAPI, Body, Path
from pymilvus import Collection
from starlette.responses import JSONResponse
import mysql.connector
# 连接到 Milvus 服务
from pymilvus import utility

from config import mysql_host, mysql_user, mysql_password, mysql_database, milvus_host, milvus_port
from db_process.db_tool.embedding_model import text2vec
from llm_tool.model_utils import process_response
from national_development.book_operation_lock import (
    BookOperationLockTimeout,
    wait_until_book_idle,
)
from national_development.milvus_connection import connect_milvus

app_search_vector = APIRouter()


def search_datas(collection_name, vec_field, vec_text, minRelevance, output_fields, expr, partition_name, using, start_num=0, top_n=20,
                  nprobe=32, threshold=0):
    """

    :param collection_name:
    :param vec_field:
    :param vec_text:
    :param output_fields:
    :param expr:
    :param start_num:
    :param top_n:
    :param nprobe:要搜索的群集单元数
    :return:
    """
    print(collection_name)
    collection = Collection(collection_name, using=using)
    output_fields = list(output_fields)
    schema_field_names = {field.name for field in collection.schema.fields}
    for chunk_field in ["chunk_index", "chunk_total", "chunk_start", "chunk_end", "chunk_strategy"]:
        if chunk_field in schema_field_names and chunk_field not in output_fields:
            output_fields.append(chunk_field)

    search_params = {
        "metric_type": "COSINE",
        "offset": start_num,  # the sum of `offset` in `param` and `limit` should be less than 16384.
        "ignore_growing": False,
        "param": {"nprobe": nprobe}  # , "range_filter": 0.85 radius<distance <= range_filter
    }
    if threshold:
        search_params['param']['radius'] = threshold
    try:
        # collection = Collection(collection_name)
        # collection.load()
        if expr is None:
            results = collection.search(
                data=vec_text,  # 查询向量
                anns_field=vec_field,  # 检索向量字段
                param=search_params,
                limit=top_n,  # 输出结果数量 # the sum of `limit` should be less than 16384.
                output_fields=output_fields,  # 输出字段名称
                consistency_level="Bounded",  # 一致性级别
                timeout=5,  # 超时时间, 5秒钟,
                partition_names=partition_name
            )
        else:
            results = collection.search(
                data=vec_text,  # 查询向量
                anns_field=vec_field,  # 检索向量字段
                param=search_params,
                expr=expr,
                limit=top_n,  # 输出结果数量 # the sum of `limit` should be less than 16384.
                output_fields=output_fields,  # 输出字段名称
                consistency_level="Bounded",  # 一致性级别
                timeout=5,  # 超时时间, 5秒钟
                partition_names=partition_name
            )
        # 结果转换
        res_list = []
        if results:
            for item in results[0]:
                temp_dict = dict()
                if float(item.distance) >= minRelevance:
                    temp_dict['distance'] = item.distance
                    for key in output_fields:
                        temp_dict[key] = item.entity.get(key)
                    res_list.append(temp_dict)
        return res_list
    except Exception:
        return []


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


def collection_relevance(question, host, port, db_name, collection_name, name_vec, top_n, minRelevance, vector_model, expr, partition_name):
    seg_list = jieba.cut(question)
    tokenized_text = " ".join(seg_list)
    vec_query = text2vec([tokenized_text], vector_model).tolist()
    # collection = Collection("test")
    milvus_alias = connect_milvus(db_name, host=host, port=port)
    # 连接数据库

    results = search_datas(collection_name, name_vec, vec_query, minRelevance, ["id","title", "text", "book_id", "directory_id","literature_id","directory_type","directory_name"],
                                 expr, partition_name, using=milvus_alias, start_num=1, top_n=top_n, nprobe=10, threshold=0.1)

    return results


def scalar_condition(datas, allowed_fields=None):
    # 条件筛选,https://www.milvus-io.com/boolean
    expr_list = []
    for data in datas["condition"]:
        if allowed_fields is not None and data["field_name"] not in allowed_fields:
            continue
        if data["type"] == "in":  # in类型
            field_list = '"' + '", "'.join(data["field_value"]) + '"'
            expr_list.append(f'({data["field_name"]} in [{field_list}])')

        elif data["type"] == ">":  # 大于类型
            expr_list.append(f'({data["field_name"]} > {str(data["field_value"])})')

        elif data["type"] == "<":  # 小于类型
            expr_list.append(f'({data["field_name"]} < {str(data["field_value"])})')
        elif data["type"] == "=":
            expr_list.append(f'({data["field_name"]} == "{str(data["field_value"])}")')
        elif data["type"] == "like":  # like类型
            temp_expr = []
            for code in data["field_value"]:
                temp_expr.append(f'({data["field_name"]} like "%{code}%")')
            expr_list.append('(' + ' or '.join(temp_expr) + ')')
        elif data["type"] == "array_contains":  # 数组包含类型
            expr_list.append(f'(ARRAY_CONTAINS({data["field_name"]}, "{data["field_value"]}"))')
        elif data["type"] == "array_contains_any":  # 数组包含任意类型
            expr_list.append(f'(ARRAY_CONTAINS_ANY({data["field_name"]}, {data["field_value"]}))')
        elif data["type"] == "array_contains_all":  # 数组包含所有类型
            expr_list.append(f'(ARRAY_CONTAINS_ALL({data["field_name"]}, {data["field_value"]}))')
        else:
            return JSONResponse("查询条件错误")

    expr = ' and '.join(expr_list) if expr_list else ''
    return expr


def requested_filter_fields(datas):
    return {
        str(condition.get("field_name") or "").strip()
        for condition in datas.get("condition", [])
        if str(condition.get("field_name") or "").strip()
    }


def missing_filter_fields(collection_name, database_name, filter_fields):
    if not filter_fields:
        return []

    milvus_alias = connect_milvus(database_name, host=milvus_host, port=milvus_port)
    collection = Collection(collection_name, using=milvus_alias)
    schema_fields = {field.name for field in collection.schema.fields}
    return sorted(filter_fields - schema_fields)


def _search_library_after_gate(bookId: str, data):
    # print(bookId, data)
    try:
        file_id = str(uuid.uuid4())
        filter_enabled = data.get("filter") == "1"
        filter_fields = requested_filter_fields(data) if filter_enabled else set()
        ignored_filter_fields = []
        if bookId == "all":
            with connection_mysql() as db_connection:
                cursor = db_connection.cursor()

                sql = "SELECT name FROM vector_library"
                cursor.execute(sql)
                collections_name = [row[0] for row in cursor.fetchall()]
                relevance = []
                for collection_name in collections_name:
                    sql = """
                        SELECT * FROM vector_library WHERE name=%s;
                        """
                    cursor.execute(sql, (collection_name,))
                    library_data = cursor.fetchone()
                    # 将查询结果转换为字典
                    columns = [desc[0] for desc in cursor.description]
                    library_select_data_dict = dict(zip(columns, library_data))

                    sql = """
                        SELECT name FROM vector_library_base WHERE id=%s;
                        """

                    cursor.execute(sql, (library_select_data_dict["library_id"],))
                    library_name = cursor.fetchone()[0]
                    missing_fields = missing_filter_fields(collection_name, library_name, filter_fields)
                    collection_expr = scalar_condition(data, filter_fields - set(missing_fields)) if filter_enabled else None
                    if not collection_expr:
                        collection_expr = None
                    if missing_fields:
                        print(
                            f"Ignore unsupported filter fields for collection {collection_name}: "
                            f"{', '.join(missing_fields)}"
                        )
                        ignored_filter_fields.append(
                            {"collection": collection_name, "missingFields": missing_fields}
                        )
                    result = collection_relevance(data["question"], milvus_host, milvus_port, library_name, collection_name,
                                                  'embedding', data["topN"],
                                                  data["minRelevance"], data["vector_model"], collection_expr, None)
                    relevance.extend(result)
        else:
            with connection_mysql() as db_connection:
                cursor = db_connection.cursor()
                sql = """
                    SELECT library_id FROM vector_collection WHERE partition_name = %s
                    """
                cursor.execute(sql, (f"book_{bookId}",))
                libraryId = cursor.fetchone()[0]

                select_sql = "SELECT * FROM vector_library WHERE id = %s"
                cursor.execute(select_sql, (libraryId,))
                library_data = cursor.fetchone()

                # 将查询结果转换为字典
                columns = [desc[0] for desc in cursor.description]
                library_select_data_dict = dict(zip(columns, library_data))

                select_sql_db_name = "SELECT name FROM vector_library_base WHERE id = %s"
                cursor.execute(select_sql_db_name, (library_select_data_dict["library_id"],))
                library_name = cursor.fetchone()[0]
            with connection_mysql() as db_connection:
                cursor = db_connection.cursor()
                select_sql = "SELECT name FROM vector_collection WHERE partition_name = %s"
                cursor.execute(select_sql, (f"book_{bookId}",))
                collection_name = cursor.fetchone()[0]

            #
            # connections.connect(host='127.0.0.1', port='19530', db_name=library_name)
            # connections_list = utility.list_collections()
            relevance = []
            missing_fields = missing_filter_fields(collection_name, library_name, filter_fields)
            collection_expr = scalar_condition(data, filter_fields - set(missing_fields)) if filter_enabled else None
            if not collection_expr:
                collection_expr = None
            if missing_fields:
                print(
                    f"Ignore unsupported filter fields for collection {collection_name}: "
                    f"{', '.join(missing_fields)}"
                )
                ignored_filter_fields.append(
                    {"collection": collection_name, "missingFields": missing_fields}
                )
            result = collection_relevance(data["question"], milvus_host, milvus_port, library_name, collection_name,
                                          'embedding', data["topN"],
                                          data["minRelevance"], data["vector_model"], collection_expr, [f"book_{bookId}"])
            relevance.extend(result)

        relevance = sorted(relevance, key=lambda x: x['distance'], reverse=True)
        relevance_desc_grouped = relevance[:data["topN"]]
        # relevance.sort(key=lambda x: x['resourceId'])
        # 使用 groupby 来分组
        relevance_resource_id_grouped = [list(group) for key, group in groupby(relevance_desc_grouped, key=lambda x: x['book_id'])]
        # relevance_desc_grouped = [
        #     sorted(sublist, key=lambda relevance_resource_id_grouped: relevance_resource_id_grouped['distance'], reverse=True)[:data["topN"]]
        #     for sublist in relevance_resource_id_grouped
        # ]

        result_in = []

        pa = ["book_id", "literature_id", "directory_id"]

        for partition_param in relevance_resource_id_grouped:  # 每个文件
            partitions = []
            for partition in partition_param:  # 每个片段
                file_part = str(uuid.uuid4())
                par = {
                        "text": partition['text'],
                        "relevance": partition['distance'],
                        "partitionNumber": 0,
                        "sectionNumber": 0,
                        "bookName": partition["title"],
                        "literatureId": partition["literature_id"],
                        "bookId": partition["book_id"],
                        "directoryName": partition['directory_name'],
                        "directoryId": partition["directory_id"],
                        "documentId": partition['id'],
                        "tags": [
                            {
                                "key": "__document_id",
                                "value": partition['id']
                            },
                            {
                                "key": "__file_type",
                                "value": "text/plain"
                            },
                            {
                                "key": "__file_id",
                                "value": file_id
                            },
                            {
                                "key": "__file_part",
                                "value": file_part
                            },
                            {
                                "key": "__part_n",
                                "value": "0"
                            },
                            {
                                "key": "__sect_n",
                                "value": "0"
                            }
                        ]
                    }
                for i in pa:
                    if i in partition.keys():
                        if partition.get(i):
                            par["tags"].append({"key": i, "value": partition.get(i)})
                partitions.append(par)
            result_param = {
                "link": "5628ddc0-aa73-4af8-a9a9-cb9ad72a99b5/c915c0bd-e4e1-462c-992f-3a7cb830dfc4/a8246b2939e54b1da8e88ae97adfb057",
                "fileId": "",
                "sourceContentType": "text/plain",
                "sourceName": "content.txt",
                "sourceUrl": "/download?index=5628ddc0-aa73-4af8-a9a9-cb9ad72a99b5&documentId=c915c0bd-e4e1-462c-992f-3a7cb830dfc4&filename=content.txt",
                "partitions": partitions
            }
            result_in.append(result_param)

        result = {
            "authorization": "true",
            "stateCode": 0,
            "state": "true",
            "message": "null",
            "data": {
                "query": data['question'],
                "results": result_in,
                "ignoredFilterFields": ignored_filter_fields,
            }
        }
        return JSONResponse(result)
    except Exception as err:
        print(err)
        return JSONResponse({"authorization": "true", "stateCode": 1001, "state": False, "message": err, "data": "null"})


@app_search_vector.post("/Library/{bookId}/search")
def search_library(bookId: str, data=Body()):
    if bookId != "all":
        try:
            # This is a short admission gate, not a lock held for the whole
            # search.  Concurrent searches remain parallel, but none starts
            # while the same book is still being uploaded or deleted.
            wait_until_book_idle(bookId)
        except BookOperationLockTimeout as exc:
            return JSONResponse(
                {
                    "authorization": False,
                    "stateCode": 1,
                    "state": False,
                    "message": str(exc),
                    "data": {"book_id": bookId, "status": "processing"},
                },
                status_code=409,
            )
    return _search_library_after_gate(bookId, data)
