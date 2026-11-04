import time
from fastapi import APIRouter, Body
from pymilvus.orm import utility
from starlette.responses import JSONResponse

from log.logger import Logger
from config import milvus_host, milvus_port, mysql_host, mysql_user, mysql_password, mysql_database
from db_process.db_tool.embedding_model import text2vec
from pymilvus import Collection
import mysql.connector
from national_development.milvus_connection import connect_milvus

log = Logger()
app_search_scalar = APIRouter()


# 连接数据库
def connection_mysql():
    # 创建数据库连接
    db_connection = mysql.connector.connect(
        host=mysql_host,  # MySQL 服务器地址（例如：localhost、IP 地址等）
        user=mysql_user,  # 用户名
        password=mysql_password,  # 密码
        database=mysql_database  # 要连接的数据库名称
    )
    return db_connection


@app_search_scalar.post("/Library/{libraryId}/search_scalar")
# def process_query(question="", page_num=1, page_size=20, threshold=0.75, language=['en'], **kwargs):
def process_query(libraryId: str, data=Body()):
    """
    查询任务
    :param question:
    :param page_num:
    :param page_size:
    :param threshold: 默认值提升到0.75
    :param language:
    :return:
    """
    try:
        log.info('milvus query: ' + str(data["question"]))
        st = time.time()
        all_st = time.time()

        # 查询向量库
        with connection_mysql() as db_connection:
            cursor = db_connection.cursor()
            select_sql = "SELECT * FROM vector_library WHERE id = %s"
            cursor.execute(select_sql, (libraryId,))
            library_data = cursor.fetchone()

            # 将查询结果转换为字典
            columns = [desc[0] for desc in cursor.description]
            library_select_data_dict = dict(zip(columns, library_data))

            select_sql_db_name = "SELECT name FROM vector_library_base WHERE id = %s"
            cursor.execute(select_sql_db_name, (library_select_data_dict["library_id"],))
            library_name = cursor.fetchone()[0]

        # 链接数据库
        milvus_alias = connect_milvus(library_name)
        collection_names = utility.list_collections(using=milvus_alias)
        # 文本转向量
        vec_query = text2vec([data["question"]], data["vector_model"])
        log.debug(f'{data["question"]}:文本转向量耗时: ' + str(time.time() - st))
        st = time.time()

        # 条件筛选,https://www.milvus-io.com/boolean
        expr_list = []
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
            log.error(f'查询条件类型错误:{data["type"]}')
            return JSONResponse("查询条件错误")

        expr = ' and '.join(expr_list) if expr_list else ''

        log.info('milvus expr: ' + expr)
        log.debug(f'{data["question"]}:查询条件耗时: ' + str(time.time() - st))
        st = time.time()
        # 查询
        results = []
        for collection_name in collection_names:
            collection = Collection(collection_name, using=milvus_alias)

            # 获取集合的所有字段
            fields = collection.schema.fields

            # 从字段中去除 'embedding' 字段
            field_names = [field.name for field in fields if field.name != 'embedding']

            # 构建查询条件
            expression = expr if expr else None

            # 执行查询
            result = collection.query(expr=expression, output_fields=field_names)
            for i in result:
                results.append(i)
        log.info('milvus res num: ' + str(len(results)))
        # log.debug('milvus res: ' + str(res))
        log.debug(f'{data["question"]}:milvus 查询耗时: ' + str(time.time() - st))
        log.debug(f'{data["question"]}:milvus查询总耗时：{time.time() - all_st}')
        result = {
            "authorization": "true",
            "stateCode": 0,
            "state": "true",
            "message": "null",
            "data": {
                "query": data['question'],
                "results": results
            }
        }
        return JSONResponse(result)
    except Exception as e:
        log.error(f'查询milvus异常:{e}')
        return JSONResponse("数据查询失败")
