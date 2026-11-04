#!C:\pythonCode
# -*- coding: utf-8 -*-
# @Time : 2023/10/12 16:29
# @Author : yzq
# @File : milvus_tool.py
# @Software: PyCharm
import asyncio
import time

import pandas as pd
import requests

from constant import *
from db_process.config.db_config import COLLECTION_INFO
from db_process.db_tool.embedding_model import text2vec
from db_process.db_tool.milvus_util import MilvusUtils
# from db_process.db_tool.neo4j_process import get_neo4j_datas
from log.logger import Logger

log = Logger()

# 连接数据库
milvus_db = MilvusUtils(user=MILVUS_USER, pwd=MILVUS_PWD, ip=MILVUS_HOST, port=MILVUS_PORT, db_name=MILVUS_DB,
                        collection_name=COLLECTION_INFO.get('name'))

# 输出字段
OUTPUT_FIELDS = ['id', 'medical_case', 'doctor_name', 'sources']



async def process_query_neo4j(question, doctor_name, data_num=100):
    """
    查询任务
    :param question:
    :param doctor_name:
    :return:
    """
    keywords = await get_question_keywords(question)
    log.debug('keywords: %s' % keywords)
    if not keywords:
        return []
    return get_neo4j_datas(keywords, doctor_name, limit=data_num)



async def get_question_keywords(question):
    """
    获取关键词
    :param question:
    :return:
    """
    url = URL_MEDICAL_KW + f"?question={question}"
    try:
        response = requests.post(url, timeout=10)
        if response.status_code == 200:
            return response.json().get('result', [])
        return None
    except:
        return None

# queue_papers = queue.Queue(10000)
async def process_query_milvus(question, page_num=1, page_size=10, threshold=0.75, **kwargs):
    """
    查询任务
    :param question:
    :param page_num:
    :param page_size:
    :param threshold:
    :return:
    """
    log.info('milvus query: ' + str(question))
    log.info('milvus page_num: ' + str(page_num))
    log.info('milvus page_size: ' + str(page_size))
    log.info('milvus threshold: ' + str(threshold))
    # 文本转向量
    vec_query = text2vec([question])

    # 条件筛选,https://www.milvus-io.com/boolean
    expr_list = []
    # 语言
    language = kwargs.get('language', None)
    if language and isinstance(language, list):
        language_str = '"' + '", "'.join(language) + '"'
        expr_list.append(f'(language in [{language_str}])')

    # 医生名
    doctor_name = kwargs.get('doctor_name', None)
    if doctor_name and isinstance(doctor_name, str):
        expr_list.append(f'(doctor_name == "{doctor_name}")')

    expr = ' and '.join(expr_list) if expr_list else ''

    log.info('milvus expr: ' + expr)

    # 分页设置
    start_num = (page_num - 1) * page_size
    # 查询
    res = milvus_db.search_datas(COLLECTION_INFO.get('name'), COLLECTION_INFO.get('name_vec'), vec_query, OUTPUT_FIELDS,
                                 expr, start_num=start_num, top_n=page_size, threshold=threshold)
    log.info('milvus res num: ' + str(len(res)))
    log.debug('milvus res: ' + str(res))
    if res:
        df = pd.DataFrame(res)
        df_filled = df.fillna('')
        df_filled = df[df_filled['medical_case'].notnull()]
        res = df_filled.to_dict('records')
    return res


def process_get_datas_by_id_milvus(ids):
    output_fields = ['id', 'title', 'abstract', 'year', 'language']
    res = milvus_db.get_datas_by_id(COLLECTION_INFO.get('name'), ids, output_fields)
    if res:
        res = sorted(res, key=lambda x: ids.index(x['id']))
    return res


if __name__ == '__main__':
    responses = process_query_milvus(
        "The oxygen reduction reaction of two electron transfer of nitrogen-doped carbon in the electro-Fenton system	",
        page_size=20, threshold=0.83)
    result = []
    for res in responses:
        # 临时不行
        if res["abstract"]:  # 对摘要长度做限制筛选
            if res['language'] == 'en' and len(res["abstract"].split(' ')) > 15:
                result.append(res)
            elif res['language'] == 'zh' and len(res["abstract"]) > 20:
                result.append(res)
            else:
                result.append(res)
    print(result)
    # print(process_get_datas_by_id(["000e366868f064c6d38fea9c4690c175", "006202d150f0317e28802defc671c5b2", '009e5048381411efca9f078a85f0749e']))

    for i in range(50):
        time.sleep(1)


