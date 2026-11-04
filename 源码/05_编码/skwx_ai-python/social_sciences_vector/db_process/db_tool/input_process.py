#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/10/13 10:05
@Author  : st
@File    : input_process.py
@Project : Smart_Search
@Software: PyCharm
@文件说明:

"""
import json
import os
import queue
import threading
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
from pymilvus import DataType
from pymilvus import connections, CollectionSchema, FieldSchema, Collection, utility, db
from sentence_transformers import SentenceTransformer
from constant import *
from db_process.config.db_config import COLLECTION_INFO

# 连接数据库
conn = connections.connect(user=MILVUS_USER, password=MILVUS_PWD, host=MILVUS_HOST, port=MILVUS_PORT, alias="default")
if not db.list_database().__contains__(MILVUS_DB):
    db.create_database(MILVUS_DB)
db.using_database(MILVUS_DB)

CONSISTENCY_LEVEL = "Bounded"

# 批次大小
num_items_to_get = 100

# 文献数据队列
queue_papers = queue.Queue(1000)


api_paper_num = 0
model = SentenceTransformer('/data/st/code/LLM_and_mlivus_api/tool/milvus_confirguation/data/models/embedding/m3e-base')


def text2vec(sentences, batch_size=64, normalize=True):
    embeddings = model.encode(sentences, batch_size=batch_size, device='cuda:0', normalize_embeddings=normalize)
    return embeddings



def get_time_str():
    t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
    return str(t)

data_path = '../data/input_data/朱仁康、赵炳南医案20211214.xlsx'

def create_trian():
    num = 0
    temp_list = []
    temp_abs_list = []
    datas = pd.read_excel(data_path, sheet_name=1).fillna('')
    datas = datas.to_dict(orient='records')
    for index, data in enumerate(datas):
        temp_dict = dict()
        temp_dict['medical_case'] = data['医案'].strip().replace('\n', '')
        temp_dict['sources_id'] = data['来源处的ID'].strip()
        temp_dict['doctor_name'] = data['名医'].strip()
        temp_dict['sources'] = data['资料来源'].strip().replace('\n', '')
        temp_dict['create_time'] = get_time_str()
        # 用两个字段拼接
        # temp_abs_list.append('数据集名称：' + data['name_ch'] + '\n数据集描述：' + data['describe'])
        temp_abs_list.append(data['医案'].strip().replace('\n', ''))
        temp_list.append(temp_dict)
        num += 1
        if num % num_items_to_get == 0 or index == len(datas) - 1:
            print('===================' + str(num) + '====================')
            vec_abs_list = text2vec(temp_abs_list)
            for x, y in zip(temp_list, vec_abs_list):
                x['vec_medical_case'] = y
                queue_papers.put(x)
            temp_list = []
            temp_abs_list = []
    return queue_papers


def process_insert():
    """
    数据导入任务
    :return:
    """
    # 创建集合
    res = create_collection(COLLECTION_INFO.get('name'), COLLECTION_INFO.get('fields'),
                            COLLECTION_INFO.get('num_partitions'))
    if not res:
        raise "集合创建失败：" + COLLECTION_INFO.get('name')
    num_all = 0
    num_success = 0
    while True:
        if queue_papers.empty():
            print('暂无数据，休息60秒')
            time.sleep(10)
        temp_list = []
        for _ in range(num_items_to_get):
            temp_list.append(queue_papers.get())
            num_all += 1
            if queue_papers.empty():
                break
        if not temp_list:
            continue
        num = insert_datas(temp_list, COLLECTION_INFO.get('name'))
        num_success += num
        print('Added datas:' + str(num) + '; all datas: ' + str(num_all) + '; success: ' + str(num_success))



def create_collection(collection_name, fields_list, num_partitions=1024):
    """
    创建数据库集合
    :param collection_name: 集合名称
    :param fields_list: 字段信息
    :param num_partitions: 分区数
    :return:
    """
    if utility.has_collection(collection_name):
        return True
    fields = []
    index_params = []
    for field in fields_list:
        temp_params = FieldSchema(
            name=field['name'],
            dtype=field['dtype'],
            description=field['description']
        )

        if field.__contains__('dim'):
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description'],
                dim=field['dim'],
            )

        if field.__contains__('max_length'):
            temp_params = FieldSchema(
                name=field['name'],
                dtype=field['dtype'],
                description=field['description'],
                max_length=field['max_length'],
            )
        if field.__contains__('is_primary'):
            temp_params.is_primary = field['is_primary']
            temp_params.auto_id = field['auto_id']
        fields.append(temp_params)
        # 判断是否创建索引
        if field.__contains__('index_params'):
            # 向量索引
            if field['index_params']:
                temp_params = {
                    "metric_type": field['index_params']['metric_type'],  # 'IP'
                    "index_type": field['index_params']['index_type'],  # "IVF_FLAT",
                    "params": {"nlist": field['index_params']['nlist']}  #
                }
            # 标量索引
            else:
                temp_params = None
            index_params.append((field['name'], 'index_' + field['name'], temp_params))

    schema = CollectionSchema(
        fields=fields,
        description="nstl serach"
    )
    # 创建集合
    collection = Collection(
        name=collection_name,
        schema=schema,
        using='default',
        shards_num=2,  # 要创建的集合的分片数。	[1,16]
        # num_partitions=num_partitions,  # 要创建的集合的逻辑分区数。	[1,4096]
        consistency_level=CONSISTENCY_LEVEL
    )
    # 集合创建索引
    for field_name, index_name, index_param in index_params:
        collection.create_index(
            field_name=field_name,
            index_params=index_param,
            index_name=index_name
        )
    return True


def insert_datas(datas, collection_name, partition_name=None):
    """
    插入数据
    :param collection_name:
    :param datas: 数据集 list/tuple/pandas.DataFrame
    :param partition_name: 分区名称
    :return:
    """
    if not utility.has_collection(collection_name):
        return 0
    try:
        collection = Collection(collection_name)
        res = collection.insert(datas, partition_name=partition_name)
        return res.insert_count
    except Exception as e:
        return 0


if __name__ == '__main__':
    t1 = threading.Thread(target=create_trian, name='create_trian')
    t1.start()
    time.sleep(10)
    print("启动api获取数据任务")
    t2 = threading.Thread(target=process_insert, name='process_insert')
    t2.start()
    print('启动入库任务')
