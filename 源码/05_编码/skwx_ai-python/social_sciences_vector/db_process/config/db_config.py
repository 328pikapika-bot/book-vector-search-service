#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/25 13:33
@Author  : st
@File    : db_config.py
@Project : Smart_Search
@Software: PyCharm
@文件说明: 

"""
from pymilvus import DataType

COLLECTION_INFO = {
    "name": "test",
    "name_vec": "embedding",
    "num_partitions": 1024,
    "fields": [
        {
            "name": 'id',
            "dtype": DataType.VARCHAR,
            "description": "主键ID,与源数据id对应",
            "max_length": 64,
            "is_primary": True,
            "auto_id": True
        },
        {
            "name": 'vec_medical_case',
            "description": "医案向量",
            "dtype": DataType.FLOAT_VECTOR,
            "dim": 768,
            "index_params": {
                "metric_type": "IP",
                "index_type": "IVF_FLAT",
                "nlist": 1024
            }
        },
        {
            "name": 'medical_case',
            "description": "医案原文",
            "dtype": DataType.VARCHAR,
            "max_length": 65535
        },
        {
            "name": 'sources_id',
            "description": "来源处的ID",
            "dtype": DataType.VARCHAR,
            "max_length": 4096
        },
        {
            "name": 'doctor_name',
            "description": "医生名称",
            "dtype": DataType.VARCHAR,
            "max_length": 64
        },
        {
            "name": 'sources',
            "description": "资料来源",
            "dtype": DataType.VARCHAR,
            "max_length": 65535
        },
        {
            "name": 'create_time',
            "description": "创建时间",
            "dtype": DataType.VARCHAR,
            "max_length": 50
        }
    ]
}
