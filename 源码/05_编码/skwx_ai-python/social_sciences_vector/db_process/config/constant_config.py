#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2023/8/24 10:13
@Author  : st
@File    : constant_config.py
@Project : Smart_Search
@Software: PyCharm
@文件说明: 

"""
import inspect
import os

path = inspect.getfile(inspect.currentframe())
abspath = os.path.abspath(path)
root_path = os.path.dirname(os.path.dirname(abspath))
print('===============', root_path)
M3E_BAS0E_EMBEDDING_MODEL_LOCAL_DICT = {
    "m3e-base": os.path.join(root_path, "data", 'embedding', 'm3e-base')
}

METRIC_TYPE = 'COSINE'
CONSISTENCY_LEVEL = "Bounded"

LANGUAGE_CODE = {
    'zh': '中文',
    'en': '英语',
    'ru': '俄语',
    'ja': '日语',
    'fr': '法语',
    'de': '德语',
    'ar': '阿拉伯语',
    'es': '西班牙语'
}