#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/4/22 17:09
@Author  : st
@File    : constant.py
@Project : LLM_and_mlivus_api
@Software: PyCharm
@文件说明: 

"""
# 启动进程数量控制
PROCESS_WORKERS = 1

# path = inspect.getfile(inspect.currentframe())
# abspath = os.path.abspath(path)
# root_path = os.path.dirname(os.path.dirname(abspath))

# EMBEDDING_MODEL = os.path.join(root_path, "data", 'models', 'embedding', 'm3e-base')

# RERANK_MODEL = os.path.join(root_path, 'data', 'models', 'rerank', 'nlp_rom_passage-ranking_chinese-base')
RERANK_MODEL = '/data/st/code/LLM_and_milvus_api/data/model/rerank/bge-reranker-v2-m3'


# MODEL_DIR = "/home/yinzq/program/qwen_master/model/Qwen-72B"
# MODEL_DIR = "/home/yinzq/program/qwen_master/model/Qwen1.5-72B-Chat-GPTQ-Int4/qwen/Qwen1___5-72B-Chat-GPTQ-Int4/"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"# model_name_or_path = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"# model_name_or_path = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"l_name_or_path = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat"# model_name_or_path = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-7B-Chat-GPTQ-Int4"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"
# MODEL_DIR = "/home/ninemax/st/code/Qwen1.5-main/qwen/Qwen1___5-0___5B-Chat-GPTQ-Int4"

MODEL_DIR = "/data/st/code/LLM_and_milvus_api/data/model/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"

CUDA_VISIBLE_DEVICES = "0,1,2,3"

LOG_LEVEL = "INFO"

USE_RERANK = True
USE_TACTICS = True

URL_KEYWORD = "http://localhost:8897/recive_stream_qwen/sig_lj_QA"
URL_MEDICAL_KW = 'http://localhost:8897/get_medical_keywords'


import os

# Milvus连接信息
# MILVUS_HOST = '127.0.0.1'
MILVUS_HOST = os.getenv("MILVUS_HOST", "127.0.0.1")
# MILVUS_HOST = '127.0.0.1'
# MILVUS_HOST = '127.0.0.1'

MILVUS_PORT = '19530'
MILVUS_USER = os.getenv("MILVUS_USER", "")
MILVUS_PWD = os.getenv("MILVUS_PASSWORD", "")
MILVUS_DB = os.getenv("MILVUS_DATABASE", "book_vector_demo")
MILVUS_SECURE = False


# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")

# 35
# NEO4J_URI = "bolt://localhost:7687"
# NEO4J_USER = "neo4j"
# NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")

# 18
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://127.0.0.1:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "")
NEO4J_PWD = os.getenv("NEO4J_PASSWORD", "")