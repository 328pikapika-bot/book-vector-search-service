#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/9/13 14:52
@Author  : st
@File    : main_llm.py
@Project : LLM_and_mlivus_api
@Software: PyCharm
@文件说明:
"""

import uvicorn
from fastapi import FastAPI, APIRouter
from constant import PROCESS_WORKERS
from national_development.up_collection import app_up_collection
from national_development.search_vector import app_search_vector
from national_development.search_scalar import app_search_scalar
from national_development.drop_collection import app_drop_collection
from national_development.drop_paragraph import app_drop_paragraph


# 创建一个总的FastAPI应用
app = FastAPI()

# # 创建APIRouter实例，并将子应用的路由添加到路由器中
router = APIRouter()
router.include_router(app_up_collection, prefix="")
router.include_router(app_drop_collection, prefix="")
router.include_router(app_search_vector, prefix="")
router.include_router(app_search_scalar, prefix="")
router.include_router(app_drop_paragraph, prefix="")


# 将router添加到主应用中
app.include_router(router)

port = 8006


def main():
    uvicorn.run(app="main_library_api:app", host="0.0.0.0", port=port, workers=PROCESS_WORKERS)


if __name__ == "__main__":
    main()
