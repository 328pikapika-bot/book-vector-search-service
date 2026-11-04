#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/10/15 17:28
@Author  : st
@File    : llm_glm_api.py
@Project : LLM_and_mlivus_api_medical
@Software: PyCharm
@文件说明: 

"""
import os
from zhipuai import ZhipuAI

client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY", ""))  # 请填写您自己的APIKey

# response = client.chat.completions.create(
#     model="glm-4",  # 请填写您要调用的模型名称
#     messages=[
#         {"role": "system", "content": "你是一个乐于回答各种问题的小助手，你的任务是提供专业、准确、有洞察力的建议。"},
#         {"role": "user",
#          "content": "我对太阳系的行星非常感兴趣，尤其是土星。请提供关于土星的基本信息，包括它的大小、组成、环系统以及任何独特的天文现象。"},
#     ],
#     stream=True,
# )
# for chunk in response:
#     print(chunk.choices[0].delta.content)


async def chat_stream(query, history,
                      system_message="你是一个乐于回答各种问题的中医小助手，你的任务是提供专业、准确、有洞察力的建议。"):
    messages = [
        {'role': 'system', 'content': system_message},
    ]
    for query_h, response_h in history:
        messages.append({'role': 'user', 'content': query_h})
        messages.append({'role': 'assistant', 'content': response_h})
    messages.append({'role': 'user', 'content': query})
    # 请填写您要调用的模型名称 glm-4-plus glm-4-flash
    response = client.chat.completions.create(model="glm-4-flash", messages=messages, stream=True)
    text = ""
    for chunk in response:
        text += chunk.choices[0].delta.content
        yield text

