#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/4/16 11:01
@Author  : st
@File    : llm_qwen_tokenizer.py
@Project : LLM_and_mlivus_api
@Software: PyCharm
@文件说明: 

"""
import asyncio
import json
import os
import copy
import uuid

from vllm import AsyncEngineArgs, AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from modelscope import AutoTokenizer, GenerationConfig

model_dir = "/data/st/code/LLM_and_mlivus_api/data/model/qwen/Qwen1___5-14B-Chat-GPTQ-Int4"
# GPU数量
tensor_parallel_size = 4
gpu_memory_utilization = 0.9
quantization = None  # 'gptq'
dtype = 'auto'  # 'float16'
# batch最大20条样本
max_num_seqs = 20


# vLLM模型加载
def load_vllm():
    global generation_config, tokenizer, stop_words_ids, engine
    # 模型下载
    # snapshot_download(model_dir)
    # 模型基础配置
    generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True)
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.eos_token_id = generation_config.eos_token_id
    # 推理终止词
    # stop_words_ids = [tokenizer.im_start_id, tokenizer.im_end_id, tokenizer.eos_token_id]
    stop_words_ids = generation_config.eos_token_id
    # vLLM基础配置
    args = AsyncEngineArgs(model_dir)
    args.worker_use_ray = False
    args.engine_use_ray = False
    args.tokenizer = model_dir
    args.tensor_parallel_size = tensor_parallel_size
    args.trust_remote_code = True
    args.quantization = quantization
    args.gpu_memory_utilization = gpu_memory_utilization
    args.dtype = dtype
    args.max_num_seqs = max_num_seqs  # batch最大20条样本
    # 加载模型
    os.environ['VLLM_USE_MODELSCOPE'] = 'True'
    engine = AsyncLLMEngine.from_engine_args(args)

    sampling_params = SamplingParams(stop_token_ids=stop_words_ids,
                                     early_stopping=False,
                                     top_p=generation_config.top_p,
                                     top_k=-1 if generation_config.top_k == 0 else generation_config.top_k,
                                     temperature=generation_config.temperature,
                                     repetition_penalty=generation_config.repetition_penalty,
                                     max_tokens=generation_config.max_new_tokens)

    return generation_config, tokenizer, stop_words_ids, engine, sampling_params


generation_config, tokenizer, stop_words_ids, engine, sampling_params = load_vllm()


# 用户停止句匹配
def match_user_stop_words(response_token_ids, user_stop_tokens):
    for stop_tokens in user_stop_tokens:
        if len(response_token_ids) < len(stop_tokens):
            continue
        if response_token_ids[-len(stop_tokens):] == stop_tokens:
            return True  # 命中停止句, 返回True
    return False


# 停用词清理
def remove_stop_words(token_ids, stop_words_ids):
    token_ids = copy.deepcopy(token_ids)
    while len(token_ids) > 0:
        if token_ids[-1] in stop_words_ids:
            token_ids.pop(-1)
        else:
            break
    return token_ids


def _parse_text(text):
    lines = text.split("/n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"/`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


async def _chat_stream(query, history, system_message="You are a helpful assistant."):
    conversation = [
        {'role': 'system', 'content': system_message},
    ]
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    user_stop_tokens = []
    request_id = str(uuid.uuid4().hex)
    # results_iter = engine.generate(prompt=None, sampling_params=sampling_params, prompt_token_ids=prompt_tokens, request_id=request_id)
    results_iter = engine.generate(prompt=inputs, sampling_params=sampling_params, request_id=request_id)

    async for result in results_iter:
        # 移除im_end,eos等系统停止词
        token_ids = remove_stop_words(result.outputs[0].token_ids, stop_words_ids)
        # 返回截止目前的tokens输出
        text = tokenizer.decode(token_ids)
        # yield (json.dumps({'text': text}) + '\0').encode('utf-8')
        yield text
        # 匹配用户停止词,终止推理
        if match_user_stop_words(token_ids, user_stop_tokens):
            await engine.abort(request_id)  # 终止vllm后续推理
            break
