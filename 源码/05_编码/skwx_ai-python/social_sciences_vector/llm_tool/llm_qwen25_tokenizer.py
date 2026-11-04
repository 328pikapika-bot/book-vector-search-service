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
import os
import time

from transformers import AutoTokenizer
from vllm import SamplingParams, LLM, AsyncEngineArgs, AsyncLLMEngine


MODEL_DIR = "/data/st/code/SWAQ/LLM_and_mlivus_api_swaq/data/model/Qwen/Qwen2___5-14B-Instruct"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"
tp_size = 4  # GPU数量
gpu_memory_utilization = 0.95  # GPU内存占用比例
quantization = 'gptq' if 'gptq' in MODEL_DIR.lower() else None  # 模型类型
dtype = 'half' #if 'gptq' in MODEL_DIR.lower() else 'float16'
max_num_seqs = 20  # batch最大20条样本
max_model_len = 32768  # 生成文本最大长度131072 32768
LORA_PATH = ''
enable_lora = False
if LORA_PATH:
    enable_lora = True

top_p = 0.8



def load_model_and_tokenizer(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    engine_args = AsyncEngineArgs(
        model=model_dir,
        tokenizer=model_dir,
        tensor_parallel_size=tp_size,
        dtype=dtype,
        gpu_memory_utilization=gpu_memory_utilization,
        worker_use_ray=False,
        #engine_use_ray=False,
        quantization=quantization,
        max_num_seqs=max_num_seqs,
        max_model_len=max_model_len,
        trust_remote_code=True,
        # 如果遇见 OOM 现象，建议开启下述参数
        # enable_chunked_prefill=True,
        # max_num_batched_tokens=4096
    )
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    params_dict = {
        "repetition_penalty": 1.05,
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": 8192,
    }
    sampling_params = SamplingParams(**params_dict)

    # llm = LLM(model=model_dir,
    #           tokenizer=model_dir,
    #           tensor_parallel_size=tp_size,
    #           dtype=dtype,
    #           gpu_memory_utilization=gpu_memory_utilization,
    #           worker_use_ray=False,
    #           engine_use_ray=False,
    #           quantization=quantization,
    #           max_num_seqs=max_num_seqs,
    #           max_model_len=max_model_len,
    #           trust_remote_code=True,
    #           # 如果遇见 OOM 现象，建议开启下述参数
    #           # enable_chunked_prefill=True,
    #           # max_num_batched_tokens=4096
    #           )
    return engine, tokenizer, sampling_params


engine, tokenizer, sampling_params = load_model_and_tokenizer(MODEL_DIR)


def parse_text(text):
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


async def chat_stream(query, history, system_message="You are a helpful assistant."):
    prompt = [
        {'role': 'system', 'content': system_message},
    ]
    for query_h, response_h in history:
        prompt.append({'role': 'user', 'content': query_h})
        prompt.append({'role': 'assistant', 'content': response_h})
    prompt.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    # outputs = llm.generate([inputs], sampling_params)
    # temp_text = ""
    # for output in outputs:
    #     temp_text += output.outputs[0].text
    #     yield temp_text

    results_iter = engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}")
    async for result in results_iter:
        # 移除im_end,eos等系统停止词
        token_ids = result.outputs[0].token_ids
        # 返回截止目前的tokens输出
        text = tokenizer.decode(token_ids, skip_special_tokens=True).replace("�", "")
        # yield (json.dumps({'text': text}) + '\0').encode('utf-8')
        yield text

async def chat_stream_prompt(messages=[]):
    prompt = []
    # 将query数组添加到prompt数组中
    prompt.extend(messages)

    inputs = tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True
    )
    # outputs = llm.generate([inputs], sampling_params)
    # temp_text = ""
    # for output in outputs:
    #     temp_text += output.outputs[0].text
    #     yield temp_text

    results_iter = engine.generate(inputs=inputs, sampling_params=sampling_params, request_id=f"{time.time()}")
    text_tmp = ""
    async for result in results_iter:
        # 移除im_end,eos等系统停止词
        token_ids = result.outputs[0].token_ids
        # 返回截止目前的tokens输出
        text = tokenizer.decode(token_ids, skip_special_tokens=True).replace("�", "")
        # yield (json.dumps({'text': text}) + '\0').encode('utf-8')
        if len(text_tmp) != len(text):
            yield text
        text_tmp = text
    yield text_tmp
# Avg prompt throughput: 0.1 tokens/s, Avg generation throughput: 0.0 tokens/s, Running: 6 reqs, Swapped: 0 reqs, Pending: 0 reqs, GPU KV cache usage: 0.6%, CPU KV cache usage: 0.0%
