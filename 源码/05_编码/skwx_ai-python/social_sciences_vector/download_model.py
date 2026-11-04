#模型下载
from modelscope import snapshot_download
# model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct',local_dir="./llama-3-8B")
# model_dir = snapshot_download('LLM-Research/Meta-Llama-3-8B-Instruct',local_dir="./llama-3-8B")
# model_dir = snapshot_download('Qwen/Qwen1.5-7B-Chat-GPTQ-Int4',local_dir="./model/qwen_7B_int4")
# model_dir = snapshot_download('Qwen/Qwen1.5-14B-Chat-GPTQ-Int4',local_dir="./model/qwen_14B_int4")
# model_dir = snapshot_download('ZhipuAI/chatglm2-6b-32k',local_dir="./model/chatglm2-6b-32k")
#模型下载
from modelscope import snapshot_download
model_dir = snapshot_download('AI-ModelScope/m3e-base', local_dir="./model/m3e-base")

# model_name ="/home/ninemax/work/yzq/progrtam/llm_and_milvus_api_master/model/qwen_7B_int4/"
# from transformers import AutoTokenizer
# from vllm import LLM, SamplingParams
#
# # Initialize the tokenizer
# tokenizer = AutoTokenizer.from_pretrained(model_name)
#
# # Pass the default decoding hyperparameters of Qwen2.5-7B-Instruct
# # max_tokens is for the maximum length for generation.
# sampling_params = SamplingParams(temperature=0.7, top_p=0.8, repetition_penalty=1.05, max_tokens=512)
#
# # Input the model name or path. Can be GPTQ or AWQ models.
# llm = LLM(model=model_name,quantization="gptq")
#
# # Prepare your prompts
# prompt = "你好吗？"
# messages = [
#     {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
#     {"role": "user", "content": prompt}
# ]
# text = tokenizer.apply_chat_template(
#     messages,
#     tokenize=False,
#     add_generation_prompt=True
# )
#
# # generate outputs
# outputs = llm.generate([text], sampling_params)
#
# # Print the outputs.
# for output in outputs:
#     prompt = output.prompt
#     generated_text = output.outputs[0].text
#     print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")