import asyncio
import os
from threading import Thread
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from constant import MODEL_DIR, CUDA_VISIBLE_DEVICES



os.environ['CUDA_VISIBLE_DEVICES'] = CUDA_VISIBLE_DEVICES
device = 'cuda'


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, device_map="auto", trust_remote_code=True).eval()
model.generation_config.max_new_tokens = 4096 #8192


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

def _chat_stream(query, history, system_message="You are a helpful assistant."):

    conversation = [
        {'role': 'system', 'content': system_message},
    ]
    for query_h, response_h in history:
        conversation.append({'role': 'user', 'content': query_h})
        conversation.append({'role': 'assistant', 'content': response_h})
    conversation.append({'role': 'user', 'content': query})
    inputs = tokenizer.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        return_tensors='pt',
    )
    # if model_name_or_path.__contains__('Int4'):
    #     inputs = inputs.to(device)
    # else:
    inputs = inputs.to(model.device)
    streamer = TextIteratorStreamer(tokenizer=tokenizer, skip_prompt=True, timeout=60.0, skip_special_tokens=True)

    # # 使用 run_in_executor 来调用阻塞的 model.generate 方法
    # loop = asyncio.get_event_loop()
    generation_kwargs = dict(
        input_ids=inputs,
        streamer=streamer,
    )
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    # 注意：这里假设 model.generate 是一个同步函数，它可能需要在一个线程或进程执行器中运行
    # 如果 model.generate 是一个异步函数，则直接调用即可
    # await loop.run_in_executor(None, model.generate, **generation_kwargs)
    # await asyncio.to_thread(model.generate, **generation_kwargs)
    # model.generate(generation_kwargs)

    for new_text in streamer:
        yield new_text




async def _chat_stream_1(query, history, num=''):
    print('----------:', query)
    for _ in range(10):
        await asyncio.sleep(0.2)
        print(num + '_test')
        yield num + '_test'