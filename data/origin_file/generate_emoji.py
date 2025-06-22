import os
import json
import traceback
import re
from pygments import highlight
from pygments.lexers import get_lexer_by_name
from pygments.formatters import HtmlFormatter
import html
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatter import Formatter

import keyword


import openai
import time

# 配置 OpenAI 客户端（新写法）
client = openai.OpenAI(
    api_key='EMPTY',  # 如果不需要 API key，可以填 'EMPTY' 或自定义
    # base_url='http://127.0.0.1:30003/v1'  # 替换成你的 API 地址
    base_url='http://10.200.0.53:8889/v1'  # 替换成你的 API 地址

)

def stream_response(response):
    """
    处理流式响应并实时输出到终端。
    """
    full_response = ""
    for chunk in response:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            print(content, end='', flush=True)
            full_response += content
        time.sleep(0.01)  # 微小延迟，模拟更顺畅的流式体验
    print()
    return full_response

def chat_with_deepseek(query):
    """
    与 DeepSeek 进行多轮对话。
    """
    # 初始化消息列表，包含系统提示
    messages = [
        {"role": "system", "content": "你是一个乐于助人的助手。"},
    ]

    messages.append({"role": "user", "content": query})

    # 调用新方式
    response = client.chat.completions.create(
        model='deepseek-r1',  # 替换为你的模型名
        messages=messages,
        stream=True,
        temperature=0,
    )

    full_response = stream_response(response)

    return full_response

json_path = "/mnt/sfs_turbo/xinyuzhu/ai-developer-draw/ML-Master/data/origin_file/task75_log_for_run0_25-06-12T16:48:19.json"
demo_data_save_path = "/mnt/sfs_turbo/xinyuzhu/ai-developer-draw/ML-Master/data/demos"
demo_config_save_path = "/mnt/sfs_turbo/xinyuzhu/ai-developer-draw/ML-Master/data/demo-config.json"
raw_log_path = "/mnt/sfs_turbo/xinyuzhu/ai-developer-draw/data/mcts_log_new/final"


with open(json_path,'r') as f:
    json_data = json.load(f)


for task_name in json_data:
    with open(demo_config_save_path,"r",encoding='utf-8') as f:
        config_data = json.load(f)

    task_desc_path = f"/mnt/sfs_turbo/exp_data/demo1bench/{task_name}/prepared/public/description.md"

    with open(task_desc_path,"r",encoding='utf-8') as f:
        lines = f.readlines()

    task_desc = ""
    for line in lines:
        task_desc += line

    prompt = f"""
我现在会用英语的形式给你一个机器学习的任务描述，任务描述为
{task_desc}

你需要阅读这段任务描述，然后完成三个任务
1. 给出一个最适合这个任务内容的emoji
2. 从classification, regression, computer-vision, nlp, others中挑选一个最符合这个任务的类型的主标签
3. 根据主标签，给出合适的，描述更详细的子标签

一个回答示例如下,请严格按照这个格式进行回答：
[Answer]:
emoji:🚢
tag:classification
description:Binary classification

"""
    for retry_time in range(3):
        try:
            answer = chat_with_deepseek(prompt)
            emoji = answer.split("emoji:")[1].split("\n")[0]
            category = answer.split("tag:")[1].split("\n")[0]
            description = answer.split("description:")[1].split("\n")[0]
            break
        except:
            if retry_time ==2:
                raise ValueError()
            print(f"提取回答出错，第{retry_time}次重试")


    task_already_in_demo = False
    for idx,task_dict in enumerate(config_data["demos"]):
        if task_dict["id"] == task_name:
            task_already_in_demo = True
            config_data["demos"][idx]["id"] = config_data["demos"][idx]["id"]
            config_data["demos"][idx]["icon"] = emoji
            config_data["demos"][idx]["title"] = config_data["demos"][idx]["title"]
            config_data["demos"][idx]["medal"] = config_data["demos"][idx]["medal"]
            config_data["demos"][idx]["description"] = description
            config_data["demos"][idx]["category"] = category
            config_data["demos"][idx]["file"] = config_data["demos"][idx]["file"]
            break

    if task_already_in_demo == False:
        config_data["demos"].append(
            {
                "id": f"{task_name}",
                "icon": emoji,
                "title": f"{task_name}",
                "description": description,
                "medal": "",
                "category": category,
                "file": f"{task_name}.json"
            }
        )
    with open(demo_config_save_path,"w",encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)

    





