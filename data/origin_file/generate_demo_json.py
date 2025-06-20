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

# 1. 关键字列表
PY_KEYWORDS = set(keyword.kwlist)

from pygments.token import Token
import html

# 你的自定义 token->class 对应关系
TOKEN_CLASS_MAP = {
    Token.Comment: 'comment',
    Token.Keyword: 'keyword',
    Token.Literal.String: 'string',
    Token.Literal.Number: 'number',
    Token.Name.Function: 'function',
    Token.Name.Class: 'class',
}

class CustomHtmlFormatter(Formatter):
    def __init__(self):
        super().__init__()

    def format(self, tokensource, outfile):
        for ttype, value in tokensource:
            value = html.escape(value)
            # 查找最接近的 token 类型
            cls = None
            for token_type, class_name in TOKEN_CLASS_MAP.items():
                if ttype in token_type:
                    cls = class_name
                    break
            if cls:
                outfile.write(f'<span class="{cls}">{value}</span>')
            else:
                outfile.write(value)


json_path = "/mnt/sfs_turbo/xinyuzhu/ai-developer-draw/ML-Master/data/origin_file/task75_log_for_run0_25-06-12T16:48:19.json"
demo_data_save_path = "/mnt/sfs_turbo/xinyuzhu/ai-developer-draw/ML-Master/data/demos"
demo_config_save_path = "/mnt/sfs_turbo/xinyuzhu/ai-developer-draw/ML-Master/data/demo-config.json"
raw_log_path = "/mnt/sfs_turbo/xinyuzhu/ai-developer-draw/data/mcts_log_new/final"

with open(json_path,'r') as f:
    json_data = json.load(f)

pattern = re.compile(r'\[.*?\] (\w+): (.*)')
for task_name in json_data:
    log_path = os.path.join(raw_log_path,json_data[task_name]["run_id"],"aide.log")
    code_path = os.path.join(raw_log_path,json_data[task_name]["run_id"],"best_solution.py")
    log_str = ""
    code_str = ""
    steps = [
            {
      "text": f"<span class='prompt'>ml-master@ai4ai:~$</span> python ml_master.py --task {task_name} --time-limit 12h",
      "delay": 200
    },
    ]

    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

            # 如果超过2000行，只保留前1000行和后1000行，中间插入说明行
            if len(lines) > 2000:
                trimmed_lines = (
                    lines[:1000]
                    + ['[2025-05-27 09:53:59,762] INFO: ... (omitted middle part for brevity) ...\n']
                    + lines[-1000:]
                )
                print(f"任务{task_name}日志过长，触发省略")
            else:
                trimmed_lines = lines
            for idx,line in enumerate(trimmed_lines):

                if idx == 0:
                    line = f"<span class='info'>[INFO]</span>: Starting run \"{task_name}\""
                    log_str = log_str+line
                    steps.append({"text":line,"delay":200})
                else:
                    line = line.strip()  # 去除每行前后的空白符
                    if "已分配" in line or "已设置" in line or "调用失败" in line or "完整报错" in line or "Traceback (most recent call last)" in line or "ConnectionError" in line:
                        continue
                    match = pattern.match(line)
                    if match:
                        level, message = match.groups()
                        line = f"{level}: {message}"
                        line = f"<span class='{level.lower()}'>[{level}]</span>: {message}"
                    else:
                        if "已分配" in line or "已设置" in line or "调用失败" in line or "完整报错" in line or "Traceback (most recent call last)" in line or "ConnectionError" in line:
                            continue
                        else:
                            print(line)
                            print(f"去除时间戳失败，原始报错{traceback.format_exc()}")

                    log_str = log_str+line
                    steps.append({"text":line,"delay":200})

    except:
        print(f"处理日志{log_path}失败，原始保错:{traceback.format_exc()}")
        continue

    try:
        with open(code_path, 'r', encoding='utf-8') as f:
            for idx,line in enumerate(f):
                code_str = code_str + line
    except:
        print(f"处理代码{code_path}失败，原始保错:{traceback.format_exc()}")
        continue

    steps.append(
            {
      "text": "<span class='prompt'>ml-master@ai4ai:~$</span> ",
      "delay": 200
    }
    )
    
    # 找到medal
    if json_data[task_name]["medal"] == "gold medal":
        medal = "🥇"
    elif json_data[task_name]["medal"] == "silver medal":
        medal = "🥈"
    elif json_data[task_name]["medal"] == "bronze medal":
        medal = "🥉"
    elif json_data[task_name]["medal"] == "above median":
        medal = "🔝"
    elif json_data[task_name]["medal"] == "below median":
        medal = "✅"
    else:
        medal = "❌"


    # highlight_code = highlight_python_code(code_str)
    formatter = HtmlFormatter(nowrap=True)
    highlight_code = highlight(code_str, PythonLexer(), CustomHtmlFormatter())
    demo_data = {
        "title":task_name,
        "steps":steps,
        "code":highlight_code
    }
    with open(os.path.join(demo_data_save_path,f"{task_name}.json"),"w",encoding='utf-8') as f:
        json.dump(demo_data, f, ensure_ascii=False, indent=4)
    with open(demo_config_save_path,"r",encoding='utf-8') as f:
        config_data = json.load(f)
    
    task_already_in_demo = False
    for idx,task_dict in enumerate(config_data["demos"]):
        if task_dict["id"] == task_name:
            task_already_in_demo = True
            config_data["demos"][idx]["id"] = task_name
            config_data["demos"][idx]["icon"] = ""
            config_data["demos"][idx]["title"] = task_name
            config_data["demos"][idx]["medal"] = medal
            config_data["demos"][idx]["description"] = ""
            config_data["demos"][idx]["category"] = "others"
            config_data["demos"][idx]["file"] = f"{task_name}.json"
            break

    if task_already_in_demo == False:
        config_data["demos"].append(
            {
                "id": f"{task_name}",
                "icon": "",
                "title": f"{task_name}",
                "description": "",
                "medal": medal,
                "category": "others",
                "file": f"{task_name}.json"
            }
        )
    with open(demo_config_save_path,"w",encoding='utf-8') as f:
        json.dump(config_data, f, ensure_ascii=False, indent=4)

    


