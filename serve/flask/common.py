import datetime
import os
import ast
import json
import shutil
import subprocess
import time
from typing import Optional
import uuid
from dotenv import load_dotenv

import pytz
import requests

load_dotenv()

def parse_params(data, params_config):
    """
    根据提供的参数配置解析请求数据。

    :param data: 包含参数的字典。
    :param params_config: 一个字典，键为参数名，值为包含默认值和解析函数的元组。
    :return: 解析后的参数字典。
    """
    parsed_params = {}
    for param, (default, parse_func) in params_config.items():
        value_str = data.get(param, default)
        value = parse_func(value_str) if value_str else default
        parsed_params[param] = value

    return parsed_params

def safe_literal_eval(value_str):
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str
    
def load_questions(question_file: str, begin: Optional[int], end: Optional[int]):
    """Load questions from a file."""
    questions = []
    with open(question_file, "r") as ques_file:
        for line in ques_file:
            if line:
                temp = json.loads(line)
                temp["turns"][0] = "阅读题干，并从所给选项中选出你认为最正确的一项,无需说明理由，不要有任何多余输出，仅输出选项A、B、C、D中的一个即可:" + temp["turns"][
                    0]
                questions.append(temp)
    questions = questions[begin:end]
    return questions

def get_free_gpus():
    try:
        # 执行 nvidia-smi 命令
        cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")

        # 分析输出结果
        free_gpus = []
        lines = output.strip().split("\n")
        for line in lines:
            index, memory_used = line.split(", ")
            if int(memory_used) <= 300:
                free_gpus.append(int(index))

        return free_gpus
    except Exception as e:
        print(f"Error: {e}")
        return []
    
def random_uuid() -> str:
    return str(uuid.uuid4().hex)

def is_non_empty_file(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0

def copy_file(source_file, destination_folder):
    try:
        # 使用 shutil 的 copy2 函数来复制文件，保留元数据（如修改时间）
        shutil.copy2(source_file, destination_folder)
        print("文件复制成功！")
    except FileNotFoundError:
        print("找不到源文件或目标文件夹。")
    except PermissionError:
        print("权限错误，无法复制文件。")
    except Exception as e:
        print("发生了未知错误:", e)

def get_start_time():
    start_time = time.time()
    dt_utc = datetime.datetime.fromtimestamp(start_time, tz=pytz.utc)
    dt_beijing = dt_utc.astimezone(pytz.timezone("Asia/Shanghai"))
    formatted_start_time = dt_beijing.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_start_time


def get_end_time():
    end_time = time.time()
    dt_utc = datetime.datetime.fromtimestamp(end_time, tz=pytz.utc)
    dt_beijing = dt_utc.astimezone(pytz.timezone("Asia/Shanghai"))
    formatted_end_time = dt_beijing.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_end_time

def append_dict_to_jsonl(file_path, data_dict):
    with open(file_path, 'a', encoding='utf-8') as f:
        print("save the file_path to", file_path)
        try:
            json_str = json.dumps(data_dict, ensure_ascii=False)
            f.write(json_str + '\n')
        except TypeError as e:
            print("TypeError: ", e, data_dict)
        except UnboundLocalError as e2:
            print("UnboundLocalError: ", e2, data_dict)

def read_jsonl_files(directory):
    file_dict = {}
    if not os.path.exists(directory):
        print(f"目录 '{directory}' 不存在")
        return file_dict

    files = os.listdir(directory)

    # 遍历文件列表
    for filename in files:
        if filename.endswith(".jsonl"):  # 确保文件以.jsonl结尾
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = [json.loads(line) for line in file.readlines()]
                file_dict[filename.split('.jsonl')[0]] = content

    return file_dict

def send_post_request(route, params_json):
    vcis11 = os.getenv("VCIS11")
    url = f"http://{vcis11}:5004/{route}"
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, json=params_json, headers=headers)
    
    if response.status_code == 200:
        return response.json()
    else:
        return response.status_code
    
if __name__ == "__main__":

    pass