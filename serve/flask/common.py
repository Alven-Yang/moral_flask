import ast
import json
import subprocess
from typing import Optional
import uuid


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