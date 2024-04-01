import json
import os
from openai import OpenAI


def read_rule(fn):
    with open(fn, "r", encoding="utf-8") as f:
        rules = [line.replace("\u3000", "") for line in f.readlines()]
        return rules


def read_prompt(fn):
    with open(fn, "r", encoding="utf-8") as f:
        prompt = ""
        for line in f.readlines():
            prompt += line
        return prompt


model = "gpt-3.5-turbo-1106"
user_prompt = read_prompt("prompt_v6.txt")
client = OpenAI()


def contains(s):
    first_10_chars = s[:10]
    return '第' in first_10_chars and '条' in first_10_chars


def get_response(model, user_prompt):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
        )
        return completion.choices[0].message
    except Exception as e:
        print("请求过程中发生错误:", e)
        return None


def read_jsonl(filename):
    json_objects = []
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            # 解析JSON并添加到列表中
            json_objects.append(json.loads(line))
    return json_objects


def write_json(data, filename):
    try:
        with open(filename, 'a', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False)
            file.write('\n')
    except Exception as e:
        print("写入文件时发生错误:", e)


def main():
    category = ["艺术自由", "文化保护", "审美伦理"]
    rule_ = ""
    q = 0
    n = 0
    while n <= 10:
        for rule in rules:
            if len(rule) > 30 and contains(rule):
                rule_ += rule
                q += 1
                if q >= 5:
                    print(rule_)
                    user_prompt1 = user_prompt.format(topic=topic, policy=policy, category=category[n % 3], rule=rule_)
                    response = get_response(model, user_prompt1)
                    print(response.content)
                    if response:
                        try:
                            question = json.loads(response.content)
                            print(question)
                            write_json(question, f"{policy}.jsonl")
                            n += 1
                            if n >= 10:
                                break
                        except json.JSONDecodeError as e:
                            print("解析 JSON 时发生错误:", e)
                    rule_ = ""
                    q = 0


if __name__ == "__main__":
    # items = os.listdir("./rules")
    # topic = "艺术与文化伦理"
    # for item in items:
    #     rules = read_rule(f"./rules/{item}")
    #     policy = item.replace(".txt","")
    #     main()
    directory = "./"
    jsonl_files = []
    for file in os.listdir(directory):
        # Check if the file is a jsonl file and has a Chinese character in its name
        if file.endswith(".jsonl") and any('\u4e00' <= character <= '\u9fff' for character in file):
            jsonl_files.append(file)
    questions = []
    for jsonl_file in jsonl_files:
        lines = read_jsonl(jsonl_file)
        id = 1
        line_ = lines[0]
        line_["results"] = []
        for line in lines:
            results = []
            for q in line["results"]:
                q["id"] = id
                id += 1
                results.append(q)
            line_["results"].extend(results)
        questions.append(line_)
    for question in questions:
        write_json(question, "question2.jsonl")
