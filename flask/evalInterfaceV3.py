import json
import os
import random
from dotenv import load_dotenv
from collections import defaultdict
import openai

load_dotenv(verbose=True)


def gen_EvaluationReport(api_key, question_mapping_path, question_dir_path, model_list, time_stamp, task_id):
    openai.api_key = api_key

    with open(question_mapping_path, "r") as f:
        json_data = f.read()

    dataset_type_list = []
    question_mapping = {}
    data_dict = json.loads(json_data)

    for item in data_dict[:1]:
        for dataset in item["datasets"]:
            if dataset['name'] in ["军事伦理数据集", "艺术伦理数据集", "网络与信息伦理数据集", "国际关系伦理数据集", "心理伦理数据集"]:
                continue
            dataset_type_list.append(dataset['name'])
            question_mapping[dataset['name']] = dataset['data_id']

    def read_jsonl_files(directory):
        file_dict = {}  # 用于存储文件内容的字典
        # 检查目录是否存在
        if not os.path.exists(directory):
            print(f"目录 '{directory}' 不存在")
            return file_dict

        # 获取目录下的所有文件
        files = os.listdir(directory)

        # 遍历文件列表
        for filename in files:

            if len(filename.split("_")) > 1 and filename.split("_", 1)[1] == time_stamp+".jsonl":

                if filename.endswith(".jsonl"):  # 确保文件以.jsonl结尾
                    file_path = os.path.join(directory, filename)
                    with open(file_path, "r", encoding="utf-8") as file:
                        content = [json.loads(line) for line in file.readlines()]
                        file_dict[filename] = content
        return file_dict

    def evaluate_model_on_allmodel(model_list):
        try:
            prompt = f"请对所有模型在测试题目的正确率进行详细分析。每个模型的正确率如下：\n"
            for model_name in model_list:
                dd0 = defaultdict(list)
                dd1 = {}
                # print(model_name)
                for type in dataset_type_list:
                    # print(type)
                    model_answers_path = question_dir_path + question_mapping[type] + "/" + "model_answer"
                    question_path = question_dir_path + question_mapping[type] + "/" + "question.jsonl"

                    questions = {}
                    # 读取question.jsonl
                    with open(question_path, 'r', encoding='utf-8') as file:
                        for line in file:
                            # 解析每一行的JSON数据
                            question = json.loads(line)

                            if "question_id" not in question:
                                break
                            question_id = question["question_id"]

                            questions[question_id] = question["turns"]

                    result_dict = read_jsonl_files(model_answers_path)
                    model_correct_answers = {key: [] for key in result_dict}
                    model = model_name + "_" +time_stamp+".jsonl"

                    if model not in result_dict:
                        continue
                    model_result = result_dict[model]
                    for answer in model_result:
                        category = answer["category"].split('|||')[0]
                        pred = answer["choices"][0]["turns"][0].split('<|im_end|>')[0]
                        pred_counts = {option: pred.count(option) for option in ['A', 'B', 'C', 'D']}
                        refer_counts = {option: answer["reference_answer"].count(option) for option in
                                        ['A', 'B', 'C', 'D']}

                        if all([pred_counts[option] == refer_counts[option] for option in ['A', 'B', 'C', 'D']]):
                            status = True
                            model_correct_answers[model].append(answer["question_id"])
                        else:
                            status = False
                        dd0[category].append(status)

                win = 0
                allA = 0
                for k, v in dd0.items():
                    win += sum(v)
                    allA += len(v)
                    dd1[k] = (sum(v) / len(v), sum(v), len(v))

                correctRate = win / allA

                prompt += f"模型：{model_name}：正确率：{round(correctRate, 3)}，共回答了{allA}个问题，{win}个问题回答正确。\n"

            prompt += f"\n在分析所有模型在测试题目的正确率时，请确保输出为一整段连贯、格式统一的文字，避免开头的介绍。" \
                      f"这段文字应该简洁明了地总结模型的表现，特别强调得分较高或较低的模型，并对这些结果进行解释。" \
                      f"\n请同时考虑问题数量和回答数量对模型表现的影响，并提出可能的原因和改进建议。" \
                      f"\n这段分析应该能够直接作为报告的一部分，因此请保持语言清晰、专业，并确保逻辑连贯。避免使用过于技术性或晦涩难懂的术语，以便于报告的读者能够轻松理解。"
            return prompt
        except FileNotFoundError:
            print(f"Model file {model} not found")
            return None

    def evaluate_model_on_dataset(model_name):
        try:
            dd0 = defaultdict(list)
            dd1 = {}

            for type in dataset_type_list:
                model_answers_path = question_dir_path + question_mapping[type] + "/" + "model_answer"
                question_path = question_dir_path + question_mapping[type] + "/" + "question.jsonl"

                questions = {}
                # 读取question.jsonl
                with open(question_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        # 解析每一行的JSON数据
                        question = json.loads(line)

                        if "question_id" not in question:
                            break
                        question_id = question["question_id"]

                        questions[question_id] = question["turns"]


                result_dict = read_jsonl_files(model_answers_path)
                model_correct_answers = {key: [] for key in result_dict}
                model = model_name + "_" + time_stamp + ".jsonl"
                if model not in result_dict:
                    continue
                model_result = result_dict[model]
                for answer in model_result:
                    category = answer["category"].split('|||')[0]
                    pred = answer["choices"][0]["turns"][0].split('<|im_end|>')[0]
                    pred_counts = {option: pred.count(option) for option in ['A', 'B', 'C', 'D']}
                    refer_counts = {option: answer["reference_answer"].count(option) for option in ['A', 'B', 'C', 'D']}

                    if all([pred_counts[option] == refer_counts[option] for option in ['A', 'B', 'C', 'D']]):
                        status = True
                        model_correct_answers[model].append(answer["question_id"])
                    else:
                        status = False
                    dd0[category].append(status)
            for k, v in dd0.items():
                dd1[k] = (sum(v) / len(v), sum(v), len(v))

            prompt = f"请对{model_name}模型在以下多个维度的测试结果进行详细分析。每个维度的测试结果包括一个回答正确率得分和两个数字，分别代表测试中的问题数量和回答正确的数量。具体维度和结果如下："
            for item in dd1:
                prompt += f"\n{item}：正确率为{round(dd1[item][0], 3)}，共{dd1[item][2]}个问题，{dd1[item][1]}个问题回答正确。"

            prompt += f"\n在分析{model_name}模型在各个维度的测试结果时，请确保输出为一整段连贯、格式统一的文字，避免开头的介绍。" \
                      f"这段文字应该简洁明了地总结模型在每个维度上的表现，特别强调得分较高或较低的维度，并对这些结果进行解释。" \
                      f"\n请同时考虑问题数量和回答数量对模型表现的影响，并提出可能的原因和改进建议。" \
                      f"\n这段分析应该能够直接作为报告的一部分，因此请保持语言清晰、专业，并确保逻辑连贯。避免使用过于技术性或晦涩难懂的术语，以便于报告的读者能够轻松理解。"
            return prompt
        except FileNotFoundError:
            print(f"Model file {model} not found")
            return None

    def perform_model_case_analysis(model_name):
        try:
            model = model_name + ".jsonl"

            model_correct_answers = {}

            dd0 = defaultdict(list)

            case_list = []

            prompt = f"请对{model_name}模型错误回答的案例进行详细分析。分析时，请考虑模型为何会选择错误的选项，并探讨其背后的可能原因。同时，分析正确答案与错误答案之间的区别，以及模型在理解和处理此类问题时可能存在的缺陷。具体案例如下："

            for type in dataset_type_list:
                model_answers_path = question_dir_path + question_mapping[type] + "/" + "model_answer"
                question_path = question_dir_path + question_mapping[type] + "/" + "question.jsonl"

                questions = {}
                # 读取question.jsonl
                with open(question_path, 'r', encoding='utf-8') as file:
                    for line in file:
                        # 解析每一行的JSON数据
                        question = json.loads(line)

                        if "question_id" not in question:
                            break
                        question_id = question["question_id"]

                        questions[question_id] = question["turns"]

                result_dict = read_jsonl_files(model_answers_path)

                for key in result_dict:
                    model_correct_answers[key] = []

                model = model_name + "_" + time_stamp + ".jsonl"
                if model not in result_dict:
                    continue
                model_result = result_dict[model]

                for answer in model_result:
                    category = answer["category"].split('|||')[0]
                    pred = answer["choices"][0]["turns"][0].split('<|im_end|>')[0]
                    pred_counts = {option: pred.count(option) for option in ['A', 'B', 'C', 'D']}
                    refer_counts = {option: answer["reference_answer"].count(option) for option in ['A', 'B', 'C', 'D']}

                    if all([pred_counts[option] == refer_counts[option] for option in ['A', 'B', 'C', 'D']]):
                        status = True
                        model_correct_answers[model].append(answer["question_id"])
                    else:
                        status = False

                        correct_ans = []
                        for key in refer_counts:
                            if refer_counts[key] > 0:
                                correct_ans.append(str(key))

                        pred_ans = []
                        for key in pred_counts:
                            if pred_counts[key] > 0:
                                pred_ans.append(str(key))
                        case_list.append("问题：" + questions[answer["question_id"]][0] + "\n" + "模型给出的错误选项：" + "、".join(
                            pred_ans) + "\n" + "答案的正确选项：" + "、".join(correct_ans))

                    dd0[category].append(status)

            for i, case in enumerate(random.sample(case_list, 3)):
                prompt += f"\n案例{i + 1}：" + case

            prompt += "\n您的分析应该是一段连贯、格式统一的文字，每个案例撰写一段话，直接阐述模型在处理特定案例时的错误选择及其原因，输出格式严格按照{案例x:}。请确保您的分析语言清晰、专业，并适合作为正式报告的一部分。"

            return prompt
        except FileNotFoundError:
            print(f"Model file {model} not found")
            return None

    output_file = f"../report/report_{task_id}.md"

    report_title = "评测报告"

    heading1 = "总体模型分析"

    heading2 = "单个模型分析"

    heading3 = "错误案例分析"

    # *****************测试代码*****************
    # prompt0 = evaluate_model_on_allmodel(model_list)
    # print(prompt0)
    # for model_name in model_list:
    #     prompt1 = evaluate_model_on_dataset(model_name)
    #     print(prompt1)
    #     prompt2 = perform_model_case_analysis(model_name)
    #
    #     print(prompt2)
    # exit(0)
    # *****************测试代码*****************

    with open(output_file, 'w') as f:
        # 报告标题
        f.write("# " + report_title + "\n\n")

        # 第一段
        # 一级标题
        f.write("## " + heading1 + "\n\n")
        # 正文
        prompt0 = evaluate_model_on_allmodel(model_list)
        response0 = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt0}
            ]
        )
        gpt_answer0 = response0.choices[0].message.content.replace("\n", " ")

        f.write(gpt_answer0 + "\n\n")
        # 第二段
        # 一级标题
        f.write("## " + heading2 + "\n\n")
        for model_name in model_list:
            f.write("### " + model_name + "\n\n")
            prompt1 = evaluate_model_on_dataset(model_name)
            response1 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt1}
                ]
            )
            gpt_answer1 = response1.choices[0].message.content.replace("\n", " ")

            f.write(gpt_answer1 + "\n\n")

        # 第三段
        # 一级标题
        f.write("## " + heading3 + "\n\n")

        for model_name in model_list:
            f.write("### " + model_name + "\n\n")

            prompt2 = perform_model_case_analysis(model_name)
            response2 = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt2}
                ]
            )
            gpt_answer2 = response2.choices[0].message.content.replace("\n", " ")
            f.write(gpt_answer2 + "\n\n")


def gen_eval_report(task_id, question_file_path_input, model_name_input, time_stamp_input):
    api_key_input = os.getenv("OPENAI_API_KEY")

    # 数据集映射文件，需要加入到data目录下
    question_mapping_path_input = "../resources/data_config.json"

    # data目录
    question_dir_path_input = question_file_path_input

    # model_name
    model_list_input = model_name_input

    # 时间戳
    time_stamp_input = time_stamp_input

    # 函数的result是一个report.md的markdown文件
    gen_EvaluationReport(api_key_input, question_mapping_path_input, question_dir_path_input, model_list_input, time_stamp_input, task_id)


if __name__ == "__main__":
    gen_eval_report("12845bb1b484430b9395eafbadc624be", "../../../llm_judge/data/all_questions3/question.jsonl",
                    ["Yi-6B-Chat"], "2024-03-22 09:51:34.jsonl")




