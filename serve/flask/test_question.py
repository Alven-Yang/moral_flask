from common import *
from score import ScoreCalculator

question_dir_path = "/home/Userlist/yanganwen/code/dataset/4.12/"

question_list = os.listdir(question_dir_path)

params_json = {
    'model_names': '["01ai/Yi-6B-Chat","ZhipuAI/chatglm3-6b","ZhipuAI/chatglm2-6b","baichuan-inc/Baichuan2-7B-Chat","qwen/Qwen-7B-Chat"]',
    'model_ids': '["yi_6b_chat","chatglm3","chatglm2","baichuan2-chat","qwen-7b-chat"]'
}
questions_path = []
for file in question_list:
    questions_path.append(os.path.join(question_dir_path, file))

params_json["data_ids"] = questions_path

send_post_request("run_evaluate", params_json)

score_calculator = ScoreCalculator()
result = score_calculator.variance(params_json["data_ids"])
print(result)