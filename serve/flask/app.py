import os, sys, json
import uuid
from flask import Flask, request, jsonify
import numpy as np
from common import (is_non_empty_file, parse_params, safe_literal_eval, random_uuid)
from evaluation import DATA_DIR_PATH, ModelEvaluation
from evaluation import MODEL_CONFIG, DATA_CONFIG, CONIFG_DIR_PATH
from score import ScoreCalculator
from evalInterfaceV3 import gen_eval_report

MODEL_DICT = {model["name"].split('/')[-1]: model for model in MODEL_CONFIG["models"]}
DATA_DICT = {}
for DATA_CATEGORY in DATA_CONFIG:
    for DATA in DATA_CATEGORY['datasets']:
        DATA_DICT[DATA['data_id']] = DATA
RENAME_DATA = {
    'political_ethics_dataset': '政治伦理',
    'economic_ethics_dataset': '经济伦理',
    'social_ethics_dataset': '社会伦理',
    'cultural_ethics_dataset': '文化伦理',
    'technology_ethics_dataset': '科技伦理',
    'environmental_ethics_dataset': '环境伦理',
    'medical_ethics_dataset': '医疗健康伦理',
    'education_ethics_dataset': '教育伦理',
    'professional_ethics_dataset': '职业道德伦理',
    'cyber_information_ethics_dataset': '网络伦理',
    'international_relations_ethics_dataset': '国际关系与全球伦理',
    'psychology_ethics_dataset': '心理伦理',
    'bioethics_dataset': '生物伦理学',
    'sports_ethics_dataset': '运动伦理学',
    'military_ethics_dataset': '军事伦理'
}

app = Flask(__name__)
score_calculator = ScoreCalculator()

@app.route("/run_evaluate", methods=["POST"])
def run_evaluate():
    params = request.get_json()
    params_config = {
        'task_id': (None, str),
        'model_names': ('[]', safe_literal_eval),
        'model_ids': ('[]', safe_literal_eval),
        'data_ids': ('[]', safe_literal_eval),
        'revision': (None, str),
        'question_begin': (None, int),
        'question_end': (None, int),
        'max_new_token': (1024, int),
        'num_choices': (1, int),
        'num_gpus_per_model': (1, int),
        'num_gpus_total': (1, int),
        'max_gpu_memory': (70, int),
        'dtype': (None, str),
        'cache_dir': ("/home/Userlist/madehua/model", str)
    }
    eval_task = ModelEvaluation(params, params_config)
    eval_task.run()
    return jsonify(f"{eval_task.run()}")

@app.route('/get_modelpage_list', methods=['POST'])
def get_modelpage_list():
    request_id = random_uuid()
    result = MODEL_CONFIG.copy()
    result.update({"request_id": request_id})
    return json.dumps(result, ensure_ascii=False)

@app.route('/get_modelpage_detail', methods=['POST'])
def get_modelpage_detail():
    request_id = random_uuid()
    data = request.json
    if not all(key in data for key in ['model_name']):
        return jsonify({"error": "Missing required fields in the request"}), 400

    MODEL_NAME = data.get('model_name')
    DATA_IDS = list(DATA_DICT.keys())
    score_caler = ScoreCalculator()
    print("model_name:", MODEL_NAME, "data_ids:", DATA_IDS)
    # overall_report = calculate_model_scores(DATA_IDS)
    report_per_model, report_per_data = score_caler.calculate_model_scores_dimension("moral_bench_test5")
    print("report_per_model:", report_per_model)
    print("report_per_data:", report_per_data)
    # sys_prompt = get_system_prompt()
    # report = generate_report(sys_prompt, overall_report[MODEL_ID]["error_examples"])
    report = {}
    try:
        MODEL_NAME = MODEL_NAME.split('/')[-1] if MODEL_NAME not in report_per_model else MODEL_NAME
    except AttributeError as e:
        print(e)
        return jsonify({"error": f"Model NAME '{MODEL_NAME}' not found in the report", "code": "ModelNotFound"}), 404
    if MODEL_NAME not in report_per_model:
        return jsonify({"error": f"Model NAME '{MODEL_NAME}' not found in the report", "code": "ModelNotFound"}), 404
    else:
        ability_scores = report_per_model[MODEL_NAME]["score_per_category"]
        ability_scores_array = []
        for ability, scores in ability_scores.items():
            ability_scores_array.append({"ability": ability, **scores})

        scores_per_data_id = report_per_model[MODEL_NAME]["scores_per_data_id"]
        data_id_scores = []
        for data_id, scores in scores_per_data_id.items():
            data_id_scores.append(
                {"data_id": data_id, "score": scores["correct"], "total": scores["total"],
                 "accuracy": scores["accuracy"]})
        result = {
            "request_id": str(request_id),
            "model_name": MODEL_NAME,
            "score": report_per_model[MODEL_NAME]["score_total"],
            "correct": report_per_model[MODEL_NAME]["total_correct"],
            "total": report_per_model[MODEL_NAME]["total_questions"],
            "ability_scores": ability_scores_array,
            "data_id_scores": data_id_scores,
            "model_description": MODEL_DICT.get(MODEL_NAME, {}),
            "report": report
        }
        return json.dumps(result, ensure_ascii=False)
    
@app.route('/get_datapage_list', methods=['POST'])
def get_datapage_list():
    request_id = random_uuid()
    result = DATA_CONFIG.copy()
    result.append({"request_id": request_id})
    return json.dumps(result, ensure_ascii=False)

@app.route('/get_datapage_detail', methods=['POST'])
def get_datapage_detail():
    request_id = random_uuid()
    data = request.json
    
    if not all(key in data for key in ['data_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    DATA_ID = data.get('data_id')
    DATA_RENAME = RENAME_DATA.get(DATA_ID, None)
    report_per_model, report_per_data = score_calculator.calculate_model_scores_dimension("moral_bench_test5")

    result = {
        "request_id": request_id,
        "data_id": DATA_ID,
        "data_description": DATA_DICT.get(DATA_ID, {}),
        "score": report_per_data.get(DATA_RENAME, 0),
        "model_ids": list(report_per_model.keys()),
    }
    return json.dumps(result, ensure_ascii=False)

@app.route('/get_leaderboard_detail', methods=['POST'])
def get_leaderboard_detail():
    print("get_leaderboard_detail_visablessssssssss")
    CATEGORY = ["合规性", "公平性", "知识产权", "隐私保护", "可信度"]
    filter_params = request.json
    categories = filter_params.get('categories', None)
    if categories is None:
        categories = CATEGORY.copy()
    model_sizes = filter_params.get('model_sizes', None)
    datasets = filter_params.get('datasets', None)
    print("categories:", categories, "model_sizes:", model_sizes, "datasets:", datasets)
    filtered_cates = CATEGORY.copy()
    if categories is not None:
        filtered_cates = [cate for cate in CATEGORY if cate in categories]
    filtered_models = [model["name"].split('/')[-1] for model in MODEL_CONFIG["models"]]
    if model_sizes is not None:
        filtered_models = [model for model in filtered_models if
                           any(size.lower() in model.lower() for size in model_sizes)]
    filtered_data = ["moral_bench_test5"]
    print("filtered_cates:", filtered_cates, "filtered_models:", filtered_models, "filtered_data:", filtered_data)

    report_per_model, report_per_data = score_calculator.calculate_model_scores_category("moral_bench_test5")
    aggregated_scores = {}
    for model_name in filtered_models:
        if model_name not in report_per_model:
            print("model_name not in report_per_model:", model_name)
            continue
        else:
            model_data = report_per_model[model_name]
            aggregated_scores[model_name] = {category: 0 for category in categories}
            aggregated_scores[model_name]['count'] = 0

            for category in categories:
                category_score = model_data['score_per_category'].get(category, {})
                aggregated_scores[model_name][category] = category_score.get('accuracy', 0)

            aggregated_scores[model_name]['count'] = model_data['total_questions']

    print("aggregated_scores:", aggregated_scores)

    final_data = []
    for model_name, scores in aggregated_scores.items():
        if model_name in filtered_models:
            avg_scores = {cat: scores[cat] for cat in categories}
            final_data.append({
                "模型": model_name,
                "发布日期": MODEL_DICT.get(model_name, {}).get('date', ''),
                "发布者": MODEL_DICT.get(model_name, {}).get('promulgator', ''),
                "国内/国外模型": MODEL_DICT.get(model_name, {}).get('country', ''),
                "参数量": MODEL_DICT.get(model_name, {}).get('parameters_size', ''),
                "综合": sum(avg_scores.values()) / len(categories),
                **avg_scores
            })
    print("final_data:", final_data)
    result = {
        "request_id": str(uuid.uuid4()),
        "header": [
                      "模型", "发布者", "发布日期", "国内/国外模型", "参数量", "综合"
                  ] + categories,
        "data": final_data
    }
    return json.dumps(result, ensure_ascii=False)

@app.route('/get_eval_report', methods=['POST'])
def get_eval_report():
    data = request.json
    log_folder = os.path.join(os.path.dirname(CONIFG_DIR_PATH), "log")
    log_json = None
    question_file = []
    model_name = []
    params_comfig = {
        'task_id': (None, str)
    }
    print("data",data)
    params = parse_params(data, params_comfig)
    task_id = params.get('task_id')
    print(task_id,str(task_id))
    with open(os.path.join(log_folder, "eval_log.jsonl"), 'r', encoding="utf-8") as f:
        log_lines = list(f)
    for line in reversed(log_lines):
        log = json.loads(line)
        if task_id in log.keys():
            log_json = log
            break
    try:
        if is_non_empty_file(f"./report/report_{task_id}.md"):
            pass
        else:
            for data_id in log_json[task_id]["data_ids"]:
                question_file.append(os.path.join(DATA_DIR_PATH, "data_upload", str(data_id), "question.jsonl"))
            for model in log_json[task_id]["model_names"]:
                model_name.append(model.split("/")[-1])
            time_suffix = log_json[task_id]["outputs"][0]["output"].split("/")[-1].split("_")[-1].split(".")[0]
            gen_eval_report(task_id, question_file, model_name, time_suffix)

            with open(f"./report/report_{task_id}.md", 'r', encoding="utf-8") as f:
                report = f.read()
            return report
    except:
        with open(f"./report/report.md", 'r', encoding="utf-8") as f:
            report = f.read()
        return report
    
@app.route('/cal_scores', methods=['POST'])
def cal_scores():
    data = request.json
    params_config = {
        "data_ids": ("[]", safe_literal_eval)
    }
    params = parse_params(data, params_config)
    data_ids = params["data_ids"]
    scores = []
    for data_id in data_ids:
        scores.append({data_id: score_calculator.calculate_model_scores_dimension(data_id)})
        result = {}
    for data_set_scores in scores:
        # print(type(data_set_scores), data_set_scores)
        model_score = []
        for data_id, model_scores in data_set_scores.items():
            score_all = []            
            for model, score in model_scores[0].items():
                score_all.append(score["result"])
                model_score.append(score["score_total"])
            score_all = np.array(score_all).astype(int)
            score_all_re = score_all.reshape(len(score["result"]), -1)
            variances = np.var(score_all_re, axis=1)
            result[data_id] = {}
            result[data_id]["variance"] = list(variances)
            result[data_id]["mean"] = np.mean(np.array(model_score))
            result[data_id]["var"] = np.var(np.array(model_score))
    for data_id, model_var in result.items():
        result[data_id]["index_varis0"] = [index for index, value in enumerate(model_var["variance"]) if value == 0]
    return result

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004, debug=True)
