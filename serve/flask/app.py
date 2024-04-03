import os, sys, json
from flask import Flask, request, jsonify
from common import (safe_literal_eval, random_uuid)
from evaluation import ModelEvaluation
from evaluation import MODEL_CONFIG, DATA_CONFIG

MODEL_DICT = {model["name"].split('/')[-1]: model for model in MODEL_CONFIG["models"]}
DATA_DICT = {}
for DATA_CATEGORY in DATA_CONFIG:
    for DATA in DATA_CATEGORY['datasets']:
        DATA_DICT[DATA['data_id']] = DATA

app = Flask(__name__)

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
    print("model_name:", MODEL_NAME, "data_ids:", DATA_IDS)
    # overall_report = calculate_model_scores(DATA_IDS)
    report_per_model, report_per_data = calculate_model_scores_category("moral_bench_test5")
    print("report_per_model:", report_per_model)
    print("report_per_data:", report_per_data)
    # sys_prompt = get_system_prompt()
    # report = generate_report(sys_prompt, overall_report[MODEL_ID]["error_examples"])
    report = get_cache()
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

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005, debug=True)
