import os, sys, json
from flask import Flask, request, jsonify
from common import *
from evaluation import ModelEvaluation

CONIFG_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "config"))
with open(os.path.join(CONIFG_DIR_PATH, "data_config.json"), "r", encoding="utf-8") as f:
    DATA_CONFIG = json.load(f)
print(DATA_CONFIG[0]["datasets"])
DATA_STD_ID = []
for dataset in DATA_CONFIG[0]["datasets"]:
    DATA_STD_ID.append(dataset["data_id"])
MODEL_STD_ID = []
with open(os.path.join(CONIFG_DIR_PATH, "model_config.json"), "r", encoding="utf-8") as f:
    MODEL_CONFIG = json.load(f)

app = Flask(__name__)

@app.route("run_evaluate", methods=["POST"])
def run_evaluate():
    params = request.get_json()
    params_config = {
        'task_id': (None, str),
        'model_names': ('[]', safe_literal_eval),
        'model_ids': ('[]', safe_literal_eval),
        'data_ids': ('[]', safe_literal_eval)
    }
    eval_task = ModelEvaluation(params, params_config)
    return jsonify(f"{eval_task.run()}")
