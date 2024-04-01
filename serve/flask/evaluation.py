import gc
import json
import os
import random
import time
from typing import Optional
from fastapi import params
from flask import jsonify
import shortuuid
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from common import (parse_params, load_questions, get_free_gpus, random_uuid, safe_literal_eval)
from modelscope import snapshot_download
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from fastchat.model import get_conversation_template

CONIFG_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "config"))
with open(os.path.join(CONIFG_DIR_PATH, "data_config.json"), "r", encoding="utf-8") as f:
    DATA_CONFIG = json.load(f)
DATA_STD_ID = []
for dataset in DATA_CONFIG[0]["datasets"]:
    DATA_STD_ID.append(dataset["data_id"])
MODEL_STD_NAME = []
MODEL_STD_ID = []
with open(os.path.join(CONIFG_DIR_PATH, "model_config.json"), "r", encoding="utf-8") as f:
    MODEL_CONFIG = json.load(f)
    

class ModelEvaluation:
    def __init__(self, params_json, params_config):
        self.params = params_json
        self.params_config = params_config

    def params_to_dict(self):
        return parse_params(self.params, self.params_config)
    
    def prompt_generator(self, questions, model_id):
        prompts = []
        for question in tqdm(questions):
            conv = get_conversation_template(model_id)
            qs = '\n'.join(question["turns"])
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None) # type: ignore
            prompt = conv.get_prompt()
            prompts.append(prompt)

        return prompts
    
    def run_eval(
            self,
            model_path: object,
            model_id: object,
            question_file: str,
            question_begin: int,
            question_end: int,
            answer_file: object,
            max_new_token: object,
            num_choices: object,
            num_gpus_per_model: object,
            max_gpu_memory: object,
            dtype: object,
            revision: object,
            cache_dir: str = "/root/Userlist/madehua/model",
        ) -> object:
        questions = load_questions(question_file, question_begin, question_end)
        # random shuffle the questions to balance the loading
        random.shuffle(questions)

        self.get_model_answers(
            model_path,
            model_id,
            questions,
            answer_file,
            max_new_token,
            num_choices,
            num_gpus_per_model,
            max_gpu_memory,
            dtype=dtype,
            revision=revision,
            cache_dir=cache_dir,
        )

    @torch.inference_mode()
    def get_model_answers(
            self,
            model_path,
            model_id,
            questions,
            answer_file,
            max_new_token,
            num_choices,
            num_gpus_per_model,
            max_gpu_memory,
            dtype,
            revision,
            cache_dir="/root/Userlist/madehua/model",
    ):
        print("model_path:", model_path, "model_id:", model_id, "revision:", revision)
        free_gpu_num = len(get_free_gpus())
        try:
            model_dir = snapshot_download(model_path, cache_dir=cache_dir, revision=revision, local_files_only=True)
        except ValueError:
            model_dir = snapshot_download(model_path, cache_dir=cache_dir, revision=revision,
                                        local_files_only=False)
        print("model_dir:", model_dir)
        try:
            llm = LLM(model=model_dir, trust_remote_code=True, tensor_parallel_size=max((free_gpu_num-(free_gpu_num & 1)), 1))
            # llm = LLM(model=model_dir, trust_remote_code=True)
        except (ModuleNotFoundError, AttributeError, torch.cuda.OutOfMemoryError) as e:
            print(e)
            destroy_model_parallel()
            gc.collect()
            torch.cuda.empty_cache()
            return None
        
        prompts = self.prompt_generator(questions, model_id)

        sampling_params = SamplingParams(temperature=0.0)
        outputs = llm.generate(prompts, sampling_params)
        # print("len of prompts: ", len(prompts), len(outputs))
        for idx, (question, output) in enumerate(zip(questions, outputs)):
            prompt = output.prompt
            qs = '\n'.join(question["turns"])
            generated_text = output.outputs[0].text
            os.makedirs(os.path.dirname(answer_file), exist_ok=True)
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": model_id,
                    "choices": [{"index": 0, "turns": [generated_text]}],
                    "reference_answer": question["reference_answer"],
                    "question_type": question["question_type"],
                    "category": question['category'],
                    "dimension": question.get("dimension", None),
                    # "field": question['field'],
                    # "law": question['law'],
                    "prompt": prompt,
                    "question": qs,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json, ensure_ascii=False) + "\n")

        destroy_model_parallel()
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    def run(self):
        params: dict = self.params_to_dict()
        task_id = params.get('task_id') if params.get('task_id') else random_uuid()
        model_names: list[str] = params.get('model_names') if len(params.get('model_names')) > 0 else MODEL_STD_NAME # type: ignore
        model_ids: list[str] = params.get('model_ids') if len(params.get('model_ids')) > 0 else MODEL_STD_ID # type: ignore
        data_ids: list[str] = params.get('data_ids') if len(params.get('data_ids')) > 0 else DATA_STD_ID # type: ignore
        if len(model_names) != len(model_ids):
            print(model_names, model_ids)
            return jsonify({"error": "model_names and model_ids should have the same length"}), 400
        
        






if __name__ == "__main__":
    params_config = {
        'task_id': (None, str),
        'model_names': ('[]', safe_literal_eval),
        'model_ids': ('[]', safe_literal_eval),
        'data_ids': ('[]', safe_literal_eval)
    }
    params = '{"model_names": ["ZhipuAI/chatglm3-6b"], "data_ids": ["/home/yanganwen/download/all_questions3.jsonl"], "cache_dir": "", "task_id": "ddwd"}'
    eval_task = ModelEvaluation(json.loads(params), params_config)
    a = eval_task.params_to_dict()
    print(len(a['model_names']), len(a['model_ids']), len(a['data_ids']), a['model_ids'])
