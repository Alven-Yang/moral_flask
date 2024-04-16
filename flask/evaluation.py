import gc
import json
import os
import random
import time
from typing import Optional
from fastapi import params
from flask import jsonify
from numpy import dtype
import shortuuid
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from common import (parse_params, load_questions, get_free_gpus, random_uuid, safe_literal_eval, is_non_empty_file, copy_file, get_start_time, get_end_time,
                    append_dict_to_jsonl)
from modelscope import snapshot_download
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel
from fastchat.model import get_conversation_template

CONIFG_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "resources", "config"))
DATA_DIR_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
with open(os.path.join(CONIFG_DIR_PATH, "data_config.json"), "r", encoding="utf-8") as f:
    DATA_CONFIG = json.load(f)
DATA_STD_ID = []
for dataset in DATA_CONFIG[0]["datasets"]:
    DATA_STD_ID.append(dataset["data_id"])
MODEL_STD_NAME = []
MODEL_STD_ID = []
with open(os.path.join(CONIFG_DIR_PATH, "model_config.json"), "r", encoding="utf-8") as f:
    MODEL_CONFIG = json.load(f)
    for model in MODEL_CONFIG['models']:
        MODEL_STD_NAME.append(model["name"])
        MODEL_STD_ID.append(model["model_id"])


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
            num_gpus_total: object,
            max_gpu_memory: object,
            dtype: object,
            revision: object,
            cache_dir: str,
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
            cache_dir,
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
            # llm = LLM(model=model_dir, trust_remote_code=True, tensor_parallel_size=max((free_gpu_num-(free_gpu_num & 1)), 1))
            llm = LLM(model=model_dir, trust_remote_code=True)
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
        revision: str = params.get('revision') # type: ignore
        question_begin = params.get('question_begin')
        question_end = params.get('question_end')
        max_new_token = params.get('max_new_token')
        num_choices = params.get('num_choices')
        num_gpus_per_model = params.get('num_gpus_per_model')
        num_gpus_total = params.get('num_gpus_toal')
        max_gpu_memory = params.get('max_gpu_memory')
        dtype = params.get('dtype')
        cache_dir = params.get('cache_dir')
        if len(model_names) != len(model_ids):
            print(model_names, model_ids)
            return jsonify({"error": "model_names and model_ids should have the same length"}), 400
        failed = []
        
        start_time = get_start_time()
        outputs = []

        for data_id in data_ids:
            if data_id not in DATA_STD_ID:
                if not is_non_empty_file(data_id):
                    return json.dumps({"error": f"data_id {data_id} not found"}), 400
                new_data_dir = os.path.join(os.path.join(DATA_DIR_PATH, "data_upload"), data_id.split('/')[-1].split('.')[0])
                print(new_data_dir)
                if not os.path.exists(new_data_dir) or not os.path.isdir(new_data_dir):
                    os.makedirs(new_data_dir)
                    os.makedirs(os.path.join(new_data_dir, "model_answer"))
                    copy_file(data_id, new_data_dir)
                    os.rename(os.path.join(new_data_dir, data_id.split("/")[-1]), os.path.join(new_data_dir, "question.jsonl"))
                data_id = str(data_id.split("/")[-1].split(".")[0])
                question_file = os.path.join(new_data_dir, "question.jsonl")
            else:
                question_file = os.path.join(DATA_DIR_PATH, "data_std", data_id, "question.jsonl")
            for model_name, model_id in zip(model_names, model_ids):
                model_name_saved = model_name.split('/')[-1]
                output_file = os.path.join(os.path.dirname(question_file), "model_answer", f"{model_name_saved}_{start_time}.jsonl")
                print("eval model:", model_name, model_id)
                try:
                    self.run_eval(
                        model_path=model_name, model_id=model_id, question_file=question_file,
                        question_begin=question_begin, question_end=question_end, # type: ignore
                        answer_file=output_file, max_new_token=max_new_token,
                        num_choices=num_choices, num_gpus_per_model=num_gpus_per_model,
                        num_gpus_total=num_gpus_total, max_gpu_memory=max_gpu_memory,
                        dtype=dtype, revision=revision, cache_dir=cache_dir # type: ignore
                    )
                except AttributeError as e:
                    print("eval model error:", model_name, model_id)
                    print(e)
                    failed.append({"model_id": model_id, "reason": str(e)})
                    continue
                except torch.cuda.OutOfMemoryError as e1:
                    print("eval model error:", model_name, model_id)
                    print(e1)
                    failed.append({"model_id": model_id, "reason": str(e1)})
                    continue
            temp = {"data_id": data_id,
                        "model_id": model_id, "model_name": model_name,
                        "output": output_file}
            outputs.append(temp)

        end_time = get_end_time()
        result = {
            "outputs": outputs,
            "model_names": model_names,
            "model_ids": model_ids,
            "data_ids": data_ids,
            "time_start": start_time,
            "time_end": end_time,
            "failed": failed
        }

        log_folder = os.path.join(os.path.dirname(CONIFG_DIR_PATH), "log")
        os.makedirs(log_folder, exist_ok=True)
        log_path = os.path.join(log_folder, "eval_log.jsonl")
        print("log_path:", log_path)
        append_dict_to_jsonl(log_path, {task_id: result})
        return jsonify(result)
        






if __name__ == "__main__":
    pass
