from common import *
from score import ScoreCalculator

file_path = "/home/yanganwen/download/questions_glm.jsonl"

score_calculator = ScoreCalculator()
report_per_model, report_per_data = score_calculator.calculate_model_scores_dimension("moral_bench_test5")

print(report_per_model.keys())
print(report_per_data["Llama-2-7b-chat-ms"])