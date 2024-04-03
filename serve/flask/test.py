from common import *

file_path = "/home/yanganwen/download/questions_glm.jsonl"

questions = load_questions(file_path, None, None) 
print(questions[0])