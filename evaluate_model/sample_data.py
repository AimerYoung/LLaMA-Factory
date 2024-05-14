import json
import random

file = "/home/zcl/LLaMA-Factory/evaluate_model/data/math_problem_train.json"
examples = json.load(open(file, "r", encoding="utf-8"))
print(len(examples))

train_6k = random.sample(examples,6000)

output_file_6k = "/home/zcl/LLaMA-Factory/evaluate_model/data/train_6k.json"

with open(output_file_6k, 'a', encoding='utf8') as f:
    for i in train_6k:
        json.dump(i, f, ensure_ascii=False)
        f.write(',')
        f.write('\n')
