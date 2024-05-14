import json
import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM, GenerationConfig, snapshot_download

def convert_expr(example):
    # rearrange declarations to the front
    sentences = example['fact_expressions'].split(';')
    sentences = sorted([s for s in sentences if ':' in s]) + \
        sorted([s for s in sentences if ':' not in s])
    exprs = ';'.join(sentences)
    example['expr'] = exprs + ';' + \
        ';'.join(
            list(map(lambda x: x + " = ?", example['query_expressions'].split(';'))))

    return example


model_name = "deepseek-ai/deepseek-math-7b-rl"
model_name = snapshot_download(model_name, 'master')
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")


prompt = """<｜begin▁of▁sentence｜>User\nThe logic expression of '{}' is:<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜>Assistant\n"""

file = "/home/zcl/LLaMA-Factory/data/math_problem/math_problem_test.json"
examples = json.load(open(file, "r", encoding="utf-8"))
results = []

for key, example in enumerate(examples):
    example = convert_expr(example)
    inputs = example['text'].strip()
    messages = prompt.format(inputs)

    input_tensor = tokenizer(messages,return_tensors="pt")
    outputs = model.generate(**input_tensor.to(model.device), max_new_tokens=512)

    result = tokenizer.decode(outputs[0])
    results.append(result)
    print(result)

output_filename = "/home/zcl/LLaMA-Factory/evaluate_model/outputs/DeepSeekMath-7b-rl/test.json"
with open(output_filename, 'a', encoding='utf8') as f:
    for i in results:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')