import torch
import json
from modelscope import AutoModelForCausalLM, AutoTokenizer


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


model = AutoModelForCausalLM.from_pretrained("AI-ModelScope/WizardMath-7B-V1.0", revision='v1.0.0', device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/WizardMath-7B-V1.0", revision='v1.0.0')

prompt = """### Instruction:\nPlease translate the following sentence into logic expressions:</s>\n\n### Input:\n{}</s>\n\n### Response:\n"""

file = "/home/zcl/LLaMA-Factory/data/math_problem/math_problem_test.json"
examples = json.load(open(file, "r", encoding="utf-8"))

results = []
for key, example in enumerate(examples):
    example = convert_expr(example)
    input = example['text'].strip()
    inputs = prompt.format(input)
    input_tensor = tokenizer(inputs, return_tensors="pt")

    # Generate
    generate_ids = model.generate(
        input_tensor.input_ids.to(model.device), 
        eos_token_id=[tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('</s>')],
        max_length = 2048)
    result = tokenizer.batch_decode(generate_ids)[0]
    results.append(result)
    print(result)

output_filename = "/home/zcl/LLaMA-Factory/evaluate_model/outputs/WizardMath-7b-v1/few-shot_prompt_test.json"
with open(output_filename, 'a', encoding='utf8') as f:
    for i in results:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')