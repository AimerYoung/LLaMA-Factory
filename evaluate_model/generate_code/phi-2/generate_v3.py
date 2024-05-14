import torch
import json
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

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


model = AutoModelForCausalLM.from_pretrained("/home/zcl/LLaMA-Factory/saves/phi-2/lora/sft/checkpoint-1562", torch_dtype="auto", device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/zcl/LLaMA-Factory/saves/phi-2/lora/sft/checkpoint-1562", trust_remote_code=True)

file = "/home/zcl/LLaMA-Factory/data/math_problem/math_problem_test.json"
examples = json.load(open(file, "r", encoding="utf-8"))
result = []
for key, example in enumerate(examples):
   example = convert_expr(example)
   inputs = "Q: Please translate the following sentence into logic expressions: {}\nA:".format(example['text'])

   inputs = tokenizer(inputs, return_tensors="pt", return_attention_mask=False).input_ids.to('cuda')
   outputs = model.generate(inputs, max_length=1024,eos_token_id=tokenizer.eos_token_id)
   text = tokenizer.batch_decode(outputs)[0]
   result.append(text)
   print(text)


output_filename = "/home/zcl/LLaMA-Factory/evaluate_model/outputs/phi-2-lora-semantic/outputs_5k_test.json"
with open(output_filename, 'a', encoding='utf8') as f:
    for i in result:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')

