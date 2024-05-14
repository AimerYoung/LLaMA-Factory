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


model = AutoModelForCausalLM.from_pretrained("/home/zcl/LLaMA-Factory/saves/qwen-1.5-7b-0.4lr/lora/sft/checkpoint-2910", torch_dtype="auto", device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/zcl/LLaMA-Factory/saves/qwen-1.5-7b-0.4lr/lora/sft/checkpoint-2910", trust_remote_code=True)

prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease translate the following sentence into logic expressions:{}<|im_end|>\n<|im_start|>assistant\n"

file = "/home/zcl/LLaMA-Factory/data/math_problem/math_problem_test.json"
examples = json.load(open(file, "r", encoding="utf-8"))
encoder_decoder = False
result = []
for key, example in enumerate(examples):
    example = convert_expr(example)
    if encoder_decoder:
        inputs = example['text'].strip()
    else:
        inputs = 'The translation of "{}" is:'.format(example['text'].strip())

    final_inputs = prompt.format(inputs)
   
    model_inputs = tokenizer([final_inputs], return_tensors="pt").to('cuda')

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512,
        eos_token_id=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids('<|im_end|>')])
    
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    response = tokenizer.batch_decode(generated_ids)[0] #, skip_special_tokens=True
    result.append(response)
    print(response)


output_filename = "/home/zcl/LLaMA-Factory/evaluate_model/outputs/qwen-1.5-lora-semantic/0.4lr_test.json"
with open(output_filename, 'a', encoding='utf8') as f:
    for i in result:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')