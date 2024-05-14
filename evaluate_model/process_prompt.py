import json

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

prompt = "<|im_start|>user\nPlease translate the following sentence into logic expressions:{}<|im_end|>\n<|im_start|>assistant\n{}<|im_end|>"

def process_prompt(filename):
    all_prompt = []
    prompts = json.load(open(filename, "r", encoding="utf-8"))
    for p in prompts:
        p = convert_expr(p)
        final_prompt = prompt.format(p['text'].strip(),p['expr'].strip())
        print(final_prompt)
        all_prompt.append(final_prompt)
    return all_prompt

result = process_prompt("/home/zcl/LLaMA-Factory/evaluate_model/prompt.json")

output_filename = "/home/zcl/LLaMA-Factory/evaluate_model/few-shot_prompt.json"
with open(output_filename, 'a', encoding='utf8') as f:
    for i in result:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')
