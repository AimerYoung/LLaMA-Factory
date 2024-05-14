import torch
import json
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig

prompt_qwen = r'''
<|im_start|>user
Please translate the following sentence into logic expressions:已知双曲线$\\frac{x^{2}}{3}-y^{2}=1$的左右焦点分别为$F_{1}$、$F_{2}$、$P$为双曲线右支上一点，点$Q$的坐标为$(-2,3)$，则$|P Q|+|P F_{1}|$的最小值为?<|im_end|>
<|im_start|>assistant
F1: Point;F2: Point;G: Hyperbola;P: Point;Q: Point;Coordinate(Q) = (-2, 3);Expression(G) = (x^2/3 - y^2 = 1);LeftFocus(G) = F1;PointOnCurve(P, RightPart(G));RightFocus(G) = F2;Min(Abs(LineSegmentOf(P, F1)) + Abs(LineSegmentOf(P, Q))) = ?<|im_end|>
<|im_start|>user
Please translate the following sentence into logic expressions:已知双曲线$C$: $\\frac{x^{2}}{8}-y^{2}=1$的右焦点为$F$，渐近线为$l_{1}$, $l_{2}$，过点$F$的直线$l$与$l_{1}$, $l_{2}$的交点分别为$A$、$B$. 若$A B \\perp l_{2}$，则$|A B|$=?<|im_end|>
<|im_start|>assistant
A: Point;B: Point;C: Hyperbola;F: Point;l1: Line;l2: Line;l: Line;Asymptote(C) = {l1,l2};Expression(C) = (x^2/8 - y^2 = 1);Intersection(l,l1) = A;Intersection(l,l2) = B;IsPerpendicular(LineSegmentOf(A,B),l2) = True;PointOnCurve(F,l) = True;RightFocus(C) = F;Abs(LineSegmentOf(A, B)) = ?<|im_end|>
<|im_start|>user
Please translate the following sentence into logic expressions:已知椭圆$\\frac{x^{2}}{25}+\\frac{y^{2}}{9}=1$与双曲线$\\frac{x^{2}}{a}-\\frac{y^{2}}{7}=1$焦点重合，则该双曲线的离心率为?<|im_end|>
<|im_start|>assistant
G: Hyperbola;H: Ellipse;a: Number;Expression(G) = (-y^2/7 + x^2/a = 1);Expression(H) = (x^2/25 + y^2/9 = 1);Focus(G) = Focus(H);Eccentricity(G) = ?<|im_end|>
<|im_start|>user
Please translate the following sentence into logic expressions:短轴长为$2 \\sqrt{5}$，离心率$e=\\frac{2}{3}$的椭圆的两焦点为$F_{1}$、$F_{2}$，过$F_{1}$作直线交椭圆于$A$、$B$两点，则$\\triangle A B F_{2}$周长为?<|im_end|>
<|im_start|>assistant
A: Point;B: Point;F1: Point;F2: Point;G: Ellipse;H: Line;e: Number;Eccentricity(G) = e;Focus(G) = {F1, F2};Intersection(H, G) = {A, B};Length(MinorAxis(G)) = 2*sqrt(5);PointOnCurve(F1, H) = True;e = 2/3;Perimeter(TriangleOf(A, B, F2)) = ?<|im_end|>
<|im_start|>user
Please translate the following sentence into logic expressions:已知$P$是双曲线$\\frac{x^{2}}{4}-\\frac{y^{2}}{12}=1$上的动点，$F_{1}$、$F_{2}$分别是其左、右焦点，$O$为坐标原点，则$\\frac{|P F_{1}|+|P F_{2}|}{|P O|}$的取值范围是?<|im_end|>
<|im_start|>assistant
F1: Point;F2: Point;G: Hyperbola;O: Origin;P: Point;Expression(G) = (x**2/4 - y**2/12=1);LeftFocus(G) = F1;PointOnCurve(P, G) = True;RightFocus(G) = F2;Range((Abs(LineSegmentOf(P, F1)) + Abs(LineSegmentOf(P, F2)))/Abs(LineSegmentOf(P, O))) = ?<|im_end|>
'''

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


model = AutoModelForCausalLM.from_pretrained("/home/zcl/LLaMA-Factory/saves/qwen-1.5-7b-prompt-5lr/lora/sft/checkpoint-1940", torch_dtype="auto", device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/zcl/LLaMA-Factory/saves/qwen-1.5-7b-prompt-5lr/lora/sft/checkpoint-1940", trust_remote_code=True)

prompt = "<|im_start|>user\nPlease translate the following sentence into logic expressions:{}<|im_end|>\n<|im_start|>assistant\n"

file = "/home/zcl/LLaMA-Factory/data/math_problem/math_problem_test.json"
examples = json.load(open(file, "r", encoding="utf-8"))
encoder_decoder = True
result = []
for key, example in enumerate(examples):
    example = convert_expr(example)
    if encoder_decoder:
        inputs = example['text'].strip()
    else:
        inputs = 'The translation of "{}" is:'.format(example['text'].strip())

    final_inputs = prompt_qwen + prompt.format(inputs)
   
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


output_filename = "/home/zcl/LLaMA-Factory/evaluate_model/outputs/qwen-1.5-lora-semantic/prompt_5lr_test.json"
with open(output_filename, 'a', encoding='utf8') as f:
    for i in result:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')