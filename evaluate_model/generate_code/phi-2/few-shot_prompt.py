import torch
import json
import re
from modelscope import AutoModelForCausalLM, AutoTokenizer
from transformers import GenerationConfig


prompt = r'''
Q: Please translate the following sentence into logic expressions:已知双曲线$\\frac{x^{2}}{3}-y^{2}=1$的左右焦点分别为$F_{1}$、$F_{2}$、$P$为双曲线右支上一点，点$Q$的坐标为$(-2,3)$，则$|P Q|+|P F_{1}|$的最小值为?
A: F1: Point;F2: Point;G: Hyperbola;P: Point;Q: Point;Coordinate(Q) = (-2, 3);Expression(G) = (x^2/3 - y^2 = 1);LeftFocus(G) = F1;PointOnCurve(P, RightPart(G));RightFocus(G) = F2;Min(Abs(LineSegmentOf(P, F1)) + Abs(LineSegmentOf(P, Q))) = ?

Q: Please translate the following sentence into logic expressions:已知双曲线$C$: $\\frac{x^{2}}{8}-y^{2}=1$的右焦点为$F$，渐近线为$l_{1}$, $l_{2}$，过点$F$的直线$l$与$l_{1}$, $l_{2}$的交点分别为$A$、$B$. 若$A B \\perp l_{2}$，则$|A B|$=?
A: A: Point;B: Point;C: Hyperbola;F: Point;l1: Line;l2: Line;l: Line;Asymptote(C) = {l1,l2};Expression(C) = (x^2/8 - y^2 = 1);Intersection(l,l1) = A;Intersection(l,l2) = B;IsPerpendicular(LineSegmentOf(A,B),l2) = True;PointOnCurve(F,l) = True;RightFocus(C) = F;Abs(LineSegmentOf(A, B)) = ?

Q: Please translate the following sentence into logic expressions:已知椭圆$\\frac{x^{2}}{25}+\\frac{y^{2}}{9}=1$与双曲线$\\frac{x^{2}}{a}-\\frac{y^{2}}{7}=1$焦点重合，则该双曲线的离心率为?
A: G: Hyperbola;H: Ellipse;a: Number;Expression(G) = (-y^2/7 + x^2/a = 1);Expression(H) = (x^2/25 + y^2/9 = 1);Focus(G) = Focus(H);Eccentricity(G) = ?

Q: Please translate the following sentence into logic expressions:短轴长为$2 \\sqrt{5}$，离心率$e=\\frac{2}{3}$的椭圆的两焦点为$F_{1}$、$F_{2}$，过$F_{1}$作直线交椭圆于$A$、$B$两点，则$\\triangle A B F_{2}$周长为?
A: A: Point;B: Point;F1: Point;F2: Point;G: Ellipse;H: Line;e: Number;Eccentricity(G) = e;Focus(G) = {F1, F2};Intersection(H, G) = {A, B};Length(MinorAxis(G)) = 2*sqrt(5);PointOnCurve(F1, H) = True;e = 2/3;Perimeter(TriangleOf(A, B, F2)) = ?

Q: Please translate the following sentence into logic expressions:已知$P$是双曲线$\\frac{x^{2}}{4}-\\frac{y^{2}}{12}=1$上的动点，$F_{1}$、$F_{2}$分别是其左、右焦点，$O$为坐标原点，则$\\frac{|P F_{1}|+|P F_{2}|}{|P O|}$的取值范围是?
A: F1: Point;F2: Point;G: Hyperbola;O: Origin;P: Point;Expression(G) = (x**2/4 - y**2/12=1);LeftFocus(G) = F1;PointOnCurve(P, G) = True;RightFocus(G) = F2;Range((Abs(LineSegmentOf(P, F1)) + Abs(LineSegmentOf(P, F2)))/Abs(LineSegmentOf(P, O))) = ?
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


model = AutoModelForCausalLM.from_pretrained("/home/zcl/LLaMA-Factory/saves/phi-2-2.7b-prompt-1lr/lora/sft/checkpoint-3395", torch_dtype="auto", device_map="cuda", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("/home/zcl/LLaMA-Factory/saves/phi-2-2.7b-prompt-1lr/lora/sft/checkpoint-3395", trust_remote_code=True)


file = "/home/zcl/LLaMA-Factory/data/math_problem/math_problem_test.json"
examples = json.load(open(file, "r", encoding="utf-8"))
result = []
for key, example in enumerate(examples):
   example = convert_expr(example)
   inputs = "以上是我给出的5个样例，请你分析这些样例，回答我的问题。回答中请不要重复我的问题，请直接给出答案。\nQ: Please translate the following sentence into logic expressions: {}\nA: ".format(example['text'])

   inputs = prompt + inputs
   inputs = tokenizer(inputs, return_tensors="pt", return_attention_mask=False).input_ids.to('cuda')
   outputs = model.generate(inputs, max_length=2048,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.eos_token_id)
   text = tokenizer.batch_decode(outputs)[0]
   result.append(text)
   print(text)


output_filename = "/home/zcl/LLaMA-Factory/evaluate_model/outputs/phi-2-lora-semantic/prompt_1lr_test.json"
with open(output_filename, 'a', encoding='utf8') as f:
    for i in result:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')

