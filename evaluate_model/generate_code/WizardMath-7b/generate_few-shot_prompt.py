import torch
from modelscope import AutoModelForCausalLM, AutoTokenizer
import json

few_shot_prompt = r'''
Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Please translate the following sentence into logic expressions:已知双曲线$C$: $\\frac{x^{2}}{8}-y^{2}=1$的右焦点为$F$，渐近线为$l_{1}$, $l_{2}$，过点$F$的直线$l$与$l_{1}$, $l_{2}$的交点分别为$A$、$B$. 若$A B \\perp l_{2}$，则$|A B|$=?

### Response:
A: Point;B: Point;C: Hyperbola;F: Point;l1: Line;l2: Line;l: Line;Asymptote(C) = {l1,l2};Expression(C) = (x^2/8 - y^2 = 1);Intersection(l,l1) = A;Intersection(l,l2) = B;IsPerpendicular(LineSegmentOf(A,B),l2) = True;PointOnCurve(F,l) = True;RightFocus(C) = F;Abs(LineSegmentOf(A, B)) = ?

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Please translate the following sentence into logic expressions:短轴长为$2 \\sqrt{5}$，离心率$e=\\frac{2}{3}$的椭圆的两焦点为$F_{1}$、$F_{2}$，过$F_{1}$作直线交椭圆于$A$、$B$两点，则$\\triangle A B F_{2}$周长为?

### Response:
A: Point;B: Point;F1: Point;F2: Point;G: Ellipse;H: Line;e: Number;Eccentricity(G) = e;Focus(G) = {F1, F2};Intersection(H, G) = {A, B};Length(MinorAxis(G)) = 2*sqrt(5);PointOnCurve(F1, H) = True;e = 2/3;Perimeter(TriangleOf(A, B, F2)) = ?

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Please translate the following sentence into logic expressions:已知$P$是双曲线$\\frac{x^{2}}{4}-\\frac{y^{2}}{12}=1$上的动点，$F_{1}$、$F_{2}$分别是其左、右焦点，$O$为坐标原点，则$\\frac{|P F_{1}|+|P F_{2}|}{|P O|}$的取值范围是?

### Response:
F1: Point;F2: Point;G: Hyperbola;O: Origin;P: Point;Expression(G) = (x**2/4 - y**2/12=1);LeftFocus(G) = F1;PointOnCurve(P, G) = True;RightFocus(G) = F2;Range((Abs(LineSegmentOf(P, F1)) + Abs(LineSegmentOf(P, F2)))/Abs(LineSegmentOf(P, O))) = ?

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


model = AutoModelForCausalLM.from_pretrained("AI-ModelScope/WizardMath-7B-V1.0", revision='v1.0.0', device_map='auto', torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("AI-ModelScope/WizardMath-7B-V1.0", revision='v1.0.0')

prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nPlease translate the following sentence into logic expressions:{}\n\n### Response:"""

file = "/home/zcl/LLaMA-Factory/data/math_problem/test.json"
examples = json.load(open(file, "r", encoding="utf-8"))

results = []
for key, example in enumerate(examples):
    example = convert_expr(example)
    input = example['text'].strip()
    inputs =  prompt.format(input)
    input_tensor = tokenizer(inputs,padding=False, add_special_tokens=False, return_tensors="pt")

    # Generate
    generate_ids = model.generate(
        input_tensor.input_ids.to(model.device), 
        eos_token_id=[tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids('</s>')],
        max_length = 2048)
    
    result = tokenizer.batch_decode(generate_ids,skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    results.append(result)
    print(result)

output_filename = "/home/zcl/LLaMA-Factory/evaluate_model/outputs/WizardMath-7b-v1/few-shot_prompt_test.json"
with open(output_filename, 'a', encoding='utf8') as f:
    for i in results:
        json.dump(i, f, ensure_ascii=False)
        f.write('\n')