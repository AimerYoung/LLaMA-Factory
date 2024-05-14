import json
import datasets
from typing import Any, Dict, Generator, List, Tuple

# from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
import re


_DESCRIPTION = "A program to process dataset."
_CITATION = ""
_HOMEPAGE = ""
_LICENSE = ""
_URL_Train = "math_problem_train.json"
_URL_Dev = "math_problem_dev.json"


class ExampleDataset(datasets.GeneratorBasedBuilder):

    VERSION = datasets.Version("0.0.0")

    def _info(self) -> datasets.DatasetInfo:
        features = datasets.Features({
            "input": datasets.Value("string"),
            "output": datasets.Value("string")
        })
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager) -> List[datasets.SplitGenerator]:
        file_path_train = dl_manager.download(_URL_Train)
        file_path_dev = dl_manager.download(_URL_Dev)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": file_path_train
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": file_path_dev
                }
            )
        ]

    def _generate_examples(self, filepath: str) -> Generator[Tuple[int, Dict[str, Any]], None, None]:
        example_dataset = json.load(open(filepath, "r", encoding="utf-8"))
        for key, example in enumerate(example_dataset):
            example = convert_expr(example)
            yield key, {
                        "input": few_shot_prompt + "<｜begin▁of▁sentence｜>User\nThe logic expression of '{}' is:<｜end▁of▁sentence｜>\n<｜begin▁of▁sentence｜>Assistant\n".format(example['text'].strip())+"\nAssistant:",
                        "output": example['expr'].strip()+"<｜end▁of▁sentence｜>"}

            
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



few_shot_prompt = r'''
<｜begin▁of▁sentence｜>User\nThe logic expression of '已知双曲线$\\frac{x^{2}}{3}-y^{2}=1$的左右焦点分别为$F_{1}$、$F_{2}$、$P$为双曲线右支上一点，点$Q$的坐标为$(-2,3)$，则$|P Q|+|P F_{1}|$的最小值为?' is:<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>Assistant\nF1: Point;F2: Point;G: Hyperbola;P: Point;Q: Point;Coordinate(Q) = (-2, 3);Expression(G) = (x^2/3 - y^2 = 1);LeftFocus(G) = F1;PointOnCurve(P, RightPart(G));RightFocus(G) = F2;Min(Abs(LineSegmentOf(P, F1)) + Abs(LineSegmentOf(P, Q))) = ?<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>User\nThe logic expression of '已知双曲线$C$: $\\frac{x^{2}}{8}-y^{2}=1$的右焦点为$F$，渐近线为$l_{1}$, $l_{2}$，过点$F$的直线$l$与$l_{1}$, $l_{2}$的交点分别为$A$、$B$. 若$A B \\perp l_{2}$，则$|A B|$=?' is:<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>Assistant\nA: Point;B: Point;C: Hyperbola;F: Point;l1: Line;l2: Line;l: Line;Asymptote(C) = {l1,l2};Expression(C) = (x^2/8 - y^2 = 1);Intersection(l,l1) = A;Intersection(l,l2) = B;IsPerpendicular(LineSegmentOf(A,B),l2) = True;PointOnCurve(F,l) = True;RightFocus(C) = F;Abs(LineSegmentOf(A, B)) = ?<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>User\nThe logic expression of '已知椭圆$\\frac{x^{2}}{25}+\\frac{y^{2}}{9}=1$与双曲线$\\frac{x^{2}}{a}-\\frac{y^{2}}{7}=1$焦点重合，则该双曲线的离心率为?' is:<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>Assistant\nG: Hyperbola;H: Ellipse;a: Number;Expression(G) = (-y^2/7 + x^2/a = 1);Expression(H) = (x^2/25 + y^2/9 = 1);Focus(G) = Focus(H);Eccentricity(G) = ?<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>User\nThe logic expression of '短轴长为$2 \\sqrt{5}$，离心率$e=\\frac{2}{3}$的椭圆的两焦点为$F_{1}$、$F_{2}$，过$F_{1}$作直线交椭圆于$A$、$B$两点，则$\\triangle A B F_{2}$周长为?' is:<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>Assistant\nA: Point;B: Point;F1: Point;F2: Point;G: Ellipse;H: Line;e: Number;Eccentricity(G) = e;Focus(G) = {F1, F2};Intersection(H, G) = {A, B};Length(MinorAxis(G)) = 2*sqrt(5);PointOnCurve(F1, H) = True;e = 2/3;Perimeter(TriangleOf(A, B, F2)) = ?<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>User\nThe logic expression of '已知$P$是双曲线$\\frac{x^{2}}{4}-\\frac{y^{2}}{12}=1$上的动点，$F_{1}$、$F_{2}$分别是其左、右焦点，$O$为坐标原点，则$\\frac{|P F_{1}|+|P F_{2}|}{|P O|}$的取值范围是?' is:<｜end▁of▁sentence｜>
<｜begin▁of▁sentence｜>Assistant\nF1: Point;F2: Point;G: Hyperbola;O: Origin;P: Point;Expression(G) = (x**2/4 - y**2/12=1);LeftFocus(G) = F1;PointOnCurve(P, G) = True;RightFocus(G) = F2;Range((Abs(LineSegmentOf(P, F1)) + Abs(LineSegmentOf(P, F2)))/Abs(LineSegmentOf(P, O))) = ?<｜end▁of▁sentence｜>
'''