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
            "instruction": datasets.Value("string"),
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
            yield key, {"instruction": "Please translate the following sentence into logic expressions:", 
                        "input": example['text'].strip(),
                        "output": example['expr'].strip()}


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
