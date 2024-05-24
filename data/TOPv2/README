Task Oriented Parsing v2 (TOPv2) representations for intent-slot based dialog
systems.

Provided under the CC-BY-SA license. Please cite the accompanying paper when
using this dataset -

@inproceedings{chen-etal-2020-low-resource,
    title={Low-Resource Domain Adaptation for Compositional Task-Oriented
        Semantic Parsing},
    author={Xilun Chen and Asish Ghoshal and Yashar Mehdad and Luke Zettlemoyer
        and Sonal Gupta},
    booktitle={Proceedings of the 2020 Conference on Empirical Methods in
        Natural Language Processing (EMNLP)},
    year={2020},
    publisher = "Association for Computational Linguistics"
}


CHANGELOG:
03/10/2021 (V1.1): Added the low-resource splits used in the paper.
09/18/2020 (V1.0): Initial release.


TOPv2 is a multi-domain task-oriented semantic parsing dataset. It is an extension to the TOP dataset (http://fb.me/semanticparsingdialog) with 6 additional domains and 137k new samples.

In total, TOPv2 has 8 domains (alarm, event, messaging, music, navigation, reminder, timer, weather) and 180k samples randomly split into train, eval, and test sets for each domain. Please refer to the paper for more data statistics.
Note: As TOPv2 data is provided on a per-domain basis, the UNSUPPORTED utterances in the original TOP dataset were removed as they could not be mapped to any domain.

The training, evaluation and test sets for each domain are provided as tab-separated value (TSV) files with file names of "domain_split.tsv".
The first row of each file contains the column headers, while each following row is of the format:
domain <tab> utterance <tab> semantic_parse
where the semantic_parse follows the same format as the original TOP dataset.

e.g. event <tab> Art fairs this weekend in Detroit <tab> [IN:GET_EVENT [SL:CATEGORY_EVENT Art fairs ] [SL:DATE_TIME this weekend ] in [SL:LOCATION Detroit ] ]


The low-resource splits used in our experiments are provided in the
`low_resource_splits` subdirectory, including training and validation sets from the reminder and weather domains under 10, 25, 50, 100, 250, 500 and 1000 SPIS.
