import json
from argparse import ArgumentParser
from data import get_dataset
from metric.metric import Metric


parser = ArgumentParser()
parser.add_argument('--dataset_path', default='conic10k', type=str)
parser.add_argument('--prediction_file', type=str)
parser.add_argument('--split', default='test', type=str)
parser.add_argument('--report_file', default='', type=str)

if __name__ == '__main__':
    args = parser.parse_args()

    task = args.prediction_file
    split = args.split
    report_file = args.report_file

    datas = get_dataset('/home/zcl/Conic10K/Conic10k/conic10k', encoder_decoder=True)[split]
    refs = [
        d['labels']
        for d in datas
    ]

    ques = [
        d['text']
        for d in datas
    ]

    f = open(args.prediction_file,'r',encoding='utf-8')
    result = f.readlines()

    preds = []
    for p in result:
        p = p.replace('"','')
        # p = p.split('\\nA:')[1].strip()
        # p = p.split('\\nQ:')[0].strip()
        p = p.replace('<|endoftext|>','')
        p = p.replace('<|im_end|>','')
        preds.append(p)
        print(p)

    mtc = Metric(max_workers=1)
    mtc.cmps(preds, refs, questions=ques, verbose=True)

    if report_file:
        with open(report_file, 'w') as f:
            f.write(mtc.detail())
            f.write('\n')
            f.write(f'accuracy: {mtc.accuracy}\nmi-f1: {mtc.f1}\nma-f1: {mtc.avg_f1}')
            f.close()

    print(f'accuracy: {mtc.accuracy}\nmi-f1: {mtc.f1}\nma-f1: {mtc.avg_f1}')
    
