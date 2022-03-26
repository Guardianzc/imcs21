import json
import sys
from seqeval.metrics import f1_score
from nervaluate import Evaluator


def load_json(path: str):
    """读取json文件"""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def ner_eval(gold_data, pred_data):
    """评估F1值"""
    golds = []
    preds = []
    eids = list(gold_data.keys())
    for eid in eids:
        gold_dialogue = gold_data[eid]["dialogue"]
        for sent in gold_dialogue:
            sid = sent['sentence_id']
            gold_bio = sent['BIO_label'].split(' ')
            pred_bio = pred_data[eid][sid].split(' ')
            assert len(gold_bio) == len(pred_bio)
            golds.append(gold_bio)
            preds.append(pred_bio)
    assert len(golds) == len(preds)
    f1 = f1_score(golds, preds)
    evaluator = Evaluator(golds, preds, tags=['Symptom', 'Medical_Examination', 'Drug', 'Drug_Category', 'Operation'],
                          loader="list")
    results, results_by_tag = evaluator.evaluate()

    m1 = {
        'type': results['ent_type']['f1'],
        'partial': results['partial']['f1'],
        'strict': results['strict']['f1'],
        'exact': results['exact']['f1'],
    }
    m2 = {
        'symptom': results_by_tag['Symptom']['exact']['f1'],
        'examination': results_by_tag['Medical_Examination']['exact']['f1'],
        'drug': results_by_tag['Drug']['exact']['f1'],
        'drug_category': results_by_tag['Drug_Category']['exact']['f1'],
        'operation': results_by_tag['Operation']['exact']['f1'],
    }
    print('Test F1 score {}%'.format(round(f1 * 100, 4)))
    print(m1)
    print(m2)


if __name__ == "__main__":
    grounds = load_json(sys.argv[1])
    predictions = load_json(sys.argv[2])
    ner_eval(grounds, predictions)
