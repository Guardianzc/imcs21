from nervaluate import Evaluator


true = [
    ['O', 'O', 'B-PER', 'I-PER', 'O'],
    ['O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'O'],
]

pred = [
    ['O', 'O', 'B-PER', 'I-PER', 'O'],
    ['O', 'B-LOC', 'I-LOC', 'B-LOC', 'I-LOC', 'O'],
]

evaluator = Evaluator(true, pred, tags=['Symptom', 'Medical_Examination', 'Drug', 'Drug_Category', 'Operation'], loader="list")

results, results_by_tag = evaluator.evaluate()

