import json
import numpy as np
from time import time
from sklearn.model_selection import ParameterGrid
import search_backend

param_ranges = {
'text_k1': np.arange(0.1, 1.4, 0.25),
    'text_b': np.arange(0, 1.1, 0.25),
    'title_k1': np.arange(0.1, 1.4, 0.25),
    'title_b': np.arange(0, 1.1, 0.25),
    'title_weight': np.arange(0, 10, 5),
    'penalize_unm': np.arange(0, 1.1, 0.25)
}


with open('queries_train.json', 'rt') as f:
  queries = json.load(f)
def average_precision(true_list, predicted_list, k=40):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    precisions = []
    for i,doc_id in enumerate(predicted_list):
        if doc_id in true_set:
            prec = (len(precisions)+1) / (i+1)
            precisions.append(prec)
    if len(precisions) == 0:
        return 0.0
    return round(sum(precisions)/len(precisions),3)
def precision_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(predicted_list) == 0:
        return 0.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(predicted_list), 3)
def recall_at_k(true_list, predicted_list, k):
    true_set = frozenset(true_list)
    predicted_list = predicted_list[:k]
    if len(true_set) < 1:
        return 1.0
    return round(len([1 for doc_id in predicted_list if doc_id in true_set]) / len(true_set), 3)
def f1_at_k(true_list, predicted_list, k):
    p = precision_at_k(true_list, predicted_list, k)
    r = recall_at_k(true_list, predicted_list, k)
    if p == 0.0 or r == 0.0:
        return 0.0
    return round(2.0 / (1.0/p + 1.0/r), 3)
def results_quality(true_list, predicted_list):
    p5 = precision_at_k(true_list, predicted_list, 10)
    f1_30 = f1_at_k(true_list, predicted_list, 30)
    if p5 == 0.0 or f1_30 == 0.0:
        return 0.0
    return round(2.0 / (1.0/p5 + 1.0/f1_30), 3)


param_grid = ParameterGrid(param_ranges)

# Define parameter ranges (you already have these)
search = search_backend.Search()

results = {}
for params in param_grid:
    text_k1 = params['text_k1']
    text_b = params['text_b']
    title_k1 = params['title_k1']
    title_b = params['title_b']
    title_weight = params['title_weight']
    penalize_unm = params['penalize_unm']
    
    qs_res = []
    for q, true_wids in queries.items():
        duration, ap = None, None
        t_start = time()
        res = search.search_query(q,
                    text_k1 = text_k1,
                    text_b = text_b,
                    title_k1 = title_k1,
                    title_b = title_b,
                    title_weight = title_weight,
                    penalize_unm = penalize_unm
                )
        duration = time() - t_start
        pred_wids, _ = zip(*res)
        rq = results_quality(true_wids, pred_wids)

        qs_res.append((q, duration, rq))

    results[str(params)] = (
        sum(result for _, _, result in qs_res) / len(qs_res),
        sum(dur for _, dur, _ in qs_res) / len(qs_res),
        max(dur for _, dur, _ in qs_res)
        )
    
print(results)