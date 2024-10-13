from typing import Literal
from itertools import product
import numpy as np
from clusterers import Clusterer, DistLiteral
from embeddings import Embeddings

MetricLiteral = Literal['sil', 'cal_har', 'db']

def opt_cluster(cltype: type, emb: list[Embeddings], 
                dist: DistLiteral = 'euclidian', 
                metric: MetricLiteral = 'db', 
                **kwargs) -> dict | None:
    def init_opt_score(metric: MetricLiteral) -> float:
        if metric == 'db':
            opt_score = np.inf

        else:
            opt_score = -np.inf

        return opt_score
    
    def get_score(clu: Clusterer, metric: MetricLiteral):
        if metric == 'sil':
            score = clu.sil_score

        elif metric == 'cal_har':
            score = clu.cal_har_score

        elif metric == 'db':
            score = clu.db_score

        return score

    if len(kwargs) > 1:
        combs = product(*kwargs.values())

    else:
        combs = list(kwargs.values())[0]

    opt_score = init_opt_score(metric)

    opt_clu = None
    opt_labels = []
    opt_params = {}
    for comb in combs:
        if len(kwargs) > 1:
            comb_kw = dict(zip(kwargs.keys(), comb))

        else:
            comb_kw = {list(kwargs.keys())[0]: comb}

        clu: Clusterer = cltype(**comb_kw)
        labels = clu.cluster(emb, dist)

        if labels is not None:
            score = get_score(clu, metric)

            if (score < opt_score and metric == 'db') or score > opt_score:
                opt_clu = clu
                opt_labels = labels
                opt_params = comb_kw
                opt_score = score

    if opt_clu is not None:
        centroids = opt_clu.centroids

        out = {
            'labels': opt_labels,
            'params': opt_params,
            'score': opt_score,
            'centroids': centroids
        }
        return out