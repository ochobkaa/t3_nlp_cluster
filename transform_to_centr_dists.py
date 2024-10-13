import numpy as np
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from centroid import Centroid
from embeddings import Embeddings
from paragraph_label import ParagraphLabel
from clusterers import DistLiteral

def _dist_arr_transform(vec_arr: np.ndarray, centrs: list[Centroid], 
                        dist: DistLiteral) -> list[np.ndarray]:
    centrs_arr_l = list(map(lambda c: c.vec_np, centrs))
    centrs_arr = np.stack(centrs_arr_l, axis=0)

    if dist == 'cosine':
        nvec_arr = cosine_distances(vec_arr, centrs_arr)

    elif dist == 'euclidian':
        nvec_arr = euclidean_distances(vec_arr, centrs_arr)

    nvec_arr_l = np.unstack(nvec_arr, axis=0)
    return nvec_arr_l

def emb_to_centr_dists(lab: list[Embeddings], centrs: list[Centroid], 
                             dist: DistLiteral) -> list[Embeddings]:
    t_l = list(map(lambda l: l.title, lab))
    emb_arr_l = list(map(lambda l: l.emb_np, lab))

    emb_arr = np.stack(emb_arr_l, axis=0)
    nemb_arr_l = _dist_arr_transform(emb_arr, centrs, dist)

    embs_t = list(map(
        lambda e: Embeddings(e[0], e[1].tolist()), 
        zip(t_l, nemb_arr_l)
    ))
    return embs_t

def lab_to_centr_dists(lab: list[ParagraphLabel], centrs: list[Centroid], 
                             dist: DistLiteral) -> list[ParagraphLabel]:
    t_l = list(map(lambda l: l.title, lab))
    lab_l = list(map(lambda l: l.label, lab))
    vec_arr_l = list(map(lambda l: l.vec_np, lab))

    vec_arr = np.stack(vec_arr_l, axis=0)
    nvec_arr_l = _dist_arr_transform(vec_arr, centrs, dist)

    labs_t = list(map(
        lambda e: ParagraphLabel(e[0], e[1], e[2].tolist()), 
        zip(t_l, lab_l, nvec_arr_l)
    ))
    return labs_t