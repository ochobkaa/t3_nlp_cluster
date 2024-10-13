import numpy as np
from embeddings import Embeddings

class Normalizer:
    def __init__(self, eps=1e-3):
        self.__eps = eps

    def norm(self, emb: list[Embeddings]) -> list[Embeddings]:
        eps = self.__eps

        emb_t_l = list(map(lambda e: e.title, emb))
        emb_arr_l = list(map(lambda e: e.emb_np, emb))

        emb_arr = np.stack(emb_arr_l, axis=0)

        emb_means = emb_arr.mean(axis=0, keepdims=True)
        emb_stds = emb_arr.std(axis=0, keepdims=True)

        nemb_arr = (emb_arr - emb_means) / (emb_stds + eps)
        nemb_arr_l = np.unstack(nemb_arr, axis=0)

        nemb = list(map(
            lambda e: Embeddings(e[0], e[1].tolist()),
            zip(emb_t_l, nemb_arr_l)
        ))
        return nemb