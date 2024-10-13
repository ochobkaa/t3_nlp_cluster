from abc import ABC, abstractmethod
from typing import Iterable
import numpy as np
from sklearn.decomposition import SparsePCA, IncrementalPCA
from sklearn.manifold import TSNE
from sklearn.base import TransformerMixin, BaseEstimator
from embeddings import Embeddings

class Transformer(ABC, TransformerMixin, BaseEstimator):
    pass

class DimReduce(ABC):
    def __init__(self):
        self.__trained = False

    @property
    @abstractmethod
    def _reducer(self) -> Transformer:
        pass

    def _set_trained(self):
        self.__trained = True

    def reduce(self, emb: list[Embeddings]) -> list[Embeddings]:
        reducer = self._reducer

        emb_arr_l = list(map(lambda e: e.emb_np, emb))

        emb_arr = np.stack(emb_arr_l, axis=0)

        if not self.trained:
            rd_arr = reducer.fit_transform(emb_arr)
            self._set_trained()

        else:
            rd_arr = reducer.transform(emb_arr)

        rd_arr_l = np.unstack(rd_arr, axis=0)
        new_embs = list(map(
            lambda e: Embeddings(e[0].title, e[1]), 
            zip(emb, rd_arr_l)
        ))
        return new_embs
    
    @property
    def trained(self):
        return self.__trained
    
class IpcaDimReduce(DimReduce):
    def __init__(self, dim: int):
        super().__init__()
        self.__dim = dim
        self.__reducer = IncrementalPCA(n_components=dim)

    @property
    def _reducer(self):
        return self.__reducer
    
    @property
    def explained_variance(self):
        return self.__reducer.explained_variance_ratio_
    
    @property
    def dim(self):
        return self.__dim
    
    def partial_reduce(self, emb_bs: Iterable[list[Embeddings]]) -> list[Embeddings]:
        dim = self.__dim
        reducer = self._reducer

        emb_ts = []
        emb_arr_bs = []
        for i, emb_b in enumerate(emb_bs):
            for emb in emb_b:
                t = emb.title
                emb_ts.append(t)

            emb_arr_l = list(map(lambda e: e.emb_np, emb_b))

            emb_arr = np.stack(emb_arr_l, axis=0)
            if emb_arr.shape[0] < dim and i > 0:
                prev_b = emb_arr_bs.pop()
                emb_arr = np.concat([prev_b, emb_arr], axis=0)

            emb_arr_bs.append(emb_arr)

        if not self.trained:
            for emb_arr_b in emb_arr_bs:
                reducer.partial_fit(emb_arr_b)

            self._set_trained()

        rd_arr_bs = list(map(
            lambda b: reducer.transform(b),
            emb_arr_bs
        ))

        if len(rd_arr_bs) > 1:
            rd_arr = np.concat(rd_arr_bs, axis=0)

        else:
            rd_arr = rd_arr_bs[0]

        rd_arr_l = np.unstack(rd_arr, axis=0)
        new_embs = list(map(
            lambda e: Embeddings(e[0], e[1].tolist()), 
            zip(emb_ts, rd_arr_l)
        ))
        return new_embs
    
class SpcaDimReduce(DimReduce):
    def __init__(self, dim: int, alpha=1.0, ridge=0.01, n_jobs=-1):
        super().__init__()
        self.__reducer = SparsePCA(n_components=dim, alpha=alpha, ridge_alpha=ridge, n_jobs=n_jobs)

    @property
    def _reducer(self):
        return self.__reducer
    
class TsneDimReduce(DimReduce):
    def __init__(self, perp=30):
        super().__init__()
        self.__reducer = TSNE(perplexity=perp)

    @property
    def _reducer(self):
        return self.__reducer
    
    @property
    def kl_divergence(self):
        return self.__reducer.kl_divergence_