from typing import Literal
import numpy as np
from sklearn.base import ClusterMixin
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics.pairwise import cosine_similarity
from centroid import Centroid
from embeddings import Embeddings
from paragraph_label import ParagraphLabel
from abc import ABC, abstractmethod

DistLiteral = Literal['euclidian', 'cosine']

class Clusterer(ABC):
    def cluster(self, emb: list[Embeddings], dist: DistLiteral = 'euclidian') -> list[ParagraphLabel] | None:
        clust = self._clust

        emb_t_l = list(map(lambda e: e.title, emb))
        emb_arr_l = list(map(lambda e: e.emb_np, emb))

        emb_arr = np.stack(emb_arr_l, axis=0)

        if dist == 'cosine':
            emb_arr_d = 1 - cosine_similarity(emb_arr)

        else:
            emb_arr_d = emb_arr

        labels_arr = clust.fit_predict(emb_arr_d)

        labels_l = labels_arr.tolist()
        if len(set(labels_l) - {-1}) > 1:
            self.__sil_score = silhouette_score(emb_arr, labels_arr)
            self.__cal_har_score = calinski_harabasz_score(emb_arr, labels_arr)
            self.__db_score = davies_bouldin_score(emb_arr, labels_arr)

            self.__centroids = self._calc_centroids(labels_l, emb_arr_l)

        else:
            return

        labels = list(map(
            lambda l: ParagraphLabel(l[0], l[1], l[2].tolist()),
            zip(emb_t_l, labels_l, emb_arr_l)
        ))
        labels_f = list(filter(lambda l: l.label != -1, labels))

        return labels_f
    
    def _calc_centroids(self, lab: list[int], emb_arr_l: list[np.ndarray]) -> list[Centroid]:
        lab_vecs: dict[int, list[np.ndarray]] = {}
        for l in set(lab):
            lab_vecs[l] = []

        for l, vec in zip(lab, emb_arr_l):
            lab_vecs[l].append(vec)

        centroids: list[Centroid] = []
        for l, vecs_l in lab_vecs.items():
            vecs_arr = np.stack(vecs_l, axis=0)
            lab_centr_l = vecs_arr.mean(axis=0).tolist()
            
            lab_centr = Centroid(l, lab_centr_l)
            centroids.append(lab_centr)

        return centroids
    
    @property
    @abstractmethod
    def _clust(self) -> ClusterMixin:
        pass

    @property
    def centroids(self) -> list[Centroid]:
        return self.__centroids

    @property
    def sil_score(self) -> float:
        return self.__sil_score
    
    @property
    def cal_har_score(self) -> float:
        return self.__cal_har_score
    
    @property
    def db_score(self) -> float:
        return self.__db_score
    
class DbscanClusterer(Clusterer):
    def __init__(self, eps=0.5, n_jobs=-1) -> None:
        self.__clust = DBSCAN(eps=eps, n_jobs=n_jobs)

    @property
    def _clust(self):
        return self.__clust
    
class AggClusterer(Clusterer):
    def __init__(self, n_clu=2, linkage='ward') -> None:
        self.__clust = AgglomerativeClustering(n_clusters=n_clu, linkage=linkage)

    @property
    def _clust(self):
        return self.__clust
    
class KmeansClusterer(Clusterer):
    def __init__(self, n_clu=2) -> None:
        self.__clust = KMeans(n_clusters=n_clu)

    @property
    def _clust(self):
        return self.__clust
    
class GaussMixClusterer(Clusterer):
    def __init__(self, n_clu=2, tol=0.001, reg_covar=1e-5) -> None:
        self.__clust = GaussianMixture(n_components=n_clu, tol=tol, reg_covar=reg_covar)

    @property
    def _clust(self):
        return self.__clust