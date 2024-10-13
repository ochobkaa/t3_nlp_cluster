from abc import ABC, abstractmethod
from embeddings import Embeddings
from paragraph_label import ParagraphLabel
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression

class Classifier(ABC):
    def learn(self, lab: list[ParagraphLabel], cv_splits=10, cv_repeats=5, n_jobs=-1):
        lab_l = list(map(lambda l: l.label, lab))
        lab_arr = np.array(lab_l)

        vec_arr_l = list(map(lambda l: l.vec_np, lab))
        vec_arr = np.stack(vec_arr_l, axis=0)

        self._model.fit(vec_arr, lab_arr)

        cv = RepeatedKFold(n_splits=cv_splits, n_repeats=cv_repeats)
        f1_scores = cross_val_score(
            estimator=self._model, 
            X=vec_arr, 
            y=lab_arr,
            cv=cv,
            n_jobs=n_jobs,
            scoring='f1_weighted'
        )
        
        self.__weighted_f1 = f1_scores.mean()

    def predict(self, lab: list[Embeddings]) -> list[ParagraphLabel]:
        t_l = list(map(lambda l: l.title, lab))
        emb_arr_l = list(map(lambda l: l.emb_np, lab))

        emb_arr = np.stack(emb_arr_l, axis=0)
        lab_arr = self._model.predict(emb_arr)

        lab_l = np.unstack(lab_arr, axis=0)

        labs_t = list(map(
            lambda e: ParagraphLabel(e[0], int(e[1]), e[2].tolist()), 
            zip(t_l, lab_l, emb_arr_l)
        ))
        return labs_t

    @property
    @abstractmethod
    def _model(self) -> ClassifierMixin:
        pass

    @property
    def weighted_f1(self) -> float:
        return self.__weighted_f1

class LogRegClassifier(Classifier):
    def __init__(self):
        self.__model = LogisticRegression(max_iter=500)

    @property
    def _model(self):
        return self.__model