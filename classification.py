from centroid import Centroid
from classifiers import Classifier, LogRegClassifier
from embeddings import Embeddings
from paragraph_label import ParagraphLabel
from transform_to_centr_dists import emb_to_centr_dists, lab_to_centr_dists

def learn_classifier(lab: list[ParagraphLabel], centrs: list[Centroid]) -> tuple[Classifier, float]:
    lab_t = lab_to_centr_dists(lab, centrs, 'cosine')

    clsfr = LogRegClassifier()
    clsfr.learn(lab_t)

    f1 = clsfr.weighted_f1
    return clsfr, f1

def classify(clsfr: Classifier, emb: list[Embeddings], centrs: list[Centroid]) -> list[ParagraphLabel]:
    emb_t = emb_to_centr_dists(emb, centrs, 'cosine')

    lab = clsfr.predict(emb_t)
    return lab