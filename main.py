import numpy as np
from centroid import Centroid
from classification import classify, learn_classifier
from clusterer_optimizer import opt_cluster
from clusterers import AggClusterer
from paragraph_label import ParagraphLabel
from plotting import plot
from pre_reduction import load_and_pre_reduce_embs
from save_labels import SaveData

def save_clu_result(lab: list[ParagraphLabel], centroids: list[Centroid]):
    sv = SaveData()
    sv.save(lab, 'labels_clu.json')
    sv.save(centroids, 'centroids.json')

def save_classf_result(lab: list[ParagraphLabel]):
    sv = SaveData()
    sv.save(lab, 'labels_classf.json')

if __name__ == '__main__':
    print('Loading and compressing Data for clustering...')
    ipca_clu, nemd_clu, dim_clu = load_and_pre_reduce_embs('embeddings_clu')
    print(f'Data compressed to {dim_clu} dimensions')

    print('Clustering...')
    ns_clu = list(range(2, 24))
    opt_put = opt_cluster(
        cltype=AggClusterer, 
        emb=nemd_clu,
        dist='cosine',
        metric='db',
        n_clu=ns_clu
    )

    if opt_put is not None:
        clu_labels = opt_put['labels']
        centroids = opt_put['centroids']

        print('Saving clustered data...')
        save_clu_result(clu_labels, centroids)

        print('Learning classifier on centroid cosine distances...')
        classifier, score = learn_classifier(clu_labels, centroids)
        print(f'Classifier learned with weighted F1 score {score:.2f}')

        print('Loading and compressing Data for classification...')
        _, nemd_classf, _ = load_and_pre_reduce_embs('embeddings_classf', ipca=ipca_clu)
        print('Data for classfication compressed to same dimensionality as data for clustering')

        print('Classifying data...')
        classf_labels = classify(classifier, nemd_classf, centroids)

        print('Saving classified data...')
        save_classf_result(classf_labels)

        print('Done!')
        plot(clu_labels)

    else:
        print('Failed to cluster data')