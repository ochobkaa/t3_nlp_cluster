import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dim_reducers import TsneDimReduce
from embeddings import Embeddings
from paragraph_label import ParagraphLabel

def reduce_to_2d(emb: list[Embeddings], perp=25) -> tuple[list[Embeddings], float]:
    tsne = TsneDimReduce(perp)

    emb_2d = tsne.reduce(emb)
    kl_div = tsne.kl_divergence

    return emb_2d, kl_div

def scatter_data(emb_2d: list[Embeddings], ax, lab: list[ParagraphLabel] | None = None):
    emb_2d_l = list(map(lambda e: e.emb, emb_2d))
    emb_2d_x = list(map(lambda e: e[0], emb_2d_l))
    emb_2d_y = list(map(lambda e: e[1], emb_2d_l))

    if lab is not None:
        lab_l = list(map(lambda l: l.label, lab))

    else:
        lab_l = [0 for _ in len(emb_2d)]

    emb_2d_df = pd.DataFrame({'X': emb_2d_x, 'Y': emb_2d_y, 'Label': lab_l})

    sns.scatterplot(emb_2d_df, x='X', y='Y', hue='Label', alpha=0.5, ax=ax)

def plot_labels(labels: list[ParagraphLabel], ax):
    labels_l = list(map(lambda l: str(l.label), labels))
    labels_df = pd.DataFrame({'Labels': labels_l})

    sns.histplot(labels_df, x='Labels', hue='Labels', binwidth=1, ax=ax)

def plot(emb: list[Embeddings], lab: list[ParagraphLabel]):
    fig, ax = plt.subplots(nrows=1, ncols=2)

    emb_2d, kl_div = reduce_to_2d(emb)
    plot_labels(lab, ax[0])
    scatter_data(emb_2d, ax[1], lab=lab)
    plt.show()