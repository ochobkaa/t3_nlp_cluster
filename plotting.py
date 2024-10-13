import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from dim_reducers import TsneDimReduce
from embeddings import Embeddings
from paragraph_label import ParagraphLabel

def reduce_to_2d(lab: list[ParagraphLabel], perp=35) -> tuple[list[ParagraphLabel], float]:
    tsne = TsneDimReduce(perp)

    emb_2d = tsne.reduce_lab(lab)
    kl_div = tsne.kl_divergence

    return emb_2d, kl_div

def scatter_data(lab_2d: list[ParagraphLabel], ax):
    lab_2d_vl = list(map(lambda e: e.vec, lab_2d))
    lab_2d_x = list(map(lambda e: e[0], lab_2d_vl))
    lab_2d_y = list(map(lambda e: e[1], lab_2d_vl))

    lab_2d_l = list(map(lambda l: l.label, lab_2d))

    lab_2d_df = pd.DataFrame({'X': lab_2d_x, 'Y': lab_2d_y, 'Label': lab_2d_l})

    sns.scatterplot(lab_2d_df, x='X', y='Y', hue='Label', style='Label', 
                    palette='brg', alpha=0.5, ax=ax, legend=False)

def plot_labels(labels: list[ParagraphLabel], ax):
    labels_l = list(map(lambda l: str(l.label), labels))
    labels_df = pd.DataFrame({'Labels': labels_l})

    sns.histplot(labels_df, x='Labels', hue='Labels', binwidth=1, ax=ax, legend=False)

def plot(lab: list[ParagraphLabel]):
    fig, ax = plt.subplots(nrows=1, ncols=2)
    fig.set_figwidth(18)
    fig.set_figheight(8)

    lab_2d, kl_div = reduce_to_2d(lab)

    fig.suptitle(f'Визуализация кластеров после кластеризации (KL divergence для t-SNE {kl_div:.2f})')

    plot_labels(lab_2d, ax[0])
    scatter_data(lab_2d, ax[1])
    plt.show()