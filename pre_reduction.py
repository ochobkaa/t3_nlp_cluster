from dim_reducers import DimReduce, IpcaDimReduce
from emb_loader import EmbLoader
from embeddings import Embeddings
from normalizer import Normalizer

def first_redux(embs_loader: EmbLoader, dim: int) -> tuple[DimReduce, list[Embeddings], float]:
    ipca = IpcaDimReduce(dim)
    emb_out = ipca.partial_reduce(embs_loader)
    
    exp_var = ipca.explained_variance.sum()
    return ipca, emb_out, exp_var

def opt_first_redux(emb_loader: EmbLoader, dim_vals: list[int], 
                    treshold=0.95) -> tuple[DimReduce, list[Embeddings], int]:
    for dim in sorted(dim_vals):
        ipca, emb_r1, exp_var = first_redux(emb_loader, dim)

        if exp_var > treshold:
            return ipca, emb_r1, dim
        
def normalize(emb: list[Embeddings]) -> list[Embeddings]:
    nrmzr = Normalizer()

    nemb = nrmzr.norm(emb)
    return nemb

def load_and_pre_reduce_embs(path: str, ipca: IpcaDimReduce | None = None, treshold=0.95) -> tuple[DimReduce, list[Embeddings], int]:
    emb_loader = EmbLoader(path)

    if ipca is None:
        dims_r1 = [32, 48, 64, 80, 96, 112, 128, 192, 256, 384]
        ipca, emb_r1, dim_r1 = opt_first_redux(emb_loader, dims_r1, treshold)
        
    else:
        emb_r1 = ipca.partial_reduce(emb_loader)
        dim_r1 = ipca.dim

    nemd = normalize(emb_r1)
    return ipca, nemd, dim_r1