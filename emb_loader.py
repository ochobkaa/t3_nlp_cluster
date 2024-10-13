import os
import os.path as pt
import json
from typing import Iterable
from embeddings import Embeddings

class EmbLoader:
    def __init__(self, emb_dir: str):
        self.__emb_dir = emb_dir

    def __to_embeddings(self, raw_emb: dict) -> Embeddings:
            title = raw_emb['title']
            emb_l = raw_emb['embeddings']

            emb = Embeddings(title, emb_l)
            return emb

    def __load_embeddings_file(self, path: str) -> list[Embeddings]:
        with open(path, encoding='utf-8') as ef:
            emb_raw_l = json.load(ef)

        embs = list(map(self.__to_embeddings, emb_raw_l))
        return embs

    def __load_embeddings_bs(self, embs_dir: str) -> Iterable[list[Embeddings]]:
        embs_fns = os.listdir(embs_dir)

        for embs_fn in embs_fns:
            embs_fp = pt.join(embs_dir, embs_fn)

            embs_b = self.__load_embeddings_file(embs_fp)
            yield embs_b

    def __iter__(self):
        emb_dir = self.__emb_dir
        return self.__load_embeddings_bs(emb_dir)