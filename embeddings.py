import numpy as np

class Embeddings:
    def __init__(self, title: str, emb: list[float]) -> None:
        self.__title = title
        self.__emb = emb

    @property
    def title(self) -> str:
        return self.__title
    
    @property
    def emb(self) -> list[float]:
        return self.__emb
    
    @property
    def emb_np(self) -> np.ndarray[np.float64]:
        return np.array(self.__emb, dtype=np.float64)