import numpy as np

class ParagraphLabel:
    def __init__(self, title: str, label: int, emb: list[float]):
        self.__title = title
        self.__label = label
        self.__emb = emb

    @property
    def title(self) -> str:
        return self.__title
    
    @property
    def label(self) -> int:
        return self.__label
    
    @property
    def vec(self) -> list[float]:
        return self.__emb

    @property
    def vec_np(self) -> np.ndarray:
        return np.array(self.__emb)