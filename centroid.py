import numpy as np

class Centroid:
    def __init__(self, label: int, vec: list[float]):
        self.__label = label
        self.__vec = vec

    @property
    def label(self) -> int:
        return self.__label
    
    @property
    def vec(self) -> list[float]:
        return self.__vec
    
    @property
    def vec_np(self) -> np.ndarray:
        return np.array(self.__vec)