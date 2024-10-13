import os
import os.path as pt
import json
from centroid import Centroid
from paragraph_label import ParagraphLabel

class SaveData:
    def __init__(self, ldir = ''):
        self.__ldir = ldir

    def __labels_to_dicts(self, lab: list[ParagraphLabel]) -> list[dict]:
        lab_ds = list(map(
            lambda l: {
                'title': l.title,
                'vec': l.vec,
                'label': l.label
            },
            lab
        ))
        return lab_ds
    
    def __centroids_to_dicts(self, centroids: list[Centroid]) -> list[dict]:
        cnt_ds = list(map(
            lambda l: {
                'vec': l.vec,
                'label': l.label
            },
            centroids
        ))
        return cnt_ds

    def save(self, data: list[ParagraphLabel] | list[Centroid], fn: str):
        if len(data) == 0:
            return

        ldir = self.__ldir

        if ldir and not pt.exists(ldir):
            os.makedirs(ldir)

        if type(data[0]) == ParagraphLabel:
            data_ds = self.__labels_to_dicts(data)

        else:
            data_ds = self.__centroids_to_dicts(data)

        lpt = pt.join(ldir, fn)
        with open(lpt, mode='w', encoding='utf-8') as f_out:
            json.dump(data_ds, f_out, ensure_ascii=False, indent=4)
        