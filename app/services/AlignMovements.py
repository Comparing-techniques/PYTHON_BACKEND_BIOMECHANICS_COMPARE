import numpy as np
from ..utils.preprocessing_helpers import DTW

class AlignMovementsService:
    @staticmethod
    def align_segments(seg_a: np.ndarray, seg_b: np.ndarray):
        """
        Alinea dos matrices de cuaterniones (o vectores) seg_a, seg_b.
        Devuelve seg_a_aligned, seg_b_aligned, distancia.
        """
        dist, path = DTW.align(seg_a, seg_b)
        a_idx, b_idx = zip(*path)
        return seg_a[np.array(a_idx)], seg_b[np.array(b_idx)], dist