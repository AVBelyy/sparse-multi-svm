from scipy.sparse import csr_matrix
import numpy as np

def sparse_clip(x: csr_matrix, min, max, inplace=False):
    if not inplace:
        t = x.copy()
        t.data = np.clip(x.data, min, max)
        return t
    else:
        x.data = np.clip(x.data, min, max)
        return None