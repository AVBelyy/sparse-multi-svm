import numpy as np
from scipy import sparse as ss
from lib.sparse_tools import dense_sparse_dot

class SimpleLSH:
    def __init__(self, n_features: int, hash_length: int=256):
        self._k = hash_length
        self._n_features = n_features
        self._a = np.random.randn(hash_length, n_features + 1)

    def transform(self, x: ss.csr_matrix):
        res = []
        last_elem = np.sqrt(max(0., 1. - np.dot(x.data, x.data)))
        for i in range(self._k):
            apx = dense_sparse_dot(self._a[i], x)
            apx += last_elem * self._a[i][-1]
            res.append(int(np.sign(apx) > 0))
        res = np.array(res, dtype=np.bool)
        return np.packbits(np.pad(res, (0, len(res) % 32), mode="constant", constant_values=0)
                           .reshape(-1, 8)).view(np.uint32)
