import numpy as np
from scipy import sparse as ss
from lib.sparse_tools import dense_sparse_dot

class SimpleLSH:
    def __init__(self, n_features: int, hash_length : int = 256):
        self.__k = hash_length
        self.__n_features = n_features
        self.__a = np.random.randn(hash_length, n_features + 1)

    def transform(self, x: ss.csr_matrix):
        res = []
        last_elem = np.sqrt(max(0., 1. - np.dot(x.data, x.data)))
        for i in range(self.__k):
            apx = dense_sparse_dot(self.__a[i], x)
            apx += last_elem * self.__a[i][-1]
            res.append(0 if np.sign(apx) <= 0 else 1) # TODO: treat zero sign correctly
        res = np.array(res, dtype=np.bool)
        return np.packbits(np.pad(res, (0, len(res)%32), mode='constant', constant_values=0)
                           .reshape(-1, 8)).view(np.uint32)