import numpy as np

class SimpleLSH:
    def __init__(self, n_features: int, hash_length: int=256):
        self._k = hash_length
        self._n_features = n_features
        self._a = np.random.randn(hash_length, n_features + 1)

    def transform(self, x: np.ndarray):
        last_elem = np.sqrt(max(0., 1. - np.dot(x, x)))
        x = np.append(x, last_elem).reshape(-1, 1)
        res = np.int32((np.dot(self._a, x) > 0).flatten())
        return " ".join([str(r) for r in res])
