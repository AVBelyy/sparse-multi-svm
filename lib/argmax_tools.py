import random

import nmslib
import numpy as np
import scipy.sparse as ss
from abc import ABCMeta
from abc import abstractmethod

from lib.sparse_tools import sparse_sparse_dot
from lib.lsh_tools import SimpleLSH


class BaseArgmax:
    __metaclass__ = ABCMeta

    @abstractmethod
    def query(self, xs, ys):
        pass

    @abstractmethod
    def update(self, ixs, new_values):
        pass


class BruteforceArgmax(BaseArgmax):
    def __init__(self, W):
        self.W = W
        self.min_float = np.finfo(self.W.dtype).min

    def query(self, xs, ys):
        indices = []
        for x, y in zip(xs, ys):
            max_dist, max_ixs = self.min_float, []
            for i, w in enumerate(self.W.m):
                dist = sparse_sparse_dot(w, x)
                if i != y:
                    if max_dist < dist:
                        max_dist = dist
                        max_ixs = [i]
                    elif max_dist == dist:
                        max_ixs.append(i)
            assert len(max_ixs) > 0
            max_ix = random.sample(max_ixs, 1)[0]
            indices.append(max_ix)
        return indices

    def update(self, ixs, new_values):
        pass


class RandomArgmax(BaseArgmax):
    def __init__(self, n_classes):
        self.classes = set(range(n_classes))

    def query(self, xs, ys):
        indices = []
        for x, y in zip(xs, ys):
            choice_set = self.classes - {y}
            ix = random.sample(choice_set, 1)[0]
            indices.append(ix)
        return indices

    def update(self, ixs, new_values):
        pass


class ANNArgmax(BaseArgmax):
    def __init__(self, n_classes, num_threads, method="sw-graph", is_sparse=True,
                 LSH=False, n_features=None, hash_length=256):
        if LSH:
            if n_features is None:
                raise AttributeError("n_features is not defined")
            self.lsh = SimpleLSH(n_features=n_features, hash_length=hash_length)
            self.index = nmslib.init(method=method, space="bit_hamming",
                                     data_type=nmslib.DataType.DENSE_VECTOR)
        elif is_sparse:
            self.index = nmslib.init(method=method, space="negdotprod_sparse_fast",
                                     data_type=nmslib.DataType.SPARSE_VECTOR)
        else:
            self.index = nmslib.init(method=method, space="negdotprod",
                                     data_type=nmslib.DataType.DENSE_VECTOR)
        self.num_threads = num_threads
        self.present = set()
        self.not_present = set(range(n_classes))
        self.index.createIndex({"indexThreadQty": self.num_threads})

    def take_random_zero_vector(self):
        if len(self.not_present) > 0:
            return random.sample(self.not_present, 1)[0]

    def query(self, xs, ys):
        if hasattr(self, "lsh"):
            xs = self.lsh.transform(xs)

        results = self.index.knnQueryBatch(xs, k=2, num_threads=self.num_threads)
        indices = []
        # dists = []
        for (ixs, ds), y_ in zip(results, ys):
            if len(ixs) == 0 or len(ds) == 0:
                ix = self.take_random_zero_vector()
                assert ix is not None
                dist = 0.
            else:
                if ixs[0] == y_:
                    ix, dist = ixs[1], -ds[1]
                else:
                    ix, dist = ixs[0], -ds[0]
                if dist <= 0:
                    zero_ix = self.take_random_zero_vector()
                    if zero_ix is not None and zero_ix != y_: # TODO: if zero_ix == y, sample other ix if possible
                        ix = zero_ix
                        dist = 0.
            indices.append(ix)
            # dists.append(dist)
        return indices

    def update(self, ixs: np.ndarray, new_values: ss.csr_matrix):
        # print("to del: ", ixs_del)
        if hasattr(self, "lsh"):
            new_values = self.lsh.transform(new_values)

        ixs_set = set(ixs)
        ixs_del = list(self.present & ixs_set)
        del_strategy = 0
        self.index.deleteBatch(ixs_del, del_strategy)
        self.not_present -= ixs_set
        # print("to add: ", ixs, " cur: ", self.present)
        ixs_nz, ixs_z = [], []
        for ix, v in enumerate(new_values):
            if v.nnz == 0:
                ixs_z.append(ix)
            else:
                ixs_nz.append(ix)
        self.index.addBatch(new_values[ixs_nz], ixs[ixs_nz])
        self.present |= set(ixs[ixs_nz])
        self.present -= set(ixs[ixs_z])
