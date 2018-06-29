import nmslib
import numpy as np
from abc import ABCMeta
from abc import abstractmethod

from lib.sparse_tools import sparse_sparse_dot


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
            max_dist, max_ix = self.min_float, -1
            for i, w in enumerate(self.W.m):
                dist = sparse_sparse_dot(w, x)
                if max_dist < dist and i != y:
                    max_dist = dist
                    max_ix = i
            indices.append(max_ix)
        return indices

    def update(self, ixs, new_values):
        pass


class ANNArgmax(BaseArgmax):
    def __init__(self, method="sw-graph", is_sparse=True):
        if is_sparse:
            self.index = nmslib.init(method=method, space="negdotprod_sparse_fast",
                                     data_type=nmslib.DataType.SPARSE_VECTOR)
        else:
            self.index = nmslib.init(method=method, space="negdotprod",
                                     data_type=nmslib.DataType.DENSE_VECTOR)
        self.present = set()
        # TODO: replace inadequately high values to lower ones
        self.index.createIndex({'indexThreadQty': 4})

    def query(self, xs, ys, num_threads=4):
        results = self.index.knnQueryBatch(xs, k=2, num_threads=num_threads)
        indices = []
        # dists = []
        for (ixs, ds), y in zip(results, ys):
            if len(ixs) == 0 or len(ds) == 0:
                indices.append(0)
                # dists.append(0.)
            else:
                if ixs[0] == y:
                    indices.append(ixs[1])
                    # dists.append(ds[1])
                else:
                    indices.append(ixs[0])
                    # dists.append(ds[0])
        return indices

    def update(self, ixs, new_values):
        del_strategy = 0
        ixs_set = set(ixs)
        ixs_del = list(self.present & ixs_set)
        self.index.deleteBatch(ixs_del, del_strategy)
        self.present |= ixs_set
        self.index.addBatch(new_values, ixs)
