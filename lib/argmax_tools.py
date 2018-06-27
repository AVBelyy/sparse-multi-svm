import nmslib
import numpy as np
from sklearn.metrics import pairwise_distances_argmin
from abc import ABCMeta
from abc import abstractmethod


class BaseArgmax:
    __metaclass__ = ABCMeta

    @abstractmethod
    def query(self, xs):
        pass

    @abstractmethod
    def update(self, ixs, new_values):
        pass


class BruteforceArgmax(BaseArgmax):
    def __init__(self, W):
        self.W = W

    def query(self, xs):
        return pairwise_distances_argmin(xs, self.W, metric="cosine")

    def update(self, ixs, new_values):
        self.W[ixs, :] = new_values


class ANNArgmax(BaseArgmax):
    def __init__(self, method="sw-graph", is_sparse=True):
        if is_sparse:
            self.index = nmslib.init(method=method, space="negdotprod_sparse_fast",
                                     data_type=nmslib.DataType.SPARSE_VECTOR)
        else:
            self.index = nmslib.init(method=method, space="negdotprod",
                                     data_type=nmslib.DataType.DENSE_VECTOR)
        self.present = set()
        self.index.createIndex({'indexThreadQty': 4})


    def query(self, xs, num_threads=4):
        results = self.index.knnQueryBatch(xs, k=1, num_threads=num_threads)
        indices = []
        dists = []
        for ixs, ds in results:
            if not ixs or not ds:
                indices.append(0)
                dists.append(0.)
            else:
                indices.append(ixs[0])
                dists.append(ds[0])
        return indices, dists

    def update(self, ixs, new_values):
        del_strategy = 0
        ixs_set = set(ixs)
        ixs_del = list(self.present & ixs_set)
        self.index.deleteBatch(ixs_del, del_strategy)
        self.present |= ixs_set
        self.index.addBatch(new_values, ixs)
