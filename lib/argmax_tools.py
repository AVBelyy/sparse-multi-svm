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
            self.index = nmslib.init(method=method, space="cosinesimil_sparse",
                                     data_type=nmslib.DataType.SPARSE_VECTOR)
        else:
            self.index = nmslib.init(method=method, space="cosinesimil",
                                     data_type=nmslib.DataType.DENSE_VECTOR)


    def query(self, xs, num_threads=4):
        results = index.knnQueryBatch(xs, k=1, num_threads=num_threads)
        return [x[0][0] for x in results]

    def update(self, ixs, new_values):
        del_strategy = 0 # TODO: tune strategy
        self.index.deleteBatch(ixs, del_strategy)
        self.index.addBatch(new_values, ixs)
