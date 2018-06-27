# Binary and multiclass SGD trainer for SVM a.k.a. Pegasos a.k.a. the baseline algorithm.

import os, pickle
import collections
import numpy as np
import scipy.sparse as ss

from lib.sparse_tools import dense_sparse_dot, dense_sparse_add, sparse_sparse_dot
from lib.argmax_tools import ANNArgmax
from tqdm import tqdm

# Available datasets.
datasets_names = ("LSHTC1", "DMOZ", "WIKI_Small", "WIKI_50K", "WIKI_100K")
dataset_dir = "../data"
out_dir = "../data/parsed"

# Read the dataset.

# dataset_name = "WIKI_100K"
dataset_name = "LSHTC1"

with open(os.path.join(out_dir, "%s_train.dump" % dataset_name), "rb") as fin:
    X_train = pickle.load(fin)
with open(os.path.join(out_dir, "%s_train_out.dump" % dataset_name), "rb") as fin:
    y_train = pickle.load(fin)
with open(os.path.join(out_dir, "%s_test.dump" % dataset_name), "rb") as fin:
    X_test = pickle.load(fin)
with open(os.path.join(out_dir, "%s_test_out.dump" % dataset_name), "rb") as fin:
    y_test = pickle.load(fin)

n_classes = 0
for dataset_part in ("train", "heldout", "test"):
    with open(os.path.join(out_dir, "%s_%s_out.dump" % (dataset_name, dataset_part)), "rb") as fin:
        labels = pickle.load(fin)
        n_classes = max(n_classes, max(labels) + 1)

# X_train, X_test = normalize(X_train, norm="l1"), normalize(X_test, norm="l1")

X_train = ss.hstack([X_train, np.ones(X_train.shape[0]).reshape(-1, 1)])
X_test = ss.hstack([X_test, np.ones(X_test.shape[0]).reshape(-1, 1)])
X_train, X_test = ss.csr_matrix(X_train), ss.csr_matrix(X_test)

classes_objects = collections.defaultdict(list)
for i, y in enumerate(y_train):
    classes_objects[y].append(i)

# Load Iris datasets
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
#
# iris_data = load_iris()
# X, y = iris_data["data"], iris_data["target"]
# X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
#
# # Make X sparse matrix
# X = ss.csr_matrix(X)
#
# # Split train and test
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

"""
Vector in a form: a * v
"""


class WeightVector:
    # TODO: add squared norm

    def __init__(self, dimension):
        self.d = dimension
        self.a = 1.0
        self.v = np.zeros(self.d, dtype=np.float64)

    def sparse_dot(self, u: ss.csr_matrix):
        return dense_sparse_dot(self.v, u) * self.a

    def sparse_add(self, u: ss.csr_matrix, s: float):
        dense_sparse_add(self.v, u * (s / self.a), inplace=True)

    def add(self, other, s: float):
        self.v *= self.a
        self.v += other.v * other.a * s
        self.a = 1.0

    def scale(self, s: float):
        if abs(s) < 1e-9:
            self.__init__(self.d)
        else:
            self.a *= s

    def get_snorm(self):
        return (self.a ** 2) * np.dot(self.v, self.v)


"""
Matrix in a form: a * [v_i], i=1...n
"""


class WeightMatrix:
    # TODO: add squared norm

    def __init__(self, dim):
        self.dim = n, d = dim
        self.a = 1.0
        self.m = [ss.csr_matrix((1, d), dtype=np.float32) for _ in range(n)]

    def sparse_dot(self, ix: int, v: ss.csr_matrix):
        return sparse_sparse_dot(self.m[ix], v) * self.a

    def sparse_add(self, ix: int, v: ss.csr_matrix, s: float):
        self.m[ix] += v * (s / self.a)
        return self.m[ix] * self.a

    def scale(self, s: float):
        if abs(s) < 1e-32:
            self.__init__(self.dim)
        else:
            self.a *= s


def stochastic_pegasos(X: np.array, y: np.array, pos_class: int, random_seed=None) -> np.ndarray:
    n, d = X.shape

    labels = ((y == pos_class) * 2 - 1)

    # TODO: make parameters
    max_iter = 800
    num_to_avg = 400
    lambd = 0.1
    k = 1

    if random_seed is not None:
        np.random.seed(random_seed)
    random_ids = np.random.choice(n, size=max_iter * k)

    avg_scale = min(max_iter, num_to_avg)
    avg_wv = WeightVector(d)
    wv = WeightVector(d)
    wvs = []

    for i in tqdm(range(max_iter)):
        x_ids = random_ids[i * k: (i + 1) * k]
        eta = 1. / (lambd * (i + 2))
        grad_ixs, grad_weights = [], []
        for j in x_ids:
            x = X.getrow(j)
            pred = wv.sparse_dot(x)
            label = labels[j]
            if label * pred < 1:
                grad_ixs.append(j)
                grad_weights.append(eta * label / k)
        # Scale wv
        wv.scale(1. - eta * lambd)
        # Add sub-gradients
        for grad_ix, grad_w in zip(grad_ixs, grad_weights):
            wv.sparse_add(X.getrow(grad_ix), grad_w)
        # Projection step
        wv.scale(min(1., 1. / np.sqrt(lambd * wv.get_snorm())))
        # Average weights
        if i >= max_iter - num_to_avg:
            avg_wv.add(wv, 1. / avg_scale)
            if (i + 1) % 1 == 0:
                wvs.append(avg_wv.a * avg_wv.v)
        else:
            if (i + 1) % 1 == 0:
                wvs.append(wv.a * wv.v)

    return avg_wv.a * avg_wv.v


def multi_pegasos(X: np.array, y: np.array, random_seed=None) -> WeightMatrix:
    n, d = X.shape

    # TODO: make parameters
    max_iter = 6000
    lambd = 0.0005
    k = 100

    Wyx = np.zeros(n, dtype=np.float32)

    W = WeightMatrix((n_classes, d))
    # amax = BruteforceArgmax(W)
    amax = ANNArgmax()

    if random_seed is not None:
        np.random.seed(random_seed)
    random_ids = np.random.choice(n, size=max_iter * k)

    # avg_scale = min(max_iter, num_to_avg)
    # avg_wv = WeightVector(d)

    for i in tqdm(range(max_iter)):
        x_ids = random_ids[i * k: (i + 1) * k]
        xs = X[x_ids]
        eta = 1. / (lambd * (i + 2))
        ys = y[x_ids]
        rs, dists = amax.query(xs)
        grad_ixs, grad_weights = [], []
        for j_, y_, r_, dr, x_ in zip(x_ids, ys, rs, dists, xs):
            loss = max(0, 1 + (-dr) - Wyx[j_])
            if loss > 0:
                grad_ixs.append((y_, j_))
                grad_weights.append(+eta / k)
                grad_ixs.append((r_, j_))
                grad_weights.append(-eta / k)
        # Scale weight matrix
        W.scale(1. - eta * lambd)
        # Add sub-gradients and project rows onto a sphere of r=1
        update = {}
        for (class_ix, obj_ix), grad_w in zip(grad_ixs, grad_weights):
            obj = X.getrow(obj_ix)
            upd = W.sparse_add(class_ix, obj, grad_w)
            # Incrementally update <w_yk, xk> structure
            for x_ix in classes_objects[class_ix]:
                Wyx[x_ix] += sparse_sparse_dot(X.getrow(x_ix), obj) * grad_w
            # TODO: UNNEEDED?
            # upd.data *= min(1., 1. / np.sqrt(lambd * np.dot(upd.data, upd.data)))
            if upd.nnz > 0:
                update[class_ix] = upd
                # TODO: we should delete zero vectors
        if len(update) > 0:
            class_ixs = list(update.keys())
            new_values = ss.vstack(list(update.values()))
            amax.update(class_ixs, new_values)
        if i % 250 == 0:
            print("iter #%6i, W sparsity: %.9f" % (i, sum([x.nnz for x in W.m]) / (len(W.m) * W.m[0].shape[1])))

    return W


if __name__ == "__main__":
    # Train
    W = multi_pegasos(X_train, y_train, random_seed=0)
    with open("W.dump", "wb") as fout:
        pickle.dump(W, fout)
    # clf = LogisticRegression(C=100.0, fit_intercept=False)
    # clf.fit(X_train, (y_train == pos_class))
    # wv_lr = clf.coef_.reshape(-1, 1)
    # # Predict
    # y_true = (y_test == pos_class)
    # y_pred_pegasos = (X_test.dot(wv_pegasos) > 0).T[0]
    # y_pred_lr = (X_test.dot(wv_lr) > 0).T[0]
    # print(accuracy_score(y_true, y_pred_pegasos))
    # print(accuracy_score(y_true, y_pred_lr))
    pass
