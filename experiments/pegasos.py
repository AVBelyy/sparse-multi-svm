# Binary and multiclass SGD trainer for SVM a.k.a. Pegasos a.k.a. the baseline algorithm.

import numpy as np
import scipy.sparse as ss
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from lib.sparse_tools import dense_sparse_dot, dense_sparse_add
from typing import List
from tqdm import tqdm

# Available datasets.
datasets_names = ("LSHTC1", "DMOZ", "WIKI_Small", "WIKI_50K", "WIKI_100K")
dataset_dir = "../data"
out_dir = "../data/parsed"

# Read the dataset.

# dataset_name = "WIKI_100K"
dataset_name = "LSHTC1"

# with open(os.path.join(out_dir, "%s_train.dump" % dataset_name), "rb") as fin:
#     X_train = pickle.load(fin)
# with open(os.path.join(out_dir, "%s_train_out.dump" % dataset_name), "rb") as fin:
#     y_train = pickle.load(fin)
# with open(os.path.join(out_dir, "%s_test.dump" % dataset_name), "rb") as fin:
#     X_test = pickle.load(fin)
# with open(os.path.join(out_dir, "%s_test_out.dump" % dataset_name), "rb") as fin:
#     y_test = pickle.load(fin)


# Load Iris datasets
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris_data = load_iris()
X, y = iris_data["data"], iris_data["target"]
X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])

# Make X sparse matrix
X = ss.csr_matrix(X)

# Split train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

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


if __name__ == "__main__":
    # Train
    pos_class = 1
    wv = stochastic_pegasos(X_train, y_train, pos_class=pos_class, random_seed=0)
    wv_pegasos = wv.reshape(-1, 1)
    clf = LogisticRegression(C=100.0, fit_intercept=False)
    clf.fit(X_train, (y_train == pos_class))
    wv_lr = clf.coef_.reshape(-1, 1)
    # Predict
    y_true = (y_test == pos_class)
    y_pred_pegasos = (X_test.dot(wv_pegasos) > 0).T[0]
    y_pred_lr = (X_test.dot(wv_lr) > 0).T[0]
    print(accuracy_score(y_true, y_pred_pegasos))
    print(accuracy_score(y_true, y_pred_lr))
    pass