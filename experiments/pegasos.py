# Binary and multiclass SGD trainer for SVM a.k.a. Pegasos a.k.a. the baseline algorithm.

import sys
import csv
import resource
import os, pickle
import collections

import nmslib
import numpy as np
import scipy.sparse as ss
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from lib.sparse_tools import dense_sparse_dot, dense_sparse_add, sparse_sparse_dot
from lib.argmax_tools import ANNArgmax, BruteforceArgmax, RandomArgmax
from tqdm import tqdm

from typing import Tuple

# Read the dataset.
out_dir = "../data/parsed"

# dataset_name = "WIKI_100K"
dataset_name = "LSHTC1"
is_lasso = False
gamma = 0.000005
# dataset_name = "20newsgroups"
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]
if len(sys.argv) > 2:
    is_lasso = (sys.argv[2] == "lasso")
if len(sys.argv) > 3:
    gamma = float(sys.argv[3])

num_threads = 16

if is_lasso:
    dataset_filename = "%s_lasso" % dataset_name
else:
    dataset_filename = dataset_name

use_class_sampling = True
use_dummy_loss = not is_lasso

with open(os.path.join(out_dir, "%s_train.dump" % dataset_name), "rb") as fin:
    X_train = pickle.load(fin)
with open(os.path.join(out_dir, "%s_train_out.dump" % dataset_name), "rb") as fin:
    y_train = pickle.load(fin)
with open(os.path.join(out_dir, "%s_heldout.dump" % dataset_name), "rb") as fin:
    X_heldout = pickle.load(fin)
with open(os.path.join(out_dir, "%s_heldout_out.dump" % dataset_name), "rb") as fin:
    y_heldout = pickle.load(fin)
with open(os.path.join(out_dir, "%s_test.dump" % dataset_name), "rb") as fin:
    X_test = pickle.load(fin)
with open(os.path.join(out_dir, "%s_test_out.dump" % dataset_name), "rb") as fin:
    y_test = pickle.load(fin)

n_classes = 0
for dataset_part in ("train", "heldout", "test"):
    with open(os.path.join(out_dir, "%s_%s_out.dump" % (dataset_name, dataset_part)), "rb") as fin:
        labels = pickle.load(fin)
        n_classes = max(n_classes, max(labels) + 1)

tfidf = TfidfTransformer()
tfidf.fit(X_train)
X_train = tfidf.transform(X_train, copy=False)
X_heldout = tfidf.transform(X_heldout, copy=False)
X_test = tfidf.transform(X_test, copy=False)

"""
t11 = time.time()
from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2000, algorithm="arpack", random_state=0)
X_train = X_train.astype(np.float32)
svd.fit(X_train)

X_train = ss.csr_matrix(svd.transform(X_train))
X_heldout = ss.csr_matrix(svd.transform(X_heldout))
X_test = ss.csr_matrix(svd.transform(X_test))
t12 = time.time()
"""

print("I have PID", os.getpid())

X_train = ss.hstack([X_train, np.ones(X_train.shape[0]).reshape(-1, 1)])
X_heldout = ss.hstack([X_heldout, np.ones(X_heldout.shape[0]).reshape(-1, 1)])
X_test = ss.hstack([X_test, np.ones(X_test.shape[0]).reshape(-1, 1)])
X_train, X_heldout, X_test = ss.csr_matrix(X_train), ss.csr_matrix(X_heldout), ss.csr_matrix(X_test)

classes_objects = collections.defaultdict(list)
classes_cnt = [0] * n_classes
for i, y in enumerate(y_train):
    classes_objects[y].append(i)
    classes_cnt[y] += 1
classes_cnt = np.array(classes_cnt)

predict_chunk_size = 1000


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, l.shape[0], n):
        yield l[i:i + n]


def predict_NN(X, Ws, WsT, metric="cosine"):
    y_pred = []
    for x_chunk in chunks(X, predict_chunk_size):
        if metric == "cosine":
            results = cosine_similarity(x_chunk, Ws).argmax(axis=1)
        else:
            results = np.array(x_chunk.dot(WsT).argmax(axis=1).T)[0]
        y_pred += list(results)
    return y_pred


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

    def elem_add(self, ix: int, s: float):
        self.v[ix] += (s / self.a)

    def elem_get(self, ix: int):
        return self.v[ix] * self.a

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
    def __init__(self, dim, dtype=np.float32):
        self.dim = n, d = dim
        self.dtype = dtype
        self.a = 1.0
        self.snorm = 0.
        self.m = np.array([ss.csr_matrix((1, d), dtype=dtype) for _ in range(n)])
        self.nnz = 0

    def sparse_dot(self, ix: int, v: ss.csr_matrix):
        return sparse_sparse_dot(self.m[ix], v) * self.a

    def sparse_add(self, ix: int, v: ss.csr_matrix, s: float):
        old_ix_norm = np.dot(self.m[ix].data, self.m[ix].data)
        old_nnz = self.m[ix].nnz
        self.m[ix] += v * (s / self.a)
        new_ix_norm = np.dot(self.m[ix].data, self.m[ix].data)
        new_nnz = self.m[ix].nnz
        self.snorm += (new_ix_norm - old_ix_norm) * (self.a * self.a)
        self.nnz += new_nnz - old_nnz
        return self.m[ix] * self.a

    def soft_threshold(self, ix: int, th: float):
        # Can skip updating snorm as the usage of soft_threshold and snorm is mutually exclusive
        th /= self.a
        arr = self.m[ix].toarray()
        gt_ix = (arr > +th)
        lt_ix = (arr < -th)
        eq_ix = (~gt_ix & ~lt_ix)
        arr[gt_ix] -= th
        arr[lt_ix] += th
        arr[eq_ix] = 0
        old_nnz = self.m[ix].nnz
        arr_sparse = ss.csr_matrix(arr, dtype=self.m[ix].dtype)
        new_nnz = arr_sparse.nnz
        self.m[ix] = arr_sparse
        self.nnz += new_nnz - old_nnz
        return arr_sparse * self.a

    def scale(self, s: float):
        if abs(s) < 1e-32:
            self.__init__(self.dim)
        else:
            self.a *= s
            self.snorm *= (s * s)


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


def multi_pegasos(X: np.array, y: np.array, lasso_svm=True, lsh_ann=False, random_seed=None) -> Tuple[WeightMatrix, Tuple]:
    n, d = X.shape

    # TODO: make parameters
    max_iter = 40
    eta0 = 0.1
    eta_decay_rate = 0.02

    if lasso_svm:
        k = 100 * int(np.sqrt(n_classes))
        lambd = 1.
    else:
        k = 100 * int(np.sqrt(n_classes))
        lambd = 1.

    W = WeightMatrix((n_classes, d))
    # Wyx = WeightVector(n)

    # amax1 = BruteforceArgmax(W)
    if lsh_ann:
        amax2 = ANNArgmax(n_classes, num_threads, LSH=True, n_features=d, hash_length=2048)
    else:
        amax2 = ANNArgmax(n_classes, num_threads)

    if random_seed is not None:
        np.random.seed(random_seed)
    if use_class_sampling:
        class_uniform_p = 1. / (len(classes_cnt[classes_cnt != 0]) * classes_cnt[y_train])
        random_ids = np.random.choice(n, size=max_iter * k, p=class_uniform_p)
    else:
        random_ids = np.random.choice(n, size=max_iter * k)

    # avg_scale = min(max_iter, num_to_avg)
    # avg_wv = WeightVector(d)
    amax_multiplier = 1.

    learning_time = 0.

    rs_stats = collections.Counter()
    ys_stats = collections.Counter()

    with open("log_%s_%d.txt" % (dataset_filename, os.getpid()), "w") as fout:
        fout.write("i,learning_time,maf1,mif1,amax_multiplier,nnz_sum,sparsity\n")

    # a, b = 0., 0.
    for i in tqdm(range(max_iter)):
        iter_start = time.time()
        x_ids = random_ids[i * k: (i + 1) * k]
        xs = X[x_ids]
        eta = eta0 / (1 + eta_decay_rate * i)

        ys = y[x_ids]
        # rs1 = amax1.query(xs, ys)
        rs2 = amax2.query(xs, ys)
        # keks1 = np.array([W.sparse_dot(r_, x_) for r_, x_ in zip(rs1, xs)])
        # keks2 = np.array([W.sparse_dot(r_, x_) for r_, x_ in zip(rs2, xs)])
        # kek = (keks1 - keks2)
        # assert np.all(kek >= -1e-9)
        # print(ys, rs1, rs2)
        # if np.any(kek >= 1e-9):
        #     # TODO: хотелось бы понять, почему query иногда не "видит" всех векторов в индексе
        #     print("wombat")
        # a += np.sum(kek <= 1e-9)
        # b += xs.shape[0]
        # print("Accuracy score: %.6f" % (a / b))
        rs = rs2
        grad_ixs, grad_weights = [], []

        # Collect class stats
        # rs_stats.update(rs)
        # ys_stats.update(ys)

        for j_, y_, r_, x_ in zip(x_ids, ys, rs, xs):
            if use_dummy_loss:
                loss = 1
            else:
                # loss = max(0, 1 + (-dr) - Wyx.elem_get(j_))
                # TODO: use wrx from dists
                wrx = W.sparse_dot(r_, x_)
                wyx = W.sparse_dot(y_, x_)
                loss = 1 + wrx - wyx
            if loss > 0:
                grad_ixs.append((y_, j_))
                grad_weights.append(+eta / k)
                grad_ixs.append((r_, j_))
                grad_weights.append(-eta / k)

        # Scale weight matrix and Wyx cache matrix
        if not lasso_svm:
            iter_scale = 1. - eta * lambd
            W.scale(iter_scale)
            amax_multiplier *= iter_scale
            # Wyx.scale(iter_scale)
        # Add sub-gradients and project rows onto a sphere of r=1
        amax_update = {}
        for (class_ix, obj_ix), grad_w in zip(grad_ixs, grad_weights):
            obj = X.getrow(obj_ix)
            upd = W.sparse_add(class_ix, obj, grad_w)
            # Incrementally update Wyx (<w_yk, xk>) cache matrix
            # for x_ix in classes_objects[class_ix]:
            #     Wyx.elem_add(x_ix, sparse_sparse_dot(X.getrow(x_ix), obj) * grad_w)
            upd.data /= amax_multiplier
            amax_update[class_ix] = upd
        # Do soft thresholding for lasso SVM
        if lasso_svm:
            W_ixs = list(set(ys) | set(rs))
            sparsity = W.nnz / W.dim[0] / W.dim[1]
            th = gamma * n_classes / len(W_ixs) * lambd * eta
            if th > 0:
                for class_ix in W_ixs:
                    upd = W.soft_threshold(class_ix, th)
                    amax_update[class_ix] = upd

        # Normalize weight matrix and Wyx cache matrix
        if not lasso_svm:
            # Projection step
            iter_norm = min(1., 1. / np.sqrt(lambd * W.snorm))
            W.scale(iter_norm)
            amax_multiplier *= iter_norm
            # Wyx.scale(iter_norm)
            # for class_ix, new_val in amax_update.items():
            #     snorm = np.dot(new_val.data, new_val.data)
            #     new_norm = min(1., 1. / np.sqrt(lambd * snorm))
            #     amax_update[class_ix] *= new_norm
        if len(amax_update) > 0:
            class_ixs = np.array(list(amax_update.keys()))
            new_values = ss.vstack(list(amax_update.values()))
            # print(class_ixs)
            # amax1.update(class_ixs, new_values)
            amax2.update(class_ixs, new_values)

        iter_end = time.time()
        learning_time += iter_end - iter_start

        if i % 100500 == 0 and i > 0:
            # Save intermediate W matrix
            # with open("W_%s.dump" % dataset_filename, "wb") as fout:
            #     pickle.dump(W, fout)
            # Create test index :(
            # TODO: incapsulation is broken -- fix
            # Calculate MaF1 and MiF1 heldout score
            nnz_sum = sum([x.nnz for x in W.m])
            sparsity = nnz_sum / (len(W.m) * W.m[0].shape[1])
            Ws = ss.vstack(W.m) * W.a
            WsT = ss.csr_matrix(Ws.T)
            y_pred_heldout = predict_NN(X_heldout, Ws, WsT, metric="cosine")
            # y_pred_heldout_dot = predict_NN(X_heldout, Ws, WsT, metric="dot")
            maf1 = f1_score(y_heldout, y_pred_heldout, average="macro")
            mif1 = f1_score(y_heldout, y_pred_heldout, average="micro")
            # maf1_dot = f1_score(y_heldout, y_pred_heldout_dot, average="macro")
            # mif1_dot = f1_score(y_heldout, y_pred_heldout_dot, average="micro")
            stats = [i, learning_time, maf1, mif1, amax_multiplier, nnz_sum, sparsity]
            with open("log_%s_%d.txt" % (dataset_filename, os.getpid()), "a") as fout:
                writer = csv.writer(fout)
                writer.writerow(stats)

    print("MaxRSS (in bytes): %d" % resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return W, (ys_stats, rs_stats)


if __name__ == "__main__":
    # Train
    print("processing %s ... (lasso = %d)" % (dataset_name, is_lasso))
    W, stats = multi_pegasos(X_train, y_train, lasso_svm=is_lasso, random_seed=0)
    with open("W_%s.dump" % dataset_filename, "wb") as fout:
        pickle.dump((W, stats), fout)
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
