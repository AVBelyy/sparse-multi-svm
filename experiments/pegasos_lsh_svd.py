# Binary and multiclass SGD trainer for SVM a.k.a. Pegasos a.k.a. the baseline algorithm.

import sys
import csv
import resource
import os, pickle
import collections

import nmslib
import numpy as np
import time
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import f1_score
from sklearn.metrics.pairwise import cosine_similarity

from lib.argmax_tools import ANNArgmax, BruteforceArgmax, RandomArgmax
from tqdm import tqdm

from typing import Tuple

if __name__ == "__main__":
    # Read the dataset.
    in_dir = "../data/parsed"

    # dataset_name = "WIKI_100K"
    dataset_name = "LSHTC1"
    hash_length = 64
    # dataset_name = "20newsgroups"
    if len(sys.argv) > 1:
        dataset_name = sys.argv[1]
    if len(sys.argv) > 2:
        hash_length = int(sys.argv[2])

    num_threads = 16

    dataset_filename = "%s_lsh" % dataset_name

    use_class_sampling = True
    use_dummy_loss = False # TODO: tune maybe

    with open(os.path.join(in_dir, "%s_train_out.dump" % dataset_name), "rb") as fin:
        y_train = pickle.load(fin)
    with open(os.path.join(in_dir, "%s_heldout_out.dump" % dataset_name), "rb") as fin:
        y_heldout = pickle.load(fin)
    with open(os.path.join(in_dir, "%s_test_out.dump" % dataset_name), "rb") as fin:
        y_test = pickle.load(fin)

    with open(os.path.join(in_dir, "svd/%s_train.dump" % dataset_name), "rb") as fin:
        X_train = np.fromfile(fin, dtype=np.float32).reshape(len(y_train), -1)
    with open(os.path.join(in_dir, "svd/%s_heldout.dump" % dataset_name), "rb") as fin:
        X_heldout = np.fromfile(fin, dtype=np.float32).reshape(len(y_heldout), -1)
    with open(os.path.join(in_dir, "svd/%s_test.dump" % dataset_name), "rb") as fin:
        X_test = np.fromfile(fin, dtype=np.float32).reshape(len(y_test), -1)

    n_classes = 0
    for dataset_part in ("train", "heldout", "test"):
        with open(os.path.join(in_dir, "%s_%s_out.dump" % (dataset_name, dataset_part)), "rb") as fin:
            labels = pickle.load(fin)
            n_classes = max(n_classes, max(labels) + 1)


    print("I have PID", os.getpid())

    X_train = np.hstack([X_train, np.ones(X_train.shape[0]).reshape(-1, 1)])
    X_heldout = np.hstack([X_heldout, np.ones(X_heldout.shape[0]).reshape(-1, 1)])
    X_test = np.hstack([X_test, np.ones(X_test.shape[0]).reshape(-1, 1)])

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
Dense matrix in a form: a * [v_i], i=1...n
"""

class DenseWeightMatrix:
    def __init__(self, dim, dtype=np.float32):
        self.dim = n, d = dim
        self.dtype = dtype
        self.a = 1.0
        self.snorm = 0.
        self.m = np.zeros(dim, dtype=dtype)
        self.nnz = n * d

    def dense_dot(self, ix: int, v: np.array):
        return np.dot(self.m[ix], v) * self.a

    def dense_add(self, ix: int, v: np.array, s: float):
        old_ix_norm = np.dot(self.m[ix], self.m[ix])
        self.m[ix] += v * (s / self.a)
        new_ix_norm = np.dot(self.m[ix], self.m[ix])
        self.snorm += (new_ix_norm - old_ix_norm) * (self.a * self.a)
        return self.m[ix] * self.a

    def scale(self, s: float):
        if abs(s) < 1e-32:
            self.__init__(self.dim)
        else:
            self.a *= s
            self.snorm *= (s * s)


def multi_pegasos_lsh(X: np.array, y: np.array, random_seed=None) -> Tuple[DenseWeightMatrix, Tuple]:
    n, d = X.shape

    # TODO: make parameters
    max_iter = 50

    eta0 = 0.01
    eta_decay_rate = 0.02

    k = 100 * int(np.sqrt(n_classes))
    lambd = 1.

    W = DenseWeightMatrix((n_classes, d))

    amax = ANNArgmax(n_classes, num_threads, is_lsh=True, n_features=d, hash_length=hash_length)

    if random_seed is not None:
        np.random.seed(random_seed)
    if use_class_sampling:
        class_uniform_p = 1. / (len(classes_cnt[classes_cnt != 0]) * classes_cnt[y_train])
        random_ids = np.random.choice(n, size=max_iter * k, p=class_uniform_p)
    else:
        random_ids = np.random.choice(n, size=max_iter * k)

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
        rs = amax.query(xs, ys)
        grad_ixs, grad_weights = [], []

        # Collect class stats
        # rs_stats.update(rs)
        # ys_stats.update(ys)

        for j_, y_, r_, x_ in zip(x_ids, ys, rs, xs):
            if use_dummy_loss:
                loss = 1
            else:
                wrx = W.dense_dot(r_, x_)
                wyx = W.dense_dot(y_, x_)
                loss = 1 + wrx - wyx
            if loss > 0:
                grad_ixs.append((y_, j_))
                grad_weights.append(+eta / k)
                grad_ixs.append((r_, j_))
                grad_weights.append(-eta / k)

        # Scale weight matrix and Wyx cache matrix
        iter_scale = 1. - eta * lambd
        W.scale(iter_scale)
        amax_multiplier *= iter_scale
        # Add sub-gradients and project rows onto a sphere of r=1
        amax_update = {}
        for (class_ix, obj_ix), grad_w in zip(grad_ixs, grad_weights):
            obj = X[obj_ix]
            upd = W.dense_add(class_ix, obj, grad_w)
            upd /= amax_multiplier
            amax_update[class_ix] = upd

        # Normalize weight matrix and Wyx cache matrix
        # Projection step
        iter_norm = min(1., 1. / np.sqrt(lambd * W.snorm))
        W.scale(iter_norm)
        amax_multiplier *= iter_norm
        if len(amax_update) > 0:
            class_ixs = np.array(list(amax_update.keys()))
            new_values = np.vstack(list(amax_update.values()))
            amax.update(class_ixs, new_values)

        iter_end = time.time()
        learning_time += iter_end - iter_start

        if i % 1 == 0 and i > 0:
            # Save intermediate W matrix
            # with open("W_%s.dump" % dataset_filename, "wb") as fout:
            #     pickle.dump(W, fout)
            # Calculate MaF1 and MiF1 heldout score
            nnz_sum = W.nnz
            sparsity = 1.0
            Ws = W.m # W.m * W.a
            WsT = None # Ws.T
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

    print("Learning time: %.1f" % learning_time)
    print("Non-zero elements: %d" % W.nnz)
    return W, (ys_stats, rs_stats)


if __name__ == "__main__":
    # Train
    print("processing %s ..." % dataset_name)
    W, stats = multi_pegasos_lsh(X_train, y_train, random_seed=0)
    with open("W_%s.dump" % dataset_filename, "wb") as fout:
        pickle.dump((W, stats), fout)
