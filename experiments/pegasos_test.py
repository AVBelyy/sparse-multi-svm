import os
import sys
import ctypes
import pickle
import nmslib
import itertools
import collections
import numpy as np
import scipy.sparse as ss
from sklearn.feature_extraction.text import TfidfTransformer
from lib.sparse_tools import dense_sparse_dot, dense_sparse_add, sparse_sparse_dot
from lib.argmax_tools import BruteforceArgmax, ANNArgmax
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances_argmin, jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from sklearn.utils.extmath import safe_sparse_dot
from tqdm import tqdm
from time import time
from multiprocessing import Pool, RawArray

datasets_names = ("LSHTC1", "DMOZ", "WIKI_Small", "WIKI_50K", "WIKI_100K")
dataset_dir = "../data"
out_dir = "../data/parsed"

# Read the dataset.

# dataset_name = "WIKI_Small"
# dataset_name = "DMOZ"
# dataset_name = "LSHTC1"
# dataset_name = "20newsgroups"
dataset_name = sys.argv[1]

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

n_features = X_train.shape[1]
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

classes_objects = collections.defaultdict(list)
classes_cnt = [0] * n_classes
for i, y in enumerate(y_train):
    classes_objects[y].append(i)
    classes_cnt[y] += 1
classes_cnt = np.array(classes_cnt)

X_train = ss.hstack([X_train, np.ones(X_train.shape[0]).reshape(-1, 1)])
X_heldout = ss.hstack([X_heldout, np.ones(X_heldout.shape[0]).reshape(-1, 1)])
X_test = ss.hstack([X_test, np.ones(X_test.shape[0]).reshape(-1, 1)])
X_train, X_heldout, X_test = ss.csr_matrix(X_train), ss.csr_matrix(X_heldout), ss.csr_matrix(X_test)

print("Init done")

class WeightMatrix:
    def __init__(self, dim):
        self.dim = n, d = dim
        self.a = 1.0
        self.snorm = 0.
        self.m = [ss.csr_matrix((1, d), dtype=np.float32) for _ in range(n)]

    def sparse_dot(self, ix: int, v: ss.csr_matrix):
        return sparse_sparse_dot(self.m[ix], v) * self.a

    def sparse_add(self, ix: int, v: ss.csr_matrix, s: float):
        old_ix_norm = np.dot(self.m[ix].data, self.m[ix].data)
        self.m[ix] += v * (s / self.a)
        new_ix_norm = np.dot(self.m[ix].data, self.m[ix].data)
        self.snorm += (new_ix_norm - old_ix_norm) * (self.a * self.a)
        return self.m[ix] * self.a

    def scale(self, s: float):
        if abs(s) < 1e-32:
            self.__init__(self.dim)
        else:
            self.a *= s
            self.snorm *= (s*s)

wm_filename = sys.argv[2] if len(sys.argv) > 2 else "W_%s.dump" % dataset_name
with open(wm_filename, "rb") as fin:
    W, _ = pickle.load(fin)
    # W = pickle.load(fin)

# normalize all vectors
W.a /= W.a * np.sqrt(max([np.dot(x.data, x.data) for x in W.m]))

Ws = ss.vstack(W.m) * W.a

print("Reading weight matrix done")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, l.shape[0], n):
        yield l[i:i + n]

chunk_size = 1000
# WsT = ss.csr_matrix(Ws.T)
num_threads = 12

def predict_ANN(X):
    y_pred = []
    for x_chunk in tqdm(chunks(X, chunk_size)):
        results = index.knnQueryBatch(x_chunk, k=num_candidates, num_threads=num_threads)
        for x, (nn_ids, _) in zip(x_chunk, results):
            ix = 0
            class_id = nn_ids[ix]
            y_pred.append(class_id)
    return y_pred

def share_np_array(arr):
    if arr.dtype == np.float64:
        ctype = ctypes.c_double
    elif arr.dtype == np.int32:
        ctype = ctypes.c_int
    else:
        raise NotImplementedError
    sharr = RawArray(ctype, len(arr))
    sharr_np = np.frombuffer(sharr, dtype=arr.dtype).reshape(arr.shape)
    np.copyto(sharr_np, arr)
    return sharr

def share_sparse_array(sparr):
    shsparr = {}
    shsparr["shape"] = sparr.shape
    for k, v in {"data": sparr.data, "indices": sparr.indices, "indptr": sparr.indptr}.items():
        shsparr[k] = (share_np_array(v), v.dtype, len(v))
    return tuple(shsparr.items())

def load_sparse_matrix(shsparr):
    shsparr = dict(shsparr)
    sparr = {}
    sparr["shape"] = shsparr["shape"]
    del shsparr["shape"]
    for k, (sharr, dtype, count) in shsparr.items():
        sparr[k] = np.frombuffer(sharr, dtype=dtype, count=count)
    spmat = ss.csr_matrix((sparr["data"], sparr["indices"], sparr["indptr"]), shape=sparr["shape"], copy=False)
    return spmat

Ws_shared = share_sparse_array(Ws)
Ws_worker = None

def init_worker(args):
    global Ws_worker
    Ws_worker = load_sparse_matrix(args)

def worker_func(x):
    return cosine_similarity(x, Ws_worker).argmax(axis=1)

def predict_NN(X, metric="cosine"):
    if metric != "cosine":
        raise NotImplementedError
    with Pool(processes=num_threads, initializer=lambda *x: init_worker(x), initargs=Ws_shared) as pool:
        result = pool.map(worker_func, chunks(X, chunk_size))
        y_pred = list(itertools.chain.from_iterable(result))
    return y_pred

t1 = time()
y_pred_test = predict_NN(X_test, metric="cosine")
t2 = time()
print("Predicting done")
print()

maf1 = f1_score(y_test, y_pred_test, average="macro")
mif1 = f1_score(y_test, y_pred_test, average="micro")

print("Prediction time = %.1f" % (t2 - t1))
print("Macro F1 (cosine simil) = %.6f" % maf1)
print("Micro F1 (cosine simil) = %.6f" % mif1)
