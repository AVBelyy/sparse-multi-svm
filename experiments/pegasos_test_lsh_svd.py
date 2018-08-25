import os
import sys
import ctypes
import pickle
import nmslib
import resource
import operator
import functools
import itertools
import collections
import numpy as np
import scipy.sparse as ss
from sklearn.feature_extraction.text import TfidfTransformer
from lib.argmax_tools import BruteforceArgmax, ANNArgmax
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import pairwise_distances_argmin, jaccard_similarity_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import linear_kernel
from sklearn.svm import LinearSVC
from tqdm import tqdm
from time import time
from multiprocessing import Pool, RawArray

from pegasos_lsh import DenseWeightMatrix

datasets_names = ("LSHTC1", "DMOZ", "WIKI_Small", "WIKI_50K", "WIKI_100K")
dataset_dir = "../data"
in_dir = "../data/parsed"

# Read the dataset.

# dataset_name = "WIKI_Small"
# dataset_name = "DMOZ"
# dataset_name = "LSHTC1"
# dataset_name = "20newsgroups"
dataset_name = sys.argv[1]
wm_filename = sys.argv[2] if len(sys.argv) > 2 else "W_%s.dump" % dataset_name

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

n_features = X_train.shape[1]
n_classes = 0
for dataset_part in ("train", "heldout", "test"):
    with open(os.path.join(in_dir, "%s_%s_out.dump" % (dataset_name, dataset_part)), "rb") as fin:
        labels = pickle.load(fin)
        n_classes = max(n_classes, max(labels) + 1)

classes_objects = collections.defaultdict(list)
classes_cnt = [0] * n_classes
for i, y in enumerate(y_train):
    classes_objects[y].append(i)
    classes_cnt[y] += 1
classes_cnt = np.array(classes_cnt)

X_train = np.hstack([X_train, np.ones(X_train.shape[0]).reshape(-1, 1)])
X_heldout = np.hstack([X_heldout, np.ones(X_heldout.shape[0]).reshape(-1, 1)])
X_test = np.hstack([X_test, np.ones(X_test.shape[0]).reshape(-1, 1)])

print("Init done")

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, l.shape[0], n):
        yield l[i:i + n]

chunk_size = 1000
num_threads = 16

def share_np_array(arr):
    if arr.dtype == np.float64:
        ctype = ctypes.c_double
    elif arr.dtype == np.float32:
        ctype = ctypes.c_float
    elif arr.dtype == np.int32:
        ctype = ctypes.c_int
    else:
        raise NotImplementedError
    arr_size = functools.reduce(operator.mul, arr.shape, 1)
    sharr = RawArray(ctype, arr_size)
    sharr_np = np.frombuffer(sharr, dtype=arr.dtype).reshape(arr.shape)
    np.copyto(sharr_np, arr)
    return (('shape', arr.shape), ('dtype', arr.dtype), ('data', sharr))

def load_np_array(sharr):
    sharr = dict(sharr)
    arr_size = functools.reduce(operator.mul, sharr["shape"], 1)
    arr = np.frombuffer(sharr["data"], dtype=sharr["dtype"], count=arr_size).reshape(sharr["shape"])
    return arr

W_shared = None
W_worker = None

def init_worker(args):
    global W_worker
    W_worker = load_np_array(args)

def worker_func(x):
    return cosine_similarity(x, W_worker).argmax(axis=1)

def predict_NN(X, metric="cosine"):
    if metric != "cosine":
        raise NotImplementedError
    with Pool(processes=num_threads, initializer=lambda *x: init_worker(x), initargs=W_shared) as pool:
        result = pool.map(worker_func, chunks(X, chunk_size))
        y_pred = list(itertools.chain.from_iterable(result))
    return y_pred

t11 = time()
with open(wm_filename, "rb") as fin:
    W, _ = pickle.load(fin)
t12 = time()
W_shared = share_np_array(W.m)

print("Loading done")

t21 = time()
y_pred_test = predict_NN(X_test, metric="cosine")
t22 = time()
print("Predicting done")
print()

maf1 = f1_score(y_test, y_pred_test, average="macro")
mif1 = f1_score(y_test, y_pred_test, average="micro")

memory_usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

print("Training time = %.1f" % (t12 - t11))
print("Prediction time = %.1f" % (t22 - t21))
print("Total memory (in bytes) = %d" % memory_usage)
print("Micro F1 (cosine simil) = %.6f" % mif1)
print("Macro F1 (cosine simil) = %.6f" % maf1)
