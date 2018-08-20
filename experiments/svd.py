import sys
import os, pickle
import collections

import numpy as np
import scipy.sparse as ss
import time

from sklearn.feature_extraction.text import TfidfTransformer

# Read the dataset.
in_dir = "../data/parsed"
out_dir = "../data/parsed/svd"

dataset_name = "LSHTC1"
if len(sys.argv) > 1:
    dataset_name = sys.argv[1]

with open(os.path.join(in_dir, "%s_train.dump" % dataset_name), "rb") as fin:
    X_train = pickle.load(fin)
with open(os.path.join(in_dir, "%s_heldout.dump" % dataset_name), "rb") as fin:
    X_heldout = pickle.load(fin)
with open(os.path.join(in_dir, "%s_test.dump" % dataset_name), "rb") as fin:
    X_test = pickle.load(fin)

t1 = time.time()
tfidf = TfidfTransformer()
tfidf.fit(X_train)
X_train = tfidf.transform(X_train, copy=False)
X_heldout = tfidf.transform(X_heldout, copy=False)
X_test = tfidf.transform(X_test, copy=False)

X_train = X_train.astype(np.float32)
X_heldout = X_heldout.astype(np.float32)
X_test = X_test.astype(np.float32)

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=2048, algorithm="arpack", random_state=0)
svd.fit(X_train)
t2 = time.time()

print("It took me %.1f seconds to SVD %s" % (t2 - t1, dataset_name))

with open(os.path.join(out_dir, "%s_train.dump" % dataset_name), "wb") as fout:
    X_train = svd.transform(X_train)
    X_train.tofile(fout)
    del X_train
with open(os.path.join(out_dir, "%s_heldout.dump" % dataset_name), "wb") as fout:
    X_heldout = svd.transform(X_heldout)
    X_heldout.tofile(fout)
    del X_heldout
with open(os.path.join(out_dir, "%s_test.dump" % dataset_name), "wb") as fout:
    X_test = svd.transform(X_test)
    X_test.tofile(fout)
    del X_test

