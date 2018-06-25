import os
import time
import pickle
import liblinear

datasets_names = ("LSHTC1", "DMOZ", "WIKI_Small", "WIKI_50K", "WIKI_100K")
dataset_dir = "../data"
out_dir = "../data/parsed"

# Read the dataset.

# dataset_name = "LSHTC1"
dataset_name = "WIKI_100K"

with open(os.path.join(out_dir, "%s_train.dump" % dataset_name), "rb") as fin:
    X_train = pickle.load(fin)
with open(os.path.join(out_dir, "%s_train_out.dump" % dataset_name), "rb") as fin:
    y_train = pickle.load(fin)
with open(os.path.join(out_dir, "%s_test.dump" % dataset_name), "rb") as fin:
    X_test = pickle.load(fin)
with open(os.path.join(out_dir, "%s_test_out.dump" % dataset_name), "rb") as fin:
    y_test = pickle.load(fin)

# pos_class = 33
pos_class = 1927

start_time = time.time()
y_train = (y_train == pos_class)
y_test = (y_test == pos_class)

m = liblinear.train(y_train, X_train, "-s 1 -c 1 -q")
end_time = time.time()

y_test, info, _ = liblinear.predict(y_test, X_test, m)
print(info)
print("Train time: %.3f" % (end_time - start_time))
