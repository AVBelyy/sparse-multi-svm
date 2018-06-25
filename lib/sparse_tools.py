from scipy.sparse import csr_matrix
import numpy as np

def sparse_clip(x: csr_matrix, min_, max_, inplace=False):
    if inplace:
        out = x
    else:
        out = x.copy()
    out.data = np.clip(x.data, min_, max_)
    return out

def sparse_pos_clip(a: csr_matrix):
    # TODO: optimize
    return a.multiply(a > 0)

def dense_pos_clip(a: np.array, inplace=False):
    if inplace:
        out = a
    else:
        out = a.copy()
    out[out < 0] = 0
    return out

def sparse_sparse_dot(a : csr_matrix, b: csr_matrix):
    ixs_a, ixs_b = a.indices, b.indices
    val_a, val_b = a.data, b.data
    len_a, len_b = len(ixs_a), len(ixs_b)
    i, j = 0, 0
    ans = 0
    while i < len_a and j < len_b:
        if ixs_a[i] < ixs_b[j]:
            i += 1
        elif ixs_a[i] > ixs_b[j]:
            j += 1
        else:
            ans += val_a[i] * val_b[j]
            i += 1
            j += 1
    return ans

def dense_sparse_add(a: np.array, b: csr_matrix, inplace=False):
    if inplace:
        out = a
    else:
        out = a.copy()
    out[b.indices] += b.data
    return out

def dense_sparse_mul(a: np.array, b: csr_matrix, inplace=False):
    if inplace:
        out = b
    else:
        out = b.copy()
    out.data *= a[b.indices]
    return out

def dense_sparse_dot(a: np.array, b: csr_matrix):
    return np.dot(b.data, a[b.indices])

def sparse_sub_with_clip(a: csr_matrix, c):
    out = a.copy()
    out.data -= c
    return sparse_pos_clip(out)

