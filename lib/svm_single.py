# Single machine version of Sparse SVM

from scipy.sparse import csr_matrix, csc_matrix
import numpy as np
import timeit

from lib.gradient_tools import BasicGradientUpdater, HeapGradientUpdater
from lib.sparse_tools import sparse_clip

def optimize(sparse_km: csc_matrix, gamma: float,
             regcoef: float, L1: float, eps: float, max_iter: int) -> (csc_matrix, dict):
    """
    Perform SVM on sparsified kernel matrix.
    :param sparse_km: sparsified kernel matrix.
    :param eps: epsilon value.
    :param max_iter: maximal number of iterations.
    :return: object weights.
    """

    if sparse_km.shape[0] != sparse_km.shape[1]:
        raise Exception("Kernel matrix is not a squared matrix")

    log = {"grad_norm": [], "time": []}

    N = sparse_km.shape[0]

    def grad_f(x):
        t = x.copy()
        t.data -= 1 / (2 * N * regcoef)
        return -csr_matrix((N, 1)) + sparse_km.dot(x) - \
               gamma*sparse_clip(-x, 0, None) + gamma*sparse_clip(t, 0, None)


    x0 = csr_matrix((N, 1))
    x0[0, 0] = 1/2
    grad_f0 = grad_f(x0)
    grad_min = BasicGradientUpdater(grad_f0.T)
    grad_max = BasicGradientUpdater(-grad_f0.T)

    iter_counter = 0

    start = timeit.default_timer()

    current_point = x0
    true_grad = grad_f0

    while grad_min.get_norm() > eps**2 or iter_counter < max_iter:

        #if true_grad == grad_min.get():

        log["grad_norm"].append(grad_min.get_norm())
        log["time"].append(timeit.default_timer() - start)

        i_plus = grad_max.get_coordinate()
        g_plus = -grad_max.get_value()
        i_minus = grad_min.get_coordinate()
        g_minus = grad_min.get_value()

        h_val = 1/(4*L1)*(g_plus - g_minus)
        h = csr_matrix((N, 1))
        h[i_plus, 0] = h_val
        h[i_minus, 0] = -h_val

        t = current_point.copy() # had to make this turnaround 'cause "sparse vector + constant" operation hasn't been implemented yet
        t.data -= 1 / (2 * N * regcoef)

        delta_grad = sparse_km.dot(h)
        delta_grad -= gamma*sparse_clip(-current_point - h, 0, None)
        delta_grad += gamma*sparse_clip(-current_point, 0, None)
        delta_grad += gamma*sparse_clip(t + h, 0, None)
        delta_grad -= gamma*sparse_clip(t, 0, None)

        grad_min.update(delta_grad.T)
        grad_max.update(-delta_grad.T)

        current_point += h
        true_grad = grad_f(current_point.T)
        iter_counter += 1

    return h, log