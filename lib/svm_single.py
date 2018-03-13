# Single machine version of Sparse SVM

from scipy.sparse import csr_matrix


def optimize(sparse_km: csr_matrix, eps: float, max_iter: int) -> csr_matrix:
    """
    Perform SVM on sparsified kernel matrix.
    :param sparse_km: sparsified kernel matrix.
    :param eps: epsilon value.
    :param max_iter: maximal number of iterations.
    :return: object weights.
    """
    pass