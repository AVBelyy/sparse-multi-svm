# Single machine version of kernel matrix computation

from scipy.sparse import csr_matrix


def compute_km(data: csr_matrix, kernel_type="linear") -> csr_matrix:
    if kernel_type == "linear":
        return data * data.T
    elif kernel_type == "rbf":
        raise NotImplementedError()
    else:
        raise ValueError("Unknown kernel_type")
