import numpy as np
from scipy import sparse
from scipy.sparse.linalg import eigsh


def modes(
    J_sparse,
    abs_sigma_tol=1.0e-6,
    gap_ratio=10.0,
    k_start=6,
    k_step=6,
    k_max=60,
):
    """
    Parameters
    ----------
    J_sparse : scipy.sparse.csr_matrix
        Compatibility Jacobian

    abs_sigma_tol : float
        特異値の絶対閾値

    gap_ratio : float
        σ_{i+1}/σ_i がこれを超えたら打ち切る

    k_start : int
        最初に求める特異値数

    k_step : int
        足りなければ増やす数

    k_max : int
        最大探索数

    Returns
    -------
    soft_sigma : ndarray
    soft_modes : ndarray
        各列が1つのモード
    """

    n = J_sparse.shape[1]
    A = J_sparse.T @ J_sparse

    def solve_dense():
        A_dense = A.toarray() if sparse.issparse(A) else np.asarray(A)
        eigval, eigvec = np.linalg.eigh(A_dense)
        order = np.argsort(eigval)
        eigval = np.maximum(eigval[order], 0.0)
        eigvec = eigvec[:, order]
        return np.sqrt(eigval), eigvec

    k = min(k_start, n - 1)

    if n <= 200 or k >= n - 1:
        sigma, eigvec = solve_dense()

        cut = None
        for i in range(len(sigma) - 1):
            ratio = sigma[i + 1] / max(sigma[i], 1e-30)
            if sigma[i + 1] > abs_sigma_tol and ratio > gap_ratio:
                cut = i + 1
                break

        if cut is not None:
            return sigma[:cut], eigvec[:, :cut]

        idx = np.where(sigma <= abs_sigma_tol)[0]
        if len(idx) > 0:
            return sigma[:idx[-1] + 1], eigvec[:, :idx[-1] + 1]

        return sigma, eigvec

    while True:

        print("before eigsh")
        eigval, eigvec = eigsh(
            A,
            k=k,
            which="SM"
        )
        print("after eigsh")
        order = np.argsort(eigval)

        eigval = np.maximum(eigval[order], 0.0)
        eigvec = eigvec[:, order]

        sigma = np.sqrt(eigval)

        # ギャップ判定
        cut = None

        for i in range(len(sigma) - 1):

            ratio = sigma[i + 1] / max(sigma[i], 1e-30)

            if (
                sigma[i + 1] > abs_sigma_tol
                and ratio > gap_ratio
            ):
                cut = i + 1
                break

        if cut is not None:
            return sigma[:cut], eigvec[:, :cut]

        # 絶対閾値を超えたものしか残らない場合
        if sigma[-1] > abs_sigma_tol:

            idx = np.where(sigma <= abs_sigma_tol)[0]

            if len(idx) > 0:
                return sigma[:idx[-1] + 1], eigvec[:, :idx[-1] + 1]

        if k >= min(k_max, n - 1):
            return sigma, eigvec

        k = min(k + k_step, k_max, n - 1)
