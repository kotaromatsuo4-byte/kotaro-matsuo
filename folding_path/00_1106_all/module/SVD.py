import numpy as np
import warnings
from scipy import sparse
from scipy.sparse.linalg import (eigsh,LinearOperator,ArpackNoConvergence)

"""
疎行列ヤコビアンから小特異値モードを抽出する。
Returns
-------
soft_sigma : ndarray, shape (r,)
    抽出された特異値
soft_modes : ndarray, shape (n_free, r)
    各列が1つの右特異ベクトル
"""
def modes_for_sparse(J_sparse,abs_sigma_tol,gap_ratio,k_start,k_step,k_max,sigma_floor,eigensolver_tol,eigensolver_maxiter):
    J = J_sparse.tocsr().astype(np.float64, copy=False)
    n = J.shape[1]
    A = LinearOperator(shape=(n, n),matvec=lambda v: J.T @ (J @ v),dtype=np.float64,)

    # 探索数の初期化
    k_limit = min(k_max, n - 1)
    k = min(k_start, k_limit)

    #　小固有値の探索
    while True:
        try:
            eigval, eigvec = eigsh(A,k=k,which="SM",tol=eigensolver_tol,maxiter=eigensolver_maxiter)
        except ArpackNoConvergence as e:
            raise RuntimeError(f"eigsh did not converge for k={k}. ""Check maxiter and tolerance.") from e

        # 5. 特異値を直接評価
        sigma = np.linalg.norm(J @ eigvec, axis=0)
        order = np.argsort(sigma)
        sigma = sigma[order]
        eigvec = eigvec[:, order]

        # ギャップ判定
        cut = None
        for i in range(len(sigma) - 1):
            ratio = sigma[i + 1] / max(sigma[i], sigma_floor)
            if (sigma[i + 1] > abs_sigma_tol and ratio > gap_ratio):
                cut = i + 1
                break
        if cut is not None:
            return sigma[:cut], eigvec[:, :cut]
        
        # 最大探索数に達した場合
        if k == k_limit:
            idx = np.flatnonzero(sigma <= abs_sigma_tol)
            if len(idx) == k:
                warnings.warn("All computed singular values are below " "abs_sigma_tol. More soft modes may exist.",RuntimeWarning)
            return sigma[idx], eigvec[:, idx]

        # 8. 探索数を増やす
        k = min(k + k_step, k_limit)


"""
密行列ヤコビアンから小特異値モードを抽出する。
Returns
-------
sigma : ndarray, shape (r,)
    抽出された特異値
modes : ndarray, shape (n_free, r)
    各列が1つの右特異ベクトル
"""
def modes_for_dense(J_dense, tol):
    jacobian = np.asarray(J_dense, dtype=float)
    _, singular_values, right = np.linalg.svd(jacobian, full_matrices=True)
    sigma = np.zeros(jacobian.shape[1])
    sigma[:len(singular_values)] = singular_values
    selected = np.flatnonzero(sigma <= tol)
    selected = selected[np.argsort(sigma[selected])]
    return sigma[selected], right[selected].T


"""全自由度の行列を縮約し、対応する特異値と全自由度のモードを返す。optionsは各分解関数の引数。"""
def modes(jacobian, free_index, *, backend, **options):
    free_index = np.asarray(free_index, dtype=int)
    matrix = jacobian.tocsr() if sparse.issparse(jacobian) else np.asarray(jacobian)
    reduced = matrix[:, free_index]
    if backend == "dense":
        reduced = reduced.toarray() if sparse.issparse(reduced) else reduced
        sigma, modes_free = modes_for_dense(reduced, **options)
    elif backend == "sparse":
        sigma, modes_free = modes_for_sparse(sparse.csr_matrix(reduced), **options)
    modes_full = np.zeros((matrix.shape[1], modes_free.shape[1]))
    modes_full[free_index, :] = modes_free
    return sigma, modes_full

"""支持条件で縮約せず、剛体運動を含む全座標のモードを返す"""
def modes_free(jacobian, *, backend, **options):
    matrix = jacobian.tocsr() if sparse.issparse(jacobian) else np.asarray(jacobian)
    if backend == "dense":
        matrix = matrix.toarray() if sparse.issparse(matrix) else matrix
        sigma, modes_free = modes_for_dense(matrix, **options)
    elif backend == "sparse":
        sigma, modes_free = modes_for_sparse(sparse.csr_matrix(matrix), **options)
    return sigma, modes_free
