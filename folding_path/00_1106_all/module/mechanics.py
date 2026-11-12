import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsmr

#mechanics

def index_free(num_node, fix):
    fixed = np.asarray(fix, dtype=int).reshape(-1, 2)
    return np.delete(np.arange(num_node * 3), 3 * fixed[:, 0] + fixed[:, 1])


def member_lengths(node, member):
    node = np.asarray(node, dtype=float)
    member = np.asarray(member, dtype=int).reshape(-1, 2)
    differences = node[member[:, 0]] - node[member[:, 1]]
    return np.linalg.norm(differences, axis=1)


def strain(node, member, lengths):
    lengths = np.asarray(lengths, dtype=float)
    return (member_lengths(node, member) - lengths) / lengths


def incompatibility(node, member, lengths):
    lengths = np.asarray(lengths, dtype=float)
    current_lengths = member_lengths(node, member)
    return (current_lengths**2 - lengths**2) / 2


def jacobian(node, member, *, backend):
    node = np.asarray(node, dtype=float)
    member = np.asarray(member, dtype=int).reshape(-1, 2)
    differences = node[member[:, 0]] - node[member[:, 1]]
    rows = np.repeat(np.arange(len(member)), 6)
    cols = (3 * member[:, :, None] + np.arange(3)).reshape(-1)
    values = np.stack((differences, -differences), axis=1).reshape(-1)
    J = sparse.coo_matrix((values, (rows, cols)), shape=(len(member), node.size)).tocsr()
    return J if backend == "sparse" else J.toarray()


#step

def predictor(node, displacement_modes, A, *, step_size):
    node = np.asarray(node, dtype=float)
    displacement = np.asarray(displacement_modes) @ np.asarray(A, dtype=float)
    return node + step_size * displacement.reshape(node.shape)


def corrector(node, member, lengths, free_index, *, backend, tolerance, max_iterations, max_backtracks, initial_step_size, backtrack_shrink, linear_atol, linear_btol, linear_rcond, linear_maxiter, linear_conlim):
    corrected = np.asarray(node, dtype=float).copy()
    free_index = np.asarray(free_index, dtype=int)
    for _ in range(max_iterations):
        residual = incompatibility(corrected, member, lengths)
        residual_norm = np.linalg.norm(residual)
        if residual_norm <= tolerance:
            break
        # 自由自由度に縮約して補正量を求め、全自由度に戻す。
        J = jacobian(corrected, member, backend=backend)[:, free_index]
        if backend == "sparse":
            delta_free = lsmr(J, -residual, atol=linear_atol, btol=linear_btol, maxiter=linear_maxiter, conlim=linear_conlim)[0]
        else:
            delta_free = np.linalg.lstsq(J, -residual, rcond=linear_rcond)[0]
        delta = np.zeros(corrected.size)
        delta[free_index] = delta_free
        delta = delta.reshape(corrected.shape)
        # バックトラックで残差の減る補正幅を選ぶ。
        step_size = initial_step_size
        accepted = False
        for _ in range(max_backtracks):
            candidate = corrected + step_size * delta
            if np.linalg.norm(incompatibility(candidate, member, lengths)) < residual_norm:
                corrected = candidate
                accepted = True
                break
            step_size *= backtrack_shrink
        if not accepted:
            break
    return corrected


# 評価関数

"""dimension : 2(最小二乗直線)/3(最小二乗平面)。距離二乗和をスカラーで返す。"""
def compute_flatness_potential(node, dimension):
    node = np.asarray(node, dtype=float)
    centered = node - node.mean(axis=0, keepdims=True)
    sigma = np.linalg.svd(centered, compute_uv=False)
    return float(np.sum(sigma[dimension - 1:] ** 2))


"""dimension : 2(最小二乗直線)/3(最小二乗平面)。投影後の最大節点間距離をスカラーで返す。"""
def projected_spread(node, dimension):
    node = np.asarray(node, dtype=float)
    centered = node - node.mean(axis=0, keepdims=True)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ right[:dimension - 1].T
    differences = projected[:, None, :] - projected[None, :, :]
    return float(np.max(np.linalg.norm(differences, axis=2)))
