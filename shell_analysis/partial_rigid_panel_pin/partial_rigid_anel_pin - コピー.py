import numpy as np
from pathlib import Path
from SVD import modes
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix

#loading
def load_int_table(filename, ncol=None):
    try:
        arr = np.loadtxt(filename, dtype=int)
    except ValueError:
        try:
            arr = np.loadtxt(filename, delimiter=",", dtype=int)
        except ValueError:
            if ncol is None:
                return np.array([], dtype=int)
            return np.empty((0, ncol), dtype=int)

    arr = np.asarray(arr, dtype=int)

    if arr.size == 0:
        if ncol is None:
            return np.array([], dtype=int)
        return np.empty((0, ncol), dtype=int)

    if ncol is None:
        return np.atleast_1d(arr).ravel()

    return np.asarray(arr).reshape(-1, ncol)

#model
node = np.loadtxt("00_node_initial.dat", dtype=float)
panel = np.loadtxt("01_face.dat", dtype=int)
member = np.loadtxt("02_member.dat", dtype=int)
fix_member_id = load_int_table("03_fix_crease_id.dat", ncol=None)
fix = load_int_table("04_fix.dat", ncol=2)

"""

node = np.array([
    [0.0,0.0,0.0],
    [0.5,0.0,0.0],
    [0.5,0.5,0.7071],
    [1.0,0.0,0.0],
    [1.0,1.0,0.0]
])
member = np.array([
    [0,1],[0,2],[0,4],[1,3],[1,4],[2,4],[3,4]
])

panel = np.array([
    [1,4,3],
    [0,4,1],
    [0,2,4]
])

fix_member_id = []

fix = np.array([
    [0,0],[0,1],[0,2],
    [4,0],[4,1],[4,2],
    [2,0],[2,1],[2,2]
])
"""

L0 = np.array([np.linalg.norm(node[i]-node[j]) for i,j in member])
phi0 = np.array([0 for i in fix_member_id])
#angle
def angle(a,b):
    c = np.dot(a,b) / (np.linalg.norm(a)* np.linalg.norm(b))
    c = np.clip(c,-1.0,1.0)
    return np.arccos(c)

#cot(x)
def cot(x):
    return 1/np.tan(x)

#find k,l for fold angle
def find_kl(panel, i, j):
    faces = []
    for tri in panel:
        if i in tri and j in tri:
            faces.append(tri)
    tri1, tri2 = faces
    k = [v for v in tri1 if v not in (i,j)][0]
    l = [v for v in tri2 if v not in (i,j)][0]

    return k, l

#extract free index
def index_free(num_node, fix):
    idx = np.arange(num_node * 3)
    return np.delete(idx, 3 * fix[:, 0] + fix[:, 1])


def make_dof_map(num_node, fix):
    all_dof = np.arange(num_node * 3)

    if fix is None or len(fix) == 0:
        fixed = np.array([], dtype=int)
    else:
        fix = np.array(fix, dtype=int).reshape(-1, 2)
        fixed = 3 * fix[:, 0] + fix[:, 1]

    free = np.setdiff1d(all_dof, fixed)

    dof_map = -np.ones(num_node * 3, dtype=int)
    dof_map[free] = np.arange(len(free))

    return free, dof_map


def add_grad_sparse(rows, cols, data, row, node_id, grad, dof_map):
    for a in range(3):
        global_dof = 3 * node_id + a
        col = dof_map[global_dof]

        if col >= 0:  # 固定自由度なら -1 なので無視
            rows.append(row)
            cols.append(col)
            data.append(grad[a])

#incompatibility and Jacobian_matrix
def G_dGdX_sparse(node, panel, fix_member_id, member, L0, phi0, fix):

    fix_member_id = np.array(fix_member_id, dtype=int).ravel()
    phi0 = np.array(phi0, dtype=float).ravel()

    m = member.shape[0]
    f = len(fix_member_id)
    n = node.shape[0]

    free, dof_map = make_dof_map(n, fix)

    G_bar = np.zeros((m, 1))
    G_panel = np.zeros((f, 1))

    rows = []
    cols = []
    data = []

    # bar length
    for x in range(m):
        i, j = member[x]

        diff = node[i] - node[j]

        G_bar[x] = 0.5 * (np.dot(diff, diff) - L0[x]**2)

        dG_di = diff
        dG_dj = -diff

        add_grad_sparse(rows, cols, data, x, i, dG_di, dof_map)
        add_grad_sparse(rows, cols, data, x, j, dG_dj, dof_map)

    # fold angle
    for x in range(f):
        y = fix_member_id[x]
        i, j = member[y]

        k, l = find_kl(panel, i, j)

        n_k = np.cross(node[k] - node[j], node[i] - node[j])
        n_k = n_k / np.linalg.norm(n_k)

        n_l = np.cross(node[i] - node[j], node[l] - node[j])
        n_l = n_l / np.linalg.norm(n_l)

        phi_ij = np.arccos(np.clip(np.dot(n_k, n_l), -1.0, 1.0))
        G_panel[x] = phi_ij - phi0[x]

        theta_j_ki = angle(node[k] - node[j], node[i] - node[j])
        theta_j_il = angle(node[l] - node[j], node[i] - node[j])
        theta_i_jk = angle(node[j] - node[i], node[k] - node[i])
        theta_i_lj = angle(node[l] - node[i], node[j] - node[i])

        h_k = np.sin(theta_j_ki) * np.linalg.norm(node[k] - node[j])
        h_l = np.sin(theta_j_il) * np.linalg.norm(node[l] - node[j])

        d_phi_d_k = n_k / h_k
        d_phi_d_l = n_l / h_l

        d_phi_d_i = (
            -cot(theta_j_ki) / (cot(theta_j_ki) + cot(theta_i_jk)) * d_phi_d_k
            -cot(theta_j_il) / (cot(theta_j_il) + cot(theta_i_lj)) * d_phi_d_l
        )

        d_phi_d_j = (
            -cot(theta_i_jk) / (cot(theta_j_ki) + cot(theta_i_jk)) * d_phi_d_k
            -cot(theta_i_lj) / (cot(theta_j_il) + cot(theta_i_lj)) * d_phi_d_l
        )

        row = m + x

        add_grad_sparse(rows, cols, data, row, i, d_phi_d_i, dof_map)
        add_grad_sparse(rows, cols, data, row, j, d_phi_d_j, dof_map)
        add_grad_sparse(rows, cols, data, row, k, d_phi_d_k, dof_map)
        add_grad_sparse(rows, cols, data, row, l, d_phi_d_l, dof_map)

    G = np.vstack([G_bar, G_panel])

    dGdX_sparse = coo_matrix(
        (data, (rows, cols)),
        shape=(m + f, len(free))
    ).tocsr()

    return G, dGdX_sparse, free

def expand_modes_to_full(soft_modes_free, free, num_node):
    """
    soft_modes_free : ndarray, shape = (num_free_dof, num_modes)
        自由DOFだけの固有モード

    free : ndarray
        全DOF番号における自由DOF index

    num_node : int
        節点数

    return
    ------
    soft_modes_full : ndarray, shape = (3*num_node, num_modes)
        固定DOFを0で埋めた全DOFモード
    """

    num_full_dof = 3 * num_node
    num_modes = soft_modes_free.shape[1]

    soft_modes_full = np.zeros((num_full_dof, num_modes))
    soft_modes_full[free, :] = soft_modes_free

    return soft_modes_full

G, J_free, free = G_dGdX_sparse(
    node,
    panel,
    fix_member_id,
    member,
    L0,
    phi0,
    fix
)

# 固有値解析は自由DOFだけ
soft_sigma, soft_modes_free = modes(
    J_free,
    abs_sigma_tol=1e-6,
    gap_ratio=10.0,
    k_start=12,
    k_max=100
)

# 出力用に全DOFへ戻す
soft_modes = expand_modes_to_full(
    soft_modes_free,
    free,
    node.shape[0]
)



# 保存先フォルダ
result_dir = Path("result")
mode_dir = result_dir / "mode"

result_dir.mkdir(exist_ok=True)
mode_dir.mkdir(exist_ok=True)

for old_mode in mode_dir.glob("soft_mode_*.dat"):
    old_mode.unlink()

for i in range(len(soft_sigma)):
    np.savetxt(
        mode_dir / f"soft_mode_{i}.dat",
        soft_modes[:, i].reshape(-1, 3),
        delimiter=",",
        fmt="%.6f"
    )

# sigmaと1つ前のsigmaとの比を保存
sigma_out = []

for i, sigma in enumerate(soft_sigma):
    if i == 0:
        sigma_out.append([sigma])
    else:
        ratio = sigma / soft_sigma[i - 1]
        sigma_out.append([sigma, ratio])

with open(result_dir / "soft_sigma.dat", "w") as f:
    for row in sigma_out:
        if len(row) == 1:
            f.write(f"{row[0]:.6f}\n")
        else:
            f.write(f"{row[0]:.6f},{row[1]:.6f}\n")