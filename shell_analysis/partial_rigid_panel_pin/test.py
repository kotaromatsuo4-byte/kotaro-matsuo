import numpy as np
import torch

# モデル
node0=np.loadtxt("node_initial.dat",dtype=float)
member=np.loadtxt("member.dat",dtype=int)
panel=np.loadtxt("face.dat",dtype=int)
fix_crease=np.loadtxt("fix_crease.dat",dtype=int)
fix=np.loadtxt("fix.dat",dtype=int)

"""
# =========================
# 初期形状
# =========================

node0 = np.array([
    [0.0,0.0,0.0],
    [1.0,0.0,0.0],
    [1.0,1.0,0.0],
    [0.5,0.5,0.7071],
    [0.5,0.0,0.0]
], dtype=np.float32)

member = np.array([
    [0,4],
    [4,1],
    [1,2],
    [2,3],
    [3,0],
    [2,0],
    [2,4]
], dtype=int)

panel = np.array([
    [0,2,4],
    [1,2,4],
    [0,2,3]
], dtype=int)


fix_crease = np.array([
    [2,4]
], dtype=int)


fix_crease = np.array([
], dtype=int)

# ==========================================
# boundary condition
# ==========================================

fix = np.array([
    [0,0],
    [0,1],
    [0,2],
    [4,0],
    [4,1],
    [4,2],
    [2,0],
    [2,1],
    [2,2]
])

fix=np.array([], dtype=int)

#非適合量
def incompatibility(node, member, L0):
    g = np.zeros(len(member))
    for i, (i0, i1) in enumerate(member):
        diff = node[i0] - node[i1]
        g[i] = 0.5 * (np.dot(diff, diff) - L0[i]**2)
    return g

#モードの計算
def index_free(num_node, fix):
    idx = np.arange(num_node * 3)
    return np.delete(idx, 3 * fix[:, 0] + fix[:, 1])

def dedx(node, num_node, member, num_member, index_free):
    dedx_all = np.zeros((num_member, num_node * 3))
    node_diff = node[member[:, 0]] - node[member[:, 1]]

    for i in range(num_member):
        i0, i1 = member[i]
        dedx_all[i, i0*3:(i0+1)*3] = node_diff[i]
        dedx_all[i, i1*3:(i1+1)*3] = -node_diff[i]

    return dedx_all[:, index_free]

def mode(index_free, dedx, num_node, tol=1e-8):
    _, sigma, vh = np.linalg.svd(dedx)
    rank = np.count_nonzero(sigma > tol)
    dof = dedx.shape[1] - rank

    displace_mode = np.zeros((num_node * 3, dof))
    displace_mode[index_free, :] = vh[rank:].T
    return displace_mode

"""
# =========================
# 初期長さ
# =========================

L0 = np.linalg.norm(
    node0[member[:,0]] - node0[member[:,1]],
    axis=1
)

# ==========================================
# free dof index
# ==========================================

def index_free(num_node, fix):

    all_dof = np.arange(num_node * 3)

    # 境界条件なし
    if fix is None or len(fix) == 0:
        return all_dof

    fixed_dof = 3 * fix[:,0] + fix[:,1]

    return np.setdiff1d(all_dof, fixed_dof)


# ==========================================
# face normal
# ==========================================

def face_normal(node, tri):

    i,j,k = tri

    v1 = node[j] - node[i]
    v2 = node[k] - node[i]

    n = torch.cross(v1, v2, dim=0)

    eps = 1e-12

    n = n / (torch.norm(n) + eps)

    return n


# ==========================================
# adjacent faces
# ==========================================

def find_adjacent_faces(panel, edge):

    i,j = edge

    faces = []

    for f, tri in enumerate(panel):

        if i in tri and j in tri:
            faces.append(f)

    return faces


# ==========================================
# signed dihedral angle
# ==========================================

def dihedral_angle(node, tri1, tri2, edge):

    i,j = edge

    n1 = face_normal(node, tri1)
    n2 = face_normal(node, tri2)

    e = node[j] - node[i]
    e = e / (torch.norm(e) + 1e-12)

    cos_theta = torch.dot(n1, n2)

    sin_theta = torch.dot(
        torch.cross(n1, n2, dim=0),
        e
    )

    theta = torch.atan2(sin_theta, cos_theta)

    return theta


# ==========================================
# initial edge length
# ==========================================

def initial_length(node0, member):

    return np.linalg.norm(
        node0[member[:,0]] - node0[member[:,1]],
        axis=1
    )


# ==========================================
# initial dihedral
# ==========================================

def initial_dihedral(node0, panel, fix_crease):

    # fix_creaseなし
    if fix_crease is None or len(fix_crease) == 0:
        return np.array([])

    theta0 = []

    node0_tensor = torch.tensor(node0)

    for edge in fix_crease:

        faces = find_adjacent_faces(panel, edge)

        # 隣接面が2枚ない
        if len(faces) != 2:
            continue

        f1, f2 = faces

        th = dihedral_angle(
            node0_tensor,
            panel[f1],
            panel[f2],
            edge
        )

        theta0.append(th.item())

    return np.array(theta0)


# ==========================================
# incompatibility
# ==========================================

def incompatibility(
    node,
    member,
    panel,
    fix_crease,
    L0,
    theta0
):

    g = []

    # --------------------------------------
    # edge length constraints
    # --------------------------------------

    for e, (i,j) in enumerate(member):

        diff = node[i] - node[j]

        val = 0.5 * (
            torch.dot(diff, diff)
            - L0[e]**2
        )

        g.append(val)

    # --------------------------------------
    # rigid crease constraints
    # --------------------------------------

    if fix_crease is not None and len(fix_crease) > 0:

        c_valid = 0

        for edge in fix_crease:

            faces = find_adjacent_faces(panel, edge)

            if len(faces) != 2:
                continue

            f1, f2 = faces

            theta = dihedral_angle(
                node,
                panel[f1],
                panel[f2],
                edge
            )

            val = theta - theta0[c_valid]

            g.append(val)

            c_valid += 1

    return torch.stack(g)


# ==========================================
# Jacobian
# ==========================================

def jacobian_autograd(
    node,
    member,
    panel,
    fix_crease,
    L0,
    theta0,
    index_free
):

    g = incompatibility(
        node,
        member,
        panel,
        fix_crease,
        L0,
        theta0
    )

    m = len(g)

    ndof_free = len(index_free)

    J = torch.zeros((m, ndof_free))

    for i in range(m):

        if node.grad is not None:
            node.grad.zero_()

        g[i].backward(retain_graph=True)

        grad_full = node.grad.reshape(-1)

        J[i,:] = grad_full[index_free]

        print(f"return J[{i}]")

    return g.detach(), J.detach()


# ==========================================
# mechanism mode
# ==========================================

def mode(J, num_node, index_free, tol=1e-8):

    U, S, Vh = torch.linalg.svd(J)

    rank = torch.sum(S > tol).item()

    dof = J.shape[1] - rank

    disp_mode = np.zeros((num_node * 3, dof))

    if dof > 0:
        disp_mode[index_free,:] = (
            Vh[rank:].T.detach().numpy()
        )

    return disp_mode, S,dof


# ==========================================
# free dof
# ==========================================

idx_free = index_free(len(node0), fix)

# ==========================================
# initial values
# ==========================================

L0 = initial_length(node0, member)

theta0 = initial_dihedral(
    node0,
    panel,
    fix_crease
)
print("finish initial dihedral")
# ==========================================
# node tensor
# ==========================================

node_tensor = torch.tensor(
    node0,
    dtype=torch.float32,
    requires_grad=True
)
print("finish node tensor")
# ==========================================
# Jacobian
# ==========================================

g, J = jacobian_autograd(
    node_tensor,
    member,
    panel,
    fix_crease,
    L0,
    theta0,
    idx_free
)
print("finish Jacobian")
# ==========================================
# mechanism
# ==========================================

disp_mode, S, dof = mode(
    J,
    len(node0),
    idx_free
)
print("finish mechanism")
print(dof)
print(disp_mode.shape)
