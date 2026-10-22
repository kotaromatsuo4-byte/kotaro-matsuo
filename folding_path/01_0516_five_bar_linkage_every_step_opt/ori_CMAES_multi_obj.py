import numpy as np
import os
from cmaes import CMA

# folder
root = "multi_obj_history"
os.makedirs(root, exist_ok=True)

# input

node = np.loadtxt("node_initial.dat", delimiter=',', dtype=float)
member = np.loadtxt("member.dat", delimiter=',', dtype=int)
fix = np.loadtxt("fix.dat", delimiter=',', dtype=int)



num_node = node.shape[0]
num_member = member.shape[0]


# first_order_mechanism_mode
def get_index_free(num_node, fix):
    idx = np.arange(num_node * 3)
    return np.delete(idx, 3 * fix[:, 0] + fix[:, 1])

def calc_dedx(node, num_node, member, num_member, index_free):
    dedx_all = np.zeros((num_member, num_node * 3))
    node_diff = node[member[:, 0]] - node[member[:, 1]]
    for i in range(num_member):
        i0, i1 = member[i]
        dedx_all[i, i0*3:(i0+1)*3] = node_diff[i]
        dedx_all[i, i1*3:(i1+1)*3] = -node_diff[i]

    return dedx_all[:, index_free]

def calc_mode(index_free, dedx_matrix, num_node, tol=1e-8):
    _, sigma, vh = np.linalg.svd(dedx_matrix)
    rank = np.count_nonzero(sigma > tol)
    dof = dedx_matrix.shape[1] - rank
    displace_mode = np.zeros((num_node * 3, dof))
    displace_mode[index_free, :] = vh[rank:].T

    return displace_mode


# predictor
def predictor(node, displace_mode, A):
    displacement = displace_mode @ A
    displacement = displacement.reshape(-1, 3)
    return node + displacement


# incompatibility
def incompatibility(node, member, L0):
    g = np.zeros(len(member))
    for i, (i0, i1) in enumerate(member):
        diff = node[i0] - node[i1]
        g[i] = 0.5 * (np.dot(diff, diff) - L0[i]**2)

    return g

# corrector
def corrector(node,num_node,member,num_member,L0,index_free,beta0,n_iter=20,tau=0.5):

    g = incompatibility(node, member, L0)
    gamma = calc_dedx(node,num_node,member,num_member,index_free)
    dd_free = -np.linalg.pinv(gamma) @ g
    dd = np.zeros(num_node * 3)
    dd[index_free] = dd_free
    beta = beta0

    for _ in range(n_iter):
        node_new = node + beta * dd.reshape(-1, 3)
        g_new = incompatibility(node_new, member, L0)
        if np.linalg.norm(g_new) < (1 - 1e-4 * beta) * np.linalg.norm(g):
            break
        beta *= tau

    return beta * dd.reshape(-1, 3)


# x-compactness
def compute_x_compactness(node, fix):
    x_fixed_nodes = fix[fix[:, 1] == 0,0]
    x_free_nodes = np.setdiff1d(np.arange(node.shape[0]),x_fixed_nodes)
    x_values = node[x_free_nodes,0]

    x_max = np.max(x_values)
    x_min = np.min(x_values)

    x = x_max - x_min
    return x

# flatness
def compute_flatness_z(node):
    z_values = node[:, 2]
    z_max = np.max(z_values)
    z_min = np.min(z_values)

    z = z_max - z_min
    return z


# initial
index_free = get_index_free(num_node, fix)
L0 = np.linalg.norm(node[member[:, 0]] - node[member[:, 1]],axis=1)

node_initial = node.copy()
x_range = np.max(node_initial[:, 0]) - np.min(node_initial[:, 0])
z_range = np.max(node_initial[:, 2]) - np.min(node_initial[:, 2])
lamda = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
for i in range(len(lamda)):
    node=node_initial.copy()
    lamda1 = lamda[i]
    lamda2 = (1-lamda1**2)**0.5
    print(f"history[{i}]: lamda1: {lamda1:.2f}, lamda2: {lamda2:.2f}")

    # save
    folder = os.path.join(root, f"multi_obj_history[{i}]")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{0}.dat")
    np.savetxt(filepath, node, fmt="%.6f")

    # node position update
    max_steps = 100
    flatness_tol = 0.05
    for step in range(max_steps):
        # dedx
        dedx_matrix = calc_dedx(node,num_node,member,num_member,index_free)

        # mode
        displace_mode = calc_mode(index_free,dedx_matrix,num_node,tol=1e-8)

        num_mode = displace_mode.shape[1]

        if num_mode == 0:
            print("mechanism mode not found")
            break

        # target norm
        target_norm_max = 0.5
        target_norm = target_norm_max * np.exp(-step / max_steps)
        
        # optimize A by CMA-ES
        optimizer = CMA(mean=np.zeros(num_mode),sigma=0.3)

        max_cma_iter = 200

        best_compactness = 1e10
        best_node = None

        for generation in range(max_cma_iter):
            solutions = []
            for _ in range(optimizer.population_size):
                A_raw = optimizer.ask()

                # normalize
                A_norm = np.linalg.norm(A_raw)
                if A_norm < 1e-12:
                    A = np.zeros_like(A_raw)
                else:
                    A = (target_norm* A_raw/ A_norm)
                
                # predictor
                node_pred = predictor(node,displace_mode,A)
                
                # corrector
                node_corr = node_pred.copy()

                num_corrector_iter = 20
                for _ in range(num_corrector_iter):
                    dc = corrector(node_corr,num_node,member,num_member,L0,index_free,beta0=1.0,n_iter=20,tau=0.5)
                    node_corr += dc
                    if np.linalg.norm(dc) < 1e-6:
                        break

                # objective
                x = compute_x_compactness(node_corr, fix) / x_range
                z = compute_flatness_z(node_corr) / z_range
                compactness = lamda1 * x + lamda2 * z

                # CMAES update
                solutions.append((A_raw, compactness))

                # best update
                if compactness < best_compactness:
                    best_compactness = compactness
                    best_node = node_corr.copy()

            optimizer.tell(solutions)

            if optimizer.should_stop():
                break
        node = best_node
        
        # save
        filepath = os.path.join(folder, f"{step+1}.dat")
        np.savetxt(filepath, node, fmt="%.6f")
        
        # convergence check
        z = compute_flatness_z(node) / z_range
        if z < flatness_tol:
            print(f"history[{i}]: flatness converged:", z)
            break