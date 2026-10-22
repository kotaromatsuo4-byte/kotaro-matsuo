import numpy as np
import os
from cmaes import CMA


# input

node = np.loadtxt("node_initial.dat", delimiter=',', dtype=float)
member = np.loadtxt("member.dat", delimiter=',', dtype=int)
fix = np.loadtxt("fix4.dat", delimiter=',', dtype=int)

# save
file_dir = "history4"
os.makedirs(file_dir, exist_ok=True)
filepath = os.path.join(file_dir, f"{0}.dat")
np.savetxt(filepath, node, fmt="%.6f")

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


# flatness
def compute_flatness_z(node):
    z_values = node[:, 2]
    z_max = np.max(z_values)
    z_min = np.min(z_values)

    return z_max - z_min


# initial
index_free = get_index_free(num_node, fix)
L0 = np.linalg.norm(node[member[:, 0]] - node[member[:, 1]],axis=1)



# node position update
max_steps = 100
flattness_tol = 0.05
for step in range(max_steps):

    print("step =", step)

    # dedx
    dedx_matrix = calc_dedx(node,num_node,member,num_member,index_free)

    # mode
    displace_mode = calc_mode(index_free,dedx_matrix,num_node,tol=1e-8)

    num_mode = displace_mode.shape[1]

    print("num_mode =", num_mode)

    if num_mode == 0:
        print("mechanism mode not found")
        break

    # target norm
    target_norm_max = 0.5
    target_norm = target_norm_max * np.exp(-step / max_steps)
    
    # optimize A by CMA-ES
    optimizer = CMA(mean=np.zeros(num_mode),sigma=0.3)

    max_cma_iter = 200

    best_flatness = 1e10
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
            flatness = compute_flatness_z(node_corr)

            # CMAES update
            solutions.append((A_raw, flatness))

            # best update
            if flatness < best_flatness:
                best_flatness = flatness
                best_node = node_corr.copy()

        optimizer.tell(solutions)

        print(generation, best_flatness)

        if optimizer.should_stop():
            break
    node = best_node
    
    # save
    filepath = os.path.join(file_dir, f"{step+1}.dat")
    np.savetxt(filepath, node, fmt="%.6f")
    
    # convergence check
    flatness = compute_flatness_z(node)
    if flatness < flattness_tol:
        print("flatness converged:", flatness)
        break

    z_values = node[:, 2]
    z_min = np.min(z_values)
    if z_min < -20.0:
        print("negative z value detected, stopping optimization.")
        break