import numpy as np

# input
node_initial = np.loadtxt("node_initial.dat", delimiter=',', dtype=float)
node_target = np.loadtxt("node_target.dat", delimiter=',', dtype=float)
member = np.loadtxt("member.dat", delimiter=',', dtype=int)
fix = np.loadtxt("fix.dat", delimiter=',', dtype=int)

num_node = node_initial.shape[0]
num_member = member.shape[0]

#initial member length
L0 = np.linalg.norm(node_initial[member[:, 0]] - node_initial[member[:, 1]],axis=1)

# free index
idx = np.arange(num_node * 3)
index_free = np.delete(idx, 3 * fix[:, 0] + fix[:, 1])


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

def predictor(node, displace_mode, node_target,beta0,tau,l_iter):
    grad = node_target - node #(n,3)
    grad = grad.flatten()  #(n*3,)
    deltaX = displace_mode @ (displace_mode.T @ grad)

    norm_delta = np.linalg.norm(deltaX)
    if norm_delta > 1e-12:
        deltaX /= norm_delta

    displacement = deltaX.reshape(-1, 3)
    beta = beta0
    accepted = False
    node_new = node.copy()
    residual = np.linalg.norm(node_new - node_target)
    for _ in range(l_iter):
        node_trial = node + beta * displacement
        error = np.linalg.norm(node_trial - node_target)

        if error < (1 - 1e-4 * beta) * residual:
            node_new = node_trial
            accepted = True
            break

        beta *= tau

    if not accepted:
        return node.copy()

    return node_new

def incompatibility(node, member, L0):
    g = np.zeros(len(member))
    for i, (i0, i1) in enumerate(member):
        diff = node[i0] - node[i1]
        g[i] = 0.5 * (np.dot(diff, diff) - L0[i]**2)
    return g

def corrector(node,num_node,member,num_member,L0,index_free,constraint_tolerance,beta0,n_iter,l_iter,tau):
    node_new = node.copy()
    for _ in range(n_iter):
        g = incompatibility(node_new, member, L0)
        residual = np.linalg.norm(g)
        if residual < constraint_tolerance:
            break
        gamma = calc_dedx(node_new,num_node,member,num_member,index_free)
        dd_free = -np.linalg.pinv(gamma, rcond=1.0e-8) @ g
        dd = np.zeros(num_node * 3)
        dd[index_free] = dd_free
        beta = beta0
        accepted = False
        for _ in range(l_iter):
            node_trial = node_new + beta * dd.reshape(-1, 3)
            g_new = incompatibility(node_trial, member, L0)
            if np.linalg.norm(g_new) < (1 - 1e-4 * beta) * residual:
                node_new = node_trial
                accepted = True
                break
            beta *= tau
        if not accepted:
            break
    return node_new


def step(node,num_node,member,num_member,index_free,node_target,constraint_tolerance,beta0,n_iter,l_iter,tau):
    dedx = calc_dedx(node, num_node, member, num_member, index_free)
    displace_mode = calc_mode(index_free, dedx, num_node)
    node_new = predictor(node, displace_mode, node_target,beta0,tau,l_iter)
    node_corr = corrector(node_new,num_node,member,num_member,L0,index_free,constraint_tolerance,beta0,n_iter,l_iter,tau)

    return node_corr

