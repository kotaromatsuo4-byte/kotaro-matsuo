import numpy as np
import math
from cmaes import CMA

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

def predictor(node, displace_mode, A):
    displacement = displace_mode @ A
    displacement = displacement.reshape(-1, 3)
    return node + displacement

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

def compute_gravity_potential(node,reference_point):
    z_values = node[:, 2]
    potential = np.sum(np.abs(z_values - reference_point))
    return potential

def random_step(node,num_node,member,num_member,index_free,scale,L0,constraint_tolerance,beta0,n_iter,l_iter,tau):
    dedx = calc_dedx(node, num_node, member, num_member, index_free)
    displace_mode = calc_mode(index_free, dedx, num_node)
    A = np.random.randn(displace_mode.shape[1])
    A = scale * A / (np.linalg.norm(A)+ 1e-8)
    randomised_node = predictor(node, displace_mode, A)
    node_corr = corrector(randomised_node,num_node,member,num_member,L0,index_free,constraint_tolerance,beta0,n_iter,l_iter,tau)

    return node_corr

def arbitary_step(node,num_node,member,num_member,index_free,A,scale,L0,constraint_tolerance,beta0,n_iter,l_iter,tau):
    dedx = calc_dedx(node, num_node, member, num_member, index_free)
    displace_mode = calc_mode(index_free, dedx, num_node)
    A = scale * A / (np.linalg.norm(A)+ 1e-8)
    randomised_node = predictor(node, displace_mode, A)
    node_corr = corrector(randomised_node,num_node,member,num_member,L0,index_free,constraint_tolerance,beta0,n_iter,l_iter,tau)

    return node_corr

def deterministic_step(node,member,L0,num_node,num_member,index_free,step_norm,constraint_tolerance,beta0,n_iter,l_iter,tau,cma_population,cma_generations,reference_point,cma_seed):
    dedx = calc_dedx(node,num_node,member,num_member,index_free)
    displace_mode = calc_mode(index_free,dedx,num_node,tol=1e-8)
    num_mode = displace_mode.shape[1]
    
    optimizer = CMA(
        mean=np.zeros(num_mode),
        sigma=0.3,
        population_size=cma_population,
        seed=cma_seed,
    )
    
    best_flatness = math.inf
    best_node = None
    for _ in range(cma_generations):
        solutions = []
        for _ in range(optimizer.population_size):
            A_raw = optimizer.ask()

            # normalize
            A_norm = np.linalg.norm(A_raw)
            if A_norm < 1e-12:
                A = np.zeros_like(A_raw)
            else:
                A = (step_norm* A_raw/ A_norm)
            
            # predictor
            node_pred = predictor(node,displace_mode,A)        
            # corrector
            node_corr = corrector(node_pred,num_node,member,num_member,L0,index_free,constraint_tolerance,beta0,n_iter,l_iter,tau)

            # objective
            flatness = compute_gravity_potential(node_corr, reference_point)

            # CMAES update
            solutions.append((A_raw, flatness))

            # best update
            if flatness < best_flatness:
                best_flatness = flatness
                best_node = node_corr.copy()

        optimizer.tell(solutions)

        if optimizer.should_stop():
            break
        
    return best_node
