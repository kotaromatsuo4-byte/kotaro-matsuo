import numpy as np
import os
from cmaes import CMA

# folder
root = "randomize+gravity_potential_history2"
os.makedirs(root, exist_ok=True)

# input

node = np.loadtxt("node_initial.dat", delimiter=',', dtype=float)
member = np.loadtxt("member.dat", delimiter=',', dtype=int)
fix = np.loadtxt("fix4.dat", delimiter=',', dtype=int)



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

# gravity potential
def compute_gravity_potential(node):
    reference_point = -1.707107
    z_values = node[:, 2]
    potential = np.sum(np.abs(z_values - reference_point)) # Define the pseudo-gravity potential
    return potential

# randomize
def randomise_node(node, member,fix,scale):
    num_node = node.shape[0]
    index_free = get_index_free(num_node, fix)
    num_member = member.shape[0]
    dedx = calc_dedx(node, num_node, member, num_member, index_free)
    displace_mode = calc_mode(index_free, dedx, num_node)
    A = np.random.randn(displace_mode.shape[1])
    A = scale * A / (np.linalg.norm(A)+ 1e-8)
    node_pred = predictor(node, displace_mode, A)
    # corrector
    node_corr = node_pred.copy()

    num_corrector_iter = 20
    for _ in range(num_corrector_iter):
        dc = corrector(node_corr,num_node,member,num_member,L0,index_free,beta0=1.0,n_iter=20,tau=0.5)
        node_corr += dc
        if np.linalg.norm(dc) < 1e-6:
            break
    
    randomised_node = node_corr.copy()
    return randomised_node

# segment-segment distance
def segment_distance(p1, p2, q1, q2):

    u = p2 - p1
    v = q2 - q1
    w = p1 - q1

    a = np.dot(u, u)
    b = np.dot(u, v)
    c = np.dot(v, v)
    d = np.dot(u, w)
    e = np.dot(v, w)

    D = a * c - b * b

    eps = 1e-12

    sN = 0.0
    sD = D

    tN = 0.0
    tD = D

    if D < eps:

        sN = 0.0
        sD = 1.0
        tN = e
        tD = c

    else:

        sN = (b * e - c * d)
        tN = (a * e - b * d)

        if sN < 0.0:

            sN = 0.0
            tN = e
            tD = c

        elif sN > sD:

            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:

        tN = 0.0

        if -d < 0.0:
            sN = 0.0

        elif -d > a:
            sN = sD

        else:
            sN = -d
            sD = a

    elif tN > tD:

        tN = tD

        if (-d + b) < 0.0:
            sN = 0

        elif (-d + b) > a:
            sN = sD

        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < eps else sN / sD
    tc = 0.0 if abs(tN) < eps else tN / tD

    dP = w + sc * u - tc * v

    return np.linalg.norm(dP)

# edge-edge crossing check
def edge_edge_crossing_check(node0,node1,member,dist_tol,normal_tol,cos_tol):

    n_member = len(member)

    for i in range(n_member):

        ia, ib = member[i]

        set_i = set([ia, ib])

        for j in range(i + 1, n_member):

            ja, jb = member[j]

            set_j = set([ja, jb])

            shared = list(set_i & set_j)

            # CASE 1: non shared node
            if len(shared) == 0:

                p0a = node0[ia]
                p0b = node0[ib]
                q0a = node0[ja]
                q0b = node0[jb]

                p1a = node1[ia]
                p1b = node1[ib]
                q1a = node1[ja]
                q1b = node1[jb]

                dist0 = segment_distance(p0a, p0b,q0a, q0b)
                dist1 = segment_distance(p1a, p1b,q1a, q1b)

                if dist0 > dist_tol and dist1 > dist_tol:
                    continue

                u0 = p0b - p0a
                v0 = q0b - q0a
                n0 = np.cross(u0, v0)
                norm0 = np.linalg.norm(n0)
                if norm0 < normal_tol:
                    continue
                d0 = np.dot(q0a - p0a, n0)

                u1 = p1b - p1a
                v1 = q1b - q1a
                n1 = np.cross(u1, v1)
                norm1 = np.linalg.norm(n1)
                if norm1 < normal_tol:
                    continue
                d1 = np.dot(q1a - p1a, n1)

                d0 /=np.linalg.norm(n0)+1e-8
                d1 /=np.linalg.norm(n1)+1e-8
                if d0 * d1 < -1e-4: # non-shared crossing is strictly defined.
                    return 0

            # CASE 2: one shared node
            elif len(shared) == 1:

                s = shared[0]
                a = ib if ia == s else ia
                b = jb if ja == s else ja

                u0 = node0[a] - node0[s]
                v0 = node0[b] - node0[s]
                cos0= np.dot(u0, v0) / (np.linalg.norm(u0) * np.linalg.norm(v0) + 1e-8)

                u1 = node1[a] - node1[s]
                v1 = node1[b] - node1[s]
                cos1 = np.dot(u1, v1) / (np.linalg.norm(u1) * np.linalg.norm(v1) + 1e-8)

                near0 = cos0 > cos_tol
                near1 = cos1 > cos_tol

                if not (near0 or near1): # if both are not near, then it's a crossing
                    continue

                n0 = np.cross(u0, v0)
                n1 = np.cross(u1, v1)

                n0 /= np.linalg.norm(n0) + 1e-8
                n1 /= np.linalg.norm(n1) + 1e-8
                if np.dot(n0, n1) < -1e-2: # shared-node crossing is not strictly defined. Because things don't work out if you evaluate contact too strictly.
                    return 0

            # CASE 3: two shared nodes
            else:
                continue

    return 1


# initial
index_free = get_index_free(num_node, fix)
L0 = np.linalg.norm(node[member[:, 0]] - node[member[:, 1]],axis=1)

node_initial = node.copy()

for i in range(10):
    node=node_initial.copy()

    # randomize
    scale=1.0
    node = randomise_node(node, member, fix, scale)

    # save
    folder = os.path.join(root, f"randomize+gravity_potential_history[{i}]")
    os.makedirs(folder, exist_ok=True)
    filepath = os.path.join(folder, f"{0}.dat")
    np.savetxt(filepath, node, fmt="%.6f")

    # node position update
    max_steps = 50
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
                flatness = compute_gravity_potential(node_corr)

                # CMAES update
                solutions.append((A_raw, flatness))

                # best update
                if flatness < best_flatness:
                    best_flatness = flatness
                    best_node = node_corr.copy()

            optimizer.tell(solutions)

            if optimizer.should_stop():
                break
        node = best_node
        
        # save
        filepath = os.path.join(folder, f"{step+1}.dat")
        np.savetxt(filepath, node, fmt="%.6f")
        
        # convergence check
        potential = compute_gravity_potential(node)
        if potential < flatness_tol:
            print(f"history[{i}]: gravity potential converged:", potential)
            break

        # edge-edge crossing check
        node_prev_filepath = os.path.join(folder, f"{step}.dat")
        node_prev = np.loadtxt(node_prev_filepath)
        dist_tol = 0.05
        normal_tol = 0.1
        cos_tol = 0.99
        if edge_edge_crossing_check(node_prev, node, member, dist_tol, normal_tol, cos_tol) == 0:
            print(f"history[{i}]: edge-edge crossing detected at step {step+1}, stopping optimization.")
            break

        """    
        z_values = node[:, 2]
        z_min = np.min(z_values)
        if z_min < 0:
            print("negative z value detected, stopping optimization.")
            break
        """