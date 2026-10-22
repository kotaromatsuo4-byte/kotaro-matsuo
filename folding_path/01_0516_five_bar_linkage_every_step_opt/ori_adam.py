import numpy as np
import torch
import torch.optim as optim
import os


# input

node = np.loadtxt("node_initial.dat", delimiter=',', dtype=float)
member = np.loadtxt("member.dat", delimiter=',', dtype=int)
fix = np.loadtxt("fix.dat", delimiter=',', dtype=int)

# save
file_dir = "history"
os.makedirs(file_dir, exist_ok=True)
filepath = os.path.join(file_dir, f"{0}.dat")
np.savetxt(filepath, node, fmt="%.6f")

num_node = node.shape[0]
num_member = member.shape[0]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    node_torch = torch.tensor(node,dtype=torch.float32,device=device)
    mode_torch = torch.tensor(displace_mode,dtype=torch.float32,device=device)
    displacement = mode_torch @ A
    displacement = displacement.reshape(-1, 3)

    return node_torch + displacement


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
def compute_flatness_z(node_torch):
    z_values = node_torch[:, 2]
    z_max = torch.max(z_values)
    z_min = torch.min(z_values)

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


    # multi-start optimize A

    num_starts = 10 + step//5

    best_loss = 1e10
    best_node_pred = None
    best_A = None

    target_norm_max = 0.5
    target_norm = target_norm_max * np.exp(-step / max_steps)

    tol = 1e-6
    max_iters = 5000

    for start in range(num_starts):

        # random initial value
        A_ = torch.randn(num_mode,dtype=torch.float32,device=device,requires_grad=True)
        optimizer = optim.Adam([A_],lr=1e-3)
        loss_old = 1e10

        # adam loop
        for it in range(max_iters):
            optimizer.zero_grad()
            # norm constraint
            A = (target_norm* A_/ (torch.norm(A_) + 1e-8))

            # predictor
            node_pred = predictor(node,displace_mode,A)

            # flatness
            loss = compute_flatness_z(node_pred)

            # backward
            loss.backward()

            optimizer.step()

            # convergence
            loss_new = loss.item()

            if abs(loss_old - loss_new) < tol:
                break

            loss_old = loss_new

        # best solution update
        final_loss = loss.item()

        if final_loss < best_loss:
            best_loss = final_loss
            best_A = A.detach().clone()
            best_node_pred = (node_pred.detach().cpu().numpy())

    node_pred_opt = best_node_pred

    # optimized node
    node_pred_opt = node_pred.detach().cpu().numpy()

    # corrector
    num_corrector_iter = 10
    for _ in range(num_corrector_iter):
        
        dc = corrector(node_pred_opt,num_node,member,num_member,L0,index_free,beta0=target_norm,n_iter=20,tau=0.5)
        node_pred_opt += dc

        if np.linalg.norm(dc) < 1e-6:
            node = node_pred_opt
            break
        
        node = node_pred_opt
    
    # save
    filepath = os.path.join(file_dir, f"{step+1}.dat")
    np.savetxt(filepath, node, fmt="%.6f")
    
    # convergence check
    flatness = compute_flatness_z(torch.tensor(node,dtype=torch.float32,device=device)).item()
    if flatness < flattness_tol:
        print("flatness converged:", flatness)
        break
    
    node_value = torch.tensor(node,dtype=torch.float32,device=device)
    z_values = node_value[:, 2]
    z_min = torch.min(z_values)
    if z_min < 0:
        print("negative z value detected, stopping optimization.")
        break