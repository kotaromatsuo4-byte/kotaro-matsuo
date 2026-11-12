import numpy as np
import os
from module import mechanics, SVD

# ==========================
# settings
# ==========================

# main parameters
MODEL_NAME = "five_bar_linkage"
backend = "dense" # "dense" or "sparse"

# folder
base = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(base, "model", MODEL_NAME)
root = os.path.join(base, "result", MODEL_NAME, "landscape")
os.makedirs(root, exist_ok=True)

# SVD parameters
tol = 1e-8
abs_sigma_tol = 1e-6
gap_ratio = 10.0
k_start = 6
k_step = 6
k_max = 60
sigma_floor = 1e-30
eigensolver_tol = 1e-10
eigensolver_maxiter = 1000

if backend == "dense":
    svd_options = dict(tol=tol)
else:
    svd_options = dict(abs_sigma_tol=abs_sigma_tol, gap_ratio=gap_ratio, k_start=k_start, k_step=k_step, k_max=k_max, sigma_floor=sigma_floor, eigensolver_tol=eigensolver_tol, eigensolver_maxiter=eigensolver_maxiter)

# save
save_fmt = "%.12e"


# ==========================
# main
# ==========================

# input
node = np.loadtxt(os.path.join(model_folder, "node_initial.dat"), delimiter=",", dtype=float)
member = np.loadtxt(os.path.join(model_folder, "member.dat"), delimiter=",", dtype=int)
fix = np.loadtxt(os.path.join(model_folder, "fix.dat"), delimiter=",", dtype=int)
num_node = node.shape[0]
num_member = member.shape[0]

# 初期平均部材長を1に正規化
model_scale = float(np.mean(mechanics.member_lengths(node, member)))
node = node / model_scale

# main
J = mechanics.jacobian(node, member, backend=backend)
index_free = mechanics.index_free(num_node, fix)
sigma, displace_mode = SVD.modes(J, index_free, backend=backend, **svd_options)
num_mode = displace_mode.shape[1]
