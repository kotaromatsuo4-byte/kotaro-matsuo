import numpy as np
import os
import json
from datetime import datetime, timezone, timedelta
from module import mechanics, collision, SVD

# ==========================
# functions
# ==========================

# 計算済みモードから方向・幅をランダムに選び、修正後の節点座標(num_node, 3)：x, y, zを返す。
def randomise_step(node, member, L0, index_free, displace_mode, random_scale_max, rng):
    A = rng.standard_normal(displace_mode.shape[1])
    A = A / (np.linalg.norm(A) + normalization_eps)
    step_size = rng.uniform(0, random_scale_max)
    node_pred = mechanics.predictor(node, displace_mode, A, step_size=step_size)
    node_corr = mechanics.corrector(node_pred, member, L0, index_free, **corrector_options)
    return node_corr


# ==========================
# settings
# ==========================

# main parameters
MODEL_NAME = "five_bar_linkage"
DIMENSION = 2 # 2(2D・最小二乗直線) or 3(3D・最小二乗平面)
backend = "dense" # "dense" or "sparse"
SEED = 1000

# folder
base = os.path.dirname(os.path.abspath(__file__))
model_folder = os.path.join(base, "model", MODEL_NAME)
timestamp = datetime.now(timezone(timedelta(hours=9))).strftime("%Y%m%d_%H%M%S_%f")
root = os.path.join(base, "result", MODEL_NAME, "random", timestamp)

# step
max_steps = 100
flatness_tol = 0.05
random_scale_max = 0.1
num_trials = 10
normalization_eps = 1e-12

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

# corrector parameters
tolerance = 1e-6
max_iterations = 100
max_backtracks = 10
initial_step_size = 1.0
backtrack_shrink = 0.5
linear_atol = 1e-8
linear_btol = 1e-8
linear_rcond = 1e-8
linear_maxiter = 1000
linear_conlim = 1e8

corrector_options = dict(backend=backend, tolerance=tolerance, max_iterations=max_iterations, max_backtracks=max_backtracks, initial_step_size=initial_step_size, backtrack_shrink=backtrack_shrink, linear_atol=linear_atol, linear_btol=linear_btol, linear_rcond=linear_rcond, linear_maxiter=linear_maxiter, linear_conlim=linear_conlim)

# crossing parameters
crossing_tol = 1e-6
angle_tol = 1e-10
parameter_tol = 1e-9
coefficient_root_tol = 1e-12
root_imag_tol = 1e-8
time_tol = 1e-10

collision_options = dict(distance_tolerance=crossing_tol, angle_tolerance=angle_tol, parameter_tolerance=parameter_tol, coefficient_tolerance=coefficient_root_tol, root_imag_tolerance=root_imag_tol, time_tolerance=time_tol)

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

# normalization
model_scale = float(np.mean(mechanics.member_lengths(node, member)))
node = node / model_scale

# save settings
os.makedirs(root, exist_ok=True)
config = dict(model=MODEL_NAME, model_folder=model_folder, dimension=DIMENSION, backend=backend, seed=SEED, timestamp=timestamp)
config["step"] = dict(max_steps=max_steps, flatness_tol=flatness_tol, random_scale_max=random_scale_max, num_trials=num_trials, normalization_eps=normalization_eps)
config["normalization"] = dict(method="initial mean member length = 1", model_scale=model_scale, calculation_coordinates="normalized", saved_coordinates="original")
config["svd"], config["corrector"], config["collision"] = svd_options, corrector_options, collision_options
config["objective"] = "sum of squared distances to the least-squares line (2D) or plane (3D) in normalized coordinates; success uses flatness_tol"
config["save"] = dict(fmt=save_fmt, delimiter=" ", step0="original model", step1="first accepted random step")
with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as file:
    json.dump(config, file, ensure_ascii=False, indent=2)

# initial
index_free = mechanics.index_free(num_node, fix)
L0 = mechanics.member_lengths(node, member)
node_initial = node.copy()
rng = np.random.default_rng(SEED)
checker = collision.CollisionChecker(member, node_initial, dimension=DIMENSION, **collision_options)
success = False

# main loop
for i in range(num_trials):
    node = node_initial.copy()
    checker.reset(node)
    path = [node.copy()]

    # node position update
    for step in range(max_steps):
        J = mechanics.jacobian(node, member, backend=backend)
        sigma, displace_mode = SVD.modes(J, index_free, backend=backend, **svd_options)
        num_mode = displace_mode.shape[1]
        if num_mode == 0:
            print(f"history[{i}]: no modes at step {step + 1}")
            break

        node_step = randomise_step(node, member, L0, index_free, displace_mode, random_scale_max, rng)
        
        # crossing check
        if np.linalg.norm(mechanics.incompatibility(node_step, member, L0)) > tolerance:
            print(f"history[{i}]: correction did not converge at step {step + 1}")
            break
        if checker.check(node, node_step):
            print(f"history[{i}]: crossing detected at step {step + 1}")
            break
        
        checker.accept(node, node_step)
        node = node_step
        path.append(node.copy())

        # convergence check / save
        potential = mechanics.compute_flatness_potential(node, dimension=DIMENSION)
        if potential < flatness_tol:
            folder = os.path.join(root, "random_success")
            os.makedirs(folder, exist_ok=True)
            for saved_step, saved_node in enumerate(path):
                np.savetxt(os.path.join(folder, f"step{saved_step}.dat"), saved_node * model_scale, fmt=save_fmt)
            print(f"history[{i}]: flatness potential converged: {potential}")
            print(f"saved: {folder}")
            success = True
            break

    if success:
        break

if not success:
    print(f"No success in {num_trials} trials. Settings saved: {root}")
