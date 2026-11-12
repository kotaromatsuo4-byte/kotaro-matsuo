import numpy as np
import os
import json
from datetime import datetime, timezone, timedelta
from cmaes import CMA
from module import mechanics, collision, SVD

# ==========================
# functions
# ==========================

# 重力ポテンシャルを計量とする
def compute_gravity_potential(node, *, reference_height):
    node = np.asarray(node, dtype=float)
    return float(np.sum(np.abs(node[:, 2] - reference_height)))


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
root = os.path.join(base, "result", MODEL_NAME, "every_step_opt", timestamp)

# optimization parameters
max_cma_iter = 200
cma_sigma = 0.3
coefficient_tol = 1e-12

# step
max_steps = 100
flatness_tol = 0.05
step_scale_max = 0.1
reference_height = 0.0

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
reference_height_original = reference_height
reference_height = reference_height / model_scale

# save settings
os.makedirs(root, exist_ok=True)
config = dict(model=MODEL_NAME, model_folder=model_folder, dimension=DIMENSION, backend=backend, seed=SEED, timestamp=timestamp)
config["optimization"] = dict(max_cma_iter=max_cma_iter, cma_sigma=cma_sigma, coefficient_tol=coefficient_tol)
config["step"] = dict(max_steps=max_steps, flatness_tol=flatness_tol, step_scale_max=step_scale_max, reference_height=reference_height)
config["normalization"] = dict(method="initial mean member length = 1", model_scale=model_scale, calculation_coordinates="normalized", saved_coordinates="original", reference_height_original=reference_height_original)
config["svd"], config["corrector"], config["collision"] = svd_options, corrector_options, collision_options
config["objective"] = "minimize sum(abs(z-reference_height)) over corrected candidates; success uses flatness_tol on squared distances to the least-squares line (2D) or plane (3D)"
config["optimization"]["variables"] = "mode coefficients and ro; 0 < ro < step_scale_max; initial ro mean = step_scale_max / 2; initial ro std = cma_sigma * step_scale_max"
config["save"] = dict(fmt=save_fmt, delimiter=" ", step0="original model", step1="first accepted optimized state")
with open(os.path.join(root, "config.json"), "w", encoding="utf-8") as file:
    json.dump(config, file, ensure_ascii=False, indent=2)

# initial
index_free = mechanics.index_free(num_node, fix)
L0 = mechanics.member_lengths(node, member)
node_initial = node.copy()
rng = np.random.default_rng(SEED)
checker = collision.CollisionChecker(member, node_initial, dimension=DIMENSION, **collision_options)
success = False

# main
node = node_initial.copy()
checker.reset(node)
path = [node.copy()]

# node position update
for step in range(max_steps):
    J = mechanics.jacobian(node, member, backend=backend)
    sigma, displace_mode = SVD.modes(J, index_free, backend=backend, **svd_options)
    num_mode = displace_mode.shape[1]
    if num_mode == 0:
        print(f"no modes at step {step + 1}")
        break    

    # 方向と幅を同時に最適化。幅の初期分布は上限に合わせて尺度をそろえる。
    mean = np.zeros(num_mode + 1)
    mean[-1] = step_scale_max / 2
    cov = np.eye(num_mode + 1)
    cov[-1, -1] = step_scale_max**2
    bounds = np.tile([-np.inf, np.inf], (num_mode + 1, 1))
    bounds[-1] = [np.nextafter(0.0, step_scale_max), np.nextafter(step_scale_max, 0.0)]
    optimizer = CMA(mean=mean, sigma=cma_sigma, bounds=bounds, cov=cov, seed=int(rng.integers(np.iinfo(np.int32).max)))
    best_flatness = np.inf
    best_node = None

    for generation in range(max_cma_iter):
        solutions = []
        for _ in range(optimizer.population_size):
            design_vars = optimizer.ask()
            A_raw = design_vars[:-1]
            rho = design_vars[-1]
            
            # normalize
            A_norm = np.linalg.norm(A_raw)
            if A_norm < coefficient_tol:
                A = np.zeros_like(A_raw)
            else:
                A = A_raw / A_norm

            # predictor / corrector
            node_pred = mechanics.predictor(node, displace_mode, A, step_size=rho)
            node_corr = mechanics.corrector(node_pred, member, L0, index_free, **corrector_options)

            # 未収束候補は最良候補に含めず、CMA-ESには最悪の評価値を返す。
            if np.linalg.norm(mechanics.incompatibility(node_corr, member, L0)) > tolerance:
                solutions.append((design_vars, np.finfo(float).max))
                continue

            # objective
            flatness = compute_gravity_potential(node_corr, reference_height=reference_height)
            solutions.append((design_vars, flatness))

            # best update
            if flatness < best_flatness:
                best_flatness = flatness
                best_node = node_corr.copy()

        optimizer.tell(solutions)
        if optimizer.should_stop():
            break

    # 収束した候補がないstepは打ち切る。
    if best_node is None:
        print(f"no converged candidate at step {step + 1}")
        break

    # crossing check
    if checker.check(node, best_node):
        print(f"crossing detected at step {step + 1}")
        break
    
    checker.accept(node, best_node)
    node = best_node
    path.append(node.copy())

    # convergence check / save
    potential = mechanics.compute_flatness_potential(node, dimension=DIMENSION)
    if potential < flatness_tol:
        folder = os.path.join(root, "every_step_opt_success")
        os.makedirs(folder, exist_ok=True)
        for saved_step, saved_node in enumerate(path):
            np.savetxt(os.path.join(folder, f"step{saved_step}.dat"), saved_node * model_scale, fmt=save_fmt)
        print(f"flatness potential converged: {potential}")
        print(f"saved: {folder}")
        success = True
        break
    
if not success:
    print(f"No success. Settings saved: {root}")
