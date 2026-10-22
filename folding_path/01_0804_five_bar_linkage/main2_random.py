"""
main2_random : bootstrapの生成数をこちらで指定してその後はhistoryからランダムに摂動を与える
"""
from pathlib import Path
import os

import numpy as np

from state import create_history
from roll import random_roll, deterministic_roll
from archive import update_elite_archive
from result import reset_output_directory, save_elite_paths, save_discovery_curve, save_search_tree

base_directory = Path(__file__).resolve().parent
model_directory = base_directory / "model"

# input
node_initial = np.loadtxt(model_directory / "node_initial.dat", delimiter=',', dtype=float)
member = np.loadtxt(model_directory / "member.dat", delimiter=',', dtype=int)
fix = np.loadtxt(model_directory / "fix.dat", delimiter=',', dtype=int)
num_node = node_initial.shape[0]
num_member = member.shape[0]

# free index
idx = np.arange(num_node * 3)
index_free = np.delete(idx, 3 * fix[:, 0] + fix[:, 1])

#initial member length
L0 = np.linalg.norm(node_initial[member[:, 0]] - node_initial[member[:, 1]],axis=1)

# initial potential
reference_point = 0.0
z_values = node_initial[:, 2]
initial_potential = np.sum(np.abs(z_values - reference_point))

# root state
history = create_history(node_initial, initial_potential)

"""
hiperparameters
"""
# 1stepのパラメータ
constraint_tolerance = 1.0e-10
beta0 = 1.0
n_iter = 10
l_iter = 10
tau = 0.5
cma_population = 6
cma_generations = 10
cma_seed = 0

# rollのパラメータ
target_norm_max = 0.10
random_scale = 0.20
max_rollout_steps = 100 # random stepを含む1rolloutの最大step数。random stepが挟まるとリセットされる。

# 衝突判定の閾値
flatness_tolerance = 0.05
dist_tol = 0.05
normal_tol = 0.1
cos_tol = 0.99

# archiveのパラメータ
length_scale = np.mean(L0)
layer_threshold = 0.1
overlap_tolerance = 0.05
separation_tolerance = 1.0e-5
collinearity_tolerance = 0.02
parallel_tolerance = 0.02
rms_tolerance = 0.02
max_tolerance = 0.05

# 結果図で途中状態をまとめるcellのパラメータ
state_cell_rms_tolerance = 0.05
state_cell_max_tolerance = 0.10

# bootstrapのパラメータ
num_bootstrap = 3

# 探索のパラメータ
random_seed = int(os.environ.get("FOLDING_SEED", "11"))
total_step_budget = int(os.environ.get("FOLDING_STEP_BUDGET", "5000"))
np.random.seed(random_seed) # random step用
selection_rng = np.random.default_rng(random_seed + 1) # 摂動位置選択専用乱数

"""
main loop
"""
used_steps = 0
elite_archive = []
discovery_curve = [(0, 0)]
random_edges = []


# bootstrap
for i in range(num_bootstrap):
    
    if used_steps >= total_step_budget:
        break
    
    rollout_steps = 0

    # 最初はrandom roll。
    parent_id = 0
    current = random_roll(history,parent_id,member,num_node,num_member,L0,index_free,random_scale,constraint_tolerance,beta0,n_iter,l_iter,tau,reference_point,flatness_tolerance,dist_tol,normal_tol,cos_tol)
    random_edges.append((parent_id, current.state_id))
    used_steps += 1
    rollout_steps += 1
        
    # 平坦になるか衝突するかループ数が上限に達するまでdeterministic rollを繰り返す。
    while (
        current.terminal_reason is None
        and rollout_steps < max_rollout_steps
        and used_steps < total_step_budget
    ):
        parent_id=current.state_id # 1つ前がparent
        current = deterministic_roll(history,parent_id,member,num_node,num_member,L0,index_free,target_norm_max,constraint_tolerance,beta0,n_iter,l_iter,tau,cma_population,cma_generations,cma_seed,reference_point,flatness_tolerance,dist_tol,normal_tol,cos_tol)
        used_steps += 1
        rollout_steps += 1
        
        if used_steps % 50 == 0:
            print("used steps:", used_steps)
            print("number of cells:", len(elite_archive))
            
    if current.terminal_reason == "flat":

        is_new_cell = update_elite_archive(
            elite_archive,
            history,
            current.state_id,
            member,
            length_scale,
            layer_threshold,
            overlap_tolerance,
            separation_tolerance,
            collinearity_tolerance,
            parallel_tolerance,
            rms_tolerance,
            max_tolerance,
        )

        if is_new_cell:
            discovery_curve.append((used_steps, len(elite_archive)))
    
bootstrap_used_steps = used_steps # bootstrapで使ったステップ数を記録        

# 探索
while used_steps < total_step_budget:
    
    # root・flat・collisionを除いた途中状態から摂動付与位置候補を構成
    candidate_state_ids = [state.state_id for state in history if (state.state_id != 0 and state.terminal_reason is None)]
    
    if len(candidate_state_ids) == 0: # bootstrapがすべてrandom step直後に衝突する場合などは、候補が空になりうるので探索を終了する。
        break
    
    # history内のStateNodeを一様ランダム選択
    parent_id = int(selection_rng.choice(candidate_state_ids))
    
    # 1回random roll。
    rollout_steps = 0 #rolloutのステップ数をリセット
    current = random_roll(history,parent_id,member,num_node,num_member,L0,index_free,random_scale,constraint_tolerance,beta0,n_iter,l_iter,tau,reference_point,flatness_tolerance,dist_tol,normal_tol,cos_tol)
    random_edges.append((parent_id, current.state_id))
    used_steps += 1
    rollout_steps += 1
    
    # 平坦になるか衝突するかループ数が上限に達するまでdeterministic rollを繰り返す。
    while (
        current.terminal_reason is None
        and rollout_steps < max_rollout_steps
        and used_steps < total_step_budget
    ):
        parent_id=current.state_id # 1つ前がparent
        current = deterministic_roll(history,parent_id,member,num_node,num_member,L0,index_free,target_norm_max,constraint_tolerance,beta0,n_iter,l_iter,tau,cma_population,cma_generations,cma_seed,reference_point,flatness_tolerance,dist_tol,normal_tol,cos_tol)
        used_steps += 1
        rollout_steps += 1
        
        if used_steps % 50 == 0:
            print("used steps:", used_steps)
            print("number of cells:", len(elite_archive))
    
    if current.terminal_reason == "flat":

        is_new_cell = update_elite_archive(
            elite_archive,
            history,
            current.state_id,
            member,
            length_scale,
            layer_threshold,
            overlap_tolerance,
            separation_tolerance,
            collinearity_tolerance,
            parallel_tolerance,
            rms_tolerance,
            max_tolerance,
        )

        if is_new_cell:
            discovery_curve.append((used_steps, len(elite_archive)))
            

"""
saving results
"""

# discovery curveを総予算まで延ばす
if discovery_curve[-1][0] != used_steps:
    discovery_curve.append((used_steps, len(elite_archive)))
print("bootstrap used steps:", bootstrap_used_steps)
# seedごとに結果を分けて保存する
output_directory = base_directory / "result" / "main2_random" / f"seed_{random_seed:03d}"
reset_output_directory(output_directory)
save_elite_paths(history, elite_archive, output_directory)
save_discovery_curve(discovery_curve, output_directory)
save_search_tree(
    history=history,
    elite_archive=elite_archive,
    random_edges=random_edges,
    output_directory=output_directory,
    member=member,
    reference_point=reference_point,
    length_scale=length_scale,
    state_rms_tolerance=state_cell_rms_tolerance,
    state_max_tolerance=state_cell_max_tolerance,
    layer_threshold=layer_threshold,
    overlap_tolerance=overlap_tolerance,
    separation_tolerance=separation_tolerance,
    collinearity_tolerance=collinearity_tolerance,
    parallel_tolerance=parallel_tolerance,
    elite_rms_tolerance=rms_tolerance,
    elite_max_tolerance=max_tolerance,
)
