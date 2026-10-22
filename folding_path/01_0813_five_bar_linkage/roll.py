"""StateNodeを一つだけ進めるrandom rollとdeterministic roll。"""

from __future__ import annotations
import numpy as np
from collision import edge_edge_crossing_check as edge_edge_crossing_check_3d
from collision2 import edge_edge_crossing_check as edge_edge_crossing_check_planar
from mechanics import (
    compute_gravity_potential,
    corrector,
    deterministic_step,
    random_step,
    arbitary_step,
)
from state import StateNode

# 更新後の座標を判定し、親の直下に子StateNodeを一つ追加する
def _evaluate_and_add_child(
    history,
    parent_id,
    node_after,
    member,
    reference_point,
    flatness_tolerance,
    dist_tol,
    normal_tol,
    cos_tol,
    proposed_displacement=None,
    step_type=None,
    correction_success=True,
    collision_model="planar",
):

    parent = history[parent_id]
    node_before = parent.node

    if collision_model == "planar":
        collision_check = edge_edge_crossing_check_planar
    elif collision_model == "3d":
        collision_check = edge_edge_crossing_check_3d
    else:
        raise ValueError("collision_model must be 'planar' or '3d'")

    valid = collision_check(
        node_before,node_after,member,dist_tol,normal_tol,cos_tol
    ) # 0:衝突 1:有効
    potential = compute_gravity_potential(node_after, reference_point)

    if not correction_success:
        terminal_reason = "corrector_failure"
    elif valid == 0:
        terminal_reason = "collision"
    elif potential <= flatness_tolerance:
        terminal_reason = "flat"
    else:
        terminal_reason = None

    # historyの末尾に新しい子状態を追加し、親から子への枝も記録する。
    state_id = len(history)
    child = StateNode(
        state_id=state_id,
        node=node_after.copy(),
        parent_id=parent_id,
        children_ids=[],
        depth=parent.depth + 1,
        terminal_reason=terminal_reason,
        potential=float(potential),
        incoming_displacement=(
            None
            if proposed_displacement is None
            else np.asarray(proposed_displacement, dtype=float).copy()
        ),
        incoming_step_type=step_type,
    )
    history.append(child)
    parent.children_ids.append(state_id)
    return child

# ランダムなメカニズム配合率で1 step進め、子状態を返す。
def random_roll(history,parent_id,member,num_node,num_member,L0,index_free,random_scale,constraint_tolerance,beta0,n_iter,l_iter,tau,reference_point,flatness_tolerance,dist_tol,normal_tol,cos_tol,collision_model="planar"):

    parent = history[parent_id]

    node_before = parent.node
    node_after, displacement, success, _ = random_step(
        node_before,num_node,member,num_member,index_free,random_scale,L0,
        constraint_tolerance,beta0,n_iter,l_iter,tau,return_info=True,
    )

    return _evaluate_and_add_child(
        history,parent_id,node_after,member,reference_point,flatness_tolerance,
        dist_tol,normal_tol,cos_tol,displacement,"perturbation",success,
        collision_model,
    )

# 任意のメカニズム配合率で1 step進め、子状態を返す。
def arbitrary_roll(history,parent_id,member,num_node,num_member,L0,index_free,A,scale,constraint_tolerance,beta0,n_iter,l_iter,tau,reference_point,flatness_tolerance,dist_tol,normal_tol,cos_tol,collision_model="planar"):

    parent = history[parent_id]

    node_before = parent.node
    node_after, displacement, success, _ = arbitary_step(
        node_before,num_node,member,num_member,index_free,A,scale,L0,
        constraint_tolerance,beta0,n_iter,l_iter,tau,return_info=True,
    )

    return _evaluate_and_add_child(
        history,parent_id,node_after,member,reference_point,flatness_tolerance,
        dist_tol,normal_tol,cos_tol,displacement,"perturbation",success,
        collision_model,
    )


# NNが候補生成時に計算した物理的な節点変位を、そのまま用いる摂動step。
# modeを再計算しないため、NNへ入力した変位と実際に与える変位が一致する。
def displacement_roll(
    history,
    parent_id,
    displacement,
    member,
    num_node,
    num_member,
    L0,
    index_free,
    constraint_tolerance,
    beta0,
    n_iter,
    l_iter,
    tau,
    reference_point,
    flatness_tolerance,
    dist_tol,
    normal_tol,
    cos_tol,
    collision_model="planar",
):
    node_predicted = history[parent_id].node + np.asarray(displacement, dtype=float)
    node_after, success, _ = corrector(
        node_predicted,
        num_node,
        member,
        num_member,
        L0,
        index_free,
        constraint_tolerance,
        beta0,
        n_iter,
        l_iter,
        tau,
        return_info=True,
    )
    return _evaluate_and_add_child(
        history,
        parent_id,
        node_after,
        member,
        reference_point,
        flatness_tolerance,
        dist_tol,
        normal_tol,
        cos_tol,
        proposed_displacement=displacement,
        step_type="perturbation",
        correction_success=success,
        collision_model=collision_model,
    )

# 重力ポテンシャルを最小化する方向へ1 step進め、子状態を返す
def deterministic_roll(history,parent_id,member,num_node,num_member,L0,index_free,target_norm_max,constraint_tolerance,beta0,n_iter,l_iter,tau,cma_population,cma_generations,cma_seed,reference_point,flatness_tolerance,dist_tol,normal_tol,cos_tol,collision_model="planar"):

    parent = history[parent_id]

    node_before = parent.node

    initial_scale_ratio = 0.99
    initial_potential = history[0].potential
    potential_scale = initial_potential * np.sqrt((1.0 - initial_scale_ratio) / initial_scale_ratio)
    scale_denominator = parent.potential**2 + potential_scale**2
    if scale_denominator <= 1.0e-24:
        scale_ratio = 0.0
    else:
        scale_ratio = parent.potential**2 / scale_denominator
    scale = target_norm_max * scale_ratio
    
    node_after, displacement, success, _ = deterministic_step(
        node_before,member,L0,num_node,num_member,index_free,scale,
        constraint_tolerance,beta0,n_iter,l_iter,tau,cma_population,
        cma_generations,reference_point,cma_seed,return_info=True,
    )

    return _evaluate_and_add_child(
        history,parent_id,node_after,member,reference_point,flatness_tolerance,
        dist_tol,normal_tol,cos_tol,displacement,"deterministic",success,
        collision_model,
    )
