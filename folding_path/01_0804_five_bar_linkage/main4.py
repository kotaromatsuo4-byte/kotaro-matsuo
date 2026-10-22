"""摂動回数とpotential進行度を層別したNN-guided tree探索。

この実装では、一回の候補評価を

    過去の非終端StateNodeを選ぶ
    -> 摂動を一回与える
    -> 以後は決定論的に終端までrolloutする

という単位に固定する。rollout中の全ての途中状態はhistoryに残るため、
後のbatchでその途中状態を選ぶと、tree上では二回以上の摂動を含む経路が
自然に生成される。摂動回数には上限を設けない。

StateNode自体は変更せず、rootから各状態までに含まれる摂動回数を
``perturbation_count[state_id]`` という外部dictで管理する。親候補は

    min(摂動回数, 4) x potential進行度6 bin

で層別し、各batchで最大24状態をround-robin抽出する。各親から24個の
摂動候補を作り、``random`` または ``hybrid`` selectorで4候補を実行する。
``hybrid`` は、予測新規cell獲得率、学習分布境界での情報獲得、完全random
を組み合わせる。NNの外挿をそのまま高得点にしないため、教師入力からの
距離で予測を減衰し、総step予算に合わせて予測rollout costでも割る。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import os
import time

import numpy as np

from archive import update_elite_archive, update_frontier_archive
from main3_nn import execute_candidate_rollout
from nn import (
    append_training_sample,
    candidate_support_distance,
    generate_candidates,
    predict_cost_ensemble,
    sample_frontier_parent_ids,
    score_candidates,
    select_random_candidates,
    train_cost_ensemble,
    train_ensemble,
)
from result import (
    reset_output_directory,
    save_discovery_curve,
    save_elite_paths,
    save_frontier_states,
    save_search_tree,
)
from state import create_history


def potential_progress_bin(potential, initial_potential, num_bins=6):
    """potential低下の進行度を ``0, ..., num_bins - 1`` に量子化する。

    rootを進行度0、potentialが0に近い状態を進行度1とする。摂動直後に
    rootよりpotentialが大きくなってもbin外へ出ないよう[0, 1]へclipする。
    """

    if num_bins <= 0:
        raise ValueError("num_bins must be positive")
    scale = max(float(initial_potential), 1.0e-12)
    progress = 1.0 - float(potential) / scale
    progress = float(np.clip(progress, 0.0, 1.0))
    return min(int(progress * num_bins), num_bins - 1)


def stratified_parent_ids(
    history,
    perturbation_count,
    initial_potential,
    max_parents,
    rng,
    num_progress_bins=6,
    max_count_bucket=4,
):
    """非終端historyを摂動回数×potential進行度で層別抽出する。

    各非空層から一状態ずつ取るround-robinを繰り返す。層の順番と層内の
    状態順はbatchごとに乱数化するため、特定のbucketだけを固定的に優先
    しない。rootはbootstrap専用とし、主探索からは除外する。
    """

    if max_parents <= 0:
        return [], {}

    groups = defaultdict(list)
    for state in history:
        if state.state_id == 0 or state.terminal_reason is not None:
            continue
        count = int(perturbation_count.get(state.state_id, 0))
        count_bucket = min(max(count, 0), max_count_bucket)
        progress_bin = potential_progress_bin(
            state.potential,
            initial_potential,
            num_progress_bins,
        )
        groups[(count_bucket, progress_bin)].append(state.state_id)

    if not groups:
        return [], {}

    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])

    offsets = {key: 0 for key in keys}
    selected = []
    while len(selected) < int(max_parents):
        added = False
        for key in keys:
            offset = offsets[key]
            if offset >= len(groups[key]):
                continue
            selected.append(int(groups[key][offset]))
            offsets[key] = offset + 1
            added = True
            if len(selected) >= int(max_parents):
                break
        if not added:
            break

    group_sizes = {key: len(values) for key, values in groups.items()}
    return selected, group_sizes


def _append_distinct_parent(order, candidates, selected, selected_parents):
    """順位列から、未選択かつ親も未選択の最上位候補を一つ加える。"""

    for index in order:
        parent_id = int(candidates[index]["parent_id"])
        if index in selected or parent_id in selected_parents:
            continue
        selected.append(int(index))
        selected_parents.add(parent_id)
        return True
    return False


def select_main4_candidates(
    candidates,
    new_probability,
    flat_probability,
    uncertainty,
    predicted_distance,
    support_distance,
    predicted_cost,
    training_support_radius,
    support_multiplier,
    num_guided,
    num_random,
    rng,
):
    """新規cell・情報獲得・randomを組み合わせて候補を選ぶ。

    ``P_new`` は教師分布から離れた外挿点でも大きくなり得る。そのため、
    教師近傍では1、遠方では0へ滑らかに減衰するevidence weightを掛ける。
    情報獲得scoreは、教師分布の中心でも極端な外側でもなく、support境界
    付近を優先する。両scoreを予測rollout step数で割ることで、同じ総step
    予算内での発見効率を評価する。
    """

    if len(candidates) == 0:
        return []

    radius = max(float(training_support_radius), 1.0e-12)
    scaled_distance = np.asarray(support_distance) / radius
    evidence = np.exp(
        -0.5 * (scaled_distance / max(float(support_multiplier), 1.0e-12)) ** 2
    )
    boundary_weight = np.exp(
        -0.5 * ((scaled_distance - support_multiplier) / 0.5) ** 2
    )
    cost = np.maximum(np.asarray(predicted_cost, dtype=float), 1.0)
    novelty_score = np.asarray(new_probability) * evidence / cost
    # collision/timeoutには連続descriptor教師がないため、その領域の回帰分散を
    # 情報価値と誤認しないよう、予測flat確率を掛ける。
    information_score = (
        np.asarray(flat_probability)
        * np.asarray(uncertainty)
        * boundary_weight
        / cost
    )

    novelty_order = np.argsort(-novelty_score, kind="stable").tolist()
    information_order = np.argsort(-information_score, kind="stable").tolist()
    selected_indices = []
    selected_parents = set()
    selected_kinds = []

    # guided枠をほぼ半分ずつ新規cell狙いとsupport境界の教師追加へ配る。
    num_novelty = (int(num_guided) + 1) // 2
    num_information = int(num_guided) - num_novelty
    for _ in range(num_novelty):
        if _append_distinct_parent(
            novelty_order,
            candidates,
            selected_indices,
            selected_parents,
        ):
            selected_kinds.append("guided_novelty")
    for _ in range(num_information):
        if _append_distinct_parent(
            information_order,
            candidates,
            selected_indices,
            selected_parents,
        ):
            selected_kinds.append("guided_information")

    # 親数が不足するときだけ、既選択親の別摂動を許してguided枠を埋める。
    combined_order = novelty_order + information_order
    while len(selected_indices) < int(num_guided):
        index = next(
            (i for i in combined_order if i not in selected_indices),
            None,
        )
        if index is None:
            break
        selected_indices.append(int(index))
        selected_kinds.append("guided_novelty")

    # random枠にも可能な限り別の親を割り当てる。
    remaining = [
        i for i in rng.permutation(len(candidates)).tolist()
        if i not in selected_indices
    ]
    random_indices = []
    for index in remaining:
        parent_id = int(candidates[index]["parent_id"])
        if parent_id in selected_parents:
            continue
        random_indices.append(int(index))
        selected_parents.add(parent_id)
        if len(random_indices) >= int(num_random):
            break
    for index in remaining:
        if len(random_indices) >= int(num_random):
            break
        if index not in random_indices:
            random_indices.append(int(index))

    selected = []
    for index, kind in zip(selected_indices, selected_kinds):
        candidate = candidates[index].copy()
        candidate.update(
            {
                "new_probability": float(new_probability[index]),
                "flat_probability": float(flat_probability[index]),
                "uncertainty": float(uncertainty[index]),
                "predicted_distance": float(predicted_distance[index]),
                "support_distance": float(support_distance[index]),
                "predicted_cost": float(cost[index]),
                "novelty_score": float(novelty_score[index]),
                "information_score": float(information_score[index]),
                "selection_type": kind,
            }
        )
        selected.append(candidate)

    for index in random_indices:
        candidate = candidates[index].copy()
        candidate.update(
            {
                "new_probability": float(new_probability[index]),
                "flat_probability": float(flat_probability[index]),
                "uncertainty": float(uncertainty[index]),
                "predicted_distance": float(predicted_distance[index]),
                "support_distance": float(support_distance[index]),
                "predicted_cost": float(cost[index]),
                "novelty_score": float(novelty_score[index]),
                "information_score": float(information_score[index]),
                "selection_type": "random",
            }
        )
        selected.append(candidate)

    return selected


def stop_reason_code(stop_reason):
    """診断ファイル用の停止理由codeを返す。"""

    return {
        None: -1,
        "flat": 0,
        "collision": 1,
        "timeout": 2,
    }.get(stop_reason, 3)


def selection_type_code(selection_type):
    """診断ファイル用の候補選択種別codeを返す。"""

    return {
        "bootstrap": 0,
        "initial_training": 1,
        "guided_novelty": 2,
        "guided_information": 3,
        "random": 4,
    }.get(selection_type, 5)


def validate_perturbation_counts(history, random_edges, perturbation_count):
    """保存した累積回数が祖先経路上のrandom edge数と一致するか確認する。"""

    random_edge_set = {(int(parent), int(child)) for parent, child in random_edges}
    for state in history:
        if state.parent_id is None:
            expected = 0
        else:
            expected = perturbation_count[state.parent_id] + int(
                (state.parent_id, state.state_id) in random_edge_set
            )
        if perturbation_count[state.state_id] != expected:
            raise RuntimeError(
                f"invalid perturbation_count at state {state.state_id}"
            )


def main():
    base_directory = Path(__file__).resolve().parent
    model_directory = base_directory / "model"

    node_initial = np.loadtxt(
        model_directory / "node_initial.dat",
        delimiter=",",
        dtype=float,
    )
    member = np.loadtxt(
        model_directory / "member.dat",
        delimiter=",",
        dtype=int,
    )
    fix = np.loadtxt(
        model_directory / "fix.dat",
        delimiter=",",
        dtype=int,
    )
    num_node = node_initial.shape[0]
    num_member = member.shape[0]

    index_all = np.arange(num_node * 3)
    index_free = np.delete(index_all, 3 * fix[:, 0] + fix[:, 1])
    L0 = np.linalg.norm(
        node_initial[member[:, 0]] - node_initial[member[:, 1]],
        axis=1,
    )

    reference_point = 0.0
    initial_potential = np.sum(
        np.abs(node_initial[:, 2] - reference_point)
    )
    length_scale = np.mean(L0)
    history = create_history(node_initial, initial_potential)

    # mechanics/roll parametersはmain1--main3_nnと同じにする。
    constraint_tolerance = 1.0e-10
    beta0 = 1.0
    n_iter = 10
    l_iter = 10
    tau = 0.5
    cma_population = 6
    cma_generations = 10
    cma_seed = 0

    target_norm_max = 0.10
    perturbation_scale = 0.20
    max_rollout_steps = 100

    flatness_tolerance = 0.05
    dist_tol = 0.05
    normal_tol = 0.1
    cos_tol = 0.99

    # elite/frontier descriptor parametersもmain3_nnと同じにする。
    layer_threshold = 0.1
    overlap_tolerance = 0.05
    separation_tolerance = 1.0e-5
    collinearity_tolerance = 0.02
    parallel_tolerance = 0.02
    rms_tolerance = 0.02
    max_tolerance = 0.05
    frontier_rms_tolerance = 0.05
    frontier_max_tolerance = 0.10

    # 初期教師まではselector間で完全に共通にする。環境変数は短いsmoke testと
    # 比較実験用であり、未指定なら論文用の既定値になる。
    num_bootstrap = int(os.environ.get("FOLDING_NN_BOOTSTRAP", "3"))
    initial_num_parents = int(
        os.environ.get("FOLDING_NN_INITIAL_PARENTS", "4")
    )
    initial_perturbations_per_parent = int(
        os.environ.get("FOLDING_NN_INITIAL_ACTIONS", "2")
    )
    candidate_num_parents = int(
        os.environ.get("FOLDING_NN_CANDIDATE_PARENTS", "24")
    )
    candidate_perturbations_per_parent = int(
        os.environ.get("FOLDING_NN_CANDIDATE_ACTIONS", "24")
    )
    num_guided_rollouts = int(
        os.environ.get("FOLDING_NN_GUIDED_ROLLOUTS", "2")
    )
    num_random_rollouts = int(
        os.environ.get("FOLDING_NN_RANDOM_ROLLOUTS", "2")
    )
    num_progress_bins = int(
        os.environ.get("FOLDING_MAIN4_PROGRESS_BINS", "6")
    )
    max_count_bucket = int(
        os.environ.get("FOLDING_MAIN4_MAX_COUNT_BUCKET", "4")
    )

    selector = os.environ.get("FOLDING_NN_SELECTOR", "hybrid")
    if selector not in ("hybrid", "random"):
        raise ValueError(
            "FOLDING_NN_SELECTOR must be 'hybrid' or 'random'"
        )
    if num_guided_rollouts + num_random_rollouts <= 0:
        raise ValueError("at least one rollout must be selected per batch")
    if num_progress_bins <= 0 or max_count_bucket < 0:
        raise ValueError("invalid perturbation-count stratification")

    num_models = int(os.environ.get("FOLDING_NN_MODELS", "5"))
    hidden_dimension = int(os.environ.get("FOLDING_NN_HIDDEN", "64"))
    training_epochs = int(os.environ.get("FOLDING_NN_EPOCHS", "150"))
    cost_num_models = int(os.environ.get("FOLDING_NN_COST_MODELS", "3"))
    cost_hidden_dimension = int(
        os.environ.get("FOLDING_NN_COST_HIDDEN", "32")
    )
    cost_training_epochs = int(
        os.environ.get("FOLDING_NN_COST_EPOCHS", "80")
    )
    support_multiplier = float(
        os.environ.get("FOLDING_MAIN4_SUPPORT_MULTIPLIER", "1.5")
    )
    if support_multiplier <= 0.0:
        raise ValueError("FOLDING_MAIN4_SUPPORT_MULTIPLIER must be positive")

    random_seed = int(os.environ.get("FOLDING_SEED", "11"))
    total_step_budget = int(os.environ.get("FOLDING_STEP_BUDGET", "5000"))
    result_name = os.environ.get("FOLDING_RESULT_NAME", "main4")

    np.random.seed(random_seed)
    perturbation_rng = np.random.default_rng(random_seed)
    parent_rng = np.random.default_rng(random_seed + 1)
    selector_rng = np.random.default_rng(random_seed + 2)

    used_steps = 0
    elite_archive = []
    frontier_archive = []
    discovery_curve = [(0, 0)]
    random_edges = []

    # StateNodeを変更せず、状態ごとの累積摂動回数を外部で管理する。
    perturbation_count = {0: 0}

    training_inputs = []
    training_targets = []
    training_stop_labels = []
    training_costs = []
    nn_diagnostics = []
    batch_diagnostics = []
    rollout_diagnostics = []
    mode_cache = {}
    start_time = time.perf_counter()

    def evaluate_candidate(candidate):
        """一候補を終端まで評価し、tree・archive・教師を更新する。"""

        nonlocal used_steps

        parent_id = int(candidate["parent_id"])
        parent_count = int(perturbation_count[parent_id])
        child_count = parent_count + 1
        parent_progress_bin = potential_progress_bin(
            history[parent_id].potential,
            initial_potential,
            num_progress_bins,
        )
        first_new_state_id = len(history)

        current, spent_steps, stop_reason, perturbation_child_id = (
            execute_candidate_rollout(
                history,
                candidate,
                total_step_budget - used_steps,
                member,
                num_node,
                num_member,
                L0,
                index_free,
                target_norm_max,
                max_rollout_steps,
                constraint_tolerance,
                beta0,
                n_iter,
                l_iter,
                tau,
                cma_population,
                cma_generations,
                cma_seed,
                reference_point,
                flatness_tolerance,
                dist_tol,
                normal_tol,
                cos_tol,
            )
        )
        if current is None:
            return False

        # 最初のnew stateは摂動child、その後は全てgravity childなので、
        # このrolloutで生成された状態の摂動回数は全てparent_count+1となる。
        for state_id in range(first_new_state_id, len(history)):
            perturbation_count[state_id] = child_count

        used_steps += spent_steps
        random_edges.append((parent_id, int(perturbation_child_id)))

        update_frontier_archive(
            frontier_archive,
            history,
            range(first_new_state_id, len(history)),
            reference_point,
            length_scale,
            frontier_rms_tolerance,
            frontier_max_tolerance,
        )

        is_new_cell = False
        if stop_reason == "flat":
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

        # 予算打切りには真の終端labelが無いため教師にしない。
        if stop_reason is not None:
            append_training_sample(
                training_inputs,
                training_targets,
                training_stop_labels,
                candidate["input"],
                stop_reason,
                history,
                current.state_id,
                member,
                reference_point,
                length_scale,
                layer_threshold,
                overlap_tolerance,
                separation_tolerance,
                collinearity_tolerance,
                parallel_tolerance,
            )
            training_costs.append(float(spent_steps))

        selection_type = candidate.get("selection_type")
        guided = 1 if str(selection_type).startswith("guided") else 0
        nn_diagnostics.append(
            (
                used_steps,
                len(training_inputs),
                sum(label == 0 for label in training_stop_labels),
                len(elite_archive),
                candidate.get("new_probability", np.nan),
                candidate.get("flat_probability", np.nan),
                candidate.get("uncertainty", np.nan),
                candidate.get("support_distance", np.nan),
                candidate.get("predicted_cost", np.nan),
                candidate.get("novelty_score", np.nan),
                candidate.get("information_score", np.nan),
                guided,
                1 if is_new_cell else 0,
            )
        )
        rollout_diagnostics.append(
            (
                len(rollout_diagnostics),
                used_steps,
                parent_id,
                parent_count,
                child_count,
                min(parent_count, max_count_bucket),
                parent_progress_bin,
                spent_steps,
                stop_reason_code(stop_reason),
                selection_type_code(selection_type),
                1 if is_new_cell else 0,
                len(elite_archive),
            )
        )
        return True

    # 1-a. rootから3本のbootstrapを生成する。
    for _ in range(num_bootstrap):
        if used_steps >= total_step_budget:
            break
        candidates = generate_candidates(
            history,
            [0],
            1,
            member,
            num_node,
            num_member,
            index_free,
            perturbation_scale,
            reference_point,
            length_scale,
            perturbation_rng,
            mode_cache,
        )
        if not candidates:
            break
        candidate = candidates[0]
        candidate["selection_type"] = "bootstrap"
        evaluate_candidate(candidate)

    bootstrap_used_steps = used_steps

    # 1-b. bootstrap frontierの最大4状態 × 2摂動を共通初期教師にする。
    initial_parent_ids = sample_frontier_parent_ids(
        frontier_archive,
        initial_num_parents,
        parent_rng,
    )
    initial_candidates = generate_candidates(
        history,
        initial_parent_ids,
        initial_perturbations_per_parent,
        member,
        num_node,
        num_member,
        index_free,
        perturbation_scale,
        reference_point,
        length_scale,
        perturbation_rng,
        mode_cache,
    )
    for candidate in initial_candidates:
        if used_steps >= total_step_budget:
            break
        candidate["selection_type"] = "initial_training"
        evaluate_candidate(candidate)

    initial_training_used_steps = used_steps
    initial_cell_count = len(elite_archive)

    if selector == "random":
        ensemble = None
        cost_ensemble = None
    else:
        ensemble = train_ensemble(
            training_inputs,
            training_targets,
            training_stop_labels,
            num_models=num_models,
            hidden_dimension=hidden_dimension,
            epochs=training_epochs,
            seed=random_seed + 100,
        )
        cost_ensemble = train_cost_ensemble(
            training_inputs,
            training_costs,
            num_models=cost_num_models,
            hidden=cost_hidden_dimension,
            epochs=cost_training_epochs,
            seed=random_seed + 200,
        )

    print("initial training finished")
    print("bootstrap used steps:", bootstrap_used_steps)
    print("training samples:", len(training_inputs))
    print("flat samples:", sum(label == 0 for label in training_stop_labels))
    print("number of cells:", len(elite_archive))

    # 2. 層別したtree状態から、摂動候補を予算まで評価する。
    while used_steps < total_step_budget:
        batch_start_time = time.perf_counter()
        parent_ids, group_sizes = stratified_parent_ids(
            history,
            perturbation_count,
            initial_potential,
            candidate_num_parents,
            parent_rng,
            num_progress_bins,
            max_count_bucket,
        )
        if not parent_ids:
            break

        candidates = generate_candidates(
            history,
            parent_ids,
            candidate_perturbations_per_parent,
            member,
            num_node,
            num_member,
            index_free,
            perturbation_scale,
            reference_point,
            length_scale,
            perturbation_rng,
            mode_cache,
        )
        candidate_generation_time = time.perf_counter() - batch_start_time
        if not candidates:
            break

        scoring_start_time = time.perf_counter()
        selection_count = num_guided_rollouts + num_random_rollouts
        if selector == "random" or ensemble is None or cost_ensemble is None:
            selected_candidates = select_random_candidates(
                candidates,
                selection_count,
                selector_rng,
            )
        else:
            (
                new_probability,
                uncertainty,
                predicted_distance,
                flat_probability,
            ) = score_candidates(
                ensemble,
                candidates,
                elite_archive,
                member,
                num_node,
                overlap_tolerance,
                separation_tolerance,
                collinearity_tolerance,
                parallel_tolerance,
                rms_tolerance,
                max_tolerance,
                return_flat_probability=True,
            )
            support_distance = candidate_support_distance(
                ensemble,
                candidates,
            )
            candidate_inputs = np.asarray(
                [candidate["input"] for candidate in candidates],
                dtype=float,
            )
            predicted_cost_mean, predicted_cost_std = predict_cost_ensemble(
                cost_ensemble,
                candidate_inputs,
            )
            conservative_cost = np.clip(
                predicted_cost_mean + predicted_cost_std,
                1.0,
                float(max_rollout_steps),
            )
            selected_candidates = select_main4_candidates(
                candidates,
                new_probability,
                flat_probability,
                uncertainty,
                predicted_distance,
                support_distance,
                conservative_cost,
                ensemble["training_support_radius"],
                support_multiplier,
                num_guided_rollouts,
                num_random_rollouts,
                selector_rng,
            )
        scoring_time = time.perf_counter() - scoring_start_time

        rollout_start_time = time.perf_counter()
        selected_counts = []
        for candidate in selected_candidates:
            if used_steps >= total_step_budget:
                break
            selected_counts.append(
                perturbation_count[int(candidate["parent_id"])]
            )
            evaluate_candidate(candidate)
        rollout_time = time.perf_counter() - rollout_start_time

        training_start_time = time.perf_counter()
        if selector != "random":
            ensemble = train_ensemble(
                training_inputs,
                training_targets,
                training_stop_labels,
                num_models=num_models,
                hidden_dimension=hidden_dimension,
                epochs=training_epochs,
                seed=random_seed + 100 + len(training_inputs),
            )
            cost_ensemble = train_cost_ensemble(
                training_inputs,
                training_costs,
                num_models=cost_num_models,
                hidden=cost_hidden_dimension,
                epochs=cost_training_epochs,
                seed=random_seed + 200 + len(training_inputs),
            )
        training_time = time.perf_counter() - training_start_time

        parent_counts = [perturbation_count[state_id] for state_id in parent_ids]
        batch_diagnostics.append(
            (
                used_steps,
                len(parent_ids),
                len(candidates),
                len(group_sizes),
                max(parent_counts, default=0),
                float(np.mean(selected_counts)) if selected_counts else np.nan,
                max(selected_counts, default=-1),
                len(elite_archive),
                candidate_generation_time,
                scoring_time,
                rollout_time,
                training_time,
            )
        )

        if used_steps % 100 < max_rollout_steps:
            print(
                "used steps:", used_steps,
                "cells:", len(elite_archive),
                "max perturbations:", max(perturbation_count.values()),
            )

    if discovery_curve[-1][0] != used_steps:
        discovery_curve.append((used_steps, len(elite_archive)))

    wall_time = time.perf_counter() - start_time
    validate_perturbation_counts(
        history,
        random_edges,
        perturbation_count,
    )
    output_directory = (
        base_directory
        / "result"
        / result_name
        / f"seed_{random_seed:03d}"
    )
    reset_output_directory(output_directory)
    save_elite_paths(history, elite_archive, output_directory)
    save_frontier_states(history, frontier_archive, output_directory)
    save_discovery_curve(discovery_curve, output_directory)
    save_search_tree(
        history=history,
        elite_archive=elite_archive,
        random_edges=random_edges,
        output_directory=output_directory,
        member=member,
        reference_point=reference_point,
        length_scale=length_scale,
        state_rms_tolerance=frontier_rms_tolerance,
        state_max_tolerance=frontier_max_tolerance,
        layer_threshold=layer_threshold,
        overlap_tolerance=overlap_tolerance,
        separation_tolerance=separation_tolerance,
        collinearity_tolerance=collinearity_tolerance,
        parallel_tolerance=parallel_tolerance,
        elite_rms_tolerance=rms_tolerance,
        elite_max_tolerance=max_tolerance,
    )

    np.savetxt(
        output_directory / "nn_diagnostics.dat",
        np.asarray(nn_diagnostics, dtype=float).reshape(-1, 13),
        header=(
            "used_steps num_training num_flat num_cells "
            "predicted_new_probability predicted_flat_probability "
            "ensemble_uncertainty "
            "support_distance predicted_cost novelty_score "
            "information_score guided_selection actual_new_cell"
        ),
    )
    np.savetxt(
        output_directory / "main4_batch_diagnostics.dat",
        np.asarray(batch_diagnostics, dtype=float).reshape(-1, 12),
        header=(
            "used_steps num_parents num_candidates num_nonempty_strata "
            "max_parent_count mean_selected_parent_count "
            "max_selected_parent_count elite_count "
            "candidate_generation_seconds scoring_seconds "
            "rollout_seconds retraining_seconds"
        ),
    )
    np.savetxt(
        output_directory / "main4_rollout_diagnostics.dat",
        np.asarray(rollout_diagnostics, dtype=float).reshape(-1, 12),
        header=(
            "rollout_index used_steps parent_id parent_perturbation_count "
            "child_perturbation_count parent_count_bucket "
            "parent_progress_bin spent_steps stop_reason_code "
            "selection_type_code actual_new_cell elite_count"
        ),
    )
    np.savetxt(
        output_directory / "state_perturbation_counts.dat",
        np.asarray(sorted(perturbation_count.items()), dtype=int).reshape(-1, 2),
        fmt="%d",
        header="state_id perturbation_count",
    )
    np.savetxt(
        output_directory / "random_edges.dat",
        np.asarray(random_edges, dtype=int).reshape(-1, 2),
        fmt="%d",
        header="parent_state_id perturbation_child_state_id",
    )
    elite_count_rows = []
    for cell_index, cell in enumerate(elite_archive):
        terminal_state_id = int(cell["terminal_state_id"])
        elite_count_rows.append(
            (
                cell_index,
                terminal_state_id,
                perturbation_count[terminal_state_id],
                history[terminal_state_id].depth,
            )
        )
    np.savetxt(
        output_directory / "elite_perturbation_counts.dat",
        np.asarray(elite_count_rows, dtype=int).reshape(-1, 4),
        fmt="%d",
        header=(
            "cell_index terminal_state_id perturbation_count path_depth"
        ),
    )
    np.savez(
        output_directory / "training_data.npz",
        inputs=np.asarray(training_inputs),
        targets=np.asarray(training_targets),
        stop_labels=np.asarray(training_stop_labels),
        rollout_costs=np.asarray(training_costs),
    )
    (output_directory / "run_config.txt").write_text(
        "\n".join(
            (
                f"random_seed={random_seed}",
                f"total_step_budget={total_step_budget}",
                f"bootstrap_used_steps={bootstrap_used_steps}",
                f"initial_training_used_steps={initial_training_used_steps}",
                f"initial_cell_count={initial_cell_count}",
                "parent_source=stratified_history",
                f"selector={selector}",
                f"num_bootstrap={num_bootstrap}",
                f"initial_num_parents={initial_num_parents}",
                "initial_perturbations_per_parent="
                f"{initial_perturbations_per_parent}",
                f"candidate_num_parents={candidate_num_parents}",
                "candidate_perturbations_per_parent="
                f"{candidate_perturbations_per_parent}",
                f"num_guided_rollouts={num_guided_rollouts}",
                f"num_random_rollouts={num_random_rollouts}",
                f"num_progress_bins={num_progress_bins}",
                f"max_count_bucket={max_count_bucket}",
                "maximum_perturbation_count=unbounded",
                f"num_models={num_models}",
                f"hidden_dimension={hidden_dimension}",
                f"training_epochs={training_epochs}",
                f"cost_num_models={cost_num_models}",
                f"cost_hidden_dimension={cost_hidden_dimension}",
                f"cost_training_epochs={cost_training_epochs}",
                f"support_multiplier={support_multiplier}",
                f"target_norm_max={target_norm_max}",
                f"perturbation_scale={perturbation_scale}",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    max_count = max(perturbation_count.values(), default=0)
    np.savetxt(
        output_directory / "summary.dat",
        np.asarray(
            [[
                used_steps,
                len(elite_archive),
                len(training_inputs),
                sum(label == 0 for label in training_stop_labels),
                initial_training_used_steps,
                initial_cell_count,
                max_count,
                len(random_edges),
                wall_time,
            ]],
            dtype=float,
        ),
        header=(
            "used_steps num_cells num_training num_flat "
            "initial_training_used_steps initial_cell_count "
            "max_perturbation_count num_random_edges wall_time_seconds"
        ),
    )

    print("search finished")
    print("used steps:", used_steps)
    print("number of cells:", len(elite_archive))
    print("maximum perturbation count:", max_count)
    print("wall time [s]:", wall_time)


if __name__ == "__main__":
    main()
