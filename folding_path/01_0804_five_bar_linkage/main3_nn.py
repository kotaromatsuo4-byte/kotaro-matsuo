"""frontier上の摂動候補を終端予測NNで選ぶ探索。

流れ
1. rootからbootstrap経路を作る。
2. bootstrap経路のfrontier状態へ複数の摂動を与え、初期教師データを作る。
3. frontierまたはhistoryの状態 × 摂動数十方向をNNで一括予測する。
4. 予測新規cell確率上位とランダム候補を実rolloutする。
5. archiveと教師データを更新し、ensembleを再学習する。
"""

from pathlib import Path
import os
import time

import numpy as np

from archive import update_elite_archive, update_frontier_archive
from nn import (
    append_training_sample,
    generate_candidates,
    sample_frontier_parent_ids,
    score_candidates,
    select_candidates,
    select_parent_ids,
    select_random_candidates,
    train_ensemble,
)
from result import (
    reset_output_directory,
    save_discovery_curve,
    save_elite_paths,
    save_frontier_states,
    save_search_tree,
)
from roll import deterministic_roll, displacement_roll
from state import create_history


def execute_candidate_rollout(
    history,
    candidate,
    remaining_budget,
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
):
    """選択済み物理変位を1回与え、その後は決定論的に終端まで進める。"""
    if remaining_budget <= 0:
        return None, 0, None, None

    parent_id = candidate["parent_id"]
    current = displacement_roll(
        history,
        parent_id,
        candidate["displacement"],
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
    )
    perturbation_child_id = current.state_id
    used_steps = 1
    rollout_steps = 1

    while (
        current.terminal_reason is None
        and rollout_steps < max_rollout_steps
        and used_steps < remaining_budget
    ):
        current = deterministic_roll(
            history,
            current.state_id,
            member,
            num_node,
            num_member,
            L0,
            index_free,
            target_norm_max,
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
        used_steps += 1
        rollout_steps += 1

    if current.terminal_reason is not None:
        stop_reason = current.terminal_reason
    elif rollout_steps >= max_rollout_steps:
        stop_reason = "timeout"
    else:
        # 総予算によって途中で切れた経路には真の終端教師ラベルがない。
        stop_reason = None

    return current, used_steps, stop_reason, perturbation_child_id


def main():
    base_directory = Path(__file__).resolve().parent
    model_directory = base_directory / "model"

    # input
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

    """mechanics/roll parameters: main1〜3と同じ値"""
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

    """elite/frontier parameters: main3と同じ値"""
    layer_threshold = 0.1
    overlap_tolerance = 0.05
    separation_tolerance = 1.0e-5
    collinearity_tolerance = 0.02
    parallel_tolerance = 0.02
    rms_tolerance = 0.02
    max_tolerance = 0.05
    frontier_rms_tolerance = 0.05
    frontier_max_tolerance = 0.10

    """NN探索 parameters"""
    num_bootstrap = int(os.environ.get("FOLDING_NN_BOOTSTRAP", "3"))
    initial_num_parents = int(
        os.environ.get("FOLDING_NN_INITIAL_PARENTS", "8")
    )
    initial_perturbations_per_parent = int(
        os.environ.get("FOLDING_NN_INITIAL_ACTIONS", "2")
    )

    # 数十状態 × 数十摂動を、correctorなしで一括予測する。
    candidate_num_parents = int(
        os.environ.get("FOLDING_NN_CANDIDATE_PARENTS", "24")
    )
    candidate_perturbations_per_parent = int(
        os.environ.get("FOLDING_NN_CANDIDATE_ACTIONS", "24")
    )
    # frontier: frontier prototypeから抽出
    # history_random: 全historyを母集団として上の個数だけ抽出
    # history_all: 全ての非終端StateNodeを毎batch評価
    parent_source = os.environ.get(
        "FOLDING_NN_PARENT_SOURCE",
        "frontier",
    )

    # 一回のbatchではP_new上位2候補と完全ランダム2候補を全てrolloutする。
    # five-barの予備実験では3+1よりseed間の結果が安定した。
    num_guided_rollouts = int(
        os.environ.get("FOLDING_NN_GUIDED_ROLLOUTS", "2")
    )
    num_random_rollouts = int(
        os.environ.get("FOLDING_NN_RANDOM_ROLLOUTS", "2")
    )
    # hybrid: P_new上位とrandomを併用、guided: 全枠P_new上位、
    # random: 同じ親・摂動候補集合からNNを使わず一様選択する対照条件。
    selector = os.environ.get("FOLDING_NN_SELECTOR", "hybrid")
    if selector not in ("hybrid", "guided", "random"):
        raise ValueError(
            "FOLDING_NN_SELECTOR must be 'hybrid', 'guided', or 'random'"
        )

    num_models = int(os.environ.get("FOLDING_NN_MODELS", "5"))
    hidden_dimension = int(os.environ.get("FOLDING_NN_HIDDEN", "64"))
    training_epochs = int(os.environ.get("FOLDING_NN_EPOCHS", "500"))

    random_seed = int(os.environ.get("FOLDING_SEED", "11"))
    total_step_budget = int(os.environ.get("FOLDING_STEP_BUDGET", "5000"))
    result_name = os.environ.get("FOLDING_RESULT_NAME", "main3_nn")
    np.random.seed(random_seed)
    perturbation_rng = np.random.default_rng(random_seed)
    # 親集合と候補selectorで乱数列を分離し、selectorを変えても次batchの
    # 親抽出が乱数消費量の違いだけで変化しないようにする。
    parent_rng = np.random.default_rng(random_seed + 1)
    selector_rng = np.random.default_rng(random_seed + 2)

    used_steps = 0
    elite_archive = []
    frontier_archive = []
    discovery_curve = [(0, 0)]
    random_edges = []

    training_inputs = []
    training_targets = []
    training_stop_labels = []
    diagnostics = []
    batch_diagnostics = []
    mode_cache = {}
    start_time = time.perf_counter()

    def evaluate_candidate(candidate):
        """一候補を実行し、history・archives・教師データをまとめて更新する。"""
        nonlocal used_steps

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

        used_steps += spent_steps
        random_edges.append((candidate["parent_id"], perturbation_child_id))

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

        # 総予算による途中打切りは教師にせず、物理的な終端だけ追加する。
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

        diagnostics.append(
            (
                used_steps,
                len(training_inputs),
                sum(label == 0 for label in training_stop_labels),
                len(elite_archive),
                candidate.get("new_probability", np.nan),
                candidate.get("uncertainty", np.nan),
                1 if candidate.get("selection_type") == "guided" else 0,
                1 if is_new_cell else 0,
            )
        )
        return True

    # 1-a. rootに摂動を与えてbootstrap経路を生成する。
    for _ in range(num_bootstrap):
        if used_steps >= total_step_budget:
            break
        candidate = generate_candidates(
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
        )[0]
        candidate["selection_type"] = "bootstrap"
        evaluate_candidate(candidate)

    bootstrap_used_steps = used_steps

    # 1-b. bootstrapで得た複数frontier状態 × 複数摂動から初期教師を作る。
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

    if selector == "random":
        ensemble = None
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

    print("initial training finished")
    print("bootstrap used steps:", bootstrap_used_steps)
    print("training samples:", len(training_inputs))
    print("flat samples:", sum(label == 0 for label in training_stop_labels))
    print("number of cells:", len(elite_archive))

    # 2, 3. NN予測による候補選択、実rollout、再学習を予算まで繰り返す。
    while used_steps < total_step_budget:
        batch_start_time = time.perf_counter()
        parent_ids = select_parent_ids(
            history,
            frontier_archive,
            parent_source,
            candidate_num_parents,
            parent_rng,
        )
        if len(parent_ids) == 0:
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
        if len(candidates) == 0:
            break

        scoring_start_time = time.perf_counter()
        selection_count = num_guided_rollouts + num_random_rollouts
        if selector == "random" or ensemble is None:
            # random対照、またはflat教師がまだ無い初期段階。
            selected_candidates = select_random_candidates(
                candidates,
                selection_count,
                selector_rng,
            )
        else:
            new_probability, uncertainty, predicted_distance = score_candidates(
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
            )
            if selector == "guided":
                guided_count = selection_count
                random_count = 0
            else:
                guided_count = num_guided_rollouts
                random_count = num_random_rollouts

            selected_candidates = select_candidates(
                candidates,
                new_probability,
                uncertainty,
                predicted_distance,
                guided_count,
                random_count,
                selector_rng,
            )
        scoring_time = time.perf_counter() - scoring_start_time

        # 選んだL候補はB候補へ再選別せず、corrector後に全てrolloutする。
        rollout_start_time = time.perf_counter()
        for candidate in selected_candidates:
            if used_steps >= total_step_budget:
                break
            evaluate_candidate(candidate)
        rollout_time = time.perf_counter() - rollout_start_time

        # archive価値は毎batch変わる。教師追加後にensembleも学習し直す。
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
        training_time = time.perf_counter() - training_start_time
        batch_diagnostics.append(
            (
                used_steps,
                len(parent_ids),
                len(candidates),
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
                "training samples:", len(training_inputs),
            )

    if discovery_curve[-1][0] != used_steps:
        discovery_curve.append((used_steps, len(elite_archive)))

    wall_time = time.perf_counter() - start_time
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
        np.asarray(diagnostics, dtype=float),
        header=(
            "used_steps num_training num_flat num_cells "
            "predicted_new_probability ensemble_uncertainty "
            "guided_selection actual_new_cell"
        ),
    )
    np.savetxt(
        output_directory / "nn_batch_diagnostics.dat",
        np.asarray(batch_diagnostics, dtype=float).reshape(-1, 7),
        header=(
            "used_steps num_parents num_candidates "
            "candidate_generation_seconds scoring_seconds "
            "rollout_seconds retraining_seconds"
        ),
    )
    np.savez(
        output_directory / "training_data.npz",
        inputs=np.asarray(training_inputs),
        targets=np.asarray(training_targets),
        stop_labels=np.asarray(training_stop_labels),
    )
    (output_directory / "run_config.txt").write_text(
        "\n".join(
            (
                f"random_seed={random_seed}",
                f"total_step_budget={total_step_budget}",
                f"parent_source={parent_source}",
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
                f"num_models={num_models}",
                f"hidden_dimension={hidden_dimension}",
                f"training_epochs={training_epochs}",
                f"target_norm_max={target_norm_max}",
                f"perturbation_scale={perturbation_scale}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    np.savetxt(
        output_directory / "summary.dat",
        np.asarray(
            [[
                used_steps,
                len(elite_archive),
                len(training_inputs),
                sum(label == 0 for label in training_stop_labels),
                wall_time,
            ]],
            dtype=float,
        ),
        header="used_steps num_cells num_training num_flat wall_time_seconds",
    )

    print("search finished")
    print("used steps:", used_steps)
    print("number of cells:", len(elite_archive))
    print("wall time [s]:", wall_time)


if __name__ == "__main__":
    main()
