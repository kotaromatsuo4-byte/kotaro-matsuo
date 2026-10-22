"""GPで摂動位置とモード配合率を選ぶ多様平坦折経路探索。

探索の一回の評価は

    過去の非終端StateNodeを一つ選ぶ
    -> その状態のメカニズム空間内で摂動を与える
    -> 任意設定では、その物理方向を新しい接空間へ運んで小摂動を続ける
    -> 以後は重力ポテンシャル最小化で終端まで決定論rolloutする

である。最初はrootから決定論rolloutを一本だけ生成し、複数のroot
bootstrapは作らない。完了rolloutで新しく生成された全edgeを

    (親の節点座標, corrector前の物理変位) -> rollout終端

という教師にする。collisionをflat終端座標へ混ぜず、flat到達確率、
全終端で定義できる最小平坦度、flat条件付き終端descriptorを分けて学ぶ。

環境変数 ``FOLDING_MAIN5_SELECTOR`` は ``random``、
``transport_heuristic``、``gp_ucb``、``gp_posterior``、``gp_filter`` の
いずれか。
全手法は同じ候補生成・力学step予算を使う。

任意設定では、直前の物理摂動を現在の接空間へ射影して方向を運ぶ。
これは複数の有界な小摂動で一つの有限変形を作るためのemitterであり、
SVDモード基底の符号反転や基底内回転には依存しない。
"""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
import time

import numpy as np

from archive import update_elite_archive
from descriptor import find_last_nonflat_state_id
from gp import (
    fit_shared_kernel_gp,
    fit_surrogate_models,
    generate_tangent_candidates,
    make_terminal_output_spec,
    score_acquisition,
    terminal_target_from_history,
    transition_input,
    predict_shared_kernel_gp,
)
from mechanics import calc_dedx, calc_mode, compute_gravity_potential
from path_diagnostics import (
    apply_direct_perturbation_parent_bonus,
    apply_transport_alignment_bonus,
    state_consecutive_perturbation_counts,
    state_path_perturbation_counts,
)
from result import (
    reset_output_directory,
    save_discovery_curve,
    save_elite_paths,
    save_search_tree,
)
from roll import deterministic_roll, displacement_roll
from state import create_history, get_path_state_ids


def _env_int(name, default):
    return int(os.environ.get(name, str(default)))


def _env_float(name, default):
    return float(os.environ.get(name, str(default)))


def _load_model(model_directory):
    """1行だけのmember/fixファイルにも対応して入力モデルを読む。"""

    node = np.atleast_2d(
        np.loadtxt(model_directory / "node_initial.dat", delimiter=",", dtype=float)
    )
    member = np.atleast_2d(
        np.loadtxt(model_directory / "member.dat", delimiter=",", dtype=int)
    )
    fix = np.atleast_2d(
        np.loadtxt(model_directory / "fix.dat", delimiter=",", dtype=int)
    )
    num_node = len(node)
    index_all = np.arange(3 * num_node)
    fixed_index = 3 * fix[:, 0] + fix[:, 1]
    index_free = np.delete(index_all, fixed_index)
    member_length = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]],
        axis=1,
    )
    return node, member, fix, index_free, member_length


def _configuration():
    """比較で変更する値を含め、実行設定を一つのdictで返す。"""

    selector = os.environ.get("FOLDING_MAIN5_SELECTOR", "gp_ucb")
    if selector not in (
        "random",
        "transport_heuristic",
        "gp_ucb",
        "gp_posterior",
        "gp_filter",
        "gp_filter_shuffled",
        "gp_filter_cost",
        "gp_filter_cost_shuffled",
    ):
        raise ValueError(
            "FOLDING_MAIN5_SELECTOR must be random, transport_heuristic, "
            "gp_ucb, gp_posterior, gp_filter, gp_filter_shuffled, or "
            "gp_filter_cost, or gp_filter_cost_shuffled"
        )

    return {
        # 数値結果とアルゴリズム版を対応付けるため、動作を変えたときに更新する。
        "algorithm_version": "main5_gp_v4_2_persistent_macro_filter_diagnostics",
        "selector": selector,
        "seed": _env_int("FOLDING_SEED", 0),
        "total_step_budget": _env_int("FOLDING_STEP_BUDGET", 1200),
        "reference_point": _env_float("FOLDING_REFERENCE_POINT", 0.0),
        "constraint_tolerance": _env_float(
            "FOLDING_CONSTRAINT_TOLERANCE", 1.0e-10
        ),
        "beta0": 1.0,
        "corrector_iterations": _env_int("FOLDING_CORRECTOR_ITERATIONS", 10),
        "line_search_iterations": _env_int("FOLDING_LINE_SEARCH_ITERATIONS", 10),
        "line_search_tau": 0.5,
        "cma_population": _env_int("FOLDING_CMA_POPULATION", 6),
        "cma_generations": _env_int("FOLDING_CMA_GENERATIONS", 10),
        "cma_seed": _env_int("FOLDING_CMA_SEED", 0),
        "target_norm_max": _env_float("FOLDING_TARGET_NORM", 0.10),
        "perturbation_scale": _env_float("FOLDING_PERTURBATION_SCALE", 0.20),
        # 1は従来の単一摂動。2以上では、最初の物理変位を各更新状態の
        # 接空間へ逐次射影し、同じ向きを保つmacro-actionとして実行する。
        "perturbation_chain_length": _env_int(
            "FOLDING_MAIN5_PERTURBATION_CHAIN_LENGTH", 1
        ),
        "max_rollout_steps": _env_int("FOLDING_MAX_ROLLOUT_STEPS", 100),
        "flatness_tolerance": _env_float("FOLDING_FLATNESS_TOLERANCE", 0.05),
        "dist_tol": _env_float("FOLDING_COLLISION_DIST_TOL", 0.05),
        "normal_tol": _env_float("FOLDING_COLLISION_NORMAL_TOL", 0.10),
        "cos_tol": _env_float("FOLDING_COLLISION_COS_TOL", 0.99),
        # five-barの比較では連続なx-z平面内交差を使う。一般3次元構造では
        # FOLDING_COLLISION_MODEL=3d としてcollision.pyへ差し替えられる。
        "collision_model": os.environ.get("FOLDING_COLLISION_MODEL", "planar"),
        "layer_threshold": _env_float("FOLDING_LAYER_THRESHOLD", 0.10),
        "overlap_tolerance": _env_float("FOLDING_OVERLAP_TOLERANCE", 0.05),
        "separation_tolerance": _env_float(
            "FOLDING_SEPARATION_TOLERANCE", 1.0e-5
        ),
        "collinearity_tolerance": _env_float(
            "FOLDING_COLLINEARITY_TOLERANCE", 0.02
        ),
        "parallel_tolerance": _env_float("FOLDING_PARALLEL_TOLERANCE", 0.02),
        "rms_tolerance": _env_float("FOLDING_ELITE_RMS_TOLERANCE", 0.02),
        "max_tolerance": _env_float("FOLDING_ELITE_MAX_TOLERANCE", 0.05),
        "state_cell_rms_tolerance": 0.05,
        "state_cell_max_tolerance": 0.10,
        # 0なら全非終端StateNodeを毎回候補にする。大構造ではbounded poolに
        # できるが、教師学習には常に全完了edgeを使う。
        "candidate_parent_limit": _env_int("FOLDING_MAIN5_PARENT_LIMIT", 64),
        "directions_per_parent": _env_int("FOLDING_MAIN5_DIRECTIONS", 16),
        # 大きい摂動を複数の小摂動へ分解できるよう、摂動直後StateNodeを
        # bounded candidate poolへ最低何個残すかを指定する。0は従来挙動。
        "direct_parent_minimum": _env_int(
            "FOLDING_MAIN5_DIRECT_PARENT_MINIMUM", 0
        ),
        # rootをbounded candidate poolへ必ず含める。既定0は従来挙動。
        "include_root_parent": bool(
            _env_int("FOLDING_MAIN5_INCLUDE_ROOT_PARENT", 0)
        ),
        # 直前の物理摂動を現在の接空間へ射影した候補で、最も近い通常候補を
        # 置換する。候補数は変えないためrandom対照の位置分布を歪めない。
        "include_transported_direction": bool(
            _env_int("FOLDING_MAIN5_INCLUDE_TRANSPORTED_DIRECTION", 0)
        ),
        "transport_min_projection_ratio": _env_float(
            "FOLDING_MAIN5_TRANSPORT_MIN_RATIO", 0.10
        ),
        "epsilon_random": _env_float("FOLDING_MAIN5_EPSILON", 0.20),
        # gp_filterではparentごとの最大scoreを比べ、上位何割のparentを
        # 残すかを指定する。残したparentから一様抽選してargmax集中を防ぐ。
        "filter_keep_fraction": _env_float(
            "FOLDING_GP_FILTER_KEEP_FRACTION", 0.50
        ),
        "cost_lcb_kappa": _env_float("FOLDING_GP_COST_LCB_KAPPA", 1.0),
        "cost_offset": _env_float("FOLDING_GP_COST_OFFSET", 1.0),
        # filterが捨てる親の最良候補を少数だけ実行し、false rejectを監査する。
        # 監査rolloutも通常どおりmechanics予算へ数える。0なら無効。
        "filter_audit_rate": _env_float("FOLDING_GP_FILTER_AUDIT_RATE", 0.0),
        # 初期trunk一経路だけでは終端写像を識別できない。これは複数root
        # bootstrapではなく、主探索内で両selectorに共通なrandom校正区間。
        "cold_start_rollouts": _env_int("FOLDING_MAIN5_COLD_START_ROLLOUTS", 4),
        "refit_rollout_interval": _env_int("FOLDING_MAIN5_REFIT_INTERVAL", 3),
        # exact GPを長時間探索でも使えるよう、全教師は保存したままfit時だけ
        # rollout単位で間引ける。all_edgesは従来どおり全完了edgeを使用。
        "gp_training_mode": os.environ.get(
            "FOLDING_GP_TRAINING_MODE", "all_edges"
        ),
        "gp_balanced_edges_per_rollout": _env_int(
            "FOLDING_GP_BALANCED_EDGES_PER_ROLLOUT", 8
        ),
        "state_length_scale": _env_float("FOLDING_GP_STATE_LENGTH", 0.50),
        "action_length_scale": _env_float("FOLDING_GP_ACTION_LENGTH", 0.10),
        "signal_variance": _env_float("FOLDING_GP_SIGNAL_VARIANCE", 1.0),
        "regression_noise_std": _env_float("FOLDING_GP_NOISE", 0.08),
        "classifier_noise_std": _env_float("FOLDING_GP_CLASSIFIER_NOISE", 0.25),
        "descriptor_scale_floor": _env_float(
            "FOLDING_GP_DESCRIPTOR_SCALE_FLOOR", 0.05
        ),
        "novelty_kappa": _env_float("FOLDING_GP_KAPPA", 1.0),
        # 摂動直後StateNodeを親とするguided候補だけに掛ける乗算bonus。
        # 例えば1.0ならbase acquisitionを2倍にする。random枠には掛けない。
        "direct_parent_bonus": _env_float(
            "FOLDING_GP_DIRECT_PARENT_BONUS", 0.0
        ),
        # 直前の物理摂動を現在の接空間へtransportした方向とのcos類似度に
        # 比例するguided scoreの乗算bonus。SVD基底の符号・回転に不変。
        "transport_alignment_bonus": _env_float(
            "FOLDING_GP_TRANSPORT_ALIGNMENT_BONUS", 0.0
        ),
        "reach_weight": _env_float("FOLDING_GP_REACH_WEIGHT", 0.25),
        "cold_uncertainty_weight": _env_float(
            "FOLDING_GP_COLD_UNCERTAINTY_WEIGHT", 0.10
        ),
        # GP出力のlast-nonflat zは長さで無次元化済み。potentialの次元を持つ
        # layer_thresholdとは分けて定義する。
        "layer_uncertainty_scale": _env_float(
            "FOLDING_GP_LAYER_UNCERTAINTY_SCALE", 0.10
        ),
        "posterior_samples": _env_int("FOLDING_GP_POSTERIOR_SAMPLES", 32),
        "save_tree": bool(_env_int("FOLDING_SAVE_TREE", 1)),
        "verbose": bool(_env_int("FOLDING_VERBOSE", 1)),
        "result_name": os.environ.get(
            "FOLDING_RESULT_NAME", f"main5_{selector}"
        ),
    }


def _validate_configuration(config):
    if config["total_step_budget"] <= 0:
        raise ValueError("total_step_budget must be positive")
    if config["max_rollout_steps"] <= 0:
        raise ValueError("max_rollout_steps must be positive")
    if config["perturbation_chain_length"] <= 0:
        raise ValueError("perturbation_chain_length must be positive")
    if config["perturbation_chain_length"] > config["max_rollout_steps"]:
        raise ValueError(
            "perturbation_chain_length must not exceed max_rollout_steps"
        )
    if config["directions_per_parent"] <= 0:
        raise ValueError("directions_per_parent must be positive")
    if config["candidate_parent_limit"] < 0:
        raise ValueError("candidate_parent_limit must be nonnegative")
    if config["direct_parent_minimum"] < 0:
        raise ValueError("direct_parent_minimum must be nonnegative")
    if not 0.0 <= config["epsilon_random"] <= 1.0:
        raise ValueError("epsilon_random must be in [0, 1]")
    if not 0.0 < config["filter_keep_fraction"] <= 1.0:
        raise ValueError("filter_keep_fraction must be in (0, 1]")
    if config["cost_lcb_kappa"] < 0.0 or config["cost_offset"] <= 0.0:
        raise ValueError("cost_lcb_kappa must be nonnegative and cost_offset positive")
    if not 0.0 <= config["filter_audit_rate"] <= 1.0:
        raise ValueError("filter_audit_rate must be in [0, 1]")
    if config["cold_start_rollouts"] < 0:
        raise ValueError("cold_start_rollouts must be nonnegative")
    if config["direct_parent_bonus"] < 0.0:
        raise ValueError("direct_parent_bonus must be nonnegative")
    if config["transport_alignment_bonus"] < 0.0:
        raise ValueError("transport_alignment_bonus must be nonnegative")
    if not 0.0 <= config["transport_min_projection_ratio"] <= 1.0:
        raise ValueError("transport_min_projection_ratio must be in [0, 1]")
    if config["refit_rollout_interval"] <= 0:
        raise ValueError("refit_rollout_interval must be positive")
    if config["gp_training_mode"] not in (
        "all_edges",
        "rollout_balanced",
        "perturbation_only",
        "macro_first",
    ):
        raise ValueError(
            "gp_training_mode must be all_edges, rollout_balanced, or "
            "perturbation_only, or macro_first"
        )
    if config["gp_balanced_edges_per_rollout"] <= 0:
        raise ValueError("gp_balanced_edges_per_rollout must be positive")
    if (
        config["perturbation_chain_length"] > 1
        and config["selector"].startswith("gp_")
        and config["gp_training_mode"] != "macro_first"
    ):
        raise ValueError(
            "GP selectors with a persistent macro require "
            "FOLDING_GP_TRAINING_MODE=macro_first so train and query actions "
            "have the same meaning"
        )
    if config["posterior_samples"] <= 0:
        raise ValueError("posterior_samples must be positive")
    for name in (
        "target_norm_max",
        "perturbation_scale",
        "state_length_scale",
        "action_length_scale",
        "signal_variance",
        "regression_noise_std",
        "classifier_noise_std",
        "descriptor_scale_floor",
        "layer_uncertainty_scale",
    ):
        if config[name] <= 0.0:
            raise ValueError(f"{name} must be positive")
    if config["collision_model"] not in ("planar", "3d"):
        raise ValueError("collision_model must be planar or 3d")
    if (
        config["selector"] == "transport_heuristic"
        and not config["include_transported_direction"]
    ):
        raise ValueError(
            "transport_heuristic requires include_transported_direction"
        )


def _deterministic_roll_once(history, parent_id, model, config):
    return deterministic_roll(
        history,
        parent_id,
        model["member"],
        model["num_node"],
        model["num_member"],
        model["member_length"],
        model["index_free"],
        config["target_norm_max"],
        config["constraint_tolerance"],
        config["beta0"],
        config["corrector_iterations"],
        config["line_search_iterations"],
        config["line_search_tau"],
        config["cma_population"],
        config["cma_generations"],
        config["cma_seed"],
        config["reference_point"],
        config["flatness_tolerance"],
        config["dist_tol"],
        config["normal_tol"],
        config["cos_tol"],
        config["collision_model"],
    )


def _initial_deterministic_trunk(history, remaining_budget, model, config):
    """rootから一本だけ決定論的に終端まで進める。"""

    current = history[0]
    new_state_ids = []
    used_steps = 0
    if current.potential <= config["flatness_tolerance"]:
        return current, new_state_ids, used_steps, "flat"
    while current.terminal_reason is None and used_steps < remaining_budget:
        if used_steps >= config["max_rollout_steps"]:
            return current, new_state_ids, used_steps, "timeout"
        current = _deterministic_roll_once(
            history,
            current.state_id,
            model,
            config,
        )
        new_state_ids.append(current.state_id)
        used_steps += 1

    if current.terminal_reason is not None:
        return current, new_state_ids, used_steps, current.terminal_reason
    return current, new_state_ids, used_steps, None


def _transport_displacement(node, previous_displacement, model, config):
    """直前の物理変位を現在状態の接空間へ射影して同じnormへ戻す。

    SVDの係数ではなく物理変位を射影するため、零空間基底の符号反転・
    直交回転に依存しない。射影がほぼ消える場合はNoneを返す。
    """

    jacobian = calc_dedx(
        node,
        model["num_node"],
        model["member"],
        model["num_member"],
        model["index_free"],
    )
    mode = calc_mode(
        model["index_free"],
        jacobian,
        model["num_node"],
    )
    if mode.shape[1] == 0:
        return None

    previous = np.asarray(previous_displacement, dtype=float).reshape(-1)
    previous_norm = float(np.linalg.norm(previous))
    projection = mode @ (np.linalg.pinv(mode, rcond=1.0e-12) @ previous)
    projection_norm = float(np.linalg.norm(projection))
    projection_ratio = projection_norm / max(previous_norm, 1.0e-12)
    if (
        projection_norm <= 1.0e-12
        or projection_ratio < config["transport_min_projection_ratio"]
    ):
        return None
    return (
        config["perturbation_scale"] * projection / projection_norm
    ).reshape(model["num_node"], 3)


def _candidate_rollout(history, candidate, remaining_budget, model, config):
    """持続小摂動macroを与え、その後は決定論的に終端まで進める。"""

    if remaining_budget <= 0:
        return None, [], 0, None, None

    current = history[int(candidate["parent_id"])]
    displacement = np.asarray(candidate["displacement"], dtype=float)
    perturbation_child_ids = []
    new_state_ids = []
    used_steps = 0
    projection_failed = False

    for chain_index in range(config["perturbation_chain_length"]):
        if used_steps >= remaining_budget or current.terminal_reason is not None:
            break
        if chain_index > 0:
            displacement = _transport_displacement(
                current.node,
                displacement,
                model,
                config,
            )
            if displacement is None:
                projection_failed = True
                break

        current = displacement_roll(
            history,
            current.state_id,
            displacement,
            model["member"],
            model["num_node"],
            model["num_member"],
            model["member_length"],
            model["index_free"],
            config["constraint_tolerance"],
            config["beta0"],
            config["corrector_iterations"],
            config["line_search_iterations"],
            config["line_search_tau"],
            config["reference_point"],
            config["flatness_tolerance"],
            config["dist_tol"],
            config["normal_tol"],
            config["cos_tol"],
            config["collision_model"],
        )
        perturbation_child_ids.append(current.state_id)
        new_state_ids.append(current.state_id)
        used_steps += 1

    if not new_state_ids:
        return None, [], 0, None, []
    if projection_failed:
        return (
            current,
            new_state_ids,
            used_steps,
            "projection_failure",
            perturbation_child_ids,
        )

    while (
        current.terminal_reason is None
        and len(new_state_ids) < config["max_rollout_steps"]
        and used_steps < remaining_budget
    ):
        current = _deterministic_roll_once(
            history,
            current.state_id,
            model,
            config,
        )
        new_state_ids.append(current.state_id)
        used_steps += 1

    if current.terminal_reason is not None:
        stop_reason = current.terminal_reason
    elif len(new_state_ids) >= config["max_rollout_steps"]:
        stop_reason = "timeout"
    else:
        # 総step予算による打切り。真の終端教師には使わない。
        stop_reason = None
    return (
        current,
        new_state_ids,
        used_steps,
        stop_reason,
        perturbation_child_ids,
    )


def _new_training_data(output_dimension):
    return {
        "inputs": [],
        "flat_labels": [],
        "flatness_targets": [],
        "cost_targets": [],
        "descriptor_targets": [],
        "rollout_ids": [],
        "state_ids": [],
        "edge_kinds": [],
        "stop_reasons": [],
        "output_dimension": int(output_dimension),
    }


def _append_completed_rollout(
    data,
    history,
    new_state_ids,
    stop_reason,
    rollout_id,
    model,
    config,
    output_spec,
):
    """完了rolloutの全新規edgeを、それぞれの入力に対して教師化する。"""

    if stop_reason not in ("flat", "collision") or not new_state_ids:
        return 0

    potentials = np.asarray(
        [history[state_id].potential for state_id in new_state_ids],
        dtype=float,
    )
    suffix_minimum = np.minimum.accumulate(potentials[::-1])[::-1]
    # flat閾値を1とする無次元量。flatならlog(1+h/tol)<=log(2)。
    flatness_targets = np.log1p(
        suffix_minimum / max(config["flatness_tolerance"], 1.0e-12)
    )

    if stop_reason == "flat":
        terminal_target = terminal_target_from_history(
            history,
            new_state_ids[-1],
            config["layer_threshold"],
            output_spec,
        )
    else:
        terminal_target = np.full(
            data["output_dimension"],
            np.nan,
            dtype=float,
        )

    rollout_length = len(new_state_ids)
    for edge_index, (state_id, flatness_target) in enumerate(
        zip(new_state_ids, flatness_targets)
    ):
        state = history[state_id]
        if state.parent_id is None or state.incoming_displacement is None:
            continue
        parent = history[state.parent_id]
        context_state_id = find_last_nonflat_state_id(
            history,
            state.parent_id,
            config["layer_threshold"],
        )
        data["inputs"].append(
            transition_input(
                parent.node,
                state.incoming_displacement,
                model["index_free"],
                config["reference_point"],
                model["length_scale"],
                history[context_state_id].node,
            )
        )
        data["flat_labels"].append(stop_reason == "flat")
        data["flatness_targets"].append(float(flatness_target))
        # このedgeから同じrollout終端までに実際に使った残りstep数。
        # log1pで学習し、macro_firstならmacro全体のcostが教師になる。
        data["cost_targets"].append(
            float(np.log1p(rollout_length - edge_index))
        )
        data["descriptor_targets"].append(terminal_target.copy())
        data["rollout_ids"].append(int(rollout_id))
        data["state_ids"].append(int(state_id))
        data["edge_kinds"].append(state.incoming_step_type)
        data["stop_reasons"].append(stop_reason)
    return len(new_state_ids)


def _evenly_spaced_indices(indices, maximum):
    """順序を保ったまま、配列から最大maximum個を等間隔に残す。"""

    values = np.asarray(indices, dtype=int)
    if len(values) <= int(maximum):
        return values
    positions = np.rint(
        np.linspace(0, len(values) - 1, int(maximum))
    ).astype(int)
    return values[np.unique(positions)]


def _gp_training_indices(data, config):
    """保存済み全edgeから、GP fitに使う行だけを決定論的に選ぶ。"""

    num_rows = len(data["inputs"])
    if num_rows == 0 or config["gp_training_mode"] == "all_edges":
        return np.arange(num_rows, dtype=int)

    rollout_ids = np.asarray(data["rollout_ids"], dtype=int)
    edge_kinds = np.asarray(data["edge_kinds"], dtype="U16")
    if config["gp_training_mode"] == "perturbation_only":
        return np.flatnonzero(edge_kinds == "perturbation")
    if config["gp_training_mode"] == "macro_first":
        pieces = []
        for rollout_id in np.unique(rollout_ids):
            group = np.flatnonzero(
                (rollout_ids == rollout_id)
                & (edge_kinds == "perturbation")
            )
            if len(group) > 0:
                pieces.append(int(group[0]))
        return np.asarray(pieces, dtype=int)

    maximum = int(config["gp_balanced_edges_per_rollout"])
    pieces = []
    for rollout_id in np.unique(rollout_ids):
        group = np.flatnonzero(rollout_ids == rollout_id)
        if len(group) <= maximum:
            pieces.append(group)
            continue
        perturbation = group[edge_kinds[group] == "perturbation"]
        if len(perturbation) == 0:
            pieces.append(_evenly_spaced_indices(group, maximum))
            continue
        required = int(perturbation[0])
        remaining = group[group != required]
        other = _evenly_spaced_indices(remaining, max(maximum - 1, 0))
        pieces.append(np.sort(np.concatenate(([required], other))))

    if not pieces:
        return np.empty(0, dtype=int)
    return np.sort(np.concatenate(pieces))


def _fit_models(data, model, config):
    """完了edgeを、指定したrollout単位の規則で抽出して再学習する。"""

    input_dimension = model["input_dimension"]
    output_dimension = data["output_dimension"]
    inputs = np.asarray(data["inputs"], dtype=float).reshape(-1, input_dimension)
    flat_labels = np.asarray(data["flat_labels"], dtype=bool)
    flatness_targets = np.asarray(data["flatness_targets"], dtype=float)
    cost_targets = np.asarray(data["cost_targets"], dtype=float)
    descriptors = np.asarray(data["descriptor_targets"], dtype=float).reshape(
        -1,
        output_dimension,
    )
    indices = _gp_training_indices(data, config)
    models = fit_surrogate_models(
        inputs[indices],
        flat_labels[indices],
        flatness_targets[indices],
        descriptors[indices],
        state_dimension=model["input_state_dimension"],
        state_length_scale=config["state_length_scale"],
        action_length_scale=config["action_length_scale"],
        signal_variance=config["signal_variance"],
        regression_noise_std=config["regression_noise_std"],
        classifier_noise_std=config["classifier_noise_std"],
        descriptor_scale_floor=config["descriptor_scale_floor"],
        prior_flat_probability=0.5,
    )
    models["cost"] = (
        fit_shared_kernel_gp(
            inputs[indices],
            cost_targets[indices, None],
            model["input_state_dimension"],
            config["state_length_scale"],
            config["action_length_scale"],
            signal_variance=config["signal_variance"],
            noise_std=config["regression_noise_std"],
            output_scale_floor=0.10,
        )
        if len(indices) > 0
        else None
    )
    return models


def _stratified_parent_ids(
    history,
    perturbation_count,
    initial_potential,
    maximum,
    rng,
    direct_parent_minimum=0,
    include_root_parent=False,
):
    """摂動回数×potential進行度を偏らせず親候補を抽出する。

    ``direct_parent_minimum`` は、直前のincoming edgeが摂動である状態の
    最低枠である。祖先に摂動があるだけのdeterministic子孫とは区別し、
    小さい摂動を連続させる候補が大きなhistoryに埋もれるのを防ぐ。
    """

    valid = [state.state_id for state in history if state.terminal_reason is None]
    if maximum <= 0 or len(valid) <= maximum:
        return valid

    forced = []
    if include_root_parent and 0 in valid:
        forced.append(0)

    direct = [
        state_id
        for state_id in valid
        if history[state_id].incoming_step_type == "perturbation"
        and state_id not in forced
    ]
    rng.shuffle(direct)
    direct_capacity = max(int(maximum) - len(forced), 0)
    forced.extend(
        direct[: min(int(direct_parent_minimum), direct_capacity, len(direct))]
    )
    forced_set = set(forced)
    remaining_valid = [state_id for state_id in valid if state_id not in forced_set]

    groups = defaultdict(list)
    scale = max(float(initial_potential), 1.0e-12)
    for state_id in remaining_valid:
        state = history[state_id]
        count_bin = min(int(perturbation_count.get(state_id, 0)), 4)
        progress = np.clip(1.0 - state.potential / scale, 0.0, 1.0)
        progress_bin = min(int(progress * 6), 5)
        groups[(count_bin, progress_bin)].append(state_id)

    keys = list(groups)
    rng.shuffle(keys)
    for key in keys:
        rng.shuffle(groups[key])
    selected = list(forced)
    offset = {key: 0 for key in keys}
    while len(selected) < maximum:
        changed = False
        for key in keys:
            if offset[key] >= len(groups[key]):
                continue
            selected.append(groups[key][offset[key]])
            offset[key] += 1
            changed = True
            if len(selected) >= maximum:
                break
        if not changed:
            break
    return selected


def _action_key(candidate, index_free):
    """同じStateNodeへ同じ物理方向を二度評価しないためのkey。"""

    vector = np.asarray(candidate["displacement"]).reshape(-1)[index_free]
    norm = np.linalg.norm(vector)
    if norm > 0.0:
        vector = vector / norm
    return (int(candidate["parent_id"]),) + tuple(np.round(vector, 8))


def _candidate_pool(
    history,
    perturbation_count,
    evaluated_actions,
    initial_potential,
    model,
    config,
    parent_rng,
    action_rng,
    mode_cache,
):
    parent_ids = _stratified_parent_ids(
        history,
        perturbation_count,
        initial_potential,
        config["candidate_parent_limit"],
        parent_rng,
        config["direct_parent_minimum"],
        config["include_root_parent"],
    )
    candidates = generate_tangent_candidates(
        history,
        parent_ids,
        model["member"],
        model["index_free"],
        config["directions_per_parent"],
        config["perturbation_scale"],
        config["reference_point"],
        model["length_scale"],
        action_rng,
        mode_cache=mode_cache,
        layer_threshold=config["layer_threshold"],
        include_transported_direction=config["include_transported_direction"],
        transport_min_projection_ratio=config["transport_min_projection_ratio"],
    )
    filtered = []
    for candidate in candidates:
        key = _action_key(candidate, model["index_free"])
        if key in evaluated_actions:
            # A transported arm replaces its nearest grid arm.  Once the
            # transported arm has been evaluated, restore that still-unevaluated
            # grid arm instead of silently reducing this parent's candidate
            # count by one.
            backup = candidate.get("replaced_grid_candidate")
            if backup is not None:
                backup_key = _action_key(backup, model["index_free"])
                if backup_key not in evaluated_actions:
                    backup["action_key"] = backup_key
                    filtered.append(backup)
            continue
        candidate["action_key"] = key
        filtered.append(candidate)
    return filtered


def _update_archive(
    elite_archive,
    history,
    terminal_state_id,
    model,
    config,
):
    return update_elite_archive(
        elite_archive,
        history,
        terminal_state_id,
        model["member"],
        model["length_scale"],
        config["layer_threshold"],
        config["overlap_tolerance"],
        config["separation_tolerance"],
        config["collinearity_tolerance"],
        config["parallel_tolerance"],
        config["rms_tolerance"],
        config["max_tolerance"],
        config["reference_point"],
    )


def _discovery_auc(discovery_curve, budget):
    """累積cell数の右連続階段関数を0..budgetで積分して正規化する。"""

    curve = sorted((int(step), int(count)) for step, count in discovery_curve)
    area = 0.0
    previous_step = 0
    count = 0
    for step, next_count in curve:
        step = min(max(step, previous_step), int(budget))
        area += (step - previous_step) * count
        previous_step = step
        count = next_count
        if previous_step >= budget:
            break
    area += max(int(budget) - previous_step, 0) * count
    return area / max(float(budget), 1.0)


def _post_trunk_auc(discovery_curve, trunk_end_step, trunk_cell_count, budget):
    """共通trunk後に追加されたcellだけの平均数を返す。"""

    remaining = int(budget) - int(trunk_end_step)
    if remaining <= 0:
        return 0.0
    shifted = [(0, 0)]
    for step, count in discovery_curve:
        if step > trunk_end_step:
            shifted.append(
                (int(step) - int(trunk_end_step), int(count) - int(trunk_cell_count))
            )
    return _discovery_auc(shifted, remaining)


def _stop_code(reason):
    return {
        "flat": 0,
        "collision": 1,
        "timeout": 2,
        "corrector_failure": 3,
        None: 4,
    }.get(reason, 5)


def _discovery_event(history, terminal_state_id, step, cell_index, selection_code):
    """初回発見経路を、後のelite置換とは独立に保存する。"""

    path = get_path_state_ids(history, int(terminal_state_id))
    perturbation_ids = [
        int(state_id)
        for state_id in path[1:]
        if history[int(state_id)].incoming_step_type == "perturbation"
    ]
    return {
        "cell_index": int(cell_index),
        "discovery_step": int(step),
        "terminal_state_id": int(terminal_state_id),
        "path_depth": int(history[int(terminal_state_id)].depth),
        "path_perturbation_count": int(len(perturbation_ids)),
        "perturbation_child_state_ids": perturbation_ids,
        "path_step_types": [
            history[int(state_id)].incoming_step_type for state_id in path[1:]
        ],
        "selection_code": int(selection_code),
    }


def _save_outputs(
    base_directory,
    output_directory,
    history,
    elite_archive,
    discovery_curve,
    random_edges,
    data,
    rollout_rows,
    discovery_events,
    config,
    model,
    used_steps,
    wall_time,
    fit_time,
    score_time,
    trunk_end_step,
    trunk_cell_count,
):
    reset_output_directory(output_directory)
    save_elite_paths(history, elite_archive, output_directory)

    curve_to_save = list(discovery_curve)
    if curve_to_save[-1][0] != config["total_step_budget"]:
        curve_to_save.append(
            (config["total_step_budget"], len(elite_archive))
        )
    save_discovery_curve(curve_to_save, output_directory)

    input_dimension = model["input_dimension"]
    output_dimension = data["output_dimension"]
    np.savez_compressed(
        output_directory / "training_data.npz",
        inputs=np.asarray(data["inputs"], dtype=float).reshape(-1, input_dimension),
        flat_labels=np.asarray(data["flat_labels"], dtype=np.int8),
        flatness_targets=np.asarray(data["flatness_targets"], dtype=float),
        cost_targets=np.asarray(data["cost_targets"], dtype=float),
        descriptor_targets=np.asarray(
            data["descriptor_targets"], dtype=float
        ).reshape(-1, output_dimension),
        rollout_ids=np.asarray(data["rollout_ids"], dtype=int),
        state_ids=np.asarray(data["state_ids"], dtype=int),
        edge_kinds=np.asarray(data["edge_kinds"], dtype="U16"),
        stop_reasons=np.asarray(data["stop_reasons"], dtype="U24"),
    )

    rollout_array = np.asarray(rollout_rows, dtype=float).reshape(-1, 33)
    np.savetxt(
        output_directory / "rollout_diagnostics.dat",
        rollout_array,
        delimiter=",",
        header=(
            "rollout_id,start_step,end_step,parent_id,parent_perturbations,"
            "spent_steps,stop_code,new_cell,selection_code,acquisition,"
            "p_flat,p_reach,mean_novelty,descriptor_uncertainty,"
            "posterior_p_new,num_training_edges,parent_is_direct_perturbation,"
            "base_acquisition,bonus_factor,perturbation_child_id,terminal_state_id,"
            "predictor_displacement_norm,corrected_displacement_norm,"
            "candidate_is_transported,transport_alignment,"
            "transport_projection_ratio,candidate_count,retained_count,"
            "filter_rejected_count,filter_bypass,selected_rank_fraction,"
            "audit_rejected,audit_would_be_new"
        ),
        comments="",
    )

    auc = _discovery_auc(discovery_curve, config["total_step_budget"])
    post_trunk_auc = _post_trunk_auc(
        discovery_curve,
        trunk_end_step,
        trunk_cell_count,
        config["total_step_budget"],
    )
    stop_codes = rollout_array[:, 6] if len(rollout_array) else np.empty(0)
    selection_codes = rollout_array[:, 8] if len(rollout_array) else np.empty(0)
    direct_parent_flags = (
        rollout_array[:, 16] if len(rollout_array) else np.empty(0)
    )
    transported_flags = (
        rollout_array[:, 23] if len(rollout_array) else np.empty(0)
    )
    selected_transport_alignment = (
        rollout_array[:, 24] if len(rollout_array) else np.empty(0)
    )
    nontrunk_selection_mask = selection_codes != 0
    elite_depths = [
        int(history[cell["terminal_state_id"]].depth)
        for cell in elite_archive
    ]
    random_edge_set = {
        (int(parent_id), int(child_id)) for parent_id, child_id in random_edges
    }
    elite_perturbation_counts = []
    for cell in elite_archive:
        path = get_path_state_ids(history, cell["terminal_state_id"])
        elite_perturbation_counts.append(
            int(
                sum(
                    (parent_id, child_id) in random_edge_set
                    for parent_id, child_id in zip(path[:-1], path[1:])
                )
            )
        )
    all_path_counts = state_path_perturbation_counts(history)
    all_consecutive_counts = state_consecutive_perturbation_counts(history)
    summary = {
        "algorithm_version": config["algorithm_version"],
        "selector": config["selector"],
        "seed": config["seed"],
        "step_budget": config["total_step_budget"],
        "used_steps": int(used_steps),
        "elite_cells": int(len(elite_archive)),
        "discovery_auc": float(auc),
        "post_trunk_new_cells": int(len(elite_archive) - trunk_cell_count),
        "post_trunk_discovery_auc": float(post_trunk_auc),
        "trunk_end_step": int(trunk_end_step),
        "trunk_cell_count": int(trunk_cell_count),
        "num_states": int(len(history)),
        "num_rollouts": int(len(rollout_rows)),
        "flat_rollouts": int(np.count_nonzero(stop_codes == 0)),
        "collision_rollouts": int(np.count_nonzero(stop_codes == 1)),
        "timeout_rollouts": int(np.count_nonzero(stop_codes == 2)),
        "failed_or_censored_rollouts": int(np.count_nonzero(stop_codes >= 3)),
        "cold_start_selections": int(np.count_nonzero(selection_codes == 5)),
        "gp_guided_selections": int(
            np.count_nonzero(
                (selection_codes == 3)
                | (selection_codes == 4)
                | (selection_codes == 8)
                | (selection_codes == 9)
                | (selection_codes == 10)
                | (selection_codes == 11)
            )
        ),
        "gp_filter_selections": int(np.count_nonzero(selection_codes == 8)),
        "shuffled_filter_selections": int(
            np.count_nonzero(selection_codes == 9)
        ),
        "cost_filter_selections": int(np.count_nonzero(selection_codes == 10)),
        "cost_shuffled_filter_selections": int(
            np.count_nonzero(selection_codes == 11)
        ),
        "gp_random_selections": int(np.count_nonzero(selection_codes == 2)),
        "transport_heuristic_selections": int(
            np.count_nonzero(selection_codes == 6)
        ),
        "transport_heuristic_random_selections": int(
            np.count_nonzero(selection_codes == 7)
        ),
        "direct_perturbation_parent_selections": int(
            np.count_nonzero(direct_parent_flags == 1)
        ),
        "transported_direction_selections": int(
            np.count_nonzero(transported_flags == 1)
        ),
        "mean_selected_transport_alignment": float(
            np.mean(selected_transport_alignment[nontrunk_selection_mask])
            if np.any(nontrunk_selection_mask)
            else 0.0
        ),
        "maximum_path_perturbation_count": int(
            np.max(all_path_counts) if len(all_path_counts) else 0
        ),
        "maximum_consecutive_perturbation_count": int(
            np.max(all_consecutive_counts) if len(all_consecutive_counts) else 0
        ),
        "num_training_edges": int(len(data["inputs"])),
        "gp_fit_training_edges": int(len(_gp_training_indices(data, config))),
        "flat_training_edges": int(np.count_nonzero(data["flat_labels"])),
        "gp_input_dimension": int(model["input_dimension"]),
        "gp_output_dimension": int(data["output_dimension"]),
        "elite_path_depths": elite_depths,
        "elite_perturbation_counts": elite_perturbation_counts,
        "discovery_perturbation_counts": [
            int(event["path_perturbation_count"]) for event in discovery_events
        ],
        "wall_time_seconds": float(wall_time),
        "gp_fit_time_seconds": float(fit_time),
        "gp_score_time_seconds": float(score_time),
    }
    (output_directory / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_directory / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_directory / "discovery_events.json").write_text(
        json.dumps(discovery_events, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if config["save_tree"]:
        try:
            save_search_tree(
                history=history,
                elite_archive=elite_archive,
                random_edges=random_edges,
                output_directory=output_directory,
                member=model["member"],
                reference_point=config["reference_point"],
                length_scale=model["length_scale"],
                state_rms_tolerance=config["state_cell_rms_tolerance"],
                state_max_tolerance=config["state_cell_max_tolerance"],
                layer_threshold=config["layer_threshold"],
                overlap_tolerance=config["overlap_tolerance"],
                separation_tolerance=config["separation_tolerance"],
                collinearity_tolerance=config["collinearity_tolerance"],
                parallel_tolerance=config["parallel_tolerance"],
                elite_rms_tolerance=config["rms_tolerance"],
                elite_max_tolerance=config["max_tolerance"],
            )
        except ValueError as error:
            # 現在のinteractive treeはfive-bar三折角表示。探索自体は任意Nで
            # 動くため、非five-barでは図だけ省略して理由を保存する。
            (output_directory / "search_tree_skipped.txt").write_text(
                str(error), encoding="utf-8"
            )
    return summary


def main():
    base_directory = Path(__file__).resolve().parent
    config = _configuration()
    _validate_configuration(config)
    node_initial, member, _, index_free, member_length = _load_model(
        base_directory / "model"
    )
    length_scale = float(np.mean(member_length))
    if not np.isfinite(length_scale) or length_scale <= 0.0:
        raise ValueError("the model must have positive finite member lengths")
    initial_potential = compute_gravity_potential(
        node_initial,
        config["reference_point"],
    )
    history = create_history(node_initial, initial_potential)
    model = {
        "member": member,
        "member_length": member_length,
        "index_free": index_free,
        "num_node": len(node_initial),
        "num_member": len(member),
        "length_scale": length_scale,
        # state block = 現在座標のfree成分 + 既知last-nonflat contextのfree-z。
        "input_state_dimension": int(
            len(index_free) + np.count_nonzero(index_free % 3 == 2)
        ),
        "input_dimension": int(
            2 * len(index_free) + np.count_nonzero(index_free % 3 == 2)
        ),
    }
    output_spec = make_terminal_output_spec(
        node_initial,
        index_free,
        config["reference_point"],
        length_scale,
    )

    # 乱数用途を分離し、候補集合の乱数とepsilon-randomの乱数が干渉しない。
    seed = config["seed"]
    parent_rng = np.random.default_rng(seed + 101)
    action_rng = np.random.default_rng(seed + 202)
    selection_rng = np.random.default_rng(seed + 303)
    posterior_rng = np.random.default_rng(seed + 404)
    np.random.seed(seed)

    used_steps = 0
    rollout_id = 0
    elite_archive = []
    discovery_curve = [(0, 0)]
    discovery_events = []
    random_edges = []
    perturbation_count = {0: 0}
    evaluated_actions = set()
    mode_cache = {}
    training_data = _new_training_data(len(output_spec["active_positions"]))
    rollout_rows = []
    surrogate_models = None
    completed_rollouts_since_fit = 0
    total_fit_time = 0.0
    total_score_time = 0.0
    start_time = time.perf_counter()

    # 一本の決定論経路だけで探索木を開始する。
    trunk_start = used_steps
    current, new_ids, spent, stop_reason = _initial_deterministic_trunk(
        history,
        config["total_step_budget"] - used_steps,
        model,
        config,
    )
    used_steps += spent
    for state_id in new_ids:
        perturbation_count[state_id] = 0

    is_new_cell = False
    if current is not None and stop_reason == "flat":
        is_new_cell = _update_archive(
            elite_archive,
            history,
            current.state_id,
            model,
            config,
        )
        if is_new_cell:
            discovery_curve.append((used_steps, len(elite_archive)))
            discovery_events.append(
                _discovery_event(
                    history,
                    current.state_id,
                    used_steps,
                    len(elite_archive) - 1,
                    0,
                )
            )

    _append_completed_rollout(
        training_data,
        history,
        new_ids,
        stop_reason,
        rollout_id,
        model,
        config,
        output_spec,
    )
    rollout_rows.append(
        [
            rollout_id,
            trunk_start,
            used_steps,
            -1,
            0,
            spent,
            _stop_code(stop_reason),
            int(is_new_cell),
            0,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            len(training_data["inputs"]),
            0,
            np.nan,
            1.0,
            -1,
            -1 if current is None else current.state_id,
            np.nan,
            np.nan,
            0,
            0.0,
            0.0,
            0,
            0,
            0,
            0,
            np.nan,
            np.nan,
            np.nan,
        ]
    )
    rollout_id += 1
    completed_rollouts_since_fit += int(stop_reason in ("flat", "collision"))

    if config["verbose"]:
        print(
            f"initial trunk: steps={used_steps}, stop={stop_reason}, "
            f"cells={len(elite_archive)}"
        )

    trunk_end_step = used_steps
    trunk_cell_count = len(elite_archive)
    completed_perturbation_rollouts = 0

    while used_steps < config["total_step_budget"]:
        candidates = _candidate_pool(
            history,
            perturbation_count,
            evaluated_actions,
            initial_potential,
            model,
            config,
            parent_rng,
            action_rng,
            mode_cache,
        )
        if not candidates and config["candidate_parent_limit"] > 0:
            # bounded pool内の1DOF方向を使い切った場合は、全historyも確認する。
            old_limit = config["candidate_parent_limit"]
            config["candidate_parent_limit"] = 0
            candidates = _candidate_pool(
                history,
                perturbation_count,
                evaluated_actions,
                initial_potential,
                model,
                config,
                parent_rng,
                action_rng,
                mode_cache,
            )
            config["candidate_parent_limit"] = old_limit
        if not candidates:
            break

        diagnostics = None
        bonus_details = None
        selection_code = 1
        candidate_index = None
        candidate_count = len(candidates)
        retained_count = len(candidates)
        filter_rejected_count = 0
        filter_bypass = 0
        selected_rank_fraction = np.nan
        audit_rejected = np.nan
        audit_would_be_new = np.nan

        cold_start = (
            completed_perturbation_rollouts < config["cold_start_rollouts"]
        )
        if config["selector"] == "random":
            candidate_index = int(selection_rng.integers(len(candidates)))
            selection_code = 1
        elif config["selector"] == "transport_heuristic":
            transported = np.asarray(
                [
                    index
                    for index, candidate in enumerate(candidates)
                    if candidate.get("is_transported_direction", False)
                ],
                dtype=int,
            )
            if (
                len(transported) > 0
                and selection_rng.random() >= config["epsilon_random"]
            ):
                candidate_index = int(selection_rng.choice(transported))
                selection_code = 6
            else:
                candidate_index = int(selection_rng.integers(len(candidates)))
                selection_code = 7
        elif cold_start:
            candidate_index = int(selection_rng.integers(len(candidates)))
            selection_code = 5
            filter_bypass = int(config["selector"].startswith("gp_filter"))
        else:
            if (
                surrogate_models is None
                or completed_rollouts_since_fit
                >= config["refit_rollout_interval"]
            ):
                fit_start = time.perf_counter()
                surrogate_models = _fit_models(training_data, model, config)
                total_fit_time += time.perf_counter() - fit_start
                completed_rollouts_since_fit = 0

            query_inputs = np.asarray(
                [candidate["input"] for candidate in candidates], dtype=float
            )
            score_start = time.perf_counter()
            diagnostics = score_acquisition(
                surrogate_models,
                query_inputs,
                elite_archive,
                output_spec,
                member,
                config["overlap_tolerance"],
                config["separation_tolerance"],
                config["collinearity_tolerance"],
                config["parallel_tolerance"],
                config["rms_tolerance"],
                config["max_tolerance"],
                flatness_target_threshold=np.log(2.0),
                acquisition_kind=(
                    "ucb"
                    if config["selector"] in (
                        "gp_ucb",
                        "gp_filter",
                        "gp_filter_shuffled",
                        "gp_filter_cost",
                        "gp_filter_cost_shuffled",
                    )
                    else "posterior_new"
                ),
                novelty_kappa=config["novelty_kappa"],
                reach_weight=config["reach_weight"],
                cold_uncertainty_weight=config["cold_uncertainty_weight"],
                layer_uncertainty_scale=config["layer_uncertainty_scale"],
                num_posterior_samples=config["posterior_samples"],
                rng=posterior_rng,
            )
            direct_bonus_details = apply_direct_perturbation_parent_bonus(
                diagnostics["score"],
                candidates,
                history,
                config["direct_parent_bonus"],
            )
            transport_bonus_details = apply_transport_alignment_bonus(
                direct_bonus_details["adjusted_score"],
                candidates,
                config["transport_alignment_bonus"],
            )
            bonus_details = {
                "base_score": direct_bonus_details["base_score"],
                "bonus_factor": (
                    direct_bonus_details["bonus_factor"]
                    * transport_bonus_details["bonus_factor"]
                ),
                "transport_bonus_factor": transport_bonus_details[
                    "bonus_factor"
                ],
            }
            diagnostics["base_score"] = bonus_details["base_score"]
            diagnostics["bonus_factor"] = bonus_details["bonus_factor"]
            diagnostics["score"] = transport_bonus_details["adjusted_score"]
            if (
                config["selector"] in (
                    "gp_filter_cost",
                    "gp_filter_cost_shuffled",
                )
                and surrogate_models.get("cost") is not None
            ):
                log_cost_mean, log_cost_variance = predict_shared_kernel_gp(
                    surrogate_models["cost"], query_inputs
                )
                log_cost_lcb = (
                    log_cost_mean[:, 0]
                    - config["cost_lcb_kappa"]
                    * np.sqrt(np.maximum(log_cost_variance[:, 0], 0.0))
                )
                cost_lcb = np.maximum(np.expm1(log_cost_lcb), 1.0)
                diagnostics["score"] = diagnostics["score"] / (
                    cost_lcb + config["cost_offset"]
                )
            total_score_time += time.perf_counter() - score_start

            if selection_rng.random() < config["epsilon_random"]:
                candidate_index = int(selection_rng.integers(len(candidates)))
                selection_code = 2  # GP run内の明示的random quota
                filter_bypass = int(config["selector"].startswith("gp_filter"))
            elif config["selector"] in (
                "gp_filter",
                "gp_filter_shuffled",
                "gp_filter_cost",
                "gp_filter_cost_shuffled",
            ):
                # 各parentの最良候補だけを比較し、上位quantileをfilterとして
                # 残す。残存parentからは一様に選ぶので、GP-UCBのglobal
                # argmaxが同じbranchへ集中する問題を分離できる。
                finite_score = np.nan_to_num(
                    diagnostics["score"], nan=-np.inf, neginf=-np.inf
                )
                ranking_score = finite_score
                if config["selector"] in (
                    "gp_filter_shuffled",
                    "gp_filter_cost_shuffled",
                ):
                    # score値の分布とGP計算量は保ったまま、候補との対応だけ
                    # 無作為化する負対照。filter構造・保持率による効果と、
                    # GP予測の順位情報による効果を分ける。
                    ranking_score = finite_score[
                        selection_rng.permutation(len(finite_score))
                    ]
                best_by_parent = {}
                for index, candidate_item in enumerate(candidates):
                    candidate_parent = int(candidate_item["parent_id"])
                    old_index = best_by_parent.get(candidate_parent)
                    if (
                        old_index is None
                        or ranking_score[index] > ranking_score[old_index]
                    ):
                        best_by_parent[candidate_parent] = int(index)
                parent_best_indices = np.asarray(
                    list(best_by_parent.values()), dtype=int
                )
                keep_count = max(
                    1,
                    int(
                        np.ceil(
                            config["filter_keep_fraction"]
                            * len(parent_best_indices)
                        )
                    ),
                )
                parent_order = parent_best_indices[
                    np.argsort(
                        -ranking_score[parent_best_indices], kind="stable"
                    )
                ]
                retained = parent_order[:keep_count]
                rejected_parent_best = parent_order[keep_count:]
                retained_count = int(len(retained))
                filter_rejected_count = int(candidate_count - retained_count)
                if (
                    len(rejected_parent_best) > 0
                    and selection_rng.random() < config["filter_audit_rate"]
                ):
                    candidate_index = int(
                        selection_rng.choice(rejected_parent_best)
                    )
                    audit_rejected = 1.0
                else:
                    candidate_index = int(selection_rng.choice(retained))
                    audit_rejected = 0.0
                selection_code = (
                    9 if config["selector"] == "gp_filter_shuffled"
                    else (
                        11
                        if config["selector"] == "gp_filter_cost_shuffled"
                        else (10 if config["selector"] == "gp_filter_cost" else 8)
                    )
                )
            else:
                finite_score = np.nan_to_num(
                    diagnostics["score"], nan=-np.inf, neginf=-np.inf
                )
                maximum = np.max(finite_score)
                tied = np.flatnonzero(
                    np.isclose(finite_score, maximum, rtol=1.0e-12, atol=1.0e-12)
                )
                candidate_index = int(selection_rng.choice(tied))
                selection_code = 3 if config["selector"] == "gp_ucb" else 4

        if diagnostics is not None:
            finite_for_rank = np.nan_to_num(
                diagnostics["score"], nan=-np.inf, neginf=-np.inf
            )
            rank_order = np.argsort(-finite_for_rank, kind="stable")
            rank_position = int(np.flatnonzero(rank_order == candidate_index)[0])
            selected_rank_fraction = rank_position / max(len(candidates) - 1, 1)

        candidate = candidates[candidate_index]
        evaluated_actions.add(candidate["action_key"])
        parent_id = int(candidate["parent_id"])
        parent_count = int(perturbation_count[parent_id])
        start_step = used_steps
        first_new_state_id = len(history)
        current, new_ids, spent, stop_reason, perturbation_child_ids = (
            _candidate_rollout(
                history,
                candidate,
                config["total_step_budget"] - used_steps,
                model,
                config,
            )
        )
        if current is None:
            break
        used_steps += spent
        perturbation_increment = len(perturbation_child_ids)
        for state_id in range(first_new_state_id, len(history)):
            perturbation_count[state_id] = parent_count + perturbation_increment
        perturbation_parent_id = parent_id
        for perturbation_child_id in perturbation_child_ids:
            random_edges.append((perturbation_parent_id, perturbation_child_id))
            perturbation_parent_id = perturbation_child_id

        is_new_cell = False
        if stop_reason == "flat":
            is_new_cell = _update_archive(
                elite_archive,
                history,
                current.state_id,
                model,
                config,
            )
            if is_new_cell:
                discovery_curve.append((used_steps, len(elite_archive)))
                discovery_events.append(
                    _discovery_event(
                        history,
                        current.state_id,
                        used_steps,
                        len(elite_archive) - 1,
                        selection_code,
                    )
                )

        if audit_rejected == 1.0:
            audit_would_be_new = float(is_new_cell)

        added = _append_completed_rollout(
            training_data,
            history,
            new_ids,
            stop_reason,
            rollout_id,
            model,
            config,
            output_spec,
        )
        if added:
            completed_rollouts_since_fit += 1
            completed_perturbation_rollouts += 1

        if diagnostics is None:
            diagnostic_values = [np.nan] * 6
        else:
            diagnostic_values = [
                diagnostics["score"][candidate_index],
                diagnostics["flat_probability"][candidate_index],
                diagnostics["reach_probability"][candidate_index],
                diagnostics["mean_novelty"][candidate_index],
                diagnostics["descriptor_uncertainty"][candidate_index],
                diagnostics["posterior_new_probability"][candidate_index],
            ]
        parent_is_direct = int(
            history[parent_id].incoming_step_type == "perturbation"
        )
        if bonus_details is None:
            base_acquisition = np.nan
            bonus_factor = 1.0
        else:
            base_acquisition = bonus_details["base_score"][candidate_index]
            bonus_factor = bonus_details["bonus_factor"][candidate_index]
        predictor_norm = float(np.linalg.norm(candidate["displacement"]))
        first_perturbation_child_id = int(perturbation_child_ids[0])
        corrected_norm = float(
            np.linalg.norm(
                history[first_perturbation_child_id].node
                - history[parent_id].node
            )
        )
        candidate_is_transported = int(
            candidate.get("is_transported_direction", False)
        )
        transport_alignment = float(
            candidate.get("transport_alignment", 0.0)
        )
        transport_projection_ratio = float(
            candidate.get("transport_projection_ratio", 0.0)
        )
        rollout_rows.append(
            [
                rollout_id,
                start_step,
                used_steps,
                parent_id,
                parent_count,
                spent,
                _stop_code(stop_reason),
                int(is_new_cell),
                selection_code,
                *diagnostic_values,
                len(training_data["inputs"]),
                parent_is_direct,
                base_acquisition,
                bonus_factor,
                first_perturbation_child_id,
                current.state_id,
                predictor_norm,
                corrected_norm,
                candidate_is_transported,
                transport_alignment,
                transport_projection_ratio,
                candidate_count,
                retained_count,
                filter_rejected_count,
                filter_bypass,
                selected_rank_fraction,
                audit_rejected,
                audit_would_be_new,
            ]
        )
        rollout_id += 1

        if config["verbose"] and (
            is_new_cell or used_steps == config["total_step_budget"]
        ):
            print(
                f"steps={used_steps}, stop={stop_reason}, "
                f"cells={len(elite_archive)}, training_edges={len(training_data['inputs'])}"
            )

    wall_time = time.perf_counter() - start_time
    output_directory = (
        base_directory
        / "result"
        / config["result_name"]
        / f"seed_{config['seed']:03d}"
    )
    summary = _save_outputs(
        base_directory,
        output_directory,
        history,
        elite_archive,
        discovery_curve,
        random_edges,
        training_data,
        rollout_rows,
        discovery_events,
        config,
        model,
        used_steps,
        wall_time,
        total_fit_time,
        total_score_time,
        trunk_end_step,
        trunk_cell_count,
    )
    print(json.dumps(summary, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    main()
