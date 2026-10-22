"""同じ物理方向を局所接空間へ射影し直す連続小摂動emitterの診断。

rootの2次元メカニズム空間で一方向を選び、各摂動後の状態でその物理方向を
新しい接空間へ射影し直す。指定回数の小摂動を一stepずつcorrector・collision
判定しながら加えた後、通常の重力ポテンシャル最小化rolloutを行う。

終端targetは候補生成にも方向選択にも使わない。root方向は円周上の等角度grid
であり、chain length 1と2に同じ角度集合を使う。これは探索手法そのものでは
なく、大きいroot摂動を複数の連続な小摂動へ置き換えられるかを調べる診断である。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from archive import update_elite_archive
from mechanics import calc_dedx, calc_mode, compute_gravity_potential
from roll import deterministic_roll, displacement_roll
from state import create_history, get_path_state_ids


def load_model(model_directory):
    node = np.atleast_2d(
        np.loadtxt(model_directory / "node_initial.dat", delimiter=",", dtype=float)
    )
    member = np.atleast_2d(
        np.loadtxt(model_directory / "member.dat", delimiter=",", dtype=int)
    )
    fix = np.atleast_2d(
        np.loadtxt(model_directory / "fix.dat", delimiter=",", dtype=int)
    )
    index_all = np.arange(node.size)
    index_free = np.delete(index_all, 3 * fix[:, 0] + fix[:, 1])
    member_length = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]],
        axis=1,
    )
    return {
        "node": node,
        "member": member,
        "index_free": index_free,
        "member_length": member_length,
        "num_node": len(node),
        "num_member": len(member),
        "length_scale": float(np.mean(member_length)),
    }


def tangent_mode(node, model):
    jacobian = calc_dedx(
        node,
        model["num_node"],
        model["member"],
        model["num_member"],
        model["index_free"],
    )
    return calc_mode(model["index_free"], jacobian, model["num_node"])


def root_reference_directions(model, num_directions, phase_fraction=0.5):
    """rootの2DOF接空間における等角度の物理単位方向を返す。"""

    mode = tangent_mode(model["node"], model)
    if mode.shape[1] != 2:
        raise ValueError(
            "five_bar_persistent_emitter requires a two-dimensional root mode"
        )
    angles = 2.0 * np.pi * (
        np.arange(int(num_directions), dtype=float) + float(phase_fraction)
    ) / int(num_directions)
    coefficients = np.column_stack((np.cos(angles), np.sin(angles)))
    directions = (mode @ coefficients.T).T.reshape(-1, model["num_node"], 3)
    return angles, directions


def projected_displacement(node, reference_direction, scale, model):
    """rootで定めた物理方向を現在状態の接空間へ射影し、scaleへ規格化する。"""

    mode = tangent_mode(node, model)
    reference = np.asarray(reference_direction, dtype=float).reshape(-1)
    projection = mode @ (mode.T @ reference)
    norm = float(np.linalg.norm(projection))
    if norm <= 1.0e-12:
        return None, norm
    return (float(scale) * projection / norm).reshape(model["num_node"], 3), norm


def deterministic_roll_once(history, parent_id, model, parameters):
    return deterministic_roll(
        history,
        parent_id,
        model["member"],
        model["num_node"],
        model["num_member"],
        model["member_length"],
        model["index_free"],
        parameters["target_norm_max"],
        parameters["constraint_tolerance"],
        parameters["beta0"],
        parameters["corrector_iterations"],
        parameters["line_search_iterations"],
        parameters["line_search_tau"],
        parameters["cma_population"],
        parameters["cma_generations"],
        parameters["cma_seed"],
        parameters["reference_point"],
        parameters["flatness_tolerance"],
        parameters["dist_tol"],
        parameters["normal_tol"],
        parameters["cos_tol"],
        "planar",
    )


def perturbation_roll_once(history, parent_id, displacement, model, parameters):
    return displacement_roll(
        history,
        parent_id,
        displacement,
        model["member"],
        model["num_node"],
        model["num_member"],
        model["member_length"],
        model["index_free"],
        parameters["constraint_tolerance"],
        parameters["beta0"],
        parameters["corrector_iterations"],
        parameters["line_search_iterations"],
        parameters["line_search_tau"],
        parameters["reference_point"],
        parameters["flatness_tolerance"],
        parameters["dist_tol"],
        parameters["normal_tol"],
        parameters["cos_tol"],
        "planar",
    )


def archive_update(elite_archive, history, terminal_state_id, model, parameters):
    return update_elite_archive(
        elite_archive,
        history,
        terminal_state_id,
        model["member"],
        model["length_scale"],
        parameters["layer_threshold"],
        parameters["overlap_tolerance"],
        parameters["separation_tolerance"],
        parameters["collinearity_tolerance"],
        parameters["parallel_tolerance"],
        parameters["rms_tolerance"],
        parameters["max_tolerance"],
        parameters["reference_point"],
    )


def perturbation_count_on_path(history, terminal_state_id):
    return sum(
        history[state_id].incoming_step_type == "perturbation"
        for state_id in get_path_state_ids(history, terminal_state_id)
    )


def execute_persistent_rollout(
    history,
    reference_direction,
    scale,
    chain_length,
    model,
    parameters,
):
    """rootから連続小摂動を加え、非終端なら決定論的に終端まで進める。"""

    current = history[0]
    perturbation_edges = []
    projection_norms = []
    proposed_norms = []
    corrected_norms = []
    used_steps = 0

    for _ in range(int(chain_length)):
        displacement, projection_norm = projected_displacement(
            current.node,
            reference_direction,
            scale,
            model,
        )
        projection_norms.append(float(projection_norm))
        if displacement is None:
            return current, used_steps, "projection_failure", {
                "perturbation_edges": perturbation_edges,
                "projection_norms": projection_norms,
                "proposed_norms": proposed_norms,
                "corrected_norms": corrected_norms,
            }

        parent = current
        current = perturbation_roll_once(
            history,
            parent.state_id,
            displacement,
            model,
            parameters,
        )
        perturbation_edges.append([parent.state_id, current.state_id])
        proposed_norms.append(float(np.linalg.norm(displacement)))
        corrected_norms.append(float(np.linalg.norm(current.node - parent.node)))
        used_steps += 1
        if current.terminal_reason is not None:
            return current, used_steps, current.terminal_reason, {
                "perturbation_edges": perturbation_edges,
                "projection_norms": projection_norms,
                "proposed_norms": proposed_norms,
                "corrected_norms": corrected_norms,
            }

    while (
        current.terminal_reason is None
        and used_steps < parameters["max_rollout_steps"]
    ):
        current = deterministic_roll_once(
            history,
            current.state_id,
            model,
            parameters,
        )
        used_steps += 1

    stop_reason = current.terminal_reason
    if stop_reason is None:
        stop_reason = "timeout"
    return current, used_steps, stop_reason, {
        "perturbation_edges": perturbation_edges,
        "projection_norms": projection_norms,
        "proposed_norms": proposed_norms,
        "corrected_norms": corrected_norms,
    }


def run_configuration(model, scale, chain_length, num_directions, parameters):
    """一つのscale・chain lengthについて全root方向を評価する。"""

    initial_potential = compute_gravity_potential(
        model["node"],
        parameters["reference_point"],
    )
    history = create_history(model["node"], initial_potential)
    elite_archive = []
    angles, directions = root_reference_directions(model, num_directions)
    rollout_rows = []
    mechanics_steps = 0
    discovery_curve = [[0, 0]]

    for direction_index, (angle, direction) in enumerate(zip(angles, directions)):
        current, spent, stop_reason, diagnostics = execute_persistent_rollout(
            history,
            direction,
            scale,
            chain_length,
            model,
            parameters,
        )
        mechanics_steps += spent
        new_cell = False
        if stop_reason == "flat":
            new_cell = archive_update(
                elite_archive,
                history,
                current.state_id,
                model,
                parameters,
            )
            if new_cell:
                discovery_curve.append([mechanics_steps, len(elite_archive)])

        rollout_rows.append(
            {
                "direction_index": direction_index,
                "angle_radians": float(angle),
                "spent_steps": spent,
                "cumulative_mechanics_steps": mechanics_steps,
                "stop_reason": stop_reason,
                "new_cell": bool(new_cell),
                "num_cells": len(elite_archive),
                **diagnostics,
            }
        )
        if len(elite_archive) >= parameters["target_cell_count"]:
            break

    terminal_counts = {}
    for row in rollout_rows:
        terminal_counts[row["stop_reason"]] = terminal_counts.get(row["stop_reason"], 0) + 1
    elite_perturbation_counts = [
        int(
            perturbation_count_on_path(
                history,
                cell["terminal_state_id"],
            )
        )
        for cell in elite_archive
    ]
    elite_cells = []
    for cell_index, cell in enumerate(elite_archive):
        terminal_state_id = int(cell["terminal_state_id"])
        elite_cells.append(
            {
                "cell_index": cell_index,
                "terminal_state_id": terminal_state_id,
                "x_signature": np.rint(
                    history[terminal_state_id].node[:, 0]
                ).astype(int).tolist(),
                "layer_key": [
                    list(map(int, entry)) for entry in cell["layer_key"]
                ],
                "perturbation_count": int(
                    perturbation_count_on_path(history, terminal_state_id)
                ),
            }
        )
    all_corrected_norms = [
        value
        for row in rollout_rows
        for value in row["corrected_norms"]
    ]
    all_projection_norms = [
        value
        for row in rollout_rows
        for value in row["projection_norms"]
    ]

    return {
        "scale": float(scale),
        "chain_length": int(chain_length),
        "num_directions_available": int(num_directions),
        "num_directions_executed": len(rollout_rows),
        "mechanics_steps": mechanics_steps,
        "num_cells": len(elite_archive),
        "reached_target_cell_count": (
            len(elite_archive) >= parameters["target_cell_count"]
        ),
        "terminal_counts": terminal_counts,
        "elite_perturbation_counts": elite_perturbation_counts,
        "maximum_corrected_perturbation_norm": max(
            all_corrected_norms,
            default=0.0,
        ),
        "minimum_reference_projection_norm": min(
            all_projection_norms,
            default=0.0,
        ),
        "elite_cells": elite_cells,
        "discovery_curve": discovery_curve,
        "rollouts": rollout_rows,
    }


def default_parameters():
    return {
        "reference_point": 0.0,
        "constraint_tolerance": 1.0e-10,
        "beta0": 1.0,
        "corrector_iterations": 10,
        "line_search_iterations": 10,
        "line_search_tau": 0.5,
        "cma_population": 6,
        "cma_generations": 10,
        "cma_seed": 0,
        "target_norm_max": 0.10,
        "max_rollout_steps": 100,
        "flatness_tolerance": 0.05,
        "dist_tol": 0.05,
        "normal_tol": 0.10,
        "cos_tol": 0.99,
        "layer_threshold": 0.10,
        "overlap_tolerance": 0.05,
        "separation_tolerance": 1.0e-5,
        "collinearity_tolerance": 0.02,
        "parallel_tolerance": 0.02,
        "rms_tolerance": 0.02,
        "max_tolerance": 0.05,
        "target_cell_count": 6,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scales", type=float, nargs="+", default=[0.15, 0.20])
    parser.add_argument("--chain-lengths", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--num-directions", type=int, default=72)
    parser.add_argument(
        "--output-directory",
        default="result/persistent_direction_emitter",
    )
    return parser.parse_args()


def summary_as_markdown(report):
    lines = [
        "# Persistent-direction emitter diagnostic",
        "",
        "rootで選んだ物理方向を各更新状態の局所接空間へ射影し直し、targetを",
        "使わずに連続小摂動を加えた結果である。chain length 1は同じ方向gridを",
        "用いた比較基準である。",
        "",
        "| scale | chain length | directions | mechanics steps | cells | reached 6 | flat | collision | max actual perturbation norm |",
        "|---:|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for result in report["results"]:
        terminal = result["terminal_counts"]
        lines.append(
            f"| {result['scale']:.2f} | {result['chain_length']} | "
            f"{result['num_directions_executed']} | {result['mechanics_steps']} | "
            f"{result['num_cells']} | {result['reached_target_cell_count']} | "
            f"{terminal.get('flat', 0)} | {terminal.get('collision', 0)} | "
            f"{result['maximum_corrected_perturbation_norm']:.6f} |"
        )
    lines.extend(
        [
            "",
            "`mechanics steps` は摂動stepと決定論stepを同じ1 stepとして数える。",
            "6 cell未到達は可到達性の否定ではなく、この固定方向emitterと有限角度grid",
            "では回収できなかったことだけを意味する。",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    base_directory = Path(__file__).resolve().parent
    arguments = parse_arguments()
    parameters = default_parameters()
    model = load_model(base_directory / "model")

    results = []
    for scale in arguments.scales:
        for chain_length in arguments.chain_lengths:
            result = run_configuration(
                model,
                scale,
                chain_length,
                arguments.num_directions,
                parameters,
            )
            results.append(result)
            print(
                f"scale={scale:.2f}, K={chain_length}: "
                f"cells={result['num_cells']}, "
                f"steps={result['mechanics_steps']}"
            )

    report = {
        "method": "persistent physical direction reprojected to each local tangent space",
        "parameters": parameters,
        "results": results,
    }
    output_directory = Path(arguments.output_directory)
    if not output_directory.is_absolute():
        output_directory = base_directory / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "persistent_emitter_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_directory / "persistent_emitter_report.md").write_text(
        summary_as_markdown(report),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
