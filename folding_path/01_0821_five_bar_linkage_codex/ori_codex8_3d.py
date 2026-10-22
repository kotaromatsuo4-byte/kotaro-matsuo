"""Three-dimensional adapter for the additive Codex8 reward ablation."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys

import numpy as np

import collision_3d
import ori_codex2 as mechanics
import ori_codex6_rejected_crossing as rejected
import ori_codex6_rejected_crossing_3d as rejected3d
import ori_codex6_zero_reject as zero_reject
import ori_codex6_zero_reject_3d as zero3d
import ori_codex8 as ori8


ADAPTER_NAME = "ori_codex8_3d"
ADAPTER_VERSION = "1.0.0"

load_problem_data = rejected3d.load_problem_data
support_compatible_rigid_basis = rejected3d.support_compatible_rigid_basis
add_rigid_motion_diagnostics = rejected3d.add_rigid_motion_diagnostics

_BASE_ORI8_MAKE_ENVIRONMENT = ori8.make_environment
_BASE_ORI8_RUN_EPISODE = ori8.run_episode
_BASE_ORI8_AGGREGATE = ori8.aggregate_stochastic_evaluations


def make_environment(args, node, member, fix):
    node = np.asarray(node, dtype=float)
    if node.ndim != 2 or node.shape[1] != 3:
        raise ValueError(f"{ADAPTER_NAME} can only construct a 3-D environment")
    environment = _BASE_ORI8_MAKE_ENVIRONMENT(args, node, member, fix)
    environment.rigid_motion_diagnostics_enabled = bool(
        getattr(args, "rigid_motion_diagnostics", False)
    )
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _BASE_ORI8_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    return add_rigid_motion_diagnostics(env, summary, nodes, actions)


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _BASE_ORI8_AGGREGATE(evaluations, episode=episode)
    if result is None:
        return None
    result["rigid_motion_diagnostics_enabled_rate"] = float(np.mean([
        row.get("rigid_motion_diagnostics_enabled", 0)
        for row in evaluations
    ]))
    result["rigid_motion_diagnostics_applicable_rate"] = float(np.mean([
        row.get("rigid_motion_diagnostics_applicable", 0)
        for row in evaluations
    ]))
    for field, aggregate_name in {
        "mean_commanded_rigid_energy_fraction": (
            "mean_commanded_rigid_energy_fraction"
        ),
        "mean_actual_rigid_energy_fraction": (
            "mean_actual_rigid_energy_fraction"
        ),
        "valid_path_rigid_energy_fraction": (
            "mean_valid_path_rigid_energy_fraction"
        ),
        "total_rigid_path_length_normalised": (
            "mean_total_rigid_path_length_normalised"
        ),
        "total_internal_path_length_normalised": (
            "mean_total_internal_path_length_normalised"
        ),
    }.items():
        values = [
            float(row[field]) for row in evaluations
            if row.get(field) is not None
        ]
        result[aggregate_name] = float(np.mean(values)) if values else None
    return result


def _build_metadata(args, node, member, fix, files):
    metadata = rejected3d._build_metadata(args, node, member, fix, files)
    metadata.update({
        "adapter": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
        "ori8_candidate": ori8.CANDIDATE_NAME,
        "ori8_candidate_schema_version": ori8.CANDIDATE_SCHEMA_VERSION,
        "ori8_parent_candidate": zero_reject.CANDIDATE_NAME,
        "ori8_reward_formula": copy.deepcopy(ori8.REWARD_FORMULA),
        "ori8_reward_formula_sha256": ori8.REWARD_FORMULA_SHA256,
        "ori8_transition_formula": copy.deepcopy(ori8.TRANSITION_FORMULA),
        "ori8_transition_formula_sha256": ori8.TRANSITION_FORMULA_SHA256,
        "reward_parameters": ori8.reward_parameters(args),
        "observation_contract": ori8.OBSERVATION_CONTRACT,
        "pomdp_caveat": ori8.POMDP_CAVEAT,
        "parent_sources_unmodified": True,
    })
    return metadata


@contextmanager
def patched_three_dimensional_runtime(args, member):
    pairs = collision_3d.nonadjacent_member_pairs(member)
    collision_callback = collision_3d.make_legacy_signature_checker(
        member,
        pair_indices=pairs,
        distance_tolerance=float(args.crossing_tolerance),
    )
    original = (
        mechanics.has_proper_self_crossing,
        ori8.make_environment,
        ori8.run_episode,
        ori8.aggregate_stochastic_evaluations,
    )
    mechanics.has_proper_self_crossing = collision_callback
    ori8.make_environment = make_environment
    ori8.run_episode = run_episode
    ori8.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    try:
        yield collision_callback
    finally:
        (
            mechanics.has_proper_self_crossing,
            ori8.make_environment,
            ori8.run_episode,
            ori8.aggregate_stochastic_evaluations,
        ) = original


def train(args):
    ori8._validate_ori8_args(args)
    if int(args.flatness_subspace_dimension) != 2:
        raise ValueError(
            "The Codex8 3-D adapter requires --flatness-subspace-dimension 2"
        )
    node, member, fix, files = load_problem_data(args.data_dir)
    args.data_dir = str(Path(args.data_dir).resolve())
    if not hasattr(args, "selected_model"):
        args.selected_model = Path(args.data_dir).name
    metadata = _build_metadata(args, node, member, fix, files)
    args.three_dimensional_adapter = metadata

    with patched_three_dimensional_runtime(args, member):
        output_directory, summary = ori8.train(args)

    output_directory = Path(output_directory)
    summary["three_dimensional_adapter"] = metadata
    with open(output_directory / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config.update({
        "three_dimensional_adapter": metadata,
        "effective_output_directory": str(output_directory.resolve()),
        "candidate_entrypoint": Path(__file__).name,
    })
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    return output_directory, summary


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    selected, remaining = ori8._ori8_parser().parse_known_args(arguments)
    if not rejected._option_was_supplied(remaining, zero_reject.CLI_FLAG):
        remaining = [zero_reject.CLI_FLAG, *remaining]
    if not rejected._option_was_supplied(
        remaining, "--ori6-spectral-weight"
    ):
        remaining = ["--ori6-spectral-weight", "0", *remaining]
    args = zero3d.parse_args(remaining)
    ori8._bind_ori8_args(args, selected, arguments)
    if not rejected._option_was_supplied(arguments, "--output-dir"):
        base_name = Path(args.output_dir).name
        args.output_dir = str(Path("ori_codex8_3d_results") / base_name)
    return args


if __name__ == "__main__":
    train(parse_args())
