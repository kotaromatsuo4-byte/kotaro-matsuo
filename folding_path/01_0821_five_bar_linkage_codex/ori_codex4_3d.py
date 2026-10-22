"""Three-dimensional experiment adapter for :mod:`ori_codex4`.

This module deliberately leaves the ori_codex4 state, action, reward, and
Actor--Critic update untouched.  It only selects three-dimensional model
defaults, installs the continuous 3-D bar-crossing detector for the duration
of a run, and adds logging-only rigid-motion diagnostics.

Examples
--------
Four-unit torus (target planar diameter / maximum member length = 2)::

    python ori_codex4_3d.py --model 4unit_torus --target-spread 2

Waterbomb target diameters 3 and 3 sqrt(2), respectively::

    python ori_codex4_3d.py --model waterbomb --target-spread 1
    python ori_codex4_3d.py --model waterbomb --target-spread 1.4142135623730951

Unless ``--max-step-width`` is supplied, the predictor width is one tenth of
the largest initial member length.  Every ordinary ori_codex4/ori_codex3 CLI
option remains available and a user-supplied option takes precedence.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import uuid

import numpy as np

import collision_3d
import ori_codex2 as mechanics
import ori_codex4 as ori4


_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
_ORIGINAL_ORI4_MAKE_ENVIRONMENT = ori4.make_environment
_ORIGINAL_ORI4_RUN_EPISODE = ori4.run_episode
_ORIGINAL_ORI4_AGGREGATE_EVALUATIONS = (
    ori4.aggregate_stochastic_evaluations
)
_MODEL_DEFAULT_TARGETS = {
    "4unit_torus": 2.0,
    "waterbomb": 1.0,
}


def _option_was_supplied(arguments, option):
    return any(
        argument == option or argument.startswith(f"{option}=")
        for argument in arguments
    )


def _safe_slug(value):
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_.-")
    return text or "model"


def _number_slug(value):
    return f"{float(value):.8g}".replace("-", "m").replace(".", "p")


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_problem_data(data_directory):
    """Load and validate one 3-D truss model without changing its supports."""

    data_directory = Path(data_directory).resolve()
    required = {
        "node": data_directory / "node_initial.dat",
        "member": data_directory / "member.dat",
        "fix": data_directory / "fix.dat",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The 3-D model is missing required data files: " + ", ".join(missing)
        )
    node = np.atleast_2d(np.loadtxt(required["node"], delimiter=","))
    member = np.atleast_2d(np.loadtxt(
        required["member"], delimiter=",", dtype=int
    ))
    fix = np.atleast_2d(np.loadtxt(
        required["fix"], delimiter=",", dtype=int
    ))
    if node.ndim != 2 or node.shape[1] != 3:
        raise ValueError("ori_codex4_3d requires node coordinates with shape (N, 3).")
    if member.ndim != 2 or member.shape[1] != 2:
        raise ValueError("member.dat must have two node indices per member.")
    if fix.ndim != 2 or fix.shape[1] != 2:
        raise ValueError("fix.dat must have node and coordinate index columns.")
    if not np.all(np.isfinite(node)):
        raise ValueError("node_initial.dat contains a non-finite coordinate.")
    if len(member) and (np.min(member) < 0 or np.max(member) >= len(node)):
        raise ValueError("member.dat contains a node index outside node_initial.dat.")
    if len(fix) and (
        np.min(fix[:, 0]) < 0
        or np.max(fix[:, 0]) >= len(node)
        or np.min(fix[:, 1]) < 0
        or np.max(fix[:, 1]) >= 3
    ):
        raise ValueError("fix.dat contains an invalid node or coordinate index.")
    return node, member, fix, required


def _maximum_initial_member_length(node, member):
    if not len(member):
        raise ValueError("At least one member is required.")
    lengths = np.linalg.norm(node[member[:, 0]] - node[member[:, 1]], axis=1)
    maximum = float(np.max(lengths))
    if not np.isfinite(maximum) or maximum <= 0.0:
        raise ValueError("The largest initial member length must be positive.")
    return maximum


def _orthonormal_columns(matrix, relative_tolerance=1.0e-10):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("matrix must be two-dimensional.")
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular_values) or singular_values[0] <= 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    tolerance = max(
        relative_tolerance * singular_values[0],
        np.finfo(float).eps * max(matrix.shape) * singular_values[0],
    )
    rank = int(np.sum(singular_values > tolerance))
    return left[:, :rank]


def _coefficient_nullspace(matrix, number_of_columns):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape[0] == 0:
        return np.eye(number_of_columns, dtype=float)
    _, singular_values, right_transpose = np.linalg.svd(
        matrix, full_matrices=True
    )
    largest = float(singular_values[0]) if len(singular_values) else 0.0
    tolerance = np.finfo(float).eps * max(matrix.shape) * max(largest, 1.0)
    rank = int(np.sum(singular_values > tolerance))
    return right_transpose[rank:, :].T.copy()


def support_compatible_rigid_basis(node, fix):
    """Return an orthonormal basis of infinitesimal rigid motions allowed by fix.

    The six ambient generators are three translations and three rotations.
    Their coefficient combinations are restricted only by the exact scalar
    degrees of freedom listed in ``fix.dat``.  No additional support is added.
    For the supplied torus this produces its three rotations about fixed node
    10; for the corrected waterbomb it produces rank zero.
    """

    node = np.asarray(node, dtype=float)
    fix = np.asarray(fix, dtype=int).reshape(-1, 2)
    if node.ndim != 2 or node.shape[1] != 3:
        raise ValueError("node must have shape (N, 3).")
    coordinate_size = node.size
    origin = np.mean(node, axis=0)
    generators = np.zeros((coordinate_size, 6), dtype=float)
    for coordinate in range(3):
        generators[coordinate::3, coordinate] = 1.0
    relative = node - origin
    axes = np.eye(3)
    for axis_index, axis in enumerate(axes):
        generators[:, 3 + axis_index] = np.cross(
            np.broadcast_to(axis, relative.shape), relative
        ).reshape(-1)

    fixed_indices = np.asarray(
        [3 * int(node_index) + int(coordinate) for node_index, coordinate in fix],
        dtype=int,
    )
    coefficient_basis = _coefficient_nullspace(
        generators[fixed_indices, :], generators.shape[1]
    )
    compatible = generators @ coefficient_basis
    if len(fixed_indices):
        compatible[fixed_indices, :] = 0.0
    return _orthonormal_columns(compatible)


def _projection_components(vector, basis):
    vector = np.asarray(vector, dtype=float).reshape(-1)
    basis = np.asarray(basis, dtype=float)
    if basis.shape[1]:
        rigid = basis @ (basis.T @ vector)
    else:
        rigid = np.zeros_like(vector)
    internal = vector - rigid
    norm_squared = float(np.dot(vector, vector))
    rigid_squared = float(np.dot(rigid, rigid))
    fraction = rigid_squared / norm_squared if norm_squared > 1.0e-24 else 0.0
    return rigid, internal, float(np.clip(fraction, 0.0, 1.0))


def add_rigid_motion_diagnostics(env, summary, nodes, actions):
    """Add diagnostics after a rollout; never affect mechanics or reward."""

    enabled = bool(getattr(env, "rigid_motion_diagnostics_enabled", False))
    summary["rigid_motion_diagnostics_enabled"] = int(enabled)
    if not enabled:
        summary.update({
            "rigid_motion_diagnostics_applicable": 0,
            "minimum_support_compatible_rigid_rank": None,
            "maximum_support_compatible_rigid_rank": None,
            "mean_commanded_rigid_energy_fraction": None,
            "mean_actual_rigid_energy_fraction": None,
            "valid_path_rigid_energy_fraction": None,
            "total_rigid_path_length_normalised": None,
            "total_internal_path_length_normalised": None,
        })
        return summary, nodes, actions

    commanded_fractions = []
    actual_fractions = []
    ranks = []
    valid_rigid_length = 0.0
    valid_internal_length = 0.0
    valid_rigid_energy = 0.0
    valid_actual_energy = 0.0
    scale = max(float(env.coordinate_scale), 1.0e-12)

    for index, action in enumerate(actions):
        before = np.asarray(nodes[index], dtype=float)
        after = np.asarray(nodes[index + 1], dtype=float)
        rigid_basis = support_compatible_rigid_basis(before, env.fix)
        rigid_rank = int(rigid_basis.shape[1])
        ranks.append(rigid_rank)

        commanded = np.asarray(action["physical_direction"], dtype=float)
        commanded_rigid, commanded_internal, commanded_fraction = (
            _projection_components(commanded, rigid_basis)
        )
        actual = (after - before).reshape(-1)
        actual_rigid, actual_internal, actual_fraction = (
            _projection_components(actual, rigid_basis)
        )
        commanded_fractions.append(commanded_fraction)
        actual_fractions.append(actual_fraction)

        modes = np.asarray(action.get("mode_basis", []), dtype=float)
        if modes.ndim == 2 and modes.shape[0] == before.size:
            mechanism_basis = _orthonormal_columns(modes)
        else:
            mechanism_basis = np.zeros((before.size, 0), dtype=float)
        if rigid_rank and mechanism_basis.shape[1]:
            capture_values = np.linalg.svd(
                rigid_basis.T @ mechanism_basis, compute_uv=False
            )
            captured_rank = int(np.sum(capture_values >= 1.0 - 1.0e-7))
            minimum_capture = float(np.min(capture_values))
        else:
            captured_rank = 0
            minimum_capture = None

        valid_transition = bool(action.get("after_state_valid", 1))
        rigid_length = float(np.linalg.norm(actual_rigid))
        internal_length = float(np.linalg.norm(actual_internal))
        actual_energy = float(np.dot(actual, actual))
        rigid_energy = float(np.dot(actual_rigid, actual_rigid))
        if valid_transition:
            valid_rigid_length += rigid_length
            valid_internal_length += internal_length
            valid_actual_energy += actual_energy
            valid_rigid_energy += rigid_energy

        action.update({
            "rigid_diagnostic_valid_transition": int(valid_transition),
            "support_compatible_rigid_rank": rigid_rank,
            "mechanism_rigid_mode_rank": captured_rank,
            "minimum_rigid_mode_capture": minimum_capture,
            "commanded_rigid_energy_fraction": commanded_fraction,
            "commanded_internal_energy_fraction": 1.0 - commanded_fraction,
            "commanded_rigid_direction_norm": float(
                np.linalg.norm(commanded_rigid)
            ),
            "commanded_internal_direction_norm": float(
                np.linalg.norm(commanded_internal)
            ),
            "actual_rigid_energy_fraction": actual_fraction,
            "actual_internal_energy_fraction": 1.0 - actual_fraction,
            "actual_rigid_displacement": rigid_length,
            "actual_internal_displacement": internal_length,
            "normalised_actual_rigid_displacement": rigid_length / scale,
            "normalised_actual_internal_displacement": internal_length / scale,
        })

    applicable = bool(ranks and max(ranks) > 0)
    summary.update({
        "rigid_motion_diagnostics_applicable": int(applicable),
        "minimum_support_compatible_rigid_rank": (
            int(min(ranks)) if ranks else 0
        ),
        "maximum_support_compatible_rigid_rank": (
            int(max(ranks)) if ranks else 0
        ),
        "mean_commanded_rigid_energy_fraction": float(
            np.mean(commanded_fractions) if commanded_fractions else 0.0
        ),
        "mean_actual_rigid_energy_fraction": float(
            np.mean(actual_fractions) if actual_fractions else 0.0
        ),
        "valid_path_rigid_energy_fraction": float(
            valid_rigid_energy / valid_actual_energy
            if valid_actual_energy > 1.0e-24 else 0.0
        ),
        "total_rigid_path_length_normalised": float(
            valid_rigid_length / scale
        ),
        "total_internal_path_length_normalised": float(
            valid_internal_length / scale
        ),
    })
    return summary, nodes, actions


def make_environment(args, node, member, fix):
    """Attach logging flags to the unmodified ori_codex4 environment."""

    node = np.asarray(node, dtype=float)
    if node.ndim != 2 or node.shape[1] != 3:
        raise ValueError("ori_codex4_3d can only construct a 3-D environment.")
    environment = _ORIGINAL_ORI4_MAKE_ENVIRONMENT(args, node, member, fix)
    environment.rigid_motion_diagnostics_enabled = bool(
        getattr(args, "rigid_motion_diagnostics", False)
    )
    return environment


def run_episode(env, agent, deterministic=False, training=False):
    summary, nodes, actions = _ORIGINAL_ORI4_RUN_EPISODE(
        env, agent, deterministic=deterministic, training=training
    )
    return add_rigid_motion_diagnostics(env, summary, nodes, actions)


def aggregate_stochastic_evaluations(evaluations, episode=None):
    result = _ORIGINAL_ORI4_AGGREGATE_EVALUATIONS(
        evaluations, episode=episode
    )
    if result is None:
        return None
    result["rigid_motion_diagnostics_enabled_rate"] = float(np.mean([
        row.get("rigid_motion_diagnostics_enabled", 0) for row in evaluations
    ]))
    result["rigid_motion_diagnostics_applicable_rate"] = float(np.mean([
        row.get("rigid_motion_diagnostics_applicable", 0) for row in evaluations
    ]))
    aggregate_names = {
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
    }
    for field, aggregate_name in aggregate_names.items():
        values = [
            float(row[field]) for row in evaluations
            if row.get(field) is not None
        ]
        result[aggregate_name] = float(np.mean(values)) if values else None
    return result


def _build_metadata(args, node, member, fix, files):
    maximum_length = _maximum_initial_member_length(node, member)
    pairs = collision_3d.nonadjacent_member_pairs(member)
    collision_settings = collision_3d.ContinuousBarCollisionChecker(
        member,
        pair_indices=pairs,
        distance_tolerance=float(args.crossing_tolerance),
    )
    free = mechanics.index_free(len(node), fix)
    initial_modes = mechanics.mode(
        free, mechanics.dedx(node, member, free), len(node)
    )
    rigid_rank = int(support_compatible_rigid_basis(node, fix).shape[1])
    initial_mode_count = int(initial_modes.shape[1])
    return {
        "adapter": "ori_codex4_3d",
        "adapter_version": "1.0.0",
        "model_name": str(getattr(args, "selected_model", Path(args.data_dir).name)),
        "data_directory": str(Path(args.data_dir).resolve()),
        "spatial_dimension": 3,
        "flatness_subspace_dimension": int(args.flatness_subspace_dimension),
        "node_count": int(len(node)),
        "member_count": int(len(member)),
        "fixed_scalar_dof_count": int(len(fix)),
        "support_conditions_modified": False,
        "initial_maximum_member_length": maximum_length,
        "max_step_width": float(args.max_step_width),
        "step_width_rule": str(getattr(args, "step_width_rule", "user_supplied")),
        "target_spread": float(args.target_spread),
        "spread_definition": (
            "diameter after projection onto the best-fit plane, divided by "
            "the maximum initial member length"
        ),
        "collision_backend": collision_3d.COLLISION_BACKEND_NAME,
        "collision_backend_version": collision_3d.COLLISION_BACKEND_VERSION,
        "collision_semantics": collision_3d.COLLISION_SEMANTICS,
        "collision_nonadjacent_pair_count": int(len(pairs)),
        "collision_parameters": {
            "distance_tolerance": collision_settings.distance_tolerance,
            "time_tolerance": collision_settings.time_tolerance,
            "parameter_tolerance": collision_settings.parameter_tolerance,
            "parallel_sine_tolerance": (
                collision_settings.parallel_sine_tolerance
            ),
            "coefficient_tolerance": (
                collision_settings.coefficient_tolerance
            ),
        },
        "rigid_motion_diagnostics": bool(args.rigid_motion_diagnostics),
        "rigid_motion_diagnostic_effect": "logging_only",
        "initial_support_compatible_rigid_rank": rigid_rank,
        "initial_mechanism_mode_count": initial_mode_count,
        "uniform_random_initial_expected_rigid_energy_fraction": (
            float(rigid_rank / initial_mode_count)
            if initial_mode_count else 0.0
        ),
        "model_file_sha256": {
            name: _sha256(path) for name, path in files.items()
        },
    }


@contextmanager
def patched_three_dimensional_runtime(args, member):
    """Install all adapter hooks and restore them even after an exception."""

    pairs = collision_3d.nonadjacent_member_pairs(member)
    collision_callback = collision_3d.make_legacy_signature_checker(
        member,
        pair_indices=pairs,
        distance_tolerance=float(args.crossing_tolerance),
    )
    original = (
        mechanics.has_proper_self_crossing,
        ori4.make_environment,
        ori4.run_episode,
        ori4.aggregate_stochastic_evaluations,
    )
    mechanics.has_proper_self_crossing = collision_callback
    ori4.make_environment = make_environment
    ori4.run_episode = run_episode
    ori4.aggregate_stochastic_evaluations = aggregate_stochastic_evaluations
    try:
        yield collision_callback
    finally:
        (
            mechanics.has_proper_self_crossing,
            ori4.make_environment,
            ori4.run_episode,
            ori4.aggregate_stochastic_evaluations,
        ) = original


def train(args):
    """Train/evaluate ori_codex4 with 3-D collision and audit metadata."""

    if int(args.flatness_subspace_dimension) != 2:
        raise ValueError(
            "The 3-D adapter requires --flatness-subspace-dimension 2."
        )
    node, member, fix, files = load_problem_data(args.data_dir)
    args.data_dir = str(Path(args.data_dir).resolve())
    if not hasattr(args, "selected_model"):
        args.selected_model = Path(args.data_dir).name
    if not hasattr(args, "rigid_motion_diagnostics"):
        args.rigid_motion_diagnostics = args.selected_model == "4unit_torus"
    metadata = _build_metadata(args, node, member, fix, files)
    args.three_dimensional_adapter = metadata

    with patched_three_dimensional_runtime(args, member):
        output_directory, summary = ori4.train(args)

    output_directory = Path(output_directory)
    summary["three_dimensional_adapter"] = metadata
    with open(output_directory / "summary.json", "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as file:
        config = json.load(file)
    config["three_dimensional_adapter"] = metadata
    config["effective_output_directory"] = str(output_directory.resolve())
    with open(config_path, "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)
    return output_directory, summary


def parse_args(argv=None):
    """Parse adapter options first, then the complete ori_codex4 CLI."""

    arguments = sys.argv[1:] if argv is None else list(argv)
    adapter_parser = argparse.ArgumentParser(add_help=False)
    adapter_parser.add_argument(
        "--model", choices=tuple(_MODEL_DEFAULT_TARGETS), default=None
    )
    diagnostic_group = adapter_parser.add_mutually_exclusive_group()
    diagnostic_group.add_argument(
        "--rigid-diagnostics", dest="rigid_diagnostics", action="store_true"
    )
    diagnostic_group.add_argument(
        "--no-rigid-diagnostics", dest="rigid_diagnostics", action="store_false"
    )
    adapter_parser.set_defaults(rigid_diagnostics=None)
    adapter_options, remaining = adapter_parser.parse_known_args(arguments)

    model_for_default_path = adapter_options.model or "4unit_torus"
    injected = ["--flatness-subspace-dimension", "2"]
    if not _option_was_supplied(remaining, "--data-dir"):
        injected.extend([
            "--data-dir",
            str(_SCRIPT_DIRECTORY / "model" / model_for_default_path),
        ])
    args = ori4.parse_args([*injected, *remaining])
    args.data_dir = str(Path(args.data_dir).resolve())
    inferred_model = Path(args.data_dir).name
    args.selected_model = adapter_options.model or inferred_model

    node, member, _, _ = load_problem_data(args.data_dir)
    maximum_length = _maximum_initial_member_length(node, member)
    if not _option_was_supplied(remaining, "--max-step-width"):
        args.max_step_width = 0.1 * maximum_length
        args.step_width_rule = "0.1_times_initial_maximum_member_length"
    else:
        args.step_width_rule = "user_supplied"

    if not _option_was_supplied(remaining, "--target-spread"):
        args.target_spread = _MODEL_DEFAULT_TARGETS.get(inferred_model, 2.0)

    if adapter_options.rigid_diagnostics is None:
        args.rigid_motion_diagnostics = inferred_model == "4unit_torus"
    else:
        args.rigid_motion_diagnostics = bool(adapter_options.rigid_diagnostics)

    if not _option_was_supplied(remaining, "--output-dir"):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        unique_suffix = uuid.uuid4().hex[:8]
        run_name = (
            f"{_safe_slug(inferred_model)}_spread{_number_slug(args.target_spread)}_"
            f"h{_number_slug(args.max_step_width)}_seed{int(args.seed)}_"
            f"{timestamp}_{unique_suffix}"
        )
        args.output_dir = str(Path("ori_codex4_3d_results") / run_name)
    return args


if __name__ == "__main__":
    train(parse_args())
