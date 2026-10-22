"""Build one interactive potential-coloured forest from five-bar search runs.

The historical ``search_tree.html`` files contain the complete StateNode point
cloud in three signed joint angles, but not the cached potential.  For a planar
five-bar chain with fixed endpoints, the three internal angles and the fixed
link lengths determine the Cartesian state.  This script reconstructs that
state and evaluates the same quantity used by ``mechanics.py``::

    U(q) = sum_i abs(z_i - reference_point)

Newer runs can instead provide ``state_tree.npz``.  The ``potential_tree_v1``
schema written by ``main9.py`` is supported, together with a few plural aliases
for forward compatibility.

By default every ``*_five_bar_linkage/result`` directory beside this script is
scanned.  If a run directory has both input formats, ``state_tree.npz`` wins.
The output is a single browser-only Plotly document.  It initially shows all
recovered states; selecting one run adds that run's deterministic and
perturbation edges.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np


PLOTLY_CDN = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
HTML_SOURCE_NAME = "search_tree.html"
NPZ_SOURCE_NAME = "state_tree.npz"


@dataclass(slots=True)
class FiveBarModel:
    """Fixed geometry needed to invert the three internal joint angles."""

    model_directory: Path
    chain: np.ndarray
    link_lengths: np.ndarray
    initial_nodes: np.ndarray
    start_xz: complex
    endpoint_vector_xz: complex


@dataclass(slots=True)
class RunTree:
    """A compact, format-independent search tree."""

    label: str
    source_path: Path
    source_format: str
    state_id: np.ndarray
    parent_id: np.ndarray
    depth: np.ndarray
    joint_angle_deg: np.ndarray
    potential: np.ndarray
    cell_label: list[str]
    terminal_reason: list[str]
    incoming_step_type: list[str]
    outgoing_perturbations: np.ndarray
    reference_point: float
    closure_error_max: float
    potential_validation_error_max: float | None

    @property
    def number_of_states(self) -> int:
        return int(len(self.state_id))


def _ordered_five_bar_chain(member: np.ndarray, number_of_nodes: int) -> np.ndarray:
    """Return the node order along a single four-link, five-node chain."""

    member = np.asarray(member, dtype=int)
    if number_of_nodes != 5 or member.shape != (4, 2):
        raise ValueError(
            "expected a five-node, four-member chain; "
            f"got {number_of_nodes} nodes and member shape {member.shape}"
        )

    neighbours: list[list[int]] = [[] for _ in range(number_of_nodes)]
    for node_i, node_j in member:
        neighbours[int(node_i)].append(int(node_j))
        neighbours[int(node_j)].append(int(node_i))

    end_nodes = [index for index, adjacent in enumerate(neighbours) if len(adjacent) == 1]
    if len(end_nodes) != 2 or any(len(adjacent) > 2 for adjacent in neighbours):
        raise ValueError("member.dat is not a single open five-bar chain")

    chain = [min(end_nodes)]
    previous: int | None = None
    current = chain[0]
    while len(chain) < number_of_nodes:
        candidates = [node for node in neighbours[current] if node != previous]
        if len(candidates) != 1:
            raise ValueError("could not order the five-bar chain")
        next_node = candidates[0]
        chain.append(next_node)
        previous, current = current, next_node

    if chain[-1] != max(end_nodes):
        raise ValueError("unexpected five-bar endpoint ordering")
    return np.asarray(chain, dtype=int)


def _load_model(model_directory: Path) -> FiveBarModel:
    model_directory = model_directory.resolve()
    node_path = model_directory / "node_initial.dat"
    member_path = model_directory / "member.dat"
    initial_nodes = np.loadtxt(node_path, delimiter=",", dtype=float)
    member = np.loadtxt(member_path, delimiter=",", dtype=int)
    chain = _ordered_five_bar_chain(member, len(initial_nodes))
    ordered_nodes = initial_nodes[chain]
    link_lengths = np.linalg.norm(np.diff(ordered_nodes, axis=0), axis=1)

    start = complex(float(ordered_nodes[0, 0]), float(ordered_nodes[0, 2]))
    finish = complex(float(ordered_nodes[-1, 0]), float(ordered_nodes[-1, 2]))
    endpoint_vector = finish - start
    if abs(endpoint_vector) <= np.finfo(float).eps:
        raise ValueError("the two fixed endpoints coincide")
    if np.ptp(initial_nodes[:, 1]) > 1.0e-10:
        raise ValueError("the fixed model is not in the x-z plane")

    return FiveBarModel(
        model_directory=model_directory,
        chain=chain,
        link_lengths=link_lengths,
        initial_nodes=initial_nodes,
        start_xz=start,
        endpoint_vector_xz=endpoint_vector,
    )


def _find_model_directory(source_path: Path) -> Path:
    """Find the nearest linkage ``model`` directory above an input file."""

    for ancestor in (source_path.parent, *source_path.parents):
        candidate = ancestor / "model"
        if (candidate / "node_initial.dat").is_file() and (
            candidate / "member.dat"
        ).is_file():
            return candidate
    raise FileNotFoundError(f"no five-bar model found above {source_path}")


def _reconstruct_nodes_from_angles(
    joint_angle_deg: np.ndarray,
    model: FiveBarModel,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct Cartesian nodes and closure error from signed joint angles.

    If ``phi_j`` is the orientation of link ``j``, the angle convention used in
    ``result._five_bar_joint_angles`` gives

        phi_j = phi_(j-1) + pi + theta_j.

    The remaining global rotation is fixed by the endpoint vector.  The input
    HTML keeps double-precision angles, so the nominal-link closure discrepancy
    is normally at round-off level.
    """

    angles = np.asarray(joint_angle_deg, dtype=float)
    if angles.ndim != 2 or angles.shape[1] != 3:
        raise ValueError(f"joint angles must have shape (N, 3), got {angles.shape}")
    if not np.all(np.isfinite(angles)):
        raise ValueError("joint angles contain non-finite values")

    theta = np.radians(angles)
    relative_orientation = np.empty((len(theta), 4), dtype=float)
    relative_orientation[:, 0] = 0.0
    relative_orientation[:, 1] = np.pi + theta[:, 0]
    relative_orientation[:, 2] = 2.0 * np.pi + theta[:, 0] + theta[:, 1]
    relative_orientation[:, 3] = 3.0 * np.pi + np.sum(theta, axis=1)

    relative_links = (
        model.link_lengths[None, :]
        * np.exp(1j * relative_orientation)
    )
    relative_endpoint = np.sum(relative_links, axis=1)
    if np.any(np.abs(relative_endpoint) <= np.finfo(float).eps):
        raise ValueError("joint angles imply a zero endpoint vector")

    fixed_endpoint = model.endpoint_vector_xz
    rotation = np.exp(
        1j * (np.angle(fixed_endpoint) - np.angle(relative_endpoint))
    )
    oriented_links = relative_links * rotation[:, None]

    ordered_xz = np.empty((len(theta), 5), dtype=complex)
    ordered_xz[:, 0] = model.start_xz
    ordered_xz[:, 1:] = model.start_xz + np.cumsum(oriented_links, axis=1)

    nodes = np.zeros((len(theta), 5, 3), dtype=float)
    nodes[:, model.chain, 0] = ordered_xz.real
    nodes[:, model.chain, 2] = ordered_xz.imag
    nodes[:, :, 1] = model.initial_nodes[None, :, 1]

    closure_error = np.abs(ordered_xz[:, -1] - (model.start_xz + fixed_endpoint))
    return nodes, closure_error


def _potential_from_nodes(nodes: np.ndarray, reference_point: float) -> np.ndarray:
    """Evaluate the exact project potential used by ``mechanics.py``."""

    nodes = np.asarray(nodes, dtype=float)
    if nodes.ndim != 3 or nodes.shape[2] != 3:
        raise ValueError(f"nodes must have shape (N, num_nodes, 3), got {nodes.shape}")
    potential = np.sum(np.abs(nodes[:, :, 2] - float(reference_point)), axis=1)
    if not np.all(np.isfinite(potential)):
        raise ValueError("reconstructed potential contains non-finite values")
    return potential


def _joint_angles_from_nodes(
    nodes: np.ndarray,
    parent_id: np.ndarray,
    state_id: np.ndarray,
    model: FiveBarModel,
) -> np.ndarray:
    """Recover the same branch-continuous angles used by the tree HTML."""

    nodes = np.asarray(nodes, dtype=float)
    raw = np.empty((len(nodes), 3), dtype=float)
    for angle_index, center_node in enumerate(model.chain[1:-1]):
        previous_node = model.chain[angle_index]
        next_node = model.chain[angle_index + 2]
        vector_previous = nodes[:, previous_node][:, [0, 2]] - nodes[:, center_node][
            :, [0, 2]
        ]
        vector_next = nodes[:, next_node][:, [0, 2]] - nodes[:, center_node][
            :, [0, 2]
        ]
        cross = (
            vector_previous[:, 0] * vector_next[:, 1]
            - vector_previous[:, 1] * vector_next[:, 0]
        )
        dot = np.sum(vector_previous * vector_next, axis=1)
        raw[:, angle_index] = np.arctan2(cross, dot)

    id_to_index = {int(value): index for index, value in enumerate(state_id)}
    continuous = np.empty_like(raw)
    completed: set[int] = set()

    def resolve(index: int, active: set[int]) -> None:
        if index in completed:
            return
        if index in active:
            raise ValueError("cycle detected in parent_id")
        active.add(index)
        parent = int(parent_id[index])
        if parent < 0:
            continuous[index] = raw[index]
        else:
            if parent not in id_to_index:
                raise ValueError(f"parent state {parent} is missing")
            parent_index = id_to_index[parent]
            resolve(parent_index, active)
            difference = (raw[index] - continuous[parent_index] + np.pi) % (
                2.0 * np.pi
            ) - np.pi
            continuous[index] = continuous[parent_index] + difference
        active.remove(index)
        completed.add(index)

    for state_index in range(len(nodes)):
        resolve(state_index, set())
    return np.degrees(continuous)


def _calculate_depth(state_id: np.ndarray, parent_id: np.ndarray) -> np.ndarray:
    id_to_index = {int(value): index for index, value in enumerate(state_id)}
    depth = np.full(len(state_id), -1, dtype=np.int32)

    def resolve(index: int, active: set[int]) -> int:
        if depth[index] >= 0:
            return int(depth[index])
        if index in active:
            raise ValueError("cycle detected in parent_id")
        active.add(index)
        parent = int(parent_id[index])
        if parent < 0:
            value = 0
        else:
            if parent not in id_to_index:
                raise ValueError(f"parent state {parent} is missing")
            value = resolve(id_to_index[parent], active) + 1
        active.remove(index)
        depth[index] = value
        return value

    for state_index in range(len(state_id)):
        resolve(state_index, set())
    return depth


def _normalise_step_type(value: Any, is_root: bool = False) -> str:
    if is_root:
        return "root"
    text = str(value).strip().lower()
    if text in {"1", "true", "random", "perturbation", "perturbed"}:
        return "perturbation"
    if "perturb" in text or "random" in text:
        return "perturbation"
    return "deterministic"


def _string_array(values: Iterable[Any], length: int, default: str) -> list[str]:
    result = [default if value is None else str(value) for value in values]
    if len(result) != length:
        raise ValueError(f"metadata length {len(result)} does not match {length} states")
    return result


def _reference_point_for_html(path: Path, fallback: float) -> float:
    config_json = path.parent / "config.json"
    if config_json.is_file():
        try:
            config = json.loads(config_json.read_text(encoding="utf-8"))
            if "reference_point" in config:
                return float(config["reference_point"])
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            pass

    for filename in ("run_config.txt", "config.txt"):
        config_text = path.parent / filename
        if not config_text.is_file():
            continue
        try:
            text = config_text.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        match = re.search(
            r"(?:^|\n)\s*reference_point\s*[:=]\s*"
            r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
            text,
        )
        if match:
            return float(match.group(1))
    return float(fallback)


def _decode_traces_from_html(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    marker = "const traces = "
    marker_index = text.find(marker)
    if marker_index < 0:
        raise ValueError("could not find 'const traces =' in HTML")
    traces, _ = json.JSONDecoder().raw_decode(text[marker_index + len(marker) :])
    if not isinstance(traces, list):
        raise ValueError("HTML traces value is not a JSON array")
    return traces


def _load_html_tree(
    path: Path,
    label: str,
    model: FiveBarModel,
    reference_point_fallback: float,
) -> RunTree:
    traces = _decode_traces_from_html(path)
    state_trace = next(
        (
            trace
            for trace in traces
            if trace.get("name") == "StateNode"
            and trace.get("type") == "scatter3d"
        ),
        None,
    )
    if state_trace is None:
        raise ValueError("no scatter3d trace named StateNode")

    joint_angle_deg = np.column_stack(
        [state_trace.get("x", []), state_trace.get("y", []), state_trace.get("z", [])]
    ).astype(float)
    number_of_states = len(joint_angle_deg)
    if number_of_states == 0:
        raise ValueError("StateNode trace is empty")

    custom_data = state_trace.get("customdata")
    if custom_data is None or len(custom_data) != number_of_states:
        custom_data = [[] for _ in range(number_of_states)]

    def field(row: Any, index: int, default: Any) -> Any:
        return row[index] if isinstance(row, (list, tuple)) and len(row) > index else default

    state_id = np.asarray(
        [int(field(row, 0, index)) for index, row in enumerate(custom_data)],
        dtype=np.int64,
    )
    parent_id = np.asarray(
        [int(field(row, 1, -1)) for row in custom_data], dtype=np.int64
    )
    depth_values = [field(row, 2, None) for row in custom_data]
    if all(value is not None for value in depth_values):
        depth = np.asarray(depth_values, dtype=np.int32)
    else:
        depth = _calculate_depth(state_id, parent_id)
    cell_label = [str(field(row, 3, "unknown")) for row in custom_data]
    terminal_reason = [str(field(row, 4, "nonterminal")) for row in custom_data]
    incoming_step_type = [
        _normalise_step_type(field(row, 5, 0), int(parent_id[index]) < 0)
        for index, row in enumerate(custom_data)
    ]
    outgoing_perturbations = np.asarray(
        [int(field(row, 6, 0)) for row in custom_data], dtype=np.int32
    )

    reference_point = _reference_point_for_html(path, reference_point_fallback)
    reconstructed_nodes, closure_error = _reconstruct_nodes_from_angles(
        joint_angle_deg, model
    )
    potential = _potential_from_nodes(reconstructed_nodes, reference_point)
    return RunTree(
        label=label,
        source_path=path,
        source_format="search_tree.html",
        state_id=state_id,
        parent_id=parent_id,
        depth=depth,
        joint_angle_deg=joint_angle_deg,
        potential=potential,
        cell_label=cell_label,
        terminal_reason=terminal_reason,
        incoming_step_type=incoming_step_type,
        outgoing_perturbations=outgoing_perturbations,
        reference_point=reference_point,
        closure_error_max=float(np.max(closure_error)),
        potential_validation_error_max=None,
    )


def _first_npz(data: Any, names: Iterable[str]) -> np.ndarray | None:
    for name in names:
        if name in data.files:
            return np.asarray(data[name])
    return None


def _npz_scalar(data: Any, names: Iterable[str], default: Any) -> Any:
    value = _first_npz(data, names)
    if value is None:
        return default
    if value.size != 1:
        raise ValueError(f"expected scalar key from {tuple(names)}, got shape {value.shape}")
    return value.reshape(-1)[0].item()


def _load_npz_tree(
    path: Path,
    label: str,
    model: FiveBarModel,
    reference_point_fallback: float,
) -> RunTree:
    with np.load(path, allow_pickle=True) as data:
        reference_point = float(
            _npz_scalar(data, ("reference_point", "z_reference"), reference_point_fallback)
        )
        node_array = _first_npz(data, ("node", "nodes", "state_nodes"))
        angle_array = _first_npz(
            data,
            (
                "joint_angle_deg",
                "joint_angles_deg",
                "angles_deg",
                "coordinates",
            ),
        )
        cached_potential = _first_npz(data, ("potential", "potentials"))

        count_candidates = [
            len(array)
            for array in (node_array, angle_array, cached_potential)
            if array is not None and array.ndim >= 1
        ]
        if not count_candidates:
            raise ValueError("NPZ has none of node, joint_angle_deg, or potential")
        number_of_states = int(count_candidates[0])
        if any(count != number_of_states for count in count_candidates):
            raise ValueError("NPZ state arrays have inconsistent lengths")

        state_id_array = _first_npz(data, ("state_id", "state_ids"))
        state_id = (
            np.arange(number_of_states, dtype=np.int64)
            if state_id_array is None
            else np.asarray(state_id_array, dtype=np.int64).reshape(-1)
        )
        parent_array = _first_npz(data, ("parent_id", "parent_ids", "parents"))
        parent_id = (
            np.full(number_of_states, -1, dtype=np.int64)
            if parent_array is None
            else np.asarray(parent_array, dtype=np.int64).reshape(-1)
        )
        if len(state_id) != number_of_states or len(parent_id) != number_of_states:
            raise ValueError("NPZ state_id or parent_id length mismatch")

        depth_array = _first_npz(data, ("depth", "depths"))
        depth = (
            _calculate_depth(state_id, parent_id)
            if depth_array is None
            else np.asarray(depth_array, dtype=np.int32).reshape(-1)
        )

        nodes: np.ndarray | None
        if node_array is not None:
            nodes = np.asarray(node_array, dtype=float)
            if nodes.ndim != 3 or nodes.shape[0] != number_of_states:
                raise ValueError(f"unexpected node shape {nodes.shape}")
        else:
            nodes = None

        if angle_array is not None:
            joint_angle_deg = np.asarray(angle_array, dtype=float)
            if joint_angle_deg.shape != (number_of_states, 3):
                raise ValueError(f"unexpected joint_angle_deg shape {joint_angle_deg.shape}")
        elif nodes is not None:
            joint_angle_deg = _joint_angles_from_nodes(nodes, parent_id, state_id, model)
        else:
            raise ValueError("NPZ needs node coordinates or joint angles for plotting")

        if nodes is None:
            nodes, closure_error = _reconstruct_nodes_from_angles(joint_angle_deg, model)
        else:
            closure_error = np.zeros(number_of_states, dtype=float)
            endpoint_start = nodes[:, model.chain[0]][:, [0, 2]]
            endpoint_finish = nodes[:, model.chain[-1]][:, [0, 2]]
            fixed_vector = np.asarray(
                [model.endpoint_vector_xz.real, model.endpoint_vector_xz.imag]
            )
            closure_error = np.linalg.norm(
                endpoint_finish - endpoint_start - fixed_vector[None, :], axis=1
            )

        recomputed_potential = _potential_from_nodes(nodes, reference_point)
        validation_error: float | None = None
        if cached_potential is not None:
            cached = np.asarray(cached_potential, dtype=float).reshape(-1)
            if len(cached) != number_of_states or not np.all(np.isfinite(cached)):
                raise ValueError("cached potential is invalid")
            validation_error = float(np.max(np.abs(cached - recomputed_potential)))
            cache_tolerance_array = _first_npz(data, ("potential_cache_tolerance",))
            if cache_tolerance_array is not None:
                if cache_tolerance_array.size != 1:
                    raise ValueError("potential_cache_tolerance must be scalar")
                cache_tolerance = float(cache_tolerance_array.reshape(-1)[0])
                numerical_slack = 16.0 * np.finfo(float).eps * max(
                    1.0, float(np.max(np.abs(cached)))
                )
                if validation_error > cache_tolerance + numerical_slack:
                    raise ValueError(
                        "cached potential differs from node coordinates by "
                        f"{validation_error:.3e}, above tolerance "
                        f"{cache_tolerance:.3e}"
                    )
            # potential_tree_v1 preserves the StateNode cache as float64.
            # Keep it after validating it against the saved coordinates.
            potential = cached
        else:
            potential = recomputed_potential

        cell_array = _first_npz(data, ("cell_label", "cell_labels", "cell"))
        cell_label = (
            ["unknown"] * number_of_states
            if cell_array is None
            else _string_array(np.asarray(cell_array).reshape(-1), number_of_states, "unknown")
        )
        terminal_array = _first_npz(
            data, ("terminal_reason", "terminal_reasons", "terminal")
        )
        terminal_reason = (
            ["nonterminal"] * number_of_states
            if terminal_array is None
            else _string_array(
                np.asarray(terminal_array).reshape(-1), number_of_states, "nonterminal"
            )
        )
        step_array = _first_npz(
            data, ("incoming_step_type", "incoming_step_types", "step_type")
        )
        if step_array is None:
            incoming_step_type = [
                "root" if int(parent_id[index]) < 0 else "deterministic"
                for index in range(number_of_states)
            ]
        else:
            raw_step = np.asarray(step_array).reshape(-1)
            if len(raw_step) != number_of_states:
                raise ValueError("incoming_step_type length mismatch")
            incoming_step_type = [
                _normalise_step_type(raw_step[index], int(parent_id[index]) < 0)
                for index in range(number_of_states)
            ]

    outgoing_perturbations = np.zeros(number_of_states, dtype=np.int32)
    id_to_index = {int(value): index for index, value in enumerate(state_id)}
    for parent, step_type in zip(parent_id, incoming_step_type):
        if int(parent) >= 0 and step_type == "perturbation" and int(parent) in id_to_index:
            outgoing_perturbations[id_to_index[int(parent)]] += 1

    return RunTree(
        label=label,
        source_path=path,
        source_format="state_tree.npz",
        state_id=state_id,
        parent_id=parent_id,
        depth=depth,
        joint_angle_deg=joint_angle_deg,
        potential=np.asarray(potential, dtype=float),
        cell_label=cell_label,
        terminal_reason=terminal_reason,
        incoming_step_type=incoming_step_type,
        outgoing_perturbations=outgoing_perturbations,
        reference_point=reference_point,
        closure_error_max=float(np.max(closure_error)),
        potential_validation_error_max=validation_error,
    )


def _default_input_roots(script_directory: Path) -> list[Path]:
    workspace = script_directory.parent
    roots = [
        directory / "result"
        for directory in sorted(workspace.glob("*_five_bar_linkage"))
        if (directory / "result").is_dir()
    ]
    if not roots and (script_directory / "result").is_dir():
        roots = [script_directory / "result"]
    return roots


def _discover_sources(input_roots: Iterable[Path]) -> list[Path]:
    """Find one preferred source per run directory."""

    by_directory: dict[Path, Path] = {}
    for raw_root in input_roots:
        root = raw_root.resolve()
        if root.is_file() and root.name in {HTML_SOURCE_NAME, NPZ_SOURCE_NAME}:
            candidates = [root]
        elif root.is_dir():
            candidates = [
                *root.rglob(HTML_SOURCE_NAME),
                *root.rglob(NPZ_SOURCE_NAME),
            ]
        else:
            continue
        for candidate in candidates:
            current = by_directory.get(candidate.parent)
            if current is None or candidate.name == NPZ_SOURCE_NAME:
                by_directory[candidate.parent] = candidate
    return sorted(by_directory.values(), key=lambda value: str(value).lower())


def _common_label_root(input_roots: list[Path]) -> Path:
    resolved = [root.resolve() if root.is_dir() else root.resolve().parent for root in input_roots]
    if not resolved:
        return Path.cwd().resolve()
    try:
        # os.path.commonpath handles Windows drive semantics correctly.
        return Path(os.path.commonpath([str(root) for root in resolved]))
    except (ValueError, OSError):
        return Path.cwd().resolve()


def _run_label(path: Path, label_root: Path) -> str:
    try:
        relative_parent = path.parent.resolve().relative_to(label_root.resolve())
        return relative_parent.as_posix()
    except ValueError:
        return path.parent.as_posix()


def _validate_run(run: RunTree) -> None:
    count = run.number_of_states
    arrays = (
        run.state_id,
        run.parent_id,
        run.depth,
        run.joint_angle_deg,
        run.potential,
        run.outgoing_perturbations,
    )
    if any(len(array) != count for array in arrays):
        raise ValueError("state array length mismatch")
    if len(set(map(int, run.state_id))) != count:
        raise ValueError("state_id values are not unique")
    if not np.all(np.isfinite(run.joint_angle_deg)) or not np.all(
        np.isfinite(run.potential)
    ):
        raise ValueError("non-finite angle or potential")
    if np.any(run.potential < 0.0):
        raise ValueError("potential must be non-negative")
    if len(run.cell_label) != count or len(run.terminal_reason) != count:
        raise ValueError("string metadata length mismatch")
    if len(run.incoming_step_type) != count:
        raise ValueError("incoming_step_type length mismatch")


def _json_float_list(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values, dtype=float)]


def _run_to_json(run: RunTree) -> dict[str, Any]:
    validation_error = run.potential_validation_error_max
    return {
        "label": run.label,
        "source": run.source_path.as_posix(),
        "format": run.source_format,
        "reference": float(run.reference_point),
        "closureMax": float(run.closure_error_max),
        "potentialValidationMax": (
            None if validation_error is None else float(validation_error)
        ),
        "id": [int(value) for value in run.state_id],
        "parent": [int(value) for value in run.parent_id],
        "depth": [int(value) for value in run.depth],
        "x": _json_float_list(run.joint_angle_deg[:, 0]),
        "y": _json_float_list(run.joint_angle_deg[:, 1]),
        "z": _json_float_list(run.joint_angle_deg[:, 2]),
        "u": _json_float_list(run.potential),
        "cell": run.cell_label,
        "terminal": run.terminal_reason,
        "edge": run.incoming_step_type,
        "outgoingPerturbations": [int(value) for value in run.outgoing_perturbations],
    }


def _html_document(runs: list[RunTree]) -> str:
    payload = [_run_to_json(run) for run in runs]
    all_potential = np.concatenate([run.potential for run in runs])
    potential_min = float(np.min(all_potential))
    potential_max = float(np.max(all_potential))
    if potential_min == potential_max:
        linear_min = potential_min - 0.5
        linear_max = potential_max + 0.5
    else:
        linear_min = potential_min
        linear_max = potential_max

    positive = all_potential[all_potential > 0.0]
    if len(positive):
        log_epsilon = max(1.0e-15, float(np.min(positive)) * 1.0e-6)
    else:
        log_epsilon = 1.0e-15
    log_values = np.log10(all_potential + log_epsilon)
    log_min = float(np.min(log_values))
    log_max = float(np.max(log_values))
    if log_min == log_max:
        log_min -= 0.5
        log_max += 0.5

    summary = {
        "runCount": len(runs),
        "stateCount": int(sum(run.number_of_states for run in runs)),
        "potentialMin": potential_min,
        "potentialMax": potential_max,
        "linearMin": linear_min,
        "linearMax": linear_max,
        "logMin": log_min,
        "logMax": log_max,
        "logEpsilon": log_epsilon,
        "generatedUtc": datetime.now(timezone.utc).isoformat(),
    }
    payload_json = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).replace("</", "<\\/")
    summary_json = json.dumps(summary, separators=(",", ":"), allow_nan=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Five-bar gravity-potential forest</title>
<script src="{PLOTLY_CDN}"></script>
<style>
:root {{ color-scheme: light; font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }}
html, body {{ width: 100%; height: 100%; margin: 0; }}
body {{ display: flex; flex-direction: column; overflow: hidden; background: #ffffff; color: #171717; }}
#controls {{ display: flex; flex-wrap: wrap; align-items: end; gap: 10px 18px; padding: 10px 14px 8px; border-bottom: 1px solid color-mix(in srgb, CanvasText 22%, transparent); }}
#controls label {{ display: grid; gap: 3px; font-size: 12px; }}
#controls select {{ min-width: 15rem; max-width: min(52rem, 78vw); padding: 4px 6px; font: inherit; }}
#controls #scale-select {{ min-width: 10rem; }}
#status {{ flex: 1 1 22rem; min-width: 15rem; padding-bottom: 4px; font-size: 12px; color: color-mix(in srgb, CanvasText 72%, transparent); }}
#forest {{ flex: 1 1 auto; min-height: 0; width: 100%; }}
@media (max-width: 560px) {{
  #controls {{ gap: 8px; }}
  #controls label, #controls select, #status {{ width: 100%; min-width: 0; max-width: none; }}
}}
</style>
</head>
<body>
<div id="controls">
  <label>Run
    <select id="run-select" aria-label="Run to display"></select>
  </label>
  <label>Potential color scale
    <select id="scale-select" aria-label="Potential color scale">
      <option value="linear">Linear U</option>
      <option value="log">Log10 U</option>
    </select>
  </label>
  <div id="status" role="status" aria-live="polite"></div>
</div>
<div id="forest" role="img" aria-label="Three-dimensional five-bar search forest coloured by gravity potential"></div>
<script>
"use strict";
const runs = {payload_json};
const summary = {summary_json};
const plot = document.getElementById("forest");
const runSelect = document.getElementById("run-select");
const scaleSelect = document.getElementById("scale-select");
const status = document.getElementById("status");

const all = {{x: [], y: [], z: [], u: [], custom: []}};
for (const run of runs) {{
  for (let i = 0; i < run.id.length; i += 1) {{
    all.x.push(run.x[i]); all.y.push(run.y[i]); all.z.push(run.z[i]); all.u.push(run.u[i]);
    all.custom.push([run.label, run.id[i], run.parent[i], run.depth[i], run.cell[i], run.terminal[i], run.edge[i], run.u[i], run.outgoingPerturbations[i]]);
  }}
}}

const allOption = document.createElement("option");
allOption.value = "all";
allOption.textContent = `All runs (${{summary.runCount}} runs, ${{summary.stateCount.toLocaleString()}} states)`;
runSelect.appendChild(allOption);
runs.forEach((run, index) => {{
  const option = document.createElement("option");
  option.value = String(index);
  option.textContent = `${{run.label}} (${{run.id.length.toLocaleString()}} states)`;
  runSelect.appendChild(option);
}});

function transformedPotential(values) {{
  if (scaleSelect.value === "log") {{
    return values.map(value => Math.log10(value + summary.logEpsilon));
  }}
  return values;
}}

function colourSettings() {{
  if (scaleSelect.value === "log") {{
    return {{cmin: summary.logMin, cmax: summary.logMax, title: `log10(U + ${{summary.logEpsilon.toExponential(1)}})`}};
  }}
  return {{cmin: summary.linearMin, cmax: summary.linearMax, title: "U = sum_i |z_i - z_ref|"}};
}}

function markerFor(values, selectedRun) {{
  const colour = colourSettings();
  return {{
    size: selectedRun ? 3.5 : 2.0,
    color: transformedPotential(values),
    colorscale: "Viridis",
    cmin: colour.cmin,
    cmax: colour.cmax,
    opacity: selectedRun ? 0.88 : 0.58,
    showscale: true,
    colorbar: {{title: {{text: colour.title}}, thickness: 18, len: 0.82}}
  }};
}}

function highlightedMarkerFor(values, kind, selectedRun) {{
  const colour = colourSettings();
  const isRoot = kind === "root";
  return {{
    size: isRoot ? (selectedRun ? 11 : 8) : (selectedRun ? 8 : 6),
    symbol: isRoot ? "diamond" : "square",
    color: transformedPotential(values),
    colorscale: "Viridis",
    cmin: colour.cmin,
    cmax: colour.cmax,
    opacity: 1.0,
    showscale: false,
    line: {{
      color: isRoot ? "#111111" : "#00876c",
      width: isRoot ? 4 : 3
    }}
  }};
}}

function customData(run) {{
  const output = [];
  for (let i = 0; i < run.id.length; i += 1) {{
    output.push([run.label, run.id[i], run.parent[i], run.depth[i], run.cell[i], run.terminal[i], run.edge[i], run.u[i], run.outgoingPerturbations[i]]);
  }}
  return output;
}}

function highlightedPoints(points, kind) {{
  const output = {{x: [], y: [], z: [], u: [], custom: []}};
  for (let i = 0; i < points.custom.length; i += 1) {{
    const row = points.custom[i];
    const matches = kind === "root" ? row[2] === -1 : row[5] === "flat";
    if (!matches) continue;
    output.x.push(points.x[i]); output.y.push(points.y[i]); output.z.push(points.z[i]);
    output.u.push(points.u[i]); output.custom.push(row);
  }}
  return output;
}}

function edgesFor(run, perturbation) {{
  const x = [], y = [], z = [];
  const indexOf = new Map(run.id.map((stateId, index) => [stateId, index]));
  for (let child = 0; child < run.id.length; child += 1) {{
    const parentIndex = indexOf.get(run.parent[child]);
    if (parentIndex === undefined) continue;
    const isPerturbation = run.edge[child] === "perturbation";
    if (isPerturbation !== perturbation) continue;
    x.push(run.x[parentIndex], run.x[child], null);
    y.push(run.y[parentIndex], run.y[child], null);
    z.push(run.z[parentIndex], run.z[child], null);
  }}
  return {{x, y, z}};
}}

const hoverTemplate =
  "run=%{{customdata[0]}}<br>state=%{{customdata[1]}}<br>parent=%{{customdata[2]}}" +
  "<br>depth=%{{customdata[3]}}<br>cell=%{{customdata[4]}}<br>terminal=%{{customdata[5]}}" +
  "<br>incoming=%{{customdata[6]}}<br>outgoing perturbations=%{{customdata[8]}}" +
  "<br><b>raw U=%{{customdata[7]:.9g}}</b>" +
  "<br>theta_1=%{{x:.3f}} deg<br>theta_2=%{{y:.3f}} deg<br>theta_3=%{{z:.3f}} deg<extra></extra>";

const allRoots = highlightedPoints(all, "root");
const allFlats = highlightedPoints(all, "flat");

const traces = [
  {{
    type: "scatter3d", mode: "markers", name: "StateNode",
    x: all.x, y: all.y, z: all.z, customdata: all.custom,
    marker: markerFor(all.u, false), hovertemplate: hoverTemplate
  }},
  {{
    type: "scatter3d", mode: "lines", name: "deterministic step",
    x: [], y: [], z: [], visible: false, hoverinfo: "skip",
    line: {{color: "rgba(40,40,40,0.48)", width: 2}}
  }},
  {{
    type: "scatter3d", mode: "lines", name: "perturbation step",
    x: [], y: [], z: [], visible: false, hoverinfo: "skip",
    line: {{color: "rgba(214,39,40,0.86)", width: 4}}
  }},
  {{
    type: "scatter3d", mode: "markers", name: "Root (parent = -1)",
    x: allRoots.x, y: allRoots.y, z: allRoots.z, customdata: allRoots.custom,
    marker: highlightedMarkerFor(allRoots.u, "root", false),
    hovertemplate: hoverTemplate,
    legendrank: 10
  }},
  {{
    type: "scatter3d", mode: "markers", name: "Final flat state (terminal = flat)",
    x: allFlats.x, y: allFlats.y, z: allFlats.z, customdata: allFlats.custom,
    marker: highlightedMarkerFor(allFlats.u, "flat", false),
    hovertemplate: hoverTemplate,
    legendrank: 20
  }}
];

const layout = {{
  title: {{text: "Five-bar configuration-space forest - gravity potential", x: 0.5}},
  template: "plotly_white",
  margin: {{l: 8, r: 8, b: 8, t: 56}},
  legend: {{x: 0.01, y: 0.99}},
  hovermode: "closest",
  uirevision: "potential-forest-camera",
  scene: {{
    xaxis: {{title: "theta_1 [deg]"}}, yaxis: {{title: "theta_2 [deg]"}}, zaxis: {{title: "theta_3 [deg]"}},
    aspectmode: "data", dragmode: "orbit", camera: {{eye: {{x: 1.45, y: -1.65, z: 1.10}}}}
  }}
}};
const config = {{responsive: true, scrollZoom: true, displaylogo: false}};

function updateStatus(run) {{
  if (run === null) {{
    const roots = allRoots.x.length;
    const flats = allFlats.x.length;
    status.textContent = `${{summary.runCount}} runs / ${{summary.stateCount.toLocaleString()}} states | ${{roots.toLocaleString()}} roots / ${{flats.toLocaleString()}} flat terminals | global U ${{summary.potentialMin.toPrecision(5)}} to ${{summary.potentialMax.toPrecision(5)}} | select one run to show its edges`;
  }} else {{
    let minimum = Infinity, maximum = -Infinity;
    for (const value of run.u) {{ minimum = Math.min(minimum, value); maximum = Math.max(maximum, value); }}
    const runPoints = {{x: run.x, y: run.y, z: run.z, u: run.u, custom: customData(run)}};
    const roots = highlightedPoints(runPoints, "root").x.length;
    const flats = highlightedPoints(runPoints, "flat").x.length;
    status.textContent = `${{run.id.length.toLocaleString()}} states | ${{roots.toLocaleString()}} root / ${{flats.toLocaleString()}} flat terminals | U ${{minimum.toPrecision(5)}} to ${{maximum.toPrecision(5)}} | ${{run.format}} | global color range`;
  }}
}}

function updateView() {{
  const run = runSelect.value === "all" ? null : runs[Number(runSelect.value)];
  const points = run === null ? all : {{x: run.x, y: run.y, z: run.z, u: run.u, custom: customData(run)}};
  const deterministic = run === null ? {{x: [], y: [], z: []}} : edgesFor(run, false);
  const perturbation = run === null ? {{x: [], y: [], z: []}} : edgesFor(run, true);
  const roots = highlightedPoints(points, "root");
  const flats = highlightedPoints(points, "flat");
  const marker = markerFor(points.u, run !== null);
  Plotly.restyle(plot, {{x: [points.x], y: [points.y], z: [points.z], customdata: [points.custom], marker: [marker]}}, [0]);
  Plotly.restyle(plot, {{x: [deterministic.x], y: [deterministic.y], z: [deterministic.z], visible: run !== null}}, [1]);
  Plotly.restyle(plot, {{x: [perturbation.x], y: [perturbation.y], z: [perturbation.z], visible: run !== null}}, [2]);
  Plotly.restyle(plot, {{x: [roots.x], y: [roots.y], z: [roots.z], customdata: [roots.custom], marker: [highlightedMarkerFor(roots.u, "root", run !== null)]}}, [3]);
  Plotly.restyle(plot, {{x: [flats.x], y: [flats.y], z: [flats.z], customdata: [flats.custom], marker: [highlightedMarkerFor(flats.u, "flat", run !== null)]}}, [4]);
  const subtitle = run === null
    ? `${{summary.stateCount.toLocaleString()}} states from ${{summary.runCount}} runs`
    : `${{run.label}} - ${{run.id.length.toLocaleString()}} states`;
  Plotly.relayout(plot, {{"title.text": `Five-bar configuration-space forest - gravity potential<br><sup>${{subtitle}}</sup>`}});
  updateStatus(run);
}}

Plotly.newPlot(plot, traces, layout, config).then(() => {{
  updateStatus(null);
  runSelect.addEventListener("change", updateView);
  scaleSelect.addEventListener("change", updateView);
}});
</script>
</body>
</html>
"""


def _build_parser(script_directory: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recover five-bar StateNodes from search_tree.html/state_tree.npz "
            "and build a potential-coloured interactive forest."
        )
    )
    parser.add_argument(
        "--input-root",
        action="append",
        type=Path,
        default=None,
        help=(
            "file or directory to scan; repeatable. Default: every sibling "
            "*_five_bar_linkage/result directory"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_directory / "result" / "gravity_potential_forest.html",
        help="output HTML path",
    )
    parser.add_argument(
        "--reference-point",
        type=float,
        default=0.0,
        help="fallback z reference when an input has no saved value",
    )
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help="optional regular expression applied to source paths",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="limit sorted runs (primarily for smoke tests)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="stop at the first unreadable run instead of reporting and skipping it",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    script_directory = Path(__file__).resolve().parent
    parser = _build_parser(script_directory)
    args = parser.parse_args(argv)

    input_roots = (
        [path.resolve() for path in args.input_root]
        if args.input_root
        else _default_input_roots(script_directory)
    )
    if not input_roots:
        parser.error("no input roots found")
    if args.max_runs is not None and args.max_runs <= 0:
        parser.error("--max-runs must be positive")
    include_pattern = re.compile(args.include) if args.include else None

    sources = _discover_sources(input_roots)
    if include_pattern is not None:
        sources = [source for source in sources if include_pattern.search(str(source))]
    if args.max_runs is not None:
        sources = sources[: args.max_runs]
    if not sources:
        parser.error("no search_tree.html or state_tree.npz inputs found")

    label_root = _common_label_root(input_roots)
    model_cache: dict[Path, FiveBarModel] = {}
    runs: list[RunTree] = []
    skipped: list[tuple[Path, str]] = []

    for source in sources:
        try:
            model_directory = _find_model_directory(source)
            model = model_cache.get(model_directory)
            if model is None:
                model = _load_model(model_directory)
                model_cache[model_directory] = model
            label = _run_label(source, label_root)
            if source.name == NPZ_SOURCE_NAME:
                run = _load_npz_tree(
                    source, label, model, float(args.reference_point)
                )
            else:
                run = _load_html_tree(
                    source, label, model, float(args.reference_point)
                )
            _validate_run(run)
            runs.append(run)
        except Exception as error:  # report the run; strict mode re-raises below
            if args.strict:
                raise
            skipped.append((source, f"{type(error).__name__}: {error}"))

    if not runs:
        for source, reason in skipped:
            print(f"SKIP {source}: {reason}", file=sys.stderr)
        raise RuntimeError("no run could be recovered")

    runs.sort(key=lambda run: run.label.lower())
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_html_document(runs), encoding="utf-8")

    number_of_states = sum(run.number_of_states for run in runs)
    potential_min = min(float(np.min(run.potential)) for run in runs)
    potential_max = max(float(np.max(run.potential)) for run in runs)
    closure_max = max(run.closure_error_max for run in runs)
    print(
        f"Wrote {output_path}\n"
        f"Recovered {len(runs)} runs / {number_of_states} states; "
        f"U=[{potential_min:.9g}, {potential_max:.9g}], "
        f"maximum closure error={closure_max:.3e}"
    )
    if skipped:
        print(f"Skipped {len(skipped)} inputs:", file=sys.stderr)
        for source, reason in skipped:
            print(f"  {source}: {reason}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
