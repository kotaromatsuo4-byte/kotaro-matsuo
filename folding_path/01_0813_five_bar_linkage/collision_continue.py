"""Collision-continuing rollout helpers and two-layer terminal archive.

This module deliberately separates three concepts that were coupled in
``roll.py``:

* collision is an *edge observation*, not a terminal condition;
* rollout completion is caused by flatness or a numerical/algorithmic limit;
* terminal-cell coverage and collision-free path feasibility are recorded
  separately.

The functions are intentionally independent of ``main8.py``/``main9.py``.
Callers can keep per-state metadata in an external dictionary without changing
``StateNode``.  The existing collision implementations are reused verbatim, so
``collision_model='planar'`` has the same swept, x-z-plane meaning as
``collision2.py`` and ``collision_model='3d'`` has the legacy meaning from
``collision.py``.

Stop-reason semantics
---------------------
``flat``
    A completed rollout and the only outcome eligible for terminal archives.
``corrector_failure``
    The proposed state was not put on the constraint manifold.  Do not keep
    integrating from it and do not use it as a completed terminal teacher.
``timeout``
    The rollout-specific deterministic horizon was exhausted.  This is a
    right-censored rollout, not a terminal shape.
``global_budget``
    The experiment-wide mechanics budget was exhausted before flatness.  This
    is also right-censored.  It is kept distinct from ``timeout`` so learning
    code cannot mistake an accounting boundary for dynamics.

A transport/projection failure is not a physical terminal state.  A macro
caller should end the perturbation chain at the last valid state and start its
deterministic suffix there.  If no suffix can be run because the global budget
is empty, the result is ``global_budget``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from collision import edge_edge_crossing_check as _collision_check_3d
from collision2 import edge_edge_crossing_check as _collision_check_planar
from descriptor import (
    same_terminal_shape,
    terminal_continuous_descriptor,
    terminal_descriptor,
)
from mechanics import compute_gravity_potential, corrector, deterministic_step
from state import StateNode
from state import get_path_state_ids


FLAT = "flat"
CORRECTOR_FAILURE = "corrector_failure"
TIMEOUT = "timeout"
GLOBAL_BUDGET = "global_budget"

_COMPLETED_REASONS = frozenset((FLAT,))
_CENSORED_REASONS = frozenset((TIMEOUT, GLOBAL_BUDGET))


@dataclass(frozen=True, slots=True)
class EdgeAssessment:
    """Measured properties of one accepted parent-to-child edge.

    ``collision`` is a boolean edge event.  Consequently
    ``collision_edge_count`` below counts colliding integration edges, not the
    number of colliding member pairs.  This matches the information exposed by
    the existing collision modules, both of which stop at the first pair.

    ``geometric_length`` is the normalized RMS physical node displacement.
    Summing it gives a discretized arc length rather than merely a tree depth.
    """

    collision: bool
    geometric_length: float


@dataclass(frozen=True, slots=True)
class PathQuality:
    """Cumulative root-to-state path measurements."""

    collision_edge_count: int = 0
    geometric_path_length: float = 0.0
    edge_count: int = 0
    perturbation_count: int = 0

    @classmethod
    def zero(cls) -> "PathQuality":
        return cls()

    @property
    def collision_free(self) -> bool:
        return self.collision_edge_count == 0

    @property
    def path_has_collision(self) -> bool:
        return not self.collision_free

    def as_dict(self) -> dict:
        return {
            "collision_edge_count": int(self.collision_edge_count),
            "geometric_path_length": float(self.geometric_path_length),
            "edge_count": int(self.edge_count),
            "perturbation_count": int(self.perturbation_count),
            "collision_free": bool(self.collision_free),
            "path_has_collision": bool(self.path_has_collision),
        }


@dataclass(frozen=True, slots=True)
class ArchiveUpdate:
    """Result of one :func:`update_dual_archive` call."""

    accepted: bool
    cell_index: int | None
    new_kinematic_cell: bool = False
    new_feasible_cell: bool = False
    kinematic_elite_replaced: bool = False
    feasible_elite_replaced: bool = False
    rejection_reason: str | None = None

    def as_dict(self) -> dict:
        return {
            "accepted": bool(self.accepted),
            "cell_index": (
                None if self.cell_index is None else int(self.cell_index)
            ),
            "new_kinematic": bool(self.new_kinematic_cell),
            "new_feasible": bool(self.new_feasible_cell),
            "kinematic_replaced": bool(self.kinematic_elite_replaced),
            "feasible_replaced": bool(self.feasible_elite_replaced),
            "rejection_reason": self.rejection_reason,
        }


def _validate_nodes(node0, node1):
    node0_array = np.asarray(node0, dtype=float)
    node1_array = np.asarray(node1, dtype=float)
    if node0_array.shape != node1_array.shape:
        raise ValueError("node0 and node1 must have the same shape")
    if node0_array.ndim != 2 or node0_array.shape[1] != 3:
        raise ValueError("nodes must have shape (num_node, 3)")
    if not np.all(np.isfinite(node0_array)) or not np.all(np.isfinite(node1_array)):
        raise ValueError("node coordinates must be finite")
    return node0_array, node1_array


def edge_collision(
    node0,
    node1,
    member,
    dist_tol,
    normal_tol,
    cos_tol,
    collision_model="planar",
):
    """Return ``True`` iff the existing checker marks this edge invalid.

    Existing collision functions return ``0`` for collision and ``1`` for a
    valid edge.  This wrapper exposes ordinary boolean semantics and does not
    attach any terminal meaning to the result.
    """

    node0_array, node1_array = _validate_nodes(node0, node1)
    if collision_model == "planar":
        checker = _collision_check_planar
    elif collision_model == "3d":
        checker = _collision_check_3d
    else:
        raise ValueError("collision_model must be 'planar' or '3d'")

    valid = checker(
        node0_array,
        node1_array,
        member,
        dist_tol,
        normal_tol,
        cos_tol,
    )
    return bool(valid == 0)


def edge_geometric_length(node0, node1, length_scale=1.0, index_free=None):
    """Return normalized RMS node displacement for a single edge.

    With ``index_free=None``, all nodes contribute.  With flattened free-DOF
    indices, only those coordinates contribute and normalization uses the
    number of distinct participating nodes.  Fixed nodes therefore do not
    dilute the path length.  ``length_scale`` makes values comparable across
    geometrically scaled copies of a structure.
    """

    node0_array, node1_array = _validate_nodes(node0, node1)
    scale = float(length_scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("length_scale must be finite and positive")

    difference = (node1_array - node0_array).reshape(-1)
    if index_free is None:
        divisor = max(int(node0_array.shape[0]), 1)
        squared_norm = float(np.dot(difference, difference))
    else:
        free = np.asarray(index_free, dtype=int).reshape(-1)
        if free.size == 0:
            return 0.0
        if np.any(free < 0) or np.any(free >= difference.size):
            raise ValueError("index_free contains an out-of-range coordinate")
        selected = difference[free]
        squared_norm = float(np.dot(selected, selected))
        divisor = max(int(np.unique(free // 3).size), 1)

    return float(np.sqrt(squared_norm / divisor) / scale)


def assess_edge(
    node0,
    node1,
    member,
    dist_tol,
    normal_tol,
    cos_tol,
    collision_model="planar",
    length_scale=1.0,
    index_free=None,
):
    """Return collision and geometric-length measurements for one edge."""

    return EdgeAssessment(
        collision=edge_collision(
            node0,
            node1,
            member,
            dist_tol,
            normal_tol,
            cos_tol,
            collision_model,
        ),
        geometric_length=edge_geometric_length(
            node0,
            node1,
            length_scale,
            index_free,
        ),
    )


def _coerce_path_quality(quality):
    if isinstance(quality, PathQuality):
        return quality
    if isinstance(quality, Mapping):
        return PathQuality(
            collision_edge_count=int(quality["collision_edge_count"]),
            geometric_path_length=float(quality["geometric_path_length"]),
            edge_count=int(quality["edge_count"]),
            perturbation_count=int(quality.get("perturbation_count", 0)),
        )
    raise TypeError("quality must be PathQuality or a compatible mapping")


def advance_path_quality(parent_quality, edge_assessment, step_type=None):
    """Purely extend cumulative quality by one accepted edge."""

    parent = _coerce_path_quality(parent_quality)
    if not isinstance(edge_assessment, EdgeAssessment):
        if isinstance(edge_assessment, Mapping):
            edge_assessment = EdgeAssessment(
                collision=bool(edge_assessment["collision"]),
                geometric_length=float(edge_assessment["geometric_length"]),
            )
        else:
            raise TypeError(
                "edge_assessment must be EdgeAssessment or a compatible mapping"
            )
    length = float(edge_assessment.geometric_length)
    if not np.isfinite(length) or length < 0.0:
        raise ValueError("edge geometric length must be finite and nonnegative")
    return PathQuality(
        collision_edge_count=(
            parent.collision_edge_count + int(edge_assessment.collision)
        ),
        geometric_path_length=parent.geometric_path_length + length,
        edge_count=parent.edge_count + 1,
        perturbation_count=(
            parent.perturbation_count + int(step_type == "perturbation")
        ),
    )


def initialize_path_metrics(root_state_id=0):
    """Create external per-state cumulative metrics for a root StateNode.

    The dictionary is intentionally external to ``StateNode`` so existing
    history/result code remains compatible.  Values are immutable
    :class:`PathQuality` objects; extending one child never mutates its sibling
    or parent metrics.
    """

    return {int(root_state_id): PathQuality.zero()}


def add_child_continuing_collision(
    history,
    parent_id,
    node_after,
    member,
    reference_point,
    flatness_tolerance,
    dist_tol,
    normal_tol,
    cos_tol,
    path_metrics,
    proposed_displacement=None,
    step_type=None,
    correction_success=True,
    collision_model="planar",
    length_scale=1.0,
    index_free=None,
):
    """Append one child while recording, but never terminating on, collision.

    This is the collision-continuing counterpart of
    ``roll._evaluate_and_add_child``.  It mutates only the caller-owned
    ``history`` tree and ``path_metrics`` dictionary.  A colliding but
    corrected, non-flat child receives ``terminal_reason=None``; a colliding
    flat child receives ``terminal_reason='flat'`` and remains kinematically
    archive-eligible while its cumulative metrics make it infeasible.
    """

    parent_id = int(parent_id)
    if parent_id not in path_metrics:
        raise KeyError(
            f"path_metrics has no cumulative quality for parent {parent_id}"
        )
    parent = history[parent_id]
    node_after_array = np.asarray(node_after, dtype=float)
    assessment = assess_edge(
        parent.node,
        node_after_array,
        member,
        dist_tol,
        normal_tol,
        cos_tol,
        collision_model,
        length_scale,
        index_free,
    )
    potential = compute_gravity_potential(node_after_array, reference_point)
    terminal_reason = continuation_terminal_reason(
        correction_success,
        potential,
        flatness_tolerance,
    )

    state_id = len(history)
    child = StateNode(
        state_id=state_id,
        node=node_after_array.copy(),
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
    path_metrics[state_id] = advance_path_quality(
        path_metrics[parent_id],
        assessment,
        step_type,
    )
    return child, bool(assessment.collision)


def displacement_step_continue(
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
    path_metrics,
    collision_model="planar",
    length_scale=1.0,
):
    """Correct a proposed physical displacement and append its child.

    Corrector failure is terminal and must not be followed by another step.
    Collision is returned as a boolean observation and never changes the
    child's terminal reason.
    """

    displacement_array = np.asarray(displacement, dtype=float)
    node_predicted = history[int(parent_id)].node + displacement_array
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
    return add_child_continuing_collision(
        history,
        parent_id,
        node_after,
        member,
        reference_point,
        flatness_tolerance,
        dist_tol,
        normal_tol,
        cos_tol,
        path_metrics,
        proposed_displacement=displacement_array,
        step_type="perturbation",
        correction_success=success,
        collision_model=collision_model,
        length_scale=length_scale,
        index_free=index_free,
    )


def deterministic_step_continue(
    history,
    parent_id,
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
    path_metrics,
    collision_model="planar",
    length_scale=1.0,
):
    """Take one existing deterministic mechanics step without collision stop."""

    parent = history[int(parent_id)]
    initial_scale_ratio = 0.99
    initial_potential = float(history[0].potential)
    potential_scale = initial_potential * np.sqrt(
        (1.0 - initial_scale_ratio) / initial_scale_ratio
    )
    scale_denominator = parent.potential**2 + potential_scale**2
    if scale_denominator <= 1.0e-24:
        scale = 0.0
    else:
        scale = target_norm_max * parent.potential**2 / scale_denominator

    node_after, displacement, success, _ = deterministic_step(
        parent.node,
        member,
        L0,
        num_node,
        num_member,
        index_free,
        scale,
        constraint_tolerance,
        beta0,
        n_iter,
        l_iter,
        tau,
        cma_population,
        cma_generations,
        reference_point,
        cma_seed,
        return_info=True,
    )
    return add_child_continuing_collision(
        history,
        parent_id,
        node_after,
        member,
        reference_point,
        flatness_tolerance,
        dist_tol,
        normal_tol,
        cos_tol,
        path_metrics,
        proposed_displacement=displacement,
        step_type="deterministic",
        correction_success=success,
        collision_model=collision_model,
        length_scale=length_scale,
        index_free=index_free,
    )


def path_quality_from_history(
    history,
    state_id,
    member,
    dist_tol,
    normal_tol,
    cos_tol,
    collision_model="planar",
    length_scale=1.0,
    index_free=None,
):
    """Recompute root-to-state quality without requiring stored metadata."""

    path = get_path_state_ids(history, int(state_id))
    quality = PathQuality.zero()
    for parent_id, child_id in zip(path[:-1], path[1:]):
        parent = history[int(parent_id)]
        child = history[int(child_id)]
        assessment = assess_edge(
            parent.node,
            child.node,
            member,
            dist_tol,
            normal_tol,
            cos_tol,
            collision_model,
            length_scale,
            index_free,
        )
        quality = advance_path_quality(
            quality,
            assessment,
            getattr(child, "incoming_step_type", None),
        )
    return quality


def continuation_terminal_reason(
    correction_success,
    potential,
    flatness_tolerance,
):
    """Classify an accepted edge while deliberately ignoring collision."""

    if not bool(correction_success):
        return CORRECTOR_FAILURE
    potential_value = float(potential)
    tolerance = float(flatness_tolerance)
    if not np.isfinite(potential_value) or not np.isfinite(tolerance):
        raise ValueError("potential and flatness_tolerance must be finite")
    if potential_value <= tolerance:
        return FLAT
    return None


def reclassify_legacy_terminal_reason(
    legacy_reason,
    potential,
    flatness_tolerance,
):
    """Translate a ``roll.py`` child into collision-continuing semantics.

    A legacy ``collision`` child necessarily passed the corrector because
    ``roll.py`` gives corrector failure higher priority.  Re-evaluating only
    flatness is therefore sufficient.  Algorithmic rollout reasons are kept
    unchanged if a caller passes them through this helper.
    """

    if legacy_reason == CORRECTOR_FAILURE:
        return CORRECTOR_FAILURE
    if legacy_reason in (None, FLAT, "collision"):
        return continuation_terminal_reason(
            True,
            potential,
            flatness_tolerance,
        )
    if legacy_reason in (TIMEOUT, GLOBAL_BUDGET):
        return legacy_reason
    raise ValueError(f"unsupported legacy terminal reason: {legacy_reason!r}")


def classify_rollout_stop(
    current_terminal_reason,
    steps_executed,
    max_rollout_steps,
    remaining_budget_at_start,
):
    """Resolve physical, horizon, and global-budget stopping semantics.

    The caller invokes this after it can no longer take another suffix step.
    If both limits are numerically equal, ``timeout`` takes precedence because
    the rollout-specific horizon alone would have stopped the rollout at that
    point.  A return value of ``None`` means that no stopping condition is yet
    justified and the caller should continue.
    """

    if current_terminal_reason in (FLAT, CORRECTOR_FAILURE):
        return current_terminal_reason
    if current_terminal_reason not in (None, "collision"):
        if current_terminal_reason in (TIMEOUT, GLOBAL_BUDGET):
            return current_terminal_reason
        raise ValueError(
            f"unsupported current terminal reason: {current_terminal_reason!r}"
        )

    steps = int(steps_executed)
    horizon = int(max_rollout_steps)
    budget = int(remaining_budget_at_start)
    if steps < 0 or horizon <= 0 or budget < 0:
        raise ValueError(
            "steps_executed >= 0, max_rollout_steps > 0, and "
            "remaining_budget_at_start >= 0 are required"
        )
    if steps >= horizon and horizon <= budget:
        return TIMEOUT
    if steps >= budget and budget < horizon:
        return GLOBAL_BUDGET
    return None


def is_completed_rollout(reason):
    """Return whether ``reason`` denotes an archive-eligible terminal shape."""

    return reason in _COMPLETED_REASONS


def is_censored_rollout(reason):
    """Return whether ``reason`` denotes an incomplete right-censored rollout."""

    return reason in _CENSORED_REASONS


def feasibility_first_quality_key(quality):
    """Lexicographic key that never trades feasibility for a shorter path."""

    value = _coerce_path_quality(quality)
    return (
        int(not value.collision_free),
        int(value.collision_edge_count),
        float(value.geometric_path_length),
        int(value.perturbation_count),
        int(value.edge_count),
    )


def kinematic_quality_key(quality):
    """Shortest-path key when collision is intentionally ignored."""

    value = _coerce_path_quality(quality)
    return (
        float(value.geometric_path_length),
        int(value.perturbation_count),
        int(value.edge_count),
        int(value.collision_edge_count),
    )


def feasible_quality_key(quality):
    """Shortest feasible-path key; return ``None`` for an infeasible path."""

    value = _coerce_path_quality(quality)
    if not value.collision_free:
        return None
    return (
        float(value.geometric_path_length),
        int(value.perturbation_count),
        int(value.edge_count),
    )


def is_better_quality(candidate, incumbent, mode="feasibility_first"):
    """Compare path qualities using an explicit, scale-free policy."""

    if incumbent is None:
        return True
    if mode == "feasibility_first":
        key_function = feasibility_first_quality_key
    elif mode == "kinematic":
        key_function = kinematic_quality_key
    elif mode == "feasible":
        candidate_key = feasible_quality_key(candidate)
        incumbent_key = feasible_quality_key(incumbent)
        if candidate_key is None:
            return False
        if incumbent_key is None:
            return True
        return candidate_key < incumbent_key
    else:
        raise ValueError(
            "mode must be 'feasibility_first', 'kinematic', or 'feasible'"
        )
    return key_function(candidate) < key_function(incumbent)


def new_dual_archive():
    """Return an empty kinematic/feasible terminal-cell archive."""

    return []


def find_dual_archive_cell(
    archive,
    shape_descriptor,
    layer_key,
    rms_tolerance,
    max_tolerance,
):
    """Return the matching stable cell index, or ``None``."""

    for cell_index, cell in enumerate(archive):
        if layer_key != cell["layer_key"]:
            continue
        if same_terminal_shape(
            shape_descriptor,
            cell["shape_descriptor"],
            rms_tolerance,
            max_tolerance,
        ):
            return int(cell_index)
    return None


def _elite_record(terminal_state_id, continuous_descriptor, quality):
    value = _coerce_path_quality(quality)
    return {
        "terminal_state_id": int(terminal_state_id),
        "continuous_descriptor": np.asarray(
            continuous_descriptor,
            dtype=float,
        ).copy(),
        "quality": value.as_dict(),
    }


def update_dual_archive(
    archive,
    history,
    terminal_state_id,
    quality,
    member,
    length_scale,
    layer_threshold,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
    rms_tolerance,
    max_tolerance,
    reference_point=0.0,
    require_flat=True,
):
    """Update kinematic coverage and collision-free elites for one flat path.

    ``archive`` is a list of stable cell dictionaries.  Every accepted flat
    endpoint creates/updates ``kinematic_elite`` with the feasibility-first
    quality policy.  A collision-free path also creates/updates
    ``feasible_elite`` using geometric path length.  Thus cell count and
    feasible-cell count can evolve independently.

    With the default ``require_flat=True``, a StateNode whose terminal reason
    is not ``flat`` is rejected.  This prevents timeouts, global-budget
    censoring, and corrector failures from silently becoming cells.  A caller
    using immutable legacy nodes may pass ``require_flat=False`` only after it
    has independently classified the endpoint as flat with
    :func:`continuation_terminal_reason`.
    """

    terminal_state_id = int(terminal_state_id)
    terminal_state = history[terminal_state_id]
    if require_flat and terminal_state.terminal_reason != FLAT:
        return ArchiveUpdate(
            accepted=False,
            cell_index=None,
            rejection_reason="not_flat",
        )

    path_quality = _coerce_path_quality(quality)
    shape_descriptor, layer_key = terminal_descriptor(
        history,
        terminal_state_id,
        member,
        length_scale,
        layer_threshold,
        overlap_tolerance,
        separation_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
    )
    if any(upper_member == -1 for _, _, upper_member in layer_key):
        return ArchiveUpdate(
            accepted=False,
            cell_index=None,
            rejection_reason="ambiguous_layer_order",
        )
    continuous_descriptor = terminal_continuous_descriptor(
        history,
        terminal_state_id,
        length_scale,
        layer_threshold,
        reference_point,
    )
    record = _elite_record(
        terminal_state_id,
        continuous_descriptor,
        path_quality,
    )
    cell_index = find_dual_archive_cell(
        archive,
        shape_descriptor,
        layer_key,
        rms_tolerance,
        max_tolerance,
    )

    if cell_index is None:
        feasible_record = record if path_quality.collision_free else None
        archive.append(
            {
                "shape_descriptor": np.asarray(
                    shape_descriptor,
                    dtype=float,
                ).copy(),
                "layer_key": layer_key,
                "first_terminal_state_id": terminal_state_id,
                "kinematic_elite": record,
                "feasible_elite": feasible_record,
            }
        )
        return ArchiveUpdate(
            accepted=True,
            cell_index=len(archive) - 1,
            new_kinematic_cell=True,
            new_feasible_cell=bool(path_quality.collision_free),
            kinematic_elite_replaced=True,
            feasible_elite_replaced=bool(path_quality.collision_free),
        )

    cell = archive[cell_index]
    kinematic_replaced = False
    incumbent_kinematic = cell.get("kinematic_elite")
    if incumbent_kinematic is None or is_better_quality(
        path_quality,
        incumbent_kinematic["quality"],
        mode="feasibility_first",
    ):
        cell["kinematic_elite"] = record
        kinematic_replaced = True

    feasible_replaced = False
    new_feasible_cell = False
    if path_quality.collision_free:
        incumbent_feasible = cell.get("feasible_elite")
        if incumbent_feasible is None:
            new_feasible_cell = True
        if incumbent_feasible is None or is_better_quality(
            path_quality,
            incumbent_feasible["quality"],
            mode="feasible",
        ):
            cell["feasible_elite"] = record
            feasible_replaced = True

    return ArchiveUpdate(
        accepted=True,
        cell_index=cell_index,
        new_feasible_cell=new_feasible_cell,
        kinematic_elite_replaced=kinematic_replaced,
        feasible_elite_replaced=feasible_replaced,
    )


def update_dual_archive_from_metrics(
    archive,
    history,
    terminal_state_id,
    path_metrics,
    member,
    length_scale,
    layer_threshold,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
    rms_tolerance,
    max_tolerance,
    reference_point=0.0,
    require_flat=True,
):
    """Path-metadata adapter returning a plain main-loop-friendly dictionary."""

    terminal_state_id = int(terminal_state_id)
    if terminal_state_id not in path_metrics:
        raise KeyError(
            "path_metrics has no cumulative quality for terminal state "
            f"{terminal_state_id}"
        )
    result = update_dual_archive(
        archive,
        history,
        terminal_state_id,
        path_metrics[terminal_state_id],
        member,
        length_scale,
        layer_threshold,
        overlap_tolerance,
        separation_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
        rms_tolerance,
        max_tolerance,
        reference_point,
        require_flat,
    )
    return result.as_dict()


def dual_archive_counts(archive):
    """Return ``(kinematic_cell_count, feasible_cell_count)``."""

    return (
        int(len(archive)),
        int(sum(cell.get("feasible_elite") is not None for cell in archive)),
    )


def archive_layer_view(archive, layer="kinematic"):
    """Return legacy-shaped elite dictionaries for result-saving functions.

    The returned list is detached from the archive records, while descriptor
    arrays are shared read-only by convention.  Feasible view omits cells that
    have only been reached by colliding paths.
    """

    if layer not in ("kinematic", "feasible"):
        raise ValueError("layer must be 'kinematic' or 'feasible'")
    record_name = f"{layer}_elite"
    result = []
    for cell in archive:
        record = cell.get(record_name)
        if record is None:
            continue
        result.append(
            {
                "shape_descriptor": cell["shape_descriptor"],
                "layer_key": cell["layer_key"],
                "continuous_descriptor": record["continuous_descriptor"],
                "terminal_state_id": int(record["terminal_state_id"]),
                "quality": dict(record["quality"]),
            }
        )
    return result
