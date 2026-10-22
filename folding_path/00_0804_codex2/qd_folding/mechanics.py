"""Five-bar mechanics and deterministic rollout primitives.

The search algorithms share this module and differ only in how they choose a
restart state or an intervention sequence.  A path is stored as a matrix with
shape ``(3 * number_of_nodes, number_of_states)``.  Column zero is the root
state, so ``path.shape[1] - 1`` is the full root-to-terminal path length.

``compute_steps`` has deliberately different semantics: it counts only update
attempts newly performed by the current call.  A reused prefix therefore
contributes to path length, but not to the new computation budget.  Both an
intervention attempt and one deterministic gravity update consume one compute
step.  A failed update is counted too, preventing repeated failures from
becoming free under a finite budget.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from cmaes import CMA

from .config import ExperimentConfig


Array = np.ndarray


@dataclass(frozen=True, slots=True)
class Model:
    """Bar-and-joint model and quantities derived from its input files."""

    node_initial: Array
    member: Array
    fix: Array
    index_free: Array
    length_initial: Array

    @property
    def num_node(self) -> int:
        return int(self.node_initial.shape[0])

    @property
    def num_member(self) -> int:
        return int(self.member.shape[0])

    @property
    def vector_size(self) -> int:
        return 3 * self.num_node

    @property
    def length_scale(self) -> float:
        return float(np.mean(self.length_initial))


@dataclass(slots=True)
class StepResult:
    """Result of one corrected mechanism update.

    ``action`` is the executed predictor displacement in global nodal
    coordinates, flattened to length ``3 * num_node``.  Keeping actions in this
    fixed coordinate system avoids exposing the sign and ordering ambiguity of
    SVD null-space bases to sequence search.
    """

    node: Array | None
    action: Array
    coefficients: Array
    residual: float
    terminal_reason: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.node is not None

    @property
    def collided(self) -> bool:
        return self.terminal_reason == "collision"


@dataclass(frozen=True, slots=True)
class Intervention:
    """One latent intervention after ``wait_steps`` deterministic updates."""

    wait_steps: int
    latent: Array


def load_model(data_dir: str | Path) -> Model:
    """Load ``node_initial.dat``, ``member.dat``, and ``fix.dat``."""

    folder = Path(data_dir)
    node = np.atleast_2d(
        np.loadtxt(folder / "node_initial.dat", delimiter=",", dtype=float)
    )
    member = np.atleast_2d(
        np.loadtxt(folder / "member.dat", delimiter=",", dtype=int)
    )
    fix = np.atleast_2d(
        np.loadtxt(folder / "fix.dat", delimiter=",", dtype=int)
    )
    index = np.arange(node.size, dtype=int)
    fixed_index = 3 * fix[:, 0] + fix[:, 1]
    index_free = np.delete(index, fixed_index)
    length_initial = np.linalg.norm(
        node[member[:, 0]] - node[member[:, 1]], axis=1
    )
    return Model(
        node_initial=node,
        member=member,
        fix=fix,
        index_free=index_free,
        length_initial=length_initial,
    )


def node_to_vector(node: Array) -> Array:
    """Flatten an ``(N, 3)`` nodal configuration."""

    return np.asarray(node, dtype=float).reshape(-1)


def vector_to_node(vector: Array, model: Model) -> Array:
    """Reshape a flat nodal vector using ``model.num_node``."""

    return np.asarray(vector, dtype=float).reshape(model.num_node, 3)


def start_path(node: Array) -> Array:
    """Create a one-state path matrix."""

    return node_to_vector(node)[:, None]


def append_state(path: Array, node: Array) -> Array:
    """Return ``path`` with ``node`` appended as its final column."""

    return np.column_stack((np.asarray(path, dtype=float), node_to_vector(node)))


def path_nodes(path: Array, model: Model) -> list[Array]:
    """Convert a path matrix to a list of copied nodal configurations."""

    matrix = np.asarray(path, dtype=float)
    return [
        vector_to_node(matrix[:, index], model).copy()
        for index in range(matrix.shape[1])
    ]


def constraint_jacobian(node: Array, model: Model) -> Array:
    """Return the member-length constraint Jacobian on free coordinates."""

    jacobian_all = np.zeros((model.num_member, model.vector_size), dtype=float)
    differences = node[model.member[:, 0]] - node[model.member[:, 1]]
    for row, (first, second) in enumerate(model.member):
        jacobian_all[row, 3 * first : 3 * (first + 1)] = differences[row]
        jacobian_all[row, 3 * second : 3 * (second + 1)] = -differences[row]
    return jacobian_all[:, model.index_free]


def calc_dedx(node: Array, model: Model) -> Array:
    """Compatibility alias for :func:`constraint_jacobian`."""

    return constraint_jacobian(node, model)


def mechanism_modes(
    node: Array,
    model: Model,
    tolerance: float = 1.0e-8,
) -> Array:
    """Return an orthonormal global-coordinate basis of infinitesimal modes."""

    jacobian = constraint_jacobian(node, model)
    _, singular_values, vh = np.linalg.svd(jacobian, full_matrices=True)
    rank = int(np.count_nonzero(singular_values > tolerance))
    free_modes = vh[rank:].T
    modes = np.zeros((model.vector_size, free_modes.shape[1]), dtype=float)
    modes[model.index_free, :] = free_modes
    return modes


def calc_mode(
    node: Array,
    model: Model,
    tolerance: float = 1.0e-8,
) -> Array:
    """Compatibility alias for :func:`mechanism_modes`."""

    return mechanism_modes(node, model, tolerance)


def get_modes(
    node: Array,
    model: Model,
    tolerance: float = 1.0e-8,
) -> Array:
    """Compatibility alias for :func:`mechanism_modes`."""

    return mechanism_modes(node, model, tolerance)


def predictor(node: Array, modes: Array, coefficients: Array) -> Array:
    """Apply an infinitesimal mechanism combination before correction."""

    displacement = modes @ np.asarray(coefficients, dtype=float).reshape(-1)
    return np.asarray(node, dtype=float) + displacement.reshape(-1, 3)


def incompatibility(node: Array, model: Model) -> Array:
    """Squared member-length constraint residuals."""

    differences = node[model.member[:, 0]] - node[model.member[:, 1]]
    squared_lengths = np.einsum("ij,ij->i", differences, differences)
    return 0.5 * (squared_lengths - model.length_initial**2)


def corrector_increment(
    node: Array,
    model: Model,
    beta0: float = 1.0,
    line_search_iterations: int = 20,
    tau: float = 0.5,
) -> Array:
    """Compute one damped pseudoinverse corrector increment."""

    residual = incompatibility(node, model)
    jacobian = constraint_jacobian(node, model)
    displacement_free = -np.linalg.pinv(jacobian) @ residual
    displacement = np.zeros(model.vector_size, dtype=float)
    displacement[model.index_free] = displacement_free

    residual_norm = float(np.linalg.norm(residual))
    if residual_norm == 0.0:
        return np.zeros_like(node, dtype=float)

    beta = float(beta0)
    for _ in range(line_search_iterations):
        candidate = node + beta * displacement.reshape(-1, 3)
        candidate_norm = float(np.linalg.norm(incompatibility(candidate, model)))
        if candidate_norm < (1.0 - 1.0e-4 * beta) * residual_norm:
            break
        beta *= tau
    return beta * displacement.reshape(-1, 3)


def correct_configuration(
    node: Array,
    model: Model,
    max_iterations: int,
    tolerance: float,
) -> tuple[Array, float, bool]:
    """Project a predicted configuration back to the constraint manifold."""

    corrected = np.asarray(node, dtype=float).copy()
    for _ in range(max_iterations):
        increment = corrector_increment(corrected, model)
        corrected += increment
        if np.linalg.norm(increment) < tolerance:
            break
    residual = float(np.linalg.norm(incompatibility(corrected, model)))
    finite = bool(np.all(np.isfinite(corrected)) and np.isfinite(residual))
    return corrected, residual, finite


def gravity_potential(node: Array) -> float:
    """Flatness-driving potential used by the deterministic local optimizer."""

    return float(np.sum(np.abs(np.asarray(node, dtype=float)[:, 2])))


def cross2(first: Array, second: Array) -> float:
    """Two-dimensional scalar cross product."""

    return float(first[0] * second[1] - first[1] * second[0])


def proper_segment_intersection(
    first_start: Array,
    first_end: Array,
    second_start: Array,
    second_end: Array,
    tolerance: float,
) -> bool:
    """Whether two projected segments intersect strictly in their interiors."""

    first_direction = first_end - first_start
    second_direction = second_end - second_start
    denominator = cross2(first_direction, second_direction)
    if abs(denominator) <= tolerance:
        return False
    first_parameter = (
        cross2(second_start - first_start, second_direction) / denominator
    )
    second_parameter = (
        cross2(second_start - first_start, first_direction) / denominator
    )
    return (
        tolerance < first_parameter < 1.0 - tolerance
        and tolerance < second_parameter < 1.0 - tolerance
    )


def has_member_collision(
    node: Array,
    model: Model,
    tolerance: float,
) -> bool:
    """Detect proper intersections of non-adjacent members in the x-z plane."""

    points = np.asarray(node, dtype=float)[:, [0, 2]]
    for first_index, (first_start, first_end) in enumerate(model.member):
        for second_index in range(first_index + 1, model.num_member):
            second_start, second_end = model.member[second_index]
            if (
                len(
                    {
                        int(first_start),
                        int(first_end),
                        int(second_start),
                        int(second_end),
                    }
                )
                < 4
            ):
                continue
            if proper_segment_intersection(
                points[first_start],
                points[first_end],
                points[second_start],
                points[second_end],
                tolerance,
            ):
                return True
    return False


def sample_uniform_ball(
    dimension: int,
    radius: float,
    rng: np.random.Generator,
) -> Array:
    """Sample uniformly with respect to volume inside a Euclidean ball."""

    if dimension < 0:
        raise ValueError("dimension must be non-negative")
    if radius < 0.0:
        raise ValueError("radius must be non-negative")
    if dimension == 0 or radius == 0.0:
        return np.zeros(dimension, dtype=float)

    direction = rng.normal(size=dimension)
    direction_norm = float(np.linalg.norm(direction))
    while direction_norm < 1.0e-15:
        direction = rng.normal(size=dimension)
        direction_norm = float(np.linalg.norm(direction))
    radial_fraction = float(rng.random()) ** (1.0 / dimension)
    return radius * radial_fraction * direction / direction_norm


def clamp_norm(vector: Array, maximum_norm: float) -> Array:
    """Clamp a vector to a closed Euclidean ball."""

    value = np.asarray(vector, dtype=float).reshape(-1)
    norm = float(np.linalg.norm(value))
    if norm <= maximum_norm:
        return value.copy()
    return maximum_norm * value / max(norm, 1.0e-15)


def deterministic_state_seed(
    node: Array,
    absolute_step: int,
    base_seed: int,
) -> int:
    """Create a stable local-CMA seed from state, depth, and experiment seed."""

    digest = hashlib.blake2b(digest_size=8)
    digest.update(
        np.asarray([base_seed, absolute_step], dtype=np.int64).tobytes()
    )
    digest.update(
        np.round(np.asarray(node, dtype=float), decimals=10)
        .astype(np.float64)
        .tobytes()
    )
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


def _corrected_step_result(
    node: Array,
    modes: Array,
    coefficients: Array,
    model: Model,
    config: ExperimentConfig,
) -> StepResult:
    """Predict, correct, and classify one mechanism update."""

    coefficient_vector = np.asarray(coefficients, dtype=float).reshape(-1)
    predictor_displacement = modes @ coefficient_vector
    predicted = node + predictor_displacement.reshape(-1, 3)
    corrected, residual, finite = correct_configuration(
        predicted,
        model,
        config.corrector_iterations,
        config.corrector_tolerance,
    )
    if not finite or residual > config.residual_tolerance:
        return StepResult(
            node=None,
            action=predictor_displacement.copy(),
            coefficients=coefficient_vector.copy(),
            residual=residual,
            terminal_reason="correction_failed",
        )

    reason = (
        "collision"
        if has_member_collision(
            corrected, model, config.collision_tolerance
        )
        else None
    )
    return StepResult(
        node=corrected,
        action=predictor_displacement.copy(),
        coefficients=coefficient_vector.copy(),
        residual=residual,
        terminal_reason=reason,
    )


def apply_mode_perturbation(
    node: Array,
    coefficients: Array,
    model: Model,
    config: ExperimentConfig,
    maximum_norm: float | None = None,
) -> StepResult:
    """Apply bounded coefficients in the current SVD mechanism basis."""

    modes = mechanism_modes(node, model)
    if modes.shape[1] == 0:
        return StepResult(
            node=None,
            action=np.zeros(model.vector_size),
            coefficients=np.empty(0),
            residual=math.inf,
            terminal_reason="no_mechanism",
        )
    coefficient_vector = np.asarray(coefficients, dtype=float).reshape(-1)
    if coefficient_vector.size != modes.shape[1]:
        raise ValueError(
            "coefficient dimension does not match the current mechanism basis"
        )
    limit = (
        config.perturb_max_norm
        if maximum_norm is None
        else float(maximum_norm)
    )
    bounded = clamp_norm(coefficient_vector, limit)
    return _corrected_step_result(node, modes, bounded, model, config)


def apply_latent_perturbation(
    node: Array,
    latent_displacement: Array,
    model: Model,
    config: ExperimentConfig,
    maximum_norm: float | None = None,
) -> StepResult:
    """Project a fixed-dimensional nodal latent vector onto current modes.

    The latent input always has ``3 * num_node`` entries.  Its physical
    direction is therefore stable even if an SVD mode changes sign or swaps
    order.  Only its projection onto the current infinitesimal mechanism space
    is executed.
    """

    latent = np.asarray(latent_displacement, dtype=float).reshape(-1)
    if latent.size != model.vector_size:
        raise ValueError(
            f"latent displacement must have {model.vector_size} entries"
        )
    modes = mechanism_modes(node, model)
    if modes.shape[1] == 0:
        return StepResult(
            node=None,
            action=np.zeros(model.vector_size),
            coefficients=np.empty(0),
            residual=math.inf,
            terminal_reason="no_mechanism",
        )

    limit = (
        config.perturb_max_norm
        if maximum_norm is None
        else float(maximum_norm)
    )
    requested_norm = min(float(np.linalg.norm(latent)), limit)
    coefficients = modes.T @ latent
    projected = modes @ coefficients
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= 1.0e-14 or requested_norm <= 1.0e-14:
        return StepResult(
            node=None,
            action=np.zeros(model.vector_size),
            coefficients=np.zeros(modes.shape[1]),
            residual=math.inf,
            terminal_reason="no_projected_perturbation",
        )
    projected = requested_norm * projected / projected_norm
    coefficients = modes.T @ projected
    return _corrected_step_result(
        node, modes, coefficients, model, config
    )


def deterministic_one_step(
    node: Array,
    absolute_step: int,
    model: Model,
    config: ExperimentConfig,
) -> StepResult:
    """Choose one deterministic gravity-reducing mechanism update by CMA-ES.

    Candidate collision is not penalized inside the optimizer.  If the selected
    minimum-potential update collides, that corrected collision configuration
    is returned and becomes the explicit rollout endpoint.
    """

    modes = mechanism_modes(node, model)
    dimension = modes.shape[1]
    if dimension == 0:
        return StepResult(
            node=None,
            action=np.zeros(model.vector_size),
            coefficients=np.empty(0),
            residual=math.inf,
            terminal_reason="no_mechanism",
        )

    target_norm = config.gravity_step_norm * np.exp(
        -absolute_step / max(config.max_path_steps, 1)
    )
    seed = deterministic_state_seed(node, absolute_step, config.seed)
    optimizer = CMA(
        mean=np.zeros(dimension),
        sigma=config.local_cma_sigma,
        population_size=config.local_cma_population,
        seed=seed,
    )
    best: StepResult | None = None
    best_potential = math.inf

    for _ in range(config.local_cma_generations):
        solutions: list[tuple[Array, float]] = []
        for _ in range(optimizer.population_size):
            raw = optimizer.ask()
            coefficients = clamp_norm(raw, target_norm)
            # Every nonzero proposal uses the same deterministic step length.
            raw_norm = float(np.linalg.norm(coefficients))
            if raw_norm > 1.0e-15:
                coefficients *= target_norm / raw_norm
            result = _corrected_step_result(
                node, modes, coefficients, model, config
            )
            objective = (
                1.0e12
                if result.node is None
                else gravity_potential(result.node)
            )
            solutions.append((raw, float(objective)))
            if result.node is not None and objective < best_potential:
                best = result
                best_potential = float(objective)
        optimizer.tell(solutions)
        if optimizer.should_stop():
            break

    if best is None:
        return StepResult(
            node=None,
            action=np.zeros(model.vector_size),
            coefficients=np.zeros(dimension),
            residual=math.inf,
            terminal_reason="optimizer_failed",
        )
    return best


def optimize_one_folding_step(
    node: Array,
    absolute_step: int,
    model: Model,
    config: ExperimentConfig,
) -> StepResult:
    """Compatibility alias for :func:`deterministic_one_step`."""

    return deterministic_one_step(node, absolute_step, model, config)


def _normalise_prefix(
    start_node: Array,
    prefix_path: Array | None,
    prefix_actions: Sequence[Array] | None,
    prefix_action_kinds: Sequence[str] | None,
    model: Model,
) -> tuple[Array, list[Array], list[str]]:
    """Validate and copy a root-to-start prefix."""

    if prefix_path is None:
        path = start_path(start_node)
    else:
        path = np.asarray(prefix_path, dtype=float).copy()
        if path.ndim != 2 or path.shape[0] != model.vector_size:
            raise ValueError("prefix_path has an incompatible shape")
        if not np.allclose(
            path[:, -1], node_to_vector(start_node), rtol=0.0, atol=1.0e-10
        ):
            raise ValueError("start_node must equal the final prefix state")

    actions = [
        np.asarray(action, dtype=float).reshape(-1).copy()
        for action in (prefix_actions or [])
    ]
    if len(actions) != path.shape[1] - 1:
        raise ValueError("prefix must have exactly one action per transition")
    if prefix_action_kinds is None:
        action_kinds = ["prefix"] * len(actions)
    else:
        action_kinds = list(prefix_action_kinds)
        if len(action_kinds) != len(actions):
            raise ValueError(
                "prefix_action_kinds must have one entry per action"
            )
    return path, actions, action_kinds


def make_trajectory(
    path: Array,
    actions: Sequence[Array],
    source: str,
    terminal_reason: str,
    new_start_step: int,
    intervention_count: int,
    compute_steps: int,
    model: Model,
    action_kinds: Sequence[str] | None = None,
    intervention_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Build the common trajectory dictionary required by all algorithms."""

    matrix = np.asarray(path, dtype=float)
    action_values = [
        np.asarray(action, dtype=float).reshape(-1).copy()
        for action in actions
    ]
    if matrix.ndim != 2 or matrix.shape[0] != model.vector_size:
        raise ValueError("path has an incompatible shape")
    if len(action_values) != matrix.shape[1] - 1:
        raise ValueError("path must contain exactly one more state than actions")
    final_node = vector_to_node(matrix[:, -1], model)
    final_flatness = gravity_potential(final_node)
    collided = terminal_reason == "collision"
    failed = terminal_reason in {
        "correction_failed",
        "no_mechanism",
        "no_projected_perturbation",
        "optimizer_failed",
        "invalid_sequence",
    }
    trajectory = {
        "path": matrix.copy(),
        "actions": action_values,
        "source": str(source),
        "terminal_reason": str(terminal_reason),
        "is_flat": terminal_reason == "flat",
        "collided": collided,
        "failed": failed,
        "new_start_step": int(new_start_step),
        "intervention_count": int(intervention_count),
        "compute_steps": int(compute_steps),
        "final_flatness": float(final_flatness),
        "action_kinds": (
            list(action_kinds)
            if action_kinds is not None
            else ["unknown"] * len(action_values)
        ),
        "intervention_steps": list(intervention_steps or []),
    }
    return trajectory


def _initial_terminal_reason(
    node: Array,
    model: Model,
    config: ExperimentConfig,
) -> str | None:
    if has_member_collision(node, model, config.collision_tolerance):
        return "collision"
    if gravity_potential(node) <= config.flatness_tolerance:
        return "flat"
    return None


def rollout_from_prefix(
    start_node: Array,
    prefix_path: Array,
    prefix_actions: Sequence[Array],
    model: Model,
    config: ExperimentConfig,
    source: str,
    remaining_compute_steps: int | None = None,
    *,
    prefix_action_kinds: Sequence[str] | None = None,
    prefix_intervention_count: int = 0,
    new_start_step: int | None = None,
    prefix_intervention_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Continue deterministically from a reusable full root prefix."""

    path, actions, action_kinds = _normalise_prefix(
        start_node,
        prefix_path,
        prefix_actions,
        prefix_action_kinds,
        model,
    )
    restart_step = (
        path.shape[1] - 1 if new_start_step is None else int(new_start_step)
    )
    intervention_steps = list(prefix_intervention_steps or [])
    budget = (
        config.max_path_steps
        if remaining_compute_steps is None
        else max(int(remaining_compute_steps), 0)
    )
    compute_steps = 0
    node = np.asarray(start_node, dtype=float).copy()

    reason = _initial_terminal_reason(node, model, config)
    if reason is not None:
        return make_trajectory(
            path,
            actions,
            source,
            reason,
            restart_step,
            prefix_intervention_count,
            compute_steps,
            model,
            action_kinds,
            intervention_steps,
        )

    no_progress_count = 0
    previous_potential = gravity_potential(node)
    while True:
        absolute_step = path.shape[1] - 1
        if absolute_step >= config.max_path_steps:
            reason = "max_path_steps"
            break
        if compute_steps >= budget:
            reason = "budget_exhausted"
            break

        compute_steps += 1
        result = deterministic_one_step(node, absolute_step, model, config)
        if result.node is None:
            reason = result.terminal_reason or "optimizer_failed"
            break

        node = result.node
        path = append_state(path, node)
        actions.append(result.action.copy())
        action_kinds.append("gravity")
        if result.collided:
            reason = "collision"
            break

        potential = gravity_potential(node)
        if potential <= config.flatness_tolerance:
            reason = "flat"
            break
        improvement = previous_potential - potential
        if improvement <= config.no_progress_tolerance:
            no_progress_count += 1
        else:
            no_progress_count = 0
        previous_potential = potential
        if no_progress_count >= config.no_progress_patience:
            reason = "no_progress"
            break

    return make_trajectory(
        path,
        actions,
        source,
        reason,
        restart_step,
        prefix_intervention_count,
        compute_steps,
        model,
        action_kinds,
        intervention_steps,
    )


def rollout_mode_perturbation(
    prefix_path: Array,
    prefix_actions: Sequence[Array],
    coefficients: Array,
    model: Model,
    config: ExperimentConfig,
    source: str,
    remaining_compute_steps: int,
    *,
    prefix_action_kinds: Sequence[str] | None = None,
    prefix_intervention_count: int = 0,
    prefix_intervention_steps: Sequence[int] | None = None,
    maximum_norm: float | None = None,
) -> dict[str, Any]:
    """Perturb the final prefix state once, then roll to an endpoint."""

    parent_node = vector_to_node(prefix_path[:, -1], model)
    path, actions, action_kinds = _normalise_prefix(
        parent_node,
        prefix_path,
        prefix_actions,
        prefix_action_kinds,
        model,
    )
    parent_step = path.shape[1] - 1
    intervention_steps = list(prefix_intervention_steps or [])

    initial_reason = _initial_terminal_reason(parent_node, model, config)
    if initial_reason is not None:
        return make_trajectory(
            path,
            actions,
            source,
            initial_reason,
            parent_step,
            prefix_intervention_count,
            0,
            model,
            action_kinds,
            intervention_steps,
        )
    if parent_step >= config.max_path_steps:
        return make_trajectory(
            path,
            actions,
            source,
            "max_path_steps",
            parent_step,
            prefix_intervention_count,
            0,
            model,
            action_kinds,
            intervention_steps,
        )
    if remaining_compute_steps <= 0:
        return make_trajectory(
            path,
            actions,
            source,
            "budget_exhausted",
            parent_step,
            prefix_intervention_count,
            0,
            model,
            action_kinds,
            intervention_steps,
        )

    result = apply_mode_perturbation(
        parent_node,
        coefficients,
        model,
        config,
        maximum_norm=maximum_norm,
    )
    compute_steps = 1
    intervention_count = prefix_intervention_count + 1
    if result.node is None:
        return make_trajectory(
            path,
            actions,
            source,
            result.terminal_reason or "correction_failed",
            parent_step,
            intervention_count,
            compute_steps,
            model,
            action_kinds,
            intervention_steps,
        )

    path = append_state(path, result.node)
    actions.append(result.action.copy())
    action_kinds.append("intervention")
    intervention_steps.append(path.shape[1] - 1)
    if result.collided:
        return make_trajectory(
            path,
            actions,
            source,
            "collision",
            parent_step,
            intervention_count,
            compute_steps,
            model,
            action_kinds,
            intervention_steps,
        )
    if gravity_potential(result.node) <= config.flatness_tolerance:
        return make_trajectory(
            path,
            actions,
            source,
            "flat",
            parent_step,
            intervention_count,
            compute_steps,
            model,
            action_kinds,
            intervention_steps,
        )

    continuation = rollout_from_prefix(
        result.node,
        path,
        actions,
        model,
        config,
        source,
        remaining_compute_steps=max(remaining_compute_steps - 1, 0),
        prefix_action_kinds=action_kinds,
        prefix_intervention_count=intervention_count,
        new_start_step=parent_step,
        prefix_intervention_steps=intervention_steps,
    )
    continuation["compute_steps"] += compute_steps
    return continuation


def rollout_random_perturbation(
    prefix_path: Array,
    prefix_actions: Sequence[Array],
    model: Model,
    config: ExperimentConfig,
    rng: np.random.Generator,
    source: str,
    remaining_compute_steps: int,
    *,
    prefix_action_kinds: Sequence[str] | None = None,
    prefix_intervention_count: int = 0,
    prefix_intervention_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Sample one mode coefficient uniformly in a ball and roll to an endpoint."""

    parent_node = vector_to_node(prefix_path[:, -1], model)
    dimension = mechanism_modes(parent_node, model).shape[1]
    coefficients = sample_uniform_ball(
        dimension, config.perturb_max_norm, rng
    )
    return rollout_mode_perturbation(
        prefix_path,
        prefix_actions,
        coefficients,
        model,
        config,
        source,
        remaining_compute_steps,
        prefix_action_kinds=prefix_action_kinds,
        prefix_intervention_count=prefix_intervention_count,
        prefix_intervention_steps=prefix_intervention_steps,
    )


def _coerce_intervention(value: Intervention | dict[str, Any] | Sequence[Any]) -> Intervention:
    if isinstance(value, Intervention):
        return value
    if isinstance(value, dict):
        wait = value.get("wait_steps", value.get("wait"))
        if wait is None or "latent" not in value:
            raise ValueError("intervention dict needs wait_steps and latent")
        return Intervention(int(wait), np.asarray(value["latent"], dtype=float))
    if len(value) != 2:
        raise ValueError("intervention tuple must be (wait_steps, latent)")
    return Intervention(int(value[0]), np.asarray(value[1], dtype=float))


def rollout_with_interventions(
    start_node: Array,
    interventions: Iterable[
        Intervention | dict[str, Any] | Sequence[Any]
    ],
    model: Model,
    config: ExperimentConfig,
    source: str,
    remaining_compute_steps: int,
    *,
    prefix_path: Array | None = None,
    prefix_actions: Sequence[Array] | None = None,
    prefix_action_kinds: Sequence[str] | None = None,
    prefix_intervention_count: int = 0,
    prefix_intervention_steps: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Evaluate a variable-length sequence of wait-and-perturb interventions."""

    sequence = [_coerce_intervention(value) for value in interventions]
    if len(sequence) > config.max_interventions:
        raise ValueError("intervention sequence exceeds max_interventions")
    if any(item.wait_steps < 0 for item in sequence):
        raise ValueError("wait_steps must be non-negative")

    path, actions, action_kinds = _normalise_prefix(
        start_node,
        prefix_path,
        prefix_actions,
        prefix_action_kinds,
        model,
    )
    restart_step = path.shape[1] - 1
    intervention_count = int(prefix_intervention_count)
    intervention_steps = list(prefix_intervention_steps or [])
    compute_steps = 0
    node = np.asarray(start_node, dtype=float).copy()
    budget = max(int(remaining_compute_steps), 0)
    no_progress_count = 0
    previous_potential = gravity_potential(node)

    reason = _initial_terminal_reason(node, model, config)
    if reason is not None:
        return make_trajectory(
            path,
            actions,
            source,
            reason,
            restart_step,
            intervention_count,
            compute_steps,
            model,
            action_kinds,
            intervention_steps,
        )

    def advance_gravity_once() -> str | None:
        nonlocal path
        nonlocal actions
        nonlocal action_kinds
        nonlocal compute_steps
        nonlocal node
        nonlocal no_progress_count
        nonlocal previous_potential

        if path.shape[1] - 1 >= config.max_path_steps:
            return "max_path_steps"
        if compute_steps >= budget:
            return "budget_exhausted"
        compute_steps += 1
        result = deterministic_one_step(
            node, path.shape[1] - 1, model, config
        )
        if result.node is None:
            return result.terminal_reason or "optimizer_failed"
        node = result.node
        path = append_state(path, node)
        actions.append(result.action.copy())
        action_kinds.append("gravity")
        if result.collided:
            return "collision"
        potential = gravity_potential(node)
        if potential <= config.flatness_tolerance:
            return "flat"
        improvement = previous_potential - potential
        if improvement <= config.no_progress_tolerance:
            no_progress_count += 1
        else:
            no_progress_count = 0
        previous_potential = potential
        if no_progress_count >= config.no_progress_patience:
            return "no_progress"
        return None

    for intervention in sequence:
        for _ in range(intervention.wait_steps):
            reason = advance_gravity_once()
            if reason is not None:
                return make_trajectory(
                    path,
                    actions,
                    source,
                    reason,
                    restart_step,
                    intervention_count,
                    compute_steps,
                    model,
                    action_kinds,
                    intervention_steps,
                )

        if path.shape[1] - 1 >= config.max_path_steps:
            reason = "max_path_steps"
            break
        if compute_steps >= budget:
            reason = "budget_exhausted"
            break
        compute_steps += 1
        result = apply_latent_perturbation(
            node,
            intervention.latent,
            model,
            config,
        )
        intervention_count += 1
        if result.node is None:
            reason = result.terminal_reason or "correction_failed"
            break
        node = result.node
        path = append_state(path, node)
        actions.append(result.action.copy())
        action_kinds.append("intervention")
        intervention_steps.append(path.shape[1] - 1)
        previous_potential = gravity_potential(node)
        no_progress_count = 0
        if result.collided:
            reason = "collision"
            break
        if previous_potential <= config.flatness_tolerance:
            reason = "flat"
            break
    else:
        reason = None

    while reason is None:
        reason = advance_gravity_once()

    return make_trajectory(
        path,
        actions,
        source,
        reason,
        restart_step,
        intervention_count,
        compute_steps,
        model,
        action_kinds,
        intervention_steps,
    )


def rollout_intervention_sequence(
    start_node: Array,
    interventions: Iterable[
        Intervention | dict[str, Any] | Sequence[Any]
    ],
    model: Model,
    config: ExperimentConfig,
    source: str,
    remaining_compute_steps: int,
    **kwargs: Any,
) -> dict[str, Any]:
    """Compatibility alias for :func:`rollout_with_interventions`."""

    return rollout_with_interventions(
        start_node,
        interventions,
        model,
        config,
        source,
        remaining_compute_steps,
        **kwargs,
    )


def generate_bootstrap_paths(
    model: Model,
    config: ExperimentConfig,
    rng: np.random.Generator | None = None,
) -> list[dict[str, Any]]:
    """Generate the common three-path bootstrap from bounded imperfections."""

    generator = (
        np.random.default_rng(config.seed) if rng is None else rng
    )
    root = model.node_initial.copy()
    root_path = start_path(root)
    modes = mechanism_modes(root, model)
    trajectories: list[dict[str, Any]] = []
    for run in range(config.bootstrap_count):
        coefficients = sample_uniform_ball(
            modes.shape[1], config.initial_imperfection, generator
        )
        trajectory = rollout_mode_perturbation(
            root_path,
            [],
            coefficients,
            model,
            config,
            source=f"bootstrap_{run:03d}",
            remaining_compute_steps=config.max_path_steps,
            maximum_norm=config.initial_imperfection,
        )
        trajectories.append(trajectory)
    return trajectories


__all__ = [
    "Intervention",
    "Model",
    "StepResult",
    "append_state",
    "apply_latent_perturbation",
    "apply_mode_perturbation",
    "calc_dedx",
    "calc_mode",
    "clamp_norm",
    "constraint_jacobian",
    "correct_configuration",
    "corrector_increment",
    "cross2",
    "deterministic_one_step",
    "deterministic_state_seed",
    "generate_bootstrap_paths",
    "get_modes",
    "gravity_potential",
    "has_member_collision",
    "incompatibility",
    "load_model",
    "make_trajectory",
    "mechanism_modes",
    "node_to_vector",
    "optimize_one_folding_step",
    "path_nodes",
    "predictor",
    "proper_segment_intersection",
    "rollout_from_prefix",
    "rollout_intervention_sequence",
    "rollout_mode_perturbation",
    "rollout_random_perturbation",
    "rollout_with_interventions",
    "sample_uniform_ball",
    "start_path",
    "vector_to_node",
]
