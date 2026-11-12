"""Codex6-compatible folding environment for the five common tasks.

The two deliberate reward revisions are local to this file: self-crossing
proposals are zero-reward rejected no-ops, and the joint milestones are a
hard state-potential difference rather than episode-local one-shot flags.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np

try:  # Supports both package imports and ``python 00_0822_all/main.py``.
    from . import mechanics
except ImportError:  # pragma: no cover - used by the script entry point
    import mechanics


def _setting(config, name, default=None, *aliases):
    """Read one setting from either a mapping or a simple namespace."""

    for key in (name, *aliases):
        if isinstance(config, Mapping) and key in config:
            return config[key]
        if hasattr(config, key):
            return getattr(config, key)
    return default


def _load_data(data, config):
    if data is None:
        data = _setting(
            config, "data", None, "data_dir", "model_dir", "model_directory"
        )
    if isinstance(data, (str, Path)):
        directory = Path(data)
        node = np.loadtxt(
            directory / "node_initial.dat", delimiter=",", dtype=float
        )
        member = np.loadtxt(directory / "member.dat", delimiter=",", dtype=int)
        fix = np.loadtxt(directory / "fix.dat", delimiter=",", dtype=int)
    elif isinstance(data, Mapping):
        node = data.get("node_initial", data.get("node"))
        member, fix = data["member"], data["fix"]
    else:
        node, member, fix = data
    return (
        np.asarray(node, dtype=float).reshape(-1, 3),
        np.asarray(member, dtype=int).reshape(-1, 2),
        np.asarray(fix, dtype=int).reshape(-1, 2),
    )


def endpoint_reward_components(
    flatness_before,
    flatness_after,
    spread_before,
    spread_after,
    target_spread,
    flatness_clip=2.0,
    flatness_weight=1.0,
    target_weight=1.0,
):
    """Return the conservative flatness and squared spread-error increments."""

    bounded_before = float(np.clip(flatness_before, 0.0, flatness_clip))
    bounded_after = float(np.clip(flatness_after, 0.0, flatness_clip))
    error_before = float(spread_before) - float(target_spread)
    error_after = float(spread_after) - float(target_spread)
    flatness_reward = float(
        flatness_weight * (bounded_before - bounded_after)
    )
    target_reward = float(
        target_weight
        * (error_before * error_before - error_after * error_after)
    )
    return {
        "bounded_flatness_before": bounded_before,
        "bounded_flatness_after": bounded_after,
        "target_energy_before": float(error_before * error_before),
        "target_energy_after": float(error_after * error_after),
        "flatness": flatness_reward,
        "target": target_reward,
        "total": float(flatness_reward + target_reward),
    }


class OrigamiEnv:
    """Fixed-start, variable-mode environment shared by all five tasks.

    ``config`` may be a dictionary or namespace. ``data`` may be a model
    directory, a ``(node, member, fix)`` tuple, or a mapping containing those
    arrays. Concise setting names and saved Codex6 ``ori5_*``/``ori6_*`` names
    are both accepted.
    """

    def __init__(self, config, data=None):
        self.config = config
        self.node0, self.member, self.fix = _load_data(data, config)
        self.node = self.node0.copy()
        self.num_node = len(self.node0)
        self.num_member = len(self.member)
        self.coordinate_size = self.node0.size
        self.free = mechanics.index_free(self.num_node, self.fix)
        self.lengths = np.linalg.norm(
            self.node0[self.member[:, 0]] - self.node0[self.member[:, 1]],
            axis=1,
        )
        self.coordinate_scale = max(float(np.max(self.lengths)), 1.0e-8)

        # Mechanical transition.
        self.max_steps = int(_setting(config, "max_steps", 80))
        self.max_step_width = float(_setting(config, "max_step_width", 0.5))
        self.min_step_width = float(
            _setting(config, "min_step_width", 1.0e-4)
        )
        self.projection_shrink = float(
            _setting(config, "projection_shrink", 0.5)
        )
        self.max_correction_ratio = float(
            _setting(config, "max_correction_ratio", 1.0)
        )
        self.max_actual_step_ratio = float(
            _setting(config, "max_actual_step_ratio", 1.25)
        )
        self.strain_tolerance = float(
            _setting(config, "strain_tolerance", 1.0e-8)
        )
        self.crossing_tolerance = float(
            _setting(config, "crossing_tolerance", 1.0e-10)
        )
        self.crossing_penalty = 0.0
        self.crossing_policy = "reject_candidate_as_zero_reward_noop"

        # Goal geometry.
        self.flatness_subspace_dimension = int(
            _setting(config, "flatness_subspace_dimension", 1)
        )
        self.evaluation_flatness_tolerance = float(
            _setting(config, "flatness_tolerance", 1.0e-3)
        )
        self.active_flatness_tolerance = self.evaluation_flatness_tolerance
        self.training_flatness_tolerances = tuple(float(value) for value in
            _setting(
                config,
                "training_flatness_tolerances",
                (self.evaluation_flatness_tolerance,),
            )
        )
        self.training_flatness_milestones = tuple(int(value) for value in
            _setting(config, "training_flatness_milestones", ())
        )
        self.training_episode_count = 0
        self.target_spread = float(_setting(config, "target_spread", 2.0))
        self.spread_tolerance = float(
            _setting(config, "spread_tolerance", 0.05)
        )

        # Endpoint potentials.
        self.flatness_clip = float(_setting(
            config,
            "endpoint_flatness_clip",
            2.0,
            "flatness_clip",
            "ori5_flatness_clip",
        ))
        self.flatness_weight = float(_setting(
            config,
            "endpoint_flatness_weight",
            1.0,
            "flatness_progress_weight",
        ))
        self.target_weight = float(_setting(
            config,
            "endpoint_target_weight",
            1.0,
            "target_progress_weight",
        ))

        # Continuous spread-frontier state potential Phi_D.
        self.frontier_bonus = float(_setting(
            config,
            "frontier_bonus",
            0.9,
            "spread_frontier_bonus",
            "ori6_spread_frontier_bonus",
        ))
        self.frontier_mode = str(_setting(
            config,
            "frontier_mode",
            "state_potential",
            "ori6_spread_frontier_mode",
        ))
        self.frontier_flatness_cap = float(_setting(
            config,
            "frontier_flatness_cap",
            2.0,
            "ori6_spread_frontier_flatness_cap",
        ))
        self.frontier_initial_factor = float(_setting(
            config,
            "frontier_initial_factor",
            2.0,
            "spread_frontier_initial_factor",
            "ori5_spread_milestone_initial_factor",
        ))

        # Hard joint state potential Phi_J. There are deliberately no flags.
        self.joint_flatness_thresholds = tuple(float(value) for value in
            _setting(
                config,
                "joint_flatness_thresholds",
                (0.5, 0.1, 0.01),
                "ori5_joint_flatness_thresholds",
            )
        )
        self.joint_bonuses = tuple(float(value) for value in _setting(
            config,
            "joint_bonuses",
            (0.2, 0.3, 0.5),
            "joint_potential_values",
            "ori5_joint_milestone_bonuses",
        ))

        # Reset-referenced spectral-subspace potential.
        self.spectral_weight = float(_setting(
            config, "spectral_weight", 0.05, "ori6_spectral_weight"
        ))
        self.spectral_relative_epsilon = float(_setting(
            config,
            "spectral_relative_epsilon",
            1.0e-12,
            "ori6_spectral_relative_epsilon",
        ))

        # Fixed and graded terminal terms.
        self.numerical_failure_penalty = abs(float(_setting(
            config, "numerical_failure_penalty", 10.0
        )))
        self.no_mode_penalty = abs(float(
            _setting(config, "no_mode_penalty", 5.0)
        ))
        self.wrong_flat_penalty = abs(float(_setting(
            config,
            "wrong_flat_penalty",
            2.0,
            "ori5_wrong_flat_penalty",
        )))
        self.timeout_penalty = abs(float(_setting(
            config,
            "terminal_timeout_penalty",
            1.0,
            "ori5_timeout_penalty",
            "timeout_penalty",
        )))
        self.success_bonus = float(_setting(
            config,
            "terminal_success_bonus",
            10.0,
            "ori5_success_bonus",
            "success_bonus",
        ))
        self.quality_weight = float(_setting(
            config,
            "terminal_quality_weight",
            1.0,
            "quality_weight",
            "ori5_quality_weight",
        ))
        self.speed_weight = float(_setting(
            config,
            "terminal_speed_weight",
            0.5,
            "success_speed_bonus_weight",
            "ori5_success_speed_bonus_weight",
        ))
        self.near_miss_flatness_far = float(
            _setting(config, "near_miss_flatness_far", 1.0)
        )
        self.near_miss_spread_error_far = float(
            _setting(config, "near_miss_spread_error_far", 1.0)
        )

        self.initial_flatness_potential = mechanics.compute_flatness_potential(
            self.node0, self.flatness_subspace_dimension
        )
        if self.initial_flatness_potential <= 1.0e-12:
            raise ValueError("The initial flatness potential must be positive.")

        self.crossing_pairs = mechanics.nonadjacent_bar_pairs(self.member)
        self.spatial_collision = bool(_setting(
            config,
            "spatial_collision",
            self.flatness_subspace_dimension == 2,
            "spatial_crossing",
        ))
        self.collision_checker = (
            mechanics.ContinuousBarCollisionChecker(
                self.member,
                distance_tolerance=max(self.crossing_tolerance, 0.0),
                pair_indices=self.crossing_pairs,
            )
            if self.spatial_collision else None
        )
        self._mode_cache_key = None
        self._mode_cache = None
        self.reset()

    def configure_episode_tolerance(self, training=True):
        """Select the Codex6 curriculum tolerance (strict by default)."""

        if not training:
            self.active_flatness_tolerance = self.evaluation_flatness_tolerance
            return
        self.training_episode_count += 1
        phase = sum(
            self.training_episode_count > milestone
            for milestone in self.training_flatness_milestones
        )
        self.active_flatness_tolerance = self.training_flatness_tolerances[phase]

    @staticmethod
    def _node_key(node):
        value = np.ascontiguousarray(np.asarray(node, dtype="<f8"))
        return value.shape, value.tobytes(order="C")

    def reset(self):
        self.node = self.node0.copy()
        self.step_count = 0
        self._mode_cache_key = None
        self._mode_cache = None

        initial = self.compactness_metrics()
        self.initial_spread_error = float(initial["spread_error"])
        self.frontier_eligible = bool(
            self.initial_spread_error
            > self.frontier_initial_factor * self.spread_tolerance
        )
        self.current_frontier_potential = float(
            self.spread_frontier_components(initial)["potential"]
        )
        self.current_joint_potential = float(self.joint_potential(initial))

        reference = mechanics.spectral_subspace_geometry(
            self.node,
            self.flatness_subspace_dimension,
            reference_projector=None,
            relative_epsilon=self.spectral_relative_epsilon,
        )
        self.spectral_reference_projector = np.asarray(
            reference["projector"], dtype=float
        ).copy()
        self.current_spectral_potential = 0.0

        self.cumulative_flatness_reward = 0.0
        self.cumulative_target_reward = 0.0
        self.cumulative_frontier_reward = 0.0
        self.cumulative_joint_reward = 0.0
        self.cumulative_spectral_reward = 0.0
        self.reward_trace = []
        return self.get_state()

    def get_state(self):
        """Coordinates only, normalised by the fixed initial member scale."""

        return (self.node / self.coordinate_scale).flatten().astype(np.float32)

    def get_modes(self):
        key = self._node_key(self.node)
        if key != self._mode_cache_key:
            jacobian = mechanics.dedx(self.node, self.member, self.free)
            self._mode_cache = mechanics.mode(
                self.free, jacobian, self.num_node
            )
            self._mode_cache_key = key
        return np.asarray(self._mode_cache, dtype=float).copy()

    @property
    def num_mode(self):
        return int(self.get_modes().shape[1])

    def flatness_metrics(self, node=None):
        coordinates = self.node if node is None else np.asarray(node, dtype=float)
        return mechanics.flatness_metrics(
            coordinates,
            self.flatness_subspace_dimension,
            self.initial_flatness_potential,
            self.coordinate_scale,
        )

    def compactness_metrics(self, node=None):
        coordinates = self.node if node is None else np.asarray(node, dtype=float)
        potential, ratio, rms = self.flatness_metrics(coordinates)
        spread = mechanics.projected_spread(
            coordinates,
            self.flatness_subspace_dimension,
            self.coordinate_scale,
        )
        spread_deviation = float(spread - self.target_spread)
        return {
            "flatness_potential": float(potential),
            "flatness_ratio": float(ratio),
            "flatness_rms": float(rms),
            "spread": float(spread),
            "spread_deviation": spread_deviation,
            "spread_error": abs(spread_deviation),
            "target_deviation_energy": float(spread_deviation * spread_deviation),
        }

    def near_miss_quality(self, flatness_ratio, spread_error):
        denominator = np.log(
            self.near_miss_flatness_far / self.evaluation_flatness_tolerance
        )
        q_flatness = np.log(
            self.near_miss_flatness_far
            / max(float(flatness_ratio), self.evaluation_flatness_tolerance)
        ) / denominator
        q_flatness = float(np.clip(q_flatness, 0.0, 1.0))
        q_spread = (
            self.near_miss_spread_error_far - abs(float(spread_error))
        ) / (
            self.near_miss_spread_error_far - self.spread_tolerance
        )
        q_spread = float(np.clip(q_spread, 0.0, 1.0))
        return q_flatness, q_spread, float(q_flatness * q_spread)

    def spread_frontier_components(self, metrics):
        if not getattr(self, "frontier_eligible", False):
            progress = 0.0
        else:
            denominator = self.initial_spread_error - self.spread_tolerance
            progress = (
                self.initial_spread_error - float(metrics["spread_error"])
            ) / denominator
            progress = float(np.clip(progress, 0.0, 1.0))

        if self.frontier_mode == "state_potential":
            gate = 1.0
        elif self.frontier_mode == "flatness_gated_state_potential":
            gate = (
                self.frontier_flatness_cap - float(metrics["flatness_ratio"])
            ) / (
                self.frontier_flatness_cap
                - self.evaluation_flatness_tolerance
            )
            gate = float(np.clip(gate, 0.0, 1.0))
        else:
            raise ValueError(f"Unknown frontier mode: {self.frontier_mode}")
        return {
            "progress": progress,
            "flatness_gate": gate,
            "potential": float(self.frontier_bonus * progress * gate),
        }

    def joint_potential(self, metrics):
        """Hard, flag-free joint potential Phi_J(s)."""

        if float(metrics["spread_error"]) > self.spread_tolerance:
            return 0.0
        flatness = float(metrics["flatness_ratio"])
        return float(sum(
            bonus
            for threshold, bonus in zip(
                self.joint_flatness_thresholds, self.joint_bonuses
            )
            if flatness <= threshold
        ))

    def spectral_subspace_metrics(self, node=None):
        coordinates = self.node if node is None else np.asarray(node, dtype=float)
        return mechanics.spectral_subspace_geometry(
            coordinates,
            self.flatness_subspace_dimension,
            reference_projector=self.spectral_reference_projector,
            relative_epsilon=self.spectral_relative_epsilon,
        )

    def _project_candidate(self, direction, requested_width):
        return mechanics.project_candidate(
            self.node,
            direction,
            requested_width,
            self.member,
            self.lengths,
            self.free,
            max_step_width=self.max_step_width,
            min_step_width=self.min_step_width,
            projection_shrink=self.projection_shrink,
            strain_tolerance=self.strain_tolerance,
            max_correction_ratio=self.max_correction_ratio,
            max_actual_step_ratio=self.max_actual_step_ratio,
        )

    def _has_crossing(self, before, after):
        return bool(mechanics.has_proper_self_crossing(
            before,
            after,
            self.member,
            self.crossing_pairs,
            self.crossing_tolerance,
            spatial=self.spatial_collision,
            checker=self.collision_checker,
        ))

    def _crossing_info(
        self,
        metrics,
        requested_width,
        projection_attempts,
        mode_coefficients,
    ):
        timeout = self.step_count >= self.max_steps
        qF, qD, quality = self.near_miss_quality(
            metrics["flatness_ratio"], metrics["spread_error"]
        )
        fixed = -self.timeout_penalty if timeout else 0.0
        graded = self.quality_weight * quality if timeout else 0.0
        reward = float(fixed + graded) if timeout else 0.0
        info = dict(metrics)
        info.update({
            "success": False,
            "strict_success": False,
            "strict_flat": bool(
                metrics["flatness_ratio"] <= self.evaluation_flatness_tolerance
            ),
            "wrong_flat": False,
            "self_crossing": False,
            "self_crossing_candidate": True,
            "rejected_crossing": True,
            "correction_failed": False,
            "no_modes": False,
            "timeout": timeout,
            "termination_reason": "timeout" if timeout else "running",
            "after_state_valid": 0,
            "requested_width": float(requested_width),
            "accepted_width": 0.0,
            "projection_attempts": int(projection_attempts),
            "correction_norm": 0.0,
            "actual_displacement": 0.0,
            "mode_count": int(len(mode_coefficients)),
            "mode_coefficients": np.asarray(mode_coefficients, dtype=float).copy(),
            "reward_flatness": 0.0,
            "reward_target": 0.0,
            "reward_frontier": 0.0,
            "reward_joint": 0.0,
            "reward_spectral": 0.0,
            "reward_terminal_fixed": fixed,
            "reward_terminal_quality": graded,
            "reward_terminal_speed": 0.0,
            "reward_flatness_rollback": 0.0,
            "reward_frontier_rollback": 0.0,
            "reward_joint_rollback": 0.0,
            "reward_spectral_rollback": 0.0,
            "reward_total": reward,
            "quality_flatness": qF if timeout else 0.0,
            "quality_spread": qD if timeout else 0.0,
            "quality": quality if timeout else 0.0,
            "terminal_flatness_ratio": float(metrics["flatness_ratio"]),
        })
        self.reward_trace.append(info.copy())
        return self.get_state(), reward, timeout, info

    def step(self, physical_direction, requested_width, mode_coefficients):
        node_before = self.node.copy()
        metrics_before = self.compactness_metrics(node_before)
        frontier_before = self.spread_frontier_components(metrics_before)
        joint_before = self.joint_potential(metrics_before)
        spectral_before = self.spectral_subspace_metrics(node_before)

        (
            candidate,
            accepted_width,
            correction_failed,
            projection_attempts,
            strain,
            correction_norm,
            actual_displacement,
        ) = self._project_candidate(physical_direction, requested_width)
        self.step_count += 1

        crossing = bool(
            not correction_failed
            and abs(accepted_width) > 0.0
            and self._has_crossing(node_before, candidate)
        )
        if crossing:
            # Candidate is never committed. All potential ledgers stay
            # untouched, so an ordinary rejection has exactly zero reward.
            return self._crossing_info(
                metrics_before,
                requested_width,
                projection_attempts,
                mode_coefficients,
            )

        if not correction_failed:
            self.node = candidate
            self._mode_cache_key = None
            self._mode_cache = None

        metrics_after = self.compactness_metrics()
        transition_valid = not correction_failed
        accepted_valid = bool(
            transition_valid and abs(float(accepted_width)) > 0.0
        )

        if transition_valid:
            endpoint = endpoint_reward_components(
                metrics_before["flatness_ratio"],
                metrics_after["flatness_ratio"],
                metrics_before["spread"],
                metrics_after["spread"],
                self.target_spread,
                self.flatness_clip,
                self.flatness_weight,
                self.target_weight,
            )
            frontier_after = self.spread_frontier_components(metrics_after)
            joint_after = self.joint_potential(metrics_after)
            frontier_reward = float(
                frontier_after["potential"] - frontier_before["potential"]
            )
            joint_reward = float(joint_after - joint_before)
        else:
            endpoint = endpoint_reward_components(
                metrics_before["flatness_ratio"],
                metrics_before["flatness_ratio"],
                metrics_before["spread"],
                metrics_before["spread"],
                self.target_spread,
                self.flatness_clip,
                self.flatness_weight,
                self.target_weight,
            )
            frontier_after = dict(frontier_before)
            joint_after = joint_before
            frontier_reward = joint_reward = 0.0

        if accepted_valid:
            spectral_after = self.spectral_subspace_metrics(self.node)
            spectral_progress = float(
                self.spectral_weight
                * (spectral_after["potential"] - spectral_before["potential"])
            )
        else:
            spectral_after = spectral_before
            spectral_progress = 0.0

        flatness_reward = float(endpoint["flatness"])
        target_reward = float(endpoint["target"])
        self.cumulative_flatness_reward += flatness_reward
        self.cumulative_target_reward += target_reward
        self.cumulative_frontier_reward += frontier_reward
        self.cumulative_joint_reward += joint_reward
        self.cumulative_spectral_reward += spectral_progress
        self.current_frontier_potential = float(frontier_after["potential"])
        self.current_joint_potential = float(joint_after)
        self.current_spectral_potential = float(spectral_after["potential"])

        flat_reached = bool(
            transition_valid
            and metrics_after["flatness_ratio"] <= self.active_flatness_tolerance
        )
        success = bool(
            flat_reached
            and metrics_after["spread_error"] <= self.spread_tolerance
        )
        wrong_flat = bool(flat_reached and not success)
        no_modes = bool(
            transition_valid and not flat_reached and self.get_modes().shape[1] == 0
        )

        if correction_failed:
            reason = "correction_failed"
        elif success:
            reason = "success"
        elif wrong_flat:
            reason = "wrong_flat"
        elif no_modes:
            reason = "no_modes"
        elif self.step_count >= self.max_steps:
            reason = "timeout"
        else:
            reason = "running"

        qF_after, qD_after, quality_after = self.near_miss_quality(
            metrics_after["flatness_ratio"], metrics_after["spread_error"]
        )
        terminal_fixed = terminal_quality = terminal_speed = 0.0
        rollback_flatness = rollback_frontier = rollback_joint = 0.0
        rollback_spectral = 0.0
        qF_used, qD_used, quality_used = qF_after, qD_after, quality_after

        if reason == "correction_failed":
            terminal_fixed = -self.numerical_failure_penalty
            rollback_frontier = -self.cumulative_frontier_reward
            rollback_joint = -self.cumulative_joint_reward
            rollback_spectral = -self.cumulative_spectral_reward
            qF_used = qD_used = quality_used = 0.0
        elif reason == "success":
            terminal_fixed = self.success_bonus
            terminal_quality = self.quality_weight * quality_after
            terminal_speed = self.speed_weight * (
                1.0 - self.step_count / self.max_steps
            )
        elif reason == "wrong_flat":
            terminal_fixed = -self.wrong_flat_penalty
            terminal_quality = self.quality_weight * qD_after
            rollback_flatness = -self.cumulative_flatness_reward
            rollback_frontier = -self.cumulative_frontier_reward
            rollback_joint = -self.cumulative_joint_reward
            rollback_spectral = -self.cumulative_spectral_reward
            qF_used, quality_used = 0.0, qD_after
        elif reason == "no_modes":
            terminal_fixed = -self.no_mode_penalty
            rollback_frontier = -self.cumulative_frontier_reward
            rollback_spectral = -self.cumulative_spectral_reward
            qF_used = qD_used = quality_used = 0.0
        elif reason == "timeout":
            terminal_fixed = -self.timeout_penalty
            terminal_quality = self.quality_weight * quality_after

        if reason == "correction_failed":
            reward = float(
                terminal_fixed
                + rollback_frontier
                + rollback_joint
                + rollback_spectral
            )
        else:
            reward = float(
                flatness_reward
                + target_reward
                + frontier_reward
                + joint_reward
                + spectral_progress
                + terminal_fixed
                + terminal_quality
                + terminal_speed
                + rollback_flatness
                + rollback_frontier
                + rollback_joint
                + rollback_spectral
            )

        strict_flat = bool(
            metrics_after["flatness_ratio"]
            <= self.evaluation_flatness_tolerance
        )
        strict_success = bool(
            strict_flat and metrics_after["spread_error"] <= self.spread_tolerance
        )
        info = dict(metrics_after)
        info.update({
            "success": success,
            "strict_success": strict_success,
            "strict_flat": strict_flat,
            "wrong_flat": wrong_flat,
            "self_crossing": False,
            "self_crossing_candidate": False,
            "rejected_crossing": False,
            "correction_failed": correction_failed,
            "no_modes": no_modes,
            "timeout": reason == "timeout",
            "termination_reason": reason,
            "after_state_valid": int(transition_valid),
            "requested_width": float(requested_width),
            "accepted_width": float(accepted_width),
            "projection_attempts": int(projection_attempts),
            "strain": float(strain),
            "correction_norm": float(correction_norm),
            "actual_displacement": float(actual_displacement),
            "correction_to_width_ratio": float(
                correction_norm / max(abs(accepted_width), self.min_step_width)
            ),
            "actual_to_width_ratio": float(
                actual_displacement / max(abs(accepted_width), self.min_step_width)
            ),
            "mode_count": int(len(mode_coefficients)),
            "mode_coefficients": np.asarray(mode_coefficients, dtype=float).copy(),
            "frontier_progress": float(frontier_after["progress"]),
            "frontier_potential": float(frontier_after["potential"]),
            "joint_potential": float(joint_after),
            "spectral_rho": float(spectral_after["rho"]),
            "spectral_delta": float(spectral_after["delta"]),
            "spectral_potential": float(spectral_after["potential"]),
            "reward_flatness": flatness_reward,
            "reward_target": target_reward,
            "reward_frontier": frontier_reward,
            "reward_joint": joint_reward,
            "reward_spectral": spectral_progress,
            "reward_terminal_fixed": float(terminal_fixed),
            "reward_terminal_quality": float(terminal_quality),
            "reward_terminal_speed": float(terminal_speed),
            "reward_flatness_rollback": float(rollback_flatness),
            "reward_frontier_rollback": float(rollback_frontier),
            "reward_joint_rollback": float(rollback_joint),
            "reward_spectral_rollback": float(rollback_spectral),
            "reward_total": reward,
            "quality_flatness": float(qF_used),
            "quality_spread": float(qD_used),
            "quality": float(quality_used),
            "terminal_flatness_ratio": float(metrics_after["flatness_ratio"]),
            "flatness_progress_reward": flatness_reward,
            "target_progress_reward": target_reward,
            "terminal_reward_component": float(
                terminal_fixed
                + terminal_quality
                + terminal_speed
                + rollback_flatness
                + rollback_frontier
                + rollback_joint
                + rollback_spectral
            ),
        })
        self.reward_trace.append(info.copy())
        return self.get_state(), reward, reason != "running", info


__all__ = ["OrigamiEnv", "endpoint_reward_components"]
