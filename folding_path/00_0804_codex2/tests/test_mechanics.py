from pathlib import Path
from unittest.mock import patch

import numpy as np

from qd_folding.config import ExperimentConfig
from qd_folding.mechanics import (
    Intervention,
    StepResult,
    constraint_jacobian,
    deterministic_one_step,
    generate_bootstrap_paths,
    get_modes,
    has_member_collision,
    load_model,
    rollout_from_prefix,
    rollout_with_interventions,
    sample_uniform_ball,
    start_path,
    vector_to_node,
)


DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def small_config(**overrides) -> ExperimentConfig:
    values = {
        "max_path_steps": 6,
        "local_cma_generations": 1,
        "local_cma_population": 4,
        "no_progress_patience": 6,
    }
    values.update(overrides)
    return ExperimentConfig(**values)


def test_model_modes_and_uniform_ball() -> None:
    model = load_model(DATA_DIR)
    assert constraint_jacobian(model.node_initial, model).shape == (4, 6)
    assert get_modes(model.node_initial, model).shape == (15, 2)

    rng = np.random.default_rng(3)
    samples = np.asarray(
        [sample_uniform_ball(2, 0.22, rng) for _ in range(1_000)]
    )
    assert np.max(np.linalg.norm(samples, axis=1)) <= 0.22 + 1.0e-12


def test_deterministic_step_is_reproducible() -> None:
    model = load_model(DATA_DIR)
    config = small_config(local_cma_generations=2)
    first = deterministic_one_step(model.node_initial, 0, model, config)
    second = deterministic_one_step(model.node_initial, 0, model, config)

    assert first.node is not None
    assert second.node is not None
    np.testing.assert_array_equal(first.node, second.node)
    np.testing.assert_array_equal(first.action, second.action)


def test_bootstrap_contract_and_budget() -> None:
    model = load_model(DATA_DIR)
    config = small_config(bootstrap_count=3)
    trajectories = generate_bootstrap_paths(
        model, config, np.random.default_rng(config.seed)
    )
    required = {
        "path",
        "actions",
        "source",
        "terminal_reason",
        "is_flat",
        "collided",
        "failed",
        "new_start_step",
        "intervention_count",
        "compute_steps",
        "final_flatness",
    }

    assert len(trajectories) == 3
    for trajectory in trajectories:
        assert required <= set(trajectory)
        assert trajectory["path"].shape[1] - 1 == len(
            trajectory["actions"]
        )
        assert trajectory["compute_steps"] <= config.max_path_steps
        assert trajectory["new_start_step"] == 0
        assert trajectory["intervention_count"] == 1


def test_prefix_length_is_full_but_compute_steps_are_new_only() -> None:
    model = load_model(DATA_DIR)
    config = small_config(max_path_steps=8)
    first = rollout_from_prefix(
        model.node_initial,
        start_path(model.node_initial),
        [],
        model,
        config,
        "first",
        remaining_compute_steps=2,
    )
    prefix_steps = first["path"].shape[1] - 1
    assert first["compute_steps"] == 2

    last = vector_to_node(first["path"][:, -1], model)
    second = rollout_from_prefix(
        last,
        first["path"],
        first["actions"],
        model,
        config,
        "second",
        remaining_compute_steps=1,
        prefix_action_kinds=first["action_kinds"],
    )
    assert second["compute_steps"] == 1
    assert second["path"].shape[1] - 1 == prefix_steps + 1


def test_collision_configuration_is_appended_as_terminal_state() -> None:
    model = load_model(DATA_DIR)
    config = small_config()
    collision_node = np.array(
        [
            [-1.0, 0.0, -1.0],
            [1.0, 0.0, 1.0],
            [-1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [2.0, 0.0, 0.0],
        ]
    )
    assert has_member_collision(
        collision_node, model, config.collision_tolerance
    )
    mocked_result = StepResult(
        node=collision_node,
        action=np.zeros(model.vector_size),
        coefficients=np.zeros(2),
        residual=0.0,
        terminal_reason="collision",
    )

    with patch(
        "qd_folding.mechanics.deterministic_one_step",
        return_value=mocked_result,
    ):
        trajectory = rollout_from_prefix(
            model.node_initial,
            start_path(model.node_initial),
            [],
            model,
            config,
            "collision",
            remaining_compute_steps=1,
        )

    assert trajectory["terminal_reason"] == "collision"
    assert trajectory["collided"]
    assert trajectory["compute_steps"] == 1
    assert trajectory["path"].shape[1] == 2
    np.testing.assert_allclose(
        vector_to_node(trajectory["path"][:, -1], model),
        collision_node,
    )


def test_intervention_sequence_obeys_budget() -> None:
    model = load_model(DATA_DIR)
    config = small_config(max_interventions=2)
    sequence = [
        Intervention(wait_steps=0, latent=np.full(model.vector_size, 0.01)),
        Intervention(wait_steps=1, latent=np.full(model.vector_size, -0.01)),
    ]
    trajectory = rollout_with_interventions(
        model.node_initial,
        sequence,
        model,
        config,
        "sequence",
        remaining_compute_steps=2,
    )
    assert trajectory["compute_steps"] <= 2
    assert trajectory["path"].shape[1] - 1 == len(trajectory["actions"])
