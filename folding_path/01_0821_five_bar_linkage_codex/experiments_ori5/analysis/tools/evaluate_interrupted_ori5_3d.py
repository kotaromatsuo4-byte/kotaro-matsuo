"""Evaluate ori5 policies saved at an interrupted 3-D checkpoint.

This is deliberately a derived-artifact utility.  It reads a canonical run,
refuses to overwrite an existing output directory, and writes only CSV/JSON
files below the requested derived analysis directory.  It neither resumes
training nor writes into the canonical run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY_DIRECTORY = SCRIPT_PATH.parents[3]
if str(REPOSITORY_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_DIRECTORY))

import ori_codex2 as mechanics  # noqa: E402
import ori_codex5_3d as adapter  # noqa: E402


SCHEMA_VERSION = 1
EXPECTED_REWARD_DESIGN = "ori5_endpoint_potential_fixed_milestones"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    try:
        name = str(path.relative_to(root.resolve())) if root else str(path)
    except ValueError:
        name = str(path)
    stat = path.stat()
    return {
        "path": name,
        "size_bytes": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write an empty evaluation: {path.name}")
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def load_checkpoint(path: Path) -> dict[str, Any]:
    # These are trusted, locally generated training artifacts.  weights_only
    # cannot reconstruct the saved NumPy RNG tuple in current PyTorch.
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or "model_state_dict" not in value:
        raise ValueError(f"Invalid ori5 checkpoint: {path}")
    return value


def expected_evaluation_episodes(checkpoint_episode: int, interval: int) -> list[int]:
    result = [1]
    result.extend(range(interval, checkpoint_episode + 1, interval))
    return sorted(set(result))


def validate_partial_run(run_directory: Path) -> dict[str, Any]:
    required = {
        name: run_directory / name
        for name in (
            "run_config.json",
            "training_history.csv",
            "update_history.csv",
            "deterministic_evaluation_history.csv",
            "stochastic_evaluation_history.csv",
            "checkpoint.pt",
            "best_policy.pt",
            "best_policy_metrics.json",
        )
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing partial-run artifacts: {missing}")

    config = json.loads(required["run_config.json"].read_text(encoding="utf-8"))
    checkpoint = load_checkpoint(required["checkpoint.pt"])
    best = load_checkpoint(required["best_policy.pt"])
    best_metrics = json.loads(
        required["best_policy_metrics.json"].read_text(encoding="utf-8")
    )
    checkpoint_episode = int(checkpoint["episode"])
    batch_size = int(config["batch_size"])
    eval_interval = int(config["eval_interval"])

    if config.get("selected_model") != "4unit_torus":
        raise ValueError("This audit expects selected_model=4unit_torus.")
    if config.get("reward_design") != EXPECTED_REWARD_DESIGN:
        raise ValueError("Unexpected reward design in run_config.json.")
    if int(config["episodes"]) <= checkpoint_episode:
        raise ValueError("The checkpoint does not represent an interrupted run.")
    if checkpoint_episode % int(config["checkpoint_every"]):
        raise ValueError("Interrupted artifact is not on an atomic checkpoint boundary.")
    if checkpoint.get("config") != config:
        raise ValueError("checkpoint config and run_config.json are not identical.")

    history = read_csv(required["training_history.csv"])
    updates = read_csv(required["update_history.csv"])
    deterministic = read_csv(required["deterministic_evaluation_history.csv"])
    stochastic = read_csv(required["stochastic_evaluation_history.csv"])

    history_episodes = [int(float(row["episode"])) for row in history]
    if history_episodes != list(range(1, checkpoint_episode + 1)):
        raise ValueError("training_history episode axis is not exactly 1..checkpoint.")
    update_episodes = [int(float(row["episode"])) for row in updates]
    expected_updates = list(range(batch_size, checkpoint_episode + 1, batch_size))
    if update_episodes != expected_updates:
        raise ValueError("update_history does not contain every completed batch.")
    expected_evaluations = expected_evaluation_episodes(
        checkpoint_episode, eval_interval
    )
    for name, rows in (("deterministic", deterministic), ("stochastic", stochastic)):
        episodes = [int(float(row["episode"])) for row in rows]
        if episodes != expected_evaluations:
            raise ValueError(f"{name} evaluation history has an unexpected episode axis.")

    expected_stochastic_count = int(config["periodic_stochastic_eval_episodes"])
    if any(
        int(float(row["evaluation_episodes"])) != expected_stochastic_count
        for row in stochastic
    ):
        raise ValueError("Periodic stochastic evaluation count changed within the run.")

    nonfinite_rewards = 0
    maximum_episode_error = 0.0
    maximum_step_error = 0.0
    nonfinite_components = 0
    for row in history:
        for field in (
            "total_reward",
            "ori5_reward_decomposition_error",
            "ori5_maximum_abs_step_reward_decomposition_error",
        ):
            if not math.isfinite(float(row[field])):
                nonfinite_rewards += 1
        maximum_episode_error = max(
            maximum_episode_error,
            abs(float(row["ori5_reward_decomposition_error"])),
        )
        maximum_step_error = max(
            maximum_step_error,
            abs(float(row["ori5_maximum_abs_step_reward_decomposition_error"])),
        )
        nonfinite_components += int(float(row["ori5_nonfinite_reward_component_count"]))
    if nonfinite_rewards or nonfinite_components:
        raise ValueError("Non-finite ori5 reward components were recorded.")
    if maximum_episode_error > 1.0e-10 or maximum_step_error > 1.0e-10:
        raise ValueError("ori5 reward decomposition identity failed.")

    best_episode = int(best["episode"])
    if best_episode != int(best_metrics["episode"]):
        raise ValueError("best_policy episode and metrics episode differ.")
    if best.get("evaluation_metrics") != best_metrics.get("selection_evaluation"):
        raise ValueError("best_policy saved evaluation does not match its JSON metrics.")
    if list(best.get("evaluation_score")) != list(best_metrics.get("selection_score")):
        raise ValueError("best_policy saved score does not match its JSON metrics.")

    absent_final_artifacts = [
        name for name in (
            "policy.pt",
            "evaluation_summary.csv",
            "best_policy_evaluation_summary.csv",
            "summary.json",
        )
        if not (run_directory / name).exists()
    ]
    return {
        "status": "valid_partial_checkpoint",
        "checkpoint_episode": checkpoint_episode,
        "planned_episodes": int(config["episodes"]),
        "training_history_rows": len(history),
        "update_history_rows": len(updates),
        "deterministic_evaluation_history_rows": len(deterministic),
        "stochastic_evaluation_history_rows": len(stochastic),
        "evaluation_history_episodes": expected_evaluations,
        "best_policy_episode": best_episode,
        "maximum_abs_reward_decomposition_error": maximum_episode_error,
        "maximum_abs_step_reward_decomposition_error": maximum_step_error,
        "nonfinite_reward_component_count": nonfinite_components,
        "expected_absent_final_artifacts": absent_final_artifacts,
    }


def build_agent(args: SimpleNamespace, checkpoint: dict[str, Any]):
    model = mechanics.VariableModeActorCritic(
        coordinate_size=int(checkpoint["config"]["coordinate_size"]),
        hidden_size=int(args.hidden_size),
        variance_bias=float(args.variance_bias),
        width_mean_limit=float(args.width_mean_limit),
    )
    agent = mechanics.ActorCriticAgent(
        model=model,
        max_step_width=float(args.max_step_width),
        gamma=float(args.gamma),
        actor_learning_rate=float(args.actor_learning_rate),
        critic_learning_rate=float(args.critic_learning_rate),
        entropy_coefficient=float(args.entropy_coefficient),
        value_coefficient=float(args.value_coefficient),
        actor_gradient_clip=float(args.actor_gradient_clip),
        critic_gradient_clip=float(args.critic_gradient_clip),
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return agent


def evaluate_checkpoint(
    args: SimpleNamespace,
    node: np.ndarray,
    member: np.ndarray,
    fix: np.ndarray,
    checkpoint: dict[str, Any],
    episodes: int,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    environment = adapter.make_environment(args, node, member, fix)
    agent = build_agent(args, checkpoint)
    evaluations = []
    for evaluation_index in range(episodes):
        case_seed = int(seed) + evaluation_index
        torch.manual_seed(case_seed)
        np.random.seed(case_seed)
        with torch.no_grad():
            summary, _, _ = adapter.run_episode(
                environment, agent, deterministic=False, training=False
            )
        summary["evaluation_episode"] = evaluation_index
        summary["evaluation_seed"] = case_seed
        evaluations.append(summary)
    aggregate = adapter.aggregate_stochastic_evaluations(
        evaluations, episode=int(checkpoint["episode"])
    )
    return evaluations, aggregate


def metrics_close(actual: Any, expected: Any, path: str = "") -> list[str]:
    errors: list[str] = []
    if isinstance(expected, dict):
        for key, value in expected.items():
            child = f"{path}.{key}" if path else key
            if key not in actual:
                errors.append(f"{child}: missing")
            else:
                errors.extend(metrics_close(actual[key], value, child))
    elif expected is None:
        if actual is not None:
            errors.append(f"{path}: {actual!r} != None")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=2e-12):
            errors.append(f"{path}: {actual!r} != {expected!r}")
    elif actual != expected:
        errors.append(f"{path}: {actual!r} != {expected!r}")
    return errors


def aggregate_replay_comparison(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    """Describe stochastic replay drift without hiding discrete outcomes.

    A long nonlinear rollout can differ at round-off level across processes
    even under identical explicit seeds.  Discrete terminal outcomes are the
    acceptance-critical fingerprint; continuous aggregate drift is retained
    explicitly instead of being silently rounded away.
    """

    discrete_fields = (
        "episode",
        "evaluation_episodes",
        "success_rate",
        "self_crossing_rate",
        "mean_steps",
        "flat_reached_rate",
        "wrong_flat_rate",
        "strict_compact_success_rate",
        "relaxed_only_success_rate",
        "curriculum_waypoint_rate",
        "maximum_curriculum_waypoint_bonus_count",
        "ever_flat_reached_rate",
        "ever_wrong_flat_rate",
        "near_miss_terminal_eligible_rate",
    )
    discrete_differences = {
        field: {"replay": actual.get(field), "saved": expected.get(field)}
        for field in discrete_fields
        if actual.get(field) != expected.get(field)
    }
    continuous_deltas = {}
    for field, saved in expected.items():
        replay = actual.get(field)
        if (
            field in discrete_fields
            or replay is None
            or saved is None
            or isinstance(replay, bool)
            or isinstance(saved, bool)
            or not isinstance(replay, (int, float))
            or not isinstance(saved, (int, float))
        ):
            continue
        continuous_deltas[field] = float(replay) - float(saved)
    return {
        "discrete_terminal_outcomes_matched": not discrete_differences,
        "discrete_differences": discrete_differences,
        "maximum_absolute_continuous_aggregate_delta": float(max(
            [abs(value) for value in continuous_deltas.values()] or [0.0]
        )),
        "continuous_aggregate_deltas_replay_minus_saved": continuous_deltas,
        "saved_aggregate": expected,
        "replayed_aggregate": actual,
        "interpretation": (
            "The explicit seeds and discrete terminal outcomes reproduce. "
            "Small continuous drift is reported as cross-process floating-"
            "point replay drift, not treated as a new learning result."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--evaluation-episodes", type=int, default=100)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument(
        "--skip-best-selection-replay", action="store_true",
        help="Skip the 50-case replay that fingerprints the evaluator.",
    )
    options = parser.parse_args(argv)
    if options.evaluation_episodes <= 0:
        parser.error("--evaluation-episodes must be positive")

    run_directory = options.run_dir.resolve()
    output_directory = options.output_dir.resolve()
    if output_directory.exists():
        parser.error("--output-dir must not exist; derived artifacts are immutable")
    partial_validation = validate_partial_run(run_directory)
    config = json.loads((run_directory / "run_config.json").read_text(encoding="utf-8"))
    evaluation_seed = (
        int(config["final_evaluation_seed"])
        if options.evaluation_seed is None
        else int(options.evaluation_seed)
    )
    if evaluation_seed != int(config["final_evaluation_seed"]):
        parser.error("Evaluation seed must equal the recorded final_evaluation_seed")

    canonical_names = (
        "run_config.json",
        "training_history.csv",
        "update_history.csv",
        "deterministic_evaluation_history.csv",
        "stochastic_evaluation_history.csv",
        "checkpoint.pt",
        "best_policy.pt",
        "best_policy_metrics.json",
    )
    canonical_before = {
        name: file_record(run_directory / name, run_directory)
        for name in canonical_names
    }

    checkpoint = load_checkpoint(run_directory / "checkpoint.pt")
    best_checkpoint = load_checkpoint(run_directory / "best_policy.pt")
    args = SimpleNamespace(**config)
    node, member, fix, model_files = adapter.load_problem_data(args.data_dir)

    torch.set_num_threads(1)
    with adapter.patched_three_dimensional_runtime(args, member):
        best_selection_replay = None
        if not options.skip_best_selection_replay:
            replay_rows, replay_aggregate = evaluate_checkpoint(
                args, node, member, fix, best_checkpoint,
                int(config["periodic_stochastic_eval_episodes"]),
                int(config["selection_evaluation_seed"]),
            )
            expected = json.loads(
                (run_directory / "best_policy_metrics.json").read_text(encoding="utf-8")
            )["selection_evaluation"]
            replay_comparison = aggregate_replay_comparison(
                replay_aggregate, expected
            )
            if not replay_comparison["discrete_terminal_outcomes_matched"]:
                raise RuntimeError(
                    "Best-policy selection replay changed discrete terminal outcomes: "
                    + repr(replay_comparison["discrete_differences"])
                )
            best_selection_replay = {
                "evaluation_episodes": len(replay_rows),
                "evaluation_seed": int(config["selection_evaluation_seed"]),
                **replay_comparison,
            }

        checkpoint_rows, checkpoint_aggregate = evaluate_checkpoint(
            args, node, member, fix, checkpoint,
            options.evaluation_episodes, evaluation_seed,
        )
        best_rows, best_aggregate = evaluate_checkpoint(
            args, node, member, fix, best_checkpoint,
            options.evaluation_episodes, evaluation_seed,
        )

    canonical_after = {
        name: file_record(run_directory / name, run_directory)
        for name in canonical_names
    }
    if canonical_after != canonical_before:
        raise RuntimeError("A canonical run artifact changed during derived evaluation.")

    runtime_paths = [
        REPOSITORY_DIRECTORY / name
        for name in (
            "ori_codex2.py",
            "ori_codex3.py",
            "ori_codex4.py",
            "ori_codex4_3d.py",
            "ori_codex5.py",
            "ori_codex5_3d.py",
            "collision_3d.py",
        )
    ]
    model_file_records = {
        key: file_record(Path(value), Path(args.data_dir))
        for key, value in model_files.items()
    }

    output_directory.mkdir(parents=True, exist_ok=False)
    write_csv(output_directory / "checkpoint_ep8192_evaluation.csv", checkpoint_rows)
    write_csv(output_directory / "best_policy_ep1000_evaluation.csv", best_rows)

    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "user_stopped_at_checkpoint",
        "checkpoint_episode": int(checkpoint["episode"]),
        "last_checkpoint_held_out_evaluation": checkpoint_aggregate,
        "best_policy_episode": int(best_checkpoint["episode"]),
        "best_policy_held_out_evaluation": best_aggregate,
        "evaluation_protocol": {
            "evaluation_episodes": int(options.evaluation_episodes),
            "first_seed": evaluation_seed,
            "last_seed": evaluation_seed + int(options.evaluation_episodes) - 1,
            "seed_rule": "case_seed = recorded final_evaluation_seed + evaluation_index",
            "deterministic": False,
            "training": False,
            "common_random_numbers_between_policies": True,
            "collision_backend": config["three_dimensional_adapter"]["collision_backend"],
            "collision_backend_version": config["three_dimensional_adapter"]["collision_backend_version"],
        },
        "best_selection_replay": best_selection_replay,
        "partial_artifact_validation": partial_validation,
        "provenance": {
            "canonical_run_directory": str(run_directory),
            "canonical_files_unchanged_during_evaluation": True,
            "canonical_files": canonical_before,
            "runtime_files": {
                path.name: file_record(path, REPOSITORY_DIRECTORY)
                for path in runtime_paths
            },
            "model_files": model_file_records,
            "evaluator": file_record(SCRIPT_PATH, REPOSITORY_DIRECTORY),
            "python_version": sys.version,
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
    }
    write_json(output_directory / "evaluation_summary.json", result)
    output_records = {
        path.name: file_record(path, output_directory)
        for path in sorted(output_directory.iterdir())
    }
    write_json(output_directory / "output_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "files": output_records,
    })
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Derived results: {output_directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
