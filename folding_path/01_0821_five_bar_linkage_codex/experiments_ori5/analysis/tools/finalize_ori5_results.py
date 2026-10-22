"""Build the audited, user-facing ori_codex5 final result bundle."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys


SCRIPT_PATH = Path(__file__).resolve()
REPOSITORY = SCRIPT_PATH.parents[3]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

import analyze_ori_codex5_experiments as analyzer  # noqa: E402
import ori_codex5_first_discovery as discovery  # noqa: E402
import run_ori_codex5_experiments as runner  # noqa: E402


ANALYSIS = REPOSITORY / "experiments_ori5" / "analysis"
REWARD_SHA256 = discovery.FIXED_REWARD_SHA256
FULL_MANIFEST = (
    REPOSITORY / "experiments_ori5" / "manifests"
    / "full_seed2101_e7df0e12f833.json"
)
CALIBRATION_ROOT = REPOSITORY / "w5" / "u5"
REPLAYS = {
    "five_d2": {
        "root": REPOSITORY / "w5" / "d2",
        "full": REPOSITORY / "experiments_ori5" / "full"
        / "five_d2" / "B" / "seed2101",
        "problem": "five_bar",
        "initial_model_sha256": (
            "817710797be360cf866fb14883e4d7035db6adcab96caedab456f07bec832fe1"
        ),
        "actions_sha256": (
            "3dfbadb8af4291bb02e6ca481970053d014c507c861dd864848ed48a67266b45"
        ),
    },
    "five_d3": {
        "root": REPOSITORY / "w5" / "d3",
        "full": REPOSITORY / "experiments_ori5" / "full"
        / "five_d3" / "B" / "seed2101",
        "problem": "five_bar",
        "initial_model_sha256": (
            "817710797be360cf866fb14883e4d7035db6adcab96caedab456f07bec832fe1"
        ),
        "actions_sha256": (
            "923407e0e77bb3fa09cbf5d126678ac088423844e355d1278526db643f61721f"
        ),
    },
    "watersqrt2": {
        "root": REPOSITORY / "w5" / "ws",
        "full": REPOSITORY / "experiments_ori5" / "full"
        / "watersqrt2" / "B" / "seed2101",
        "problem": "waterbomb",
        "initial_model_sha256": (
            "119f906f52478ca09ac77b666a21bb463c3a714399317a6e0d1a07f74810013f"
        ),
        "actions_sha256": (
            "6d11ebe6bef93f6a7e1eaebf86f9108ccc2ae7066798c3ed55b4b810b065ef41"
        ),
    },
}

EPISODE_FIELD_MAP = {
    "episode": "episode",
    "mechanical_steps": "steps",
    "projector_calls": "projection_attempts",
    "strict_compact_success": "strict_compact_success",
    "termination_reason": "termination_reason",
    "total_reward": "total_reward",
    "flatness_ratio": "flatness_ratio",
    "spread": "spread",
    "spread_error": "spread_error",
    "self_crossing": "self_crossing",
    "correction_failed": "correction_failed",
    "timeout": "timeout",
    "near_miss_q": "near_miss_q",
    "near_miss_bonus": "near_miss_bonus",
    "ori5_total_flatness_progress_reward": (
        "ori5_total_flatness_progress_reward"
    ),
    "ori5_total_target_progress_reward": "ori5_total_target_progress_reward",
    "ori5_total_spread_milestone_reward": (
        "ori5_total_spread_milestone_reward"
    ),
    "ori5_total_joint_milestone_reward": (
        "ori5_total_joint_milestone_reward"
    ),
    "ori5_total_milestone_reward": "ori5_total_milestone_reward",
    "ori5_total_terminal_fixed_reward": "ori5_total_terminal_fixed_reward",
    "ori5_total_terminal_quality_reward": (
        "ori5_total_terminal_quality_reward"
    ),
    "ori5_total_terminal_speed_reward": "ori5_total_terminal_speed_reward",
    "ori5_total_terminal_correction_reward": (
        "ori5_total_terminal_correction_reward"
    ),
    "ori5_reward_decomposition_error": "ori5_reward_decomposition_error",
    "ori5_nonfinite_reward_component_count": (
        "ori5_nonfinite_reward_component_count"
    ),
    "ori5_last_valid_flatness_ratio": "ori5_last_valid_flatness_ratio",
    "ori5_last_valid_spread": "ori5_last_valid_spread",
}
UPDATE_FIELDS = (
    "episode", "loss", "actor_loss", "critic_loss", "entropy",
    "actor_gradient_norm", "critic_gradient_norm", "actor_learning_rate",
    "critic_learning_rate",
)


def load_json(path: Path):
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def numeric_equal(left, right, tolerance=1.0e-13) -> bool:
    try:
        return math.isclose(
            float(left), float(right), rel_tol=0.0, abs_tol=tolerance
        )
    except (TypeError, ValueError):
        return left == right


def file_hash_map(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): sha256(path)
        for path in directory.rglob("*") if path.is_file()
    }


def validate_replay(task: str, specification: dict) -> dict:
    trial = specification["root"] / "trials" / "learning" / "seed2101"
    summary = load_json(trial / "summary.json")
    root_config = load_json(specification["root"] / "run_config.json")
    aggregate = discovery.aggregate_results(
        specification["root"], require_complete=True
    )
    replay_rows = load_csv(trial / "episode_history.csv")
    full_rows = load_csv(specification["full"] / "training_history.csv")
    expected_rows = full_rows[:len(replay_rows)]
    mismatches = []
    cumulative_mechanics = 0
    cumulative_projectors = 0
    for index, (replay, full) in enumerate(
        zip(replay_rows, expected_rows), start=1
    ):
        for replay_field, full_field in EPISODE_FIELD_MAP.items():
            if not numeric_equal(replay[replay_field], full[full_field]):
                mismatches.append({
                    "episode": index, "field": replay_field,
                    "replay": replay[replay_field], "full": full[full_field],
                })
        cumulative_mechanics += int(float(full["steps"]))
        cumulative_projectors += int(float(full["projection_attempts"]))
        if int(replay["cumulative_mechanical_steps"]) != cumulative_mechanics:
            mismatches.append({
                "episode": index, "field": "cumulative_mechanical_steps"
            })
        if int(replay["cumulative_projector_calls"]) != cumulative_projectors:
            mismatches.append({
                "episode": index, "field": "cumulative_projector_calls"
            })

    replay_updates = load_csv(trial / "update_history.csv")
    full_updates = load_csv(specification["full"] / "update_history.csv")
    update_mismatches = []
    for index, (replay, full) in enumerate(
        zip(replay_updates, full_updates[:len(replay_updates)]), start=1
    ):
        for field in UPDATE_FIELDS:
            if not numeric_equal(replay[field], full[field]):
                update_mismatches.append({
                    "update": index, "field": field,
                    "replay": replay[field], "full": full[field],
                })

    replay_path = file_hash_map(trial / "first_hit_path")
    full_path = file_hash_map(specification["full"] / "first_success_path")
    path_equal = replay_path == full_path
    actions_sha256 = replay_path.get("actions.json")
    checks = {
        "aggregate_complete": aggregate.get("complete") is True,
        "summary_complete": summary.get("complete") is True,
        "success": summary.get("success") == 1,
        "not_censored": summary.get("censored") == 0,
        "reward_identity": (
            summary.get("reward_constants_sha256") == REWARD_SHA256
            and root_config.get("reward_constants_sha256") == REWARD_SHA256
        ),
        "problem_identity": (
            summary.get("benchmark_problem") == specification["problem"]
        ),
        "initial_model_identity": (
            summary.get("initial_model_sha256")
            == specification["initial_model_sha256"]
        ),
        "episode_prefix_equal": not mismatches,
        "update_prefix_equal": not update_mismatches,
        "first_hit_path_byte_equal": path_equal,
        "actions_identity": actions_sha256 == specification["actions_sha256"],
        "nonfinite_reward_components_zero": (
            summary.get("reward_audit", {}).get(
                "nonfinite_reward_component_count"
            ) == 0
        ),
    }
    return {
        "task": task,
        "exact_with_respect_to_all_saved_invariants": all(checks.values()),
        "checks": checks,
        "episode_rows_compared": len(replay_rows),
        "episode_mismatch_count": len(mismatches),
        "episode_mismatches_first10": mismatches[:10],
        "update_rows_compared": len(replay_updates),
        "update_mismatch_count": len(update_mismatches),
        "update_mismatches_first10": update_mismatches[:10],
        "first_hit_path_file_count": len(replay_path),
        "actions_sha256": actions_sha256,
        "summary": summary,
        "raw_root": str(specification["root"].relative_to(REPOSITORY)),
    }


def validate_calibration() -> tuple[dict, dict[str, dict]]:
    result = load_json(CALIBRATION_ROOT / "calibration_results.json")
    blocks = load_csv(CALIBRATION_ROOT / "timing_blocks.csv")
    task_rows = load_csv(CALIBRATION_ROOT / "task_summary.csv")
    identity_keys = (
        "schema_version", "calibrator_version", "protocol_content_sha256",
        "protocol_file_sha256", "baseline_file_sha256",
        "reward_constants_sha256", "selected_tasks", "task_identity",
        "source_sha256",
    )
    identity = {key: result[key] for key in identity_keys}
    finite_blocks = all(
        math.isfinite(float(row["rollout_wall_seconds"]))
        and math.isfinite(float(row["mechanical_steps_per_wall_second"]))
        and int(float(row["nonfinite_reward_component_count"])) == 0
        and abs(float(row["maximum_abs_reward_decomposition_error"]))
        <= 1.0e-8
        for row in blocks
    )
    checks = {
        "complete": result.get("complete") is True,
        "all_five_tasks": result.get("selected_tasks") == [
            "five_d2", "five_d3", "torus2", "water1", "watersqrt2"
        ],
        "all_protocol_tasks_selected": (
            result.get("all_protocol_tasks_selected") is True
        ),
        "single_process_protocol": result.get("measurement_definition", {}).get(
            "single_process"
        ) is True,
        "not_direct_random_first_discovery_wall": (
            result.get("measurement_definition", {}).get(
                "direct_uniform_first_discovery_wall_measurement"
            ) is False
        ),
        "not_direct_rl_wall": result.get("measurement_definition", {}).get(
            "direct_rl_wall_measurement"
        ) is False,
        "reward_identity": result.get("reward_constants_sha256") == REWARD_SHA256,
        "protocol_identity": result.get("protocol_file_sha256") == (
            "cfb64192cab10cde6c70ea629bc97cca1a0cda007112be792a07acc4d82fec21"
        ),
        "fingerprint_recomputed": (
            result.get("run_fingerprint") == discovery._canonical_digest(identity)
        ),
        "timing_blocks_90": len(blocks) == 90,
        "task_rows_5": len(task_rows) == 5,
        "measured_mechanics_368640": sum(
            int(float(row["mechanical_steps"])) for row in blocks
        ) == 368_640,
        "finite_numeric_audit": finite_blocks,
    }
    return ({
        "valid": all(checks.values()),
        "checks": checks,
        "run_fingerprint": result.get("run_fingerprint"),
        "totals": result.get("totals"),
        "raw_root": str(CALIBRATION_ROOT.relative_to(REPOSITORY)),
    }, {row["task"]: row for row in task_rows})


def optional_number(row: dict, key: str):
    value = row.get(key)
    return None if value in (None, "") else float(value)


def jsonable_number(value):
    if value is None:
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def write_markdown(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8", newline="\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def plot_calibrated_wall_comparison(
    replays: dict[str, dict], calibration: dict[str, dict],
) -> list[dict]:
    """Plot direct RL replay time against calibrated random wall-equivalent RMST."""

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    tasks = ("five_d2", "five_d3", "watersqrt2")
    labels = ("five D=2", "five D=3", "water D=sqrt(2)")
    annotations = (
        "random/RL = 6.95x",
        "0 policy updates; not learning-attributable",
        "RL is 5.35x slower",
    )
    rows = []
    for task in tasks:
        direct = float(replays[task]["summary"]["compute_wall_seconds"])
        random_wall = float(
            calibration[task]["calibrated_random_wall_rmst_seconds"]
        )
        lower = float(
            calibration[task][
                "calibrated_random_wall_rmst_ci_lower_seconds"
            ]
        )
        upper = float(
            calibration[task][
                "calibrated_random_wall_rmst_ci_upper_seconds"
            ]
        )
        rows.append({
            "task": task,
            "direct_rl_rollout_plus_update_seconds": direct,
            "calibrated_random_wall_rmst_seconds": random_wall,
            "calibrated_random_wall_rmst_ci_lower_seconds": lower,
            "calibrated_random_wall_rmst_ci_upper_seconds": upper,
            "ratio_calibrated_random_over_direct_rl": random_wall / direct,
            "interpretation": (
                "not_learning_attributable_zero_updates"
                if task == "five_d3" else
                "rl_slower_in_wall" if task == "watersqrt2" else
                "rl_faster_in_wall"
            ),
        })

    x = np.arange(len(tasks), dtype=float)
    width = 0.34
    direct_values = np.asarray([
        row["direct_rl_rollout_plus_update_seconds"] for row in rows
    ])
    random_values = np.asarray([
        row["calibrated_random_wall_rmst_seconds"] for row in rows
    ])
    lower_errors = random_values - np.asarray([
        row["calibrated_random_wall_rmst_ci_lower_seconds"] for row in rows
    ])
    upper_errors = np.asarray([
        row["calibrated_random_wall_rmst_ci_upper_seconds"] for row in rows
    ]) - random_values

    figure, axis = plt.subplots(figsize=(10.8, 6.4))
    direct_bars = axis.bar(
        x - width / 2.0, direct_values, width,
        color="tab:green", label="ori5 direct replay: rollout + update",
    )
    random_bars = axis.bar(
        x + width / 2.0, random_values, width,
        color="tab:blue", alpha=0.82,
        yerr=np.vstack((lower_errors, upper_errors)), capsize=5,
        label="Uniform random: calibrated wall-equivalent RMST (95% CI)",
    )
    axis.set_yscale("log")
    axis.set_ylabel("Seconds (log scale)")
    axis.set_xticks(x, labels)
    axis.grid(axis="y", which="both", alpha=0.23)
    figure.suptitle(
        "First strict-flat discovery: direct ori5 replay vs calibrated random RMST",
        y=0.975, fontsize=15,
    )
    figure.text(
        0.5, 0.925,
        "Descriptive comparison; random bars are calibrated, not direct wall trials",
        ha="center", va="center", fontsize=9.5,
    )
    figure.legend(
        handles=(direct_bars, random_bars),
        labels=(
            "ori5 direct replay: rollout + update",
            "Uniform random: calibrated wall-equivalent RMST (95% CI)",
        ),
        loc="upper center", bbox_to_anchor=(0.5, 0.89), ncol=2,
        fontsize=8.5,
    )
    for bars in (direct_bars, random_bars):
        for bar in bars:
            axis.annotate(
                f"{bar.get_height():,.2f} s",
                (bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
                xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=8,
            )
    note_x = (0.20, 0.50, 0.80)
    for position, annotation in zip(note_x, annotations):
        figure.text(
            position, 0.025, annotation, ha="center", va="bottom",
            fontsize=8.3, fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.25", "facecolor": "white",
                "edgecolor": "0.75", "alpha": 0.95,
            },
        )
    figure.subplots_adjust(top=0.81, bottom=0.17, left=0.08, right=0.985)
    figure.savefig(
        ANALYSIS / "final_ori5_calibrated_wall_comparison.png", dpi=180,
    )
    plt.close(figure)
    analyzer.write_csv(
        ANALYSIS / "final_ori5_calibrated_wall_comparison.csv", rows,
    )
    runner.atomic_json(
        ANALYSIS / "final_ori5_calibrated_wall_comparison.json",
        {
            "schema_version": 1,
            "measurement_note": (
                "RL is one exact seed-2101 replay. Random is a current-machine "
                "calibrated wall-equivalent common-censor RMST, not a direct "
                "first-discovery wall measurement. Ratios are descriptive."
            ),
            "tasks": rows,
        },
    )
    return rows


def main() -> int:
    analysis_report = load_json(ANALYSIS / "full_analysis_report.json")
    run_rows = load_csv(ANALYSIS / "full_run_summary.csv")
    run_by_task = {row["task"]: row for row in run_rows}
    random_audit = load_json(ANALYSIS / "random_baseline_compatibility.json")
    manifest = load_json(FULL_MANIFEST)
    replays = {
        task: validate_replay(task, specification)
        for task, specification in REPLAYS.items()
    }
    calibration_audit, calibration = validate_calibration()
    wall_comparison = plot_calibrated_wall_comparison(replays, calibration)

    png_names = [
        *(f"full_{task}_profiles.png" for task in runner.TASKS),
        "full_random_first_hit_comparison.png",
        *(f"full_{task}_random_ecdf_survival.png" for task in runner.TASKS),
        "final_ori5_calibrated_wall_comparison.png",
    ]
    from PIL import Image
    png_audit = []
    for name in png_names:
        path = ANALYSIS / name
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            png_audit.append({
                "path": name, "sha256": sha256(path),
                "size_bytes": path.stat().st_size,
                "width": image.width, "height": image.height,
                "readable": True,
            })

    task_records = []
    task_order = list(runner.TASKS)
    for task in task_order:
        row = run_by_task[task]
        baseline = random_audit["tasks"][task]["random_summary"]
        first_episode = optional_number(row, "first_success_episode")
        first_mechanics = optional_number(row, "first_success_mechanical_steps")
        random_events = baseline.get("first_success_mechanical_steps", [])
        random_before = (
            sum(float(value) < first_mechanics for value in random_events)
            if first_mechanics is not None else None
        )
        replay = replays.get(task)
        direct_rl_wall = (
            float(replay["summary"]["compute_wall_seconds"])
            if replay else None
        )
        calibrated_wall = float(
            calibration[task]["calibrated_random_wall_rmst_seconds"]
        )
        wall_ratio = (
            calibrated_wall / direct_rl_wall if direct_rl_wall else None
        )
        updates = optional_number(row, "policy_updates_before_first_success")
        if task == "five_d2":
            conclusion = (
                "supports_learning_speed_stability_and_spread_control_with_"
                "residual_crossing"
            )
        elif task == "five_d3":
            conclusion = (
                "supports_stability_and_spread_control_but_first_hit_not_"
                "attributable_to_learning_zero_updates"
            )
        elif task == "watersqrt2":
            conclusion = (
                "supports_stability_and_spread_control_but_first_hit_"
                "advantage_is_weak_in_mechanics_and_loses_in_wall_time"
            )
        elif task == "torus2":
            conclusion = "failed_flatness_near_miss_dead_zone_user_censored_ep8192"
        else:
            conclusion = "failed_self_crossing_attractor"
        task_records.append({
            "task": task,
            "profile": row["profile"],
            "artifact_status": row["artifact_status"],
            "training_episodes": int(float(row["episodes"])),
            "training_total_mechanical_steps": int(float(
                row["training_total_mechanical_steps"]
            )),
            "first_success_episode": jsonable_number(first_episode),
            "first_success_mechanical_steps": jsonable_number(first_mechanics),
            "policy_updates_before_first_success": jsonable_number(updates),
            "training_success_rate": float(row["training_success_rate"]),
            "tail_success_rate": float(row["tail_success_rate"]),
            "tail_crossing_rate": float(row["tail_crossing_rate"]),
            "heldout_success_rate": float(row["heldout_strict_success_rate"]),
            "heldout_crossing_rate": float(row["heldout_crossing_rate"]),
            "heldout_wrong_flat_rate": float(row["heldout_wrong_flat_rate"]),
            "heldout_residual_rate": float(
                row["heldout_timeout_or_other_residual_rate"]
            ),
            "heldout_mean_flatness_ratio": float(
                row["heldout_mean_flatness_ratio"]
            ),
            "heldout_mean_spread_error": float(
                row["heldout_mean_spread_error"]
            ),
            "random_search_count": int(baseline["search_count"]),
            "random_success_rate": float(baseline["success_rate"]),
            "random_median_first_success_mechanical_steps": baseline.get(
                "median_first_success_mechanical_steps"
            ),
            "random_rmst_episode": float(
                baseline["restricted_mean_first_success_episode"]
            ),
            "random_rmst_mechanical_steps": float(
                baseline["restricted_mean_first_success_mechanical_steps"]
            ),
            "mechanics_rmst_ratio_random_over_rl": optional_number(
                row, "rl_first_hit_mechanical_rmst_speedup"
            ),
            "random_successes_before_rl_first_hit": random_before,
            "random_successes_total": len(random_events),
            "direct_rl_rollout_plus_update_seconds": direct_rl_wall,
            "calibrated_random_wall_rmst_seconds": calibrated_wall,
            "calibrated_random_wall_rmst_ci_lower_seconds": float(
                calibration[task][
                    "calibrated_random_wall_rmst_ci_lower_seconds"
                ]
            ),
            "calibrated_random_wall_rmst_ci_upper_seconds": float(
                calibration[task][
                    "calibrated_random_wall_rmst_ci_upper_seconds"
                ]
            ),
            "wall_ratio_calibrated_random_over_direct_rl": wall_ratio,
            "exact_wall_replay_saved_invariants": (
                replay["exact_with_respect_to_all_saved_invariants"]
                if replay else None
            ),
            "conclusion": conclusion,
        })

    global_checks = {
        "analysis_endpoint_protocol_valid": (
            analysis_report.get("accepted_endpoint_collection") is True
        ),
        "four_natural_completions_one_user_checkpoint": (
            analysis_report.get("strict_natural_completions") == 4
            and analysis_report.get("accepted_user_stopped_checkpoints") == 1
        ),
        "reward_invariant": analysis_report.get("reward_invariant") is True,
        "random_baselines_compatible": (
            analysis_report.get("random_baselines_compatible") is True
        ),
        "full_source_identity_unchanged": (
            manifest.get("source_sha256") == runner.source_hashes()
        ),
        "exact_success_replays": all(
            replay["exact_with_respect_to_all_saved_invariants"]
            for replay in replays.values()
        ),
        "uniform_calibration_valid": calibration_audit["valid"],
        "all_final_pngs_readable": all(
            row["readable"] and row["size_bytes"] > 10_000
            for row in png_audit
        ),
    }
    payload = {
        "schema_version": 1,
        "reward_constants_sha256": REWARD_SHA256,
        "global_valid": all(global_checks.values()),
        "global_checks": global_checks,
        "interpretation": {
            "direct_rl_wall": (
                "Single seed-2101 exact replay; rollout plus Actor-Critic "
                "update time, excluding initialization and artifact I/O."
            ),
            "random_wall": (
                "Current-machine calibrated wall-equivalent RMST from fixed "
                "uniform prefixes; not a direct random first-discovery wall run."
            ),
            "wall_ratio": (
                "Descriptive only: calibrated multi-seed random RMST divided "
                "by one exact learning-seed replay time."
            ),
        },
        "torus_endpoint": {
            "status": "user_stopped_at_checkpoint",
            "primary_episode": 8192,
            "exploratory_stdout_last_episode_ignored": 8800,
            "derived_evaluation": str(
                analyzer.USER_STOPPED_DERIVED_EVALUATION.relative_to(REPOSITORY)
            ),
        },
        "tasks": task_records,
        "wall_replay_validation": replays,
        "uniform_calibration_validation": calibration_audit,
        "calibrated_wall_comparison": wall_comparison,
        "png_audit": png_audit,
        "failed_max_path_attempts_excluded": [
            "experiments_ori5/analysis/walltime_calibration/"
            "rl_first_discovery_single_process/five_d3",
            "experiments_ori5/first_discovery_replays/"
            "five_d3_B_seed2101_budget1033",
        ],
    }
    runner.atomic_json(ANALYSIS / "final_ori5_comparison.json", payload)
    analyzer.write_csv(ANALYSIS / "final_ori5_comparison.csv", task_records)
    runner.atomic_json(
        ANALYSIS / "final_ori5_wall_replay_validation.json",
        {
            "schema_version": 1,
            "all_exact": global_checks["exact_success_replays"],
            "replays": replays,
            "uniform_calibration": calibration_audit,
        },
    )
    runner.atomic_json(
        ANALYSIS / "final_ori5_artifact_qa.json",
        {
            "schema_version": 1, "valid": all(global_checks.values()),
            "checks": global_checks, "pngs": png_audit,
        },
    )

    by_task = {row["task"]: row for row in task_records}
    d2, d3 = by_task["five_d2"], by_task["five_d3"]
    torus, water1 = by_task["torus2"], by_task["water1"]
    water2 = by_task["watersqrt2"]
    table_lines = [
        "| 課題 | endpoint | 初回hit (episode / mechanics / updates) | "
        "random RMST mechanics比 | final 100 success / crossing | spread error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in task_records:
        hit = (
            f"{row['first_success_episode']} / "
            f"{row['first_success_mechanical_steps']:,} / "
            f"{row['policy_updates_before_first_success']}"
            if row["first_success_episode"] is not None else "未発見"
        )
        ratio = (
            f"{row['mechanics_rmst_ratio_random_over_rl']:.3f}x"
            if row["mechanics_rmst_ratio_random_over_rl"] is not None else "—"
        )
        table_lines.append(
            f"| {row['task']} | {row['artifact_status']} | {hit} | {ratio} | "
            f"{row['heldout_success_rate']:.3f} / "
            f"{row['heldout_crossing_rate']:.3f} | "
            f"{row['heldout_mean_spread_error']:.6g} |"
        )

    markdown = f"""# ori_codex5 最終検証結果

同一の報酬定数を固定し、five-bar 2課題、4unit torus、waterbomb 2課題を検証した。4課題は10,000 episodeを自然完了した。torus はユーザー停止のため、最後の原子的 checkpoint である episode 8,192 のみを一次データとし、stdoutだけに残った episode 8,193–8,800 は集計から除外した。

## 結果

{chr(10).join(table_lines)}

- **five D=2**: 初回hitは73 policy updates後で、random RMSTに対して mechanics 15.90倍、直接RL wall 1,176.80秒に対して calibrated random wall RMST 8,174.40秒（記述比6.95倍）だった。final 100の成功率は0.88、spread errorは{d2['heldout_mean_spread_error']:.3g}。ただし crossingが0.12残る。
- **five D=3**: final 100は成功率1.00で安定した。一方、初回hitはepisode 3、**policy update 0回**であり、初回発見の速さを学習効果とは帰属できない。これは未更新NNの確率的方策による初期発見である。
- **waterbomb D=√2**: 初回hit後に学習が進み、final 100成功率0.93、crossing 0、spread error {water2['heldout_mean_spread_error']:.3g}を得た。ただし mechanics上のrandom RMST比は1.047倍にすぎず、random中央値10,718.5 stepsはRLの17,324 stepsより早い。random 10試行中6試行がRLより先に発見した。さらに直接RL wall 110.83秒に対し calibrated random wall RMSTは20.71秒で、wall上はRLが約5.35倍遅い。
- **torus D=2**: 8,192 episodeがすべてtimeoutで、success/crossing/wrong-flatは0。全episodeで qF=q=0、joint milestone 0、spread-only milestoneは3/8,192だけだった。final-seed 100評価も全timeout（平均 F={torus['heldout_mean_flatness_ratio']:.6g}, spread error={torus['heldout_mean_spread_error']:.6g}）。flatness near-miss閾値へ入れないdead zoneが原因であり、この報酬形態の汎用性を支持しない。
- **waterbomb D=1**: success 0、last 1,000 crossing率0.999、final 100 crossing率1.00で、自己交差attractorへ収束した。near-miss自体は初期に少数存在したが、安定した回避方向を与えられなかった。

したがって、同一報酬形態はfive-bar 2課題とwaterbomb D=√2で「学習後の安定折りたたみ・spread制御」を示したが、「全構造で初回発見をrandomより高速化する」ことは示せない。D=2 five-barのみ、更新後の初回hitとmechanics/wall双方の明確な優位を確認できた。D=3 five-barの初回hitは学習前、waterbomb D=√2はwallでrandomに負け、torusとwaterbomb D=1は未成功である。

## Wall-time の定義

RL値はseed 2101を同一configで再実行し、full runのepisode/update/成功pathと全保存項目が一致した場合のみ採用した `rollout_wall_seconds + update_wall_seconds` である。random値は、同じidle machineで固定uniform prefixを測定し、保存済みcommon-censor mechanics RMSTを換算した calibrated wall-equivalentである。randomの直接first-discovery wall測定ではなく、比は単一RL seedと複数random seed RMSTの記述比較である。

## 図

- [five D=2 learning curve](full_five_d2_profiles.png)
- [five D=3 learning curve](full_five_d3_profiles.png)
- [torus D=2 learning curve（episode 8192でcensor）](full_torus2_profiles.png)
- [waterbomb D=1 learning curve](full_water1_profiles.png)
- [waterbomb D=√2 learning curve](full_watersqrt2_profiles.png)
- [全5課題の初回発見 mechanics 比較](full_random_first_hit_comparison.png)
- episode/mechanics ECDF: [five D=2](full_five_d2_random_ecdf_survival.png), [five D=3](full_five_d3_random_ecdf_survival.png), [torus](full_torus2_random_ecdf_survival.png), [water D=1](full_water1_random_ecdf_survival.png), [water D=√2](full_watersqrt2_random_ecdf_survival.png)

## 機械可読データ

- [最終比較CSV](final_ori5_comparison.csv)
- [最終比較JSON](final_ori5_comparison.json)
- [wall replay検証](final_ori5_wall_replay_validation.json)
- [artifact QA](final_ori5_artifact_qa.json)
- [torus派生評価](derived_torus_user_stopped_ep8192/evaluation_summary.json)
"""
    report_table_lines = [
        "| \u6761\u4ef6 | endpoint | \u521d\u56dehit: episode / mechanics / updates | "
        "random/RL mechanics RMST\u6bd4 | final 100: success / crossing | spread error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in task_records:
        report_hit = (
            f"{row['first_success_episode']} / "
            f"{row['first_success_mechanical_steps']:,} / "
            f"{row['policy_updates_before_first_success']}"
            if row["first_success_episode"] is not None
            else "\u672a\u767a\u898b"
        )
        report_ratio = (
            f"{row['mechanics_rmst_ratio_random_over_rl']:.3f}x"
            if row["mechanics_rmst_ratio_random_over_rl"] is not None
            else "--"
        )
        report_table_lines.append(
            f"| {row['task']} | {row['artifact_status']} | {report_hit} | "
            f"{report_ratio} | {row['heldout_success_rate']:.3f} / "
            f"{row['heldout_crossing_rate']:.3f} | "
            f"{row['heldout_mean_spread_error']:.6g} |"
        )

    markdown = f"""# ori_codex5 \u6700\u7d42\u691c\u8a3c\u7d50\u679c

\u5171\u901a\u306e\u5831\u916c\u5b9a\u6570\u3092\u56fa\u5b9a\u3057\u3001optimizer\u7b49\u306e\u30cf\u30a4\u30d1\u30fc\u30d1\u30e9\u30e1\u30fc\u30bf\u306e\u307f\u3092\u8abf\u6574\u3057\u30665\u6761\u4ef6\u3092\u691c\u8a3c\u3057\u305f\u3002five-bar D=2/D=3\u3001waterbomb D=1/D=sqrt(2)\u306f10,000 episode\u3092\u5b8c\u8d70\u3057\u305f\u3002torus D=2\u306f\u30e6\u30fc\u30b6\u30fc\u505c\u6b62\u306e\u305f\u3081\u3001\u6700\u5f8c\u306eatomic checkpoint\u3067\u3042\u308bepisode 8,192\u3092\u6b63\u5f0f\u306a\u6253\u5207\u308a\u70b9\u3068\u3057\u3001stdout\u3060\u3051\u306b\u6b8b\u3063\u305fepisode 8,193\u20138,800\u306f\u96c6\u8a08\u304b\u3089\u9664\u5916\u3057\u305f\u3002

## \u7d50\u679c\u4e00\u89a7

{chr(10).join(report_table_lines)}

## \u6761\u4ef6\u5225\u306e\u89e3\u91c8

- **five-bar D=2**: episode 585\u3001233,824 mechanics step\u300173 policy updates\u5f8c\u306b\u521d\u56de\u6210\u529f\u3057\u305f\u3002random mechanics RMST\u6bd4\u306f15.898\u500d\u3001\u76f4\u63a5\u8a08\u6642\u3057\u305fRL\u306f{d2['direct_rl_rollout_plus_update_seconds']:.2f}\u79d2\u3001\u6821\u6b63random RMST\u306f{d2['calibrated_random_wall_rmst_seconds']:.2f}\u79d2\uff08\u8a18\u8ff0\u6bd4{d2['wall_ratio_calibrated_random_over_direct_rl']:.2f}\u500d\uff09\u3002final 100\u306f\u6210\u529f0.88\u3001\u81ea\u5df1\u4ea4\u5dee0.12\u3001\u5e73\u5747spread error {d2['heldout_mean_spread_error']:.3g}\u3002\u5b66\u7fd2\u306b\u3088\u308b\u521d\u56de\u767a\u898b\u3001\u5b66\u7fd2\u5f8c\u306e\u5b89\u5b9a\u5316\u3001spread\u5236\u5fa1\u3092\u78ba\u8a8d\u3057\u305f\u304c\u3001\u4ea4\u5dee\u30e2\u30fc\u30c9\u306f\u6b8b\u3063\u305f\u3002
- **five-bar D=3**: final 100\u306f\u6210\u529f1.00\u3001\u81ea\u5df1\u4ea4\u5dee0\u3002\u521d\u56de\u6210\u529f\u306fepisode 3\u30011,033 mechanics step\u3060\u304c\u3001policy update\u306f0\u56de\u3067\u3042\u308b\u3002\u3057\u305f\u304c\u3063\u3066\u901f\u3044\u521d\u56de\u767a\u898b\u306f\u672a\u66f4\u65b0NN\u306e\u521d\u671f\u65b9\u7b56\u5206\u5e03\u306b\u3088\u308b\u3082\u306e\u3067\u3001\u5831\u916c\u5b66\u7fd2\u306e\u52b9\u679c\u3068\u306f\u4e3b\u5f35\u3067\u304d\u306a\u3044\u3002
- **waterbomb D=sqrt(2)**: episode 47\u300117,324 mechanics step\u30015 updates\u5f8c\u306b\u521d\u56de\u6210\u529f\u3057\u3001final 100\u306f\u6210\u529f0.93\u3001\u81ea\u5df1\u4ea4\u5dee0\u3001\u5e73\u5747spread error {water2['heldout_mean_spread_error']:.3g}\u3002\u305f\u3060\u3057random mechanics RMST\u306f18,132.5\u3067\u512a\u4f4d\u306f1.047\u500d\u306b\u3059\u304e\u305a\u3001random\u4e2d\u592e\u502410,718.5\u306fRL\u3088\u308a\u65e9\u3044\u3002random 10\u8a66\u884c\u4e2d6\u8a66\u884c\u3082RL\u3088\u308a\u65e9\u304b\u3063\u305f\u3002\u3055\u3089\u306bRL\u76f4\u63a5\u8a08\u6642{water2['direct_rl_rollout_plus_update_seconds']:.2f}\u79d2\u306b\u5bfe\u3057\u6821\u6b63random RMST\u306f{water2['calibrated_random_wall_rmst_seconds']:.2f}\u79d2\u3067\u3001wall\u4e0a\u306fRL\u304c\u7d04{1.0 / water2['wall_ratio_calibrated_random_over_direct_rl']:.2f}\u500d\u9045\u3044\u3002
- **torus D=2**: episode 8,192\u307e\u3067\u6210\u529f\u30fb\u81ea\u5df1\u4ea4\u5dee\u30fbwrong-flat\u306f\u3059\u3079\u30660\u3067\u3001\u5168episode\u304ctimeout\u3060\u3063\u305f\u3002qF=q=0\u3001joint milestone\u306f0\u3001spread-only milestone\u306f3/8,192\u3060\u3051\u3067\u3001final-seed 100\u8a55\u4fa1\u3082\u5168timeout\u3060\u3063\u305f\u3002flatness near-miss\u95be\u5024\u3078\u5165\u308c\u306a\u3044dead zone\u3067\u3042\u308a\u3001\u3053\u306e\u5171\u901a\u5831\u916c\u3067\u306f\u8a98\u5c0e\u4fe1\u53f7\u304c\u4e0d\u8db3\u3057\u305f\u3002
- **waterbomb D=1**: \u6210\u529f0\u3002last 1,000\u306e\u81ea\u5df1\u4ea4\u5dee\u73870.999\u3001final 100\u306e\u81ea\u5df1\u4ea4\u5dee\u73871.00\u3067\u3001\u81ea\u5df1\u4ea4\u5deeattractor\u3078\u53ce\u675f\u3057\u305f\u3002spread milestone\u3092\u5b89\u5b9a\u3057\u3066\u5f97\u308b\u4ea4\u5dee\u7d4c\u8def\u304ctimeout\u3088\u308a\u9ad8\u8a55\u4fa1\u306b\u306a\u308a\u3001\u56de\u907f\u65b9\u7b56\u3092\u5b66\u3079\u306a\u304b\u3063\u305f\u3002

## \u521d\u56de\u767a\u898b\u3068random\u6bd4\u8f03

episode/mechanics\u6bd4\u8f03\u306b\u306f\u65e2\u5b58\u306e\u4e92\u63db\u306a\u4e00\u69d8random\u8a66\u884c\u3092\u7528\u3044\u3001\u5404\u6761\u4ef6\u3067\u5171\u901a\u306e\u884c\u653f\u6253\u5207\u308a\u4e0a\u9650\u306b\u518dcensor\u3057\u305fRMST\u3092\u4f7f\u3063\u305f\u3002five-bar D=2\u3067\u306f\u660e\u78ba\u306brandom\u3088\u308a\u65e9\u304b\u3063\u305f\u3002\u4e00\u65b9\u3001D=3\u306f0 update\u306a\u306e\u3067\u5b66\u7fd2\u52b9\u679c\u3092\u793a\u3055\u305a\u3001waterbomb D=sqrt(2)\u306emechanics\u512a\u4f4d\u306f\u5f31\u304fwall\u3067\u306f\u8ca0\u3051\u305f\u3002torus\u3068waterbomb D=1\u306fRL\u30fbrandom\u3068\u3082\u767a\u898b\u306a\u3057\u3067\u3042\u308a\u3001\u901f\u5ea6\u6bd4\u306f\u5b9a\u7fa9\u3057\u306a\u3044\u3002

wall\u6bd4\u8f03\u306eRL\u5024\u306f\u3001seed 2101\u3092\u540c\u4e00\u8a2d\u5b9a\u3067\u518d\u5b9f\u884c\u3057\u3001full run\u306e\u4fdd\u5b58\u6e08\u307fepisode/update/first-success path\u3068\u4e00\u81f4\u3057\u305f\u5834\u5408\u306e rollout + Actor\u2013Critic update\u6642\u9593\u3067\u3042\u308b\u3002random\u5024\u306f\u3001\u540c\u3058idle machine\u4e0a\u306e\u56fa\u5b9auniform prefix throughput\u3067common-censor mechanics RMST\u3092\u63db\u7b97\u3057\u305f\u6821\u6b63wall-equivalent\u3067\u3001random first-discovery wall\u3092\u76f4\u63a5\u8a08\u6642\u3057\u305f\u5024\u3067\u306f\u306a\u3044\u3002\u5358\u4e00RL seed\u5bfe\u8907\u6570random seed RMST\u306e\u8a18\u8ff0\u6bd4\u8f03\u3067\u3042\u308a\u3001\u63a8\u6e2c\u7d71\u8a08\u7684\u306aspeedup\u3067\u306f\u306a\u3044\u3002

## \u7dcf\u5408\u7d50\u8ad6

\u5171\u901a\u5831\u916c\u306ffive-bar D=2\u3001five-bar D=3\u3001waterbomb D=sqrt(2)\u3067\u5b66\u7fd2\u5f8c\u306e\u5b89\u5b9a\u6298\u308a\u7573\u307f\u3068spread\u5236\u5fa1\u3092\u5b9f\u73fe\u3057\u305f\u3002\u3057\u304b\u3057\u3001\u5168\u69cb\u9020\u3092\u5305\u62ec\u3057\u3066\u300c\u521d\u56de\u767a\u898b\u3092random\u3088\u308a\u9ad8\u901f\u5316\u3059\u308b\u300d\u3068\u306f\u7d50\u8ad6\u3067\u304d\u306a\u3044\u3002\u5b66\u7fd2\u306b\u3088\u308b\u660e\u78ba\u306a\u521d\u56de\u9ad8\u901f\u5316\u3092\u78ba\u8a8d\u3067\u304d\u305f\u306e\u306ffive-bar D=2\u3060\u3051\u3067\u3042\u308b\u3002torus\u306eflatness dead zone\u3068waterbomb D=1\u306e\u81ea\u5df1\u4ea4\u5deeattractor\u3092\u89e3\u6d88\u3059\u308b\u306b\u306f\u3001\u5831\u916c\u5f62\u614b\u81ea\u4f53\u306e\u8ffd\u52a0\u6539\u8a02\u304c\u5fc5\u8981\u3067\u3042\u308a\u3001\u30cf\u30a4\u30d1\u30fc\u30d1\u30e9\u30e1\u30fc\u30bf\u8abf\u6574\u3060\u3051\u3067\u306f\u4e0d\u5341\u5206\u3060\u3063\u305f\u3002

## \u56f3

- [five-bar D=2 learning curve](full_five_d2_profiles.png)
- [five-bar D=3 learning curve](full_five_d3_profiles.png)
- [torus D=2 learning curve\uff08episode 8,192\u3067\u6253\u5207\u308a\uff09](full_torus2_profiles.png)
- [waterbomb D=1 learning curve](full_water1_profiles.png)
- [waterbomb D=sqrt(2) learning curve](full_watersqrt2_profiles.png)
- [\u51685\u6761\u4ef6\u306e\u521d\u56de\u767a\u898bmechanics\u6bd4\u8f03](full_random_first_hit_comparison.png)
- [\u6210\u529f3\u6761\u4ef6\u306e\u6821\u6b63wall\u6bd4\u8f03](final_ori5_calibrated_wall_comparison.png)
- episode/mechanics ECDF: [five D=2](full_five_d2_random_ecdf_survival.png), [five D=3](full_five_d3_random_ecdf_survival.png), [torus](full_torus2_random_ecdf_survival.png), [water D=1](full_water1_random_ecdf_survival.png), [water D=sqrt(2)](full_watersqrt2_random_ecdf_survival.png)

## \u6a5f\u68b0\u53ef\u8aad\u30c7\u30fc\u30bf

- [\u6700\u7d42\u6bd4\u8f03CSV](final_ori5_comparison.csv)
- [\u6700\u7d42\u6bd4\u8f03JSON](final_ori5_comparison.json)
- [\u6821\u6b63wall\u6bd4\u8f03CSV](final_ori5_calibrated_wall_comparison.csv)
- [wall replay\u691c\u8a3c](final_ori5_wall_replay_validation.json)
- [artifact QA](final_ori5_artifact_qa.json)
- [torus\u6d3e\u751f\u8a55\u4fa1](derived_torus_user_stopped_ep8192/evaluation_summary.json)
"""
    write_markdown(ANALYSIS / "final_ori5_results_ja.md", markdown)
    print(json.dumps({
        "valid": payload["global_valid"],
        "comparison_csv": str(ANALYSIS / "final_ori5_comparison.csv"),
        "comparison_json": str(ANALYSIS / "final_ori5_comparison.json"),
        "report": str(ANALYSIS / "final_ori5_results_ja.md"),
    }, indent=2, ensure_ascii=False))
    return 0 if payload["global_valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
