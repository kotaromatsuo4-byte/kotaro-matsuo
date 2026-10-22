"""Standalone finalizer for the active ori6 zero-reject staged experiment.

This file is intentionally outside the immutable runner/core/analyzer binding.
It refuses to write while a staged child is alive and writes only to the
separate ``experiments_ori6/analysis/zero_reject_reward_final`` directory.
The current run is compared descriptively with the old P0X2RCS05 seed-4101
two-task diagnostic through its exact episode-6144 atomic prefix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import analyze_ori_codex6_zero_reject_staged as staged


SCRIPT_DIRECTORY = Path(__file__).resolve().parent
ACTIVE_ROOT = (
    SCRIPT_DIRECTORY / "experiments_ori6" / "staged" /
    "zero_reject_reward" / "P0X2RCS05ZR0R1"
)
OUTPUT_DIRECTORY = (
    SCRIPT_DIRECTORY / "experiments_ori6" / "analysis" /
    "zero_reject_reward_final"
)
OLD_ROOT = (
    SCRIPT_DIRECTORY / "experiments_ori6" / "full" /
    "two_task_full_diagnostic" / "P0X2RCS05"
)
TASKS = ("torus2", "water1")
PROFILES = staged.PROFILES
SEEDS = staged.SEEDS
OLD_ENDPOINT = 6144


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def read_json(path: Path, default=None):
    if not path.is_file():
        return default
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    process_query_limited_information = 0x1000
    still_active = 259
    handle = ctypes.windll.kernel32.OpenProcess(
        process_query_limited_information, False, int(pid)
    )
    if not handle:
        return False
    exit_code = ctypes.c_ulong()
    try:
        ok = ctypes.windll.kernel32.GetExitCodeProcess(
            handle, ctypes.byref(exit_code)
        )
        return bool(ok and exit_code.value == still_active)
    finally:
        ctypes.windll.kernel32.CloseHandle(handle)


def staged_statuses(root: Path) -> list[dict]:
    records = []
    for task in TASKS:
        for seed in SEEDS:
            path = root / "_runner_logs" / task / f"seed{seed}" / "status.json"
            payload = read_json(path, {})
            records.append({
                "task": task,
                "seed": seed,
                "path": str(path.resolve()),
                "payload": payload,
                "pid_alive": process_is_alive(int(payload.get("pid", 0) or 0)),
            })
    return records


def assert_safe_to_finalize(root: Path) -> tuple[dict, list[dict]]:
    statuses = staged_statuses(root)
    alive = [record for record in statuses if record["pid_alive"]]
    if alive:
        labels = ", ".join(
            f"{record['task']}/seed{record['seed']} pid={record['payload'].get('pid')}"
            for record in alive
        )
        raise RuntimeError(f"Refusing to finalize while staged children are alive: {labels}")
    final_summary = read_json(root / "analysis" / "final_summary.json")
    if final_summary is None:
        raise FileNotFoundError(
            "Runner final_summary.json is absent; wait for safe runner finalization."
        )
    return final_summary, statuses


def torch_episode(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        import torch
        payload = torch.load(path, map_location="cpu", weights_only=False)
        episode = int(payload.get("episode", 0))
        return episode if episode > 0 else None
    except (OSError, RuntimeError, EOFError, ValueError):
        return None


def csv_last_episode(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        with open(path, "rb") as handle:
            handle.seek(max(0, path.stat().st_size - 65536))
            lines = [
                line for line in handle.read().decode("utf-8", errors="ignore").splitlines()
                if line.strip()
            ]
        return int(float(next(csv.reader([lines[-1]]))[-1])) if lines else None
    except (OSError, csv.Error, ValueError, IndexError):
        return None


def authoritative_endpoint(run_directory: Path) -> dict:
    checkpoint = torch_episode(run_directory / "checkpoint.pt")
    policy = torch_episode(run_directory / "policy.pt")
    history = csv_last_episode(run_directory / "training_history.csv")
    updates = csv_last_episode(run_directory / "update_history.csv")
    complete = (
        policy == 10000
        and history == 10000
        and updates == 10000
        and (run_directory / "summary.json").is_file()
        and (run_directory / "run_config.json").is_file()
    )
    if complete:
        return {
            "episode": 10000,
            "authority": "complete_final_policy_history_summary",
            "checkpoint_episode": checkpoint,
            "policy_episode": policy,
            "history_episode": history,
            "update_episode": updates,
            "runner_checkpoint_10000_gap": checkpoint != 10000,
        }
    return {
        "episode": checkpoint or 0,
        "authority": "periodic_atomic_checkpoint",
        "checkpoint_episode": checkpoint,
        "policy_episode": policy,
        "history_episode": history,
        "update_episode": updates,
        "runner_checkpoint_10000_gap": False,
    }


def _mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.fmean(values)) if values else math.nan


def _median(values: Iterable[float]) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(values)) if values else math.nan


def enrich_record(record: dict, task: str) -> dict:
    result = dict(record)
    if task == "water1":
        result["wrong_basin"] = bool(
            not record["invalid"]
            and record["flatness"] <= 0.5
            and record["spread_error"] >= 0.30
        )
        result["collapsed_basin"] = False
    else:
        result["wrong_basin"] = False
        result["collapsed_basin"] = bool(
            not record["invalid"]
            and record["flatness"] >= 0.9
            and record["spread_error"] >= 0.30
        )
    return result


def summary(records: Sequence[dict], start: int, end: int) -> dict:
    rows = [row for row in records if start <= row["episode"] <= end]
    n = len(rows)
    return {
        **staged.window_summary(rows, start, end),
        "wrong_basin_count": sum(row["wrong_basin"] for row in rows),
        "wrong_basin_rate": sum(row["wrong_basin"] for row in rows) / n if n else 0.0,
        "collapsed_basin_count": sum(row["collapsed_basin"] for row in rows),
        "collapsed_basin_rate": sum(row["collapsed_basin"] for row in rows) / n if n else 0.0,
        "reject_episode_count": sum(row["rejects"] > 0 for row in rows),
        "reject_episode_rate": sum(row["rejects"] > 0 for row in rows) / n if n else 0.0,
        "total_rejected_actions": sum(row["rejects"] for row in rows),
    }


def load_run(
    dataset: str, task: str, seed: int, run_directory: Path,
    endpoint: int, selected: bool,
) -> dict:
    raw_rows, history_errors = staged.read_history(
        run_directory / "training_history.csv", endpoint
    )
    updates, update_errors = staged.read_updates(
        run_directory / "update_history.csv", endpoint
    )
    records = [enrich_record(staged.normalise(row), task) for row in raw_rows]
    raw_by_episode = {
        int(float(row["episode"])): row for row in raw_rows
    }
    for record in records:
        raw = raw_by_episode[record["episode"]]
        record.update({
            "rejected_reward": staged._float(
                raw, ("ori6_total_rejected_crossing_reward",), 0.0
            ),
            "zero_reject_timeout_reward": staged._float(
                raw, ("ori6_zero_reject_timeout_reward",), 0.0
            ),
            "max_reject_streak": staged._int(
                raw, ("ori6_max_consecutive_rejected_crossings",), 0
            ),
        })
    first = summary(records, 1, min(1024, endpoint))
    recent = summary(records, max(1, endpoint - 1023), endpoint)
    recent_updates = staged.update_window_summary(
        updates, max(1, endpoint - 511), endpoint
    )
    windows = []
    for start in range(1, endpoint + 1, 512):
        end = min(endpoint, start + 511)
        windows.append({
            **summary(records, start, end),
            **staged.update_window_summary(updates, start, end),
        })
    errors = [*history_errors, *update_errors]
    if dataset == "current":
        if recent["max_abs_zero_reject_ordinary_reward"] > 1.0e-12:
            errors.append("ordinary reject reward was nonzero")
        config = staged.validate_config(
            run_directory / "run_config.json", task, seed
        )
        if not config["pass"]:
            errors.extend(f"config:{value}" for value in config["errors"])
    else:
        config = None
    return {
        "dataset": dataset,
        "task": task,
        "seed": seed,
        "selected": selected,
        "run_directory": str(run_directory.resolve()),
        "endpoint": endpoint,
        "integrity_pass": not errors,
        "integrity_errors": errors,
        "config_validation": config,
        "first_1024": first,
        "last_1024": recent,
        "last_512_update_diagnostics": recent_updates,
        "windows_512": windows,
        "records": records,
        "updates": updates,
    }


def combined_scope_summary(runs: Sequence[dict]) -> dict:
    if not runs:
        return {}
    first_rows = []
    recent_rows = []
    recent_updates = []
    for run in runs:
        first_rows.extend(
            row for row in run["records"] if row["episode"] <= min(1024, run["endpoint"])
        )
        recent_rows.extend(
            row for row in run["records"]
            if row["episode"] >= max(1, run["endpoint"] - 1023)
        )
        recent_updates.extend(
            row for row in run["updates"]
            if staged._int(row, ("episode",)) >= max(1, run["endpoint"] - 511)
        )
    # Give concatenated rows a local episode range for the generic summarizer.
    def summarize_arbitrary(rows):
        n = len(rows)
        return {
            "episode_count": n,
            "mean_return": _mean(row["reward"] for row in rows),
            "median_flatness": _median(row["flatness"] for row in rows),
            "median_spread_error": _median(row["spread_error"] for row in rows),
            "constructive_count": sum(row["constructive"] for row in rows),
            "constructive_rate": sum(row["constructive"] for row in rows) / n if n else 0.0,
            "valid_joint_count": sum(row["joint"] for row in rows),
            "valid_joint_rate": sum(row["joint"] for row in rows) / n if n else 0.0,
            "path_joint_discovery_count": sum(row["path_joint_discovery"] for row in rows),
            "strict_count": sum(row["strict"] for row in rows),
            "strict_rate": sum(row["strict"] for row in rows) / n if n else 0.0,
            "wrong_basin_count": sum(row["wrong_basin"] for row in rows),
            "wrong_basin_rate": sum(row["wrong_basin"] for row in rows) / n if n else 0.0,
            "collapsed_basin_count": sum(row["collapsed_basin"] for row in rows),
            "collapsed_basin_rate": sum(row["collapsed_basin"] for row in rows) / n if n else 0.0,
            "reject_episode_rate": sum(row["rejects"] > 0 for row in rows) / n if n else 0.0,
            "mean_reject_count": _mean(row["rejects"] for row in rows),
            "total_rejected_actions": sum(row["rejects"] for row in rows),
            "mechanical_steps": sum(row["steps"] for row in rows),
        }
    return {
        "seed_count": len(runs),
        "seeds": [run["seed"] for run in runs],
        "endpoints": {str(run["seed"]): run["endpoint"] for run in runs},
        "first_1024": summarize_arbitrary(first_rows),
        "last_1024": summarize_arbitrary(recent_rows),
        "last_512_update_diagnostics": staged.update_window_summary(
            recent_updates, 1, max(
                [staged._int(row, ("episode",)) for row in recent_updates] or [1]
            )
        ),
        "integrity_pass": all(run["integrity_pass"] for run in runs),
    }


def input_hashes(runs: Sequence[dict], extra_paths: Sequence[Path]) -> dict:
    result = {}
    for run in runs:
        directory = Path(run["run_directory"])
        for name in (
            "training_history.csv", "update_history.csv", "run_config.json",
            "checkpoint.pt", "policy.pt", "summary.json",
        ):
            path = directory / name
            if path.is_file():
                result[str(path.resolve())] = file_sha256(path)
    for path in extra_paths:
        if path.is_file():
            result[str(path.resolve())] = file_sha256(path)
    return result


def serialisable(value):
    if isinstance(value, dict):
        return {
            key: serialisable(item) for key, item in value.items()
            if key not in {"records", "updates"}
        }
    if isinstance(value, list):
        return [serialisable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_payload(root: Path) -> dict:
    runner_final, statuses = assert_safe_to_finalize(root)
    selected = {
        task: int(seed) for task, seed in runner_final.get("selected_seeds", {}).items()
    }
    current_runs = []
    endpoint_authorities = {}
    status_map = {
        (record["task"], record["seed"]): record["payload"] for record in statuses
    }
    for task in TASKS:
        for seed in SEEDS:
            directory = root / task / PROFILES[task] / f"seed{seed}"
            authority = authoritative_endpoint(directory)
            endpoint_authorities[f"{task}/seed{seed}"] = authority
            if authority["episode"] <= 0:
                continue
            current_runs.append(load_run(
                "current", task, seed, directory, authority["episode"],
                selected.get(task) == seed,
            ))
    old_runs = []
    for task in TASKS:
        directory = OLD_ROOT / task / PROFILES[task] / "seed4101"
        old_runs.append(load_run(
            "old_P0X2RCS05", task, 4101, directory, OLD_ENDPOINT, True
        ))
    scopes = {}
    comparison = {}
    for task in TASKS:
        task_current = [run for run in current_runs if run["task"] == task]
        selected_runs = [run for run in task_current if run["selected"]]
        primary = selected_runs or task_current
        old = [run for run in old_runs if run["task"] == task]
        current_summary = combined_scope_summary(primary)
        old_summary = combined_scope_summary(old)
        scopes[task] = {
            "screen_all_current_seeds": combined_scope_summary(task_current),
            "primary_current_scope": current_summary,
            "old_seed4101_scope": old_summary,
        }
        current_last = current_summary.get("last_1024", {})
        old_last = old_summary.get("last_1024", {})
        comparison[task] = {
            "scope_warning": (
                "descriptive only: current fresh seeds/task-specific tuning versus "
                "old seed4101; post4096 may be selected-seed only"
            ),
            "mean_return_delta": current_last.get("mean_return", math.nan) - old_last.get("mean_return", math.nan),
            "median_flatness_delta": current_last.get("median_flatness", math.nan) - old_last.get("median_flatness", math.nan),
            "median_spread_error_delta": current_last.get("median_spread_error", math.nan) - old_last.get("median_spread_error", math.nan),
            "constructive_rate_delta": current_last.get("constructive_rate", 0.0) - old_last.get("constructive_rate", 0.0),
            "terminal_joint_rate_delta": current_last.get("valid_joint_rate", 0.0) - old_last.get("valid_joint_rate", 0.0),
            "strict_count_current": current_last.get("strict_count", 0),
            "strict_count_old": old_last.get("strict_count", 0),
            "wrong_basin_rate_delta": current_last.get("wrong_basin_rate", 0.0) - old_last.get("wrong_basin_rate", 0.0),
        }
    decision_path = root / "analysis" / "decision_history.json"
    binding_path = root / "execution_manifest_binding.json"
    payload = {
        "schema_version": 1,
        "analysis_type": "ori6_zero_reject_staged_final_comparison",
        "generated_utc": utc_now(),
        "active_experiment_root": str(root.resolve()),
        "old_reference_root": str(OLD_ROOT.resolve()),
        "runner_final_summary": runner_final,
        "runner_statuses": statuses,
        "runner_decisions": read_json(decision_path, {}).get("decisions", []),
        "selected_seeds": selected,
        "endpoint_authorities": endpoint_authorities,
        "current_runs": current_runs,
        "old_runs": old_runs,
        "scopes": scopes,
        "old_comparison": comparison,
        "integrity": {
            "all_current_runs_pass": all(run["integrity_pass"] for run in current_runs),
            "all_old_prefixes_pass": all(run["integrity_pass"] for run in old_runs),
            "ordinary_zero_reject_invariant": all(
                run["last_1024"]["max_abs_zero_reject_ordinary_reward"] <= 1.0e-12
                for run in current_runs
            ),
            "runner_10000_checkpoint_gap_detected": any(
                item["runner_checkpoint_10000_gap"]
                for item in endpoint_authorities.values()
            ),
        },
    }
    payload["input_sha256"] = input_hashes(
        [*current_runs, *old_runs], [
            root / "analysis" / "final_summary.json", decision_path, binding_path,
        ]
    )
    return payload


def write_windows_csv(payload: dict, path: Path) -> None:
    rows = []
    for run in [*payload["current_runs"], *payload["old_runs"]]:
        for window in run["windows_512"]:
            rows.append({
                "dataset": run["dataset"], "task": run["task"],
                "seed": run["seed"], "selected": run["selected"],
                "run_endpoint": run["endpoint"], **window,
            })
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def rolling(values, window=200):
    import numpy as np
    array = np.asarray(values, dtype=float)
    if not len(array):
        return array
    width = min(window, len(array))
    return np.convolve(array, np.ones(width) / width, mode="valid")


def plot_learning(payload: dict, path: Path, mechanics_axis: bool = False) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(5, 2, figsize=(15, 16), constrained_layout=True)
    colors = {4201: "#1f77b4", 4202: "#ff7f0e", 4203: "#2ca02c", 4101: "#111111"}
    for column, task in enumerate(TASKS):
        runs = [
            run for run in [*payload["current_runs"], *payload["old_runs"]]
            if run["task"] == task
        ]
        for run in runs:
            records = run["records"]
            if mechanics_axis:
                x_full = np.cumsum([row["steps"] for row in records])
                xlabel = "cumulative mechanical steps"
            else:
                x_full = np.array([row["episode"] for row in records])
                xlabel = "episode"
            width = min(200, len(records))
            x = x_full[width - 1:]
            old = run["dataset"] != "current"
            label = (
                "old seed4101" if old else
                f"ZR seed{run['seed']}" + (" selected" if run["selected"] else " screen")
            )
            style = "--" if old else "-"
            alpha = 0.9 if old or run["selected"] else 0.45
            linewidth = 1.8 if old or run["selected"] else 1.0
            color = colors[run["seed"]]
            metrics = (
                [row["reward"] for row in records],
                [max(1.0e-8, row["flatness"]) for row in records],
                [row["spread_error"] for row in records],
                [row["constructive"] for row in records],
                [row["wrong_basin"] or row["collapsed_basin"] for row in records],
            )
            for row_index, values in enumerate(metrics):
                axes[row_index, column].plot(
                    x, rolling(values, 200), style, color=color, alpha=alpha,
                    linewidth=linewidth, label=label,
                )
            joint_x = [
                x_full[index] for index, row in enumerate(records) if row["joint"]
            ]
            strict_x = [
                x_full[index] for index, row in enumerate(records) if row["strict"]
            ]
            if joint_x:
                axes[3, column].scatter(joint_x, [1.02] * len(joint_x), s=8, marker="|", color=color, alpha=alpha)
            if strict_x:
                axes[3, column].scatter(strict_x, [1.08] * len(strict_x), s=18, marker="*", color=color, alpha=alpha)
        labels = (
            "episodic return (MA200)", "terminal flatness F (MA200)",
            "terminal spread error (MA200)",
            "constructive rate; | terminal joint; * strict", 
            "wrong/collapsed basin rate (MA200)",
        )
        for row_index, label in enumerate(labels):
            axes[row_index, column].set_ylabel(label)
            axes[row_index, column].set_xlabel(xlabel)
            axes[row_index, column].grid(alpha=0.2)
        axes[1, column].set_yscale("log")
        axes[2, column].axhline(0.1, color="#666666", linestyle=":", linewidth=0.9)
        axes[0, column].set_title(task)
        axes[0, column].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Zero-reject vs old P0X2RCS05 learning curves (descriptive, unmatched seeds)"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_optimizer(payload: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(6, 2, figsize=(15, 18), constrained_layout=True)
    colors = {4201: "#1f77b4", 4202: "#ff7f0e", 4203: "#2ca02c", 4101: "#111111"}
    for column, task in enumerate(TASKS):
        runs = [
            run for run in [*payload["current_runs"], *payload["old_runs"]]
            if run["task"] == task
        ]
        for run in runs:
            updates = run["updates"]
            episodes = np.array([staged._int(row, ("episode",)) for row in updates])
            old = run["dataset"] != "current"
            label = "old seed4101" if old else f"ZR seed{run['seed']}" + (" selected" if run["selected"] else " screen")
            style = "--" if old else "-"
            alpha = 0.9 if old or run["selected"] else 0.4
            width = 8
            x = episodes[width - 1:]
            update_metrics = (
                [max(1e-12, staged._float(row, ("critic_loss",))) for row in updates],
                [staged._float(row, ("critic_gradient_norm",)) for row in updates],
                [staged._float(row, ("actor_gradient_norm",)) for row in updates],
                [staged._float(row, ("entropy",)) for row in updates],
                [staged._float(row, ("actor_learning_rate",)) for row in updates],
            )
            for index, values in enumerate(update_metrics):
                axes[index, column].plot(
                    x, rolling(values, width), style, color=colors[run["seed"]],
                    alpha=alpha, linewidth=1.3, label=label,
                )
            records = run["records"]
            record_x = np.array([row["episode"] for row in records])
            width_record = min(200, len(records))
            axes[5, column].plot(
                record_x[width_record - 1:],
                rolling([row["reject_rate"] for row in records], width_record),
                style, color=colors[run["seed"]], alpha=alpha, linewidth=1.3,
                label=label,
            )
        labels = (
            "critic loss (MA8 updates)", "critic gradient norm (MA8)",
            "actor gradient norm (MA8)", "entropy (MA8)",
            "applied actor learning rate (MA8)", "rejected-action rate (MA200)",
        )
        for row_index, label in enumerate(labels):
            axes[row_index, column].set_ylabel(label)
            axes[row_index, column].set_xlabel("episode")
            axes[row_index, column].grid(alpha=0.2)
        axes[0, column].set_yscale("log")
        axes[1, column].axhline(1.0, color="#666666", linestyle=":", linewidth=0.9)
        axes[2, column].axhline(0.5, color="#666666", linestyle=":", linewidth=0.9)
        axes[0, column].set_title(task)
        axes[0, column].legend(fontsize=8, loc="best")
    figure.suptitle("Optimizer, exploration, and reject diagnostics")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def f(value, digits=4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}g}"


def write_report(payload: dict, path: Path) -> None:
    lines = [
        "# zero-reject段階実験 最終報告",
        "",
        f"生成時刻: {payload['generated_utc']}",
        "",
        "この比較はcurrentのfresh seed/task別調整と、旧P0X2RCS05 seed4101を比べる記述的比較であり、paired seed比較ではない。4096以降は選択された1 seedだけのproduction diagnosticで、3-seed安定性確認ではない。",
        "",
        "## 実験設計と停止判定",
        "",
        "- 通常reject報酬は0。horizon rejectだけ通常timeout `-1 + q(last safe)` を1回与える。",
        "- torusはactor LRの最初のmilestoneを1528へ移動。water D=1はnear-miss spread cutoffを0.40へ変更。",
        "- 4096で2/3 seedの持続的target-plane信号を要求し、通過時はpromising seedだけから1 seedを選ぶ。",
        "- productionは二つの連続checkpointで崩壊warningが出た場合に停止する。",
        "",
    ]
    decisions = payload["runner_decisions"]
    if decisions:
        lines.extend(["### 実際のselection/stop履歴", ""])
        for decision in decisions:
            lines.append(
                f"- {decision.get('task')} ep{decision.get('checkpoint_episode')}: "
                f"{decision.get('rule')}; promote={decision.get('promote')}, "
                f"stop={decision.get('stop')}, selected={decision.get('selected_seed')}; "
                + "; ".join(decision.get("reasons", []))
            )
        lines.append("")
    lines.extend([
        "## 最終結果と旧6144比較",
        "",
        "| task | current scope/endpoints | return | terminal F | spread error | constructive | terminal joint | strict | wrong/collapse basin | old return/F/eD |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for task in TASKS:
        current = payload["scopes"][task]["primary_current_scope"]
        old = payload["scopes"][task]["old_seed4101_scope"]
        c = current["last_1024"]
        o = old["last_1024"]
        basin = c["wrong_basin_rate"] if task == "water1" else c["collapsed_basin_rate"]
        lines.append(
            f"| {task} | seeds {current['seeds']}; {current['endpoints']} | "
            f"{f(c['mean_return'])} | {f(c['median_flatness'])} | {f(c['median_spread_error'])} | "
            f"{c['constructive_rate']:.2%} | {c['valid_joint_rate']:.2%} | {c['strict_count']} | "
            f"{basin:.2%} | {f(o['mean_return'])}/{f(o['median_flatness'])}/{f(o['median_spread_error'])} |"
        )
    lines.extend(["", "### seed別", "", 
        "| task | seed | selected | endpoint | return | F | eD | constructive | terminal joint | path joint | strict | wrong/collapse | reject episode/action |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in payload["current_runs"]:
        recent = run["last_1024"]
        basin = recent["wrong_basin_rate"] if run["task"] == "water1" else recent["collapsed_basin_rate"]
        lines.append(
            f"| {run['task']} | {run['seed']} | {run['selected']} | {run['endpoint']} | "
            f"{f(recent['mean_return'])} | {f(recent['median_flatness'])} | {f(recent['median_spread_error'])} | "
            f"{recent['constructive_rate']:.2%} | {recent['valid_joint_rate']:.2%} | "
            f"{recent['path_joint_discovery_count']} | {recent['strict_count']} | {basin:.2%} | "
            f"{recent['reject_episode_rate']:.2%}/{recent['total_rejected_actions']} |"
        )
    lines.extend(["", "## optimizer・critic診断（最終512 episode）", "",
        "| task | seed | critic loss mean/p95/max | critic grad mean/p95/max; >1 | actor grad mean/p95/max; >0.5 | entropy | actor LR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for run in payload["current_runs"]:
        u = run["last_512_update_diagnostics"]
        lines.append(
            f"| {run['task']} | {run['seed']} | {f(u['critic_loss_mean'])}/{f(u['critic_loss_p95'])}/{f(u['critic_loss_max'])} | "
            f"{f(u['critic_gradient_norm_mean'])}/{f(u['critic_gradient_norm_p95'])}/{f(u['critic_gradient_norm_max'])}; {u['critic_gradient_over_clip_fraction']:.1%} | "
            f"{f(u['actor_gradient_norm_mean'])}/{f(u['actor_gradient_norm_p95'])}/{f(u['actor_gradient_norm_max'])}; {u['actor_gradient_over_clip_fraction']:.1%} | "
            f"{f(u['entropy_mean'])} | {f(u['actor_learning_rate_last'])} |"
        )
    lines.extend([
        "", "critic gradientの`>1`、actor gradientの`>0.5`は記録されたpre-clip normがclip閾値を超えた更新割合。critic shockと方策保持崩壊を区別するため、全512窓のmean/p95/maxもCSV・JSONへ保存した。",
        "", "## 整合性と解釈上の注意", "",
        f"- current整合性: {payload['integrity']['all_current_runs_pass']}",
        f"- 旧6144 prefix整合性: {payload['integrity']['all_old_prefixes_pass']}",
        f"- 通常zero-reject報酬が全て0: {payload['integrity']['ordinary_zero_reject_invariant']}",
    ])
    if payload["integrity"]["runner_10000_checkpoint_gap_detected"]:
        lines.append(
            "- runnerは10000が512の倍数でないためcheckpoint.pt=9728を参照するが、postprocessorはpolicy.pt episode10000・連続history/update・summary/configの完全集合を確認して10000を正式endpointとして回収した。"
        )
    lines.extend([
        "- Waterのwrong-plane指標は`F<=0.5かつspread error>=0.30`。Torusのcollapse指標は`F>=0.9かつspread error>=0.30`。いずれもstrict successとは別の診断指標。",
        "- strict successが出ても、最後1024でconstructive/terminal-jointが維持されなければ安定化とは呼ばない。",
        "", "## 出力", "",
        "- `learning_curve_episode_comparison.png`: episode軸のreturn/F/eD/constructive・joint・strict/wrong basin",
        "- `learning_curve_mechanics_comparison.png`: 累積mechanical steps軸の同比較",
        "- `optimizer_reject_diagnostics.png`: critic/actor/entropy/LR/reject",
        "- `final_windows_512.csv`: reward・geometry・success信号・wrong basin・reject・optimizerの全512窓",
        "- `final_metrics.json`: endpoint authority、selection/stop、全指標、入力SHA-256",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_all(payload: dict, output: Path) -> dict:
    if output.resolve() == ACTIVE_ROOT.resolve() or ACTIVE_ROOT.resolve() in output.resolve().parents:
        raise RuntimeError("Finalizer output must stay outside the active experiment root")
    output.mkdir(parents=True, exist_ok=True)
    metrics = output / "final_metrics.json"
    windows = output / "final_windows_512.csv"
    report = output / "final_report_ja.md"
    episode_plot = output / "learning_curve_episode_comparison.png"
    mechanics_plot = output / "learning_curve_mechanics_comparison.png"
    optimizer_plot = output / "optimizer_reject_diagnostics.png"
    atomic_json(metrics, serialisable(payload))
    write_windows_csv(payload, windows)
    plot_learning(payload, episode_plot, mechanics_axis=False)
    plot_learning(payload, mechanics_plot, mechanics_axis=True)
    plot_optimizer(payload, optimizer_plot)
    write_report(payload, report)
    return {
        "metrics": str(metrics.resolve()),
        "windows": str(windows.resolve()),
        "report": str(report.resolve()),
        "episode_plot": str(episode_plot.resolve()),
        "mechanics_plot": str(mechanics_plot.resolve()),
        "optimizer_plot": str(optimizer_plot.resolve()),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-root", type=Path, default=ACTIVE_ROOT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--check-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    root = options.experiment_root.resolve()
    if options.check_ready:
        try:
            final, statuses = assert_safe_to_finalize(root)
            print(json.dumps({
                "ready": True,
                "selected_seeds": final.get("selected_seeds", {}),
                "statuses": statuses,
            }, ensure_ascii=False, indent=2))
            return 0
        except (RuntimeError, FileNotFoundError) as error:
            print(json.dumps({"ready": False, "reason": str(error)}, ensure_ascii=False, indent=2))
            return 3
    payload = build_payload(root)
    artifacts = write_all(payload, options.output.resolve())
    print(json.dumps({
        "integrity": payload["integrity"], "artifacts": artifacts,
    }, ensure_ascii=False, indent=2))
    return 0 if (
        payload["integrity"]["all_current_runs_pass"]
        and payload["integrity"]["all_old_prefixes_pass"]
        and payload["integrity"]["ordinary_zero_reject_invariant"]
    ) else 2


if __name__ == "__main__":
    raise SystemExit(main())
