"""Add a direct reward-rise / flatness-decline figure without changing formal artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import analyze_ori_codex6_zero_reject_staged as staged
import finalize_ori_codex6_zero_reject_combined as combined


HERE = Path(__file__).resolve().parent
FINAL_DIR = HERE / "experiments_ori6/analysis/zero_reject_reward_final"
MAIN_METRICS = FINAL_DIR / "final_metrics.json"
MAIN_MANIFEST = FINAL_DIR / "artifact_sha256.json"
PNG = FINAL_DIR / "reward_flatness_direct_learning_curves.png"
MANIFEST = FINAL_DIR / "reward_flatness_direct_learning_curves_sha256.json"
TEST_SOURCE = HERE / "test_plot_ori_codex6_reward_flatness_direct.py"


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def cumulative_mean(values):
    import numpy as np
    array = np.asarray(values, dtype=float)
    return np.cumsum(array) / np.arange(1, len(array) + 1)


def rolling_mean(values, width: int = 200):
    import numpy as np
    array = np.asarray(values, dtype=float)
    actual = min(width, len(array))
    return np.convolve(array, np.ones(actual) / actual, mode="valid")


def rolling_median(values, width: int = 200):
    import numpy as np
    array = np.asarray(values, dtype=float)
    actual = min(width, len(array))
    windows = np.lib.stride_tricks.sliding_window_view(array, actual)
    return np.median(windows, axis=-1)


def load_series(run: dict) -> dict:
    endpoint = int(run["endpoint"])
    directory = Path(run["run_directory"])
    raw, errors = staged.read_history(directory / "training_history.csv", endpoint)
    if errors or len(raw) != endpoint:
        raise RuntimeError(
            f"history integrity failed for {run['label']}: rows={len(raw)}, errors={errors}"
        )
    records = [staged.normalise(row) for row in raw]
    return {
        "label": run["label"],
        "dataset": run["dataset"],
        "task": run["task"],
        "seed": run["seed"],
        "endpoint": endpoint,
        "history": str((directory / "training_history.csv").resolve()),
        "episode": [row["episode"] for row in records],
        "reward": [row["reward"] for row in records],
        "flatness": [max(1.0e-8, row["flatness"]) for row in records],
    }


def selected_runs(main: dict) -> dict[str, list[dict]]:
    runs = main.get("runs", [])
    torus = [
        run for run in runs
        if run.get("task") == "torus2"
        and (
            (run.get("dataset") == "r1_zero_reject" and run.get("seed") == 4203)
            or run.get("dataset") == "old_P0X2RCS05"
        )
    ]
    water = [
        run for run in runs
        if run.get("task") == "water1"
        and (
            run.get("dataset") == "water_fallback_far_0p12"
            and run.get("seed") == 4301
            or run.get("dataset") == "r1_zero_reject"
            or run.get("dataset") == "old_P0X2RCS05"
        )
    ]
    if len(torus) != 2 or len(water) != 5:
        raise RuntimeError(f"unexpected run selection: torus={len(torus)}, water={len(water)}")
    selected_water = next(
        run for run in water if run["dataset"] == "water_fallback_far_0p12"
    )
    if int(selected_water["endpoint"]) != 9728:
        raise RuntimeError(
            "direct figure must use formal water endpoint 9728, got "
            f"{selected_water['endpoint']}"
        )
    return {"torus2": torus, "water1": water}


def style(series: dict) -> dict:
    if series["dataset"] == "old_P0X2RCS05":
        return {"color": "#222222", "linestyle": "--", "linewidth": 2.2, "alpha": 0.95}
    if series["dataset"] == "water_fallback_far_0p12":
        return {"color": "#16853b", "linestyle": "-", "linewidth": 2.8, "alpha": 1.0}
    if series["task"] == "torus2":
        return {"color": "#2468b4", "linestyle": "-", "linewidth": 2.8, "alpha": 1.0}
    colors = {4201: "#c45a17", 4202: "#df7f35", 4203: "#eba56e"}
    return {"color": colors[series["seed"]], "linestyle": "-", "linewidth": 1.5, "alpha": 0.75}


def short_label(series: dict) -> str:
    if series["dataset"] == "old_P0X2RCS05":
        return "old seed4101 @6144"
    if series["dataset"] == "water_fallback_far_0p12":
        return "fallback .12 seed4301 @9728 (formal)"
    if series["task"] == "torus2":
        return "zero-reject seed4203 @10000"
    return f"R1 .40 seed{series['seed']} @4096 (rejected)"


def render(series_by_task: dict[str, list[dict]], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure, axes = plt.subplots(2, 2, figsize=(15.5, 10), constrained_layout=True)
    for column, task in enumerate(("torus2", "water1")):
        for series in series_by_task[task]:
            episodes = np.asarray(series["episode"])
            reward = np.asarray(series["reward"])
            flatness = np.asarray(series["flatness"])
            width = min(200, len(episodes))
            current_style = style(series)
            label = short_label(series)
            axes[0, column].plot(
                episodes,
                cumulative_mean(reward),
                label=label,
                **current_style,
            )
            axes[0, column].plot(
                episodes[width - 1:], rolling_mean(reward, width),
                color=current_style["color"], linestyle=":",
                linewidth=max(0.8, current_style["linewidth"] * 0.55),
                alpha=current_style["alpha"] * 0.45,
            )
            axes[1, column].plot(
                episodes[width - 1:], rolling_median(flatness, width),
                label=label,
                **current_style,
            )
            end_return = cumulative_mean(reward)[-1]
            end_flatness = rolling_median(flatness, width)[-1]
            for axis, value in ((axes[0, column], end_return), (axes[1, column], end_flatness)):
                axis.scatter(
                    [series["endpoint"]], [value], s=26,
                    color=current_style["color"], zorder=5,
                )
            if (
                series["dataset"] == "water_fallback_far_0p12"
                or series["task"] == "torus2" and series["dataset"] == "r1_zero_reject"
            ):
                axes[0, column].annotate(
                    f"endpoint {series['endpoint']}",
                    (series["endpoint"], end_return), xytext=(-6, 8),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=current_style["color"],
                )
                axes[1, column].annotate(
                    f"endpoint {series['endpoint']}",
                    (series["endpoint"], end_flatness), xytext=(-6, -14),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=current_style["color"],
                )
        axes[0, column].axvline(4096, color="#777777", linestyle=":", linewidth=1.0)
        axes[1, column].axvline(4096, color="#777777", linestyle=":", linewidth=1.0)
        axes[0, column].axhline(0.0, color="#777777", linewidth=0.7)
        axes[1, column].axhline(0.001, color="#777777", linestyle="--", linewidth=0.9)
        axes[0, column].set_title(
            "torus D=2: selected vs old" if task == "torus2"
            else "water D=1: fallback selected vs R1 .40 and old"
        )
        axes[0, column].set_ylabel("cumulative mean episodic return\n(dotted: MA200)")
        axes[1, column].set_ylabel("terminal flatness F, rolling median 200")
        axes[1, column].set_yscale("log")
        for row in (0, 1):
            axes[row, column].set_xlabel("episode")
            axes[row, column].grid(alpha=0.22)
            axes[row, column].legend(fontsize=8, loc="best")
    figure.suptitle(
        "Direct learning evidence: cumulative return rises while terminal flatness falls\n"
        "Water uses formal prefix 9728 only; training-only episode 10000 is excluded"
    )
    figure.savefig(path, dpi=180)
    plt.close(figure)


def verify_main_manifest() -> str:
    manifest = read_json(MAIN_MANIFEST)
    for raw, expected in manifest.get("artifact_sha256", {}).items():
        path = Path(raw)
        if not path.is_file() or combined.file_sha256(path) != expected:
            raise RuntimeError(f"main artifact changed: {path}")
    return combined.file_sha256(MAIN_MANIFEST)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-ready", action="store_true")
    options = parser.parse_args(argv)
    main_manifest_sha = verify_main_manifest()
    existing = [str(path) for path in (PNG, MANIFEST) if path.exists()]
    if options.check_ready:
        print(json.dumps({
            "ready": not existing, "existing": existing,
            "main_manifest_sha256": main_manifest_sha,
        }, ensure_ascii=False, indent=2))
        return 0 if not existing else 3
    if existing:
        raise FileExistsError("refusing direct-figure overwrite: " + ", ".join(existing))
    main_metrics = read_json(MAIN_METRICS)
    selected = selected_runs(main_metrics)
    series = {
        task: [load_series(run) for run in task_runs]
        for task, task_runs in selected.items()
    }
    input_paths = {
        str(Path(item["history"]).resolve())
        for task_series in series.values() for item in task_series
    }
    input_paths.update({
        str(MAIN_METRICS.resolve()), str(MAIN_MANIFEST.resolve()),
        str(Path(__file__).resolve()), str(TEST_SOURCE.resolve()),
    })
    input_hashes = {
        raw: combined.file_sha256(Path(raw)) for raw in sorted(input_paths)
    }
    render(series, PNG)
    current = {raw: combined.file_sha256(Path(raw)) for raw in sorted(input_paths)}
    if current != input_hashes:
        raise RuntimeError("direct-figure input changed during render")
    combined.atomic_json(MANIFEST, {
        "schema_version": 1,
        "generated_utc": combined.utc_now(),
        "scope": "formal torus10000 and water9728 only; excludes water training-only10000",
        "main_artifact_manifest_sha256": main_manifest_sha,
        "input_hash_set_sha256": combined.canonical_digest(input_hashes),
        "input_sha256": input_hashes,
        "artifact_sha256": {str(PNG.resolve()): combined.file_sha256(PNG)},
        "source_sha256": {
            str(Path(__file__).resolve()): combined.file_sha256(Path(__file__)),
            str(TEST_SOURCE.resolve()): combined.file_sha256(TEST_SOURCE),
        },
        "formal_endpoints": {"torus2": 10000, "water1": 9728},
    })
    print(json.dumps({
        "png": str(PNG.resolve()), "png_sha256": combined.file_sha256(PNG),
        "manifest": str(MANIFEST.resolve()),
        "manifest_sha256": combined.file_sha256(MANIFEST),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
