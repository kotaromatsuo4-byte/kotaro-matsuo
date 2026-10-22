"""Reproducible numerical output for individual and comparative runs."""

from __future__ import annotations

import csv
from importlib.metadata import PackageNotFoundError, version
import json
import math
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .mechanics import Model, vector_to_node


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return value


def write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(value), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(_json_value(row.get(key)), ensure_ascii=False)
                        if isinstance(
                            row.get(key), (dict, list, tuple, np.ndarray)
                        )
                        else row.get(key)
                    )
                    for key in fields
                }
            )


def runtime_metadata() -> dict:
    try:
        cmaes_version = version("cmaes")
    except PackageNotFoundError:
        cmaes_version = "not-installed"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "cmaes": cmaes_version,
    }


def _trajectory_metadata(trajectory: dict) -> dict:
    return {
        "history_id": trajectory.get("history_id"),
        "source": trajectory["source"],
        "terminal_reason": trajectory["terminal_reason"],
        "is_flat": trajectory["is_flat"],
        "collided": trajectory["collided"],
        "failed": trajectory["failed"],
        "logical_steps": trajectory["path"].shape[1] - 1,
        "compute_steps": trajectory["compute_steps"],
        "new_start_step": trajectory["new_start_step"],
        "intervention_count": trajectory["intervention_count"],
        "final_flatness": trajectory["final_flatness"],
    }


def save_trajectory(
    trajectory: dict, folder: Path, model: Model, *, save_nodes: bool = True
) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    np.save(folder / "path.npy", trajectory["path"])
    actions = np.asarray(trajectory["actions"], dtype=float)
    if actions.size == 0:
        actions = np.empty((0, model.vector_size), dtype=float)
    else:
        actions = actions.reshape(-1, model.vector_size)
    np.save(
        folder / "actions.npy",
        actions,
        allow_pickle=False,
    )
    if save_nodes:
        nodes = np.asarray(
            [
                vector_to_node(trajectory["path"][:, step], model)
                for step in range(trajectory["path"].shape[1])
            ]
        )
        np.save(folder / "nodes.npy", nodes)
    write_json(_trajectory_metadata(trajectory), folder / "metadata.json")


def save_method_result(
    result: dict,
    model: Model,
    config,
    output: Path,
    *,
    clean: bool = True,
) -> None:
    if clean and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(config.to_dict(), output / "config.json")
    write_json(runtime_metadata(), output / "runtime.json")
    write_json(result["summary"], output / "summary.json")
    write_csv(result["search_log"], output / "search_log.csv")
    write_csv(result.get("repository_log", []), output / "repository_log.csv")

    archive_rows: list[dict] = []
    solutions = output / "solutions"
    for index, (key, elite) in enumerate(
        sorted(result["outcome_archive"].cells.items())
    ):
        folder = solutions / f"cell_{index:04d}"
        save_trajectory(elite["trajectory"], folder, model)
        metadata = {
            "cell_index": index,
            "shape": key[0],
            "layer_order": key[1],
            "logical_steps": elite["logical_steps"],
            "layer_observation_step": elite["layer_observation_step"],
            "evaluation_index": elite["evaluation_index"],
            "method": elite["method"],
            "candidate_metadata": elite["candidate_metadata"],
        }
        write_json(metadata, folder / "cell.json")
        archive_rows.append(metadata)
    write_csv(archive_rows, output / "outcome_archive.csv")

    if config.save_all_histories:
        histories = output / "histories"
        history_rows = []
        for index, trajectory in enumerate(result["history"].trajectories):
            folder = histories / f"trajectory_{index:05d}"
            save_trajectory(trajectory, folder, model, save_nodes=False)
            history_rows.append(_trajectory_metadata(trajectory))
        write_csv(history_rows, output / "history_index.csv")


def save_comparison(results: dict[str, dict], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    summary_rows = [result["summary"] for result in results.values()]
    write_csv(summary_rows, output / "comparison_summary.csv")

    curves: list[dict] = []
    for method, result in results.items():
        for row in result["search_log"]:
            curves.append(
                {
                    "method": method,
                    "search_steps": row["search_steps"],
                    "outcome_cells": row["outcome_cells"],
                    "shape_count": row["shape_count"],
                    "layer_count": row["layer_count"],
                }
            )
    write_csv(curves, output / "discovery_curves.csv")

    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    for method, result in results.items():
        rows = result["search_log"]
        if not rows:
            continue
        x_values = [0] + [row["search_steps"] for row in rows]
        initial = result["summary"]["bootstrap_outcome_cells"]
        y_values = [initial] + [row["outcome_cells"] for row in rows]
        axis.step(x_values, y_values, where="post", label=method)
    axis.set_xlabel("Newly simulated search steps")
    axis.set_ylabel("Discovered terminal cells")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "discovery_curves.png", dpi=180)
    plt.close(figure)
