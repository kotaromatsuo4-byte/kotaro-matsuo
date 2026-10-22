"""保存済み探索結果が従来のfive-bar canonical cellを何個覆うか調べる。

探索中のelite archiveは重なり順も区別するため、理論上のcanonical 6より多い
cellを持つ場合がある。このscriptは探索には一切介入せず、保存された各cellの
代表経路を現在のdescriptorで読み直し、``reachability_report.json`` に保存した
canonical witnessへ既存archiveと同じ距離規則で照合する。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re

import numpy as np

from archive import find_elite_cell
from descriptor import terminal_descriptor
from five_bar_reachability_report import (
    load_five_bar_model,
    load_saved_path,
    path_as_history,
)


def _cell_directories(seed_directory):
    """``cell[0]`` 形式のdirectoryをcell番号順に返す。"""

    paths = [
        path
        for path in seed_directory.iterdir()
        if path.is_dir() and re.fullmatch(r"cell\[\d+\]", path.name)
    ]
    return sorted(paths, key=lambda path: int(re.search(r"\d+", path.name).group()))


def _saved_descriptor(cell_directory, model, parameters):
    """保存代表経路から現在の終端shapeとlayer keyを復元する。"""

    nodes = load_saved_path(cell_directory)
    history = path_as_history(nodes, parameters["reference_point"])
    return terminal_descriptor(
        history,
        len(history) - 1,
        model["member"],
        model["length_scale"],
        parameters["layer_threshold"],
        parameters["overlap_tolerance"],
        parameters["separation_tolerance"],
        parameters["collinearity_tolerance"],
        parameters["parallel_tolerance"],
    )


def _canonical_archive(reference_report):
    """reference JSONのtargetを``find_elite_cell``用のarchiveへ変換する。"""

    return [
        {
            "shape_descriptor": np.asarray(
                target["prototype_shape_descriptor"],
                dtype=float,
            ),
            "layer_key": tuple(tuple(entry) for entry in target["layer_key"]),
        }
        for target in reference_report["targets"]
    ]


def inspect_seed(seed_directory, canonical_archive, model, parameters):
    """一seedの全保存cellをcanonical targetへ照合する。"""

    cells = []
    covered = set()
    for cell_directory in _cell_directories(seed_directory):
        shape, layer_key = _saved_descriptor(cell_directory, model, parameters)
        target_id = find_elite_cell(
            canonical_archive,
            np.asarray(shape, dtype=float),
            tuple(layer_key),
            parameters["rms_tolerance"],
            parameters["max_tolerance"],
        )
        if target_id is not None:
            covered.add(int(target_id))
        cells.append(
            {
                "cell": cell_directory.name,
                "canonical_target_id": (
                    None if target_id is None else int(target_id)
                ),
                "x_signature": np.rint(
                    np.asarray(shape, dtype=float).reshape(-1, 2)[:, 0]
                ).astype(int).tolist(),
                "layer_key": [list(map(int, entry)) for entry in layer_key],
            }
        )

    return {
        "seed_directory": str(seed_directory.resolve()),
        "archive_cell_count": len(cells),
        "canonical_covered_ids": sorted(covered),
        "canonical_coverage_count": len(covered),
        "extra_cell_count": sum(
            cell["canonical_target_id"] is None for cell in cells
        ),
        "cells": cells,
    }


def inspect_results(result_directories, reference_report, model):
    """複数result directoryを同じcanonical archiveで監査する。"""

    parameters = reference_report["descriptor_parameters"]
    canonical_archive = _canonical_archive(reference_report)
    results = []
    for result_directory in result_directories:
        seed_reports = [
            inspect_seed(seed_directory, canonical_archive, model, parameters)
            for seed_directory in sorted(result_directory.glob("seed_*"))
            if seed_directory.is_dir()
        ]
        results.append(
            {
                "result_name": result_directory.name,
                "result_directory": str(result_directory.resolve()),
                "seeds": seed_reports,
            }
        )
    return {
        "reference_report": reference_report.get("source_directories", []),
        "canonical_target_count": len(canonical_archive),
        "results": results,
    }


def write_summary_csv(report, output_path):
    """seed単位のcanonical被覆を短いCSVにも保存する。"""

    with output_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "result_name",
                "seed",
                "archive_cell_count",
                "canonical_coverage_count",
                "extra_cell_count",
                "canonical_covered_ids",
            ),
        )
        writer.writeheader()
        for result in report["results"]:
            for seed in result["seeds"]:
                writer.writerow(
                    {
                        "result_name": result["result_name"],
                        "seed": Path(seed["seed_directory"]).name,
                        "archive_cell_count": seed["archive_cell_count"],
                        "canonical_coverage_count": seed[
                            "canonical_coverage_count"
                        ],
                        "extra_cell_count": seed["extra_cell_count"],
                        "canonical_covered_ids": " ".join(
                            map(str, seed["canonical_covered_ids"])
                        ),
                    }
                )


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference",
        default="result/reachability_audit/reachability_report.json",
        help="canonical witnessを保存したreachability report JSON。",
    )
    parser.add_argument(
        "--result",
        action="append",
        required=True,
        help="監査するresult/<name> directory。複数回指定可能。",
    )
    parser.add_argument(
        "--output",
        default="result/canonical_coverage.json",
        help="詳細JSONの保存先。隣に同名CSVも保存する。",
    )
    return parser.parse_args()


def main():
    base_directory = Path(__file__).resolve().parent
    arguments = parse_arguments()

    reference_path = Path(arguments.reference)
    if not reference_path.is_absolute():
        reference_path = base_directory / reference_path
    result_directories = []
    for value in arguments.result:
        path = Path(value)
        result_directories.append(
            path if path.is_absolute() else base_directory / path
        )
    output_path = Path(arguments.output)
    if not output_path.is_absolute():
        output_path = base_directory / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_report = json.loads(reference_path.read_text(encoding="utf-8"))
    model = load_five_bar_model(base_directory / "model")
    report = inspect_results(result_directories, reference_report, model)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = output_path.with_suffix(".csv")
    write_summary_csv(report, csv_path)

    for result in report["results"]:
        for seed in result["seeds"]:
            print(
                result["result_name"],
                Path(seed["seed_directory"]).name,
                f"archive={seed['archive_cell_count']}",
                f"canonical={seed['canonical_coverage_count']}",
                f"extra={seed['extra_cell_count']}",
            )
    print(f"json: {output_path}")
    print(f"csv: {csv_path}")


if __name__ == "__main__":
    main()
