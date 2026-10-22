"""既存のfive-bar経路から、小摂動による6 cellの可到達性を再検査する。

このスクリプトは探索を再実行しない。``01_0804_five_bar_linkage`` に保存された
経路を現在の ``descriptor.py`` と ``collision2.py`` で読み直し、次を確認する。

* 保存経路の終端を現在のelite cell判定で再分類したときのcell数
* 各経路のcorrector後の最大節点移動ノルム
* 各edgeを現在の連続衝突判定で再検査した結果
* 探索木HTMLに保存された親子関係から復元した、初回発見経路と代表経路の
  摂動回数

探索コードとは独立したread-only監査であり、``main5.py`` はimportも変更もしない。
出力だけを ``result/reachability_audit`` 以下へ保存する。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import numpy as np

from archive import find_elite_cell
from collision2 import edge_edge_crossing_check
from descriptor import terminal_descriptor
from mechanics import compute_gravity_potential, incompatibility
from state import StateNode


DEFAULT_SOURCE_RELATIVE_PATHS = (
    "../01_0804_five_bar_linkage/result/main4_warm4_hybrid/seed_001",
    "../01_0804_five_bar_linkage/result/nn_pool_frontier_random_v2/seed_003",
)


def load_five_bar_model(model_directory):
    """節点、部材、初期部材長および代表長さを読む。"""

    node_initial = np.atleast_2d(
        np.loadtxt(model_directory / "node_initial.dat", delimiter=",", dtype=float)
    )
    member = np.atleast_2d(
        np.loadtxt(model_directory / "member.dat", delimiter=",", dtype=int)
    )
    member_length = np.linalg.norm(
        node_initial[member[:, 0]] - node_initial[member[:, 1]],
        axis=1,
    )
    return {
        "node_initial": node_initial,
        "member": member,
        "member_length": member_length,
        "length_scale": float(np.mean(member_length)),
    }


def load_run_configuration(source_directory):
    """過去runの ``key=value`` 設定を文字列のdictとして読む。"""

    configuration_path = source_directory / "run_config.txt"
    configuration = {}
    if not configuration_path.exists():
        return configuration
    for line in configuration_path.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        configuration[key.strip()] = value.strip()
    return configuration


def numbered_dat_files(cell_directory):
    """cell内の ``0.dat, 1.dat, ...`` を時刻順に返す。"""

    return sorted(cell_directory.glob("*.dat"), key=lambda path: int(path.stem))


def load_saved_path(cell_directory):
    """一つのcellに保存された代表経路の全節点座標を読む。"""

    files = numbered_dat_files(cell_directory)
    return [np.atleast_2d(np.loadtxt(path, dtype=float)) for path in files]


def path_as_history(nodes, reference_point=0.0):
    """descriptorをそのまま再利用するため、保存経路をStateNode列へ戻す。"""

    history = []
    for state_id, node in enumerate(nodes):
        history.append(
            StateNode(
                state_id=state_id,
                node=np.asarray(node, dtype=float),
                parent_id=None if state_id == 0 else state_id - 1,
                children_ids=([] if state_id + 1 == len(nodes) else [state_id + 1]),
                depth=state_id,
                terminal_reason="flat" if state_id + 1 == len(nodes) else None,
                potential=compute_gravity_potential(node, reference_point),
            )
        )
    return history


def extract_plotly_traces(search_tree_path):
    """result.pyがHTMLへ埋め込んだPlotly trace JSONを取り出す。"""

    text = search_tree_path.read_text(encoding="utf-8")
    match = re.search(
        r"const traces = (\[.*?\]);\s*const layout",
        text,
        flags=re.DOTALL,
    )
    if match is None:
        raise ValueError(f"Plotly traces were not found in {search_tree_path}")
    return json.loads(match.group(1))


def tree_metadata(search_tree_path):
    """全stateの親子関係と、elite cellに属する終端stateをHTMLから復元する。"""

    traces = extract_plotly_traces(search_tree_path)
    state_trace = next(trace for trace in traces if trace.get("name") == "StateNode")
    state_rows = {
        int(row[0]): {
            "state_id": int(row[0]),
            "parent_id": int(row[1]),
            "depth": int(row[2]),
            "incoming_perturbation": bool(int(row[5])),
        }
        for row in state_trace["customdata"]
    }

    terminal_ids_by_cell = {}
    representative_id_by_cell = {}
    for trace in traces:
        name = trace.get("name", "")
        cell_match = re.fullmatch(r"elite cell E\[(\d+)\]", name)
        if cell_match is not None:
            terminal_ids_by_cell[int(cell_match.group(1))] = [
                int(state_id) for state_id in trace["customdata"]
            ]

        representative_match = re.fullmatch(r"E\[(\d+)\] representative", name)
        if representative_match is not None:
            id_match = re.search(
                r"representative state=(\d+)",
                trace.get("hovertemplate", ""),
            )
            if id_match is not None:
                representative_id_by_cell[int(representative_match.group(1))] = int(
                    id_match.group(1)
                )

    return {
        "states": state_rows,
        "terminal_ids_by_cell": terminal_ids_by_cell,
        "representative_id_by_cell": representative_id_by_cell,
    }


def path_state_ids_from_tree(states, terminal_state_id):
    """探索木上のrootから指定終端までのstate_id列を返す。"""

    path = []
    state_id = int(terminal_state_id)
    while state_id >= 0:
        path.append(state_id)
        state_id = states[state_id]["parent_id"]
    return list(reversed(path))


def perturbation_data_on_tree_path(states, terminal_state_id):
    """経路上の摂動edge数と、そのchildのdepth/state_idを返す。"""

    path = path_state_ids_from_tree(states, terminal_state_id)
    positions = [
        {
            "depth": states[state_id]["depth"],
            "child_state_id": state_id,
        }
        for state_id in path
        if states[state_id]["incoming_perturbation"]
    ]
    return {
        "count": len(positions),
        "positions": positions,
    }


def first_and_representative_tree_data(metadata, cell_index):
    """cellの初回発見終端と、保存代表終端の摂動情報を返す。"""

    terminal_ids = metadata["terminal_ids_by_cell"].get(cell_index, [])
    representative_id = metadata["representative_id_by_cell"].get(cell_index)
    first_id = min(terminal_ids) if terminal_ids else None

    first_data = None
    if first_id is not None:
        first_data = {
            "terminal_state_id": first_id,
            **perturbation_data_on_tree_path(metadata["states"], first_id),
        }

    representative_data = None
    if representative_id is not None:
        representative_data = {
            "terminal_state_id": representative_id,
            **perturbation_data_on_tree_path(
                metadata["states"],
                representative_id,
            ),
        }

    return {
        "first_discovery": first_data,
        "representative": representative_data,
    }


def corrected_step_norms(nodes):
    """corrector後の保存座標間で測った各edgeのFrobenius normを返す。"""

    return np.asarray(
        [np.linalg.norm(node1 - node0) for node0, node1 in zip(nodes[:-1], nodes[1:])],
        dtype=float,
    )


def collision_edges(nodes, member):
    """現在のcollision2で無効と判定される保存pathのedge indexを返す。"""

    return [
        edge_index
        for edge_index, (node0, node1) in enumerate(zip(nodes[:-1], nodes[1:]))
        if edge_edge_crossing_check(
            node0,
            node1,
            member,
            0.05,
            0.10,
            0.99,
        )
        == 0
    ]


def maximum_constraint_residual(nodes, member, member_length):
    """保存丸め後座標について、全時刻中の最大拘束残差を返す。"""

    return max(
        float(np.linalg.norm(incompatibility(node, member, member_length)))
        for node in nodes
    )


def inspect_saved_cell(
    cell_directory,
    cell_index,
    metadata,
    model,
    descriptor_parameters,
):
    """一つの保存代表経路を現行の物理・cell判定で監査する。"""

    nodes = load_saved_path(cell_directory)
    history = path_as_history(nodes, descriptor_parameters["reference_point"])
    shape_descriptor, layer_key = terminal_descriptor(
        history,
        len(history) - 1,
        model["member"],
        model["length_scale"],
        descriptor_parameters["layer_threshold"],
        descriptor_parameters["overlap_tolerance"],
        descriptor_parameters["separation_tolerance"],
        descriptor_parameters["collinearity_tolerance"],
        descriptor_parameters["parallel_tolerance"],
    )
    norms = corrected_step_norms(nodes)
    invalid_edges = collision_edges(nodes, model["member"])
    tree_data = first_and_representative_tree_data(metadata, cell_index)

    return {
        "source_directory": str(cell_directory.parent.resolve()),
        "source_cell": cell_directory.name,
        "source_cell_index": int(cell_index),
        "shape_descriptor": np.asarray(shape_descriptor, dtype=float).tolist(),
        "x_signature": np.rint(nodes[-1][:, 0]).astype(int).tolist(),
        "layer_key": [list(map(int, entry)) for entry in layer_key],
        "num_states": len(nodes),
        "num_edges": max(len(nodes) - 1, 0),
        "maximum_corrected_step_norm": (
            float(np.max(norms)) if len(norms) else 0.0
        ),
        "p95_corrected_step_norm": (
            float(np.quantile(norms, 0.95)) if len(norms) else 0.0
        ),
        "collision_edge_count": len(invalid_edges),
        "collision_edge_indices": invalid_edges,
        "maximum_constraint_residual_after_dat_rounding": maximum_constraint_residual(
            nodes,
            model["member"],
            model["member_length"],
        ),
        **tree_data,
    }


def collect_witnesses(source_directories, model, descriptor_parameters):
    """指定runの全cell代表経路を監査して返す。"""

    witnesses = []
    for source_directory in source_directories:
        metadata = tree_metadata(source_directory / "search_tree.html")
        cell_directories = sorted(
            (
                path
                for path in source_directory.glob("cell*")
                if re.fullmatch(r"cell\[\d+\]", path.name)
            ),
            key=lambda path: int(re.search(r"\d+", path.name).group()),
        )
        for cell_directory in cell_directories:
            cell_index = int(re.search(r"\d+", cell_directory.name).group())
            witnesses.append(
                inspect_saved_cell(
                    cell_directory,
                    cell_index,
                    metadata,
                    model,
                    descriptor_parameters,
                )
            )
    return witnesses


def group_witnesses_into_targets(witnesses, rms_tolerance, max_tolerance):
    """現elite archiveと同じ距離規則で、run横断のwitnessをcellへまとめる。"""

    target_cells = []
    for witness in witnesses:
        layer_key = tuple(tuple(entry) for entry in witness["layer_key"])
        cell_index = find_elite_cell(
            target_cells,
            np.asarray(witness["shape_descriptor"], dtype=float),
            layer_key,
            rms_tolerance,
            max_tolerance,
        )
        if cell_index is None:
            cell_index = len(target_cells)
            target_cells.append(
                {
                    "shape_descriptor": witness["shape_descriptor"],
                    "layer_key": layer_key,
                    "x_signature": witness["x_signature"],
                    "witnesses": [],
                }
            )
        target_cells[cell_index]["witnesses"].append(witness)

    return [
        {
            "target_id": target_id,
            "x_signature": target["x_signature"],
            "layer_key": [list(entry) for entry in target["layer_key"]],
            "prototype_shape_descriptor": target["shape_descriptor"],
            "witnesses": target["witnesses"],
        }
        for target_id, target in enumerate(target_cells)
    ]


def make_report(base_directory, source_directories):
    """監査レポート全体をJSON化可能なdictとして作る。"""

    model = load_five_bar_model(base_directory / "model")
    descriptor_parameters = {
        "reference_point": 0.0,
        "layer_threshold": 0.10,
        "overlap_tolerance": 0.05,
        "separation_tolerance": 1.0e-5,
        "collinearity_tolerance": 0.02,
        "parallel_tolerance": 0.02,
        "rms_tolerance": 0.02,
        "max_tolerance": 0.05,
    }
    witnesses = collect_witnesses(
        source_directories,
        model,
        descriptor_parameters,
    )
    targets = group_witnesses_into_targets(
        witnesses,
        descriptor_parameters["rms_tolerance"],
        descriptor_parameters["max_tolerance"],
    )

    return {
        "purpose": (
            "read-only revalidation of small-perturbation five-bar paths with "
            "the current descriptor and collision2"
        ),
        "source_directories": [str(path.resolve()) for path in source_directories],
        "source_run_configurations": [
            {
                "source_directory": str(path.resolve()),
                "configuration": load_run_configuration(path),
            }
            for path in source_directories
        ],
        "descriptor_parameters": descriptor_parameters,
        "num_saved_witnesses": len(witnesses),
        "num_unique_target_cells": len(targets),
        "all_witnesses_collision_free": all(
            witness["collision_edge_count"] == 0 for witness in witnesses
        ),
        "maximum_corrected_step_norm_overall": max(
            witness["maximum_corrected_step_norm"] for witness in witnesses
        ),
        "maximum_constraint_residual_after_dat_rounding_overall": max(
            witness["maximum_constraint_residual_after_dat_rounding"]
            for witness in witnesses
        ),
        "targets": targets,
    }


def compact_layer_key(layer_key):
    """Markdown表向けにlayer keyを短い文字列へする。"""

    return " ".join(f"({i},{j},{upper})" for i, j, upper in layer_key)


def report_as_markdown(report):
    """JSONレポートの重要項目を読みやすいMarkdownへ整形する。"""

    lines = [
        "# Five-bar small-perturbation reachability audit",
        "",
        "現在の `descriptor.py` と `collision2.py` を用いて、過去に保存された経路を",
        "再分類・再検査した結果である。探索そのものは再実行していない。",
        "",
        "## Summary",
        "",
        f"- 保存witness数: {report['num_saved_witnesses']}",
        f"- run横断の固有cell数: {report['num_unique_target_cells']}",
        "- 全保存edgeが現collision2で有効: "
        f"{report['all_witnesses_collision_free']}",
        "- corrector後の最大保存step norm: "
        f"{report['maximum_corrected_step_norm_overall']:.6f}",
        "- `.dat` 丸め後の最大拘束残差: "
        f"{report['maximum_constraint_residual_after_dat_rounding_overall']:.3e}",
        "",
        "## Source runs",
        "",
        "| source | selector | budget | perturbation scale | deterministic max norm |",
        "|---|---|---:|---:|---:|",
    ]
    for source in report["source_run_configurations"]:
        configuration = source["configuration"]
        source_path = Path(source["source_directory"])
        source_label = f"{source_path.parent.name}/{source_path.name}"
        lines.append(
            f"| `{source_label}` | {configuration.get('selector', '-')} | "
            f"{configuration.get('total_step_budget', '-')} | "
            f"{configuration.get('perturbation_scale', '-')} | "
            f"{configuration.get('target_norm_max', '-')} |"
        )

    lines.extend(
        [
            "",
            "## Six target cells",
            "",
            "| target | terminal x signature | layer key | witnesses |",
            "|---:|---|---|---:|",
        ]
    )
    for target in report["targets"]:
        lines.append(
            f"| {target['target_id']} | `{target['x_signature']}` | "
            f"`{compact_layer_key(target['layer_key'])}` | "
            f"{len(target['witnesses'])} |"
        )

    lines.extend(
        [
            "",
            "## Witness paths",
            "",
            "`first K` はそのrunでcellが最初に発見された終端までの摂動edge数、",
            "`representative K` は保存された代表経路上の摂動edge数である。",
            "",
            "| target | source | cell | first K | representative K | max step norm | collisions | max residual |",
            "|---:|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target in report["targets"]:
        for witness in target["witnesses"]:
            first = witness["first_discovery"]
            representative = witness["representative"]
            source = Path(witness["source_directory"])
            source_label = f"{source.parent.name}/{source.name}"
            lines.append(
                f"| {target['target_id']} | `{source_label}` | "
                f"`{witness['source_cell']}` | "
                f"{first['count'] if first else '-'} | "
                f"{representative['count'] if representative else '-'} | "
                f"{witness['maximum_corrected_step_norm']:.6f} | "
                f"{witness['collision_edge_count']} | "
                f"{witness['maximum_constraint_residual_after_dat_rounding']:.3e} |"
            )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "この監査で6 cellが得られる場合、現在の小摂動上限と平面内衝突判定の下で",
            "各cellへ至る数値的witnessがrun横断で存在する。単一runで6 cellを回収できる",
            "ことや、各cellに一回の摂動では到達不可能であることまでは証明しない。",
            "初回発見の摂動回数は、複数摂動を優先する探索設計の経験的根拠として用いる。",
            "",
        ]
    )
    return "\n".join(lines)


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        help="保存runのseed directory。複数回指定可能。",
    )
    parser.add_argument(
        "--output-directory",
        default="result/reachability_audit",
        help="JSON/Markdownを保存するdirectory。",
    )
    return parser.parse_args()


def main():
    base_directory = Path(__file__).resolve().parent
    arguments = parse_arguments()
    source_values = arguments.source or DEFAULT_SOURCE_RELATIVE_PATHS
    source_directories = [
        (base_directory / value).resolve() if not Path(value).is_absolute() else Path(value)
        for value in source_values
    ]
    output_directory = Path(arguments.output_directory)
    if not output_directory.is_absolute():
        output_directory = base_directory / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)

    report = make_report(base_directory, source_directories)
    json_path = output_directory / "reachability_report.json"
    markdown_path = output_directory / "reachability_report.md"
    json_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    markdown_path.write_text(report_as_markdown(report), encoding="utf-8")

    print(f"unique target cells: {report['num_unique_target_cells']}")
    print(f"all collision-free: {report['all_witnesses_collision_free']}")
    print(f"json: {json_path}")
    print(f"markdown: {markdown_path}")


if __name__ == "__main__":
    main()
