import json
import os
from collections import Counter
from pathlib import Path
import shutil
import stat
import time

import matplotlib

# GUIを持たない実行環境でもpngを保存できるbackendを明示する。
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from descriptor import (
    same_terminal_shape,
    state_coordinate_descriptor,
    terminal_descriptor,
)
from state import get_path_state_ids


# OneDrive上の読み取り専用属性を外してから、古い結果を削除する。
def _remove_readonly(function, path, error_info):
    os.chmod(path, stat.S_IWRITE)
    function(path)


# 同じseedの古い結果を消し、空の出力フォルダを作る。
def reset_output_directory(output_directory):
    output_directory = Path(output_directory)
    if output_directory.exists():
        for attempt in range(10):
            try:
                shutil.rmtree(output_directory, onerror=_remove_readonly)
                break
            except OSError:
                if attempt == 9:
                    raise
                time.sleep(0.2)
    output_directory.mkdir(parents=True)


# 各elite cellの代表経路を、時刻ごとの節点座標datとして保存する。
def save_elite_paths(history, elite_archive, output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    for cell_index, cell in enumerate(elite_archive):
        cell_directory = output_directory / f"cell[{cell_index}]"
        cell_directory.mkdir(exist_ok=True)
        path_state_ids = get_path_state_ids(
            history,
            cell["terminal_state_id"],
        )

        for step, state_id in enumerate(path_state_ids):
            np.savetxt(
                cell_directory / f"{step}.dat",
                history[state_id].node,
                fmt="%.8f",
            )


# 最終frontier archiveの各cellについて、代表状態の節点座標を保存する。
def save_frontier_states(history, frontier_archive, output_directory):
    frontier_directory = Path(output_directory) / "frontier"
    frontier_directory.mkdir(parents=True, exist_ok=True)

    for cell_index, cell in enumerate(frontier_archive):
        np.savetxt(
            frontier_directory / f"{cell_index}.dat",
            history[cell["state_id"]].node,
            fmt="%.8f",
        )


# mechanics step数に対する発見済みelite cell数をdatとpngで保存する。
def save_discovery_curve(discovery_curve, output_directory):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    discovery_curve = np.asarray(discovery_curve, dtype=int)
    np.savetxt(
        output_directory / "discovery_curve.dat",
        discovery_curve,
        fmt="%d",
        header="mechanics_step number_of_cells",
    )

    plt.figure(figsize=(6, 4))
    plt.step(discovery_curve[:, 0], discovery_curve[:, 1], where="post")
    plt.xlabel("Mechanics step")
    plt.ylabel("Number of discovered cells")
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_directory / "discovery_curve.png", dpi=200)
    plt.close()


def _cluster_state_ids(
    history,
    state_ids,
    reference_point,
    length_scale,
    rms_tolerance,
    max_tolerance,
):
    """探索時のfrontierと同じ固定prototype規則で状態をcell分けする。"""

    cells = []
    prototype_descriptors = []

    for state_id in state_ids:
        descriptor = state_coordinate_descriptor(
            history[state_id].node,
            reference_point,
            length_scale,
        )

        cell_index = None
        if prototype_descriptors:
            prototypes = np.asarray(prototype_descriptors).reshape(
                -1,
                history[state_id].node.shape[0],
                3,
            )
            difference = prototypes - descriptor.reshape(1, -1, 3)
            node_distance = np.linalg.norm(difference, axis=2)
            rms_distance = np.sqrt(np.mean(node_distance**2, axis=1))
            max_distance = np.max(node_distance, axis=1)
            matching_cells = np.flatnonzero(
                (rms_distance <= rms_tolerance)
                & (max_distance <= max_tolerance)
            )
            if len(matching_cells) > 0:
                cell_index = int(matching_cells[0])

        if cell_index is None:
            cells.append(
                {
                    "state_descriptor": descriptor,
                    "prototype_state_id": state_id,
                    "state_ids": [state_id],
                }
            )
            prototype_descriptors.append(descriptor)
        else:
            cells[cell_index]["state_ids"].append(state_id)

    return cells


def _classify_flat_states(
    history,
    elite_archive,
    member,
    length_scale,
    layer_threshold,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
    rms_tolerance,
    max_tolerance,
):
    """全flat終端を、探索終了時のelite cellへ割り当てる。"""

    terminal_cell_index = {}

    for state in history:
        if state.terminal_reason != "flat":
            continue

        shape_descriptor, layer_key = terminal_descriptor(
            history,
            state.state_id,
            member,
            length_scale,
            layer_threshold,
            overlap_tolerance,
            separation_tolerance,
            collinearity_tolerance,
            parallel_tolerance,
        )

        if any(upper_member == -1 for _, _, upper_member in layer_key):
            continue

        for cell_index, cell in enumerate(elite_archive):
            if layer_key != cell["layer_key"]:
                continue
            if same_terminal_shape(
                shape_descriptor,
                cell["shape_descriptor"],
                rms_tolerance,
                max_tolerance,
            ):
                terminal_cell_index[state.state_id] = cell_index
                break

    return terminal_cell_index


def _build_display_cells(
    history,
    elite_archive,
    terminal_cell_index,
    reference_point,
    length_scale,
    state_rms_tolerance,
    state_max_tolerance,
):
    """全StateNodeを、図上のいずれか一つのcellへ割り当てる。"""

    display_cells = [
        {
            "kind": "root",
            "cell_index": 0,
            "label": "root",
            "prototype_state_id": 0,
            "state_ids": [0],
        }
    ]
    display_cell_of_state = {0: 0}

    # 非終端状態からfrontier cellを一括再構成する。
    frontier_state_ids = [
        state.state_id
        for state in history
        if state.state_id != 0 and state.terminal_reason is None
    ]
    frontier_cells = _cluster_state_ids(
        history,
        frontier_state_ids,
        reference_point,
        length_scale,
        state_rms_tolerance,
        state_max_tolerance,
    )
    for cell_index, cell in enumerate(frontier_cells):
        display_cell_id = len(display_cells)
        display_cells.append(
            {
                "kind": "frontier",
                "cell_index": cell_index,
                "label": f"F[{cell_index}]",
                "prototype_state_id": cell["prototype_state_id"],
                "state_ids": cell["state_ids"],
            }
        )
        for state_id in cell["state_ids"]:
            display_cell_of_state[state_id] = display_cell_id

    # 重なり順まで判別できたflat終端をelite cellへ入れる。
    for cell_index, elite_cell in enumerate(elite_archive):
        state_ids = sorted(
            state_id
            for state_id, index in terminal_cell_index.items()
            if index == cell_index
        )
        if not state_ids:
            continue

        display_cell_id = len(display_cells)
        display_cells.append(
            {
                "kind": "elite",
                "cell_index": cell_index,
                "label": f"E[{cell_index}]",
                "prototype_state_id": elite_cell["terminal_state_id"],
                "state_ids": state_ids,
            }
        )
        for state_id in state_ids:
            display_cell_of_state[state_id] = display_cell_id

    # 重なり順を判別できないflat終端も、状態距離でまとめる。
    unclassified_flat_ids = [
        state.state_id
        for state in history
        if state.terminal_reason == "flat"
        and state.state_id not in terminal_cell_index
    ]
    unclassified_flat_cells = _cluster_state_ids(
        history,
        unclassified_flat_ids,
        reference_point,
        length_scale,
        state_rms_tolerance,
        state_max_tolerance,
    )
    for cell_index, cell in enumerate(unclassified_flat_cells):
        display_cell_id = len(display_cells)
        display_cells.append(
            {
                "kind": "unclassified_flat",
                "cell_index": cell_index,
                "label": f"U[{cell_index}]",
                "prototype_state_id": cell["prototype_state_id"],
                "state_ids": cell["state_ids"],
            }
        )
        for state_id in cell["state_ids"]:
            display_cell_of_state[state_id] = display_cell_id

    # 衝突終端も、状態距離でまとめる。
    collision_state_ids = [
        state.state_id
        for state in history
        if state.terminal_reason == "collision"
    ]
    collision_cells = _cluster_state_ids(
        history,
        collision_state_ids,
        reference_point,
        length_scale,
        state_rms_tolerance,
        state_max_tolerance,
    )
    for cell_index, cell in enumerate(collision_cells):
        display_cell_id = len(display_cells)
        display_cells.append(
            {
                "kind": "collision",
                "cell_index": cell_index,
                "label": f"X[{cell_index}]",
                "prototype_state_id": cell["prototype_state_id"],
                "state_ids": cell["state_ids"],
            }
        )
        for state_id in cell["state_ids"]:
            display_cell_of_state[state_id] = display_cell_id

    return display_cells, display_cell_of_state


def _ordered_five_bar_chain(member, number_of_nodes):
    """five-barが一本の鎖なら、端から端までの節点番号を返す。"""

    if number_of_nodes != 5 or len(member) != 4:
        return None

    neighbours = [[] for _ in range(number_of_nodes)]
    for node_i, node_j in np.asarray(member, dtype=int):
        neighbours[node_i].append(node_j)
        neighbours[node_j].append(node_i)

    end_nodes = [i for i, adjacent in enumerate(neighbours) if len(adjacent) == 1]
    if len(end_nodes) != 2 or any(len(adjacent) > 2 for adjacent in neighbours):
        return None

    # 節点番号が小さい側を始点にして、角度の符号と軸順を固定する。
    chain = [min(end_nodes)]
    previous = None
    current = chain[0]
    while len(chain) < number_of_nodes:
        candidates = [node for node in neighbours[current] if node != previous]
        if len(candidates) != 1:
            return None
        next_node = candidates[0]
        chain.append(next_node)
        previous, current = current, next_node

    if chain[-1] != max(end_nodes):
        return None
    return chain


def _five_bar_joint_angles(history, member):
    """節点1～3の符号付き内角を、treeに沿って連続化して返す。"""

    chain = _ordered_five_bar_chain(member, history[0].node.shape[0])
    if chain is None:
        return None, None

    # 三角だけで形状を記述できるのは、両端が固定された平面鎖の場合。
    end_nodes = [chain[0], chain[-1]]
    root_end_positions = history[0].node[end_nodes]
    endpoints_are_fixed = all(
        np.allclose(state.node[end_nodes], root_end_positions)
        for state in history
    )
    chain_is_in_xz_plane = all(
        np.ptp(state.node[:, 1]) <= 1.0e-10 for state in history
    )
    if not endpoints_are_fixed or not chain_is_in_xz_plane:
        return None, None

    raw_angles = np.zeros((len(history), 3))
    internal_nodes = chain[1:-1]

    for state in history:
        node = state.node
        for angle_index, center_node in enumerate(internal_nodes):
            previous_node = chain[angle_index]
            next_node = chain[angle_index + 2]

            # 中心節点から両隣へ向かう二部材をx-z平面へ射影する。
            vector_previous = node[previous_node, [0, 2]] - node[center_node, [0, 2]]
            vector_next = node[next_node, [0, 2]] - node[center_node, [0, 2]]
            cross = (
                vector_previous[0] * vector_next[1]
                - vector_previous[1] * vector_next[0]
            )
            dot = np.dot(vector_previous, vector_next)
            raw_angles[state.state_id, angle_index] = np.arctan2(cross, dot)

    # atan2の+-pi境界で枝が飛ばないよう、各childをparentに最も近い角度へ移す。
    continuous_angles = np.zeros_like(raw_angles)
    continuous_angles[0] = raw_angles[0]
    for state in history[1:]:
        parent_angles = continuous_angles[state.parent_id]
        angle_difference = (
            raw_angles[state.state_id] - parent_angles + np.pi
        ) % (2.0 * np.pi) - np.pi
        continuous_angles[state.state_id] = parent_angles + angle_difference

    angle_labels = [f"theta_{node_id} [deg]" for node_id in internal_nodes]
    return np.degrees(continuous_angles), angle_labels


def _ellipsoid_mesh(points, padding):
    """三次元点群をゆるく囲む、塗りつぶし用の楕円体meshを返す。"""

    points = np.asarray(points, dtype=float)
    center = np.mean(points, axis=0)
    centered = points - center

    if len(points) > 1:
        covariance = centered.T @ centered / len(points)
        _, axes = np.linalg.eigh(covariance)
        projected = centered @ axes
        radii = np.max(np.abs(projected), axis=0) + padding
    else:
        axes = np.eye(3)
        radii = np.full(3, padding)

    longitude_count = 12
    latitude_interval_count = 6

    # 北極、緯度ring、南極の順に頂点を作る。
    local_vertices = [[0.0, 0.0, radii[2]]]
    for latitude_index in range(1, latitude_interval_count):
        latitude = np.pi * latitude_index / latitude_interval_count
        for longitude_index in range(longitude_count):
            longitude = 2.0 * np.pi * longitude_index / longitude_count
            local_vertices.append(
                [
                    radii[0] * np.sin(latitude) * np.cos(longitude),
                    radii[1] * np.sin(latitude) * np.sin(longitude),
                    radii[2] * np.cos(latitude),
                ]
            )
    local_vertices.append([0.0, 0.0, -radii[2]])
    vertices = center + np.asarray(local_vertices) @ axes.T

    face_i = []
    face_j = []
    face_k = []

    # 北極と最初の緯度ringをつなぐ。
    first_ring_start = 1
    for longitude_index in range(longitude_count):
        next_longitude = (longitude_index + 1) % longitude_count
        face_i.append(0)
        face_j.append(first_ring_start + longitude_index)
        face_k.append(first_ring_start + next_longitude)

    # 隣り合う緯度ring間の四角を、二つの三角形に分ける。
    number_of_rings = latitude_interval_count - 1
    for ring_index in range(number_of_rings - 1):
        upper_ring_start = 1 + ring_index * longitude_count
        lower_ring_start = upper_ring_start + longitude_count
        for longitude_index in range(longitude_count):
            next_longitude = (longitude_index + 1) % longitude_count
            upper = upper_ring_start + longitude_index
            upper_next = upper_ring_start + next_longitude
            lower = lower_ring_start + longitude_index
            lower_next = lower_ring_start + next_longitude
            face_i.extend((upper, upper_next))
            face_j.extend((lower, lower))
            face_k.extend((upper_next, lower_next))

    # 最後の緯度ringと南極をつなぐ。
    south_pole = len(vertices) - 1
    last_ring_start = south_pole - longitude_count
    for longitude_index in range(longitude_count):
        next_longitude = (longitude_index + 1) % longitude_count
        face_i.append(last_ring_start + longitude_index)
        face_j.append(south_pole)
        face_k.append(last_ring_start + next_longitude)

    return vertices, face_i, face_j, face_k


def _joined_coordinates(lines):
    """複数の三次元線を、Plotlyの一つのline traceへまとめる。"""

    x_values = []
    y_values = []
    z_values = []
    for line in lines:
        line = np.asarray(line)
        x_values.extend(line[:, 0].tolist())
        y_values.extend(line[:, 1].tolist())
        z_values.extend(line[:, 2].tolist())
        x_values.append(None)
        y_values.append(None)
        z_values.append(None)
    return x_values, y_values, z_values


def _edge_lines(history, coordinates, random_edge_set):
    deterministic_lines = []
    random_lines = []
    for state in history[1:]:
        edge = (state.parent_id, state.state_id)
        line = coordinates[[state.parent_id, state.state_id]]
        if edge in random_edge_set:
            random_lines.append(line)
        else:
            deterministic_lines.append(line)
    return deterministic_lines, random_lines


def _cell_color(cell, perturbed, elite_colors):
    if cell["kind"] == "root":
        return "#7b61a8"
    if cell["kind"] == "frontier":
        return "#d62728" if perturbed else "#75bfe8"
    if cell["kind"] == "elite":
        return elite_colors[cell["cell_index"] % len(elite_colors)]
    if cell["kind"] == "unclassified_flat":
        return "#e68613"
    return "#7f7f7f"


def _json_default(value):
    """numpyの数値だけを通常のPython数値へ変換する。"""

    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _interactive_tree_traces(
    history,
    display_cells,
    display_cell_of_state,
    coordinates,
    random_edge_set,
    elite_archive,
    terminal_cell_index,
    perturbation_count_of_state,
):
    """Plotlyへ渡す三次元探索木のtrace群を作る。"""

    traces = []
    elite_colors = [
        "#2ca02c",
        "#9467bd",
        "#17becf",
        "#bcbd22",
        "#e377c2",
        "#8c564b",
    ]
    extent = np.ptp(coordinates, axis=0)
    cloud_padding = max(1.0e-6, 0.008 * max(np.linalg.norm(extent), 1.0))

    # cellを表す半透明の楕円体。面にマウスを置くとcell情報を表示する。
    shown_legend_groups = set()
    for cell in display_cells:
        perturbed = (
            cell["kind"] in ("root", "frontier")
            and cell["perturbation_count"] > 0
        )
        if cell["kind"] == "frontier" and perturbed:
            legend_group = "perturbed_frontier"
            legend_name = "perturbed frontier cell"
        else:
            legend_group = cell["kind"]
            legend_name = {
                "root": "root cell",
                "frontier": "frontier cell",
                "elite": "elite cell",
                "unclassified_flat": "unclassified flat cell",
                "collision": "collision cell",
            }[cell["kind"]]

        vertices, face_i, face_j, face_k = _ellipsoid_mesh(
            coordinates[cell["state_ids"]],
            cloud_padding,
        )
        cell_information = (
            f'{cell["label"]}<br>'
            f'states={len(cell["state_ids"])}<br>'
            f'perturbations={cell["perturbation_count"]}<br>'
            f'prototype state={cell["prototype_state_id"]}'
        )
        traces.append(
            {
                "type": "mesh3d",
                "x": vertices[:, 0].tolist(),
                "y": vertices[:, 1].tolist(),
                "z": vertices[:, 2].tolist(),
                "i": face_i,
                "j": face_j,
                "k": face_k,
                "name": legend_name,
                "legendgroup": legend_group,
                "showlegend": legend_group not in shown_legend_groups,
                "color": _cell_color(cell, perturbed, elite_colors),
                "flatshading": False,
                "lighting": {
                    "ambient": 0.90,
                    "diffuse": 0.35,
                    "specular": 0.05,
                    "roughness": 1.0,
                    "fresnel": 0.0,
                },
                "opacity": (
                    0.22
                    if perturbed or cell["kind"] == "elite"
                    else 0.10
                ),
                "meta": cell_information,
                "hovertemplate": "%{meta}<extra></extra>",
            }
        )
        shown_legend_groups.add(legend_group)

    deterministic_lines, random_lines = _edge_lines(
        history,
        coordinates,
        random_edge_set,
    )
    if deterministic_lines:
        x_values, y_values, z_values = _joined_coordinates(deterministic_lines)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": x_values,
                "y": y_values,
                "z": z_values,
                "name": "deterministic step",
                "line": {"color": "#202020", "width": 2.0},
                "opacity": 0.58,
                "hoverinfo": "skip",
            }
        )
    if random_lines:
        x_values, y_values, z_values = _joined_coordinates(random_lines)
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": x_values,
                "y": y_values,
                "z": z_values,
                "name": "random step",
                "line": {"color": "#d62728", "width": 5.0},
                "opacity": 0.90,
                "hoverinfo": "skip",
            }
        )

    state_information = []
    for state in history:
        display_cell = display_cells[display_cell_of_state[state.state_id]]
        parent_id = -1 if state.parent_id is None else state.parent_id
        incoming_random = int((parent_id, state.state_id) in random_edge_set)
        state_information.append(
            [
                state.state_id,
                parent_id,
                state.depth,
                display_cell["label"],
                "nonterminal" if state.terminal_reason is None else state.terminal_reason,
                incoming_random,
                perturbation_count_of_state[state.state_id],
            ]
        )

    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": coordinates[:, 0].tolist(),
            "y": coordinates[:, 1].tolist(),
            "z": coordinates[:, 2].tolist(),
            "name": "StateNode",
            "marker": {"size": 2.8, "color": "#111111", "opacity": 0.82},
            "customdata": state_information,
            "hovertemplate": (
                "state=%{customdata[0]}<br>"
                "parent=%{customdata[1]}<br>"
                "depth=%{customdata[2]}<br>"
                "cell=%{customdata[3]}<br>"
                "terminal=%{customdata[4]}<br>"
                "incoming random=%{customdata[5]}<br>"
                "outgoing perturbations=%{customdata[6]}<br>"
                "theta_1=%{x:.2f} deg<br>"
                "theta_2=%{y:.2f} deg<br>"
                "theta_3=%{z:.2f} deg"
                "<extra></extra>"
            ),
        }
    )

    traces.append(
        {
            "type": "scatter3d",
            "mode": "markers",
            "x": [float(coordinates[0, 0])],
            "y": [float(coordinates[0, 1])],
            "z": [float(coordinates[0, 2])],
            "name": "root",
            "marker": {
                "size": 8,
                "color": "#1f77b4",
                "line": {"color": "#111111", "width": 1},
            },
            "hovertemplate": "root state 0<extra></extra>",
        }
    )

    selected_state_ids = sorted(perturbation_count_of_state)
    if selected_state_ids:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": coordinates[selected_state_ids, 0].tolist(),
                "y": coordinates[selected_state_ids, 1].tolist(),
                "z": coordinates[selected_state_ids, 2].tolist(),
                "name": "perturbed state",
                "marker": {
                    "size": [
                        5.5
                        + 2.5
                        * np.log1p(perturbation_count_of_state[state_id])
                        for state_id in selected_state_ids
                    ],
                    "symbol": "circle-open",
                    "color": "#d62728",
                    "line": {"color": "#d62728", "width": 2},
                },
                "customdata": [
                    [state_id, perturbation_count_of_state[state_id]]
                    for state_id in selected_state_ids
                ],
                "hovertemplate": (
                    "perturbed state=%{customdata[0]}<br>"
                    "count=%{customdata[1]}<extra></extra>"
                ),
            }
        )

    for cell_index, elite_cell in enumerate(elite_archive):
        state_ids = [
            state_id
            for state_id, index in terminal_cell_index.items()
            if index == cell_index
        ]
        if not state_ids:
            continue
        color = elite_colors[cell_index % len(elite_colors)]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": coordinates[state_ids, 0].tolist(),
                "y": coordinates[state_ids, 1].tolist(),
                "z": coordinates[state_ids, 2].tolist(),
                "name": f"elite cell E[{cell_index}]",
                "marker": {
                    "size": 5.5,
                    "color": color,
                    "line": {"color": "#111111", "width": 0.8},
                },
                "customdata": state_ids,
                "hovertemplate": (
                    f"E[{cell_index}] terminal state=%{{customdata}}"
                    "<extra></extra>"
                ),
            }
        )
        representative_state_id = elite_cell["terminal_state_id"]
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": [float(coordinates[representative_state_id, 0])],
                "y": [float(coordinates[representative_state_id, 1])],
                "z": [float(coordinates[representative_state_id, 2])],
                "name": f"E[{cell_index}] representative",
                "marker": {
                    "size": 9,
                    "symbol": "diamond",
                    "color": color,
                    "line": {"color": "#111111", "width": 1.2},
                },
                "hovertemplate": (
                    f"E[{cell_index}] representative state="
                    f"{representative_state_id}<extra></extra>"
                ),
            }
        )

    unclassified_flat_ids = [
        state.state_id
        for state in history
        if state.terminal_reason == "flat"
        and state.state_id not in terminal_cell_index
    ]
    if unclassified_flat_ids:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": coordinates[unclassified_flat_ids, 0].tolist(),
                "y": coordinates[unclassified_flat_ids, 1].tolist(),
                "z": coordinates[unclassified_flat_ids, 2].tolist(),
                "name": "unclassified flat",
                "marker": {"size": 6, "symbol": "diamond", "color": "#e68613"},
                "customdata": unclassified_flat_ids,
                "hovertemplate": "flat state=%{customdata}<extra></extra>",
            }
        )

    collision_state_ids = [
        state.state_id
        for state in history
        if state.terminal_reason == "collision"
    ]
    if collision_state_ids:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": coordinates[collision_state_ids, 0].tolist(),
                "y": coordinates[collision_state_ids, 1].tolist(),
                "z": coordinates[collision_state_ids, 2].tolist(),
                "name": "collision",
                "marker": {"size": 5.5, "symbol": "x", "color": "#666666"},
                "customdata": collision_state_ids,
                "hovertemplate": "collision state=%{customdata}<extra></extra>",
            }
        )

    return traces


def _save_interactive_three_dimensional_tree(
    history,
    display_cells,
    display_cell_of_state,
    coordinates,
    random_edge_set,
    elite_archive,
    terminal_cell_index,
    perturbation_count_of_state,
    angle_labels,
    output_directory,
):
    """マウスで回転・移動・拡大できる三次元探索木をHTMLで保存する。"""

    traces = _interactive_tree_traces(
        history,
        display_cells,
        display_cell_of_state,
        coordinates,
        random_edge_set,
        elite_archive,
        terminal_cell_index,
        perturbation_count_of_state,
    )
    layout = {
        "title": {
            "text": (
                "Search tree in three signed joint angles"
                f"<br><sup>{len(history)} states, "
                f"{len(elite_archive)} elite cells</sup>"
            ),
            "x": 0.5,
        },
        "template": "plotly_white",
        "margin": {"l": 10, "r": 10, "b": 10, "t": 80},
        "legend": {"x": 0.01, "y": 0.99},
        "hovermode": "closest",
        "scene": {
            "xaxis": {"title": angle_labels[0]},
            "yaxis": {"title": angle_labels[1]},
            "zaxis": {"title": angle_labels[2]},
            "aspectmode": "data",
            "dragmode": "orbit",
            "camera": {"eye": {"x": 1.45, "y": -1.65, "z": 1.10}},
        },
    }
    config = {
        "responsive": True,
        "scrollZoom": True,
        "displaylogo": False,
        "modeBarButtonsToRemove": ["toImage"],
    }

    # Python側にPlotlyを追加インストールせず、ブラウザ側だけで描画する。
    html_document = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Five-bar linkage search tree</title>
<script src="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"></script>
<style>
html, body, #search-tree {{ width: 100%; height: 100%; margin: 0; }}
body {{ overflow: hidden; }}
</style>
</head>
<body>
<div id="search-tree"></div>
<script>
const traces = {json.dumps(traces, ensure_ascii=False, separators=(",", ":"), default=_json_default, allow_nan=False)};
const layout = {json.dumps(layout, ensure_ascii=False, separators=(",", ":"), default=_json_default, allow_nan=False)};
const config = {json.dumps(config, ensure_ascii=False, separators=(",", ":"), default=_json_default, allow_nan=False)};
Plotly.newPlot("search-tree", traces, layout, config);
</script>
</body>
</html>
"""
    (Path(output_directory) / "search_tree.html").write_text(
        html_document,
        encoding="utf-8",
    )


def save_search_tree(
    history,
    elite_archive,
    random_edges,
    output_directory,
    member,
    reference_point,
    length_scale,
    state_rms_tolerance,
    state_max_tolerance,
    layer_threshold,
    overlap_tolerance,
    separation_tolerance,
    collinearity_tolerance,
    parallel_tolerance,
    elite_rms_tolerance,
    elite_max_tolerance,
):
    """三折角空間上の探索木、cell領域、摂動回数を探索終了後に保存する。"""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    random_edge_set = {
        (int(parent_id), int(child_id)) for parent_id, child_id in random_edges
    }

    # 一つのstateから出るdeterministic childは高々一つであることを確認する。
    for state in history:
        number_of_deterministic_children = sum(
            (state.state_id, child_id) not in random_edge_set
            for child_id in state.children_ids
        )
        if number_of_deterministic_children > 1:
            raise ValueError(
                f"state {state.state_id} has more than one deterministic child"
            )

    # cell分類はmain loop中には行わず、探索終了後にhistory全体から再構成する。
    terminal_cell_index = _classify_flat_states(
        history,
        elite_archive,
        member,
        length_scale,
        layer_threshold,
        overlap_tolerance,
        separation_tolerance,
        collinearity_tolerance,
        parallel_tolerance,
        elite_rms_tolerance,
        elite_max_tolerance,
    )
    display_cells, display_cell_of_state = _build_display_cells(
        history,
        elite_archive,
        terminal_cell_index,
        reference_point,
        length_scale,
        state_rms_tolerance,
        state_max_tolerance,
    )

    perturbation_count_of_state = Counter(
        int(parent_id) for parent_id, _ in random_edges
    )
    for cell in display_cells:
        cell["perturbation_count"] = sum(
            perturbation_count_of_state[state_id]
            for state_id in cell["state_ids"]
        )

    coordinates, angle_labels = _five_bar_joint_angles(history, member)
    if coordinates is None:
        raise ValueError(
            "The interactive search tree requires a planar five-bar linkage "
            "with both end nodes fixed."
        )

    _save_interactive_three_dimensional_tree(
        history,
        display_cells,
        display_cell_of_state,
        coordinates,
        random_edge_set,
        elite_archive,
        terminal_cell_index,
        perturbation_count_of_state,
        angle_labels,
        output_directory,
    )
