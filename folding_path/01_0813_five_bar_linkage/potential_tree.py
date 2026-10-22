"""Persist and visualize a five-bar search tree by gravity potential.

The mechanics code already caches the gravity potential on every ``StateNode``.
This module writes a compact, machine-readable copy of the complete tree and a
Plotly HTML view whose three axes are the unwrapped signed joint angles.  The
HTML embeds all tree data; only the Plotly JavaScript library is loaded from a
CDN when the file is opened.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


_PLOTLY_CDN = (
    "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"
)


def _ordered_five_bar_chain(member, number_of_nodes):
    """Return the node IDs along an open five-node/four-member chain."""
    member = np.asarray(member, dtype=int)
    if number_of_nodes != 5 or member.shape != (4, 2):
        return None

    neighbours = [[] for _ in range(number_of_nodes)]
    for node_i, node_j in member:
        if not (
            0 <= int(node_i) < number_of_nodes
            and 0 <= int(node_j) < number_of_nodes
        ):
            return None
        neighbours[int(node_i)].append(int(node_j))
        neighbours[int(node_j)].append(int(node_i))

    end_nodes = [
        node_id
        for node_id, adjacent in enumerate(neighbours)
        if len(adjacent) == 1
    ]
    if len(end_nodes) != 2 or any(len(adjacent) > 2 for adjacent in neighbours):
        return None

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
    if chain[-1] != max(end_nodes) or len(set(chain)) != number_of_nodes:
        return None
    return chain


def _validate_and_pack_history(history):
    """Convert a StateNode history into dense arrays and check tree invariants."""
    if not history:
        raise ValueError("cannot save an empty state history")

    number_of_states = len(history)
    root_shape = np.asarray(history[0].node).shape
    if len(root_shape) != 2 or root_shape[1] != 3:
        raise ValueError("each state node must have shape (number_of_nodes, 3)")

    nodes = np.empty((number_of_states, *root_shape), dtype=np.float64)
    state_ids = np.arange(number_of_states, dtype=np.int32)
    parent_ids = np.full(number_of_states, -1, dtype=np.int32)
    depths = np.empty(number_of_states, dtype=np.int32)
    potentials = np.empty(number_of_states, dtype=np.float64)
    child_counts = np.empty(number_of_states, dtype=np.int32)
    terminal_reasons = []
    incoming_step_types = []

    for expected_id, state in enumerate(history):
        if int(state.state_id) != expected_id:
            raise ValueError(
                "history must be ordered so state_id equals its list index"
            )
        node = np.asarray(state.node, dtype=np.float64)
        if node.shape != root_shape:
            raise ValueError("all states must contain the same number of nodes")
        if not np.all(np.isfinite(node)):
            raise ValueError(f"state {expected_id} contains non-finite coordinates")

        if state.parent_id is None:
            if expected_id != 0:
                raise ValueError("only state 0 may have no parent")
            parent_id = -1
        else:
            parent_id = int(state.parent_id)
            if not 0 <= parent_id < expected_id:
                raise ValueError(
                    f"state {expected_id} has invalid parent {parent_id}"
                )

        potential = float(state.potential)
        if not np.isfinite(potential):
            raise ValueError(f"state {expected_id} has non-finite potential")

        nodes[expected_id] = node
        parent_ids[expected_id] = parent_id
        depths[expected_id] = int(state.depth)
        potentials[expected_id] = potential
        child_counts[expected_id] = len(state.children_ids)
        terminal_reasons.append(
            "nonterminal"
            if state.terminal_reason is None
            else str(state.terminal_reason)
        )
        incoming_step_types.append(
            "root"
            if state.incoming_step_type is None
            else str(state.incoming_step_type)
        )

    return {
        "nodes": nodes,
        "state_id": state_ids,
        "parent_id": parent_ids,
        "depth": depths,
        "potential": potentials,
        "child_count": child_counts,
        "terminal_reason": np.asarray(terminal_reasons, dtype="U32"),
        "incoming_step_type": np.asarray(incoming_step_types, dtype="U24"),
    }


def _joint_angles(nodes, parent_ids, member):
    """Calculate parent-continuous signed joint angles in degrees."""
    chain = _ordered_five_bar_chain(member, nodes.shape[1])
    if chain is None:
        raise ValueError(
            "the potential tree requires a five-node open chain with four members"
        )

    end_nodes = [chain[0], chain[-1]]
    if not np.allclose(nodes[:, end_nodes], nodes[0, end_nodes]):
        raise ValueError(
            "the potential tree requires both end nodes to remain fixed"
        )
    if np.any(np.ptp(nodes[:, :, 1], axis=1) > 1.0e-10):
        raise ValueError("the potential tree requires configurations in the x-z plane")

    raw = np.zeros((len(nodes), 3), dtype=np.float64)
    internal_nodes = chain[1:-1]
    for angle_index, center_node in enumerate(internal_nodes):
        previous_node = chain[angle_index]
        next_node = chain[angle_index + 2]
        vector_previous = (
            nodes[:, previous_node][:, [0, 2]]
            - nodes[:, center_node][:, [0, 2]]
        )
        vector_next = (
            nodes[:, next_node][:, [0, 2]]
            - nodes[:, center_node][:, [0, 2]]
        )
        cross = (
            vector_previous[:, 0] * vector_next[:, 1]
            - vector_previous[:, 1] * vector_next[:, 0]
        )
        dot = np.sum(vector_previous * vector_next, axis=1)
        raw[:, angle_index] = np.arctan2(cross, dot)

    continuous = np.zeros_like(raw)
    continuous[0] = raw[0]
    for state_id in range(1, len(nodes)):
        parent_id = int(parent_ids[state_id])
        difference = (
            raw[state_id] - continuous[parent_id] + np.pi
        ) % (2.0 * np.pi) - np.pi
        continuous[state_id] = continuous[parent_id] + difference

    labels = [f"theta_{node_id} [deg]" for node_id in internal_nodes]
    return np.degrees(continuous), np.asarray(chain, dtype=np.int16), labels


def _joined_edges(coordinates, parent_ids, incoming_step_types, target_type):
    """Return Plotly line coordinates joined by ``None`` separators."""
    xyz = [[], [], []]
    for state_id in range(1, len(coordinates)):
        if str(incoming_step_types[state_id]) != target_type:
            continue
        parent_id = int(parent_ids[state_id])
        for axis in range(3):
            xyz[axis].extend(
                (
                    float(coordinates[parent_id, axis]),
                    float(coordinates[state_id, axis]),
                    None,
                )
            )
    return xyz


def _json(value):
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )


def _potential_html(
    data,
    angles,
    angle_labels,
    reference_point,
    output_path,
):
    potentials = np.asarray(data["potential"], dtype=np.float64)
    root_scale = max(abs(float(potentials[0])), 1.0e-15)
    normalized = potentials / root_scale
    positive = potentials[potentials > 0.0]
    if positive.size:
        log_floor = max(
            float(np.min(positive)) * 0.1,
            float(np.max(positive)) * 1.0e-12,
            1.0e-15,
        )
    else:
        log_floor = 1.0e-15
    log_potential = np.log10(np.maximum(potentials, log_floor))

    deterministic_xyz = _joined_edges(
        angles,
        data["parent_id"],
        data["incoming_step_type"],
        "deterministic",
    )
    perturbation_xyz = _joined_edges(
        angles,
        data["parent_id"],
        data["incoming_step_type"],
        "perturbation",
    )

    traces = []
    if deterministic_xyz[0]:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": deterministic_xyz[0],
                "y": deterministic_xyz[1],
                "z": deterministic_xyz[2],
                "name": "deterministic edge",
                "line": {"color": "#888888", "width": 1.5},
                "opacity": 0.45,
                "hoverinfo": "skip",
            }
        )
    if perturbation_xyz[0]:
        traces.append(
            {
                "type": "scatter3d",
                "mode": "lines",
                "x": perturbation_xyz[0],
                "y": perturbation_xyz[1],
                "z": perturbation_xyz[2],
                "name": "persistent perturbation edge",
                "line": {"color": "#d1495b", "width": 2.5},
                "opacity": 0.72,
                "hoverinfo": "skip",
            }
        )

    custom_data = [
        [
            int(data["state_id"][state_id]),
            int(data["parent_id"][state_id]),
            int(data["depth"][state_id]),
            float(potentials[state_id]),
            float(normalized[state_id]),
            str(data["incoming_step_type"][state_id]),
            str(data["terminal_reason"][state_id]),
            int(data["child_count"][state_id]),
        ]
        for state_id in range(len(potentials))
    ]
    hover_template = (
        "state=%{customdata[0]}<br>"
        "parent=%{customdata[1]}<br>"
        "depth=%{customdata[2]}<br>"
        "potential=%{customdata[3]:.7g}<br>"
        "potential/root=%{customdata[4]:.5g}<br>"
        "incoming=%{customdata[5]}<br>"
        "terminal=%{customdata[6]}<br>"
        "children=%{customdata[7]}<br>"
        f"{angle_labels[0]}=%{{x:.3f}}<br>"
        f"{angle_labels[1]}=%{{y:.3f}}<br>"
        f"{angle_labels[2]}=%{{z:.3f}}"
        "<extra></extra>"
    )

    # Keep gravity potential as the fill-color encoding for every node.  Root
    # and flat terminal states are split into overlay traces so that shape,
    # size, outline, and legend text identify them independently of color.
    terminal_reasons = np.asarray(data["terminal_reason"], dtype=str)
    root_mask = np.zeros(len(potentials), dtype=bool)
    root_mask[0] = True
    flat_mask = terminal_reasons == "flat"
    ordinary_mask = ~(root_mask | flat_mask)

    node_specs = [
        {
            "name": "sampled StateNode (circle)",
            "indices": np.flatnonzero(ordinary_mask),
            "symbol": "circle",
            "size": 3.0,
            "outline": "rgba(20,20,20,0.32)",
            "outline_width": 0.5,
            "opacity": 0.92,
        },
        {
            "name": "flat terminal (square; terminal_reason='flat')",
            "indices": np.flatnonzero(flat_mask),
            "symbol": "square",
            "size": 9.0,
            "outline": "#00876c",
            "outline_width": 2.6,
            "opacity": 1.0,
        },
        {
            "name": "root StateNode (diamond)",
            "indices": np.asarray([0], dtype=int),
            "symbol": "diamond",
            "size": 12.0,
            "outline": "#111111",
            "outline_width": 3.2,
            "opacity": 1.0,
        },
    ]
    colorbar_owner = next(
        spec for spec in node_specs if spec["indices"].size
    )
    node_trace_metadata = []
    colorbar_trace_index = None
    for spec in node_specs:
        indices = spec["indices"]
        if not indices.size:
            continue
        owns_colorbar = spec is colorbar_owner
        trace_index = len(traces)
        marker = {
            "size": spec["size"],
            "symbol": spec["symbol"],
            "color": potentials[indices].tolist(),
            "colorscale": "Cividis",
            "cmin": float(np.min(potentials)),
            "cmax": float(np.max(potentials)),
            "showscale": owns_colorbar,
            "opacity": spec["opacity"],
            "line": {
                "color": spec["outline"],
                "width": spec["outline_width"],
            },
        }
        if owns_colorbar:
            colorbar_trace_index = trace_index
            marker["colorbar"] = {
                "title": {"text": "U"},
                "thickness": 18,
                "len": 0.72,
                "tickformat": ".3g",
            }
        traces.append(
            {
                "type": "scatter3d",
                "mode": "markers",
                "x": angles[indices, 0].tolist(),
                "y": angles[indices, 1].tolist(),
                "z": angles[indices, 2].tolist(),
                "name": spec["name"],
                "marker": marker,
                "customdata": [custom_data[int(index)] for index in indices],
                "hovertemplate": hover_template,
            }
        )
        node_trace_metadata.append(
            {
                "trace_index": trace_index,
                "linear": potentials[indices].tolist(),
                "log": log_potential[indices].tolist(),
            }
        )

    title = (
        "Five-bar sampled configuration tree colored by gravity potential"
        f"<br><sup>{len(potentials):,} states; root=diamond; "
        f"flat terminals={int(np.count_nonzero(flat_mask))} squares; "
        "angles are unwrapped along each parent branch</sup>"
    )
    layout = {
        "title": {"text": title, "x": 0.5},
        "template": "plotly_white",
        "margin": {"l": 8, "r": 8, "b": 8, "t": 92},
        "legend": {
            "x": 0.01,
            "y": 0.01,
            "xanchor": "left",
            "yanchor": "bottom",
        },
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
    plot_config = {
        "responsive": True,
        "scrollZoom": True,
        "displaylogo": False,
    }
    summary = {
        "states": int(len(potentials)),
        "minimum": float(np.min(potentials)),
        "median": float(np.median(potentials)),
        "maximum": float(np.max(potentials)),
        "root": float(potentials[0]),
        "reference_point": float(reference_point),
        "flat_terminals": int(np.count_nonzero(flat_mask)),
    }

    html_document = f"""<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Five-bar gravity-potential tree</title>
<script src="{_PLOTLY_CDN}"></script>
<style>
html, body {{ width: 100%; height: 100%; margin: 0; font-family: sans-serif; }}
body {{ overflow: hidden; }}
#potential-tree {{ position: absolute; inset: 0; }}
#controls {{
  position: absolute; z-index: 10; top: 10px; left: 10px; padding: 9px 11px;
  border: 1px solid #c9c9c9; border-radius: 7px;
  background: rgba(255,255,255,0.92); box-shadow: 0 1px 4px rgba(0,0,0,0.12);
  font-size: 12px; line-height: 1.45;
}}
#controls button {{
  margin: 6px 3px 0 0; padding: 4px 9px; border: 1px solid #999;
  border-radius: 4px; background: white; cursor: pointer;
}}
#controls button.active {{ color: white; background: #3264a8; border-color: #3264a8; }}
</style>
</head>
<body>
<div id="potential-tree"></div>
<div id="controls">
  <div><b>Node color</b>: <span id="scale-label">U (linear)</span></div>
  <div>U=&Sigma;<sub>i</sub>|z<sub>i</sub>&minus;z<sub>ref</sub>|,
       z<sub>ref</sub>={summary['reference_point']:.7g}</div>
  <div>states={summary['states']:,}; min={summary['minimum']:.5g};
       median={summary['median']:.5g}; max={summary['maximum']:.5g}</div>
  <div><b>root</b>: black-outlined diamond;
       <b>flat terminal</b>: green-outlined square
       ({summary['flat_terminals']:,})</div>
  <button id="linear-button" class="active" onclick="setPotentialScale('linear')">Linear</button>
  <button id="log-button" onclick="setPotentialScale('log')">Log10</button>
</div>
<script>
const traces = {_json(traces)};
const layout = {_json(layout)};
const plotConfig = {_json(plot_config)};
const nodeTraceMetadata = {_json(node_trace_metadata)};
const colorbarTraceIndex = {colorbar_trace_index};
const linearRange = [{float(np.min(potentials))}, {float(np.max(potentials))}];
const logRange = [{float(np.min(log_potential))}, {float(np.max(log_potential))}];
const plot = document.getElementById("potential-tree");
Plotly.newPlot(plot, traces, layout, plotConfig);
function setPotentialScale(mode) {{
  const isLog = mode === "log";
  const range = isLog ? logRange : linearRange;
  const title = isLog ? "log10 U" : "U";
  nodeTraceMetadata.forEach((metadata) => {{
    const values = isLog ? metadata.log : metadata.linear;
    Plotly.restyle(plot, {{
      "marker.color": [values],
      "marker.cmin": [range[0]],
      "marker.cmax": [range[1]]
    }}, [metadata.trace_index]);
  }});
  Plotly.restyle(plot, {{
    "marker.colorbar.title.text": [title]
  }}, [colorbarTraceIndex]);
  document.getElementById("scale-label").textContent =
    isLog ? "log10 U" : "U (linear)";
  document.getElementById("linear-button").classList.toggle("active", !isLog);
  document.getElementById("log-button").classList.toggle("active", isLog);
}}
</script>
</body>
</html>
"""
    output_path.write_text(html_document, encoding="utf-8")


def save_potential_tree(
    history,
    output_directory,
    member,
    reference_point,
):
    """Save ``state_tree.npz`` and ``potential_tree.html`` for one run.

    Potentials used for coloring come from the value cached on each StateNode.
    They are independently recomputed once here so the NPZ records the maximum
    cache discrepancy as an integrity check.
    """
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    data = _validate_and_pack_history(history)
    angles, chain, angle_labels = _joint_angles(
        data["nodes"], data["parent_id"], member
    )

    recomputed = np.sum(
        np.abs(data["nodes"][:, :, 2] - float(reference_point)), axis=1
    )
    maximum_cache_error = float(
        np.max(np.abs(recomputed - data["potential"]))
    )
    cache_tolerance = 1.0e-10 * max(
        1.0,
        float(np.max(np.abs(recomputed))),
        float(np.max(np.abs(data["potential"]))),
    )
    if maximum_cache_error > cache_tolerance:
        raise ValueError(
            "cached gravity potential does not match "
            "sum(abs(z-reference_point)): maximum error "
            f"{maximum_cache_error:.6g} exceeds {cache_tolerance:.6g}"
        )

    np.savez_compressed(
        output_directory / "state_tree.npz",
        schema_version=np.asarray("potential_tree_v1"),
        state_id=data["state_id"],
        parent_id=data["parent_id"],
        depth=data["depth"],
        child_count=data["child_count"],
        node=data["nodes"],
        joint_angle_deg=angles.astype(np.float32),
        joint_angle_label=np.asarray(angle_labels, dtype="U32"),
        chain_node_id=chain,
        potential=data["potential"],
        terminal_reason=data["terminal_reason"],
        incoming_step_type=data["incoming_step_type"],
        reference_point=np.asarray(float(reference_point)),
        maximum_potential_cache_error=np.asarray(maximum_cache_error),
        potential_cache_tolerance=np.asarray(cache_tolerance),
    )
    _potential_html(
        data,
        angles,
        angle_labels,
        reference_point,
        output_directory / "potential_tree.html",
    )
    return {
        "state_count": int(len(history)),
        "potential_minimum": float(np.min(data["potential"])),
        "potential_median": float(np.median(data["potential"])),
        "potential_maximum": float(np.max(data["potential"])),
        "maximum_potential_cache_error": maximum_cache_error,
    }


__all__ = ["save_potential_tree"]
