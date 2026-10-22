"""Pure helpers for perturbation-depth bonuses and delayed path credit.

The search tree already stores the authoritative edge type on every
``StateNode``.  These helpers deliberately derive perturbation depth from
``incoming_step_type`` instead of relying on a second, parallel edge list.

Nothing in this file changes the search.  ``main5.py`` can later import the
bonus function and can save the returned diagnostics, while the ancestry
functions can be tested independently with a small synthetic tree.
"""

from __future__ import annotations

import numpy as np

from state import get_path_state_ids


def state_path_perturbation_counts(history):
    """Return the number of perturbation edges from root to every state.

    ``history[state_id]`` is an invariant used throughout the current code.
    This function checks that invariant, the parent ordering, and the stored
    depth while computing the count in one forward pass.
    """
    counts = np.zeros(len(history), dtype=int)
    allowed_step_types = {None, "deterministic", "perturbation"}

    for index, state in enumerate(history):
        if int(state.state_id) != index:
            raise ValueError("history index and state_id differ")
        if state.incoming_step_type not in allowed_step_types:
            raise ValueError(
                f"unknown incoming step type at state {index}: "
                f"{state.incoming_step_type}"
            )

        if state.parent_id is None:
            if index != 0 or int(state.depth) != 0:
                raise ValueError("only state 0 may be the depth-zero root")
            if state.incoming_step_type is not None:
                raise ValueError("the root must not have an incoming step")
            continue

        parent_id = int(state.parent_id)
        if parent_id < 0 or parent_id >= index:
            raise ValueError("each parent must precede its child in history")
        if int(state.depth) != int(history[parent_id].depth) + 1:
            raise ValueError(f"stored depth is inconsistent at state {index}")
        counts[index] = counts[parent_id] + int(
            state.incoming_step_type == "perturbation"
        )
    return counts


def state_consecutive_perturbation_counts(history):
    """Return the perturbation-run length ending at every state."""
    counts = np.zeros(len(history), dtype=int)
    for index, state in enumerate(history):
        if int(state.state_id) != index:
            raise ValueError("history index and state_id differ")
        if state.parent_id is None:
            continue
        if int(state.parent_id) >= index:
            raise ValueError("a parent must precede its child in history")
        if state.incoming_step_type == "perturbation":
            counts[index] = counts[int(state.parent_id)] + 1
    return counts


def path_perturbation_edges(history, terminal_state_id):
    """Return ``(parent_id, child_id)`` perturbation edges on one path."""
    path = get_path_state_ids(history, int(terminal_state_id))
    return [
        (int(parent_id), int(child_id))
        for parent_id, child_id in zip(path[:-1], path[1:])
        if history[int(child_id)].incoming_step_type == "perturbation"
    ]


def elite_representative_diagnostics(history, elite_archive):
    """Describe perturbations on each final elite representative path.

    These are diagnostics for the *final representatives*.  If a cell was
    first discovered by a different path and later replaced by a shorter
    path, that first-discovery path cannot be reconstructed from the final
    archive alone and must be logged as a discovery event during the search.
    """
    counts = state_path_perturbation_counts(history)
    rows = []
    for cell_index, cell in enumerate(elite_archive):
        terminal_state_id = int(cell["terminal_state_id"])
        edges = path_perturbation_edges(history, terminal_state_id)
        rows.append(
            {
                "cell_index": int(cell_index),
                "terminal_state_id": terminal_state_id,
                "path_depth": int(history[terminal_state_id].depth),
                "path_perturbation_count": int(counts[terminal_state_id]),
                "perturbation_parent_ids": [edge[0] for edge in edges],
                "perturbation_child_ids": [edge[1] for edge in edges],
            }
        )
    return rows


def apply_perturbed_parent_bonus(
    base_scores,
    candidates,
    path_perturbation_counts,
    bonus_strength,
):
    r"""Apply a dimensionless multiplicative prior to candidate scores.

    .. math::

        a'_i = a_i\left(1 + \gamma
        \mathbf{1}[k(\operatorname{parent}_i)>0]\right).

    A multiplier is used because current UCB and posterior-new acquisitions
    have very different numerical ranges.  A zero ``bonus_strength`` is an
    exact no-op.  This is a search prior, not a GP prediction, and therefore
    belongs after ``score_acquisition`` rather than inside ``gp.py``.
    """
    base = np.asarray(base_scores, dtype=float).reshape(-1)
    if len(base) != len(candidates):
        raise ValueError("candidate and score counts differ")
    strength = float(bonus_strength)
    if strength < 0.0:
        raise ValueError("bonus_strength must be nonnegative")

    parent_counts = np.asarray(
        [path_perturbation_counts[int(candidate["parent_id"])] for candidate in candidates],
        dtype=int,
    )
    factors = 1.0 + strength * (parent_counts > 0)
    adjusted = base * factors
    return {
        "base_score": base.copy(),
        "adjusted_score": adjusted,
        "path_bonus": adjusted - base,
        "bonus_factor": factors,
        "parent_path_perturbations": parent_counts,
    }


def apply_direct_perturbation_parent_bonus(
    base_scores,
    candidates,
    history,
    bonus_strength,
):
    r"""Boost actions whose parent was itself created by a perturbation.

    This differs from :func:`apply_perturbed_parent_bonus`.  A deterministic
    descendant of an old perturbation has a positive ancestry count, but it
    is not a consecutive small perturbation.  To approximate one large
    action by a continuous sequence of bounded actions, the relevant event is

    .. math::

        \operatorname{incoming\_step\_type}(s_{\rm parent})
        = \text{perturbation}.

    The multiplier is applied only to guided scores.  Random and cold-start
    selections remain unchanged in ``main5.py``.
    """
    base = np.asarray(base_scores, dtype=float).reshape(-1)
    if len(base) != len(candidates):
        raise ValueError("candidate and score counts differ")
    strength = float(bonus_strength)
    if strength < 0.0:
        raise ValueError("bonus_strength must be nonnegative")

    is_direct = np.asarray(
        [
            history[int(candidate["parent_id"])].incoming_step_type
            == "perturbation"
            for candidate in candidates
        ],
        dtype=bool,
    )
    factors = 1.0 + strength * is_direct.astype(float)
    adjusted = base * factors
    return {
        "base_score": base.copy(),
        "adjusted_score": adjusted,
        "path_bonus": adjusted - base,
        "bonus_factor": factors,
        "parent_is_direct_perturbation": is_direct,
    }


def apply_transport_alignment_bonus(
    base_scores,
    candidates,
    bonus_strength,
):
    r"""Prefer continuation of the preceding physical perturbation direction.

    Candidate generation stores

    .. math::

        \rho_i = \max\!\left(0,
        \frac{\bm v_i^T\bm P_{T_s}\bm v_{\rm prev}}
        {\|\bm v_i\|\,\|\bm P_{T_s}\bm v_{\rm prev}\|}\right),

    where :math:`\bm P_{T_s}` is the current tangent-space projector.  The
    multiplier ``1 + bonus_strength * rho_i`` therefore rewards only a
    physically aligned continuation, unlike a binary ancestry bonus that
    treats every direction from a previously perturbed state equally.
    """
    base = np.asarray(base_scores, dtype=float).reshape(-1)
    if len(base) != len(candidates):
        raise ValueError("candidate and score counts differ")
    strength = float(bonus_strength)
    if strength < 0.0:
        raise ValueError("bonus_strength must be nonnegative")

    alignment = np.asarray(
        [
            np.clip(float(candidate.get("transport_alignment", 0.0)), 0.0, 1.0)
            for candidate in candidates
        ],
        dtype=float,
    )
    factors = 1.0 + strength * alignment
    adjusted = base * factors
    return {
        "base_score": base.copy(),
        "adjusted_score": adjusted,
        "path_bonus": adjusted - base,
        "bonus_factor": factors,
        "transport_alignment": alignment,
        "is_transported_direction": np.asarray(
            [
                bool(candidate.get("is_transported_direction", False))
                for candidate in candidates
            ],
            dtype=bool,
        ),
    }


def candidate_pool_path_summary(
    candidates,
    base_scores,
    adjusted_scores,
    path_perturbation_counts,
    selected_index=None,
):
    """Summarize whether multi-perturbation parents are present but low-ranked.

    The returned dictionary is suitable for one row per search iteration.
    Counts and score summaries use bins 0, 1, 2, 3, and 4 or more.  Saving
    this row prevents the selected candidate alone from hiding candidate-set
    starvation or systematic model under-ranking.
    """
    base = np.asarray(base_scores, dtype=float).reshape(-1)
    adjusted = np.asarray(adjusted_scores, dtype=float).reshape(-1)
    if len(candidates) != len(base) or len(base) != len(adjusted):
        raise ValueError("candidate and score counts differ")

    parent_counts = np.asarray(
        [path_perturbation_counts[int(candidate["parent_id"])] for candidate in candidates],
        dtype=int,
    )
    clipped = np.minimum(parent_counts, 4)
    row = {
        "candidate_count": int(len(candidates)),
        "candidate_parent_count": int(
            len({int(candidate["parent_id"]) for candidate in candidates})
        ),
    }
    for count_bin in range(5):
        mask = clipped == count_bin
        label = "4plus" if count_bin == 4 else str(count_bin)
        row[f"count_k{label}"] = int(np.count_nonzero(mask))
        row[f"base_median_k{label}"] = (
            float(np.nanmedian(base[mask])) if np.any(mask) else np.nan
        )
        row[f"base_max_k{label}"] = (
            float(np.nanmax(base[mask])) if np.any(mask) else np.nan
        )

    if selected_index is None:
        row.update(
            {
                "selected_index": -1,
                "selected_parent_path_perturbations": -1,
                "selected_base_rank": -1,
                "selected_adjusted_rank": -1,
            }
        )
        return row

    selected = int(selected_index)
    if selected < 0 or selected >= len(candidates):
        raise IndexError("selected candidate index is outside the pool")

    # Stable sorting makes a tied rank deterministic for diagnostics only.
    base_order = np.argsort(-np.nan_to_num(base, nan=-np.inf), kind="stable")
    adjusted_order = np.argsort(
        -np.nan_to_num(adjusted, nan=-np.inf), kind="stable"
    )
    row.update(
        {
            "selected_index": selected,
            "selected_parent_path_perturbations": int(parent_counts[selected]),
            "selected_base_rank": int(np.flatnonzero(base_order == selected)[0] + 1),
            "selected_adjusted_rank": int(
                np.flatnonzero(adjusted_order == selected)[0] + 1
            ),
        }
    )
    return row


def perturbation_delayed_credit(history, terminal_records):
    """Assign retrospective terminal credit to every perturbation edge.

    ``terminal_records`` is an iterable of dictionaries containing at least
    ``terminal_state_id``.  For first-discovery diagnosis, main5 should append
    a record immediately whenever a new cell is created; using only the final
    elite archive instead measures final-representative credit.

    A perturbation is a stepping stone when it lies on a credited terminal
    path but at least one further perturbation was needed.  This detects the
    case that receives zero immediate new-cell reward but enables a later
    discovery.
    """
    counts = state_path_perturbation_counts(history)
    records = [dict(record) for record in terminal_records]
    record_paths = [
        set(get_path_state_ids(history, int(record["terminal_state_id"])))
        for record in records
    ]
    rows = []

    for child in history:
        if child.incoming_step_type != "perturbation":
            continue
        child_id = int(child.state_id)
        credited_indices = [
            index for index, path in enumerate(record_paths) if child_id in path
        ]
        remaining_counts = [
            int(counts[int(records[index]["terminal_state_id"])] - counts[child_id])
            for index in credited_indices
        ]
        positive_remaining = [value for value in remaining_counts if value > 0]
        rows.append(
            {
                "perturbation_parent_id": int(child.parent_id),
                "perturbation_child_id": child_id,
                "path_perturbations_after_edge": int(counts[child_id]),
                "credited_terminal_count": int(len(credited_indices)),
                "credited_record_indices": credited_indices,
                "minimum_additional_perturbations": (
                    int(min(remaining_counts)) if remaining_counts else -1
                ),
                "minimum_positive_additional_perturbations": (
                    int(min(positive_remaining)) if positive_remaining else -1
                ),
                "is_delayed_stepping_stone": bool(positive_remaining),
            }
        )
    return rows


def final_elite_terminal_records(elite_archive):
    """Adapt a final archive for ``perturbation_delayed_credit``."""
    return [
        {
            "cell_index": int(cell_index),
            "terminal_state_id": int(cell["terminal_state_id"]),
            "record_kind": "final_representative",
        }
        for cell_index, cell in enumerate(elite_archive)
    ]
