"""Read-only counterfactual audit of the water near-miss spread cutoff.

This utility never replays mechanics and never edits a training run.  It replaces
only the terminal spread-quality factor in already-recorded episodes, holding the
visited state/action trajectory and every other reward component fixed.  The
result therefore measures immediate reward separation, not the policy that would
have been learned under the alternative reward.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Iterable, Sequence


CUTOFFS = (0.10, 0.12, 0.15, 0.40)
SPREAD_TOLERANCE = 0.05
CURRENT_ENDPOINT = 2048
CURRENT_ROOT = Path(
    "experiments_ori6/staged/zero_reject_reward/P0X2RCS05ZR0R1"
)
OUTPUT_ROOT = Path(
    "experiments_ori6/analysis/water_cutoff_counterfactual_20260827"
)


@dataclass(frozen=True)
class Source:
    label: str
    history: Path
    endpoint: int | None
    mutable: bool = False


def sources() -> list[Source]:
    items = [
        Source(
            f"current_zero_reject_seed{seed}_ep{CURRENT_ENDPOINT}",
            CURRENT_ROOT / "water1" / "B" / f"seed{seed}" / "training_history.csv",
            CURRENT_ENDPOINT,
            True,
        )
        for seed in (4201, 4202, 4203)
    ]
    items.append(Source(
        "old_two_task_seed4101_ep6144",
        Path("experiments_ori6/full/two_task_full_diagnostic/P0X2RCS05")
        / "water1/B/seed4101/training_history.csv",
        6144,
    ))
    items.extend(
        Source(
            f"saved_spectral_screen_seed{seed}",
            Path("experiments_ori6/screen/spectral_subspace_diagnostic/P0X2RCS05")
            / f"water1/B/seed{seed}/training_history.csv",
            None,
        )
        for seed in (3201, 3202, 3203)
    )
    return items


def _float(row: dict[str, str], *keys: str, default: float = math.nan) -> float:
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            try:
                parsed = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(parsed):
                return parsed
    return default


def qd(spread_error: float, cutoff: float) -> float:
    value = (cutoff - abs(spread_error)) / (cutoff - SPREAD_TOLERANCE)
    return min(1.0, max(0.0, value))


def cohort(flatness: float, spread_error: float) -> str:
    if flatness <= 0.001 and spread_error <= 0.05:
        return "strict"
    if flatness <= 0.5 and spread_error <= 0.10:
        return "constructive"
    if flatness <= 0.5 and 0.10 < spread_error <= 0.15:
        return "boundary_flat_0p10_0p15"
    if flatness <= 0.001 and spread_error > 0.15:
        return "wrong_flat_far"
    if spread_error >= 0.30:
        return "far_spread"
    return "other"


def load_source(source: Source) -> list[dict]:
    if not source.history.is_file():
        raise FileNotFoundError(source.history)
    rows: list[dict] = []
    with source.history.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            episode = int(raw["episode"])
            if source.endpoint is not None and episode > source.endpoint:
                break
            flatness = _float(
                raw, "ori5_last_valid_flatness_ratio", "flatness_ratio"
            )
            spread_error = abs(_float(
                raw, "ori5_last_valid_spread_error", "spread_error"
            ))
            reward = _float(raw, "total_reward")
            qf = _float(raw, "near_miss_qF", default=0.0)
            actual_quality = _float(
                raw,
                "ori6_total_terminal_quality_reward",
                "ori5_total_terminal_quality_reward",
                default=0.0,
            )
            quality_source = (
                raw.get("ori6_terminal_quality_source")
                or raw.get("ori5_terminal_quality_source")
                or "none"
            )
            # wrong_flat intentionally receives spread-only credit.  Every
            # other quality-bearing terminal uses qF*qD.
            spread_only = quality_source == "after_valid_spread_only"
            eligible = quality_source != "none" and abs(actual_quality) > 0.0
            # A zero old quality can still be eligible because q was clipped
            # to zero.  Termination/source semantics are authoritative.
            eligible = quality_source not in {"", "none"}
            base = {
                "dataset": source.label,
                "episode": episode,
                "termination_reason": raw.get("termination_reason", ""),
                "quality_source": quality_source,
                "flatness": flatness,
                "spread_error": spread_error,
                "cohort": cohort(flatness, spread_error),
                "actual_return": reward,
                "actual_terminal_quality": actual_quality,
                "qF": qf,
                "spread_only": int(spread_only),
                "eligible": int(eligible),
            }
            for cutoff in CUTOFFS:
                label = f"{cutoff:.2f}"
                new_qd = qd(spread_error, cutoff)
                new_quality = (
                    new_qd if spread_only else qf * new_qd
                ) if eligible else 0.0
                base[f"qD_{label}"] = new_qd
                base[f"quality_{label}"] = new_quality
                base[f"return_{label}"] = reward - actual_quality + new_quality
            rows.append(base)
    expected = source.endpoint
    if expected is not None:
        episodes = [row["episode"] for row in rows]
        if episodes != list(range(1, expected + 1)):
            raise RuntimeError(
                f"{source.label}: non-atomic/non-contiguous prefix to {expected}"
            )
    return rows


def _safe_mean(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return mean(finite) if finite else None


def _safe_median(values: Iterable[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return median(finite) if finite else None


def summarize(rows: Sequence[dict]) -> list[dict]:
    result: list[dict] = []
    datasets = sorted({row["dataset"] for row in rows})
    for dataset in datasets + ["CURRENT_POOLED", "ALL_SAVED_OLD"]:
        if dataset == "CURRENT_POOLED":
            selected = [row for row in rows if row["dataset"].startswith("current_")]
        elif dataset == "ALL_SAVED_OLD":
            selected = [row for row in rows if not row["dataset"].startswith("current_")]
        else:
            selected = [row for row in rows if row["dataset"] == dataset]
        for group in (
            "all", "strict", "constructive", "boundary_flat_0p10_0p15",
            "wrong_flat_far", "far_spread",
        ):
            group_rows = selected if group == "all" else [
                row for row in selected if row["cohort"] == group
            ]
            for cutoff in CUTOFFS:
                label = f"{cutoff:.2f}"
                qualities = [row[f"quality_{label}"] for row in group_rows]
                returns = [row[f"return_{label}"] for row in group_rows]
                result.append({
                    "dataset": dataset,
                    "cohort": group,
                    "cutoff": cutoff,
                    "episode_count": len(group_rows),
                    "positive_quality_count": sum(value > 0 for value in qualities),
                    "mean_quality": _safe_mean(qualities),
                    "median_quality": _safe_median(qualities),
                    "mean_return": _safe_mean(returns),
                    "median_return": _safe_median(returns),
                    "positive_return_rate": (
                        sum(value > 0 for value in returns) / len(returns)
                        if returns else None
                    ),
                })
    return result


def load_saved_success_path() -> dict:
    path = Path(
        "experiments_ori6/screen/spectral_subspace_diagnostic/P0X2RCS05/"
        "water1/B/seed3203/first_success_path/actions.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    actions = payload if isinstance(payload, list) else payload.get("actions", [])
    if not actions:
        raise RuntimeError(f"no actions in {path}")
    terminal = actions[-1]
    flatness = float(terminal["flatness_ratio"])
    spread_error = abs(float(terminal["spread_error"]))
    return {
        "path": str(path.resolve()),
        "steps": len(actions),
        "termination_reason": terminal.get("termination_reason"),
        "flatness": flatness,
        "spread_error": spread_error,
        "strict": flatness <= 0.001 and spread_error <= 0.05,
        "qD_by_cutoff": {
            f"{cutoff:.2f}": qd(spread_error, cutoff) for cutoff in CUTOFFS
        },
    }


def recommendation(summary: Sequence[dict]) -> dict:
    pooled = {
        (row["cohort"], row["cutoff"]): row
        for row in summary if row["dataset"] == "CURRENT_POOLED"
    }
    boundary = {
        cutoff: pooled[("boundary_flat_0p10_0p15", cutoff)]
        for cutoff in CUTOFFS
    }
    # 0.12 is preregistered as the smallest relaxation above the 0.10
    # constructive gate that almost eliminates the observed 0.114--0.118
    # attractor while leaving a graded signal throughout 0.05--0.10.
    return {
        "recommended_cutoff": 0.12,
        "reason": (
            "0.12 is the smallest cutoff above the 0.10 constructive gate: "
            "it preserves a graded target-plane signal for 0.05<eD<0.10, "
            "while eD=0.114--0.118 receives only qD=0.029--0.086.  Cutoff "
            "0.15 still gives that boundary attractor qD=0.32--0.36; cutoff "
            "0.10 removes it completely but also drives the whole 0.05--0.10 "
            "approach band linearly to zero at the gate."
        ),
        "current_boundary_metrics": boundary,
        "interpretation_limit": (
            "Recorded-trajectory terminal-reward counterfactual only; it does "
            "not identify the policy distribution learned after retraining."
        ),
    }


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, payload: dict) -> None:
    rec = payload["recommendation"]
    summary = payload["summary"]
    selected = [
        row for row in summary
        if row["dataset"] == "CURRENT_POOLED"
        and row["cohort"] in {"all", "constructive", "boundary_flat_0p10_0p15"}
    ]
    lines = [
        "# Water D=1 near-miss spread cutoff counterfactual",
        "",
        "## Conclusion",
        "",
        f"Recommended fallback cutoff: **{rec['recommended_cutoff']:.2f}**.",
        "",
        rec["reason"],
        "",
        "This is a one-component, recorded-trajectory counterfactual. It is not a retraining result.",
        "",
        "## Current zero-reject checkpoint 2048 (three-seed pooled)",
        "",
        "| Cohort | cutoff | N | positive quality | mean quality | mean return | positive return rate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        lines.append(
            "| {cohort} | {cutoff:.2f} | {episode_count} | "
            "{positive_quality_count} | {mean_quality:.6f} | {mean_return:.6f} | "
            "{positive_return_rate:.4f} |".format(
                **{key: (0.0 if value is None else value) for key, value in row.items()}
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    all_rows: list[dict] = []
    source_meta = []
    for source in sources():
        rows = load_source(source)
        all_rows.extend(rows)
        source_meta.append({
            "label": source.label,
            "path": str(source.history.resolve()),
            "endpoint": rows[-1]["episode"] if rows else None,
            "episode_count": len(rows),
            "active_prefix_only": source.mutable,
        })
    summary = summarize(all_rows)
    payload = {
        "analysis_type": "water_near_miss_cutoff_recorded_trajectory_counterfactual",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "formula": "qD=clip((far-eD)/(far-0.05),0,1)",
        "cutoffs": list(CUTOFFS),
        "sources": source_meta,
        "saved_success_path": load_saved_success_path(),
        "summary": summary,
        "recommendation": recommendation(summary),
    }
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "water_cutoff_counterfactual.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "water_cutoff_episode_counterfactual.csv", all_rows)
    write_csv(args.output / "water_cutoff_summary.csv", summary)
    write_report(args.output / "water_cutoff_report.md", payload)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "recommended_cutoff": payload["recommendation"]["recommended_cutoff"],
        "episode_rows": len(all_rows),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
