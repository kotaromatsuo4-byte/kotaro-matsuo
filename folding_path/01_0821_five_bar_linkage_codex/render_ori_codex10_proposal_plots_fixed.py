"""Additive, fail-closed padded proposal plots for the frozen C10NS1 report.

The immutable formal report is an input, never an output.  This renderer binds
the exact formal output inventory and frozen C10 finalizer, recreates only the
three proposal-clock figures with explicit top padding, and publishes to one
fresh sibling directory after input rehash and a final PID-zero check.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import finalize_ori_codex10_nonswap_staged as c10final


SCRIPT_DIR = Path(__file__).resolve().parent
FORMAL_ROOT = (
    SCRIPT_DIR / "experiments_ori10" / "analysis" / "C10NS1_final"
).resolve()
OUTPUT = (
    SCRIPT_DIR / "experiments_ori10" / "analysis" / "C10NS1_visual_fix"
).resolve()
C10_ROOT = (SCRIPT_DIR / "experiments_ori10" / "C10NS1").resolve()
TASKS = ("five_d2", "five_d3", "watersqrt2")
SEEDS = (9101, 9102, 9103)
SELF_NAMES = (
    "render_ori_codex10_proposal_plots_fixed.py",
    "test_render_ori_codex10_proposal_plots_fixed.py",
)
EXPECTED_C10_FINALIZER_SHA256 = (
    "e3e6b121bdcefd62f7c720145ed49eb92fb92f65d5e728b8dd57127345197d45"
)
EXPECTED_C10_FINALIZER_TEST_SHA256 = (
    "5ced4493fa027eb9032873d8ad553110ad770e767c350092b3131b29bb96370e"
)
EXPECTED_FORMAL_OUTPUT_HASHES_SHA256 = (
    "877277a303c8e85b5c766b91bbd4e6517c39f345bbd32f143d0aa048a063be88"
)
EXPECTED_FORMAL_FILES = {
    "final_report.json", "historical_c5_anchors.csv", "input_hashes.json",
    "job_summary.csv", "learning_curve_five_d2_episode.png",
    "learning_curve_five_d2_proposals.png",
    "learning_curve_five_d3_episode.png",
    "learning_curve_five_d3_proposals.png",
    "learning_curve_watersqrt2_episode.png",
    "learning_curve_watersqrt2_proposals.png", "output_hashes.json",
    "paired_seed_comparisons.csv", "report_ja.md", "screen_comparison.csv",
    "screen_optimizer_comparison.png", "selected_fixed_evaluations.csv",
    "selected_learning_curves.csv", "visual_qa.json",
}
CURVE_FIELDS = (
    "task", "arm", "seed", "selected", "episode", "cumulative_proposals",
    "steps", "strict", "constructive_C", "joint_J",
    "terminal_flatness_F", "spread_error_e", "return_diagnostic",
    "rejected_crossing_rate", "spectral_terminal_potential",
    "spectral_reward", "strict_rate_256", "constructive_C_rate_256",
    "joint_J_rate_256", "terminal_flatness_F_median_256",
    "spread_error_e_median_256", "return_diagnostic_mean_256",
    "rejected_crossing_rate_mean_256", "spectral_potential_mean_256",
    "spectral_reward_mean_256", "strict_rate_512",
    "constructive_C_rate_512", "joint_J_rate_512",
    "terminal_flatness_F_median_512", "spread_error_e_median_512",
    "return_diagnostic_mean_512", "rejected_crossing_rate_mean_512",
    "spectral_potential_mean_512", "spectral_reward_mean_512",
)
FIXED_FIELDS = (
    "task", "arm", "seed", "selected", "episode", "cumulative_proposals",
    "evaluation_episodes", "strict_compact_success_rate",
)
CURVE_INTEGER_FIELDS = {
    "seed", "episode", "cumulative_proposals", "steps", "strict",
    "constructive_C", "joint_J",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_bytes(path: Path) -> bytes:
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise RuntimeError(f"Input changed while reading: {path}")
    return data


def require_sha256(value, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdefABCDEF" for character in value):
        raise RuntimeError(f"{label} is not an exact SHA-256 digest")
    return value.lower()


def record_input(path: Path, inputs: dict[str, str]) -> bytes:
    path = Path(path).resolve()
    data = stable_bytes(path)
    digest = sha256_bytes(data)
    previous = inputs.get(str(path))
    if previous is not None and previous != digest:
        raise RuntimeError(f"Input changed between reads: {path}")
    inputs[str(path)] = digest
    return data


def verify_inputs(inputs: dict[str, str]) -> None:
    for raw_path, expected in sorted(inputs.items()):
        path = Path(raw_path)
        if not path.is_file() or sha256_bytes(stable_bytes(path)) != expected:
            raise RuntimeError(f"Consumed input changed before publish: {path}")


def validate_self_approval(approved_script_sha256: str,
                           approved_test_sha256: str,
                           inputs: dict[str, str]) -> dict[str, str]:
    approved = {
        SELF_NAMES[0]: require_sha256(
            approved_script_sha256, "approved renderer SHA-256"
        ),
        SELF_NAMES[1]: require_sha256(
            approved_test_sha256, "approved renderer-test SHA-256"
        ),
    }
    for name, expected in approved.items():
        observed = sha256_bytes(record_input(SCRIPT_DIR / name, inputs))
        if observed != expected:
            raise RuntimeError(f"Externally approved SHA-256 mismatch: {name}")
    return approved


def validate_formal_inventory(
    formal_root: Path,
    expected_inventory_sha256: str,
    expected_files: set[str],
    inputs: dict[str, str],
) -> tuple[dict[str, str], dict[str, bytes]]:
    formal_root = Path(formal_root).resolve()
    inventory_path = formal_root / "output_hashes.json"
    inventory_bytes = record_input(inventory_path, inputs)
    if sha256_bytes(inventory_bytes) != require_sha256(
        expected_inventory_sha256, "formal output-hashes SHA-256"
    ):
        raise RuntimeError("Formal output_hashes.json authority mismatch")
    inventory = json.loads(inventory_bytes.decode("utf-8-sig"))
    if not isinstance(inventory, dict):
        raise RuntimeError("Formal output inventory is not an object")
    expected_inventory_names = set(expected_files) - {"output_hashes.json"}
    if set(inventory) != expected_inventory_names:
        raise RuntimeError("Formal output inventory has missing/unexpected entries")
    actual_entries = list(formal_root.iterdir())
    if any(not path.is_file() for path in actual_entries):
        raise RuntimeError("Formal output directory contains a non-file entry")
    actual_names = {path.name for path in actual_entries}
    if actual_names != set(expected_files):
        raise RuntimeError("Formal output directory has missing/unexpected files")
    payload_bytes: dict[str, bytes] = {"output_hashes.json": inventory_bytes}
    checked: dict[str, str] = {}
    for name in sorted(inventory):
        expected = require_sha256(inventory[name], f"formal artifact {name}")
        data = record_input(formal_root / name, inputs)
        observed = sha256_bytes(data)
        if observed != expected:
            raise RuntimeError(f"Formal artifact hash mismatch: {name}")
        checked[name] = observed
        payload_bytes[name] = data
    return checked, payload_bytes


def _csv_rows(data: bytes, expected_fields: tuple[str, ...], label: str) -> list[dict]:
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
    if tuple(reader.fieldnames or ()) != expected_fields:
        raise RuntimeError(f"{label} columns differ from frozen formal schema")
    rows = list(reader)
    if not rows:
        raise RuntimeError(f"{label} is empty")
    return rows


def _exact_int(value, label: str, minimum: int | None = None,
               maximum: int | None = None) -> int:
    result = c10final.exact_int(value, label, minimum, maximum)
    return result


def _finite(value, label: str, minimum: float | None = None,
            maximum: float | None = None) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise RuntimeError(f"{label} is not numeric") from error
    if not math.isfinite(result):
        raise RuntimeError(f"{label} is not finite")
    if minimum is not None and result < minimum:
        raise RuntimeError(f"{label} is below {minimum}")
    if maximum is not None and result > maximum:
        raise RuntimeError(f"{label} is above {maximum}")
    return result


def load_plot_payload(formal_bytes: dict[str, bytes]) -> dict:
    report = json.loads(formal_bytes["final_report.json"].decode("utf-8-sig"))
    if not isinstance(report, dict):
        raise RuntimeError("Formal report is not an object")
    arms = report.get("ever_selected_arms")
    if not isinstance(arms, dict) or set(arms) != set(TASKS):
        raise RuntimeError("Formal report lacks exact selected-arm task set")
    curves = _csv_rows(
        formal_bytes["selected_learning_curves.csv"], CURVE_FIELDS,
        "selected learning curves",
    )
    parsed_curves = []
    curve_groups: dict[tuple[str, int], list[dict]] = {}
    for index, row in enumerate(curves, 2):
        task, arm = row["task"], row["arm"]
        if task not in TASKS or arm != arms[task] or row["selected"] != "True":
            raise RuntimeError(f"Unexpected selected curve identity at CSV row {index}")
        parsed = {"task": task, "arm": arm, "selected": True}
        for field in CURVE_INTEGER_FIELDS:
            bounds = {
                "seed": (min(SEEDS), max(SEEDS)), "episode": (1, 10000),
                "cumulative_proposals": (1, None), "steps": (1, 400),
                "strict": (0, 1), "constructive_C": (0, 1),
                "joint_J": (0, 1),
            }[field]
            parsed[field] = _exact_int(
                row[field], f"curve row {index} {field}", *bounds
            )
        if parsed["seed"] not in SEEDS:
            raise RuntimeError(f"Unexpected selected seed at CSV row {index}")
        for field in CURVE_FIELDS:
            if field in {"task", "arm", "selected"} | CURVE_INTEGER_FIELDS:
                continue
            parsed[field] = _finite(row[field], f"curve row {index} {field}")
        parsed_curves.append(parsed)
        curve_groups.setdefault((task, parsed["seed"]), []).append(parsed)
    jobs = report.get("jobs")
    if not isinstance(jobs, list):
        raise RuntimeError("Formal report jobs are missing")
    job_map = {
        (job.get("task"), job.get("arm"), job.get("seed")): job for job in jobs
        if isinstance(job, dict)
    }
    expected_groups = {(task, seed) for task in TASKS for seed in SEEDS}
    if set(curve_groups) != expected_groups:
        raise RuntimeError("Selected curves do not contain exact task/seed groups")
    for task, seed in sorted(expected_groups):
        rows = curve_groups[(task, seed)]
        job = job_map.get((task, arms[task], seed))
        if job is None:
            raise RuntimeError(f"Selected formal job missing: {task}/s{seed}")
        endpoint = _exact_int(
            job.get("formal_endpoint"), f"formal endpoint {task}/s{seed}", 1, 10000
        )
        if [row["episode"] for row in rows] != list(range(1, endpoint + 1)):
            raise RuntimeError(f"Selected curve episodes are not contiguous: {task}/s{seed}")
        cumulative = [row["cumulative_proposals"] for row in rows]
        if any(right <= left for left, right in zip(cumulative, cumulative[1:])):
            raise RuntimeError(f"Proposal clock is not strictly increasing: {task}/s{seed}")
    fixed = _csv_rows(
        formal_bytes["selected_fixed_evaluations.csv"], FIXED_FIELDS,
        "selected fixed evaluations",
    )
    parsed_fixed = []
    fixed_groups: dict[tuple[str, int], list[dict]] = {}
    for index, row in enumerate(fixed, 2):
        task, arm = row["task"], row["arm"]
        if task not in TASKS or arm != arms[task] or row["selected"] != "True":
            raise RuntimeError(f"Unexpected fixed-eval identity at CSV row {index}")
        parsed = {
            "task": task, "arm": arm, "selected": True,
            "seed": _exact_int(row["seed"], f"fixed row {index} seed"),
            "episode": _exact_int(
                row["episode"], f"fixed row {index} episode", 1, 10000
            ),
            "cumulative_proposals": _exact_int(
                row["cumulative_proposals"],
                f"fixed row {index} cumulative proposals", 1,
            ),
            "evaluation_episodes": _exact_int(
                row["evaluation_episodes"],
                f"fixed row {index} evaluation episodes", 16, 16,
            ),
            "strict_compact_success_rate": _finite(
                row["strict_compact_success_rate"],
                f"fixed row {index} strict rate", 0.0, 1.0,
            ),
        }
        if parsed["seed"] not in SEEDS:
            raise RuntimeError(f"Unexpected fixed-eval seed at CSV row {index}")
        parsed_fixed.append(parsed)
        fixed_groups.setdefault((task, parsed["seed"]), []).append(parsed)
    if set(fixed_groups) != expected_groups:
        raise RuntimeError("Fixed evaluations do not contain exact task/seed groups")
    for task, seed in sorted(expected_groups):
        job = job_map[(task, arms[task], seed)]
        endpoint = _exact_int(job["formal_endpoint"], "fixed formal endpoint", 1, 10000)
        rows = fixed_groups[(task, seed)]
        if rows[-1]["episode"] != endpoint:
            raise RuntimeError(f"Fixed evaluation lacks formal endpoint: {task}/s{seed}")
    report["selected_episode_curves"] = parsed_curves
    report["selected_fixed_evaluations"] = parsed_fixed
    return report


def render_padded_proposal_plot(payload: dict, task: str, path: Path) -> None:
    """Reuse the frozen data renderer while imposing explicit safe margins."""
    import matplotlib
    matplotlib.use("Agg")

    rc = {
        "figure.constrained_layout.h_pad": 0.12,
        "figure.constrained_layout.hspace": 0.08,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.18,
    }
    with matplotlib.rc_context(rc):
        c10final.render_selected_plot(
            payload, task, Path(path), proposals_axis=True
        )


def visual_qa(path: Path) -> dict:
    from PIL import Image, ImageChops, ImageStat

    path = Path(path)
    with Image.open(path) as image:
        image.load()
        rgb = image.convert("RGB")
        width, height = rgb.size
        standard_deviation = max(ImageStat.Stat(rgb).stddev)
        difference = ImageChops.difference(
            rgb, Image.new("RGB", rgb.size, (255, 255, 255))
        ).convert("L")
        mask = difference.point(lambda value: 255 if value > 10 else 0)
        bounding_box = mask.getbbox()
    if bounding_box is None:
        raise RuntimeError(f"Proposal plot is blank: {path}")
    top_margin_px = bounding_box[1]
    if path.stat().st_size < 5_000 or width < 1_600 or height < 1_200 \
            or standard_deviation < 1.0 or top_margin_px < 20:
        raise RuntimeError(f"Proposal plot failed padded visual QA: {path}")
    return {
        "file": path.name, "bytes": path.stat().st_size,
        "width": width, "height": height,
        "top_content_margin_px": top_margin_px,
        "maximum_channel_standard_deviation": standard_deviation,
        "sha256": sha256_file(path), "pass": True,
    }


def write_json(path: Path, payload) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True,
                   allow_nan=False) + "\n",
        encoding="utf-8", newline="\n",
    )


def verify_before_publish(inputs: dict[str, str], active_checker) -> None:
    verify_inputs(inputs)
    active = active_checker()
    if active:
        raise RuntimeError(f"Active C10 PID(s) appeared before publish: {active}")


def build_bundle(payload: dict, formal_inventory: dict[str, str],
                 approvals: dict[str, str], inputs: dict[str, str],
                 output: Path = OUTPUT) -> dict:
    output = Path(output).resolve()
    if output != OUTPUT or output.exists():
        raise RuntimeError(f"Output must be the fresh fixed path: {OUTPUT}")
    output.parent.mkdir(parents=True, exist_ok=True)
    build = Path(tempfile.mkdtemp(prefix="tmp-c10-visual-fix-", dir=output.parent))
    try:
        plot_paths = []
        for task in TASKS:
            path = build / f"learning_curve_{task}_proposals_padded.png"
            render_padded_proposal_plot(payload, task, path)
            plot_paths.append(path)
        qa = [visual_qa(path) for path in plot_paths]
        binding = {
            "schema_version": 1,
            "purpose": "additive_top_padding_fix_for_c10ns1_proposal_plots",
            "formal_root": str(FORMAL_ROOT),
            "formal_output_hashes_sha256": EXPECTED_FORMAL_OUTPUT_HASHES_SHA256,
            "formal_file_sha256": formal_inventory,
            "frozen_c10_finalizer_sha256": EXPECTED_C10_FINALIZER_SHA256,
            "frozen_c10_finalizer_test_sha256": EXPECTED_C10_FINALIZER_TEST_SHA256,
            "approved_renderer_source_sha256": approvals,
            "layout": {
                "suptitle_position": "frozen_renderer_auto_constrained",
                "constrained_h_pad_inches": 0.12,
                "constrained_hspace": 0.08, "bbox_inches": "tight",
                "pad_inches": 0.18, "minimum_top_content_margin_px": 20,
            },
            "formal_output_unchanged": True,
        }
        write_json(build / "source_binding.json", binding)
        write_json(build / "visual_qa.json", qa)
        write_json(build / "input_hashes.json", dict(sorted(inputs.items())))
        hashes = {
            path.name: sha256_file(path) for path in sorted(build.iterdir())
            if path.is_file()
        }
        write_json(build / "output_hashes.json", hashes)
        verify_before_publish(
            inputs, lambda: c10final.active_c10_pids(C10_ROOT)
        )
        os.replace(build, output)
    finally:
        if build.exists():
            shutil.rmtree(build)
    return {
        "output": str(output),
        "files": sorted(path.name for path in output.iterdir()),
        "output_hashes_sha256": sha256_file(output / "output_hashes.json"),
    }


def analyze(approved_script_sha256: str,
            approved_test_sha256: str) -> tuple[
                dict, dict[str, str], dict[str, str], dict[str, str]
            ]:
    inputs: dict[str, str] = {}
    approvals = validate_self_approval(
        approved_script_sha256, approved_test_sha256, inputs
    )
    c10_sources = {
        "finalize_ori_codex10_nonswap_staged.py": EXPECTED_C10_FINALIZER_SHA256,
        "test_finalize_ori_codex10_nonswap_staged.py":
            EXPECTED_C10_FINALIZER_TEST_SHA256,
    }
    for name, expected in c10_sources.items():
        if sha256_bytes(record_input(SCRIPT_DIR / name, inputs)) != expected:
            raise RuntimeError(f"Frozen C10 finalizer authority mismatch: {name}")
    if c10final.active_c10_pids(C10_ROOT):
        raise RuntimeError("C10 trainer/supervisor is still active")
    formal_inventory, formal_bytes = validate_formal_inventory(
        FORMAL_ROOT, EXPECTED_FORMAL_OUTPUT_HASHES_SHA256,
        EXPECTED_FORMAL_FILES, inputs,
    )
    payload = load_plot_payload(formal_bytes)
    verify_inputs(inputs)
    return payload, formal_inventory, approvals, inputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--approved-renderer-sha256", required=True)
    parser.add_argument("--approved-test-sha256", required=True)
    options = parser.parse_args(argv)
    payload, inventory, approvals, inputs = analyze(
        options.approved_renderer_sha256, options.approved_test_sha256
    )
    result = build_bundle(payload, inventory, approvals, inputs)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == "__main__":
    main()
