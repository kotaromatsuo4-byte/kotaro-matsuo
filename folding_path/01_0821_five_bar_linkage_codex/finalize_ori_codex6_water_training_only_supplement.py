"""Create a non-authority water episode-10000 supplement without changing formal outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import analyze_ori_codex6_zero_reject_staged as staged
import finalize_ori_codex6_zero_reject_combined as combined
import run_ori_codex6_zero_reject_water_fallback as fallback


HERE = Path(__file__).resolve().parent
RUN = HERE / (
    "experiments_ori6/staged/zero_reject_water_fallback/"
    "P0X2RCS05ZR0WF12/water1/B/seed4301"
)
FALLBACK_ROOT = RUN.parents[2]
FINAL_DIR = HERE / "experiments_ori6/analysis/zero_reject_reward_final"
MAIN_METRICS = FINAL_DIR / "final_metrics.json"
MAIN_REPORT = FINAL_DIR / "final_report_ja.md"
MAIN_MANIFEST = FINAL_DIR / "artifact_sha256.json"
OUTPUTS = {
    "json": FINAL_DIR / "water_training_only_10000_non_authority.json",
    "csv": FINAL_DIR / "water_training_only_10000_non_authority.csv",
    "report": FINAL_DIR / "water_training_only_10000_non_authority_ja.md",
    "manifest": FINAL_DIR / "water_training_only_10000_non_authority_sha256.json",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def verify_main_artifacts() -> dict:
    manifest = read_json(MAIN_MANIFEST)
    expected = manifest.get("artifact_sha256", {})
    if not isinstance(expected, dict) or not expected:
        raise RuntimeError("main artifact manifest has no artifact hash map")
    mismatches = []
    for raw_path, digest in expected.items():
        path = Path(raw_path)
        actual = combined.file_sha256(path) if path.is_file() else None
        if actual != digest:
            mismatches.append(f"{path}: expected={digest}, actual={actual}")
    if mismatches:
        raise RuntimeError("main artifact verification failed: " + "; ".join(mismatches))
    return manifest


def input_paths() -> list[Path]:
    paths = [
        RUN / "training_history.csv",
        RUN / "update_history.csv",
        RUN / "run_config.json",
        RUN / "checkpoint.pt",
        RUN / "policy.pt",
        RUN / "stochastic_evaluation_history.csv",
        RUN / "deterministic_evaluation_history.csv",
        FALLBACK_ROOT / "analysis/final_summary.json",
        MAIN_METRICS,
        MAIN_REPORT,
        MAIN_MANIFEST,
        Path(__file__).resolve(),
        HERE / "test_finalize_ori_codex6_water_training_only_supplement.py",
    ]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing supplement input: " + ", ".join(missing))
    return [path.resolve() for path in paths]


def exact_input_hashes(paths: list[Path]) -> dict[str, str]:
    return {str(path): combined.file_sha256(path) for path in paths}


def selected_main_run(main: dict) -> dict:
    matches = [
        run for run in main.get("runs", [])
        if run.get("dataset") == "water_fallback_far_0p12"
        and run.get("task") == "water1" and run.get("seed") == 4301
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one main fallback run, got {len(matches)}")
    return matches[0]


def evaluation_payload() -> list[dict]:
    stochastic_raw = combined._last_csv_row(
        RUN / "stochastic_evaluation_history.csv", 10000
    )
    deterministic_raw = combined._last_csv_row(
        RUN / "deterministic_evaluation_history.csv", 10000
    )
    if not stochastic_raw or not deterministic_raw:
        raise RuntimeError("episode-10000 periodic evaluation rows are absent")
    stochastic = combined._evaluation_from_stochastic(
        stochastic_raw,
        "stochastic_evaluation_history.csv:last<=10000",
        False,
    )
    stochastic.update({
        "self_crossing_rate": staged._float(
            stochastic_raw, ("self_crossing_rate",), math.nan
        ),
        "strict_compact_success_rate": staged._float(
            stochastic_raw, ("strict_compact_success_rate", "success_rate"), math.nan
        ),
        "authority_status": "training-only periodic diagnostic; not final held-out",
    })
    deterministic = combined._evaluation_from_deterministic(
        deterministic_raw,
        "deterministic_evaluation_history.csv:last<=10000",
        False,
    )
    deterministic.update({
        "termination_reason": deterministic_raw.get("termination_reason"),
        "timeout": staged._int(deterministic_raw, ("timeout",), 0),
        "self_crossing": staged._int(deterministic_raw, ("self_crossing",), 0),
        "authority_status": "training-only periodic diagnostic; not final held-out",
    })
    return [stochastic, deterministic]


def build_payload() -> dict:
    if not FINAL_DIR.is_dir():
        raise FileNotFoundError(FINAL_DIR)
    existing = [str(path) for path in OUTPUTS.values() if path.exists()]
    if existing:
        raise FileExistsError("refusing supplement overwrite: " + ", ".join(existing))
    main_manifest = verify_main_artifacts()
    main = read_json(MAIN_METRICS)
    formal_run = selected_main_run(main)
    final_summary = read_json(FALLBACK_ROOT / "analysis/final_summary.json")
    authority = combined.authoritative_endpoint(RUN)
    raw_rows, history_errors = staged.read_history(RUN / "training_history.csv", 10000)
    updates, update_errors = staged.read_updates(RUN / "update_history.csv", 10000)
    records = [
        combined.enrich_record(staged.normalise(raw), raw, "water1")
        for raw in raw_rows
    ]
    first_1024 = combined.window_summary(records, updates, 1, 1024)
    last_1024 = combined.window_summary(records, updates, 8977, 10000)
    last_512 = combined.window_summary(records, updates, 9489, 10000)
    config = fallback.validate_fallback_config(RUN / "run_config.json", 4301)
    evaluations = evaluation_payload()
    global_decomposition = max(
        [row["decomposition_max"] for row in records] or [math.inf]
    )
    global_nonfinite = sum(row["nonfinite_count"] for row in records)
    max_ordinary = max(
        [abs(row["zero_reject_ordinary_reward"]) for row in records] or [math.inf]
    )
    checks = [
        {
            "check": "formal authority remains periodic atomic 9728",
            "pass": authority.get("episode") == 9728
            and authority.get("authority") == "periodic_atomic_checkpoint"
            and int(final_summary.get("endpoint", 0)) == 9728
            and int(formal_run.get("endpoint", 0)) == 9728,
            "detail": json.dumps({
                "postprocessor_authority": authority,
                "runner_final_endpoint": final_summary.get("endpoint"),
                "main_formal_endpoint": formal_run.get("endpoint"),
            }, ensure_ascii=False),
        },
        {
            "check": "training/update rows are contiguous through 10000",
            "pass": not history_errors and not update_errors
            and len(raw_rows) == 10000 and len(updates) == 1250,
            "detail": f"history={len(raw_rows)}, updates={len(updates)}, "
            f"history_errors={history_errors}, update_errors={update_errors}",
        },
        {
            "check": "policy/history/update reached 10000 but final summary is absent",
            "pass": authority.get("policy_episode") == 10000
            and authority.get("history_episode") == 10000
            and authority.get("update_episode") == 10000
            and not (RUN / "summary.json").exists(),
            "detail": json.dumps(authority, ensure_ascii=False),
        },
        {
            "check": "full training-only reward/config integrity",
            "pass": bool(config.get("pass")) and global_decomposition <= 1.0e-8
            and global_nonfinite == 0 and max_ordinary <= 1.0e-12,
            "detail": f"config={config.get('pass')}, max_decomp={global_decomposition:.17g}, "
            f"nonfinite={global_nonfinite}, max_ordinary_reject_reward={max_ordinary:.17g}",
        },
        {
            "check": "episode-10000 evaluation is periodic n=16, never held-out",
            "pass": evaluations[0]["episode"] == 10000
            and evaluations[0]["sample_count"] == 16
            and evaluations[0]["evaluation_mode"] == "periodic_stochastic"
            and evaluations[1]["evaluation_mode"] == "periodic_deterministic_single",
            "detail": json.dumps(evaluations, ensure_ascii=False),
        },
    ]
    source_hashes = exact_input_hashes(input_paths())
    return {
        "schema_version": 1,
        "analysis_type": "water_training_only_10000_non_authority_supplement",
        "generated_utc": combined.utc_now(),
        "authority_warning_ja": (
            "これはtraining-only episode 10000補遺であり、正式な最終authorityではない。"
            "final held-out 64-rolloutとsummary.jsonが未完了のため、正式結論はatomic checkpoint 9728。"
        ),
        "formal_authority": {
            "episode": 9728,
            "main_report": str(MAIN_REPORT.resolve()),
            "main_metrics": str(MAIN_METRICS.resolve()),
            "main_artifact_manifest_sha256": combined.file_sha256(MAIN_MANIFEST),
            "runner_final_summary": final_summary,
            "endpoint_evidence": authority,
        },
        "training_only_10000": {
            "seed": 4301,
            "first_1024": first_1024,
            "last_1024": last_1024,
            "last_512": last_512,
            "evaluations": evaluations,
            "global_max_abs_reward_decomposition_error": global_decomposition,
            "global_nonfinite_reward_component_count": global_nonfinite,
            "global_max_abs_ordinary_reject_reward": max_ordinary,
        },
        "formal_9728_last_1024": formal_run["last_1024"],
        "comparison_training10000_minus_formal9728": {
            "mean_return_delta": (
                last_1024["mean_return"] - formal_run["last_1024"]["mean_return"]
            ),
            "median_flatness_delta": (
                last_1024["median_flatness"]
                - formal_run["last_1024"]["median_flatness"]
            ),
            "median_spread_error_delta": (
                last_1024["median_spread_error"]
                - formal_run["last_1024"]["median_spread_error"]
            ),
            "strict_rate_delta": (
                last_1024["strict_rate"] - formal_run["last_1024"]["strict_rate"]
            ),
        },
        "integrity": {"pass": all(item["pass"] for item in checks), "checks": checks},
        "input_sha256": source_hashes,
        "input_hash_set_sha256": combined.canonical_digest(source_hashes),
        "main_manifest_verified_artifact_count": len(
            main_manifest.get("artifact_sha256", {})
        ),
    }


def csv_rows(payload: dict) -> list[dict]:
    training = payload["training_only_10000"]
    rows = []
    for label, window in (
        ("formal_authority_9728_last1024", payload["formal_9728_last_1024"]),
        ("training_only_10000_last1024", training["last_1024"]),
        ("training_only_10000_last512", training["last_512"]),
    ):
        rows.append({"row_type": "window", "label": label, **window})
    for evaluation in training["evaluations"]:
        rows.append({
            "row_type": "evaluation", "label": evaluation["evaluation_mode"],
            **evaluation,
        })
    for check in payload["integrity"]["checks"]:
        rows.append({"row_type": "integrity", "label": check["check"], **check})
    return rows


def report_text(payload: dict) -> str:
    training = payload["training_only_10000"]
    last = training["last_1024"]
    last512 = training["last_512"]
    stochastic, deterministic = training["evaluations"]
    return "\n".join([
        "# water training-only episode 10000 非authority補遺",
        "",
        "> **重要:** これは正式な最終結果ではない。final held-out 64-rolloutと"
        "`summary.json`が安全期限までに完了しなかったため、正式authorityは"
        "atomic checkpoint **9728** のままである。",
        "",
        f"正式報告: [{MAIN_REPORT.name}]({MAIN_REPORT.name})",
        "",
        "## training-only 10000 の再計算値",
        "",
        f"- last1024 (ep8977–10000): mean return {last['mean_return']:.6g}, "
        f"terminal F median {last['median_flatness']:.6g}, spread error median "
        f"{last['median_spread_error']:.6g}, strict {last['strict_count']}/1024 "
        f"({last['strict_rate']:.1%})。",
        f"- last512 (ep9489–10000): strict {last512['strict_count']}/512 "
        f"({last512['strict_rate']:.1%})。",
        f"- periodic stochastic ep10000: n={stochastic['sample_count']}, success "
        f"{stochastic['success_rate']:.1%}, self-crossing "
        f"{stochastic['self_crossing_rate']:.1%}, mean return "
        f"{stochastic['mean_return']:.6g}。これはfinal held-outではない。",
        f"- periodic deterministic ep10000: success "
        f"{deterministic['success_rate']:.0%}, termination="
        f"{deterministic.get('termination_reason')}, F="
        f"{deterministic['mean_flatness']:.6g}, eD="
        f"{deterministic['mean_spread_error']:.6g}。1 trajectoryであり成功率推定ではない。",
        "",
        "## authorityを9728に維持する理由",
        "",
        "policy.pt、training_history.csv、update_history.csvは10000まで到達したが、"
        "final held-out評価中にdeadline stopとなりsummary.jsonが存在しない。"
        "したがってcomplete final-policy setの条件を満たさず、checkpoint.pt=9728を"
        "正式authorityとして保守的に採用する。10000値は学習trajectoryの補足証拠に限定する。",
        "",
        "## 整合性",
        "",
        f"- supplement integrity: {'PASS' if payload['integrity']['pass'] else 'FAIL'}",
        f"- input hash set SHA-256: `{payload['input_hash_set_sha256']}`",
        f"- main artifact manifest SHA-256: "
        f"`{payload['formal_authority']['main_artifact_manifest_sha256']}`",
        "- raw training/update/evaluation CSV、policy/checkpoint、formal final summary、"
        "main report/metrics/manifest、supplement source/testをSHA-256でbindした。",
        "",
    ])


def verify_inputs(payload: dict) -> None:
    current = exact_input_hashes([Path(path) for path in payload["input_sha256"]])
    if current != payload["input_sha256"]:
        raise RuntimeError("supplement input changed during generation")


def write(payload: dict) -> dict:
    combined.atomic_json(OUTPUTS["json"], combined.serialisable(payload))
    combined.write_csv(OUTPUTS["csv"], csv_rows(payload))
    temporary = OUTPUTS["report"].with_suffix(".md.tmp")
    temporary.write_text(report_text(payload), encoding="utf-8")
    temporary.replace(OUTPUTS["report"])
    verify_inputs(payload)
    artifacts = {
        str(path.resolve()): combined.file_sha256(path)
        for key, path in OUTPUTS.items() if key != "manifest"
    }
    source_hashes = {
        str(path.resolve()): combined.file_sha256(path)
        for path in (
            Path(__file__),
            HERE / "test_finalize_ori_codex6_water_training_only_supplement.py",
        )
    }
    combined.atomic_json(OUTPUTS["manifest"], {
        "schema_version": 1,
        "generated_utc": combined.utc_now(),
        "authority_warning": "training-only 10000 is non-authority; formal endpoint=9728",
        "main_artifact_manifest_sha256": combined.file_sha256(MAIN_MANIFEST),
        "input_hash_set_sha256": payload["input_hash_set_sha256"],
        "artifact_sha256": artifacts,
        "postprocessor_source_sha256": source_hashes,
    })
    return {
        **{key: str(path.resolve()) for key, path in OUTPUTS.items()},
        "manifest_sha256": combined.file_sha256(OUTPUTS["manifest"]),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-ready", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    options = parse_args(argv)
    if options.check_ready:
        verify_main_artifacts()
        existing = [str(path) for path in OUTPUTS.values() if path.exists()]
        print(json.dumps({
            "ready": not existing,
            "formal_endpoint": read_json(
                FALLBACK_ROOT / "analysis/final_summary.json"
            ).get("endpoint"),
            "supplement_outputs_absent": not existing,
            "existing": existing,
        }, ensure_ascii=False, indent=2))
        return 0 if not existing else 3
    payload = build_payload()
    if not payload["integrity"]["pass"]:
        raise RuntimeError("supplement integrity failed; refusing output")
    print(json.dumps({
        "integrity_pass": True,
        "formal_endpoint": 9728,
        "training_only_endpoint": 10000,
        "outputs": write(payload),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
