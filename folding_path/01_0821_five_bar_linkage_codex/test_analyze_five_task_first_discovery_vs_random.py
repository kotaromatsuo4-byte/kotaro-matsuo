import csv
from pathlib import Path

import analyze_five_task_first_discovery_vs_random as analysis


def write_history(path: Path, steps, hits, policy=""):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        fields = ["episode", "steps", "projection_attempts", "strict_compact_success"]
        if policy:
            fields += ["ori6_crossing_policy", "ori6_zero_reject_total_reward"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, (step, hit) in enumerate(zip(steps, hits), 1):
            row = {
                "episode": index,
                "steps": step,
                "projection_attempts": max(0, step - 1),
                "strict_compact_success": int(hit),
            }
            if policy:
                row.update({"ori6_crossing_policy": policy, "ori6_zero_reject_total_reward": 0.0})
            writer.writerow(row)


def test_learning_first_hit_uses_cumulative_action_proposals(tmp_path):
    history = tmp_path / "history.csv"
    write_history(history, [4, 3, 5], [False, True, False])
    spec = analysis.TaskSpec(
        "five_d3", 7, "historical_ori5_actor_critic",
        (analysis.LearningRun(1, "history.csv", True),), "test", (6101,),
    )
    manifest = []
    record, errors = analysis.read_learning_history(tmp_path, spec, spec.learning_runs[0], manifest)
    assert not errors
    assert record["first_hit_proposals"] == 7
    assert record["first_hit_projector_calls"] == 5
    assert record["first_hit_episode"] == 2


def test_capped_rmst_and_strict_random_faster_tie():
    spec = analysis.TaskSpec("five_d3", 10, "learn", tuple(), "test", (6101, 6102, 6103))
    rows = [
        {"task": "five_d3", "arm": "learning", "available": True, "adequate_followup": True,
         "event": 1, "first_hit_proposals": 10, "observed_proposals": 10, "primary": True, "seed": 1},
        {"task": "five_d3", "arm": "random", "available": True, "adequate_followup": True,
         "event": 1, "first_hit_proposals": 4, "observed_proposals": 4, "seed": 6101},
        {"task": "five_d3", "arm": "random", "available": True, "adequate_followup": True,
         "event": 1, "first_hit_proposals": 10, "observed_proposals": 10, "seed": 6102},
        {"task": "five_d3", "arm": "random", "available": True, "adequate_followup": True,
         "event": 0, "first_hit_proposals": None, "observed_proposals": 10, "seed": 6103},
    ]
    summary = analysis.summarize_task(spec, rows)
    assert summary["random_faster_count"] == 1
    assert summary["random_tie_count"] == 1
    assert summary["random"]["capped_rmst_proposals"] == 8
    assert summary["random_rmst_over_primary_learning_hit"] == 0.8


def test_zero_reject_policy_and_reward_are_audited(tmp_path):
    history = tmp_path / "history.csv"
    policy = "reject_candidate_as_zero_reward_noop_and_continue"
    write_history(history, [5], [True], policy)
    spec = analysis.TaskSpec(
        "torus2", 5, "ori6_zero_reject_actor_critic",
        (analysis.LearningRun(1, "history.csv", True),), "test", (6101,),
    )
    record, errors = analysis.read_learning_history(tmp_path, spec, spec.learning_runs[0], [])
    assert not errors
    assert record["crossing_policies"] == [policy]
    assert record["max_abs_zero_reject_reward"] == 0.0


def test_historical_random_is_recensored_at_learning_hit(tmp_path):
    relative = analysis.HISTORICAL_RANDOM_CSV["five_d2"]
    authority = tmp_path / relative
    authority.parent.mkdir(parents=True, exist_ok=True)
    with authority.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["search_seed", "success", "mechanical_steps", "first_success_mechanical_steps"],
        )
        writer.writeheader()
        writer.writerow({
            "search_seed": 10, "success": 1, "mechanical_steps": 7,
            "first_success_mechanical_steps": 7,
        })
        writer.writerow({
            "search_seed": 11, "success": 1, "mechanical_steps": 20,
            "first_success_mechanical_steps": 20,
        })
    spec = analysis.TaskSpec("five_d2", 10, "learn", tuple(), "test", (10, 11))
    records, errors = analysis.read_historical_random_trials(tmp_path, (spec,), [])
    assert not errors
    assert records[0]["event"] == 1
    assert records[0]["observed_proposals"] == 7
    assert records[1]["event"] == 0
    assert records[1]["observed_proposals"] == 10
    assert records[1]["source_event"] == 1
    assert records[1]["source_first_hit_proposals"] == 20


def test_utf8_writer_rejects_replacement_character(tmp_path):
    valid = tmp_path / "valid.md"
    analysis.write_utf8_checked(valid, "日本語の報告\n")
    assert valid.read_text(encoding="utf-8") == "日本語の報告\n"
    try:
        analysis.write_utf8_checked(tmp_path / "invalid.md", "bad\ufffdtext")
    except ValueError:
        pass
    else:
        raise AssertionError("replacement character must be rejected")
