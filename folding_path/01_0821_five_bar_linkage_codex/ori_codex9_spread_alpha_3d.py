"""Three-dimensional water-D=1 adapter for the Codex9 spread-alpha screen."""

from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path
import sys

import ori_codex8 as ori8
import ori_codex8_3d as ori8_3d
import ori_codex9_spread_alpha as ori9


ADAPTER_NAME = "ori_codex9_spread_alpha_3d"
ADAPTER_VERSION = "1.0.0"


@contextmanager
def patched_three_dimensional_alpha_runtime():
    original = (
        ori8_3d._BASE_ORI8_MAKE_ENVIRONMENT,
        ori8_3d._BASE_ORI8_RUN_EPISODE,
        ori8_3d._BASE_ORI8_AGGREGATE,
        ori8.reward_parameters,
    )
    ori8_3d._BASE_ORI8_MAKE_ENVIRONMENT = ori9.make_environment
    ori8_3d._BASE_ORI8_RUN_EPISODE = ori9.run_episode
    ori8_3d._BASE_ORI8_AGGREGATE = ori9.aggregate_stochastic_evaluations
    ori8.reward_parameters = ori9.reward_parameters
    try:
        yield
    finally:
        (
            ori8_3d._BASE_ORI8_MAKE_ENVIRONMENT,
            ori8_3d._BASE_ORI8_RUN_EPISODE,
            ori8_3d._BASE_ORI8_AGGREGATE,
            ori8.reward_parameters,
        ) = original


def train(args):
    ori9._validate_ori9_args(args)
    parameters = ori9.reward_parameters(args)
    with patched_three_dimensional_alpha_runtime():
        output_directory, summary = ori8_3d.train(args)

    output_directory = Path(output_directory)
    additions = {
        "ori9_candidate": ori9.CANDIDATE_NAME,
        "ori9_candidate_schema_version": ori9.CANDIDATE_SCHEMA_VERSION,
        "ori9_parent_candidate": ori8.CANDIDATE_NAME,
        "ori9_reward_formula": copy.deepcopy(ori9.REWARD_FORMULA),
        "ori9_reward_formula_sha256": ori9.REWARD_FORMULA_SHA256,
        "ori9_spread_alpha": float(args.ori9_spread_alpha),
        "ori9_reward_parameters": parameters,
        "candidate_entrypoint": Path(__file__).name,
        "parent_sources_unmodified": True,
        "adapter": ADAPTER_NAME,
        "adapter_version": ADAPTER_VERSION,
    }
    summary.update(additions)
    config_path = output_directory / "run_config.json"
    with open(config_path, encoding="utf-8") as handle:
        config = json.load(handle)
    config.update(additions)
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    with open(output_directory / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return output_directory, summary


def parse_args(argv=None):
    arguments = sys.argv[1:] if argv is None else list(argv)
    selected, remaining = ori9._ori9_parser().parse_known_args(arguments)
    args = ori8_3d.parse_args(remaining)
    ori9._bind_ori9_args(args, selected)
    return args


if __name__ == "__main__":
    train(parse_args())
