"""Target-spread-3 entry point for the ori_codex4 experiment."""

from __future__ import annotations

import sys

import ori_codex4


def parse_args(argv=None):
    user_arguments = sys.argv[1:] if argv is None else list(argv)
    return ori_codex4.parse_args([
        "--target-spread", "3.0",
        "--output-dir",
        (
            "ori_codex4_2_target_spread_3_h010_valid_terminal_near_miss_"
            "actor_critic_10000_results"
        ),
        *user_arguments,
    ])


if __name__ == "__main__":
    ori_codex4.train(parse_args())
