"""Target-spread-3 entry point for the ori_codex3 experiment."""

from __future__ import annotations

import sys

import ori_codex3


def parse_args():
    # Prepending defaults lets an explicit user option later on the command
    # line override either value in the usual argparse manner.
    return ori_codex3.parse_args([
        "--target-spread", "3.0",
        "--output-dir",
        (
            "ori_codex3_2_h010_batch8_target_spread_3_"
            "actor_critic_10000_results"
        ),
        *sys.argv[1:],
    ])


if __name__ == "__main__":
    ori_codex3.train(parse_args())
