"""Method 1: NN mode mixture with flatness-potential line-search width."""

import argparse

from single_solution_common import add_common_arguments, run


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_common_arguments(
        parser, "single_solution_linesearch_least_squares_line_10000_results"
    )
    parser.set_defaults(
        batch_size=100,
        max_steps=50,
        max_step_width=0.25,
        gamma=0.98,
        learning_rate=3e-4,
        entropy_coefficient=0.001,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args(), method="line_search")
