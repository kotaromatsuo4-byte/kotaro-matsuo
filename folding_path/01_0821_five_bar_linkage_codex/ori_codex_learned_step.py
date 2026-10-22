"""Method 2: NN mode mixture and step width via state -> modes -> Tanh -> 2."""

import argparse

from single_solution_common import add_common_arguments, run


def parse_args():
    parser = argparse.ArgumentParser()
    parser = add_common_arguments(
        parser, "single_solution_learned_step_least_squares_line_10000_results"
    )
    parser.set_defaults(
        batch_size=100,
        max_steps=50,
        max_step_width=0.35,
        gamma=0.99,
        learning_rate=1e-3,
        entropy_coefficient=0.001,
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args(), method="learned_step")
