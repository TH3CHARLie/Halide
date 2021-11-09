"""
Generate a large volume of samples for a specified app
"""

import sys
import argparse
from utils import set_up_environment


def generate_data(app: str, output_dir: str, batch_size: int, num_batch: int):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--app", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, required=False, default=40)
    parser.add_argument("--num-batch", type=int, required=False, default=50)
    args = parser.parse_args()

    # setting env vars
    set_up_environment(args.config)
    generate_data()