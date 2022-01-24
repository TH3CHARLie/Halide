#!/usr/bin/env python
import os
import sys
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, action="store", help="existing sample directory")
    parser.add_argument('--batch', required=True, action="store", help="number of batches")
    parser.add_argument('--size', required=True, action="store", help="size of a batch")
    parser.add_argument('--bin', required=True, action="store", help="autoschedule binary directory")
    args = parser.parse_args()
    return args


def run_featurization(dir, num_batch, batch_size, bin):
    command_template = ""
    DAG = "/home/xuanda/dev/Halide/training_scripts/bilateral_DAG.txt"
    O = "/home/xuanda/dev/Halide/training_scripts/feature_ordering.txt"
    for i in range(num_batch):
        for j in range(batch_size):
            B = "{:04d}".format(i + 1)
            S = "{:04d}".format(j)
            per_sample_dir = f"{dir}/batch_{i + 1}_0/{j}"
            fname = f"bilateral_grid_batch_{B}_sample_{S}"
            parse_command = f'python3 /home/xuanda/dev/Halide/training_scripts/parse_runtime.py --input {per_sample_dir}/bench.txt --output {per_sample_dir}/runtime.txt'
            os.system(parse_command)
            command = f'{bin}/featurization_to_sample {per_sample_dir}/{fname}.featurization {per_sample_dir}/runtime.txt {DAG} {O} "" {B}{S} {per_sample_dir}/{fname}.newsample {per_sample_dir}/{fname}.metadata'
            print(command)
            os.system(command)

def main():
    args = parse_args()
    run_featurization(args.dir, int(args.batch), int(args.size), args.bin)


if __name__ == "__main__":
    main()