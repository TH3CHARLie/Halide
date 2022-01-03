#!/usr/bin/env python
import os
import sys
import argparse
import subprocess

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True, action="store", help="path to prediction with filename")
    args = parser.parse_args()
    return args

def compute_seed(sample_name):
    """
    bilateral_grid_batch_0055_sample_0018.sample
    """
    tokens = sample_name.split('_')
    batch = tokens[3]
    sample = tokens[5].split('.')[0]
    seed = batch + sample
    return batch, sample, seed


def parse_seeds(sample_file):
    lines = []
    with open(sample_file) as f:
        for line in f:
            lines.append(line.strip())
    sample_paths = []
    for line in lines:
        tokens = line.split()
        sample_paths.append(tokens[0][:-1])

    sample_names = [x[x.rfind('/') + 1:] for x in sample_paths]
    sample_folders = [x[:x.rfind('/') + 1] for x in sample_paths]
    for name, folder in zip(sample_names, sample_folders):
        compile_log_file = folder + 'compile_log.txt'
        _, _, seed = compute_seed(name)
        os.system(f"cp {compile_log_file} ~/dev/Halide/seeds/compile_log_{seed}.txt")


def main():
    args = parse_args()
    parse_seeds(args.samples)

if __name__ == "__main__":
    main()