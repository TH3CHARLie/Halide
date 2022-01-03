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
    batches, samples, seeds = [], [], []
    for x in sample_names:
        b, s, seed = compute_seed(x)
        batches.append(b)
        samples.append(s)
        seeds.append(seed)
    return batches, samples, seeds


def run(batches, samples, seeds):
    for batch, sample, seed in zip(batches, samples, seeds):
        batch = int(batch)
        sample = int(sample)
        os.system(f"export MY_SEED={seed} && cd ~/dev/Halide/paper_scripts && bash generate_autotune_results.sh 1 0 0 0 0 && mv ~/dev/Halide/apps/bilateral_grid/autotuned_samples/batch_1_0/0/counters.txt ~/dev/Halide/apps/bilateral_grid/autotuned_samples-2021-11-22-15-35-02/batch_{batch}_0/{sample}/ && rm -rf ~/dev/Halide/apps/bilateral_grid/autotuned_samples/" )



def main():
    args = parse_args()
    batches, samples, seeds = parse_seeds(args.samples)
    run(batches, samples, seeds)

if __name__ == "__main__":
    main()