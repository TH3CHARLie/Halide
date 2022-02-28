import sys
import argparse
import os
from time import sleep

BENCHMARK_TIMES = 3


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
        batches.append(int(b))
        samples.append(int(s))
        seeds.append(seed)
    return batches, samples, seeds


def benchmark_selected(prefix, batches, samples):
    DAG = "/home/xuanda/dev/Halide/training_scripts/bilateral_DAG.txt"
    O = "/home/xuanda/dev/Halide/training_scripts/feature_ordering.txt"
    bin = "../bin"
    for batch, sample in zip(batches, samples):
        directory = f"{prefix}/batch_{batch}_0/{sample}"
        B = "{:04d}".format(batch)
        S = "{:04d}".format(sample)
        for i in range(BENCHMARK_TIMES):
            parse_runtime_command = f"python3 /home/xuanda/dev/Halide/training_scripts/parse_runtime.py --input {directory}/bench_{i}.txt --output {directory}/benchmark_runtimes_{i}.txt"
            print(parse_runtime_command)
            os.system(parse_runtime_command)
        # sleep(1)
        parse_command = f'ls -d {directory}/* | python3 /home/xuanda/dev/Halide/training_scripts/aggregate_runtime.py {directory}/runtimes.txt'
        print(parse_command)
        os.system(parse_command)
        fname = f"bilateral_grid_batch_{B}_sample_{S}"
        command = f'{bin}/featurization_to_sample {directory}/{fname}.featurization {directory}/runtimes.txt {DAG} {O} "" {B}{S} {directory}/{fname}.newsample {directory}/{fname}.metadata'
        print(command)
        os.system(command)


def main():
    infile = sys.argv[1]
    prefix = sys.argv[2]
    batches, samples, _ = parse_seeds(infile)
    benchmark_selected(prefix, batches, samples)

if __name__ == "__main__":
    main()

