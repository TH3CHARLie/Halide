import sys
import argparse
import os
from time import sleep

BENCHMARK_TIMES = 5
GAP = 13

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
    for batch, sample in zip(batches, samples):
        directory = f"{prefix}/batch_{batch}_0/{sample}/"
        print("benchmarking " + directory)
        for i in range(BENCHMARK_TIMES):
            print(f"times {i}")
            command = f"HL_NUM_THREADS=20 {directory}/bench --estimate_all --benchmarks=all --benchmark_min_time=100 | tee {directory}/bench_{i+GAP}.txt"
            os.system(command)
            sleep(1)
    print("reverse order")
    for batch, sample in zip(batches[::-1], samples[::-1]):
        directory = f"{prefix}/batch_{batch}_0/{sample}/"
        print("benchmarking " + directory)
        for i in range(BENCHMARK_TIMES):
            print(f"times {i}")
            command = f"HL_NUM_THREADS=20 {directory}/bench --estimate_all --benchmarks=all --benchmark_min_time=100 | tee {directory}/bench_{i+GAP+BENCHMARK_TIMES}.txt"
            os.system(command)
            sleep(1)


def main():
    infile = sys.argv[1]
    prefix = sys.argv[2]
    batches, samples, _ = parse_seeds(infile)
    benchmark_selected(prefix, batches, samples)

if __name__ == "__main__":
    main()

