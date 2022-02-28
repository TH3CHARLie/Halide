import sys
import argparse
import os
from time import sleep
import math

BENCHMARK_TIMES = 3

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, action="store", help="path to prediction with filename")
    parser.add_argument('--prefix', required=True, action="store", help="path to the data prefix")
    parser.add_argument('--output', required=True, action="store", help="path to scatter data file")
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

def parse_seeds(lines):
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

def generate_scatter_data(infile, prefix, outfile):
    target = ""
    if infile.find("per_stage") != -1:
        target = infile[infile.find("per_stage") + 10 : -4]
    print(target)
    lines = []
    with open(infile) as f:
        for line in f:
            lines.append(line.strip())
    batches, samples, seeds = parse_seeds(lines)
    gt_data = []
    predicted_data = []
    error_data = []
    for line in lines:
        tokens = line.split()
        print(tokens)
        prediction = float(tokens[1][:-1])
        gt = float(tokens[2])
        gt_data.append(gt)
        predicted_data.append(prediction)
    # compute errors
    errors = []
    cnt = 0
    for batch, sample in zip(batches, samples):
        per_stage_runtimes = []
        direcotry = f"{prefix}/batch_{batch}_0/{sample}"
        print("generate error for " + direcotry)
        for i in range(BENCHMARK_TIMES):
            file = f"{direcotry}/benchmark_runtimes_{i}.txt"
            with open(file) as f:
                cnt = 0
                for line in f:
                    tokens = line.split()
                    name = tokens[0]
                    per_stage = float(tokens[1])
                    if name == target:
                        per_stage_runtimes.append(per_stage)
        print(per_stage_runtimes)
        if len(per_stage_runtimes) > 0:
            acc = 0.0
            for r in per_stage_runtimes:
                acc = acc + (r - gt_data[cnt]) * (r - gt_data[cnt])
            acc /= BENCHMARK_TIMES
            val = math.sqrt(acc)
            errors.append(val)
        else:
            errors.append(0)
        cnt += 1
    with open(outfile, "w") as f:
        for pt, gt, err in zip(predicted_data, gt_data, errors):
            f.write(f"{pt}, {gt}, {err}\n")



def main():
    args = parse_args()
    generate_scatter_data(args.input, args.prefix, args.output)

if __name__ == "__main__":
    main()