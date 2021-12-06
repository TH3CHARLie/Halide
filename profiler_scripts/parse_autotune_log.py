import os
import sys
import argparse
from common import FeatureDict, SampleDict, FuncPerf, ProgramPerf
import pprint
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples', required=True, action="store", help="path to prediction with filename")
    parser.add_argument('--output', required=True, action="store", help="path to csv output")
    args = parser.parse_args()
    return args


def parse_autotune_log(sample_file):
    lines = []
    with open(sample_file) as f:
        for line in f:
            lines.append(line.strip())
    actual_runtimes = []
    predicted_runtimes = []
    sample_paths = []
    for line in lines:
        tokens = line.split()
        sample_paths.append(tokens[0][:-1])
        predicted_runtimes.append(float(tokens[1][:-1]))
        actual_runtimes.append(float(tokens[2]))

    sample_names = [x[x.rfind('/') + 1:] for x in sample_paths] 
    sample_folders = [x[:x.rfind('/') + 1] for x in sample_paths]

    samples = []
    for name, folder, actual_runtime, predicted_runtime in zip(sample_names, sample_folders, actual_runtimes, predicted_runtimes):
        compile_log_file = folder + 'compile_log.txt'
        benchmark_log_file = folder + 'bench.txt'
        feature_dict = parse_compile_log(name, compile_log_file)
        perf_dict = parse_benchmark_log(name, benchmark_log_file)
        sample_dict = SampleDict(name, actual_runtime, predicted_runtime, feature_dict, perf_dict)
        samples.append(sample_dict)
    return samples


def parse_compile_log(sample_name, compile_log_file):
    lines = []
    with open(compile_log_file, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    prev = -1
    stages = {}
    flag = False
    for i in range(len(lines)):
        if lines[i].find("Schedule features for") != -1 or (flag and lines[i].find("State with ") != -1):
            if lines[i].find("Schedule features for") != -1:
                flag = True
            if prev != -1:
                # process prev to this point
                stage_name = lines[prev].split()[3]
                stage_features = {}
                for line in lines[prev + 1:i]:
                    tokens = line.split()
                    feature_name = tokens[0][:-1]
                    value = float(tokens[1])
                    stage_features[feature_name] = value
                stages[stage_name] = stage_features
                prev = i
            else:
                prev = i
    # feature_dict = FeatureDict(stages)
    # return feature_dict
    return stages


def parse_benchmark_log(sample_name, compile_log_file):
    lines = []
    with open(compile_log_file, "r") as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines]
    # skip "Benchmark"
    lines = lines[2:]
    avg_threads = float(lines[2].split()[3])
    heap_allocations = float(lines[3].split()[2])
    peak_heap_usage = float(lines[3].split()[6])
    func_perf_counters = []
    for line in lines[4:]:
        tokens = line.split()
        stage_name = tokens[1][:-1]
        stage_runtime = float(tokens[2][:-2])
        stage_threads = float(tokens[5])
        stage_peak_usage = -1
        stage_stack_space = -1
        stage_num_allocations = -1
        stage_avg_size_allocations = -1
        idx = 6
        while idx < len(tokens):
            if tokens[idx] == "peak":
                stage_peak_usage = float(tokens[idx + 1])
            elif tokens[idx] == "stack":
                stage_stack_space = float(tokens[idx + 1])
            elif tokens[idx] == "num":
                stage_num_allocations = float(tokens[idx + 1])
            elif tokens[idx] == "avg":
                stage_avg_size_allocations = float(tokens[idx + 1])
            idx += 2
        func_perf_counter = FuncPerf(stage_name, stage_runtime, stage_threads, 
                                            stage_peak_usage, stage_stack_space, 
                                            stage_num_allocations, stage_avg_size_allocations)
        func_perf_counters.append(func_perf_counter)
    program_perf_counter = ProgramPerf(avg_threads, heap_allocations, peak_heap_usage, func_perf_counters)
    return program_perf_counter


def main():
    args = parse_args()
    samples = parse_autotune_log(args.samples)
    with open(args.output, "wb") as f:
        pickle.dump(samples, f)


if __name__ == "__main__":
    main()
