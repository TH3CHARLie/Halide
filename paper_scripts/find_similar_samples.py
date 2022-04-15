import argparse
import sys
import pprint
import math
import os
import struct

TARGET_STAGE = "blury"
TARGET_STAGE_IDX = 2

NUM_STAGES = 9
HEAD2_W = 39
FEATURES_PER_STAGE = 326

def parse_schedule_features_from_sample(filename):
    size = os.stat(filename).st_size
    num_floats = size // 4
    with open(filename, "rb") as f:
        raw_results = list(struct.unpack('f'* num_floats, f.read(4 * num_floats)))
        schedule_features = []
        for i in range(NUM_STAGES):
            stage_features = []
            for x in range(HEAD2_W):
                f = raw_results[i * FEATURES_PER_STAGE + x]
                stage_features.append(f)
            schedule_features.append(stage_features)
        return schedule_features


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
    return stages

def compute_diff(name1, feat1, name2, feat2):
    # print(f"computing feature diff between {name1} and {name2}")
    feat1 = feat1[TARGET_STAGE_IDX]
    feat2 = feat2[TARGET_STAGE_IDX]
    diff = 0
    for k in range(len(feat1)):
        v1 = feat1[k]
        v2 = feat2[k]
        diff += abs(v1 - v2)
    # print(f"feature diff between {name1} and {name2} is {diff}")
    return diff

def parse_runtimes(runtime_file, stage_name):
    with open(runtime_file, "r") as f:
        for line in f:
            tokens = line.split()
            name = tokens[0]
            runtime = float(tokens[1])
            if name == stage_name:
                return runtime
        return 0.0

def main():
    compile_log_prefix = "autotuned_samples-2021-11-22-15-35-02"
    sample_prefix = "autotuned_samples-2022-01-24-22-27-46"
    sample_file = sys.argv[1]
    prediction_files = sys.argv[2:]
    lines = []
    with open(sample_file) as f:
        for line in f:
            lines.append(line.strip())
    sample_paths = []
    for line in lines:
        tokens = line.split()
        sample_paths.append(tokens[0])
    names = []
    features = []
    runtimes = []
    sample_names = [x[x.rfind('/') + 1:] for x in sample_paths]
    sample_folders = [x[:x.rfind('/') + 1] for x in sample_paths]
    # compile_log_folders = [x.replace(sample_prefix, compile_log_prefix) for x in sample_folders]
    # for name, folder in zip(sample_names, compile_log_folders):
    #     compile_log_file = folder + "compile_log.txt"
    #     feature_dict = parse_compile_log(name, compile_log_file)
    #     names.append(name)
    #     features.append(feature_dict)
    for name, path in zip(sample_names, sample_paths):
        sample_features = parse_schedule_features_from_sample(path)
        names.append(name)
        features.append(sample_features)
    for name, folder in zip(sample_names, sample_folders):
        runtime_file = folder + "runtimes.txt"
        runtime = parse_runtimes(runtime_file, TARGET_STAGE)
        runtimes.append(runtime)

    predicted_runtimes = {}
    for prediction_file in prediction_files:
        with open(prediction_file, "r") as f:
            for line in f:
                line = line.strip()
                tokens = line.split()
                x = tokens[0]
                sample_name = x[x.rfind('/') + 1:-1]
                predicted = float(tokens[1][:-1])
                if sample_name in predicted_runtimes:
                    print(f"{sample_name} in {prediction_file} already appeared before")
                predicted_runtimes[sample_name] = predicted
            print(f"{prediction_file} collected, {len(predicted_runtimes)} collected so far")
    diffs_with_idx = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = compute_diff(names[i], features[i], names[j], features[j])
            diffs_with_idx.append((diff, (i, j)))
    sorted_diffs = sorted(diffs_with_idx, key=lambda x: x[0])
    TOP_K = 1000
    print(f"dump top {TOP_K} minimum diff sample pairs")
    i = 0
    cnt = 0
    candidates = []
    while cnt < TOP_K and i < len(sorted_diffs):
        diff = sorted_diffs[i][0]
        idx_i, idx_j = sorted_diffs[i][1]
        i += 1
        name_1 = names[idx_i]
        name_2 = names[idx_j]
        if name_1 in predicted_runtimes and name_2 in predicted_runtimes:
            predicted_1 = predicted_runtimes[name_1]
            predicted_2 = predicted_runtimes[name_2]
            actual_1 = runtimes[idx_i]
            actual_2 = runtimes[idx_j]
            if diff < 200 and actual_1 != actual_2:
                cnt += 1
                factor = max(actual_1, actual_2) / min(actual_1, actual_2)
                candidates.append((name_1, name_2, diff, actual_1, actual_2, factor))
    candidates = sorted(candidates, key=lambda x: x[5], reverse=True)
    for c in candidates:
        name_1, name_2, diff, actual_1, actual_2, factor = c 
        print("diff between {} and {}: {}, runtimes: {:.3f}, {:.3f} factor: {:.3f}".format(name_1, name_2, diff, actual_1, actual_2, factor))
    num_pairs = 0
    for x in sorted_diffs:
        diff = x[0]
        idx_i, idx_j = x[1]
        if diff == 0 and (runtimes[idx_i] != runtimes[idx_j]):
            num_pairs += 1
    print(f"{num_pairs} pairs")

if __name__ == "__main__":
    main()