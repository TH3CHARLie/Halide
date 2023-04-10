import sys
import matplotlib.pyplot as plt
import math

"""
Parse excluded pipeline log files and get exclueded and bad pipeline ids
"""

PREFIX = "/home/xuanda/halide-samples/random_pipeline/a/"


def transform_path(path):
    batch_id = int(path[path.find("batch") + 6 : path.find("sample") - 1])
    sample_id = int(path[path.find("sample") + 7 : path.find("sample") + 11])
    new_path = PREFIX + f"batch_{batch_id}/{sample_id}/{path}"
    return new_path

def get_batch_id(path):
    batch_id = int(path[path.find("batch") + 6 : path.find("sample") - 1])
    return batch_id

if __name__ == "__main__":
    infile = sys.argv[1]
    paths = []
    excluded_paths = []
    very_bad_paths = []
    remaining_paths = []
    lines = []
    with open(infile, "r") as f:
        for line in f:
            lines.append(line)
    bad_batch_ids = set()
    excluded_batch_ids = set()
    remaining_batch_ids = set()
    for i, line in enumerate(lines):
        prev_tokens = []
        if i != 0:
            prev_tokens = lines[i - 1].split()
        tokens = line.split()
        if len(tokens) > 1 and tokens[0] == "Unique":
            paths.append(tokens[2])
        if len(tokens) > 1 and tokens[0] == "excluding" and tokens[1] == "schedule:":
            batch_id = get_batch_id(tokens[2])
            excluded_paths.append(tokens[2])
            if (len(prev_tokens) > 1 and prev_tokens[0] == "excluding" and prev_tokens[1] == "pipeline" and float(prev_tokens[11]) < 0.4):
                bad_batch_ids.add(batch_id)
                very_bad_paths.append(tokens[2])
            elif batch_id in bad_batch_ids:
                very_bad_paths.append(tokens[2])
            
            

    for path in paths:
        if path not in excluded_paths:
            remaining_paths.append(path)
    with open("excluded_samples.txt", "w") as o:
        for path in excluded_paths:
            batch_id = get_batch_id(path)
            excluded_batch_ids.add(batch_id)
            o.write(transform_path(path) + '\n')
    with open("remaining_samples.txt", "w") as o:
        for path in remaining_paths:
            batch_id = get_batch_id(path)
            remaining_batch_ids.add(batch_id)
            o.write(transform_path(path) + '\n')
    with open("bad_samples.txt", "w") as o:
        for path in very_bad_paths:
            o.write(transform_path(path) + '\n')

    print(list(excluded_batch_ids))
    print()
    print(list(remaining_batch_ids))
    