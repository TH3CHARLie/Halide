import argparse
import sys
import pprint
import math
import os
import struct

TARGET_STAGE = "blurz"
TARGET_STAGE_IDX = 4

NUM_STAGES = 9
HEAD2_W = 39
FEATURES_PER_STAGE = 326

schedule_feature_names = [
    "num_realizations",
    "num_productions",
    "points_computed_per_realization",
    "points_computed_per_production",
    "points_computed_total",
    "points_computed_minimum",
    "innermost_loop_extent",
    "innermost_pure_loop_extent",
    "unrolled_loop_extent",
    "inner_parallelism",
    "outer_parallelism",
    "bytes_at_realization",
    "bytes_at_production",
    "bytes_at_root",
    "innermost_bytes_at_realization",
    "innermost_bytes_at_production",
    "innermost_bytes_at_root",
    "inlined_calls",
    "unique_bytes_read_per_realization",
    "unique_lines_read_per_realization",
    "allocation_bytes_read_per_realizatio",
    "working_set",
    "vector_size",
    "native_vector_size",
    "num_vectors",
    "num_scalars",
    "scalar_loads_per_vector",
    "vector_loads_per_vector",
    "scalar_loads_per_scalar",
    "bytes_at_task",
    "innermost_bytes_at_task",
    "unique_bytes_read_per_vector",
    "unique_lines_read_per_vector",
    "unique_bytes_read_per_task",
    "unique_lines_read_per_task",
    "working_set_at_task",
    "working_set_at_production",
    "working_set_at_realization",
    "working_set_at_root"
]

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

def main():
    sample_path = sys.argv[1]
    sample_name = sample_path[sample_path.rfind('/') + 1:]
    sample_features = parse_schedule_features_from_sample(sample_path)
    print(f"sample_features of {TARGET_STAGE} for sample {sample_path}")
    for (feat, feat_name) in zip(sample_features[TARGET_STAGE_IDX], schedule_feature_names):
        print(f"{feat_name}: {feat}")

if __name__ == "__main__":
    main()