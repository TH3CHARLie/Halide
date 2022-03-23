import os
import struct

filename = "../apps/bilateral_grid/autotuned_samples-2022-01-24-22-27-46/batch_216_0/2/bilateral_grid_batch_0216_sample_0002.newsample"
NUM_STAGES = 9
HEAD2_W = 39
FEATURES_PER_STAGE = 326
MAGIC_IDX = 4

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

