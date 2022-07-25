import sys
import os

APPS = ["bgu", "bilateral_grid", "camera_pipe", "hist", "iir_blur", "local_laplacian", "max_filter", "nl_means", "stencil_chain"]

BATCH_SIZE = 40

NUM_BATCH = 1000

for app in APPS:
    mkdir_command = f"mkdir -p /home/xuanday/dev/data-train-from-scratch/{app}/"
    os.system(mkdir_command)
    for i in range(1, NUM_BATCH + 1):
        for j in range(0, BATCH_SIZE):
            format_i = "{:04d}".format(i)
            format_j = "{:04d}".format(j)
            command = f"cp /sensei-fs/users/xuanday/data/{app}/autotuned_samples/batch_{i}_0/{j}/{app}_batch_{format_i}_sample_{format_j}.sample /home/xuanday/dev/data-train-from-scratch/{app}"
            print(command)
            os.system(command)