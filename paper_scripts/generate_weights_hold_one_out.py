import sys
import os

APPS = ["bgu", "bilateral_grid", "camera_pipe", "hist", "iir_blur", "lens_blur", "local_laplacian", "max_filter", "nl_means", "stencil_chain"]

for app in APPS:
    command = f'find /home/xuanda/dev/data-profiler-off/ -name "{app}_*.sample" > {app}_samples.txt'
    os.system(command)

for app in APPS:
    mkdir_command = f"mkdir -p {app}/"
    os.system(mkdir_command)
    for other_app in APPS:
        if app == other_app:
            continue
        os.system(f"cat {other_app}_samples.txt >> {app}/{app}_hold_one_out_samples.txt")
    print(f"Training hold-one-out weights for {app}")
    os.system(f"bash /home/xuanda/dev/Halide/paper_scripts/train_unique.sh . wrong.weights {app}/updated_1e-3.weights {app}/training_prediction 15000 {app}/{app}_hold_one_out_samples.txt 0.001 | tee {app}/training_loss.txt")
