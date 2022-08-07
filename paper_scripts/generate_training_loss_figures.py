import sys
import os

APPS = ["bgu", "bilateral_grid", "camera_pipe", "hist", "iir_blur", "lens_blur", "local_laplacian", "max_filter", "nl_means", "stencil_chain"]


for app in APPS:
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/plot_training_log.py {app}/training_loss.txt {app}")
    os.system(f"mkdir -p {app}/figures/")
    os.system(f"mv training_unique_loss.png {app}/figures")
    os.system(f"mv training_unique_loss_log.png {app}/figures")
