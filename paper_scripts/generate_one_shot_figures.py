import sys
import os

APPS = ["bgu", "bilateral_grid", "camera_pipe", "hist", "iir_blur", "lens_blur", "local_laplacian", "max_filter", "nl_means", "stencil_chain"]

for app in APPS:
    os.system(f"bash /home/xuanday/dev/Halide/paper_scripts/train_unique.sh . wrong.weights {app}/tmp.weights {app}/training_prediction 10 {app}_samples.txt 0.001 | tee {app}/tmp.log")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/parse_unique_samples.py {app}/tmp.log /home/xuanday/dev/data-profiler-off/{app}")
    os.system(f"mv unique_* {app}/")
    os.system(f"bash /home/xuanday/dev/Halide/paper_scripts/predict_unique.sh . {app}/updated_1e-3.weights {app}/prediction {app}/unique_prediction_samples.txt")
    os.system(f"mv lower_bound_predictions.txt {app}/")
    os.system(f"mv upper_bound_predictions.txt {app}/")
    os.system(f"mkdir -p {app}/figures/")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/scatter_with_ranges.py --lower {app}/lower_bound_predictions.txt --app {app} --output {app}/figures --upper {app}/upper_bound_predictions.txt")
