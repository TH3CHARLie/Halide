import sys
import os

APPS = ["bgu", "bilateral_grid", "camera_pipe", "hist", "iir_blur", "lens_blur", "local_laplacian", "max_filter", "nl_means", "stencil_chain"]


for app in APPS:
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/parse_unique_samples.py {app}/training_loss.txt /home/xuanday/dev/data-profiler-off/ true")
    os.system(f"head -n 1024 unique_samples.txt > unique_samples_first_1024.txt")
    os.system(f"bash /home/xuanday/dev/Halide/paper_scripts/predict_unique.sh . {app}/updated_1e-3.weights {app}/training_set_prediction unique_samples_first_1024.txt")
    os.system(f"mv lower_bound_predictions.txt {app}/lower_bound_predictions_on_training_set.txt")
    os.system(f"mv upper_bound_predictions.txt {app}/upper_bound_predictions_on_training_set.txt")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/scatter_with_ranges.py --lower {app}/lower_bound_predictions_on_training_set.txt --app {app}_training_first_batch --output {app}/figures --upper {app}/upper_bound_predictions_on_training_set.txt")
    os.system("rm unique_samples.txt unique_prediction_samples.txt unique_training_samples.txt unique_samples_first_1024.txt")
    
