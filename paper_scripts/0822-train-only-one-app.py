import sys
import os

APPS = ["bgu", "bilateral_grid", "camera_pipe", "harris", "hist", "iir_blur", "interpolate", "max_filter", "nl_means", "stencil_chain", "unsharp"]

for app in APPS:
    os.system(f"mkdir -p {app}")
    os.system(f'find /home/xuanday/dev/data-final/ -name "{app}_*.sample" > {app}/{app}_samples.txt')
    os.system(f'head -n 1024 {app}/{app}_samples.txt > {app}/{app}_train_samples.txt')
    os.system(f'tail -n 40 {app}/{app}_samples.txt > {app}/{app}_test_samples.txt')
    os.system(f"bash /home/xuanday/dev/Halide/paper_scripts/train_unique.sh . wrong.weights {app}/updated.weights {app}/training_prediction 10000 {app}/{app}_train_samples.txt 0.005 | tee {app}/training_loss.txt")
    os.system(f"mkdir -p {app}/figures")
    os.system(f"bash /home/xuanday/dev/Halide/paper_scripts/predict_unique.sh . {app}/updated.weights prediction {app}/{app}_train_samples.txt")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/scatter_with_ranges.py --lower lower_bound_predictions.txt --app {app}_train --output {app}/figures --upper upper_bound_predictions.txt")
    os.system(f"bash /home/xuanday/dev/Halide/paper_scripts/predict_unique.sh . {app}/updated.weights prediction {app}/{app}_test_samples.txt")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/scatter_with_ranges.py --lower lower_bound_predictions.txt --app {app}_test --output {app}/figures --upper upper_bound_predictions.txt")
    os.system("rm prediction prediction_with_filename lower_bound_predictions.txt upper_bound_predictions.txt updated_back.weights")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/plot_training_log.py {app}/training_loss.txt {app} Loss")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/plot_training_log.py {app}/training_loss.txt {app} Lower_Bound_Term_1_Loss")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/plot_training_log.py {app}/training_loss.txt {app} Lower_Bound_Term_2_Loss")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/plot_training_log.py {app}/training_loss.txt {app} Upper_Bound_Term_1_Loss")
    os.system(f"python3 /home/xuanday/dev/Halide/paper_scripts/plot_training_log.py {app}/training_loss.txt {app} Upper_Bound_Term_2_Loss")
    os.system(f"mv *.png {app}/figures")