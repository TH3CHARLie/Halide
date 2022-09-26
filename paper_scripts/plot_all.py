import sys
import os
from datetime import datetime
import glob
from scatter import plot

APPS = "bilateral_grid camera_pipe conv_layer hist iir_blur interpolate lens_blur local_laplacian max_filter nl_means".split()
# APPS = "bgu bilateral_grid camera_pipe conv_layer hist iir_blur interpolate lens_blur local_laplacian max_filter nl_means stencil_chain unsharp".split()
ROOT_PATH = "/home/xuanda/dev/Halide/apps/"

OUTPUT = "/home/xuanda/dev/Halide/paper_scripts/tmp_plots"


if __name__ == "__main__":
    for app in APPS:
        app_dir = ROOT_PATH + app
        autotune_dirs = list(filter(lambda x: os.path.isdir(os.path.join(app_dir, x)) 
                                and x.find("autotuned_samples") != -1 , os.listdir(app_dir)))
        latest_autotune_dir = sorted(autotune_dirs, key=lambda s: datetime.strptime(s[s.find('-') + 1 : ], "%Y-%m-%d-%H-%M-%S"))[-1]
        predictions = os.path.join(app_dir, latest_autotune_dir, "predictions_tmp")
        print(f"app: {app} predictions: {predictions}")
        plot(predictions, app, OUTPUT)
