import sys
import os
import argparse

# APPS = ["bgu", "bilateral_grid", "camera_pipe", "hist", "iir_blur", "lens_blur", "local_laplacian", "max_filter", "nl_means", "stencil_chain"]
APPS = ["camera_pipe", "hist", "iir_blur", "max_filter", "stencil_chain"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, action="store", help="path to sensei fs where samples are stored")
    parser.add_argument('--target', required=True, action="store", help="path to local fs where samples are going to be stored")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source = args.source
    target = args.target
    for app in APPS:
        mkdir_command = f"mkdir -p {target}/{app}/"
        os.system(mkdir_command)
        os.system(f'find {source}/{app} -name "*.sample" > sensei_fs_{app}_samples.txt')
        with open(f"sensei_fs_{app}_samples.txt", "r") as f:
            for line in f:
                filename = line.strip()
                copy_command = f"cp {filename} {target}/{app}"
                print(copy_command)
                os.system(copy_command)
        os.system(f"rm sensei_fs_{app}_samples.txt")


if __name__ == "__main__":
    main()
