import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True, action="store", help="path to folders where samples are stored")
    parser.add_argument('--target', required=True, action="store", help="new path where non-unique samples will be placed")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    source = args.source
    target = args.target
    mkdir_command = f"mkdir -p {target}"
    os.system(mkdir_command)
    os.system(f'find {source}/ -name "*.sample" > tmp_samples.txt')
    os.system(f"bash /home/xuanda/dev/Halide/paper_scripts/train_unique.sh . wrong.weights tmp.weights tmp_prediction 10 tmp_samples.txt 0.001 | tee tmp_log.txt")
    os.system(f"python3 /home/xuanda/dev/Halide/paper_scripts/parse_unique_samples.py --input tmp_log.txt --prefix {source}")
    with open(f"unique_samples.txt", "r") as f:
        for line in f:
            filename = line.strip()
            copy_command = f"cp {filename} {target}/"
            print(copy_command)
            os.system(copy_command)
    os.system(f"rm tmp_samples.txt tmp.weights tmp_prediction* tmp_log.txt unique_samples.txt unique_training_samples.txt unique_prediction_samples.txt")


if __name__ == "__main__":
    main()
