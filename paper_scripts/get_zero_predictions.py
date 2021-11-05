import os
import sys

if __name__ == "__main__":
    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    filenames = []
    with open(input_file, "r") as f:
        for line in f:
            filename = line.split(',')[0]
            filenames.append(filename)
    print(filenames)
    sample_dir = output_dir + "/samples"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
        if not os.path.exists(sample_dir):
            os.mkdir(sample_dir)
    for file in filenames:
        os.system(f"cp {file} {sample_dir}")