import sys
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, action="store", help="path to training log which contains unique sample output")
    parser.add_argument('--prefix', required=True, action="store", help="path to local fs where samples are going to be stored")
    parser.add_argument('--parse-app-name', dest='parse_app_name', action="store_true", help="whether or not parse the app name")
    parser.set_defaults(parse_app_name=False)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    infile = args.input
    prefix = args.prefix
    parse_app_name = args.parse_app_name
    paths = []
    with open(infile, "r") as f:
        for line in f:
            tokens = line.split()
            if len(tokens) > 1 and tokens[0] == "Unique":
                paths.append(tokens[2])
    new_paths = []
    for path in paths:
        batch_id = int(path[path.find("batch") + 6 : path.find("batch") + 10])
        sample_id = int(path[path.find("sample") + 7 : path.find("sample") + 11])
        if parse_app_name:
            app_name = path[: path.find("batch") - 1]
            new_path = prefix + f"/{app_name}/{path}"
        else:
            new_path = prefix + f"/{path}"
        new_paths.append(new_path)
    print(f"num of unique samples: {len(new_paths)}")
    with open("unique_training_samples.txt", "w") as o:
        for path in new_paths[40:]:
            o.write(path + '\n')
    with open("unique_prediction_samples.txt", "w") as o:
        for path in new_paths[:40]:
            o.write(path + '\n')
    with open("unique_samples.txt", "w") as o:
        for path in new_paths:
            o.write(path + '\n')
