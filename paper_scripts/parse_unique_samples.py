import sys
import math

if __name__ == "__main__":
    infile = sys.argv[1]
    prefix = sys.argv[2]
    flag = False
    if len(sys.argv) > 3:
        flag = True
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
        if flag:
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
