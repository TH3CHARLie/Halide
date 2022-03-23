import sys

def main():
    argc = len(sys.argv)
    num_files = argc - 1
    predicted_total = {}
    actual_total = {}
    for i in range(1, argc):
        with open(sys.argv[i], "r") as f:
            for line in f:
                tokens = line.split()
                filename = tokens[0][:-1]
                predicted = float(tokens[1][:-1])
                actual = float(tokens[2])
                if i == 1:
                    predicted_total[filename] = predicted
                    actual_total[filename] = actual
                else:
                    if filename in predicted_total:
                        predicted_total[filename] += predicted
                        actual_total[filename] += actual
    with open("combined_prediction", "w") as f:
        for k in predicted_total.keys():
            f.write(f"{predicted_total[k]}, {actual_total[k]}\n")

if __name__ == "__main__":
    main()