import numpy as np
import sys


def main():
    infile = sys.argv[1]
    cnt = 0
    false_positive_cnt = 0
    false_negative_cnt = 0
    with open(infile, "r") as f:
        for line in f:
            tokens = line.split(",")
            predicted = float(tokens[0])
            actual = float(tokens[1])
            if predicted - actual > (0.1 * actual):
                false_negative_cnt += 1
            if actual - predicted > (0.1 * actual):
                false_positive_cnt += 1
            cnt += 1
    print(f"false positive rate: {false_positive_cnt / cnt}")
    print(f"false negative rate: {false_negative_cnt / cnt}")




if __name__ == "__main__":
    main()