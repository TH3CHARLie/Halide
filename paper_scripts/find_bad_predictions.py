import sys

"""
Read prediction file (in runtime), find top-k false-positive & false-negative
"""

if __name__ == "__main__":
    infile = sys.argv[1]
    K = int(sys.argv[2])
    records = []
    with open(infile, "r") as f:
        for line in f:
            tokens = line.split(",")
            name = tokens[0]
            predicted = float(tokens[1])
            actual = float(tokens[2])
            # we are looking for predicted is faster than actual to spot
            # any problems
            ratio = actual / predicted
            records.append((name, predicted, actual, ratio))
    sorted_records = sorted(records, key=lambda i : i[3], reverse=True)
    print(f"Top{K} predicted much faster than actual:")
    for item in sorted_records[:K]:
        print(f"{item[0]}, {item[1]}, {item[2]}, {item[2] / item[1]}")
    print(f"Top{K} predicted much slower than actual:")
    for item in sorted_records[-K:]:
        print(f"{item[0]}, {item[1]}, {item[2]}, {item[1] / item[2]}")
    