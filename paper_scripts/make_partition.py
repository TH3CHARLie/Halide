import sys

if __name__ == "__main__":
    infile = sys.argv[1] 
    filenames = []
    with open(infile, "r") as f:
        for line in f:
            filenames.append(line.strip())
    splits = len(filenames) // 1024 + 1
    print(f'reading {len(filenames)} samples, split into {splits} parts')
    for i in range(splits):
        print(f"writing partition_part{i}.txt")
        with open(f"partition_part{i}.txt", "w") as f:
            for j in range(i * 1024, min((i + 1) * 1024, len(filenames)), 1):
                f.write(filenames[j] + '\n')

