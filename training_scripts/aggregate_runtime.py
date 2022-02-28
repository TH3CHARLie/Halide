from select import select
import sys

filenames = sys.stdin.read().split()
output_filename = sys.argv[1]
runtimes = []

is_first_file = True
selected_filenames = []
for filename in filenames:
    if filename.find("benchmark_runtimes") != -1:
        selected_filenames.append(filename)
for file in selected_filenames:
    with open(file, "r") as f:
        cnt = 0
        for line in f:
            line = line.strip()
            if line == "":
                continue
            tokens = line.split()
            name = tokens[0]
            runtime = float(tokens[1])
            if is_first_file:
                runtimes.append((name, runtime))
            else:
                runtimes[cnt] = (runtimes[cnt][0], runtimes[cnt][1] + runtime)
            cnt += 1
    is_first_file = False


length = len(selected_filenames)

with open(output_filename, "w") as f:
    for runtime in runtimes:
        f.write(f'{runtime[0]} {runtime[1] / length}\n')
