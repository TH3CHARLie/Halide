import sys
import os
import subprocess


"""
diff using imagestack tool
"""

samples = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])

bad_pipelines = []
const_pipelines = []
cnt_bad_pipelines = 0
cnt_total_pipelines = end - start
for i in range(start, end):
    print(f"Running diff for batch_{i}")
    os.system("mkdir tmp_playground")
    folder = f"{samples}/batch_{i}/"
    compare_target_idx = -1
    for j in range(0, 24):
        bench_file = f"{folder}/{j}/bench"
        if os.path.exists(bench_file):
            if compare_target_idx == -1:
                compare_target_idx = j
        else:
            continue
        os.system(f'{bench_file} --estimate_all "input=random:42:[2000,2000,3]" "--output_extents=[2000,2000,3]" "output=tmp_playground/batch_{i}_{j}.tmp"')
    print("compare_target_idx", compare_target_idx)
    is_good_pipeline = True
    for j in range(0, 24):
        if not os.path.exists(f"tmp_playground/batch_{i}_{j}.tmp"):
            continue
        if j == compare_target_idx:
            continue
        os.system(f"ImageStack -load tmp_playground/batch_{i}_{compare_target_idx}.tmp -load tmp_playground/batch_{i}_{j}.tmp -subtract -statistics | tee tmp_playground/result.txt")
        with open("tmp_playground/result.txt", "r") as f:
            for line in f:
                if line.find("Means:") != -1:
                    tokens = line.split()
                    mean = float(tokens[1])
                    print("mean: ", mean)
                    if abs(mean) > 1e-4:
                        is_good_pipeline = False
                        break
    if not is_good_pipeline:
        bad_pipelines.append(i)
        cnt_bad_pipelines += 1
    else:
        os.system(f"ImageStack -load tmp_playground/batch_{i}_{compare_target_idx}.tmp  -statistics | tee tmp_playground/result.txt")
        with open("tmp_playground/result.txt", "r") as f:
            for line in f:
                if line.find("Variance:") != -1:
                    tokens = line.split()
                    variance = float(tokens[1])
                    print("variance: ", variance)
                    if variance == 0.0:
                        const_pipelines.append(i)

    os.system("rm -r tmp_playground")
    print(f"total pipelines: {cnt_total_pipelines} disagree pipelines: {cnt_bad_pipelines}")
    print(bad_pipelines)
    print("Here are const pipelines:")
    print(const_pipelines)
