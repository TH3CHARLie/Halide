import sys

"""
Remove certain pipelines from samples.txt, based on their index
"""

TO_BE_EXCLUDED_IDS = [10002, 10005, 10008, 10013, 10018, 10040, 10047, 10048, 10051, 10052, 10064, 10071, 10077, 10078, 10084, 10086, 10088, 10091, 10098, 10100, 10101, 10105, 10107, 10108, 10112, 10117, 10135, 10137, 10141, 10143, 10145, 10147, 10148, 10149, 10150, 10154, 10161, 10164, 10166, 10189, 10192, 10193, 10195, 10196, 10199, 10209, 10210, 10213, 10214, 10215, 10216, 10229, 10236, 10238, 10239]

infile = sys.argv[1]
outfile = sys.argv[2]
samples = []
with open(infile, "r") as f:
    for line in f:
        exclude = False
        for pid in TO_BE_EXCLUDED_IDS:
            if line.find(str(pid)) != -1:
                exclude = True
                break
        if not exclude:
            samples.append(line.strip())

with open(outfile, "w") as f:
    for s in samples:
        f.write(s + "\n")
