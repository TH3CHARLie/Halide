import sys
import os
import subprocess

def main():
    executable = sys.argv[1]
    file_dir = sys.argv[2]
    pipelines = []
    for file in os.listdir(file_dir):
        if file.endswith(".hlpipe"):
            pipelines.append(os.path.join(file_dir, file))
            if (file == "blur_0.hlpipe"):
                print("Found blur_0.hlpipe")
    print("Found %d pipelines" % len(pipelines))
    bad_pipelines = []
    for f in sorted(pipelines):
        # if the errorcode is non-zero, record the pipeline name
        print("Running %s" % f)
        result = subprocess.run("%s %s" % (executable, f), shell=True)
        if result.returncode != 0:
            bad_pipelines.append(f)
    print("bad pipelines are:")
    print(bad_pipelines)

main()
