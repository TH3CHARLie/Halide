import sys
import os
import subprocess

fmt_str = "/home/xuanda/dev/Serializer/Halide/build/fuzz_res/blur_{}.hlpipe"

hash_set = set()

def main():
    executable = sys.argv[1]
    file_cnt = int(sys.argv[2])
    bad_pipeline_ids = []
    for i in range(file_cnt):
        filename = fmt_str.format(i)
        if os.path.exists(filename):
            md5result = subprocess.run("md5sum %s" % (filename), shell=True, stdout=subprocess.PIPE)
            md5hash = int(md5result.stdout.decode('utf-8').split()[0], 16)
            if md5hash in hash_set:
                print(f"skipping pipeline id {i} current bad pipeline count: {len(bad_pipeline_ids)} unique pipelines count: {len(hash_set)}")
                continue
            else:
                hash_set.add(md5hash)
                print(f"running pipeline id {i} current bad pipeline count: {len(bad_pipeline_ids)} unique pipelines count: {len(hash_set)}")
                result = subprocess.run("%s %s" % (executable, filename), shell=True)
                if result.returncode != 0:
                    print(f"detected bad pipeline id {i}", file=sys.stderr)
                    bad_pipeline_ids.append(i)
    print("bad pipelines are:")
    print(bad_pipeline_ids)
    print(f"unique pipelines count: {len(hash_set)}")

main()
