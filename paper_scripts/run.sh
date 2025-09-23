DATA_DIR=$1
find $DATA_DIR -name "*.hlpipe" > "pipelines.txt"
xargs md5sum < "pipelines.txt" > "pipelines_md5.txt"
./test/fuzz_driver/fuzz_driver_load_unique "pipelines_md5.txt"
NUM=$(ls -lah | grep "unique_pipelines" | wc -l)
echo "Number of unique pipelines: $NUM"
# for i in {0..0}; do
#     echo "Running unique_pipelines_md5_${i}.txt"
#     ./test/fuzz_driver/fuzz_driver_driver "unique_pipelines_md5_${i}.txt" 2>"stderr_${i}.txt" 1>"stdout_${i}.txt"
# done
