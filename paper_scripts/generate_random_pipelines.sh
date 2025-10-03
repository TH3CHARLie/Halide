#!/bin/bash

for ((seed=0;seed<16;seed++)); do
    /home/xuanday/dev/Halide/apps/random_pipeline/build/random_pipeline.generator -g random_pipeline -f random_pipeline_${seed} -o /home/xuanday/dev/Halide/random_pipeline_output -e hlpipe,stmt target=x86-64-avx2-linux seed=${seed}
done