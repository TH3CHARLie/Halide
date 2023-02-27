#!/bin/bash
FIRST=$1
LAST=$2
FOLDER=$3
CORES=24
mkdir -p ${FOLDER}
for ((seed=${FIRST};seed<${LAST};seed++)); do
    rm -rf samples
    bash ../../src/autoschedulers/adams2019/autotune_loop.sh bin/random_pipeline.generator random_pipeline x86-64-avx2-linux-strict_float ../../src/autoschedulers/adams2019/baseline.weights ../../distrib/bin ../../distrib samples $CORES $seed seed=${seed} | tee batch_stages_${seed}.log 
    cd samples
    mv batch_1_0 batch_${seed}
    mv batch_${seed} ../${FOLDER}/
    mv ../batch_stages_${seed}.log ../${FOLDER}/batch_${seed}/
    cd ..
done