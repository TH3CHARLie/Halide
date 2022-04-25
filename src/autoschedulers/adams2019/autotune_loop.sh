#!/bin/bash

# Build the generator to autotune. This script will be autotuning the
# autoscheduler's cost model training pipeline, which is large enough
# to be interesting.
if [ $# -lt 6 -o $# -gt 8 ]; then
  echo "Usage: $0 /path/to/some.generator generatorname halide_target weights_file autoschedule_bin_dir halide_distrib_path samples_out_path [generator_args_sets]"
  exit
fi

set -eu

#trap "exit" INT TERM
#trap "kill 0" EXIT

GENERATOR=${1}
PIPELINE=${2}
HL_TARGET=${3}
START_WEIGHTS_FILE=${4}
AUTOSCHED_BIN=${5}
HALIDE_DISTRIB_PATH=${6}
SAMPLES=${7}
echo "sample dirs"
echo ${SAMPLES}

# Read the generator-arg sets into an array. Each set is delimited
# by space; multiple values within each set are are delimited with ;
# e.g. "set1arg1=1;set1arg2=foo set2=bar set3arg1=3.14;set4arg2=42"
if [ $# -ge 8 ]; then
    IFS=' ' read -r -a GENERATOR_ARGS_SETS_ARRAY <<< "${8}"
else
    declare -a GENERATOR_ARGS_SETS_ARRAY=
fi

# Ensure the length is at least 1
if [ ${#GENERATOR_ARGS_SETS_ARRAY[@]} -eq 0 ]; then
    GENERATOR_ARGS_SETS_ARRAY=( '' )
fi

COMPILATION_TIMEOUT=600s
BENCHMARKING_TIMEOUT=60s

if [ -z ${HL_TARGET} ]; then
# Use the host target -- but remove features that we don't want to train
# for by default, at least not yet (most notably, AVX512).
HL_TARGET=`${AUTOSCHED_BIN}/get_host_target avx512 avx512_knl avx512_skylake avx512_cannonlake`
fi
echo Training target is: ${HL_TARGET}

if [ -z ${GENERATOR} ]; then
GENERATOR=./bin/demo.generator
fi

if [ -z ${PIPELINE} ]; then
PIPELINE=demo
fi

mkdir -p ${SAMPLES}

WEIGHTS=${SAMPLES}/updated.weights
if [[ -f ${WEIGHTS} ]]; then
    echo Using existing weights "${WEIGHTS}"
else
    # Only copy over the weights if we don't have any already,
    # so that restarted jobs can continue from where they left off
    cp ${START_WEIGHTS_FILE} ${WEIGHTS}
    echo Copying starting weights from ${START_WEIGHTS_FILE} to ${WEIGHTS}
fi

# We could add this unconditionally, but it's easier to wade thru
# results if we only add if needed
for F in disable_llvm_loop_opt; do
    if [[ ! ${HL_TARGET} =~ .*${F}.* ]]; then
        HL_TARGET="${HL_TARGET}-${F}"
    fi
done

# A batch of this many samples is built in parallel, and then
# benchmarked serially.
BATCH_SIZE=40

TIMEOUT_CMD="timeout"
if [ $(uname -s) = "Darwin" ] && ! which $TIMEOUT_CMD 2>&1 >/dev/null; then
    # OSX doesn't have timeout; gtimeout is equivalent and available via Homebrew
    TIMEOUT_CMD="gtimeout"
    if ! which $TIMEOUT_CMD 2>&1 >/dev/null; then
        echo "Can't find the command 'gtimeout'. Run 'brew install coreutils' to install it."
        exit 1
    fi
fi

# Build a single featurization of the pipeline with a random schedule
make_featurization() {
    # echo ${SAMPLES}
    D=${1}
    SEED=${2}
    FNAME=${3}
    EXTRA_GENERATOR_ARGS=${4}
    mkdir -p ${D}
    rm -f "${D}/${FNAME}.featurization"
    rm -f "${D}/${FNAME}.sample"
    if [[ $D == */0 ]]; then
        # Sample 0 in each batch is best effort beam search, with no randomness
        dropout=100
        beam=32
    else
        # The other samples are random probes biased by the cost model
        dropout=1  # 1% chance of operating entirely greedily
        beam=1
    fi

    # Note: generate sample program
    HL_SEED=${SEED} \
        HL_WEIGHTS_DIR=${WEIGHTS} \
        HL_RANDOM_DROPOUT=${dropout} \
        HL_BEAM_SIZE=${beam} \
        HL_MACHINE_PARAMS=20,24000000,40 \
        SAMPLES=${SAMPLES} \
        ${TIMEOUT_CMD} -k ${COMPILATION_TIMEOUT} ${COMPILATION_TIMEOUT} \
        ${GENERATOR} \
        -g ${PIPELINE} \
        -f ${FNAME} \
        -o ${D} \
        -e stmt,assembly,static_library,c_header,registration,schedule,featurization \
        target=${HL_TARGET} \
        auto_schedule=true \
        ${EXTRA_GENERATOR_ARGS} \
        -s Adams2019 \
        # -p ${AUTOSCHED_BIN}/libautoschedule_adams2019.so \
        2> ${D}/compile_log.txt || echo "Compilation failed or timed out for ${D}"


    # We don't need image I/O for this purpose,
    # so leave out libpng and libjpeg

    # Note: compile sample program
    c++ \
        -std=c++17 \
        -I ${HALIDE_DISTRIB_PATH}/include \
        -I/${PAPI_DIR}/include \
        -L/${PAPI_DIR}/lib \
        ${HALIDE_DISTRIB_PATH}/tools/RunGenMain.cpp \
        ${D}/*.registration.cpp \
        ${D}/*.a \
        -o ${D}/bench \
        -DHALIDE_NO_PNG -DHALIDE_NO_JPEG \
        -ldl -lpthread \
        -lpapi
}


# Benchmark one of the random samples
benchmark_sample() {
    sleep 1 # Give CPU clocks a chance to spin back up if we're thermally throttling
    D=${1}
    export PAPI_TARGET_EVENTS="PAPI_L1_DCM,PAPI_L1_ICM,PAPI_L2_DCM,PAPI_L2_ICM,PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM,PAPI_CA_SNP,PAPI_CA_SHR,PAPI_CA_CLN,PAPI_CA_ITV,PAPI_L3_LDM,PAPI_TLB_DM,PAPI_TLB_IM,PAPI_L1_LDM,PAPI_L1_STM,PAPI_L2_LDM,PAPI_L2_STM,PAPI_PRF_DM,PAPI_MEM_WCY,PAPI_STL_ICY,PAPI_FUL_ICY,PAPI_STL_CCY,PAPI_FUL_CCY,PAPI_BR_UCN,PAPI_BR_CN,PAPI_BR_TKN,PAPI_BR_NTK,PAPI_BR_MSP,PAPI_BR_PRC,PAPI_TOT_INS,PAPI_LD_INS,PAPI_SR_INS,PAPI_BR_INS,PAPI_RES_STL,PAPI_TOT_CYC,PAPI_LST_INS,PAPI_L2_DCA,PAPI_L3_DCA,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L2_DCW,PAPI_L3_DCW,PAPI_L2_ICH,PAPI_L2_ICA,PAPI_L3_ICA,PAPI_L2_ICR,PAPI_L3_ICR,PAPI_L2_TCA,PAPI_L3_TCA,PAPI_L2_TCR,PAPI_L3_TCR,PAPI_L2_TCW,PAPI_L3_TCW,PAPI_SP_OPS,PAPI_DP_OPS,PAPI_VEC_SP,PAPI_VEC_DP,PAPI_REF_CYC"
    export PAPI_OUTPUT_DIRECTORY=${D}
    while [[ ! -z "$PAPI_TARGET_EVENTS" ]]; do
        result=($(python3 ${HALIDE_DISTRIB_PATH}/../paper_scripts/set_events.py))
        export PAPI_EVENTS=${result[0]}
        if [[ ${#result[@]} -eq 1 ]]
        then
            export PAPI_TARGET_EVENTS=""
        else
            export PAPI_TARGET_EVENTS=${result[1]}
        fi

        HL_NUM_THREADS=20 \
            ${TIMEOUT_CMD} -k ${BENCHMARKING_TIMEOUT} ${BENCHMARKING_TIMEOUT} \
            ${D}/bench \
            --estimate_all \
            --benchmarks=all \
                | tee ${D}/bench.txt || echo "Benchmarking failed or timed out for ${D}"
    done
    # Add the runtime, pipeline id, and schedule id to the feature file
    R=$(cat ${D}/bench.txt | head -n1 | cut -d' ' -f8)
    P=$3
    S=$2
    FNAME=$4
    ${AUTOSCHED_BIN}/featurization_to_sample ${D}/${FNAME}.featurization $R $P $S ${D}/${FNAME}.sample || echo "featurization_to_sample failed for ${D} (probably because benchmarking failed)"
}

# Don't clobber existing samples
FIRST=$(ls -d ${SAMPLES}/batch_* 2>/dev/null | sed -e "s|.*/batch_||;s|_.*||" | sort -n | tail -n1)

if [ $(uname -s) = "Darwin" ]; then
    LOCAL_CORES=`sysctl -n hw.ncpu`
else
    LOCAL_CORES=`nproc`
fi
echo Local number of cores detected as ${LOCAL_CORES}

# NUM_BATCHES="1"
NUM_BATCHES=${NUM_BATCHES:-1}
echo "Num batches: ${NUM_BATCHES}"

for ((BATCH_ID=$((FIRST+1));BATCH_ID<$((FIRST+1+NUM_BATCHES));BATCH_ID++)); do
    SECONDS=0

    for ((EXTRA_ARGS_IDX=0;EXTRA_ARGS_IDX<${#GENERATOR_ARGS_SETS_ARRAY[@]};EXTRA_ARGS_IDX++)); do

        # Compile a batch of samples using the generator in parallel
        DIR=${SAMPLES}/batch_${BATCH_ID}_${EXTRA_ARGS_IDX}

        # Copy the weights being used into the batch folder so that we can repro failures
        mkdir -p ${DIR}/
        cp ${WEIGHTS} ${DIR}/used.weights

        EXTRA_GENERATOR_ARGS=${GENERATOR_ARGS_SETS_ARRAY[EXTRA_ARGS_IDX]/;/ }
        if [ ! -z "${EXTRA_GENERATOR_ARGS}" ]; then
            echo "Adding extra generator args (${EXTRA_GENERATOR_ARGS}) for batch_${BATCH_ID}"
        fi

        echo ${EXTRA_GENERATOR_ARGS} > ${DIR}/extra_generator_args.txt

        # Do parallel compilation in batches, so that machines with fewer than BATCH_SIZE cores
        # don't get swamped and timeout unnecessarily
        echo -n Compiling ${BATCH_SIZE} samples
        for ((SAMPLE_ID=0;SAMPLE_ID<${BATCH_SIZE};SAMPLE_ID++)); do
            while [[ 1 ]]; do
                RUNNING=$(jobs -r | wc -l)
                if [[ RUNNING -ge LOCAL_CORES ]]; then
                    sleep 1
                else
                    break
                fi
            done

            S=$(printf "%04d%04d" $BATCH_ID $SAMPLE_ID)

            # Xuanda: Uncomment the following line to reproduce outlier samples
            # bottom left two outliers
            # S=00290022
            # S=04350034
            # top right two outliers
            # S=03160017
            # S=00290022
            FNAME=$(printf "%s_batch_%04d_sample_%04d" ${PIPELINE} $BATCH_ID $SAMPLE_ID)
            make_featurization "${DIR}/${SAMPLE_ID}" $S $FNAME "$EXTRA_GENERATOR_ARGS" &
            echo -n .
        done
        wait
        echo  done.

        # benchmark them serially using rungen
        for ((SAMPLE_ID=0;SAMPLE_ID<${BATCH_SIZE};SAMPLE_ID++)); do
            S=$(printf "%04d%04d" $BATCH_ID $SAMPLE_ID)
            FNAME=$(printf "%s_batch_%04d_sample_%04d" ${PIPELINE} $BATCH_ID $SAMPLE_ID)
            benchmark_sample "${DIR}/${SAMPLE_ID}" $S $EXTRA_ARGS_IDX $FNAME
        done

        # retrain model weights on all samples seen so far
        echo Retraining model...

        find ${SAMPLES} -name "*.sample" | \
            ${AUTOSCHED_BIN}/retrain_cost_model \
                --epochs=${BATCH_SIZE} \
                --rates="0.0001" \
                --num_cores=20 \
                --initial_weights=${WEIGHTS} \
                --weights_out=${WEIGHTS} \
                --predict_only="0" \
                --best_benchmark=${SAMPLES}/best.${PIPELINE}.benchmark.txt \
                --best_schedule=${SAMPLES}/best.${PIPELINE}.schedule.h \
                --predictions_file=${SAMPLES}/tmp_predictions \
                --verbose="0" \
                --partition_schedules="0" \
                --limit="0"
    done

    echo Batch ${BATCH_ID} took ${SECONDS} seconds to compile, benchmark, and retrain
done
