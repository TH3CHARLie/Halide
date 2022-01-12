#!/bin/bash

if [[ $# -ne 5 && $# -ne 6 ]]; then
    echo "Usage: $0 max_iterations resume train_only predict_only compare_with_metrics app"
    exit
fi

set -e

MAX_ITERATIONS=${1}
RESUME=${2}
TRAIN_ONLY=${3}
PREDICT_ONLY=${4}
COMPARE_WITH_METRICS=${5}
APP=${6}

if [[ $PREDICT_ONLY == 1 && $TRAIN_ONLY == 1 ]]; then
    echo "At most one of train_only and predict_only can be set to 1."
    exit
fi

if [[ $PREDICT_ONLY == 1 ]]; then
    echo "Predict only mode: ON"
fi


BEST_SCHEDULES_DIR=$(dirname $0)/best

HALIDE_ROOT="/home/xuanda/dev/Halide"

source $(dirname $0)/utils.sh

export CXX="g++"

export HL_MACHINE_PARAMS=20,24000000,160

# export HL_PERMIT_FAILED_UNROLL=1

export AUTOSCHED_BIN="/home/xuanda/dev/Halide/bin"
echo "AUTOSCHED_BIN set to ${AUTOSCHED_BIN}"
export AUTOSCHED_SRC="/home/xuanda/dev/Halide/src/autoschedulers/adams2019"
echo "AUTOSCHED_SRC set to ${AUTOSCHED_SRC}"
export HALIDE_DISTRIB_PATH="/home/xuanda/dev/Halide/distrib"

export SEARCH_SPACE_OPTIONS=01111
echo "SEARCH_SPACE_OPTIONS set to ${SEARCH_SPACE_OPTIONS}"
echo
# export PAPI_EVENTS="PAPI_TOT_CYC"
export PAPI_EVENTS="PAPI_L1_DCM,PAPI_L1_ICM,PAPI_L2_DCM,PAPI_L2_ICM,PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM,PAPI_CA_SNP,PAPI_CA_SHR,PAPI_CA_CLN,PAPI_CA_ITV,PAPI_L3_LDM,PAPI_TLB_DM,PAPI_TLB_IM,PAPI_L1_LDM,PAPI_L1_STM,PAPI_L2_LDM,PAPI_L2_STM,PAPI_PRF_DM,PAPI_MEM_WCY,PAPI_STL_ICY,PAPI_FUL_ICY,PAPI_STL_CCY,PAPI_FUL_CCY,PAPI_BR_UCN,PAPI_BR_CN,PAPI_BR_TKN,PAPI_BR_NTK,PAPI_BR_MSP,PAPI_BR_PRC,PAPI_TOT_INS,PAPI_LD_INS,PAPI_SR_INS,PAPI_BR_INS,PAPI_RES_STL,PAPI_TOT_CYC,PAPI_LST_INS,PAPI_L2_DCA,PAPI_L3_DCA,PAPI_L2_DCR,PAPI_L3_DCR,PAPI_L2_DCW,PAPI_L3_DCW,PAPI_L2_ICH,PAPI_L2_ICA,PAPI_L3_ICA,PAPI_L2_ICR,PAPI_L3_ICR,PAPI_L2_TCA,PAPI_L3_TCA,PAPI_L2_TCR,PAPI_L3_TCR,PAPI_L2_TCW,PAPI_L3_TCW,PAPI_SP_OPS,PAPI_DP_OPS,PAPI_VEC_SP,PAPI_VEC_DP,PAPI_REF_CYC"
export PAPI_OUTPUT_DIRECTORY="/home/xuanda/dev/Halide/results-perf-counter"


if [ ! -v HL_TARGET ]; then
    get_host_target ${HALIDE_ROOT} HL_TARGET
fi

export HL_TARGET=${HL_TARGET}

echo "HL_TARGET set to ${HL_TARGET}"

DEFAULT_SAMPLES_DIR_NAME="${SAMPLES_DIR:-autotuned_samples}"

CURRENT_DATE_TIME="`date +%Y-%m-%d-%H-%M-%S`";

if [ -z $APP ]; then
    APPS="bilateral_grid"
    # APPS="bgu bilateral_grid camera_pipe conv_layer hist iir_blur interpolate lens_blur local_laplacian max_filter nl_means stencil_chain unsharp"
    # APPS="bgu bilateral_grid blur local_laplacian nl_means lens_blur camera_pipe hist max_filter interpolate conv_layer"
else
    APPS=${APP}
fi

NUM_APPS=0
for app in $APPS; do
    NUM_APPS=$((NUM_APPS + 1))
done

echo "Autotuning on $APPS for $MAX_ITERATIONS iteration(s)"

for app in $APPS; do
    SECONDS=0
    APP_DIR="${HALIDE_ROOT}/apps/${app}"

    unset -v LATEST_SAMPLES_DIR
    for f in "$APP_DIR/${DEFAULT_SAMPLES_DIR_NAME}"*; do
        if [[ ! -d $f ]]; then
           continue
       fi

        if [[ -z ${LATEST_SAMPLES_DIR+x} || $f -nt $LATEST_SAMPLES_DIR ]]; then
            LATEST_SAMPLES_DIR=$f
        fi
    done

#     if [[ ${RESUME} -eq 1 && -z ${LATEST_SAMPLES_DIR+x} ]]; then
#         SAMPLES_DIR=${LATEST_SAMPLES_DIR}
#         echo "Resuming from existing run: ${SAMPLES_DIR}"
#     else
        while [[ 1 ]]; do
            SAMPLES_DIR_NAME=${DEFAULT_SAMPLES_DIR_NAME}-${CURRENT_DATE_TIME}
            SAMPLES_DIR="${APP_DIR}/${SAMPLES_DIR_NAME}"

            if [[ ! -d ${SAMPLES_DIR} ]]; then
                break
            fi

            sleep 1
            CURRENT_DATE_TIME="`date +%Y-%m-%d-%H-%M-%S`";
        done
        SAMPLES_DIR="${APP_DIR}/${SAMPLES_DIR_NAME}"
        echo "Starting new run in: ${SAMPLES_DIR}"
#     fi

    OUTPUT_FILE="${SAMPLES_DIR}/autotune_out.txt"
    PREDICTIONS_FILE="${SAMPLES_DIR}/predictions"
    PREDICTIONS_WITH_FILENAMES_FILE="${SAMPLES_DIR}/predictions_with_filenames"
    OUTLIERS_FILE="${SAMPLES_DIR}/outliers"
    BEST_TIMES_FILE="${SAMPLES_DIR}/best_times"

    mkdir -p ${SAMPLES_DIR}
    touch ${OUTPUT_FILE}

    if [[ $PREDICT_ONLY != 1 ]]; then
        NUM_BATCHES=${MAX_ITERATIONS} TRAIN_ONLY=${TRAIN_ONLY} SAMPLES_DIR=${SAMPLES_DIR} make -C ${APP_DIR} autotune | tee -a ${OUTPUT_FILE}
    fi

    WEIGHTS_FILE="${SAMPLES_DIR}/updated.weights"
    find ${SAMPLES_DIR} -name "*.sample" | \
        ${AUTOSCHED_BIN}/retrain_cost_model \
            --epochs="1" \
            --rates="0.001" \
            --num_cores=20 \
            --initial_weights=${WEIGHTS_FILE} \
            --weights_out=${WEIGHTS_FILE} \
            --predict_only="1" \
            --best_benchmark=${SAMPLES_DIR}/best.${PIPELINE}.benchmark.txt \
            --best_schedule=${SAMPLES_DIR}/best.${PIPELINE}.schedule.h \
            --predictions_file=${PREDICTIONS_WITH_FILENAMES_FILE} \
            --verbose="0" \
            --partition_schedules="0" \
            --limit="0"
    awk -F", " '{printf("%f, %f\n", $2, $3);}' ${PREDICTIONS_WITH_FILENAMES_FILE} > ${PREDICTIONS_FILE}

done
