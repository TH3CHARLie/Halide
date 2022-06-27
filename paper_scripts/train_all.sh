#!/bin/bash

if [ $# -ne 5 ]; then
  echo "Usage: $0 samples_dir init_weights_file output_weights_file predictions_file num_epochs"
  exit
fi

export AUTOSCHED_BIN="/home/xuanday/dev/Halide/bin"
SAMPLES_DIR=${1}
INIT_WEIGHTS_FILE=${2}
OUTPUT_WEIGHTS_FILE=${3}
PREDICTIONS_FILE=${4}
NUM_EPOCHS=${5}
PREDICTIONS_WITH_FILENAMES_FILE="${PREDICTIONS_FILE}_with_filename"
echo
echo "Samples directory: ${SAMPLES_DIR}"
echo "Init Weights file: ${INIT_WEIGHTS_FILE}"
echo "Output weights file: ${OUTPUT_WEIGHTS_FILE}"
echo "Saving predictions to: ${PREDICTIONS_FILE}"
echo "Number of epochs: ${NUM_EPOCHS}"

NUM_CORES=20

WEIGHTS_OUTFILE="${SAMPLES_DIR}/updated_back.weights"
find ${SAMPLES_DIR} -name "*.sample" | \
    ${AUTOSCHED_BIN}/retrain_cost_model \
        --epochs=${NUM_EPOCHS} \
        --rates="0.0001" \
        --num_cores=20 \
        --initial_weights=${INIT_WEIGHTS_FILE} \
        --weights_out=${OUTPUT_WEIGHTS_FILE} \
        --predict_only="0" \
        --best_benchmark=${SAMPLES_DIR}/best.${PIPELINE}.benchmark.txt \
        --best_schedule=${SAMPLES_DIR}/best.${PIPELINE}.schedule.h \
        --predictions_file=${PREDICTIONS_WITH_FILENAMES_FILE} \
        --verbose="0" \
        --partition_schedules="0" \
        --limit="0"

awk -F", " '{printf("%f, %f\n", $2, $3);}' ${PREDICTIONS_WITH_FILENAMES_FILE} > ${PREDICTIONS_FILE}
