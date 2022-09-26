#!/bin/bash

if [ $# -ne 4 ]; then
  echo "Usage: $0 samples_dir weights_file predictions_file sample_file"
  exit
fi

export AUTOSCHED_BIN="/home/xuanda/dev/Halide/bin"
SAMPLES_DIR=${1}
WEIGHTS_FILE=${2}
PREDICTIONS_FILE=${3}
SAMPLE_FILE=${4}
PREDICTIONS_WITH_FILENAMES_FILE="${PREDICTIONS_FILE}_with_filename"
echo
echo "Samples directory: ${SAMPLES_DIR}"
echo "Weights file: ${WEIGHTS_FILE}"
echo "Saving predictions to: ${PREDICTIONS_FILE}"

NUM_CORES=10

WEIGHTS_OUTFILE="${SAMPLES_DIR}/updated_back.weights"
cat ${SAMPLE_FILE} | \
    ${AUTOSCHED_BIN}/retrain_cost_model \
        --epochs="1" \
        --rates="0.001" \
        --num_cores=${NUM_CORES} \
        --initial_weights=${WEIGHTS_FILE} \
        --weights_out=${WEIGHTS_OUTFILE} \
        --predict_only="1" \
        --best_benchmark="" \
        --best_schedule="" \
        --predictions_file=${PREDICTIONS_WITH_FILENAMES_FILE} \
        --verbose="0" \
        --partition_schedules="0" \
        --limit="0" \
        --alpha="0.001"

awk -F", " '{printf("%f, %f\n", $2, $3);}' ${PREDICTIONS_WITH_FILENAMES_FILE} > ${PREDICTIONS_FILE}
