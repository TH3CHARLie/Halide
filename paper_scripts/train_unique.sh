#!/bin/bash

if [ $# -ne 5 ]; then
  echo "Usage: $0 init_weights_file output_weights_file predictions_file num_epochs sample_file"
  exit
fi

export AUTOSCHED_BIN="/home/xuanda/dev/abadams_random_pipeline/Halide/bin"
INIT_WEIGHTS_FILE=${1}
OUTPUT_WEIGHTS_FILE=${2}
PREDICTIONS_FILE=${3}
NUM_EPOCHS=${4}
SAMPLE_FILE=${5}
PREDICTIONS_WITH_FILENAMES_FILE="${PREDICTIONS_FILE}_with_filename"
echo
echo "Init Weights file: ${INIT_WEIGHTS_FILE}"
echo "Output weights file: ${OUTPUT_WEIGHTS_FILE}"
echo "Saving predictions to: ${PREDICTIONS_FILE}"
echo "Number of epochs: ${NUM_EPOCHS}"

NUM_CORES=10

cat ${SAMPLE_FILE} | \
    ${AUTOSCHED_BIN}/retrain_cost_model \
        --epochs=${NUM_EPOCHS} \
        --rates="0.0001" \
        --num_cores=${NUM_CORES} \
        --initial_weights=${INIT_WEIGHTS_FILE} \
        --weights_out=${OUTPUT_WEIGHTS_FILE} \
        --predict_only="0" \
        --best_benchmark="" \
        --best_schedule="" \
        --predictions_file=${PREDICTIONS_WITH_FILENAMES_FILE} \

awk -F", " '{printf("%f, %f\n", $2, $3);}' ${PREDICTIONS_WITH_FILENAMES_FILE} > ${PREDICTIONS_FILE}
