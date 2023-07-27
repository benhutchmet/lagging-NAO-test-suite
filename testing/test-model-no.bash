#!/bin/bash

# import the models list
source ~/lagging-NAO-test-suite/dictionaries.bash
# echo the multi-models list
echo "[INFO] models list: $models"

# set the usage message
USAGE_MESSAGE="Usage: test-model-no.bash <model_no>"

# check that the correct number of arguments have been passed
if [ $# -ne 1 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the model, initial year and final year
model=$1

# If model is a number
# Between 1-12
# Then model is equal to the ith element of the models array $models
if [[ $model =~ ^[0-9]+$ ]]; then
    # echo the model number
    echo "[INFO] Model number: $model"

    # Extract the model name from the models array
    # if the model array is given by
    # models="BCC-CSM2-MR MPI-ESM1-2-HR CanESM5 CMCC-CM2-SR5 HadGEM3-GC31-MM EC-Earth3 MPI-ESM1-2-LR FGOALS-f3-L MIROC6 IPSL-CM6A-LR CESM1-1-CAM5-CMIP5 NorCPM1"
    # then the model name is the ith element of the array
    # seperate the elements of the array by a space
    # and then extract the ith element of the array
    models_array=($models)

    # echo the models array
    echo "[INFO] models array: ${models_array[*]}"
    # extract the numbered element of the models array
    model=${models_array[$model-1]}

    # echo the model name
    echo "[INFO] Model name: $model"
    echo "[INFO] Extracting data for model: $model"

fi