#!/bin/bash
#
# submit-all-multi-model.calc-anoms-sub-anoms.bash
#
# Script for submitting a job for calculating the anomalies from the model mean state for a given model.
#
# For example: submit-all-multi-model.calc-anoms-sub-anoms.bash HadGEM3-GC31-MM 1960 1970 tas north-atlantic 2-5 DJF

# Make sure that the dictionaries.bash file exists
if [ ! -f $PWD/dictionaries.bash ]; then
    echo "ERROR: dictionaries.bash file does not exist"
    exit 1
fi

# Source the dictionaries.bash file
source $PWD/dictionaries.bash
# Echo the models
echo "[INFO] models: $models"

# Set the usage message
USAGE_MESSAGE="Usage: submit-all-multi-model.calc-anoms-sub-anoms.bash <model> <initial-year> <final-year> <variable> <region> <forecast-range> <season>"

# Check that the correct number of arguments have been passed
if [ $# -ne 7 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# Extract the model, initialization year, run number, variable, region, forecast range and season
model=$1
initial_year=$2
final_year=$3
variable=$4
region=$5
forecast_range=$6
season=$7

# If model is a number
# Between 1-12
# Then model is equal to the ith element of the models array $models
if [[ $model =~ ^[0-9]+$ ]]; then
    # echo the model number
    echo "[INFO] Model number: $model"

    # Convert the models string to an array
    models_array=($models)
    # Echo the models array
    echo "[INFO] models array: ${models_array[*]}"

    # Extract the numbered element of the models array
    model=${models_array[$model-1]}

    # echo the model name
    echo "[INFO] Model name: $model"
    echo "[INFO] Extracting data for model: $model"
fi

# Set the extractor script
EXTRACTOR=$PWD/process_scripts/multi-model.calc-anoms-sub-anoms.bash

# Check that the extractor script exists
# if not exit with an error
if [ ! -f $EXTRACTOR ]; then
    echo "ERROR: extractor script does not exist: $EXTRACTOR"
    exit 1
fi

# Make sure that cdo is loaded
module load jaspy

# If model=all, then run a for loop over all of the models
if [ "$model" == "all" ]; then

    # Echo that all of the models are being processed
    echo "[INFO] All of the models in list: ${models} are being processed"

    # Loop over the models
    for model in $models; do

        # Echo the model name
        echo "[INFO] Calculating anomalies from model mean state for model: $model"

        # Set up the output directory
        # For the LOTUS outputs
        OUTPUT_DIR="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/lotus-outputs"
        mkdir -p $OUTPUT_DIR

        # Loop over the years
        for year in $(seq $initial_year $final_year); do

            # Echo the year
            echo "[INFO] Calculating anomalies for year: $year"

            # Echo the output directory
            echo "[INFO] Output directory: $OUTPUT_DIR"

            # Echo info for job submission
            echo "[INFO] Submitting job for model: $model, variable: $variable, region: $region, forecast range: $forecast_range, season: $season, init_method: $init_method"

            # Submit the job
            sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/${model}.${year}.${variable}.${region}.${forecast_range}.${season}-calc-anoms.out -e $OUTPUT_DIR/${model}.${year}.${variable}.${region}.${forecast_range}.${season}-calc-anoms.err $EXTRACTOR $model $year $variable $region $forecast_range $season

        done
    done
else
    # For the inidividual models
    # Echo the model name
    echo "[INFO] Calculating anomalies from model mean state for model: $model"

    # Set up the output directory
    # For the LOTUS outputs
    OUTPUT_DIR="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/lotus-outputs"
    mkdir -p $OUTPUT_DIR

    # Echo the number of ensemble members
    echo "[INFO] Number of ensemble members: $run"

    # Loop over the years
    for year in $(seq $initial_year $final_year); do

        # Echo the year
        echo "[INFO] Calculating anomalies for year: $year"
        
        # Echo the output directory
        echo "[INFO] Output directory: $OUTPUT_DIR"

        # Echo info for job submission
        echo "[INFO] Submitting job for model: $model, variable: $variable, region: $region, forecast range: $forecast_range, season: $season, run: $run, init_method: $init_method"

        # Submit the job
        sbatch --partition=short-serial -t 5 -o $OUTPUT_DIR/${model}.${year}.${variable}.${region}.${forecast_range}.${season}-calc-anoms.out -e $OUTPUT_DIR/${model}.${year}.${variable}.${region}.${forecast_range}.${season}-calc-anoms.err $EXTRACTOR $model $year $variable $region $forecast_range $season

    done
fi

# End of script
exit 0