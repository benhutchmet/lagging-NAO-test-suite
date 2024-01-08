#!/bin/bash
#SBATCH --partition=short-serial
#SBATCH --job-name=calc-anoms-array-test
#SBATCH -o /gws/nopw/j04/canari/users/benhutch/batch_logs/calc-anoms-array-test/%j.out
#SBATCH -e /gws/nopw/j04/canari/users/benhutch/batch_logs/calc-anoms-array-test/%j.err
#SBATCH --time=10:00
#SBATCH --array=1960-2018

# Form the path for the logs folder and make sure it exists
logs_dir="/gws/nopw/j04/canari/users/benhutch/batch_logs/calc-anoms-array-test"

# If the logs directory does not exist
if [ ! -d $logs_dir ]; then
    # Make the logs directory
    mkdir -p $logs_dir
fi

# Verify that the dictionaries.bash file exists
if [ ! -f $PWD/dictionaries.bash ]; then
    echo "ERROR: dictionaries.bash file does not exist"
    exit 1
fi

# Source the dictionaries
source /home/users/benhutch/skill-maps-rose-suite/dictionaries.bash

# Echo th task id
echo "SLURM_ARRAY_TASK_ID is: ${SLURM_ARRAY_TASK_ID}"

# Echo trhe CLI's
echo "CLI arguments are: $@"
echo "Number of CLI arguments is: $#"
echo "Desired no. of arguments is: 6" # FIXME: might need to change this

# Check if the correct number of arguments were passed
if [ $# -ne 6 ]; then
    echo "Usage: sbatch calc-anoms-array-test.bash <model> <variable> <region> \
<forecast-range> <season> <pressure-level>"
    echo "Example: sbatch calc-anoms-array-test.bash HadGEM3-GC31-MM psl global \
2-9 DJFM 100000"
    exit 1
fi

# Extract the model, variable, region, forecast range and season
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5
pressure_level=$6

# Print the model, variable, region, forecast range and season
echo "Model is: $model"
echo "Variable is: $variable"
echo "Region is: $region"
echo "Forecast range is: $forecast_range"
echo "Season is: $season"
echo "Pressure level is: $pressure_level"

# Load cdo
module load jaspy

# Set the process script
process_script="/home/users/benhutch/skill-maps-rose-suite/process_scripts/multi-model.calc-anoms-sub-anoms.bash"

# Check that the process script exists
if [ ! -f $process_script ]; then
    echo "ERROR: process script does not exist: $process_script"
    exit 1
fi

# If model is all
if [ $model == "all" ]; then
    
    echo "Extracting data for all models"

    # Extract the models list using a case statement
    case $variable in
    "psl")
        models=$models
        ;;
    "sfcWind")
        models=$sfcWind_models
        ;;
    "rsds")
        models=$rsds_models
        ;;
    "tas")
        models=$tas_models
        ;;
    "tos")
        models=$tos_models
        ;;
    *)
        echo "ERROR: variable not recognized: $variable"
        exit 1
        ;;
    esac

    # Loop over the models
    for model in $models; do

        # Echo the model name
        echo "Extracting data for model: $model"

        # Echo the year which we are processing
        echo "Processing year: ${SLURM_ARRAY_TASK_ID}"

        # Run the process script as an array job
        bash $process_script ${model} ${SLURM_ARRAY_TASK_ID} ${variable} \
        ${region} ${forecast_range} ${season} ${pressure_level}

    done

    # End the script
    echo "Finished processing anomalies for  ${model} ${variable} ${region} \
    ${forecast_range} ${season} ${pressure_level}"

fi

# In the other case of individual models
echo "Extracting data for single model: $model"

# Echo the year which we are processing
echo "Processing year: ${SLURM_ARRAY_TASK_ID}"

# Run the process script as an array job
bash $process_script ${model} ${SLURM_ARRAY_TASK_ID} ${variable} \
    ${region} ${forecast_range} ${season} ${pressure_level}

# End the script
echo "Finished processing anomalies for  ${model} ${variable} ${region} \
${forecast_range} ${season} ${pressure_level}"