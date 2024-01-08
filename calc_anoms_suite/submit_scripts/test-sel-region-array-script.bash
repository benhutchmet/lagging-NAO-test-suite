#!/bin/bash
#SBATCH --job-name=ben-array-sel-season-test-years
#SBATCH --partition=short-serial
#SBATCH -o /gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-years/%j.out
#SBATCH -e /gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-years/%j.err
#SBATCH --time=10:00
#SBATCH --array=1970-1975

# FIXME: Replace 1960-1965 with 1960-2018 once tested

# Form the path for the logs folder and make sure it exists
logs_dir="/gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-region-test"

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
# Can be left as is
source /home/users/benhutch/skill-maps-rose-suite/dictionaries.bash

# Echo th task id
echo "Task id is: ${SLURM_ARRAY_TASK_ID}"

# # Set up the error log files
# ./test-sel-region-array-script.bash ${SLURM_ARRAY_TASK_ID}

# Print the CLI arguments
echo "CLI arguments are: $@"
echo "Number of CLI arguments is: $#"
echo "Desired no. of arguments is: 4"

# Check if the correct number of arguments were passed
if [ $# -ne 4 ]; then
    echo "Usage: sbatch test-sel-region-array-script.bash <model> <variable> <season> <experiment>"
    echo "Example: sbatch test-sel-region-array-script.bash HadGEM3-GC31-MM psl DJFM dcppA-hindcast"
    exit 1
fi

# Extract the model, variable, region, forecast range and season
model=$1
variable=$2
season=$3
experiment=$4

# Print the model, variable, region, forecast range and season
echo "Model is: $model"
echo "Variable is: $variable"
echo "Season is: $season"
echo "Experiment is: $experiment"

# Load cdo
module load jaspy

# Set up the process script
member_batch_script="/home/users/benhutch/lagging-NAO-test-suite/calc_anoms_suite/submit_scripts/test-sel-season-member-array.bash"

#FIXME: NENS extractor not working, but we use this in a different mode
# If model is all
if [ $model == "all" ]; then

    # Loop over the models
    for model in $models; do

        # Echo the model name
        echo "Processing model: $model"

        # Declare nameref for the nens extractor
        nens_extractor_ref=nens_extractor

        # Extract the number of ensemble members
        nens=${nens_extractor[$model]}

        # Loop over the years
        for run in $(seq 1 $nens); do

            # Echo the year
            echo "Processing run: $run"

            # Run the process script as an array job
            bash $process_script ${model} ${SLURM_ARRAY_TASK_ID} ${run} ${variable} ${region} ${forecast_range} ${season} ${experiment}

        done

    done

    # End the script
    echo "Finished processing ${model} ${variable} ${region} ${forecast_range} ${season} ${experiment} ${start_year} ${end_year}"
    exit 0

fi

# In the case of individual models
echo "Submitting single model: $model"

# Echo the year which we are processing
echo "Submitting year: ${SLURM_ARRAY_TASK_ID}"

# Echo the model
echo "Model is: $model"

# Run the batch script for submitting the ensemble member
sbatch ${member_batch_script} ${model} ${variable} ${season} ${experiment} ${SLURM_ARRAY_TASK_ID}

# End of script
echo "Finished processing ${model} ${variable} ${season} ${experiment} ${SLURM_ARRAY_TASK_ID}"