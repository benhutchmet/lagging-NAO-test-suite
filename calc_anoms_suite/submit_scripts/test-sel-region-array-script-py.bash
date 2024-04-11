#!/bin/bash
#SBATCH --job-name=ben-array-sel-season-test-years
#SBATCH --partition=short-serial
#SBATCH --mem=10000
#SBATCH -o /gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-years/sel-season-test-years-april.out
#SBATCH -e /gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-years/sel-season-test-years-april.err
#SBATCH --time=1800:00
#SBATCH --array=1960-2018

# Form the path for the logs folder and make sure it exists
logs_dir="/gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-years"

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
source /home/users/benhutch/lagging-NAO-test-suite/calc_anoms_suite/dictionaries.bash

# Echo th task id
echo "Task id is: ${SLURM_ARRAY_TASK_ID}"

# # Set up the error log files
# ./test-sel-region-array-script.bash ${SLURM_ARRAY_TASK_ID}

# Print the CLI arguments
echo "CLI arguments are: $@"
echo "Number of CLI arguments is: $#"
echo "Desired no. of arguments is: 7"

# Check if the correct number of arguments were passed
if [ $# -ne 7 ]; then
    echo "Usage: sbatch test-sel-region-array-script.bash <model> <variable> <season> <experiment> <region> <start_year> <end_year>"
    echo "Example: sbatch test-sel-region-array-script.bash HadGEM3-GC31-MM psl DJFM dcppA-hindcast global 1960 2018"
    exit 1
fi

# Extract the model, variable, region, forecast range and season
model=$1
variable=$2
season=$3
experiment=$4
region=$5
start_year=$6
end_year=$7

# Print the model, variable, region, forecast range and season
echo "Model is: $model"
echo "Variable is: $variable"
echo "Season is: $season"
echo "Experiment is: $experiment"
echo "Region is: $region"
echo "Start year is: $start_year"
echo "End year is: $end_year"

# Load cdo
module load jaspy

# Set up the process script
process_script="/home/users/benhutch/lagging-NAO-test-suite/testing/sel_reg_fcst_seasons_functions.py"

# Submit the process script
python $process_script $model $variable $season $experiment $region ${start_year} ${end_year} ${SLURM_ARRAY_TASK_ID}

# End of script
echo "Finished processing ${model} ${variable} ${season} ${experiment} ${SLURM_ARRAY_TASK_ID}"