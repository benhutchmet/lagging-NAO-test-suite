#!/bin/bash
#SBATCH --partition=test
#SBATCH --job-name=ben-array-sel-season-test-members
#SBATCH -o /gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-members/%j.out
#SBATCH -e /gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-members/%j.err
#SBATCH --time=60:00
#SBATCH --array=1-5

# TODO: Modify sbtach array and partition after testing

# Form the path for the logs folder and make sure it exists
logs_dir="/gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-members"

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
echo "Task id is: ${SLURM_ARRAY_TASK_ID}"

# Echo the CLI arguments
echo "CLI arguments are: $@"
echo "Number of CLI arguments is: $#"
echo "Desired no. of arguments is: 5"

# Check if the correct number of arguments were passed
if [ $# -ne 5 ]; then
    echo "Usage: sbatch test-sel-season-array-script.bash <model> <variable> <season> <experiment> <init-year>"
    echo "Example: sbatch test-sel-season-array-script.bash HadGEM3-GC31-MM psl DJFM dcppA-hindcast 1960"
    exit 1
fi

# Extract the model, variable, region, forecast range and season
model=$1
variable=$2
season=$3
experiment=$4
init_year=$5

# Print the model, variable, region, forecast range and season
echo "Model is: $model"
echo "Variable is: $variable"
echo "Season is: $season"
echo "Experiment is: $experiment"
echo "Init year is: $init_year"

# Load cdo
module load jaspy

# Set up the processing script
process_script="/home/users/benhutch/lagging-NAO-test-suite/calc_anoms_suite/process_scripts/multi-model.sel-region-forecast-range-season.bash"

# Check if the processing script exists
if [ ! -f $process_script ]; then
    echo "ERROR: processing script does not exist"
    exit 1
fi

# If the method is 'all' then exit with an error
if [ $model == "all" ]; then
    echo "ERROR: method cannot be 'all'"
    exit 1
fi

# Declare an empty associative array
declare -A nens_extractor

# Extract the models list using a case statement
# and the nen_extractor array
case $variable in
"psl")
    models=$models

    # Loop over and copy each key-value pair from the psl_models_nens
    for key in "${!psl_models_nens[@]}"; do
        nens_extractor[$key]=${psl_models_nens[$key]}
    done
    ;;
"sfcWind")
    models=$sfcWind_models

    # Loop over and copy each key-value pair from the sfcWind_models_nens
    for key in "${!sfcWind_models_nens[@]}"; do
        nens_extractor[$key]=${sfcWind_models_nens[$key]}
    done
    ;;
"rsds")
    models=$rsds_models

    # Loop over and copy each key-value pair from the rsds_models_nens
    for key in "${!rsds_models_nens[@]}"; do
        nens_extractor[$key]=${rsds_models_nens[$key]}
    done
    ;;
"tas")
    models=$tas_models

    # loop over and copy each key-value pair from the tas_models_nens
    for key in "${!tas_models_nens[@]}"; do
        nens_extractor[$key]=${tas_models_nens[$key]}
    done
    ;;
"tos")
    models=$tos_models

    # Loop over and copy each key-value pair from the tos_models_nens
    for key in "${!tos_models_nens[@]}"; do
        nens_extractor[$key]=${tos_models_nens[$key]}
    done
    ;;
*)
    echo "ERROR: variable not recognized: $variable"
    exit 1
    ;;
esac

# Echo the keys of the nens_extractor
echo "Keys of nens_extractor are: ${!nens_extractor[@]}"

# Echo the values of the nens_extractor
echo "Values of nens_extractor are: ${nens_extractor[@]}"

# Extract the number of ensemble members for the model
nens=${nens_extractor[$model]}

# Echo the number of ensemble members
echo "Number of ensemble members for $model is: $nens"

# If the slurm array task id is greater than the number of ensemble members
if [ $SLURM_ARRAY_TASK_ID -gt $nens ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is greater than the number of ensemble \
    members"
    echo "SLURM_ARRAY_TASK_ID is: $SLURM_ARRAY_TASK_ID"
    echo "Number of ensemble members is: $nens for model $model"
    exit 1
fi

# Echo the model which we are processing
echo "Processing model: $model"

# Echo the initialisation yeart which we are processing
echo "Processing init year: $init_year"

# Echo the ensemble member which we are processing
echo "Processing ensemble member: $SLURM_ARRAY_TASK_ID"

# Run the process script as an array job
# For the specific model, init year and ensemble member
bash $process_script ${model} ${init_year} ${SLURM_ARRAY_TASK_ID} ${variable} ${season} ${experiment}

# End of script
echo "Finished processing model $model, init year $init_year and ensemble member $SLURM_ARRAY_TASK_ID for variable $variable, season $season and experiment $experiment"
