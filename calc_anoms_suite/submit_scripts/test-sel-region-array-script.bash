#!/bin/bash
#SBATCH --job-name=ben-array-sel-season-test-years
#SBATCH --partition=short-serial
#SBATCH --mem=30000
#SBATCH -o /gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-years/sel-season-test-years-%A_%a.out
#SBATCH -e /gws/nopw/j04/canari/users/benhutch/batch_logs/ben-array-sel-season-test-years/sel-season-test-years-%A_%a.err
#SBATCH --time=60:00
#SBATCH --array=1960-2018%4

# FIXME: Replace 1960-1965 with 1960-2018 once tested

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
process_script="/home/users/benhutch/lagging-NAO-test-suite/calc_anoms_suite/process_scripts/multi-model.sel-region-forecast-range-season.bash"

# Declare an empty associative array
declare -A nens_extractor

# Extract the models list using a case statement
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

# Echo the models
echo "Models are: $models"

# Echo the values of the nens_extractor
echo "Values of nens_extractor are: ${nens_extractor[@]}"

# Echo the keys of the nens_extractor
echo "Keys of nens_extractor are: ${!nens_extractor[@]}"

# In the case of individual models
echo "Submitting single model: $model"

# Echo the year which we are processing
echo "Submitting year: ${SLURM_ARRAY_TASK_ID}"

# Declare the nens extractor
declare -p nens_extractor

# Extract the number of ensemble members for the model
nens=${nens_extractor[$model]}

# Echo the model
echo "Model is: $model"

# Echo the number of ensemble members
echo "Number of ensemble members is: $nens"

# Loop over the ensemble members
for run in $(seq 1 $nens); do

    # Echo the ensemble member
    echo "Processing ensemble member: $run"

    # If the model is not EC-Earth3 or NorCPM1
    if [ $model != "EC-Earth3" ] && [ $model != "NorCPM1" ]; then

        echo "Only one init scheme for $model"

        # Set the init scheme
        init_scheme="1"

        bash ${process_script} ${model} ${SLURM_ARRAY_TASK_ID} ${run} ${variable} ${season} ${experiment} ${init_scheme}
    else
        echo "Two init schemes for $model"
        # Loop over the init schemes
        for init_scheme in $(seq 1 2); do

            # Echo the init scheme
            echo "Processing init scheme: $init_scheme"

            # Submit the job
            bash ${process_script} ${model} ${SLURM_ARRAY_TASK_ID} ${run} ${variable} ${season} ${experiment} ${init_scheme}
        done
    fi
done

# End of script
echo "Finished processing ${model} ${variable} ${season} ${experiment} ${SLURM_ARRAY_TASK_ID}"