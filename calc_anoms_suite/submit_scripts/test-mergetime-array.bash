#!/bin/bash
#SBATCH --partition=short-serial
#SBATCH --job-name=mergetime-array-test
#SBATCH -o /gws/nopw/j04/canari/users/benhutch/batch_logs/mergetime-array-test/%j.out
#SBATCH -e /gws/nopw/j04/canari/users/benhutch/batch_logs/mergetime-array-test/%j.err
#SBATCH --time=10:00
#SBATCH --array=1-40

# Form the path for the logs folder and make sure it exists
logs_dir="/gws/nopw/j04/canari/users/benhutch/batch_logs/mergetime-array-test"

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
echo "Desired no. of arguments is: 6"

# Check if the correct number of arguments were passed
if [ $# -ne 6 ]; then
    echo "Usage: sbatch mergetime-array-test.bash <model> <variable> <region> \
    <forecast-range> <season> <pressure-level>"
    echo "Example: sbatch mergetime-array-test.bash HadGEM3-GC31-MM psl global \
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
process_script="/home/users/benhutch/skill-maps-rose-suite/process_scripts/multi-model.mergetime.bash"

# Check that the process script exists
if [ ! -f $process_script ]; then
    echo "ERROR: process script does not exist: $process_script"
    exit 1
fi

# If model == "all", then exit with an error
# NOTE: this method should be redundant now
if [ $model == "all" ]; then
    echo "ERROR: model cannot be all"
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

# Echo the ensemble member which we are processing
echo "Processing ensemble member: $SLURM_ARRAY_TASK_ID"

# If the model contains two init schemes
# We need to submit two sets of jobs
if [[ $model == "EC-Earth3" ]] || [[ $model == "NorCPM1" ]]; then

    # Echo the model name
    echo "Processing model: $model which contains two init schemes"

    # Loop over the init schemes
    for init_scheme in $(seq 1 2); do

        # Echo the init scheme
        echo "Processing init scheme: $init_scheme"

        # Run the process script as an array job
        bash $process_script ${model} ${variable} ${region} ${forecast_range} \
             ${season} ${SLURM_ARRAY_TASK_ID} ${init_scheme} ${pressure_level}

    done

fi

# If the model contains one init scheme
echo "Model is not EC-Earth3 or NorCPM1"
echo "Model is: $model and only contains one init scheme"

# Echo the ensemble member which we are processing
echo "Processing ensemble member: $SLURM_ARRAY_TASK_ID"

# Set the init scheme to 1
init_scheme=1

# Echo the init scheme
echo "Processing init scheme: $init_scheme for model $model"

# Run the process script as an array job
bash $process_script ${model} ${variable} ${region} ${forecast_range} \
     ${season} ${SLURM_ARRAY_TASK_ID} ${init_scheme} ${pressure_level}

# End of script
echo "Finished processing model $model, variable $variable, region $region, \
forecast range $forecast_range, season $season, init scheme $init_scheme and \
ensemble member $SLURM_ARRAY_TASK_ID"
