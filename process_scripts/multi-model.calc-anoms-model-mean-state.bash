#!/bin/bash
#
# multi-model.calc-anoms-model-mean-state.bash
#
# Script for calculating the model mean state for a given model.
#
# For example: calculate-model-mean-states.bash HadGEM3-GC31-MM psl DJFM

# TODO: this needs to be tested

# Set the usage message
USAGE_MESSAGE="Usage: multi-model.calc-anoms-model-mean-state.bash <model> <variable> <season>"

# Check that the correct number of arguments have been passed
if [ $# -ne 3 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# Extract the model, variable, region, forecast range and season
model=$1
variable=$2
season=$3

# Load cdo
module load jaspy

# Base directory
base_dir="/work/scratch-nopw2/benhutch/${variable}/${model}/global/all_forecast_years/${season}/outputs"

# Check that the base directory exists
if [ ! -d $base_dir ]; then
    echo "ERROR: base directory does not exist"
    exit 1
fi

# Function for processing files
process_files() {
    init_scheme=$1
    files_path="$base_dir/all-years-${season}-global-${variable}_Amon_${model}_dcppA-hindcast_s????-r*i${init_scheme}*.nc"

    # Test this path
    # all-years-DJFM-global-psl_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s????-r*i*.nc"

    # Echo the files to be processed
    echo "Calculating model mean state for: $files_path"
    echo "Calculating model mean state for model $model, variable $variable, region global, all forecast years, season $season and init scheme $init_scheme"

    # Set up the file name for the model mean state
    temp_model_mean_state="$base_dir/tmp/model_mean_state_${init_scheme}.nc"

    # Check that the model mean state file does not already exist
    # If one does exist, then delete it
    if [ -f $temp_model_mean_state ]; then
        echo "WARNING: model mean state file already exists ${init_scheme}"
        echo "WARNING: deleting existing model mean state file ${init_scheme}"
        rm -f $temp_model_mean_state
    fi

    # Take the ensemble mean of the time mean files
    cdo ensmean ${files_path} ${temp_model_mean_state}

    # Ensure that the model mean state file has been created
    if [ ! -f $temp_model_mean_state ]; then
        echo "ERROR: model mean state file not created ${init_scheme}"
        exit 1
    fi
}

# Create output directories
mkdir -p $base_dir/tmp

# Processing
case $model in
    "NorCPM1")
        process_files "i1"
        process_files "i2"
        ;;
    "EC-Earth3")
        process_files "i1"
        process_files "i2"
        ;;
    *)
        # For all other models, use a wildcard for init_scheme
        process_files "i1"
        ;;
esac

# Clean up temporary files
rm -f ${base_dir}/tmp/temp-*.nc

echo "Model mean states have been calculated for $model $variable $season and saved in $base_dir/tmp"
echo "Model mean state output file: ${temp_model_mean_state}"    

# End of script
exit 0