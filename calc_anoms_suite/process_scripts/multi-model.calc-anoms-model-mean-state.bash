#!/bin/bash
#
# multi-model.calc-anoms-model-mean-state.bash
#
# Script for calculating the model mean state for a given model.
#
# For example: calculate-model-mean-states.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF 92500

# Set the usage message
USAGE_MESSAGE="Usage: multi-model.calc-anoms-model-mean-state.bash <model> <variable> <region> <forecast-range> <season> <pressure-level>"

# Check that the correct number of arguments have been passed
if [ $# -ne 6 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# Extract the model, variable, region, forecast range and season
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5
pressure_level=$6

# Load cdo
module load jaspy

# Set up the base directory
if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    base_dir="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/plev_${pressure_level}/outputs"
else
    # Base directory
    base_dir="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/outputs"
fi

# Function for processing files
process_files() {
    init_scheme=$1
    variable=$2

    # If variable is ua or va
    # Then set up the file path differently
    if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
        files_path="$base_dir/mean-years-${forecast_range}-${season}-${region}-plev-${variable}_?mon_${model}_dcppA-hindcast_s????-r*${init_scheme}*.nc"
    else
        files_path="$base_dir/mean-years-${forecast_range}-${season}-${region}-${variable}_?mon_${model}_dcppA-hindcast_s????-r*${init_scheme}*.nc"
    fi

    # Echo the files to be processed
    echo "Calculating model mean state for: $files_path"
    echo "Calculating model mean state for model $model, variable $variable, region $region, forecast range $forecast_range, season $season and init scheme $init_scheme"

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

if [ "$variable" == "tos" ]; then
    # Processing
    case $model in
        "NorCPM1")
            process_files "i1" $variable
            process_files "i2" $variable
            ;;
        "EC-Earth3")
            process_files "i1" $variable
            process_files "i2" $variable
            ;;
        *)
            # For all other models, use a wildcard for init_scheme
            process_files "i1" $variable
            ;;
    esac
else
    # Processing
    case $model in
        "NorCPM1")
            process_files "i1" $variable
            process_files "i2" $variable
            ;;
        "EC-Earth3")
            process_files "i1" $variable
            process_files "i2" $variable
            ;;
        *)
            # For all other models, use a wildcard for init_scheme
            process_files "i1" $variable
            ;;
    esac
fi

# Clean up temporary files
rm -f ${base_dir}/tmp/temp-*.nc

echo "Model mean states have been calculated for $model $variable $region $forecast_range $season and saved in $base_dir/tmp"

# End of script
exit 0