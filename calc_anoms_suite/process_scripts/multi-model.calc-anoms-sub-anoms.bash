#!/bin/bash
#
# multi-model.calc-anoms-sub-anoms.bash
#
# Script for calculating the anomalies from the model mean state for a given model, variable, region, forecast range, and season.
#
# For example: multi-model.calc-anoms-sub-anoms.bash HadGEM3-GC31-MM 1960 psl north-atlantic 2-5 DJF 92500

# Set the usage message
USAGE_MESSAGE="Usage: multi-model.calc-anoms-sub-anoms.bash <model> <initialization-year> <variable> <season>"

# Check that the correct number of arguments have been passed
if [ $# -ne 4 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# Extract the model-year-run, variable, region, forecast range and season
model=$1
year=$2
variable=$3
season=$4

# Load cdo
module load jaspy

# Set the base directory
# Set up the base directory
if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    # base_dir="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/plev_${pressure_level}/outputs"
    echo "ERROR: pressure level not set up for ua and va"
    exit 1
else
    # Base directory
    # Example: /gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/psl/BCC-CSM2-MR/global/all_forecast_years/DJFM/outputs/
    base_dir="/work/scratch-nopw2/benhutch/${variable}/${model}/global/all_forecast_years/${season}/outputs"
fi

OUTPUT_DIR="/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/${variable}/${model}/global/all_forecast_years/${season}/outputs/anoms"

# Function for calculating anomalies
calculate_anoms() {
    # Extract the initialization scheme
    init_scheme=$1
    variable=$2

    if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
        # Set up the input file path
        # INPUT_FILES="$base_dir/mean-years-${forecast_range}-${season}-${region}-plev-${variable}_?mon_${model}_dcppA-hindcast_s${year}-r*${init_scheme}*.nc"
        echo "ERROR: pressure level not set up for ua and va"
        exit 1
    else
        # Set up the input file path
        INPUT_FILES="$base_dir/all-years-${season}-global-${variable}_?mon_${model}_dcppA-hindcast_s${year}-r*${init_scheme}*.nc"
    fi

    # Set up the model mean state file path
    MODEL_MEAN_STATE="$base_dir/tmp/model_mean_state_${init_scheme}.nc"

    # Echo the files to be processed
    echo "Calculating anomalies for: $INPUT_FILES"
    echo "Using model mean state file: $MODEL_MEAN_STATE"

    # # Check that the model mean state file exists
    if [ ! -f $MODEL_MEAN_STATE ]; then
        echo "ERROR: model mean state file not found ${init_scheme}: ${MODEL_MEAN_STATE}"
        exit 1
    fi

    # Calculate the anomalies
    # By looping over the input files
    for INPUT_FILE in $INPUT_FILES; do
        # Echo the input file
        echo "Calculating anomalies for: $INPUT_FILE"

        # Check that the input file exists
        if [ ! -f $INPUT_FILE ]; then
            echo "ERROR: input file not found: ${INPUT_FILE}"
            exit 1
        fi

        # Set up the output file name
        filename=$(basename ${INPUT_FILE})
        OUTPUT_FILE="$OUTPUT_DIR/${filename%.nc}-anoms.nc"

        # If this output file already exists, then delete it
        if [ -f $OUTPUT_FILE ]; then
            echo "WARNING: output file already exists"
            echo "WARNING: deleting existing output file"
            rm -f $OUTPUT_FILE
        fi

        # Subtract the time average of the model mean state from the input file
        cdo sub $INPUT_FILE -timavg $MODEL_MEAN_STATE $OUTPUT_FILE

        # # Calculate the anomalies
        # cdo sub $INPUT_FILE $MODEL_MEAN_STATE $OUTPUT_FILE
    done
}

# Create output directories
mkdir -p $OUTPUT_DIR

if [ "$variable" == "tos" ]; then
    # Processing
    case $model in
        "NorCPM1")
            calculate_anoms "i1" $variable
            calculate_anoms "i2" $variable
            ;;
        "EC-Earth3")
            calculate_anoms "i1" $variable
            calculate_anoms "i2" $variable
            ;;
        *)
            # For all other models, use a wildcard for init_scheme
            calculate_anoms "i1" $variable
            ;;
    esac
else
    # Processing
    case $model in
        "NorCPM1")
            calculate_anoms "i1" $variable
            calculate_anoms "i2" $variable
            ;;
        "EC-Earth3")
            calculate_anoms "i1" $variable
            calculate_anoms "i2" $variable
            ;;
        *)
            # For all other models, use a wildcard for init_scheme
        calculate_anoms "i1" $variable
            ;;
    esac
fi

echo "Anomalies calculated for year ${year}, model $model, variable $variable, region global, season $season and saved to $OUTPUT_DIR"

# End of script
exit 0