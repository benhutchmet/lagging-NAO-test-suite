#!/bin/bash
#
# Calculates the NAO index for each model, run and initialisation method
# for each season and forecast range
# NAO anomaly = azores - iceland
#    
# mutli-model.calc-NAO-updated.bash <model> <variable> \
# <forecast-range> <season> <lag> <run> <init>
#
# For example: multi-model.calc-NAO.bash HadGEM3-GC31-MM psl \
# 2-9 DJFM 4 1 1
#

# set the usage message
USAGE_MESSAGE="Usage: mutli-model.calc-NAO.bash <model> <variable> \
 <forecast-range> <season> <lag> <run> <init>"

if [ $# -ne 7 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the variables from the command line
model=$1
variable=$2
forecast_range=$3
season=$4
lag=$5
run=$6
init=$7

module load jaspy

# set the input and output directories
AZORES_BASE_DIR="/work/scratch-nopw2/benhutch/$variable/$model/azores/years_${forecast_range}/$season/outputs"
ICELAND_BASE_DIR="/work/scratch-nopw2/benhutch/$variable/$model/iceland/years_${forecast_range}/$season/outputs"
AZORES_INPUT_DIR="$AZORES_BASE_DIR/lag_${lag}_anoms"
ICELAND_INPUT_DIR="$ICELAND_BASE_DIR/lag_${lag}_anoms"
OUTPUT_DIR="/work/scratch-nopw2/benhutch/$variable/$model/NAO/years_${forecast_range}/$season/outputs"
# make the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# output from apply lag will look something like this
# years-2-9-DJFM-azores-psl_Amon_BCC-CSM2-MR_dcppA-hindcast_s-e_anoms.init-same.nc
# "years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.same-init.nc"

pattern_same="years-${forecast_range}-${season}-*-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.same-init.nc"
pattern_minus_1="years-${forecast_range}-${season}-*-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.init-minus-1.nc"
pattern_minus_2="years-${forecast_range}-${season}-*-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.init-minus-2.nc"
pattern_minus_3="years-${forecast_range}-${season}-*-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.init-minus-3.nc"

# construct the file paths for the azores files
azores_pattern_same_file="${AZORES_INPUT_DIR}/${pattern_same}"
azores_pattern_minus_1_file="${AZORES_INPUT_DIR}/${pattern_minus_1}"
azores_pattern_minus_2_file="${AZORES_INPUT_DIR}/${pattern_minus_2}"
azores_pattern_minus_3_file="${AZORES_INPUT_DIR}/${pattern_minus_3}"

# echo these patterns
echo "azores_pattern_same_file = $azores_pattern_same_file"
echo "azores_pattern_minus_1_file = $azores_pattern_minus_1_file"
echo "azores_pattern_minus_2_file = $azores_pattern_minus_2_file"
echo "azores_pattern_minus_3_file = $azores_pattern_minus_3_file"

# construct the file paths for the iceland files
iceland_pattern_same_file="${ICELAND_INPUT_DIR}/${pattern_same}"
iceland_pattern_minus_1_file="${ICELAND_INPUT_DIR}/${pattern_minus_1}"
iceland_pattern_minus_2_file="${ICELAND_INPUT_DIR}/${pattern_minus_2}"
iceland_pattern_minus_3_file="${ICELAND_INPUT_DIR}/${pattern_minus_3}"

# check if the files exist
if [ ! -f $azores_pattern_same_file ]; then
    echo "[ERROR] File $azores_pattern_same_file does not exist"
    exit 1
fi

if [ ! -f $azores_pattern_minus_1_file ]; then
    echo "[ERROR] File $azores_pattern_minus_1_file does not exist"
    exit 1
fi

if [ ! -f $azores_pattern_minus_2_file ]; then
    echo "[ERROR] File $azores_pattern_minus_2_file does not exist"
    exit 1
fi

if [ ! -f $azores_pattern_minus_3_file ]; then
    echo "[ERROR] File $azores_pattern_minus_3_file does not exist"
    exit 1
fi

if [ ! -f $iceland_pattern_same_file ]; then
    echo "[ERROR] File $iceland_pattern_same_file does not exist"
    exit 1
fi

if [ ! -f $iceland_pattern_minus_1_file ]; then
    echo "[ERROR] File $iceland_pattern_minus_1_file does not exist"
    exit 1
fi

if [ ! -f $iceland_pattern_minus_2_file ]; then
    echo "[ERROR] File $iceland_pattern_minus_2_file does not exist"
    exit 1
fi

if [ ! -f $iceland_pattern_minus_3_file ]; then
    echo "[ERROR] File $iceland_pattern_minus_3_file does not exist"
    exit 1
fi


# Set up the output files
# for the NAO
output_same="${OUTPUT_DIR}/NAO_${model}_${variable}_${region}_${season}_lag-${lag}_r${run}i${init}.same-init.nc"
output_shifted_minus_1="${OUTPUT_DIR}/NAO_${model}_${variable}_${region}_${season}_lag-${lag}_r${run}i${init}.init-minus-1.nc"
output_shifted_minus_2="${OUTPUT_DIR}/NAO_${model}_${variable}_${region}_${season}_lag-${lag}_r${run}i${init}.init-minus-2.nc"
output_shifted_minus_3="${OUTPUT_DIR}/NAO_${model}_${variable}_${region}_${season}_lag-${lag}_r${run}i${init}.init-minus-3.nc"

# if the output files already exist, echo that this will be overwritten
# and remove the existing files
# within the output directory
if [ -f "$output_same" ]; then
    echo "[WARNING] Output file already exists: $output_same"
    rm -f "$output_same"
    echo "[INFO] Removed existing file."
fi

if [ -f "$output_shifted_minus_1" ]; then
    echo "[WARNING] Output file already exists: $output_shifted_minus_1"
    rm -f "$output_shifted_minus_1"
    echo "[INFO] Removed existing file."
fi

if [ -f "$output_shifted_minus_2" ]; then
    echo "[WARNING] Output file already exists: $output_shifted_minus_2"
    rm -f "$output_shifted_minus_2"
    echo "[INFO] Removed existing file."
fi

if [ -f "$output_shifted_minus_3" ]; then
    echo "[WARNING] Output file already exists: $output_shifted_minus_3"
    rm -f "$output_shifted_minus_3"
    echo "[INFO] Removed existing file."
fi

# Special case for IPSL-CM6A-LR
# Some of the 1960 runs have not had their spatial mean taken
# use an if statement to check for this
if [ "$model" = "IPSL-CM6A-LR" ]; then
    echo "[INFO] Model is IPSL-CM6A-LR"
    echo "[INFO] Special case, taking the field means when calculating NAO anoms"

    # Take the differences between the fldmeans of the azores and iceland files
    cdo sub -fldmean ${azores_pattern_same_file} -fldmean ${iceland_pattern_same_file} $output_same
    cdo sub -fldmean ${azores_pattern_minus_1_file} -fldmean ${iceland_pattern_minus_1_file} $output_shifted_minus_1
    cdo sub -fldmean ${azores_pattern_minus_2_file} -fldmean ${iceland_pattern_minus_2_file} $output_shifted_minus_2
    cdo sub -fldmean ${azores_pattern_minus_3_file} -fldmean ${iceland_pattern_minus_3_file} $output_shifted_minus_3
else
    echo "[INFO] Model is not IPSL-CM6A-LR"

    # take the difference between the azores and iceland files
    # for init-minus-1
    cdo sub $azores_pattern_minus_1_file $iceland_pattern_minus_1_file $output_shifted_minus_1
    # for init-minus-2
    cdo sub $azores_pattern_minus_2_file $iceland_pattern_minus_2_file $output_shifted_minus_2
    # for init-minus-3
    cdo sub $azores_pattern_minus_3_file $iceland_pattern_minus_3_file $output_shifted_minus_3

    # calculate the NAO for the same init files
    cdo sub $azores_pattern_same_file $iceland_pattern_same_file $output_same
fi

# remove the temp files
rm ${OUTPUT_DIR}/*temp*.nc

# check if the output files exist
if [ ! -f $output_shifted_minus_1 ]; then
    echo "[ERROR] File $output_shifted_minus_1 does not exist"
    exit 1
fi

if [ ! -f $output_shifted_minus_2 ]; then
    echo "[ERROR] File $output_shifted_minus_2 does not exist"
    exit 1
fi

if [ ! -f $output_shifted_minus_3 ]; then
    echo "[ERROR] File $output_shifted_minus_3 does not exist"
    exit 1
fi

if [ ! -f $output_same ]; then
    echo "[ERROR] File $output_same does not exist"
    exit 1
fi

# script finished
echo "Script finished successfully"
exit 0

