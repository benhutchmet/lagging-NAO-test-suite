#!/bin/bash
#
# Calculates the NAO index for each model, run and initialisation method
# for each season and forecast range
# NAO anomaly = azores - iceland
#    
# mutli-model.calc-NAO.bash <model> <variable> \
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

# create extra temp files for iceland for init-minus-1 and init-minus-3
iceland_pattern_minus_1_file_temp_2="${OUTPUT_DIR}/iceland_${model}_${variable}_${region}_${season}_lag-${lag}_r${run}i${init}.init-minus-1.temp-2.nc"
iceland_pattern_minus_3_file_temp_2="${OUTPUT_DIR}/iceland_${model}_${variable}_${region}_${season}_lag-${lag}_r${run}i${init}.init-minus-3.temp-2.nc"

# and for azores
azores_pattern_minus_1_file_temp_2="${OUTPUT_DIR}/azores_${model}_${variable}_${region}_${season}_lag-${lag}_r${run}i${init}.init-minus-1.temp-2.nc"
azores_pattern_minus_3_file_temp_2="${OUTPUT_DIR}/azores_${model}_${variable}_${region}_${season}_lag-${lag}_r${run}i${init}.init-minus-3.temp-2.nc"

# extra hour shift
# for init-minus-1 and init-minus-3
# for both iceland and the azores
cdo shifttime,18hour $iceland_pattern_minus_1_file $iceland_pattern_minus_1_file_temp_2
cdo shifttime,18hour $iceland_pattern_minus_3_file $iceland_pattern_minus_3_file_temp_2

# for the azores
cdo shifttime,18hour $azores_pattern_minus_1_file $azores_pattern_minus_1_file_temp_2
cdo shifttime,18hour $azores_pattern_minus_3_file $azores_pattern_minus_3_file_temp_2

# echo the times of the files
echo "times of the files - iceland"
cdo showdate $iceland_pattern_same_file
cdo showdate $iceland_pattern_minus_1_file_temp_2
cdo showdate $iceland_pattern_minus_2_file
cdo showdate $iceland_pattern_minus_3_file_temp_2

echo "times of the files - azores"
cdo showdate $azores_pattern_same_file
cdo showdate $azores_pattern_minus_1_file_temp_2
cdo showdate $azores_pattern_minus_2_file
cdo showdate $azores_pattern_minus_3_file_temp_2

# take the difference between the azores and iceland files
# for init-minus-1
cdo sub $azores_pattern_minus_1_file_temp_2 $iceland_pattern_minus_1_file_temp_2 $output_shifted_minus_1
# for init-minus-2
cdo sub $azores_pattern_minus_2_file $iceland_pattern_minus_2_file $output_shifted_minus_2
# for init-minus-3
cdo sub $azores_pattern_minus_3_file_temp_2 $iceland_pattern_minus_3_file_temp_2 $output_shifted_minus_3

# calculate the NAO for the same init files
cdo sub $azores_pattern_same_file $iceland_pattern_same_file $output_same

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

