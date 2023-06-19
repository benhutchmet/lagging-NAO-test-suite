#!/bin/bash
#
# Calculates the NAO index for each model, run and initialisation method
# for each season and forecast range
# NAO anomaly = azores - iceland
#    
# mutli-model.calc-NAO.bash <model> <variable> <region> \
# <forecast-range> <season> <lag> <run> <init>
#
# For example: mutli-model.calc-NAO.bash HadGEM3-GC31-MM psl \
# azores 2-9 DJFM 4 1 1
#

# set the usage message
USAGE_MESSAGE="Usage: mutli-model.calc-NAO.bash <model> <variable> \
<region> <forecast-range> <season> <lag> <run> <init>"

if [ $# -ne 8 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the variables from the command line
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5
lag=$6
run=$7
init=$8

module load jaspy

# set the input and output directories
BASE_DIR="/work/scratch-nopw/benhutch/$variable/$model/$region/ \
years_${forecast_range}/$season/outputs"
INPUT_DIR="$BASE_DIR/lag_${lag}_anoms"
OUTPUT_DIR="$BASE_DIR/NAO"
# make the output directory if it doesn't exist
mkdir -p $OUTPUT_DIR