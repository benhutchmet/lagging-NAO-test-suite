#!/bin/bash
#
# multi-model.mergetime.bash
#
# Script to merge the time dimension of multiple files
# For each individual ensemble member (~150 ish)
#
# For example: multi-model.mergetime.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF 1 1 92500
#

USAGE_MESSAGE="Usage: multi-model.mergetime.bash <model> <variable> <region> <forecast-range> <season> <run> <init_scheme> <pressure-level>"

# check that the correct number of arguments have been passed
if [ $# -ne 8 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the model, variable, region, forecast range and season
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5

# extract the run and init_scheme
run=$6
init_scheme=$7
pressure_level=$8

# make sure that cdo is loaded
module load jaspy

if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    # anoms directory from which to extract the files
    base_dir="/work/scratch-nopw2/benhutch/$variable/$model/$region/years_${forecast_range}/$season/plev_${pressure_level}/outputs/anoms"
else
    # anoms directory from which to extract the files
    base_dir="/work/scratch-nopw2/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/anoms"
fi    

# If the variable is ua or va
# Then the files are in the format:
# mean-years-${forecast_range}-${season}-${region}-plev-${variable}_?mon_${model}_dcppA-hindcast_s????-r${run}i${init_scheme}*-anoms.nc
if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    # file pattern of the anoms files
    files_pattern="mean-years-${forecast_range}-${season}-${region}-plev-${variable}_?mon_${model}_dcppA-hindcast_s????-r${run}i${init_scheme}*-anoms.nc"
else
    # file pattern of the anoms files
    files_pattern="mean-years-${forecast_range}-${season}-${region}-${variable}_?mon_${model}_dcppA-hindcast_s????-r${run}i${init_scheme}*-anoms.nc"
fi

# set up the files
# combine the base directory and the file pattern
files="${base_dir}/${files_pattern}"

# echo the files to be merged
echo "Files to be merged: $files"

# Check that the base directory is not empty
if [ -z "$(ls -A $base_dir)" ]; then
    echo "ERROR: base directory is empty: $base_dir"
    exit 1
fi

if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    # Set the output directory with the pressure level
    OUTPUT_DIR="/home/users/benhutch/skill-maps-processed-data/${variable}/${model}/${region}/years_${forecast_range}/${season}/plev_${pressure_level}/outputs/mergetime"
else
    # set the output directory
    # send to the home directory
    OUTPUT_DIR="/home/users/benhutch/skill-maps-processed-data/${variable}/${model}/${region}/years_${forecast_range}/${season}/outputs/mergetime"
fi

mkdir -p $OUTPUT_DIR

# set the output file
mergetime_fname="mergetime_${model}_${variable}_${region}_${forecast_range}_${season}-r${run}i${init_scheme}.nc"
OUTPUT_FILE=${OUTPUT_DIR}/${mergetime_fname}

# echo the output file
echo "Output file: $OUTPUT_FILE"

# Check that the output file does not already exist
# If it does, then delete it
if [ -f $OUTPUT_FILE ]; then
    echo "WARNING: output file already exists"
    echo "WARNING: deleting existing output file"
    rm -f $OUTPUT_FILE
fi

# merge the files
cdo mergetime $files $OUTPUT_FILE
