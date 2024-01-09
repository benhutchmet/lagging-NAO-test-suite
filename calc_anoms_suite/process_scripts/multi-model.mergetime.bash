#!/bin/bash
#
# multi-model.mergetime.bash
#
# Script to merge the time dimension of multiple files
# For each individual ensemble member (~150 ish)
#
# For example: multi-model.mergetime.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF 1 1 92500
#

USAGE_MESSAGE="Usage: multi-model.mergetime.bash <model> <variable> <season> <run> <init_scheme>"

# check that the correct number of arguments have been passed
if [ $# -ne 5 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the model, variable, region, forecast range and season
model=$1
variable=$2
season=$3
run=$4
init_scheme=$5

# make sure that cdo is loaded
module load jaspy

if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    # anoms directory from which to extract the files
    # base_dir="/work/scratch-nopw2/benhutch/$variable/$model/$region/years_${forecast_range}/$season/plev_${pressure_level}/outputs/anoms"
    echo "ERROR: pressure level not set up for ua and va"
    exit 1
else
    # anoms directory from which to extract the files
    # example dir: /work/scratch-nopw2/benhutch/psl/BCC-CSM2-MR/global/all_forecast_years/DJFM/outputs/anoms
    base_dir="/work/scratch-nopw2/benhutch/$variable/$model/global/all_forecast_years/$season/outputs/anoms"
fi    

# If the variable is ua or va
# Then the files are in the format:
# mean-years-${forecast_range}-${season}-${region}-plev-${variable}_?mon_${model}_dcppA-hindcast_s????-r${run}i${init_scheme}*-anoms.nc
if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    # file pattern of the anoms files
    files_pattern="mean-years-${forecast_range}-${season}-${region}-plev-${variable}_?mon_${model}_dcppA-hindcast_s????-r${run}i${init_scheme}*-anoms.nc"
else
    # file pattern of the anoms files
    # EXample: all-years-DJFM-global-psl_Amon_BCC-CSM2-MR_dcppA-hindcast_s1975-r8i1p1f1_gn_197501-198412-anoms.nc
    files_pattern="all-years-${season}-global-${variable}_?mon_${model}_dcppA-hindcast_s????-r${run}i${init_scheme}*-anoms.nc"
fi

# set up the files
# combine the base directory and the file pattern
files="${base_dir}/${files_pattern}"

# If there are no files
# Then exit with an error
if [ ! -f $files ]; then
    echo "ERROR: no files found: $files"
    exit 1
fi

# echo the files to be merged
echo "Files to be merged: $files"

# Check that the base directory is not empty
if [ -z "$(ls -A $base_dir)" ]; then
    echo "ERROR: base directory is empty: $base_dir"
    exit 1
fi

if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    # Set the output directory with the pressure level
    # OUTPUT_DIR="/home/users/benhutch/skill-maps-processed-data/${variable}/${model}/${region}/years_${forecast_range}/${season}/plev_${pressure_level}/outputs/mergetime"
    echo "ERROR: pressure level not set up for ua and va"
    exit 1
else
    # set the output directory
    # send to the home directory
    OUTPUT_DIR="/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/${variable}/${model}/global/all_forecast_years/${season}/outputs/mergetime"
fi

mkdir -p $OUTPUT_DIR

# set the output file
mergetime_fname="mergetime_${model}_${variable}_global_all_forecast_years_${season}-r${run}i${init_scheme}.nc"
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
