#!/bin/bash
#
# multi-model.apply-lag.bash
#
# Usage: multi-model.apply-lag.bash <model> <variable> <region> <forecast-range> <season> <lag> <run> <init>
#
# Script for applying a lag to the forecast data
# lag with overlapping forecast validity periods
#
# For example: multi-model.apply-lag.bash CMCC-CM2-SR5 psl north-atlantic 2-9 DJFM 4 1 1
#

# Set the usage message
USAGE_MESSAGE="Usage: multi-model.apply-lag.bash <model> <variable> <region> <forecast-range> <season> <lag> <run> <init>"

# Check that the correct number of arguments have been passed
if [ $# -ne 8 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# Extract the model, variable, region, forecast range, season and lag
# From the command line arguments
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5
lag=$6
run=$7
init=$8

# load cdo
module load jaspy

# Set up the input directory
# We want the dir which contains the anoms data
INPUT_DIR="/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/anoms"

# Files will have filename like this:
#years-2-9-DJFM-north-atlantic-psl_Amon_CMCC-CM2-SR5_dcppA-hindcast_s2019-r9i1p1f1_gn_201911-202912-anoms.nc
fnames="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r${run}i${init}*anoms.nc"

# Set up the files
files="${INPUT_DIR}/${fnames}"

# Echo the files being used
echo "Files being processed: $files"

# Exit the script if the files don't exist
if [ ! -f $files ]; then
    echo "ERROR: files not found"
    exit 1
fi

# Set up the output directory
OUTPUT_DIR="/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/lag_${lag}_anoms"
TEMP_DIR="/work/scratch-nopw/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs/tmp"

# Create the output directory if it doesn't exist
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

# loop through the files
# and process them
for file in $files; do

    # Extract the initialization year from the input file name
    year=$(basename "$file" | sed 's/.*_s\([0-9]\{4\}\)-.*/\1/')
    echo "[INFO] year: $year"

    # if year is not four numbers
    # then exit the script
    if [[ ! $year =~ ^[0-9]{4}$ ]]; then
        echo "[ERROR] year is not four numbers"
        exit 1
    fi

    # if the value of year is less than 1955 or greater than 2025
    # return error and exit the script
    if [ $year -lt 1955 ] || [ $year -gt 2025 ]; then
        echo "[ERROR] year is out of range"
        exit 1
    fi

    # specify the start and end years
    # for the lagged forecast dates
    # for initialization of the same year
    # years 2-9
    start_date=$((year + 1))"-12-01"
    end_date_same_init=$((year + 9))"-03-31"
    # end date for initialization year-1
    end_date_init_minus_1=$((year + 8))"-12-31"
    # end date for initialization year-2
    end_date_init_minus_2=$((year + 7))"-12-31"
    # end date for initialization year-3
    end_date_init_minus_3=$((year + 6))"-12-31"

    # Echo these
    # For same year init
    echo "[INFO] same year initialization start date: $start_date, end date: $end_date_same_init"
    # For init year - 1
    echo "[INFO] init year - 1 start date: $start_date, end date: $end_date_init_minus_1"
    # For init year - 2
    echo "[INFO] init year - 2 start date: $start_date, end date: $end_date_init_minus_2"
    # For init year - 3
    echo "[INFO] init year - 3 start date: $start_date, end date: $end_date_init_minus_3"

    # Set up the temp file names
    # for the different initialization years
    # for the same year initialization
    temp_fname_same_init"(basename "$file" | sed 's/\(.*\)_s\([0-9]\{4\}\)-.*/\1_s\2-${start_date}-${end_date_same_init}-anoms-same-init.nc/')"
    echo "[INFO] temp_fname_same_init: $temp_fname_same_init"
    # construct the full path
    temp_file_same_init="${TEMP_DIR}/${temp_fname_same_init}"

    # for initialization year - 1
    temp_fname_init_minus_1"(basename "$file" | sed 's/\(.*\)_s\([0-9]\{4\}\)-.*/\1_s\2-${start_date}-${end_date_init_minus_1}-anoms.init-minus-1.nc/')"
    echo "[INFO] temp_fname_init_minus_1: $temp_fname_init_minus_1"
    # construct the full path
    temp_file_init_minus_1="${TEMP_DIR}/${temp_fname_init_minus_1}"

    # for initialization year - 2
    temp_fname_init_minus_2"(basename "$file" | sed 's/\(.*\)_s\([0-9]\{4\}\)-.*/\1_s\2-${start_date}-${end_date_init_minus_2}-anoms.init-minus-2.nc/')"
    echo "[INFO] temp_fname_init_minus_2: $temp_fname_init_minus_2"
    # construct the full path
    temp_file_init_minus_2="${TEMP_DIR}/${temp_fname_init_minus_2}"

    # for initialization year - 3
    temp_fname_init_minus_3"(basename "$file" | sed 's/\(.*\)_s\([0-9]\{4\}\)-.*/\1_s\2-${start_date}-${end_date_init_minus_3}-anoms.init-minus-3.nc/')"
    echo "[INFO] temp_fname_init_minus_3: $temp_fname_init_minus_3"
    # construct the full path
    temp_file_init_minus_3="${TEMP_DIR}/${temp_fname_init_minus_3}"

    # process the same init file
    # extract the dates
    # and save to a temp file
    cdo timmean -select,startdate=$start_date,enddate=$end_date_same_init $file $temp_file_same_init

    # find the init year - 1 file
    # within the same group of files
    # and process it
    # extract the dates
    # and save to a temp file
    file_init_minus_1=$(echo "${files[@]}" | tr ' ' '\n' | grep "$(basename "$file" | sed -E 's/.*_s([0-9]{4})-.*/\1/')-1" | head -n 1)

    # if the file doesn't exist
    # then exit the script
    if [ ! -f $file_init_minus_1 ]; then
        echo "[ERROR] file_init_minus_1 not found"
        # Echo the year
        echo "[INFO] File not found for year: $year"
        exit 1
    # otherwise print the file name
    else
        echo "[INFO] file_init_minus_1: $file_init_minus_1"
    fi

    # extract the dates
    # and save to a temp file
    cdo timmean -select,startdate=$start_date,enddate=$end_date_init_minus_1 $file_init_minus_1 $temp_file_init_minus_1

    # find the init year - 2 file
    # within the same group of files
    # and process it
    # extract the dates
    # and save to a temp file
    file_init_minus_2=$(echo "${files[@]}" | tr ' ' '\n' | grep "$(basename "$file" | sed -E 's/.*_s([0-9]{4})-.*/\1/')-2" | head -n 1)

    # if the file doesn't exist
    # then exit the script
    if [ ! -f $file_init_minus_2 ]; then
        echo "[ERROR] file_init_minus_2 not found"
        # Echo the year
        echo "[INFO] File not found for year: $year"
        exit 1
    # otherwise print the file name
    else
        echo "[INFO] file_init_minus_2: $file_init_minus_2"
    fi

    # extract the dates
    # and save to a temp file
    cdo timmean -select,startdate=$start_date,enddate=$end_date_init_minus_2 $file_init_minus_2 $temp_file_init_minus_2

    # find the init year - 3 file
    # within the same group of files
    # and process it
    # extract the dates
    # and save to a temp file
    file_init_minus_3=$(echo "${files[@]}" | tr ' ' '\n' | grep "$(basename "$file" | sed -E 's/.*_s([0-9]{4})-.*/\1/')-3" | head -n 1)

    # if the file doesn't exist
    # then exit the script
    if [ ! -f $file_init_minus_3 ]; then
        echo "[ERROR] file_init_minus_3 not found"
        # Echo the year
        echo "[INFO] File not found for year: $year"
        exit 1
    # otherwise print the file name
    else
        echo "[INFO] file_init_minus_3: $file_init_minus_3"
    fi

    # extract the dates
    # and save to a temp file
    cdo timmean -select,startdate=$start_date,enddate=$end_date_init_minus_3 $file_init_minus_3 $temp_file_init_minus_3

done

# Set up the files to be merged
# for the same year initialization
# these are all the files
# which contain same init string
files_same_init_to_merge=$(ls ${TEMP_DIR}/*anoms-same-init.nc)

# if the files don't exist
# then exit the script
if [ ! -f $files_same_init_to_merge ]; then
    echo "[ERROR] files_same_init_to_merge not found"
    exit 1
# otherwise print the file name
else
    echo "[INFO] files_same_init_to_merge: $files_same_init_to_merge"
fi

# Set up the files to be merged
# for the init year - 1
# these are all the files
# which contain init year - 1 string
files_init_minus_1_to_merge=$(ls ${TEMP_DIR}/*anoms.init-minus-1.nc)

# if the files don't exist
# then exit the script
if [ ! -f $files_init_minus_1_to_merge ]; then
    echo "[ERROR] files_init_minus_1_to_merge not found"
    exit 1
# otherwise print the file name
else
    echo "[INFO] files_init_minus_1_to_merge: $files_init_minus_1_to_merge"
fi

# Set up the files to be merged
# for the init year - 2
# these are all the files
# which contain init year - 2 string
files_init_minus_2_to_merge=$(ls ${TEMP_DIR}/*anoms.init-minus-2.nc)

# if the files don't exist
# then exit the script
if [ ! -f $files_init_minus_2_to_merge ]; then
    echo "[ERROR] files_init_minus_2_to_merge not found"
    exit 1
# otherwise print the file name
else
    echo "[INFO] files_init_minus_2_to_merge: $files_init_minus_2_to_merge"
fi

# Set up the files to be merged
# for the init year - 3
# these are all the files
# which contain init year - 3 string
files_init_minus_3_to_merge=$(ls ${TEMP_DIR}/*anoms.init-minus-3.nc)

# if the files don't exist
# then exit the script
if [ ! -f $files_init_minus_3_to_merge ]; then
    echo "[ERROR] files_init_minus_3_to_merge not found"
    exit 1
# otherwise print the file name
else
    echo "[INFO] files_init_minus_3_to_merge: $files_init_minus_3_to_merge"
fi


# extract the largest value of year from the files
# and save to a variable
# this is the last year of the forecast
# for same init files to merge
last_year_same_init=$(echo "${files_same_init_to_merge[@]}" | tr ' ' '\n' | sed -E 's/.*_s[0-9]{4}-r[0-9]{3}i[0-9]{3}p[0-9]{3}f([0-9]{4})-t[0-9]{3}.*.nc/\1/' | sort -n | tail -n 1)

# if the variable is empty
# then exit the script
if [ -z $last_year_same_init ]; then
    echo "[ERROR] last_year_same_init not found"
    exit 1
# otherwise print the file name
else
    echo "[INFO] last_year_same_init: $last_year_same_init"
fi

# extract the first value of year from the files
first_year_same_init=$(echo "${files_same_init_to_merge[@]}" | tr ' ' '\n' | sed -E 's/.*_s[0-9]{4}-r[0-9]{3}i[0-9]{3}p[0-9]{3}f([0-9]{4})-t[0-9]{3}.*.nc/\1/' | sort -n | head -n 1)

# if the variable is empty
# then exit the script
if [ -z $first_year_same_init ]; then
    echo "[ERROR] first_year_same_init not found"
    exit 1
# otherwise print the file name
else
    echo "[INFO] first_year_same_init: $first_year_same_init"
fi

# extract the largest value of year from the files
last_year_init_minus_1=$(echo "${files_init_minus_1_to_merge[@]}" | tr ' ' '\n' | sed -E 's/.*_s[0-9]{4}-r[0-9]{3}i[0-9]{3}p[0-9]{3}f([0-9]{4})-t[0-9]{3}.*.nc/\1/' | sort -n | tail -n 1)
last_year_init_minus_2=$(echo "${files_init_minus_2_to_merge[@]}" | tr ' ' '\n' | sed -E 's/.*_s[0-9]{4}-r[0-9]{3}i[0-9]{3}p[0-9]{3}f([0-9]{4})-t[0-9]{3}.*.nc/\1/' | sort -n | tail -n 1)
last_year_init_minus_3=$(echo "${files_init_minus_3_to_merge[@]}" | tr ' ' '\n' | sed -E 's/.*_s[0-9]{4}-r[0-9]{3}i[0-9]{3}p[0-9]{3}f([0-9]{4})-t[0-9]{3}.*.nc/\1/' | sort -n | tail -n 1)

# extract the first value of year from the files
first_year_init_minus_1=$(echo "${files_init_minus_1_to_merge[@]}" | tr ' ' '\n' | sed -E 's/.*_s[0-9]{4}-r[0-9]{3}i[0-9]{3}p[0-9]{3}f([0-9]{4})-t[0-9]{3}.*.nc/\1/' | sort -n | head -n 1)
first_year_init_minus_2=$(echo "${files_init_minus_2_to_merge[@]}" | tr ' ' '\n' | sed -E 's/.*_s[0-9]{4}-r[0-9]{3}i[0-9]{3}p[0-9]{3}f([0-9]{4})-t[0-9]{3}.*.nc/\1/' | sort -n | head -n 1)
first_year_init_minus_3=$(echo "${files_init_minus_3_to_merge[@]}" | tr ' ' '\n' | sed -E 's/.*_s[0-9]{4}-r[0-9]{3}i[0-9]{3}p[0-9]{3}f([0-9]{4})-t[0-9]{3}.*.nc/\1/' | sort -n | head -n 1)

# Set up the output file names
# for the same year initialization
output_fname_same_init="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s${first_year_same_init}-e${last_year_same_init}_anoms.init-same.nc"
output_fname_init_minus_1="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s${first_year_init_minus_1}-e${last_year_init_minus_1}_anoms.init-minus-1.nc"
output_fname_init_minus_2="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s${first_year_init_minus_2}-e${last_year_init_minus_2}_anoms.init-minus-2.nc"
output_fname_init_minus_3="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s${first_year_init_minus_3}-e${last_year_init_minus_3}_anoms.init-minus-3.nc"

# set up the full output file paths
output_file_same_init="${output_dir}/${output_fname_same_init}"
output_file_init_minus_1="${output_dir}/${output_fname_init_minus_1}"
output_file_init_minus_2="${output_dir}/${output_fname_init_minus_2}"
output_file_init_minus_3="${output_dir}/${output_fname_init_minus_3}"

# Merge the files
# by the time axis
# for the same year initialization
cdo mergetime $files_same_init_to_merge $output_file_same_init

# Merge the files
# by the time axis
# for the init minus 1 year
cdo mergetime $files_init_minus_1_to_merge $output_file_init_minus_1

# Merge the files
# by the time axis
# for the init minus 2 year
cdo mergetime $files_init_minus_2_to_merge $output_file_init_minus_2

# Merge the files
# by the time axis
# for the init minus 3 year
cdo mergetime $files_init_minus_3_to_merge $output_file_init_minus_3

# Remove the temporary files
rm $tmp_dir/*.nc
