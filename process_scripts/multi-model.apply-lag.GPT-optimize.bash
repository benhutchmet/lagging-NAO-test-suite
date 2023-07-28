#!/bin/bash

# set debug mode
#set -x

USAGE_MESSAGE="Usage: multi-model.apply-lag.bash <model> <variable> <region> <forecast-range> <season> <lag> <run> <init>"

# Function to extract the year from the file name
extract_year() {
    basename "$1" | sed 's/.*_s\([0-9]\{4\}\)-.*/\1/'
}

# Function to extract the required dates and save to a temp file
process_file() {
    local start_date="$1"
    local end_date="$2"
    local infile="$3"
    local outfile="$4"
    cdo timmean -select,startdate="$start_date",enddate="$end_date" "$infile" "$outfile"
}

if [ $# -ne 8 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

model=$1
variable=$2
region=$3
forecast_range=$4
season=$5
lag=$6
run=$7
init=$8

module load jaspy

BASE_DIR="/work/scratch-nopw2/benhutch/$variable/$model/$region/years_${forecast_range}/$season/outputs"
INPUT_DIR="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/outputs/anoms"
OUTPUT_DIR="${BASE_DIR}/lag_${lag}_anoms"
TEMP_DIR="${BASE_DIR}/tmp"
mkdir -p "$OUTPUT_DIR" "$TEMP_DIR"

pattern="mean-years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_s????-r${run}i${init}*anoms.nc"
files=($INPUT_DIR/$pattern)

echo "Files being processed: ${files[*]}"

if [ ! -f "${files[0]}" ]; then
    echo "ERROR: files not found"
    exit 1
fi

# ... previous part of the script

for file in "${files[@]}"; do
    year=$(extract_year "$file")

    if [[ ! $year =~ ^[0-9]{4}$ ]] || [ "$year" -lt 1955 ] || [ "$year" -gt 2025 ]; then
        echo "[ERROR] Year $year is invalid."
        exit 1
    fi

    # echo the file we are processing
    echo "Processing file: $file"

    start_date=$((year + 1))-12-01
    for ((i = 0; i <= 3; i++)); do

        # The end date is dependent on which init - i we are processing
        # First echo the init - i we are processing
        echo "[INFO] Currently processing file: init - $i"

        # For init - 0 we shift the end date back by 3 years
        # For init - 1 we shift the end date back by 2 years
        # For init - 2 we shift the end date back by 1 year
        # For init - 3 we do not shift the end date
        # Format this using an if statement
        if [ "$i" -eq 0 ]; then
            end_date=$((year + 9 - i - 3))-03-31
        elif [ "$i" -eq 1 ]; then
            end_date=$((year + 9 - i - 2))-03-31
        elif [ "$i" -eq 2 ]; then
            end_date=$((year + 9 - i - 1))-03-31
        elif [ "$i" -eq 3 ]; then
            end_date=$((year + 9 - i))-03-31
        fi
    
        # Echo the init - i case and the end date
        echo "[INFO] init - $i case, end date: $end_date for start date: $start_date"
        echo "[INFO] init year - $i start date: $start_date, end date: $end_date"

        temp_fname=$(basename "$file" | sed "s/\(.*\)_s\([0-9]\{4\}\)-.*/\1_s\2-${start_date}-${end_date}-r${run}-i${init}-anoms.init-minus-$i.nc/")
        temp_file="${TEMP_DIR}/${temp_fname}"

        # Echo the file path
        echo "[INFO] Base initialization file path: $file"

        # Modify this part to search for year - 1
        echo "[INFO] Base initialization file year: $year"
        # pattern year = year - i
        pattern_year=$((year - i))

        # print out pattern being used to search for target file
        echo "Searching for initialization year: $pattern_year"

        # search for target file
        if ! target_file=$(echo "${files[@]}" | tr ' ' '\n' | grep -E "_s${pattern_year}" | head -n 1 2> /tmp/grep_error); then
            grep_error=$(cat /tmp/grep_error)
            echo "[ERROR] Failed to search for target file: $grep_error"
            exit 1
        fi

        # print out target file
        echo "Target file: $target_file"

        process_file "$start_date" "$end_date" "$target_file" "$temp_file"
    done
done

# SOme issue in this bit now

# Move the merge steps to outside the loop
# Set up the files to be merged
files_same_init_to_merge=($TEMP_DIR/*-anoms.init-minus-0.nc)
files_init_minus_1_to_merge=($TEMP_DIR/*-anoms.init-minus-1.nc)
files_init_minus_2_to_merge=($TEMP_DIR/*-anoms.init-minus-2.nc)
files_init_minus_3_to_merge=($TEMP_DIR/*-anoms.init-minus-3.nc)

# echo the files we are trying to find
echo "Files to merge same init: ${files_same_init_to_merge[*]}"
echo "Files to merge init - 1: ${files_init_minus_1_to_merge[*]}"
echo "Files to merge init - 2: ${files_init_minus_2_to_merge[*]}"
echo "Files to merge init - 3: ${files_init_minus_3_to_merge[*]}"

# if any of these files are missing, exit with an error
if [ ! -f "${files_same_init_to_merge[0]}" ] || \
    [ ! -f "${files_init_minus_1_to_merge[0]}" ] || \
    [ ! -f "${files_init_minus_2_to_merge[0]}" ] || \
    [ ! -f "${files_init_minus_3_to_merge[0]}" ]; then
     echo "[ERROR] One or more files are missing."
     exit 1
    fi

# Create the output file names
output_fname_same_init="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.same-init.nc"
output_fname_init_minus_1="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.init-minus-1.nc"
output_fname_init_minus_2="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.init-minus-2.nc"
output_fname_init_minus_3="years-${forecast_range}-${season}-${region}-${variable}_Amon_${model}_dcppA-hindcast_anoms.r${run}i${init}.init-minus-3.nc"

# Set up the output file paths
output_file_same_init="${OUTPUT_DIR}/${output_fname_same_init}"
output_file_init_minus_1="${OUTPUT_DIR}/${output_fname_init_minus_1}"
output_file_init_minus_2="${OUTPUT_DIR}/${output_fname_init_minus_2}"
output_file_init_minus_3="${OUTPUT_DIR}/${output_fname_init_minus_3}"

# if the output files already exist, echo that this will be overwritten
# and remove the existing files
if [ -f "$output_file_same_init" ]; then
    echo "[WARNING] Output file already exists: $output_file_same_init"
    rm -f "$output_file_same_init"
    echo "[INFO] Removed existing file."
fi

if [ -f "$output_file_init_minus_1" ]; then
    echo "[WARNING] Output file already exists: $output_file_init_minus_1"
    rm -f "$output_file_init_minus_1"
    echo "[INFO] Removed existing file."
fi

if [ -f "$output_file_init_minus_2" ]; then
    echo "[WARNING] Output file already exists: $output_file_init_minus_2"
    rm -f "$output_file_init_minus_2"
    echo "[INFO] Removed existing file."
fi

if [ -f "$output_file_init_minus_3" ]; then
    echo "[WARNING] Output file already exists: $output_file_init_minus_3"
    rm -f "$output_file_init_minus_3"
    echo "[INFO] Removed existing file."
fi

# merge the files
cdo mergetime "${files_same_init_to_merge[@]}" "$output_file_same_init"
cdo mergetime "${files_init_minus_1_to_merge[@]}" "$output_file_init_minus_1"
cdo mergetime "${files_init_minus_2_to_merge[@]}" "$output_file_init_minus_2"
cdo mergetime "${files_init_minus_3_to_merge[@]}" "$output_file_init_minus_3"

# Remove the temporary files
#rm -f "${files_same_init_to_merge[@]}" "${files_init_minus_1_to_merge[@]}" "${files_init_minus_2_to_merge[@]}" "${files_init_minus_3_to_merge[@]}"

echo "[INFO] Script completed successfully."

exit 0
