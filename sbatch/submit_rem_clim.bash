#!/bin/bash
#SBATCH --partition=high-mem
#SBATCH --mem=50000
#SBATCH --time=60:00
#SBATCH --array=1-12
# TODO: Submit this as an array job for the 12 different models

# Check the number of command line arguments
if [ "$#" -ne 7 ]; then
    echo "Usage: ${model} ${variable} ${season} ${start_year} ${end_year} ${region} ${forecast_range}"
    exit
fi

# Extract the arguments
model=$1
variable=$2
season=$3
start_year=$4
end_year=$5
region=$6
forecast_range=$7

module load jaspy
python /home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/remove_model_clim.py $model $variable $season $start_year $end_year $region $forecast_range