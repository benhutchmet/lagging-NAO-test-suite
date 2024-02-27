#!/bin/bash
#SBATCH --job-name=sub-rem-clim
#SBATCH --partition=high-mem
#SBATCH --mem=50000
#SBATCH --time=1000:00
#SBATCH -o /home/users/benhutch/lagging-NAO-test-suite/logs/rem-clim-%A_%a.out
#SBATCH -e /home/users/benhutch/lagging-NAO-test-suite/logs/rem-clim-%A_%a.err
#SBATCH --mail-user=benwhutchins25@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS

# Check the number of command line arguments
if [ "$#" -ne 7 ]; then
    echo "Usage: ${model} ${variable} ${season} ${start_year} ${end_year} ${region} ${forecast_range}"
    exit
fi

# Set up the process script
process_script="/home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/remove_model_clim.py"

# Extract the arguments
model=$1
variable=$2
season=$3
start_year=$4
end_year=$5
region=$6
forecast_range=$7

module load jaspy
python $process_script $model $variable $season $start_year $end_year $region $forecast_range
