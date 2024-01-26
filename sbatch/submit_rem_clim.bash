#!/bin/bash
#SBATCH --partition=high-mem
#SBATCH --mem=50000
#SBATCH --time=60:00
#SBATCH --array=1

# Check the number of command line arguments
if [ "$#" -ne 6 ]; then
    echo "Usage: ${variable} ${season} ${start_year} ${end_year} ${region} ${forecast_range}"
    exit
fi

# Extract the arguments
variable=$1
season=$2
start_year=$3
end_year=$4
region=$5
forecast_range=$6

module load jaspy
python /home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/remove_model_clim.py ${SLURM_ARRAY_TASK_ID} $variable $season $start_year $end_year $region $forecast_range