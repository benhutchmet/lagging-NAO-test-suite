#!/bin/bash
#SBATCH --job-name=sub-rem-clim
#SBATCH --partition=high-mem
#SBATCH --mem=50000
#SBATCH --time=200:00
#SBATCH --array=3
#SBATCH -o /home/users/benhutch/lagging-NAO-test-suite/logs/rem-clim-%A_%a.out
#SBATCH -e /home/users/benhutch/lagging-NAO-test-suite/logs/rem-clim-%A_%a.err
#SBATCH --mail-user=benwhutchins25@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS

# Check the number of command line arguments
if [ "$#" -ne 6 ]; then
    echo "Usage: ${variable} ${season} ${start_year} ${end_year} ${region} ${forecast_range}"
    exit
fi

# Set up the process script
process_script="/home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/remove_model_clim.py"

# Extract the arguments
variable=$1
season=$2
start_year=$3
end_year=$4
region=$5
forecast_range=$6

module load jaspy
python $process_script ${SLURM_ARRAY_TASK_ID} $variable $season $start_year $end_year $region $forecast_range