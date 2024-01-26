#!/bin/bash
#SBATCH --job-name=sub-alt-lag
#SBATCH --partition=high-mem
#SBATCH --mem=50000
#SBATCH --time=150:00
#SBATCH -o /home/users/benhutch/lagging-NAO-test-suite/logs/alt-lag-%A_%a.out
#SBATCH -e /home/users/benhutch/lagging-NAO-test-suite/logs/alt-lag-%A_%a.err
#SBATCH --mail-user=benwhutchins25@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

# Set up the usage message
USAGE_MSG="Usage: ${variable} ${season} ${region} ${start_year} ${end_year} ${forecast_range} ${lag}"

# Print the number of command line arguments
echo "Number of command line arguments: $#"

# Check the number of command line arguments
if [ "$#" -ne 7 ]; then
    echo $USAGE_MSG
    exit
fi

process_script="/home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/alternate_lag_functions.py"

# Extract the arguments
variable=$1
season=$2
region=$3
start_year=$4
end_year=$5
forecast_range=$6
lag=$7

module load jaspy

# Echo the CLIs
echo "variable: $variable"
echo "season: $season"
echo "region: $region"
echo "start_year: $start_year"
echo "end_year: $end_year"
echo "forecast_range: $forecast_range"
echo "lag: $lag"

# Run the script
python $process_script $variable $season $region $start_year $end_year $forecast_range $lag