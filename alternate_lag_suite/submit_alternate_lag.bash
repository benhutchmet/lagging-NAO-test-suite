#!/bin/bash
#SBATCH --job-name=sub-alt-lag
#SBATCH --partition=high-mem
#SBATCH --mem=300000
#SBATCH --time=1500:00
#SBATCH -o /home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/logs/sub-alt-lag-%A_%a.out
#SBATCH -e /home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/logs/sub-alt-lag-%A_%a.err
# sbatch ~/alternate_lag_suite/alternate_lag_functions.py tas DJFM global 1961 2014 2-5 4 False False 20 100000
# sbatch ~/alternate_lag_suite/alternate_lag_functions.py tas DJFM global 1961 2014 2-5 4 False False 20 100000
# sbatch ~/alternate_lag_suite/alternate_lag_functions.py tas DJFM global 1961 2014 2-5 4 False False 20 100000

# Set up the usage message
usage_msg = "Usage: sbatch submit_alternate_lag.bash  <variable> <season> <region> <start_year> <end_year> <forecast_range> <lag> <nao_matching> <plot> <nao_matched_members> <level>"

# Check the number of arguments
if [ "$#" -ne 11 ]; then
    echo "Illegal number of parameters"
    echo $usage_msg
    exit 1
fi

# Set up the CLI args
variable=$1
season=$2
region=$3
start_year=$4
end_year=$5
forecast_range=$6
lag=$7
nao_matching=$8
plot=$9
nao_matched_members=${10}
level=${11}

# Load the required modules
module load jaspy

# set model fcst year as 1
export model_fcst_year=1

# Echo the CLI args
echo "variable: ${variable}"
echo "season: ${season}"
echo "region: ${region}"
echo "start_year: ${start_year}"
echo "end_year: ${end_year}"
echo "forecast_range: ${forecast_range}"
echo "lag: ${lag}"
echo "nao_matching: ${nao_matching}"
echo "plot: ${plot}"
echo "nao_matched_members: ${nao_matched_members}"
echo "level: ${level}"

# Set up the process script
process_script="/home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/alternate_lag_functions.py"

# Run the process script
python ${process_script} \
    ${variable} \
    ${season} \
    ${region} \
    ${start_year} \
    ${end_year} \
    ${forecast_range} \
    ${lag} \
    ${nao_matching} \
    ${plot} \
    ${nao_matched_members} \
    ${level}