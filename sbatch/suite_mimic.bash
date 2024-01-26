#!/bin/bash

# Set up the usage message
usg_msg="Usage: bash submit_suite_mimic.sh <variable> <season> <start_year> <end_year> <region> <forecast_range> <lag>"

# Echo the number of command line arguments
echo "Number of command line arguments: $#"

# Check the number of command line arguments
if [ "$#" -ne 7 ]; then
    echo $usg_msg
    exit
fi

# Extract the arguments
variable=$1
season=$2
start_year=$3
end_year=$4
region=$5
forecast_range=$6
lag=$7

# Echo the CLIs
echo "variable: $variable"
echo "season: $season"
echo "start_year: $start_year"
echo "end_year: $end_year"
echo "region: $region"
echo "forecast_range: $forecast_range"
echo "lag: $lag"

# Set up the process scripts
rem_clim_script="/home/users/benhutch/lagging-NAO-test-suite/sbatch/submit_rem_clim.bash"

# Set up the alt lag script
alt_lag_script="/home/users/benhutch/lagging-NAO-test-suite/sbatch/submit_alt_lag.bash"

# Set off the first job
jid1=$(sbatch $rem_clim_script $variable $season $start_year $end_year $region $forecast_range)
# Trim the output from Submitted batch job 12345678 to 12345678
jid1=$(echo $jid1 | tr -dc '0-9')

# Set off the second job
# After the first job is done
jid2=$(sbatch --dependency=afterok:$jid1 $alt_lag_script $variable $season $region $start_year $end_year $forecast_range $lag)
# Trim the output from Submitted batch job 12345678 to 12345678
jid2=$(echo $jid2 | tr -dc '0-9')

# Once the second job is done
# Echo that the script is done
echo "Done submitting jobs"