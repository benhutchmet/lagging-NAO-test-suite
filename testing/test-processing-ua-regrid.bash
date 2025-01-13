#!/bin/bash
#SBATCH --job-name=submit_test_processing_ua
#SBATCH --partition=high-mem
#SBATCH --mem=200000
#SBATCH --time=1400:00
#SBATCH -o /home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/logs/submit_test_processing_ua-%A_%a.out
#SBATCH -e /home/users/benhutch/lagging-NAO-test-suite/alternate_lag_suite/logs/submit_test_processing_ua-%A_%a.err

# Print the CLI arguments
echo "CLI arguments are: $@"
echo "Number of CLI arguments is: $#"
echo "Desired no. of arguments is: 7"

# Check if the correct number of arguments were passed
if [ $# -ne 7 ]; then
    echo "Usage: sbatch submit_test_processing_ua <model> <variable> <season> <experiment> <region> <start_year> <end_year>"
    echo "Example: sbatch submit_test_processing_ua HadGEM3-GC31-MM ua DJFM dcppA-hindcast global 1960 2018"
    exit 1
fi

# Extract the model, variable, region, forecast range and season
model=$1
variable=$2
season=$3
experiment=$4
region=$5
start_year=$6
end_year=$7
test_init_year=1961

# Print the model, variable, region, forecast range and season
echo "Model is: $model"
echo "Variable is: $variable"
echo "Season is: $season"
echo "Experiment is: $experiment"
echo "Region is: $region"
echo "Start year is: $start_year"
echo "End year is: $end_year"

module load jaspy
source activate bens-conda-env2

# Set up the process script
process_script="/home/users/benhutch/lagging-NAO-test-suite/testing/sel_reg_fcst_seasons_functions.py"

# Run the script
python ${process_script} $model $variable $season $experiment $region ${start_year} ${end_year} ${test_init_year}

# End of file
echo "End of file"