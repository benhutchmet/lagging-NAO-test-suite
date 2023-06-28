#!/bin/bash
#
# submit-all-multi-model.calc-NAO.bash
#
# Submit script for applying the calc-NAO.py script to all models
#
# Usage:
# bash submit-all-multi-model.calc-NAO.bash <model> <variable> <forecast-range> <season> <lag>
#
# E.g.:
# bash submit-all-multi-model.calc-NAO.bash BCC-CSM2-MR psl 2-9 DJFM 4
#

# Import the models list
source $PWD/dictionaries.bash
# If dictionaries is not found then exit
if [ $? -ne 0 ]; then
    echo "ERROR: dictionaries.bash not found!"
    exit 1
fi

# Set the usage message
USAGE_MESSAGE="Usage: bash submit-all-multi-model.calc-NAO.bash <model> <variable> <forecast-range> <season> <lag>"

# Check the number of arguments
if [ $# -ne 5 ]; then
    echo "ERROR: Incorrect number of arguments!"
    echo "$USAGE_MESSAGE"
    exit 1
fi

# Extract the arguments from the command line
model=$1
variable=$2
forecast_range=$3
season=$4
lag=$5

# Set up the extractor script to be used
EXTRACTOR=$PWD/process_scripts/multi-model.calc-NAO.bash

# Check that CDO is loaded
module load jaspy

# If model is all then loop over all models
if [ $model == "all" ]; then

# set up the model list
echo "Extracting data for all of the models: ${models}"

    # Loop over all models
    for model in "${MODELS[@]}"; do

    # Echo the model being processed
    echo "Processing ${model}"

    # Case statement to set the number of ensemble members and inits
    case $model in
        BCC-CSM2-MR)
            run=8
            init_methods=1
            ;;
        MPI-ESM1-2-HR)
            run=10
            init_methods=1
            ;;
        CanESM5)
            run=20
            init_methods=1
            ;;
        CMCC-CM2-SR5)
            run=10
            init_methods=1
            ;;
        HadGEM3-GC31-MM)
            run=10
            init_methods=1
            ;;
        EC-Earth3)
            run=10
            init_methods=4
            ;;
        MRI-ESM2-0)
            run=10
            init_methods=1
            ;;
        MPI-ESM1-2-LR)
            run=16
            init_methods=1
            ;;
        FGOALS-f3-L)
            run=9
            init_methods=1
            ;;
        CNRM-ESM2-1)
            run=10
            init_methods=1
            ;;
        MIROC6)
            run=10
            init_methods=1
            ;;
        IPSL-CM6A-LR)
            run=10
            init_methods=1
            ;;
        CESM1-1-CAM5-CMIP5)
            run=10
            init_methods=1
            ;;
        NorCPM1)
            run=10
            init_methods=2
            ;;
        *)
            echo "[ERROR] Model $model not found in dictionary"
            exit 1
            ;;
    esac

    # Set up the output dir for the LOTUS output
    OUTPUTS_DIR="/work/scratch-nopw/benhutch/${variable}/${model}/years_${forecast_range}/${season}/lotus-outputs"
    mkdir -p $OUTPUTS_DIR

        # loop over the ensemble members
        for run in $(seq 1 $run); do

            # if init_methods is greater than 1, then loop over the init methods
            if [ $init_methods -gt 1 ]; then

                # loop over the init methods
                for init in $(seq 1 $init_methods); do

                # echo the init method
                echo "[INFO] Calculating NAO for init method: $init"

                # echo the output directory
                echo "[INFO] Output directory for LOTUS output: $OUTPUTS_DIR"

                # Submit the job to LOTUS
                sbatch --partition=short-serial -t 5 -o $OUTPUTS_DIR/${model}_${variable}_${season}_${forecast_range}_${lag}_${run}_${init}.out -e $OUTPUTS_DIR/${model}_${variable}_${season}_${forecast_range}_${lag}_${run}_${init}.err $EXTRACTOR $model $variable $forecast_range $season $lag $run $init

                done
            else

            # echo the output directory
                echo "[INFO] Output directory for LOTUS output: $OUTPUTS_DIR"

                # Submit the job to LOTUS
                sbatch --partition=short-serial -t 5 -o $OUTPUTS_DIR/${model}_${variable}_${season}_${forecast_range}_${lag}_${run}.out -e $OUTPUTS_DIR/${model}_${variable}_${season}_${forecast_range}_${lag}_${run}.err $EXTRACTOR $model $variable $forecast_range $season $lag $run $init_methods

            fi
        done
    done
else

# Echo the model being processed
echo "Processing ${model}"

# Set up the output dir for the LOTUS output
OUTPUTS_DIR="/work/scratch-nopw/benhutch/${variable}/${model}/years_${forecast_range}/${season}/lotus-outputs"
mkdir -p $OUTPUTS_DIR

# Case statement to set the number of ensemble members and inits
case $model in
    BCC-CSM2-MR)
        run=8
        init_methods=1
        ;;
    MPI-ESM1-2-HR)
        run=10
        init_methods=1
        ;;
    CanESM5)
        run=20
        init_methods=1
        ;;
    CMCC-CM2-SR5)
        run=10
        init_methods=1
        ;;
    HadGEM3-GC31-MM)
        run=10
        init_methods=1
        ;;
    EC-Earth3)
        run=10
        init_methods=4
        ;;
    MRI-ESM2-0)
        run=10
        init_methods=1
        ;;
    MPI-ESM1-2-LR)
        run=16
        init_methods=1
        ;;
    FGOALS-f3-L)
        run=9
        init_methods=1
        ;;
    CNRM-ESM2-1)
        run=10
        init_methods=1
        ;;
    MIROC6)
        run=10
        init_methods=1
        ;;
    IPSL-CM6A-LR)
        run=10
        init_methods=1
        ;;
    CESM1-1-CAM5-CMIP5)
        run=10
        init_methods=1
        ;;
    NorCPM1)
        run=10
        init_methods=2
        ;;
    *)
        echo "[ERROR] Model $model not found in dictionary"
        exit 1
        ;;
esac

  # loop over the ensemble members
  for run in $(seq 1 $run); do

      # if init_methods is greater than 1, then loop over the init methods
      if [ $init_methods -gt 1 ]; then

          # loop over the init methods
          for init in $(seq 1 $init_methods); do

          # echo the init method
          echo "[INFO] Calculating NAO for init method: $init"

          # echo the output directory
          echo "[INFO] Output directory for LOTUS output: $OUTPUTS_DIR"

          # Submit the job to LOTUS
          sbatch --partition=short-serial -t 5 -o $OUTPUTS_DIR/${model}_${variable}_${season}_${forecast_range}_${lag}_${run}_${init}.out -e $OUTPUTS_DIR/${model}_${variable}_${season}_${forecast_range}_${lag}_${run}_${init}.err $EXTRACTOR $model $variable $forecast_range $season $lag $run $init

          done
      else

      # echo the output directory
          echo "[INFO] Output directory for LOTUS output: $OUTPUTS_DIR"

          # Submit the job to LOTUS
          sbatch --partition=short-serial -t 5 -o $OUTPUTS_DIR/${model}_${variable}_${season}_${forecast_range}_${lag}_${run}.out -e $OUTPUTS_DIR/${model}_${variable}_${season}_${forecast_range}_${lag}_${run}.err $EXTRACTOR $model $variable $forecast_range $season $lag $run $init_methods

      fi
  done
fi
