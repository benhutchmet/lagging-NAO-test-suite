#!/bin/bash
#
# submit-all-multi-model.apply-lag.bash
#
# Submit script for applying lag to all the year files
# for each model, run and init method
#
# Usage: submit-all-multi-model.apply-lag.bash <model> <variable> <region> <forecast-range> <season> <lag>
#
# For example: submit-all-multi-model.apply-lag.bash HadGEM3-GC31-MM psl north-atlantic 2-5 DJF 1
#

# import the models list
source $PWD/dictionaries.bash
# if dictionary.bash is not found, then exit
if [ $? -ne 0 ]; then
    echo "[ERROR] dictionaries.bash not found"
    exit 1
fi

# echo the multi-models list
echo "[INFO] models list: $models"

# set the usage message
USAGE_MESSAGE="Usage: submit-all-multi-model.apply-lag.bash <model> <variable> <region> <forecast-range> <season> <lag>"

# check that the correct number of arguments have been passed
if [ $# -ne 6 ]; then
    echo "$USAGE_MESSAGE"
    exit 1
fi

# extract the model, variable, region, forecast range, season and lag
model=$1
variable=$2
region=$3
forecast_range=$4
season=$5
lag=$6

# If model is a number
# Between 1-12
# Then model is equal to the ith element of the models array $models
if [[ $model =~ ^[0-9]+$ ]]; then
    # echo the model number
    echo "[INFO] Model number: $model"

    # Convert the models string to an array
    models_array=($models)
    # Echo the models array
    echo "[INFO] models array: ${models_array[*]}"

    # Extract the numbered element of the models array
    model=${models_array[$model-1]}

    # echo the model name
    echo "[INFO] Model name: $model"
    echo "[INFO] Extracting data for model: $model"
fi

# If the region is the number 1, then region = azores
# If the region is the number 2, then region = iceland
if [[ $region -eq 1 ]]; then
    region=azores
elif [[ $region -eq 2 ]]; then
    region=iceland
fi

# set the extractor script
#EXTRACTOR=$PWD/process_scripts/multi-model.apply-lag.bash
# Additional options for the extractor script
EXTRACTOR=$PWD/process_scripts/multi-model.apply-lag.GPT-optimize.bash

# make sure that cdo is loaded
module load jaspy

# if model=all, then run a for loop over all of the models
if [ "$model" == "all" ]; then

# set up the model list
echo "[INFO] Extracting data for all models: $models"

    for model in $models; do

    # Echo the model name
    echo "[INFO] Extracting data for model: $model"

    # Set up the number of ensemble members and initialisation methods using a case statement
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
            run=40
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

    # set the output directory
    OUTPUTS_DIR="${scratch_path}/${variable}/${model}/${region}/years_${forecast_range}/${season}/lotus-outputs"
    mkdir -p $OUTPUTS_DIR

        # loop over the ensemble members
        for run in $(seq 1 $run); do

            # if init_methods is greater than 1, then loop over the init methods
            if [ $init_methods -gt 1 ]; then

                # loop over the init methods
                for init in $(seq 1 $init_methods); do

                # echo the init method
                echo "[INFO] Extracting data for init method: $init"

                # echo the output directory
                echo "[INFO] Output directory: $OUTPUTS_DIR"

                # submit the job to LOTUS
                sbatch --partition=short-serial -t 5 -o $OUTPUTS_DIR/${model}_${variable}_${region}_${forecast_range}_${season}_${lag}_${run}_${init}.out -e $OUTPUTS_DIR/${model}_${variable}_${region}_${forecast_range}_${season}_${lag}_${run}_${init}.err $EXTRACTOR $model $variable $region $forecast_range $season $lag $run $init

                done
            else
                
                # echo the output directory
                echo "[INFO] Output directory: $OUTPUTS_DIR"
    
                # submit the job to LOTUS
                sbatch --partition=short-serial -t 5 -o $OUTPUTS_DIR/${model}_${variable}_${region}_${forecast_range}_${season}_${lag}_${run}_${init_methods}.out -e $OUTPUTS_DIR/${model}_${variable}_${region}_${forecast_range}_${season}_${lag}_${run}_${init_methods}.err $EXTRACTOR $model $variable $region $forecast_range $season $lag $run $init_methods
        
            fi
        done
    done
else

# Echo the model name
echo "[INFO] Extracting data for model: $model"

# Set up the output directory
OUTPUTS_DIR="${scratch_path}/${variable}/${model}/${region}/years_${forecast_range}/${season}/lotus-outputs"
mkdir -p $OUTPUTS_DIR

# Echo the output directory
echo "[INFO] Output directory: $OUTPUTS_DIR"

# Statement for applying the lag
echo "[INFO] Applying lag: $lag, for model: $model"

# Set up the number of ensemble members and initialisation methods using a case statement
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
        run=40
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
            echo "[INFO] Extracting data for init method: $init"

            # echo the output directory
            echo "[INFO] Output directory: $OUTPUTS_DIR"

            # submit the job to LOTUS
            sbatch --partition=short-serial -t 5 -o $OUTPUTS_DIR/${model}_${variable}_${region}_${forecast_range}_${season}_${lag}_${run}_${init}.out -e $OUTPUTS_DIR/${model}_${variable}_${region}_${forecast_range}_${season}_${lag}_${run}_${init}.err $EXTRACTOR $model $variable $region $forecast_range $season $lag $run $init

            done
        else
            
                # echo the output directory
                echo "[INFO] Output directory: $OUTPUTS_DIR"
        
                # submit the job to LOTUS
                sbatch --partition=short-serial -t 5 -o $OUTPUTS_DIR/${model}_${variable}_${region}_${forecast_range}_${season}_${lag}_${run}_${init_methods}.out -e $OUTPUTS_DIR/${model}_${variable}_${region}_${forecast_range}_${season}_${lag}_${run}_${init_methods}.err $EXTRACTOR $model $variable $region $forecast_range $season $lag $run $init_methods
        
        fi
    done
fi