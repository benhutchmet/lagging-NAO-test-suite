#!/bin/bash
#
# multi-model.sel-region-forecast-range-season.bash
#
# For example: multi-model.sel-region-forecast-range-season.bash HadGEM3-GC31-MM 1960 1 psl north-atlantic 2-5 DJFM dcppA-hindcast 92500
#
# NOTE: Seasons should be formatted using: JFMAYULGSOND
#

# source the dictionaries.bash file
source /home/users/benhutch/skill-maps-rose-suite/dictionaries.bash

# Print all of the CLI arguments
echo "CLI arguments: $@"
echo "Number of CLI arguments: $#"


# check if the correct number of arguments have been passed
if [ $# -ne 8 ]; then
    echo "Usage: multi-model.sel-region-forecast-range-season.bash <model> <initialization-year> <run-number> <variable> <region> <forecast-range> <season> <experiment>"
    exit 1
fi

# extract the data from the command line
model=$1
year=$2
run=$3
variable=$4
region=$5
forecast_range=$6
experiment=$8
# pressure_level=$9

# set up the gridspec file
grid="/home/users/benhutch/gridspec/gridspec-${region}.txt"

# if the gridspec file does not exist, exit
if [ ! -f "$grid" ]; then
    echo "[ERROR] Gridspec file not found"
    exit 1
fi

# echo the gridspec file path
echo "Gridspec file: $grid"

# model name and family
# set up an if loop for the model name
if [ "$model" == "BCC-CSM2-MR" ]; then
    model_group="BCC"
elif [ "$model" == "MPI-ESM1-2-HR" ]; then
    model_group="MPI-M"
elif [ "$model" == "CanESM5" ]; then
    model_group="CCCma"
elif [ "$model" == "CMCC-CM2-SR5" ]; then
    model_group="CMCC"
elif [ "$model" == "HadGEM3-GC31-MM" ]; then
    model_group="MOHC"
elif [ "$model" == "EC-Earth3" ]; then
    model_group="EC-Earth-Consortium"
elif [ "$model" == "EC-Earth3-HR" ]; then
    model_group="EC-Earth-Consortium"
elif [ "$model" == "MRI-ESM2-0" ]; then
    model_group="MRI"
elif [ "$model" == "MPI-ESM1-2-LR" ]; then
    model_group="DWD"
elif [ "$model" == "FGOALS-f3-L" ]; then
    model_group="CAS"
elif [ "$model" == "CNRM-ESM2-1" ]; then
    model_group="CNRM-CERFACS"
elif [ "$model" == "MIROC6" ]; then
    model_group="MIROC"
elif [ "$model" == "IPSL-CM6A-LR" ]; then
    model_group="IPSL"
elif [ "$model" == "CESM1-1-CAM5-CMIP5" ]; then
    model_group="NCAR"
elif [ "$model" == "NorCPM1" ]; then
    model_group="NCC"
else
    echo "[ERROR] Model not recognised"
    exit 1
fi

# activate the environment containing cdo
module load jaspy

# /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/psl/MIROC6
# psl_Amon_MIROC6_dcppA-hindcast_s2021-r9i1p1f1_gn_202111-203112.nc

# set up the files to be processed
# if the variable is psl
if [ "$variable" == "psl" ]; then
    # if the model is BCC-CSM2-MR or MPI-ESM1-2-HR or CanESM5 or CMCC-CM2-SR5
    if [ "$model" == "BCC-CSM2-MR" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then
        # set up the input files
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i?p?f?/Amon/psl/g?/files/d????????/*.nc"
    # for the single file models downloaded from ESGF
    elif [ "$model" == "MPI-ESM1-2-LR" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "MIROC6" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "NorCPM1" ] || [ "$model" == "HadGEM3-GC31-MM" ] || [ "$model" == "EC-Earth3" ]; then
        # set up the input files from xfc
        # check that this returns the files
        files="${canari_base_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*_g*_*.nc"
    else
        echo "[ERROR] Model not recognised for variable psl"
        exit 1
    fi
# if the variable is tas
elif [ "$variable" == "tas" ]; then
        
        # set up the models that have tas on JASMIN
        # these include NorCPM1, IPSL-CM6A-LR, MIROC6, BCC-CSM2-MR, MPI-ESM1-2-HR, CanESM5, CMCC-CM2-SR5, EC-Earth3, HadGEM3-GC31-MM 
        if [ "$model" == "NorCPM1" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "BCC-CSM2-MR" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then
        # set up the input files
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i?p?f?/Amon/tas/g?/files/d????????/*.nc"

    # for the files downloaded from ESGF
    # which includes CESM1-1-CAM5-CMIP5, FGOALS-f3-L, MPI-ESM1-2-LR
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "MPI-ESM1-2-LR" ]; then

        # set up the input files from canari
        files="${canari_base_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g*_*.nc"

    # if the model is HadGEM3 or EC-Earth3
    elif [ "$model" == "HadGEM3-GC31-MM" ]; then
        
        # set up the input files from badc
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i?p?f?/Amon/tas/g?/files/d????????/*.nc"

        # set up the merged file first
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # set up the start year
        start_year="${year}11"

        # set up the end year
        end_year=$((year + 11))"03"

        # set up the merged file name
        merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_${start_year}-${end_year}.nc

        # set up the merged file path
        merged_file_path=${merged_file_dir}/${merged_filename}

        # if the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Not overwriting $merged_file_path"
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}

    elif [ "$model" == "EC-Earth3" ]; then

        # set up the i1 and i2 input files from badc
        i1_multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i1p?f?/Amon/tas/g?/files/d????????/*.nc"
        i2_multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i2p?f?/Amon/tas/g?/files/d????????/*.nc"

        # set up the merged file dir
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # set up the start year
        start_year="${year}11"

        # set up the end year
        end_year=$((year + 11))"10"

        # set up the merged file names
        i1_merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f1_gr_${start_year}-${end_year}.nc
        i2_merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i2p1f1_gr_${start_year}-${end_year}.nc

        # set up the merged file paths
        i1_merged_file_path=${merged_file_dir}/${i1_merged_filename}
        i2_merged_file_path=${merged_file_dir}/${i2_merged_filename}

        # if the merged file already exists, do not overwrite
        if [ -f "$i1_merged_file_path" ]; then
            echo "INFO: Merged file already exists: $i1_merged_file_path"
            echo "INFO: Not overwriting $i1_merged_file_path"
        else
            echo "INFO: Merged file does not exist: $i1_merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $i1_multi_files $i1_merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # if the merged file already exists, do not overwrite
        if [ -f "$i2_merged_file_path" ]; then
            echo "INFO: Merged file already exists: $i2_merged_file_path"
            echo "INFO: Not overwriting $i2_merged_file_path"
        else
            echo "INFO: Merged file does not exist: $i2_merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $i2_multi_files $i2_merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files="${merged_file_dir}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p1f1_gr_${start_year}-${end_year}.nc"

    else
        echo "[ERROR] Model not recognised for variable tas"
        exit 1
    fi
# if the variable is rsds
elif [ "$variable" == "rsds" ]; then
    # set up the models that have rsds on JASMIN
    # thes incldue NorCPM1, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR, CanESM5, CMCC-CM2-SR5
    if [ "$model" == "NorCPM1" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then
        
        # set up the input files
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i?p?f?/Amon/rsds/g?/files/d????????/*.nc"

    # for the files downloaded from ESGF
    # for models CESM1-1-CAM5-CMIP5, FGOALS-f3-L, BCC-CSM2-MR
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "BCC-CSM2-MR" ]; then
    
        # set up the input files from canari
        files="${canari_base_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g*_*.nc"

    elif [ "$model" == "HadGEM3-GC31-MM" ]; then
        
        # set up the input files from badc
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i?p?f?/Amon/rsds/g?/files/d????????/*.nc"

        # set up the merged file first
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # set up the start year
        start_year="${year}11"

        # set up the end year
        end_year=$((year + 11))"03"

        # set up the merged file name
        merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_${start_year}-${end_year}.nc

        # set up the merged file path
        merged_file_path=${merged_file_dir}/${merged_filename}

        # if the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Not overwriting $merged_file_path"
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}

    elif [ "$model" == "EC-Earth3" ]; then

        # Set up the i1 and i2 input files
        i1_multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i1p?f?/Amon/rsds/g?/files/d????????/*.nc"
        i2_multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i2p?f?/Amon/rsds/g?/files/d????????/*.nc"

        # Set up the merged file dir
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # Set up the start year
        start_year="${year}11"

        # Set up the end year
        end_year=$((year + 11))"10"

        # Set up the merged file names
        i1_merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f1_gr_${start_year}-${end_year}.nc
        i2_merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i2p1f1_gr_${start_year}-${end_year}.nc

        # Set up the merged file paths
        i1_merged_file_path=${merged_file_dir}/${i1_merged_filename}
        i2_merged_file_path=${merged_file_dir}/${i2_merged_filename}

        # If the merged file already exists, do not overwrite
        if [ -f "$i1_merged_file_path" ]; then
            echo "INFO: Merged file already exists: $i1_merged_file_path"
            echo "INFO: Not overwriting $i1_merged_file_path"
        else
            echo "INFO: Merged file does not exist: $i1_merged_file_path"
            echo "INFO: Proceeding with script"

            # Merge the files
            cdo mergetime $i1_multi_files $i1_merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # If the merged file already exists, do not overwrite
        if [ -f "$i2_merged_file_path" ]; then
            echo "INFO: Merged file already exists: $i2_merged_file_path"
            echo "INFO: Not overwriting $i2_merged_file_path"
        else
            echo "INFO: Merged file does not exist: $i2_merged_file_path"
            echo "INFO: Proceeding with script"

            # Merge the files
            cdo mergetime $i2_multi_files $i2_merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files="${merged_file_dir}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p1f1_gr_${start_year}-${end_year}.nc"

    else
        echo "[ERROR] Model not recognised for variable rsds"
        exit 1
    fi
# if the variable is sfcWind - currently not downloaded
elif [ "$variable" == "sfcWind" ]; then
    # set up the models which have sfcWind on JASMIN
    # this includes HadGEM3-GC31-MM, EC-Earth3
    if [ "$model" == "HadGEM3-GC31-MM" ] ; then
        # set up the input files - only one initialization scheme
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i?p?f?/Amon/sfcWind/g?/files/d????????/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_*.nc"

        # merge the *.nc files into one file
        # set up the merged file first
        merged_file_dir=${canari_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # set up the start year
        # which is the year of the initialization and 11
        # for example 1960 would be 196011
        start_year="${year}11"

        # set up the end year
        # which is the year of the initialization + 11
        # for example 1960 would be 197103
        end_year=$((year + 11))"03"

        # set up the merged file name
        merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_${start_year}-${end_year}.nc

        # set up the merged file path
        merged_file_path=${merged_file_dir}/${merged_filename}

        # if the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Not overwriting $merged_file_path"
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}

    # for the files downloaded from ESGF
    elif [ "$model" == "EC-Earth3" ]; then
        # Set up the input files from canari
        # only i2 available for sfcWind currently
        #i1_multi_files="${canari_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p*f*_g*_*.nc"
        i2_multi_files="${canari_dir}/${experiment}/${variable}/${model}/data/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i2p*f*_g*_*.nc"

        # Set up the merged file dir
        merged_file_dir=${canari_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # Set up the start year
        start_year="${year}11"

        # Set up the end year
        end_year=$((year + 10))"12"

        # Set up the merged file name
        #i1_merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f1_gn_${start_year}-${end_year}.nc
        i2_merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i2p1f1_gn_${start_year}-${end_year}.nc

        # Set up the merged file path
        #i1_merged_file_path=${merged_file_dir}/${i1_merged_filename}
        i2_merged_file_path=${merged_file_dir}/${i2_merged_filename}

        # If the merged file already exists, do not overwrite
        # if [ -f "$i1_merged_file_path" ]; then
        #     echo "INFO: Merged file already exists: $i1_merged_file_path"
        #     echo "INFO: Not overwriting $i1_merged_file_path"
        # else
        #     echo "INFO: Merged file does not exist: $i1_merged_file_path"
        #     echo "INFO: Proceeding with script"

        #     # Merge the files
        #     cdo mergetime $i1_multi_files $i1_merged_file_path

        #     echo "[INFO] Finished merging files for $model"
        # fi

        # If the merged file already exists, do not overwrite
        if [ -f "$i2_merged_file_path" ]; then
            echo "INFO: Merged file already exists: $i2_merged_file_path"
            echo "INFO: Not overwriting $i2_merged_file_path"
        else
            echo "INFO: Merged file does not exist: $i2_merged_file_path"
            echo "INFO: Proceeding with script"

            # Merge the files
            cdo mergetime $i2_multi_files $i2_merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files="${merged_file_dir}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p1f1_gn_${start_year}-${end_year}.nc"

    # elif for CESM and BCC (in a different canari folder)
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "BCC-CSM2-MR" ]; then
        # Set up the input files from canari
        files="${canari_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g*_*.nc"

    # set up the remaining models downloaded from ESGF
    # this includes FGOALS-f3-L, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR, CanESM5, CMCC-CM2-SR5
    # these are in a different canari folder
    elif [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ]; then
        # set up the input files from canari
        files=${canari_dir}/${experiment}/${variable}/${model}/data/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g*_*.nc
    else
        echo "[ERROR] Model not recognised for variable sfcWind"
        exit 1
    fi
# in the case the variable is tos - SSTs
elif [ "$variable" == "tos" ]; then
    # Set up the single file models
    # which have been downloaded into my gws from ESGF
    if [ "$model" == "CanESM5" ] || [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "NorCPM1" ]; then
        # Set up the input files from canari
        # example: /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/tos/CanESM5/data
        # file example: tos_Omon_MIROC6_dcppA-hindcast_s2021-r9i1p1f1_gn_202111-203112.nc
        # specify a regular grid - gn - for CESM1-1-CAM5-CMIP5 (has both gr and gn)
        files="${canari_dir}/${experiment}/${variable}/${model}/data/${variable}_Omon_${model}_${experiment}_s${year}-r${run}i*p*f*_gn_*.nc"
    # Set up the multi-file models
    # First the HadGEM case
    elif [ "$model" == "HadGEM3-GC31-MM" ]; then
        # Set up the multi-file input files for a single initialization scheme, run and year
        # which have been downloaded into my gws from ESGF
        multi_files="${canari_dir}/${experiment}/${variable}/${model}/data/${variable}_Omon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_*.nc"

        # set up the merged file directory
        merged_file_dir=${canari_dir}/${experiment}/${variable}/${model}/data/merged_files
        mkdir -p $merged_file_dir

        # set up the start year
        start_year="${year}11"

        # set up the end year
        end_year=$((year + 11))"03"

        # set up the merged file name
        merged_filename=${variable}_Omon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_${start_year}-${end_year}.nc
        # merged file path
        merged_file_path=${merged_file_dir}/${merged_filename}

        # if the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Not overwriting $merged_file_path"
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}

    # Now the EC-Earth case
    elif [ "$model" == "EC-Earth3" ]; then
        # Set up the multi-file input files for a single initialization scheme, run and year
        # only i2 in this case
        # which have been downloaded into my gws from ESGF
        multi_files="${canari_dir}/${experiment}/${variable}/${model}/data/${variable}_Omon_${model}_${experiment}_s${year}-r${run}i2p1f1_gn_*.nc"

        # set up the merged file directory
        merged_file_dir=${canari_dir}/${experiment}/${variable}/${model}/data/merged_files
        mkdir -p $merged_file_dir

        # set up the start year
        start_year="${year}11"

        # set up the end year
        end_year=$((year + 10))"12"

        # set up the merged file name
        merged_filename=${variable}_Omon_${model}_${experiment}_s${year}-r${run}i2p1f1_gn_${start_year}-${end_year}.nc
        # merged file path
        merged_file_path=${merged_file_dir}/${merged_filename}

        # if the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Not overwriting $merged_file_path"
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}
    else
        echo "[ERROR] Model not recognised for variable tos"
        exit 1
    fi

# If the variable is ua or va
elif [ "$variable" == "ua" ]; then
    # Set up the single file models
    # which have been downloaded into my gws from ESGF
    if [ "$model" == "NorCPM1" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ] || [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "BCC-CSM2-MR" ]; then
        # Set up the input files from canari
        # example: /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/ua/CanESM5/data
        # file example: ua_Amon_MIROC6_dcppA-hindcast_s2021-r9i1p1f1_gn_202111-203112.nc
        # specify a regular grid - gn - for CESM1-1-CAM5-CMIP5 (has both gr and gn)
        # extract the first three letters from ${year}
        year_prefix=${year:0:3}
        files="${canari_base_dir}/${experiment}/${variable}/${model}/data/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g?_${year_prefix}*.nc"
    elif [ "$model" == "HadGEM3-GC31-MM" ]; then
        # Set up the input files from badc
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i?p?f?/Amon/ua/g?/files/d????????/*.nc"

        # set up the merged file first
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # set up the start year
        start_year="${year}11"
        # set up the end year
        end_year=$((year + 11))"03"

        # set up the merged file name
        merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_${start_year}-${end_year}.nc
        # set up the merged file path
        merged_file_path=${merged_file_dir}/${merged_filename}

        # if the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Not overwriting $merged_file_path"
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}

    elif [ "$model" == "EC-Earth3" ]; then

        # Set up the i1 multi files
        i1_multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i1p?f?/Amon/ua/g?/files/d????????/*.nc"

        # Set up the merged file dir
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # Set up the start year and end year
        start_year="${year}11"
        end_year=$((year + 11))"10"

        # Set up the merged file name
        i1_merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f1_gr_${start_year}-${end_year}.nc
        merged_file_path=${merged_file_dir}/${i1_merged_filename}

        # If the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Not overwriting $merged_file_path"
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # Merge the files
            cdo mergetime $i1_multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}
    else
        echo "[ERROR] Model not recognised for variable ua"
        exit 1
    fi
elif [ "$variable" == "va" ]; then
    # Set up the single files models
    # which have been downloaded into my gws from ESGF
    if [ "$model" == "NorCPM1" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ] || [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "BCC-CSM2-MR" ]; then
        # Set up the files from canari
        # extract the first three letters from ${year}
        year_prefix=${year:0:3}
        files="${canari_base_dir}/${experiment}/${variable}/${model}/data/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g?_${year_prefix}*.nc"
    # In the case of HadGEM which must be merged
    elif [ "$model" == "HadGEM3-GC31-MM" ]; then
        # Set up the input files from badc
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i?p?f?/Amon/va/g?/files/d????????/*.nc"

        # set up the merged file first
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # set up the start year
        start_year="${year}11"
        # set up the end year
        end_year=$((year + 11))"03"

        # set up the merged file name
        merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_${start_year}-${end_year}.nc
        # set up the merged file path
        merged_file_path=${merged_file_dir}/${merged_filename}

        # if the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Deleting $merged_file_path"
            rm $merged_file_path

            # merge the files
            cdo mergetime $multi_files $merged_file_path
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # merge the files
            cdo mergetime $multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}
    # In the case of EC-Earth
    elif [ "$model" == "EC-Earth3" ]; then

        # Set up the i1 multi files
        i1_multi_files="${canari_base_dir}/${experiment}/${variable}/${model}/data/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p?f?_g?_*.nc"

        # Set up the merged file dir
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files
        mkdir -p $merged_file_dir

        # Set up the start year and end year
        start_year="${year}11"
        end_year=$((year + 11))"10"

        # Set up the merged file name
        i1_merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f1_gr_${start_year}-${end_year}.nc
        merged_file_path=${merged_file_dir}/${i1_merged_filename}

        # If the merged file already exists, do not overwrite
        if [ -f "$merged_file_path" ]; then
            echo "INFO: Merged file already exists: $merged_file_path"
            echo "INFO: Not overwriting $merged_file_path"
        else
            echo "INFO: Merged file does not exist: $merged_file_path"
            echo "INFO: Proceeding with script"

            # Merge the files
            cdo mergetime $i1_multi_files $merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # Set up the input files
        files=${merged_file_path}
    else
        echo "[ERROR] Model not recognised for variable va"
        exit 1
    fi
else
    echo "[ERROR] Variable not recognised"
    exit 1
fi

# Function to select the plev
# Write a function to select the level
# E.g. we might want to select the 925 hPa level
# Function to select a specific pressure level
# select_pressure_level() {
#     input_file=$1
#     output_file=$2
#     pressure_level=$3

#     # Select the specified pressure level
#     cdo sellevel,$pressure_level $input_file $output_file
# }


# loop through the files and process them
for INPUT_FILE in $files; do

    # extract the season from the command line
    season=$7

    # If the variable is ua or va, use the pressure level in the OUTPUT_DIR
    # if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    #     # Set up the name for the output directory
    #     OUTPUT_DIR="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/plev_${pressure_level}/outputs"

    # Set up the name for the output directory
    OUTPUT_DIR="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/years_${forecast_range}/${season}/outputs"

    
    # if the output directory does not exist, create it
    if [ ! -d "$OUTPUT_DIR" ]; then
        echo "INFO: Output directory does not exist: $OUTPUT_DIR"
        echo "INFO: Creating output directory"
        mkdir -p $OUTPUT_DIR
    else
        echo "INFO: Output directory already exists: $OUTPUT_DIR"
    fi

    #     # If the variable is ua or va, select the pressure level
    # if [ "$variable" == "ua" ] || [ "$variable" == "va" ]; then
    #     # Set up the output file name
    #     base_fname=$(basename "$INPUT_FILE")
    #     pressure_level_fname="plev-${base_fname}"
    #     TEMP_FILE="$OUTPUT_DIR/temp-${base_fname}"
    #     OUTPUT_FILE="$OUTPUT_DIR/${pressure_level_fname}"

    #     # If OUTPUT_FILE already exists, do not overwrite
    #     if [ -f "$OUTPUT_FILE" ]; then
    #         echo "INFO: OUTPUT_FILE already exists: $OUTPUT_FILE"
    #         echo "INFO: Overwriting $OUTPUT_FILE"
    #         # Delete the file
    #         rm $OUTPUT_FILE

    #         # Select the pressure level
    #         select_pressure_level $INPUT_FILE $OUTPUT_FILE $pressure_level
    #     else
    #         echo "INFO: OUTPUT_FILE does not exist: $OUTPUT_FILE"
    #         echo "INFO: Proceeding with script"

    #         # Select the pressure level
    #         select_pressure_level $INPUT_FILE $OUTPUT_FILE $pressure_level

    #         echo "[INFO] Finished selecting pressure level for $model"
    #     fi

    #     # Set up the input file
    #     INPUT_FILE=$OUTPUT_FILE
    # fi

    # set up the output file names
    echo "Processing $INPUT_FILE"
    base_fname=$(basename "$INPUT_FILE")
    regridded_fname="regridded-${base_fname}"
    season_fname="years-${forecast_range}-${season}-${region}-${base_fname}"
    TEMP_FILE="$OUTPUT_DIR/temp-${base_fname}"
    REGRIDDED_FILE="$OUTPUT_DIR/${regridded_fname}"
    OUTPUT_FILE="$OUTPUT_DIR/${season_fname}"
    MEAN_FILE="$OUTPUT_DIR/mean-${season_fname}"

    # # If MEAN_FILE already exists, do not overwrite
    # if [ -f "$MEAN_FILE" ]; then
    #     echo "INFO: MEAN_FILE already exists: $MEAN_FILE"
    #     echo "INFO: Not overwriting $MEAN_FILE"
    # else
    #     echo "INFO: MEAN_FILE does not exist: $MEAN_FILE"
    #     echo "INFO: Proceeding with script"
    
    # Regrid using bilinear interpolation
    # Selects region (as long as x and y dimensions divide by 2.5)
    cdo remapbil,$grid $INPUT_FILE $REGRIDDED_FILE

    # Extract initialization year from the input file name
    year=$(basename "$REGRIDDED_FILE" | sed 's/.*_s\([0-9]\{4\}\)-.*/\1/')

    # Set IFS to '-' and read into array
    IFS='-' read -ra numbers <<< "$forecast_range"

    # Extract numbers
    forecast_start_year=${numbers[0]}
    forecast_end_year=${numbers[1]}

    echo "First number: $forecast_start_year"
    echo "Second number: $forecast_end_year"

    # Declare the month codes
    declare -A months=( ["J"]=1 ["F"]=2 ["M"]=3 ["A"]=4 ["Y"]=5 ["U"]=6 ["L"]=7 ["G"]=8 ["S"]=9 ["O"]=10 ["N"]=11 ["D"]=12 )

    # Extract the month code from the season
    start_month=${months[${season:0:1}]}
    if [[ ${#season} -eq 2 ]]; then
    end_month=${months[${season:1:1}]}
    elif [[ ${#season} -eq 3 ]]; then
    end_month=${months[${season:2:1}]}
    elif [[ ${#season} -eq 4 ]]; then
    end_month=${months[${season:3:1}]}
    else
    end_month=${months[${season:3:1}]}
    fi

    # if start_month or end_month is a single digit, add a 0 to the start
    if [[ ${#start_month} -eq 1 ]]; then
    start_month="0${start_month}"
    fi

    # for end month, add a 0 to the start
    if [[ ${#end_month} -eq 1 ]]; then
    end_month="0${end_month}"
    fi

    echo "Start month: $start_month"
    echo "End month: $end_month"

    # If the season specified is DJF, DJFM, NDJF, or NDJ, then the start year needs to be one less than the initialization year
    if [[ $season == *"DJF"* ]] || [[ $season == *"NDJ"* ]]; then
        echo "Season is DJF or NDJ"
        echo "Modifying start year by -1"
        # modify the start_year -1
        start_year=$((forecast_start_year - 1))
    else
        # leave the start_year as it is
        start_year=$forecast_start_year
    fi

    # Calculate the start and end dates for the DJFM season
    start_date=$((year + start_year))"-${start_month}-01"
    end_date=$((year + forecast_end_year))"-${end_month}-31"

    echo "Start date: $start_date"
    echo "End date: $end_date"

    # convert from JFMAYULGSOND to JFMAMJJASOND format
    # if Y is in the season, replace with M
    if [[ $season == *"Y"* ]]; then
    season=${season//Y/M}
    fi

    # if U is in the season, replace with J
    if [[ $season == *"U"* ]]; then
    season=${season//U/J}
    fi

    # if L is in the season, replace with J
    if [[ $season == *"L"* ]]; then
    season=${season//L/J}
    fi

    # if G is in the season, replace with A
    if [[ $season == *"G"* ]]; then
    season=${season//G/A}
    fi

    # echo the season
    echo "Season: $season"

    # Constrain the input file to the DJFM season
    cdo select,season=${season} "$REGRIDDED_FILE" "$TEMP_FILE"

    # Extract the 2-9 years using cdo
    cdo select,startdate="$start_date",enddate="$end_date" "$TEMP_FILE" "$OUTPUT_FILE"

    # Take the time mean of the output file
    cdo timmean "$OUTPUT_FILE" "$MEAN_FILE"

    # Remove the temporary, regridded, and original output files
    rm "$TEMP_FILE"
    rm "$REGRIDDED_FILE"
    rm "$OUTPUT_FILE"

    echo "[INFO] Finished processing: $INPUT_FILE"
    # fi
done