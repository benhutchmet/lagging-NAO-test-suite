#!/bin/bash
#
# multi-model.sel-region-forecast-range-season.bash
#
# For example: multi-model.sel-region-forecast-range-season.bash BCC-CSM2-MR 1961 1 psl ULG dcppA-hindcast 1
#
# NOTE: Seasons should be formatted using: JFMAYULGSOND
#

# source the dictionaries.bash file
source /home/users/benhutch/skill-maps-rose-suite/dictionaries.bash

# Print all of the CLI arguments
echo "CLI arguments: $@"
echo "Number of CLI arguments: $#"
echo "Correct number of CLI arguments: 7"


# check if the correct number of arguments have been passed
if [ $# -ne 7 ]; then
    echo "Usage: multi-model.sel-region-forecast-range-season.bash <model> <initialization-year> <run-number> <variable> <season> <experiment> <init_scheme>"
    echo "Example: multi-model.sel-region-forecast-range-season.bash BCC-CSM2-MR 1961 1 psl ULG dcppA-hindcast 1"
    exit 1
fi

# extract the data from the command line
model=$1
year=$2
run=$3
variable=$4
season=$5
experiment=$6
init_scheme=$7

# Set the region
# WARNING: Set to global by default
region="global"

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
#${init_scheme}


# set up the files to be processed
# if the variable is psl
if [ "$variable" == "psl" ]; then
    # if the model is BCC-CSM2-MR or MPI-ESM1-2-HR or CanESM5 or CMCC-CM2-SR5
    if [ "$model" == "BCC-CSM2-MR" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then
        # set up the input files
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/psl/g?/files/d????????/*.nc"
    # for the single file models downloaded from ESGF
    elif [ "$model" == "MPI-ESM1-2-LR" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "MIROC6" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "NorCPM1" ] || [ "$model" == "HadGEM3-GC31-MM" ] || [ "$model" == "EC-Earth3" ]; then
        # set up the input files from xfc
        # check that this returns the files
        files="${canari_base_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}*g*_*.nc"
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
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/tas/g?/files/d????????/*.nc"

    # for the files downloaded from ESGF
    # which includes CESM1-1-CAM5-CMIP5, FGOALS-f3-L, MPI-ESM1-2-LR
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "MPI-ESM1-2-LR" ]; then

        # set up the input files from canari
        files="${canari_base_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p*f*_g*_*.nc"

    # if the model is HadGEM3 or EC-Earth3
    elif [ "$model" == "HadGEM3-GC31-MM" ]; then
        
        # set up the input files from badc
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/tas/g?/files/d????????/*.nc"

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
        files="${merged_file_dir}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p1f1_gr_${start_year}-${end_year}.nc"

    else
        echo "[ERROR] Model not recognised for variable tas"
        exit 1
    fi
# if the variable is rsds
elif [ "$variable" == "rsds" ]; then
    # set up the models that have rsds on JASMIN
    # thes incldue NorCPM1, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR, CanESM5, CMCC-CM2-SR5
    if [ "$model" == "NorCPM1" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "MPI-ESM1-2-HR" ]; then
        
        # set up the input files
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/rsds/g?/files/d????????/*.nc"

    # for the files downloaded from ESGF
    # for models CESM1-1-CAM5-CMIP5, FGOALS-f3-L, BCC-CSM2-MR
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "BCC-CSM2-MR" ]; then
    
        # set up the input files from canari
        files="${canari_base_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g*_*.nc"

    # For the new files downloaded to canari
    # for models CanESM5 and CMCC-CM2-SR5
    elif [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ]; then

        # set up the input files from canari
        files="${canari_base_dir}/${experiment}/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g*_*.nc"

    elif [ "$model" == "HadGEM3-GC31-MM" ]; then
        
        # set up the input files from badc
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/rsds/g?/files/d????????/*.nc"

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
        files="${merged_file_dir}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p1f1_gr_${start_year}-${end_year}.nc"

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
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/sfcWind/g?/files/d????????/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_*.nc"

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
        files="${merged_file_dir}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p1f1_gn_${start_year}-${end_year}.nc"

    # elif for CESM and BCC (in a different canari folder)
    elif [ "$model" == "CESM1-1-CAM5-CMIP5" ] || [ "$model" == "BCC-CSM2-MR" ]; then
        # Set up the input files from canari
        files="${canari_dir}/${experiment}/data/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i*p*f*_g*_*.nc"

    # set up the remaining models downloaded from ESGF
    # this includes FGOALS-f3-L, IPSL-CM6A-LR, MIROC6, MPI-ESM1-2-HR, CanESM5, CMCC-CM2-SR5
    # these are in a different canari folder
    elif [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "MIROC6" ] || [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ]; then
        # set up the input files from canari
        files=${canari_dir}/${experiment}/${variable}/${model}/data/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p*f*_g*_*.nc

    # Elif the model is CMCC-CM2-SR5
    elif [ "$model" == "CMCC-CM2-SR5" ]; then

        # Set up the input files from canari
        files=${canari_dir}/${experiment}/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p*f*_g*_*.nc
        
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
        files="${canari_dir}/${experiment}/${variable}/${model}/data/${variable}_Omon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p*f*_gn_*.nc"
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
        files="${canari_base_dir}/${experiment}/${variable}/${model}/data/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p*f*_g?_${year_prefix}*.nc"
    elif [ "$model" == "HadGEM3-GC31-MM" ]; then
        # Set up the input files from badc
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/ua/g?/files/d????????/*.nc"

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
        files="${canari_base_dir}/${experiment}/${variable}/${model}/data/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p*f*_g?_${year_prefix}*.nc"
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
# if the variable is pr
elif [ "${variable}" == "pr" ]; then
    # Set up the single file models which already exist on JASMIN
    if [ "$model" == "MPI-ESM1-2-HR" ] || [ "$model" == "CanESM5" ] || [ "$model" == "CMCC-CM2-SR5" ] || [ "$model" == "BCC-CSM2-MR" ]; then
        # Set up the input files from badc
        files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/pr/g?/files/d????????/*.nc"

    # Set up the single file models which have been downloaded to canari
    elif [ "$model" == "MPI-ESM1-2-LR" ] || [ "$model" == "FGOALS-f3-L" ] || [ "$model" == "IPSL-CM6A-LR" ] || [ "$model" == "NorCPM1" ] || [ "$model" == "MIROC6" ]; then
        # Set up the input files from canari
        files="${canari_base_dir}/${experiment}/${variable}/${model}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p*f*_g*_*.nc"

    # Set up the multi-file models
    # First the HadGEM case
    elif [ "$model" == "HadGEM3-GC31-MM" ]; then

        # Set up the input files from badc
        multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i${init_scheme}p?f?/Amon/pr/g?/files/d????????/*.nc"

        # Set up the merged file first
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files

        # If the directory does not exist, create it
        mkdir -p $merged_file_dir

        # Set up the start year
        start_year="${year}11"

        # Set up the end year
        end_year=$((year + 11))"03"

        # Set up the merged file name
        merged_filename=${variable}_Amon_${model}_${experiment}_s${year}-r${run}i1p1f2_gn_${start_year}-${end_year}.nc

        # Set up the merged file path
        merged_file_path=${merged_file_dir}/${merged_filename}

        # If the merged file already exists, do not overwrite
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
        i1_multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i1p?f?/Amon/pr/g?/files/d????????/*.nc"
        i2_multi_files="/badc/cmip6/data/CMIP6/DCPP/$model_group/$model/${experiment}/s${year}-r${run}i2p?f?/Amon/pr/g?/files/d????????/*.nc"

        # Set up the merged file dir
        merged_file_dir=${canari_base_dir}/${experiment}/data/${variable}/${model}/merged_files

        # If the directory does not exist, create it
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
            echo "INFO: removing merged file"
            rm $i1_merged_file_path

            # Merge the files
            cdo mergetime $i1_multi_files $i1_merged_file_path
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
            echo "Removing merged file"

            # Remove the merged file
            rm $i2_merged_file_path

            # Merge the files
            cdo mergetime $i2_multi_files $i2_merged_file_path
        else
            echo "INFO: Merged file does not exist: $i2_merged_file_path"
            echo "INFO: Proceeding with script"

            # Merge the files
            cdo mergetime $i2_multi_files $i2_merged_file_path

            echo "[INFO] Finished merging files for $model"
        fi

        # TODO: Problem is here - merged files are generated but not followed through

        # Set up the input files
        files="${merged_file_dir}/${variable}_Amon_${model}_${experiment}_s${year}-r${run}i${init_scheme}p1f1_gr_${start_year}-${end_year}.nc"

        # /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/pr/EC-Earth3/merged_files/pr_Amon_EC-Earth3_dcppA-hindcast_s1993-r10i1p1f1_gr_199311-200410.nc
        # /gws/nopw/j04/canari/users/benhutch/dcppA-hindcast/data/pr/EC-Earth3/merged_files/pr_Amon_EC-Earth3_dcppA-hindcast_s1993-r10i2p1f1_gr_199311-200410.nc

    else
        echo "[ERROR] Model not recognised for variable pr"
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
# Echo the files
echo "Processing file: $files"

# Check how many files there are
# If there is more than one file, exit with an error
# If there is one file, proceed with the script
# If there are no files, exit with an error
# Set up the number of files
nfiles=$(ls -1 $files | wc -l)

# If there are no files, exit with an error
if [ $nfiles -eq 0 ]; then
    echo "[ERROR] No files found"
    exit 1
# If there is more than one file, exit with an error
elif [ $nfiles -gt 1 ]; then
    echo "[ERROR] More than one file found"
    exit 1
# If there is one file, proceed with the script
elif [ $nfiles -eq 1 ]; then
    echo "[INFO] One file found"
    echo "[INFO] Proceeding with script"
fi

# Set up the name for the output directory
# NOTE: Modified for all forecast years
OUTPUT_DIR="/work/scratch-nopw2/benhutch/${variable}/${model}/${region}/all_forecast_years/${season}/outputs"

# Set up the INPUT_FILE
INPUT_FILE=$(ls $files)

# if the output directory does not exist, create it
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "INFO: Output directory does not exist: $OUTPUT_DIR"
    echo "INFO: Creating output directory"
    mkdir -p $OUTPUT_DIR
else
    echo "INFO: Output directory already exists: $OUTPUT_DIR"
fi

# set up the output file names
echo "Processing $INPUT_FILE"
base_fname=$(basename "$INPUT_FILE")
season_fname="all-years-${season}-${region}-${base_fname}"
OUTPUT_FILE="$OUTPUT_DIR/${season_fname}"

# Check the size of the output file
OUTPUT_FILE_SIZE=$(stat -c%s "$OUTPUT_FILE")

# If the OUTPUT file already exists and has a file size greater than 10000 bytes, do not overwrite
# This is done in the other script already
# if [ -f "$OUTPUT_FILE" ] && [ $OUTPUT_FILE_SIZE -gt 10000 ]; then
#     echo "INFO: Output file already exists: $OUTPUT_FILE"
#     echo "INFO: Not overwriting $OUTPUT_FILE"
#     echo "INFO: Proceeding with script"
# else
#     echo "INFO: Output file does not exist: $OUTPUT_FILE or is too small"
#     echo "INFO: removing $OUTPUT_FILE"
#     rm $OUTPUT_FILE

#     echo "INFO: Proceeding with script"
# fi

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

# Echo that we are selecting the season
echo "[INFO] Selecting the season: $season"

# Echo that we are remapping the file
echo "[INFO] Then remapping the file"

# If the season is NDJFM or DJFM
if [ "$season" == "NDJFM" ] || [ "$season" == "DJFM" ] || [ "$season" == "ONDJFM" ]; then
    echo "[INFO] Season is NDJFM or DJFM"
    echo "[INFO] Shifting the time axis by 3 months"
    # Select the season, shift the time axis, take the year mean and remap the file
    cdo -remapbil,$grid -yearmean -shifttime,-3mo -select,season=${season} $INPUT_FILE $OUTPUT_FILE
# If the season is DJF or NDJF
elif [ "$season" == "DJF" ] || [ "$season" == "NDJF" ] || [ "$season" == "ONDJF" ]; then
    echo "[INFO] Season is DJF or NDJF"
    echo "[INFO] Shifting the time axis by 2 months"
    # Select the season, shift the time axis, take the year mean and remap the file
    cdo -remapbil,$grid -yearmean -shifttime,-2mo -select,season=${season} $INPUT_FILE $OUTPUT_FILE
# If the season is NDJ or ONDJ
elif [ "$season" == "NDJ" ] || [ "$season" == "ONDJ" ]; then
    echo "[INFO] Season is NDJ or ONDJ"
    echo "[INFO] Shifting the time axis by 1 month"
    # Select the season, shift the time axis, take the year mean and remap the file
    cdo -remapbil,$grid -yearmean -shifttime,-1mo -select,season=${season} $INPUT_FILE $OUTPUT_FILE
# Else no need to shift the time axis
else
    echo "[INFO] Season is not NDJFM, DJFM, DJF, NDJF, NDJ or ONDJ"
    echo "[INFO] Season is $season"
    echo "[INFO] Not shifting the time axis"
    # Select the season, take the year mean and remap the file
    cdo -remapbil,$grid -yearmean -select,season=${season} $INPUT_FILE $OUTPUT_FILE
fi

echo "[INFO] Finished processing: $INPUT_FILE"
echo "[INFO] Output file: $OUTPUT_FILE"