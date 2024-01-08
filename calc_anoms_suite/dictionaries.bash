#!/bin/bash

# dictionaries.bash
# file containing the list of models to be processed
# and other relevant dictionaries

# file path for the temperature files on JASMIN
tas_path="/work/xfc/vol5/user_cache/benhutch/tas/"

# file path for the downwellin surface radiation files on JASMIN
rsds_path="/work/xfc/vol5/user_cache/benhutch/rsds/"

# file path for the wind speed files on JASMIN - NOT PROPERLY DOWNLOADED YET
wind_speed_path="/work/xfc/vol5/user_cache/benhutch/sfcWind/"

# set up the file path for the gridspec files

# file path for the gridspec files on JASMIN
# this folder will contain all of the relevant gridspec files
gridspec_path="/home/users/benhutch/gridspec/"

# badc file path for the gridspec files
badc_base_path="/badc/cmip6/data/CMIP6/DCPP/"

scratch_path="/work/scratch-nopw2/benhutch"

# set up the canari base dir
canari_base_dir="/gws/nopw/j04/canari/users/benhutch"

declare -A example_array=(["key1"]="value1", ["key2"]="value2", ["key3"]="value3")

# Create an associative array for the models
declare -A psl_models_nens=(["BCC-CSM2-MR"]=8
    ["MPI-ESM1-2-HR"]=10
    ["CanESM5"]=20
    ["CMCC-CM2-SR5"]=10
    ["HadGEM3-GC31-MM"]=10
    ["EC-Earth3"]=10
    ["MPI-ESM1-2-LR"]=16
    ["FGOALS-f3-L"]=9
    ["MIROC6"]=10
    ["IPSL-CM6A-LR"]=10
    ["CESM1-1-CAM5-CMIP5"]=40
    ["NorCPM1"]=10
)

# Create an associative array for the models - tas
declare -A tas_models_nens
tas_models_nens=(["BCC-CSM2-MR"]=8
    ["MPI-ESM1-2-HR"]=10
    ["CanESM5"]=20
    ["CMCC-CM2-SR5"]=10
    ["HadGEM3-GC31-MM"]=10
    ["EC-Earth3"]=10
    ["MPI-ESM1-2-LR"]=16
    ["FGOALS-f3-L"]=9
    ["MIROC6"]=10
    ["IPSL-CM6A-LR"]=10
    ["CESM1-1-CAM5-CMIP5"]=40
    ["NorCPM1"]=10
)

# Create an associative array for the models - rsds
declare -A rsds_models_nens
rsds_models_nens=(["BCC-CSM2-MR"]=8
    ["MPI-ESM1-2-HR"]=10
    ["CanESM5"]=20
    ["CMCC-CM2-SR5"]=10
    ["HadGEM3-GC31-MM"]=10
    ["EC-Earth3"]=10
    ["MPI-ESM1-2-LR"]=16
    ["FGOALS-f3-L"]=3
    ["MIROC6"]=10
    ["IPSL-CM6A-LR"]=10
    ["CESM1-1-CAM5-CMIP5"]=40
    ["NorCPM1"]=10
)

# TODO: These will need to be changed once we have consistent moels
# more consistent models
# Create an associative array for the models - sfcWind
declare -A sfcWind_models_nens
sfcWind_models_nens=(["BCC-CSM2-MR"]=8
    ["MPI-ESM1-2-HR"]=10
    ["CanESM5"]=20
    ["CMCC-CM2-SR5"]=10
    ["HadGEM3-GC31-MM"]=10
    ["FGOALS-f3-L"]=3
    ["IPSL-CM6A-LR"]=10
    ["CESM1-1-CAM5-CMIP5"]=40
)

# Same for tos
declare -A tos_models_nens
tos_models_nens=(
    ["CanESM5"]=40
    ["HadGEM3-GC31-MM"]=10
    ["EC-Earth3"]=10
    ["FGOALS-f3-L"]=6
    ["MIROC6"]=10
    ["IPSL-CM6A-LR"]=10
    ["CESM1-1-CAM5-CMIP5"]=40
    ["NorCPM1"]=10
)

# list of models
# available on JASMIN
models="BCC-CSM2-MR MPI-ESM1-2-HR CanESM5 CMCC-CM2-SR5 HadGEM3-GC31-MM EC-Earth3 MPI-ESM1-2-LR FGOALS-f3-L MIROC6 IPSL-CM6A-LR CESM1-1-CAM5-CMIP5 NorCPM1"

single_file_models="BCC-CSM2-MR MPI-ESM1-2-HR CanESM5 CMCC-CM2-SR5 MRI-ESM2-0 MPI-ESM1-2-LR FGOALS-f3-L CNRM-ESM2-1 MIROC6 IPSL-CM6A-LR CESM1-1-CAM5-CMIP5 NorCPM1"

multi_file_models="HadGEM3-GC31-MM EC-Earth3 EC-Earth3-HR"

sfcWind_models="BCC-CSM2-MR MPI-ESM1-2-HR CanESM5 CESM1-1-CAM5-CMIP5 EC-Earth3 FGOALS-f3-L MIROC6 IPSL-CM6A-LR HadGEM3-GC31-MM"

# sfcwind missing CMCC-CM2-SR5, MPI-ESM1-2-LR, NorCPM1
sfcWind_models_numbers=1,2,3,5,6,8,9,10,11

# set up the sfcWind models list
# this is the list of models that have sfcWind data available in my case
sfcWind_models="BCC-CSM2-MR MPI-ESM1-2-HR CanESM5 HadGEM3-GC31-MM EC-Earth3 FGOALS-f3-L MIROC6 IPSL-CM6A-LR CESM1-1-CAM5-CMIP5"

# rsds missing MPI-ESM1-2-LR
rsds_models_numbers=1,2,3,4,5,6,8,9,10,11,12

# set up the rsds models list
# this is the list of models that have rsds data available in my case
rsds_models="BCC-CSM2-MR MPI-ESM1-2-HR CanESM5 CMCC-CM2-SR5 HadGEM3-GC31-MM EC-Earth3 FGOALS-f3-L MIROC6 IPSL-CM6A-LR CESM1-1-CAM5-CMIP5 NorCPM1"

tas_models_numbers=1,2,3,4,5,6,8,9,10,11,12

# set up the tas models list
# this is the list of models that have tas data available in my case
tas_models="BCC-CSM2-MR MPI-ESM1-2-HR CanESM5 CMCC-CM2-SR5 HadGEM3-GC31-MM EC-Earth3 FGOALS-f3-L MIROC6 IPSL-CM6A-LR CESM1-1-CAM5-CMIP5 NorCPM1"

psl_models_numbers=1,2,3,4,5,6,8,9,10,11,12

# set up the psl models list
# this is the list of models that have psl data available in my case
psl_models="BCC-CSM2-MR MPI-ESM1-2-HR CanESM5 CMCC-CM2-SR5 HadGEM3-GC31-MM EC-Earth3 MPI-ESM1-2-LR FGOALS-f3-L MIROC6 IPSL-CM6A-LR CESM1-1-CAM5-CMIP5 NorCPM1"

# Set up the tos models list
tos_models="CanESM5 HadGEM3-GC31-MM EC-Earth3 FGOALS-f3-L MIROC6 IPSL-CM6A-LR CESM1-1-CAM5-CMIP5 NorCPM1"

tos_model_numbers=3,5,6,8,9,10,11,12

# models downloaded from CEDA/LLNL ESGF server to xfer
# format these as arrays
ESGF_models=("NorCPM1" "EC-Earth3-HR" "CESM1-1-CAM5-CMIP5" "IPSL-CM6A-LR" "MIROC6" "CNRM-ESM2-1" "FGOALS-f3-L" "MPI-ESM1-2-LR" "MRI-ESM2-0")

# nodes for downloading from ESGF
# corresponding to the models above
ESGF_nodes=("noresg.nird.sigma2.no" "esg-dn1.nsc.liu.se" "esgf-data.ucar.edu" "vesg.ipsl.upmc.fr" "esgf-data02.diasjp.net" "esg1.umr-cnrm.fr" "esg.lasg.ac.cn" "esgf.dwd.de" "esgf-data03.diasjp.net")

# models for downloading wind speed data from ESGF
# check availability of wind speed data for these models on ESGF first
wind_speed_ESGF_models=("CESM1-1-CAM5-CMIP5" "IPSL-CM6A-LR" "MIROC6" "FGOALS-f3-L" "BCC-CSM2-MR" "MPI-ESM1-2-HR" "CanESM5" "CMCC-CM2-SR5" "EC-Earth3")

# nodes for downloading wind speed data from ESGF
# corresponding to the models above
# will need to make sure that these are the correct nodes
wind_speed_ESGF_nodes=("esgf-data.ucar.edu" "vesg.ipsl.upmc.fr" "esgf-data02.diasjp.net" "esg.lasg.ac.cn" "cmip.bcc.cma.cn" "esgf3.dkrz.de" "crd-esgf-drc.ec.gc.ca" "esgf-node2.cmcc.it" "esgf.bsc.es")

# models for downloading TAS data from ESGF
tas_ESGF_models=("CESM1-1-CAM5-CMIP5" "FGOALS-f3-L" "MPI-ESM1-2-LR")

# nodes for downloading TAS data from ESGF
# corresponding to the models above
# will need to make sure that these are the correct nodes
tas_ESGF_nodes=("esgf-data.ucar.edu" "esg.lasg.ac.cn" "esgf.dwd.de")

# models for downloading surface radiation data from ESGF
rsds_ESGF_models=("CESM1-1-CAM5-CMIP5" "FGOALS-f3-L" "BCC-CSM2-MR")

# nodes for downloading surface radiation data from ESGF
# corresponding to the models above
# will need to make sure that these are the correct nodes
rsds_ESGF_nodes=("esgf-data.ucar.edu" "esg.lasg.ac.cn" "esgf-data1.llnl.gov")

canari_dir="/gws/nopw/j04/canari/users/benhutch"

# merged_file_dir="${canari_dir}/dcppA-hindcast/data
