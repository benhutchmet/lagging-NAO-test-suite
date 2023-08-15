# Dictionaries used for the python stage of the processing

# models used
models = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CanESM5", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1"]

# models no canesm
models_no_can = [ "BCC-CSM2-MR", "MPI-ESM1-2-HR", "CMCC-CM2-SR5", "HadGEM3-GC31-MM", "EC-Earth3", "MPI-ESM1-2-LR", "FGOALS-f3-L", "MIROC6", "IPSL-CM6A-LR", "CESM1-1-CAM5-CMIP5", "NorCPM1"]

# Models used in Marcheggiani et al. (2023)
marcheg_models = ['BCC-CSM2-MR', 'MPI-ESM1-2-HR', 'CanESM5', 'CMCC-CM2-SR5', 'HadGEM3-GC31-MM', 'EC-Earth3', 'MIROC6', 'IPSL-CM6A-LR', 'CESM1-1-CAM5-CMIP5', 'NorCPM1']

# CMIP6 models used in Smith et al. (2020)
smith_cmip6_models = [ 'MPI-ESM1-2-HR', 'HadGEM3-GC31-MM', 'EC-Earth3', 'MIROC6', 'IPSL-CM6A-LR', 'CESM1-1-CAM5-CMIP5', 'NorCPM1']

# File path for the observations from ERA5
# Processed using CDO manipulation
obs = "/home/users/benhutch/ERA5_psl/nao-anomaly/nao-anomaly-ERA5-5yrRM.nc"

# long obs
obs_long = "/home/users/benhutch/multi-model/multi-model-jasmin/NAO_index_8yrRM_long.nc"

# Directory for plots
plots_dir = "/home/users/benhutch/lagging-NAO-test-suite/plots/"

# Full ERA5 dataset for processing
full_era5 = "/home/users/benhutch/ERA5/adaptor.mars.internal-1691509121.3261805-29348-4-3a487c76-fc7b-421f-b5be-7436e2eb78d7.nc"

# Set up the base directory where the processed files are stored
base_directory = "/home/users/benhutch/alternate-lag-processed-data"

# Define the path to the processed azores data file
azores_data_path = "/home/users/benhutch/lagging-NAO-test-suite/saved_data/processed_data_azores.npy"

# Define the path to the processed iceland data file
iceland_data_path = "/home/users/benhutch/lagging-NAO-test-suite/saved_data/processed_data_iceland.npy"

# saved data dir
saved_data_dir = "/home/users/benhutch/lagging-NAO-test-suite/saved_data/"