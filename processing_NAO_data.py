# !/usr/bin/env python

# Functions which are used to process the NAO data
# ------------------------------------------------
# Using bash and cdo, data has already been manipulated /
# to give the 2-9 year mean DJFM NAO index for each year /
# for the different overlapping lagged periods.
# ------------------------------------------------
# Here we will use some functions to load the data and /
# prepare it for plotting.

# Imports
import argparse
import glob
import os
import sys
import re

# Third party imports
import xarray as xr
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import cftime

# Import from dictionaries
import dictionaries as dic

# Set up a function to load the data
import re

import xarray as xr
import glob
import os

# set up the plots directory
plots_dir = dic.plots_dir

# Import the lag ensemble function from NAO_Matching
from NAO_Matching import lag_ensemble


def load_lagged_ensemble_members(forecast_range, season, models):
    """
    This function loads the individual ensemble members for each model
    into a dictionary of datasets grouped by model

    Args:
        forecast_range (str): the forecast range from the command line
        season (str): the season from the command line
        models (list): the list of models from the dictionary

    Returns:
        lagged_ensemble_members (dict): a dictionary of datasets grouped by model
    """

    # set up the pattern for the directory where the files are stored
    NAO_dir = "/work/scratch-nopw2/benhutch/psl/"

    # create an empty dictionary
    lagged_ensemble_members = {}

    # set up the argpath
    arg_path = f"/NAO/years_{forecast_range}/{season}/outputs/"

    # echo the models being used
    print("models being used: ", models)

    # define initialization schemes
    init_schemes = ['same-init', 'init-minus-1', 'init-minus-2', 'init-minus-3']

    # loop over the models
    for model in models:
        # # print the model name
        print("model name: ", model)

        # create the model path
        model_path = NAO_dir + model + arg_path

        # # print the model path
        print("model path: ", model_path)

        # create a nested dictionary for each model
        lagged_ensemble_members[model] = {init_scheme: [] for init_scheme in init_schemes}

        # find all the netCDF files for this model
        files = glob.glob(model_path + "/*.nc")
        
        # Get a list of all files in the directory
        all_files = os.listdir(model_path)
        
        # Filter the list to include only netCDF files
        nc_files = [f for f in all_files if f.endswith(".nc")]
        
        # Print the list of netCDF files
        print(nc_files)

        # Echo files
        print("searching for files:", files)

        # loop over the files
        for file in files:
            # extract the initialization scheme from the filename
            init_scheme = os.path.basename(file).split('__')[-1].split('.')[1]

            print("searching in list",os.path.basename(file).split('_'))
            
            # echo the filename
            print("filename: ", file)
            
            # echo the initialization scheme
            print("init scheme: ", init_scheme)

            # if file does not exist
            # print "File does not exist"
            # and exit the program with an error message
            if not os.path.exists(file):
                print("File does not exist")
                sys.exit(1)

            # open the file and append the dataset to the appropriate list
            ds = xr.open_dataset(file)
            lagged_ensemble_members[model][init_scheme].append(ds)

        # print the init schemes
        print("init schemes: ", init_schemes)

        # print the model
        print("model: ", model)

        # concatenate the list of datasets for each init scheme into a single dataset
        for init_scheme in init_schemes:
            lagged_ensemble_members[model][init_scheme] = xr.concat(lagged_ensemble_members[model][init_scheme],
                                                                    dim='member')

    return lagged_ensemble_members


# Function to process the model data
def process_model_datasets_by_init(datasets_by_init):
    model_times_by_model_by_init = {}
    model_nao_anoms_by_model_by_init = {}

    for model, init_data in datasets_by_init.items():
        model_times_by_init = {}
        model_nao_anoms_by_init = {}

        model_data = init_data['same-init']['psl']
        print("testing model time values", model_data['time'].values)

        for init_scheme, datasets in init_data.items():
            # Extract the data for 'psl'
            model_data = datasets['psl']

            print("model data shape", np.shape(model_data))
            print("model data", model_data)

            # Extract the 'time' variable and convert its data type
            model_time = model_data['time'].values
            model_time = model_time.astype("datetime64[Y]")

            print("model time shape", np.shape(model_time))
            print("model time", model_time)

            # Process the 'psl' data from Pa to hPa
            model_nao_anom = model_data / 100

            print("model nao anom shape", np.shape(model_nao_anom))
            print("model nao anom", model_nao_anom)

            model_times_by_init[init_scheme] = model_time
            model_nao_anoms_by_init[init_scheme] = model_nao_anom

    model_times_by_model_by_init[model] = model_times_by_init
    model_nao_anoms_by_model_by_init[model] = model_nao_anoms_by_init

    return model_times_by_model_by_init, model_nao_anoms_by_model_by_init

# Now we need a function which can process this model data further
# We want to establish the overlapping (year) times for the four different initialization schemes
# And then combine the data into one array with the same time dimension
# this array will have dimensions (model, init_scheme, time, member)
def combine_model_data(model_times_by_model_by_init, model_nao_anoms_by_model_by_init):
    combined_data_by_model = {}

    # Iterate over each model's data
    for model, model_times_by_init in model_times_by_model_by_init.items():
        model_nao_anoms_by_init = model_nao_anoms_by_model_by_init[model]

        # Use the years from the 'init-minus-3' scheme as the overlapping years
        overlapping_years = (model_times_by_init['init-minus-3'])

        print("overlapping years", overlapping_years)
        print("overlapping years shape", np.shape(overlapping_years))

        # Select and combine the data for the overlapping years
        combined_data = []
        for init_scheme in ['same-init', 'init-minus-1', 'init-minus-2', 'init-minus-3']:

            print("init scheme", init_scheme)

            # print the data we want to look at
            print("model times by init", model_times_by_init[init_scheme])
            # print the shape of the data we want to look at
            print("model times by init shape", np.shape(model_times_by_init[init_scheme]))
            # print the type of the data we want to look at
            print("type of model times by init", type(model_times_by_init[init_scheme]))

            # Select the data for the overlapping years
            overlap_mask = np.in1d(model_times_by_init[init_scheme], overlapping_years)
            overlap_mask_xr = xr.DataArray(overlap_mask, dims=['time'])

            print("overlap mask", overlap_mask)
            print("overlap mask shape", np.shape(overlap_mask))
            print("overlap mask xr", overlap_mask_xr['time'])
            print("overlap mask xr shape", np.shape(overlap_mask_xr))

            selected_data = model_nao_anoms_by_init[init_scheme].where(overlap_mask_xr, drop=True)

            # Add an 'init_scheme' dimension to the data
            selected_data = selected_data.expand_dims({'init_scheme': [init_scheme]})
            combined_data.append(selected_data)

        # Combine the data into one array with the same time dimension
        combined_data = xr.concat(combined_data, dim='init_scheme')

        combined_data_by_model[model] = combined_data

    return combined_data_by_model

# Now we want to write a function that will extract the model data into an array
# with dimensions (init_scheme*member*model, time)
# this function takes a list of models as an argument
# and also takes the combined data by model as an argument
# and returns the extracted data as an array with dimensions (init_scheme*member*model, time)
def extract_model_data(models, combined_data_by_model):
    """
    Extracts the model data from the combined data by model and returns it as an array.

    Args:
        models (list): A list of models.
        combined_data_by_model (dict): A dictionary of combined data by model.

    Returns:
        extracted_data (xarray.DataArray): The extracted data as an array with dimensions (init_scheme*member*model, time).

    Raises:
        ValueError: If the model is not found in the combined data.

    """
    try:
        extracted_data = []

        for model in models:
            if model not in combined_data_by_model:
                raise ValueError(f"Model '{model}' not found in the combined data.")

            model_data = combined_data_by_model[model]
            model_data = model_data.expand_dims({'model': [model]})
            extracted_data.append(model_data)

        extracted_data = xr.concat(extracted_data, dim='model')
        extracted_data = extracted_data.stack(init_scheme_member=('init_scheme', 'member'))
        extracted_data = extracted_data.transpose('model', 'time', 'lat', 'lon', 'init_scheme_member')

        return extracted_data

    except Exception as e:
        # Handle any other unexpected exceptions and provide a meaningful error message
        raise RuntimeError("An error occurred during data extraction.") from e


# Function to process the observations
def process_observations(obs):
    """
    Process the observations data by extracting the time, converting units, and calculating NAO anomalies.

    Parameters
    ----------
    obs : xarray.Dataset
        The xarray dataset containing the observations data.

    Returns
    -------
    obs_nao_anom : numpy.ndarray
        The processed observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.
    """
    # Extract the data for the observations
    obs_psl = obs["var151"]
    obs_time = obs_psl["time"].values

    print(np.shape(obs_psl))
    print(np.shape(obs_time))

    # Set the type for the time variable
    obs_time = obs_time.astype("datetime64[Y]")

    # Process the obs data from Pa to hPa
    obs_nao_anom = obs_psl[:] / 100

    return obs_nao_anom, obs_time

# Function to calculate the ACC and its significance
# Function to calculate the ACC and significance
def pearsonr_score(obs, model, model_times, obs_times, start_date, end_date):
    """
    Calculate the Pearson correlation coefficient and p-value between two time series,
    considering the dimensions of the model and observation time arrays.

    Parameters:
    obs (array-like): First time series (e.g., observations)
    model (array-like): Second time series (e.g., model mean)
    model_times (array-like): Datetime array corresponding to the model time series
    obs_times (array-like): Datetime array corresponding to the observation time series
    start_date (str): Start date (inclusive) in the format 'YYYY-MM-DD'
    end_date (str): End date (inclusive) in the format 'YYYY-MM-DD'

    Returns:
    tuple: Pearson correlation coefficient and p-value
    """

    # Ensure the time series are numpy arrays or pandas Series
    time_series1 = np.array(obs)
    time_series2 = np.array(model)

    # Ensure the start_date and end_date are pandas Timestamp objects
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Convert obs_times to an array of Timestamp objects
    obs_times = np.vectorize(pd.Timestamp)(obs_times)

    # print the type of model_times
    print("model times type", type(model_times))

    # print the type of obs_times
    print("obs times type", type(obs_times))

    # Convert model_times to an array of Timestamp objects
    model_times = np.vectorize(pd.Timestamp)(model_times)

    # print the type of model_times
    print("model times type post vectorize", type(model_times))

    # print the shape of model_times
    print("model times shape post vectorize", np.shape(model_times))
    print("obs times shape post vectorize", np.shape(obs_times))

    # print the values of model_times
    # print("model times values post vectorize", model_times)
    # print("obs times values post vectorize", obs_times)

    # debugging for NAO matching
    # print("model times", model_times)
    # print("model times shape", np.shape(model_times))
    # print("model times type", type(model_times))

    # Analyze dimensions of model_times and obs_times
    model_start_index = np.where(model_times == start_date)[0][0]
    model_end_index = np.where(model_times <= end_date)[0][-1]
    obs_start_index = np.where(obs_times >= start_date)[0][0]
    obs_end_index = np.where(obs_times <= end_date)[0][-1]

    # print the model start index
    # print("model start index", model_start_index)
    # print("model end index", model_end_index)
    # print("obs start index", obs_start_index)
    # print("obs end index", obs_end_index)

    # Get the filtered times seriers
    filtered_model_times = model_times[model_start_index:model_end_index+1]
    filtered_obs_times = obs_times[obs_start_index:obs_end_index+1]

    # # print the shape and values of the filtered model times
    # print("filtered model times shape", np.shape(filtered_model_times))
    # # print("filtered model times values", filtered_model_times)

    # # print the shape and values of the filtered obs times
    # print("filtered obs times shape", np.shape(filtered_obs_times))
    # # print("filtered obs times values", filtered_obs_times)

    # Filter the time series based on the analyzed dimensions
    filtered_time_series1 = time_series1[obs_start_index:obs_end_index+1]
    filtered_time_series2 = time_series2[model_start_index:model_end_index+1]

    # print the shapes of the filtered time series
    print("filtered time series 1 shape", np.shape(filtered_time_series1))
    print("filtered time series 2 shape", np.shape(filtered_time_series2))
    # print("filtered time series 1", filtered_time_series1)
    # print("filtered time series 2", filtered_time_series2)


    # Calculate the Pearson correlation coefficient and p-value
    correlation_coefficient, p_value = stats.pearsonr(filtered_time_series1, filtered_time_series2)

    return correlation_coefficient, p_value

def calculate_rpc_time(correlation_coefficient, forecast_members, model_times, start_date, end_date):
    """
    Calculate the Ratio of Predictable Components (RPC) given the correlation
    coefficient (ACC) and individual forecast members for a specific time period.

    Parameters:
    correlation_coefficient (float): Correlation coefficient (ACC)
    forecast_members (array-like): Individual forecast members
    model_times (array-like): Datetime array corresponding to the model time series
    start_date (str): Start date (inclusive) in the format 'YYYY-MM-DD'
    end_date (str): End date (inclusive) in the format 'YYYY-MM-DD'

    Returns:
    float: Ratio of Predictable Components (RPC)
    """

    # Convert the input arrays to numpy arrays
    forecast_members = np.array(forecast_members)

    # Convert the start_date and end_date to pandas Timestamp objects
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Find the start and end indices of the time period for the model
    model_start_index = np.where(model_times >= start_date)[0][0]
    model_end_index = np.where(model_times <= end_date)[0][-1]

    # Filter the forecast members based on the start and end indices
    forecast_members = forecast_members[:, model_start_index:model_end_index+1]

    # Calculate the standard deviation of the predictable signal for forecasts (σfsig)
    sigma_fsig = np.std(np.mean(forecast_members, axis=0))

    # Calculate the total standard deviation for forecasts (σftot)
    sigma_ftot = np.std(forecast_members)

    # Calculate the RPC
    rpc = correlation_coefficient / (sigma_fsig / sigma_ftot)

    return rpc

# Define a function to calulate the RPS score with time
# Where RPS = RPC * (total variance of observations / total variance of all the individual forecast members)
def calculate_rps_time(RPC, obs, forecast_members, model_times, start_date, end_date):
    """
    Calculate the Ratio of Predictable Signals (RPS) given the Ratio of Predictable
    Components (RPC), observations, individual forecast members, and a time period.

    Parameters:
    RPC (float): Ratio of Predictable Components (RPC)
    obs (array-like): Observations
    forecast_members (array-like): Individual forecast members
    model_times (array-like): Datetime array corresponding to the model time series
    start_date (str): Start date (inclusive) in the format 'YYYY-MM-DD'
    end_date (str): End date (inclusive) in the format 'YYYY-MM-DD'

    Returns:
    float: Ratio of Predictable Signals (RPS)
    """

    # Print the types of the input arrays
    print("obs type", type(obs))
    print("forecast members type", type(forecast_members))
    print("model times type", type(model_times))


    # Convert the input arrays to numpy arrays
    obs = np.array(obs)
    forecast_members = np.array(forecast_members)

    # Print types of the input arrays post conversion
    print("obs type post conversion", type(obs))
    print("forecast members type post conversion", type(forecast_members))

    # Print the obs and forecast members
    # print("obs", obs)
    # print("forecast members", forecast_members)

    # print the shape of the obs and forecast members
    print("obs shape", np.shape(obs))
    print("forecast members shape", np.shape(forecast_members))

    # Convert the start_date and end_date to pandas Timestamp objects
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # test

    # Find the start and end indices of the time period for the model
    model_start_index = np.where(model_times >= start_date)[0][0]
    model_end_index = np.where(model_times <= end_date)[0][-1]

    # Filter the forecast members based on the start and end indices
    forecast_members = forecast_members[:, model_start_index:model_end_index+1]

    # Calculate the total variance of the observations
    variance_obs = np.nanstd(obs)

    # print the variance of the observations
    print("variance obs", variance_obs)
    # print the type of the variance of the observations
    print("variance obs type", type(variance_obs))

    # Calculate the total variance of the forecast members
    variance_forecast_members = np.std(forecast_members)

    # print the variance of the forecast members
    print("variance forecast members", variance_forecast_members)
    # print the type of the variance of the forecast members
    print("variance forecast members type", type(variance_forecast_members))

    # Calculate the RPS
    RPS = RPC * (variance_obs / variance_forecast_members)

    return RPS

# Function to adjust the variance of the ensemble
# Used once the no. of ensemble members has been 4x
# through the lagging process
def adjust_variance(model_time_series, rps_short, rps_long):
    """
    Adjust the variance of an ensemble mean time series by multiplying by the RPS score. This accounts for the signal to noise issue in the ensemble mean.

    Parameters
    ----------
    model_time_series : numpy.ndarray
        The input ensemble mean time series.
    rps_short : float
        The RPS score for the short period RPC (1960-2010).
    rps_long : float
        The RPS score for the long period RPC (1960-2019).

    Returns
    -------
    model_time_series_var_adjust_short : numpy.ndarray
        The variance adjusted ensemble mean time series for the short period RPC (1960-2010).
    model_time_series_var_adjust_long : numpy.ndarray
        The variance adjusted ensemble mean time series for the long period RPC (1960-2019).
    """

    # Adjust the variance of the ensemble mean time series
    model_time_series_var_adjust_short = rps_short * model_time_series
    model_time_series_var_adjust_long = rps_long * model_time_series

    return model_time_series_var_adjust_short, model_time_series_var_adjust_long


# function for calculating the RMSE and 5-95% uncertainty intervals for the variance adjusted output
def compute_rmse_confidence_intervals(obs_nao_anoms, adjusted_lagged_model_nao_anoms, obs_time, model_time_lagged,
                                      lower_bound=5, upper_bound=95):
    """
    Compute the root-mean-square error (RMSE) between the variance-adjusted ensemble
    mean and the observations. Calculate the 5%-95% confidence intervals for the
    variance-adjusted model output.

    Parameters
    ----------
    obs_nao_anoms : numpy.ndarray
        The observed NAO anomalies time series.
    adjusted_lagged_model_nao_anoms : numpy.ndarray
        The adjusted and lagged model NAO anomalies time series.
    obs_time : numpy.ndarray
        The time array for the observed NAO anomalies.
    model_time_lagged : numpy.ndarray
        The time array for the adjusted and lagged model NAO anomalies.
    lower_bound : int, optional, default: 5
        The lower percentile bound for the confidence interval.
    upper_bound : int, optional, default: 95
        The upper percentile bound for the confidence interval.

    Returns
    -------
    conf_interval_lower : numpy.ndarray
        The lower bound of the confidence interval.
    conf_interval_upper : numpy.ndarray
        The upper bound of the confidence interval.
    """

    # Convert the obs_time to an array of years
    obs_time = obs_time.astype("datetime64[Y]").astype(int) + 1970

    model_time_lagged = model_time_lagged.astype(int) + 1970

    # print the obs time and model time
    print("shape of obs time", np.shape(obs_time))
    print("shape of model time", np.shape(model_time_lagged))
    print("obs_time", obs_time)
    print("model time", model_time_lagged)

    # Check that both obs_time and model_time_lagged are arrays of years
    assert obs_time.dtype == np.dtype('int64')
    # assert model_time_lagged.dtype == np.dtype('int64')

    # Match the years in obs_time and model_time_lagged
    common_years = np.intersect1d(obs_time, model_time_lagged)

    # Match the years in obs_time and model_time_lagged
    common_years = np.intersect1d(obs_time, model_time_lagged)

    # print the common years
    print("common years", common_years)
    # print the type of the common years
    print("common years type", type(common_years))

    # Find the indices of the common years in both arrays
    obs_indices = np.where(np.isin(obs_time, common_years))[0]
    model_indices = np.where(np.isin(model_time_lagged, common_years))[0]

    # print the obs indices and model indices
    print("obs indices", obs_indices)
    print("model indices", model_indices)

    # print the shape and type of obs_nao_anoms
    print("obs nao anoms shape", np.shape(obs_nao_anoms))
    print("obs nao anoms type", type(obs_nao_anoms))

    # Print the shape and type of adjusted_lagged_model_nao_anoms
    print("adjusted lagged model nao anoms shape", np.shape(adjusted_lagged_model_nao_anoms))
    print("adjusted lagged model nao anoms type", type(adjusted_lagged_model_nao_anoms))

    # Create new arrays with the corresponding values for the common years
    obs_nao_anoms_matched = obs_nao_anoms[obs_indices]
    adjusted_lagged_model_nao_anoms_matched = adjusted_lagged_model_nao_anoms[model_indices]

    # print the shape and type of obs_nao_anoms_matched
    print("obs nao anoms matched shape", np.shape(obs_nao_anoms_matched))
    print("obs nao anoms matched type", type(obs_nao_anoms_matched))
    print("obs nao anoms matched", obs_nao_anoms_matched)

    # print the shape and type of adjusted_lagged_model_nao_anoms_matched
    print("adjusted lagged model nao anoms matched shape", np.shape(adjusted_lagged_model_nao_anoms_matched))
    print("adjusted lagged model nao anoms matched type", type(adjusted_lagged_model_nao_anoms_matched))
    print("adjusted lagged model nao anoms matched", adjusted_lagged_model_nao_anoms_matched)

    # Compute the root-mean-square error (RMSE) between the ensemble mean and the observations
    rmse = np.sqrt(np.mean((obs_nao_anoms_matched - adjusted_lagged_model_nao_anoms_matched) ** 2, axis=0))

    # Calculate the upper z-score for the RMSE
    z_score_upper = np.percentile(rmse, upper_bound)

    # Calculate the 5% and 95% confidence intervals using the RMSE
    conf_interval_lower = adjusted_lagged_model_nao_anoms_matched - (rmse)
    conf_interval_upper = adjusted_lagged_model_nao_anoms_matched + (rmse)

    return conf_interval_lower, conf_interval_upper

# Function to calculate ensemble mean for each model
def ensemble_mean(data_array):
    return np.mean(data_array, axis=0)

# Define a function to constrain the years to the years that are in all of the model members
def constrain_years(model_data, models):
    """
    Constrains the years to the years that are in all of the models.

    Parameters:
    model_data (dict): The processed model data.
    models (list): The list of models to be plotted.

    Returns:
    constrained_data (dict): The model data with years constrained to the years that are in all of the models.
    """
    # Initialize a list to store the years for each model
    years_list = []

    # Print the models being proces
    # print("models:", models)
    
    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Extract the years
            years = member.time.dt.year.values

            # Append the years to the list of years
            years_list.append(years)

    # Find the years that are in all of the models
    common_years = list(set(years_list[0]).intersection(*years_list))

    # Print the common years for debugging
    # print("Common years:", common_years)
    # print("Common years type:", type(common_years))
    # print("Common years shape:", np.shape(common_years))

    # Initialize a dictionary to store the constrained data
    constrained_data = {}

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Extract the years
            years = member.time.dt.year.values

            # Print the years extracted from the model
            # print('model years', years)
            # print('model years shape', np.shape(years))
            
            # Find the years that are in both the model data and the common years
            years_in_both = np.intersect1d(years, common_years)

            # print("years in both shape", np.shape(years_in_both))
            # print("years in both", years_in_both)
            
            # Select only those years from the model data
            member = member.sel(time=member.time.dt.year.isin(years_in_both))

            # Add the member to the constrained data dictionary
            if model not in constrained_data:
                constrained_data[model] = []
            constrained_data[model].append(member)

    # # Print the constrained data for debugging
    # print("Constrained data:", constrained_data)

    return constrained_data

# Define a plotting function that will plot the variance adjusted lag data
def plot_ensemble_members_and_lagged_adjusted_mean_alt_lag(models, model_data, model_time, obs_nao_anom,
                                                obs_time, forecast_range, season,
                                                lagged_ensemble_members_time, lagged_adjusted_ensemble_mean_short_norm_lag,
                                                lag=4):
    """
    Plot the ensemble mean of all members from all models and each of the ensemble members, with lagged and adjusted variance applied to the grand ensemble mean.

    Parameters
    ----------
    models : dict
        A dictionary containing a list of models.
    model_data : list
        A list containing all ensemble members.
    model_time: dict
        A dictionary containing the times for all of the ensemble members.
    obs_nao_anom : numpy.ndarray
        The observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.
    forecast_range (str):
        the forecast range from the command line
    season (str):
        the season from the command line
    lag : int, optional, default: 4
        The number of years to lag the grand ensemble mean by.


    Returns
    -------
    None
    """


    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))
                           
    # Set up a list for the ensemble members
    ensemble_members = []

    # Set up a dictionary to store the number of ensemble members for each model
    ensemble_members_count = {}

    # First constrain the years to the years that are in all of the models
    model_data = constrain_years(model_data, models)

    # Print the shape of the model data
    print("model data shape", np.shape(model_data))

    # Loop over the models
    for model in models:
        # Extract the model data
        model_data_combined = model_data[model]
        # model_time = model_time[model]

        # # Set up the model time
        # model_time = list(model_time.values())[0]

        # # Echo the model time
        # print("model time", model_time)

        # Print the model data for debugging
        print("Extracting data for model:", model)

        # Set the ensemble members count to zero
        if model not in ensemble_members_count:
            ensemble_members_count[model] = 0

        # Loop over the ensemble members in the model data
        for member in model_data_combined:
            # Append each ensemble member to the list of ensemble members
            ensemble_members.append(member)

            # Extract the years
            years = member.time.dt.year.values

            # # Extract the time
            model_time = member.time.values

            # Print the years extracted from the model
            # print("years", years)
            # print("years shape", np.shape(years))

            # Increment the ensemble members count
            ensemble_members_count[model] += 1

    # Convert the counts to a list of tuples
    ensemble_members_count_list = [(model, count) for model, count in ensemble_members_count.items()]

    # Convert the list of ensemble members to an array
    ensemble_members_array = np.array(ensemble_members)

    # save the number of ensemble members
    no_ensemble_members = len(ensemble_members_array[:,0,0,0])

    # Print the dimensions of the ensemble members array
    print("ensemble members array shape", np.shape(ensemble_members_array))

    # Take the equal weighted mean of the ensemble members
    ensemble_mean = ensemble_members_array.mean(axis=0)
    ensemble_mean = ensemble_mean[:, 0, 0]                                                 
                                                       
    # print the types for the time
    print("For the obs time:", type(obs_time))
    print("For the model time:", type(years))
    # print("obs time", obs_time)
    # print("model time", years)

    # Convert the type of years
    years = years.astype(str)

    # convert the data type of years to dtype='datetime64[Y]
    years = years.astype("datetime64[Y]")

    # set model time to years
    model_time = years

    # constrain obs time
    obs_time = obs_time[7:]

    # print the shape of the obs nao anoms
    print("obs nao anoms shape", np.shape(obs_nao_anom))

    # # constrain obs nao anom
    obs_nao_anom = obs_nao_anom[7:]

    # constrain model time
    model_time = model_time[:-2]

    # constrain ensemble mean
    ensemble_mean = ensemble_mean[:-2]                                                   
                                                       
    print("model time shape", np.shape(model_time))
    # print("model time", model_time)


    # print the types for the time
    print("for the obs nao anoms:", type(obs_nao_anom))
    print("for the model nao anoms:", type(ensemble_mean))
    print("obs nao anoms shape", np.shape(obs_nao_anom))
    print("model nao anoms shape", np.shape(ensemble_mean))             
                                                
    # calculate the ACC (short and long) for the lagged grand
    # ensemble mean
    acc_score_short_lagged, _ = pearsonr_score(obs_nao_anom, ensemble_mean, model_time,
                                               obs_time, "1968-01-01", "2010-12-31")
    acc_score_long_lagged, _ = pearsonr_score(obs_nao_anom, ensemble_mean, model_time,
                                              obs_time, "1968-01-01", "2016-12-31")

    # Now use these ACC scores to calculate the RPC scores
    # For the short and long period
    rpc_short_lagged = calculate_rpc_time(acc_score_short_lagged, ensemble_members_array,
                                          model_time, "1968-01-01", "2010-12-31")
    rpc_long_lagged = calculate_rpc_time(acc_score_long_lagged, ensemble_members_array,
                                         model_time, "1968-01-01", "2016-12-31")

    # Now use the RPC scores to calculate the RPS
    # To be used in the variance adjustment
    rps_short_lagged = calculate_rps_time(rpc_short_lagged, obs_nao_anom, ensemble_members_array,
                                          model_time, "1968-01-01", "2010-12-31")
    rps_long_lagged = calculate_rps_time(rpc_long_lagged, obs_nao_anom, ensemble_members_array,
                                         model_time, "1968-01-01", "2016-12-31")

    # print these rpc scores
    print("RPC short lagged", rpc_short_lagged)
    print("RPC long lagged", rpc_long_lagged)

    # print these rps scores
    print("RPS short lagged", rps_short_lagged)
    print("RPS long lagged", rps_long_lagged)

    # apply the variance adjustment (via RPS scaling) to the
    # lagged grand ensemble mean
    lagged_adjusted_ensemble_mean_short, lagged_adjusted_ensemble_mean_long = adjust_variance(ensemble_mean,
                                                                                              rps_short_lagged,
                                                                                              rps_long_lagged)

    # Calculate the ACC scores for the lagged adjusted ensemble mean
    # for the short period and the long period
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_short,
                                                    model_time, obs_time, "1968-01-01", "2010-12-31")
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_long,
                                                  model_time, obs_time, "1968-01-01", "2016-12-31")

    # Calculate the 5-95% confidence intervals using compute_rmse_confidence_intervals
    conf_interval_lower_short, conf_interval_upper_short = compute_rmse_confidence_intervals(obs_nao_anom,
                                                                                             lagged_adjusted_ensemble_mean_short,
                                                                                             obs_time,
                                                                                             model_time)
    conf_interval_lower_long, conf_interval_upper_long = compute_rmse_confidence_intervals(obs_nao_anom,
                                                                                           lagged_adjusted_ensemble_mean_long,
                                                                                           obs_time,
                                                                                           model_time)

    # plot the RPS adjusted lagged ensemble mean
    # for both the short period RPS adjust
    # and the long period RPS adjust
    # short period:
    ax.plot(model_time, lagged_adjusted_ensemble_mean_short, color="red", label=f"DCPP-A (short RPS)")
    # long period:
    ax.plot(model_time, lagged_adjusted_ensemble_mean_long, color="red", label = f"DCPP-A (long RPS)", alpha=0.3)

    # Also plot the lagged_ensemble_members_time and the lagged_adjusted_ensemble_mean_short
    ax.plot(lagged_ensemble_members_time, lagged_adjusted_ensemble_mean_short_norm_lag, color="blue", label="DCPP-A (normal lag)", alpha=0.3)

    # Calculate the ACC for the short and long periods
    # Using the function pearsonr_score
    # For the lagged ensemble mean
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_short,
                                                    model_time, obs_time, "1968-01-01", "2010-12-31")
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_long,
                                                  model_time, obs_time, "1968-01-01", "2016-12-31")

    # # check the dimensions of the ci's before plotting
    # print("conf interval lower short", np.shape(conf_interval_lower_short))
    # print("conf interval upper short", np.shape(conf_interval_upper_short))
    # print("conf interval lower long", np.shape(conf_interval_lower_long))
    # print("conf interval upper long", np.shape(conf_interval_upper_long))
    # print("lagged ensemble members time", np.shape(lagged_ensemble_members_time))

    # Plot the confidence intervals for the short period
    ax.fill_between(model_time, conf_interval_lower_short, conf_interval_upper_short,
                    color="red", alpha=0.2)
    # for the long period
    # ax.fill_between(model_time, conf_interval_lower_long, conf_interval_upper_long, color="red",
    #                 alpha=0.25)

    # Plot ERA5 data
    ax.plot(obs_time, obs_nao_anom, color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO (hPa)")

    # check if the p-value is les than 0.01
    # Check if the p_values are less than 0.01 and set the text accordingly
    if p_value_short < 0.01 and p_value_long < 0.01:
        p_value_text_short = '< 0.01'
        p_value_text_long = '< 0.01'
    elif p_value_short < 0.01:
        p_value_text_short = '< 0.01'
        p_value_text_long = f'= {p_value_long:.2f}'
    elif p_value_long < 0.01:
        p_value_text_short = f'= {p_value_short:.2f}'
        p_value_text_long = '< 0.01'
    else:
        p_value_text_short = f'= {p_value_short:.2f}'
        p_value_text_long = f'= {p_value_long:.2f}'

    # Set the title with the ACC and RPC scores
    # the title will be formatted like this:
    # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
    ax.set_title(
        f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = {rpc_short_lagged:.2f} ({rpc_long_lagged:.2f}), N = {no_ensemble_members}")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Save the figure
    # In the plots_dir directory
    # with the lag in the filename
    # and the current date
    # and the number of ensemble members#
    fig.savefig(os.path.join(dic.plots_dir,
                             f"nao_ensemble_mean_and_individual_members_mod_lag_{lag}_{no_ensemble_members}_{forecast_range}_{season}_{datetime.now().strftime('%Y-%m-%d')}.png"),
                dpi=300)

    # Show the figure
    plt.show()

# Calculate confidence intervals for the variance adjusted lagged ensemble mean
# Second way - more simple
# Function to calculate the confidence intervals
def calculate_confidence_intervals(ensemble_members_array, lower_bound=5, upper_bound=95):
    """
    Calculate the confidence intervals of an ensemble members array using a simpler
    method that directly computes the percentiles of the ensemble members data.

    Parameters
    ----------
    ensemble_members_array : numpy.ndarray
        The array of ensemble members data.
    lower_bound : int, optional, default: 5
        The lower percentile bound for the confidence interval.
    upper_bound : int, optional, default: 95
        The upper percentile bound for the confidence interval.

    Returns
    -------
    conf_interval_lower : numpy.ndarray
        The lower bound of the confidence interval.
    conf_interval_upper : numpy.ndarray
        The upper bound of the confidence interval.
    """
    conf_interval_lower = np.percentile(ensemble_members_array, lower_bound, axis=0)
    conf_interval_upper = np.percentile(ensemble_members_array, upper_bound, axis=0)
    return conf_interval_lower, conf_interval_upper

# PLotting function for the no lag case
# Copied from multi-model-jasmin functions.py
def plot_ensemble_members_and_mean(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time):
    """
    Plot the ensemble mean of all members from all models and each of the ensemble members.

    Parameters
    ----------
    models : dict
        A dictionary containing a list of models.
    model_times_by_model : dict
        A dictionary containing model times for each model.
    model_nao_anoms_by_model : dict
        A dictionary containing model NAO anomalies for each model.
    obs_nao_anom : numpy.ndarray
        The observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.

    Returns
    -------
    None
    """

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize an empty list to store all ensemble members
    all_ensemble_members = []

    # Plot the ensemble members and calculate the ensemble mean for each model
    ensemble_means = []

    # Initialize a dictionary to store the count of ensemble members for each model
    ensemble_member_counts = {}

    # Iterate over the models
    for model_name in models:
        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        # If the model_name is not in the dictionary, initialize its count to 0
        if model_name not in ensemble_member_counts:
            ensemble_member_counts[model_name] = 0

        # Plot ensemble members
        for member in model_nao_anom:
            # ax.plot(model_time, member, color="grey", alpha=0.1, linewidth=0.5)

            # Add each member to the list of all ensemble members
            all_ensemble_members.append(member)

            # Increment the count of ensemble members for the current model
            ensemble_member_counts[model_name] += 1

        # Calculate and store ensemble mean
        ensemble_means.append(ensemble_mean(model_nao_anom))

    # Convert the ensemble_member_counts dictionary to a list of tuples
    ensemble_member_counts_list = [(model, count) for model, count in ensemble_member_counts.items()]

    # Convert the list of all ensemble members to a NumPy array
    all_ensemble_members_array = np.array(all_ensemble_members)

    # save the number of ensemble members
    no_ensemble_members = len(all_ensemble_members_array[:,0])

    # Remove the extra dimension from the all_ensemble_members_array
    # all_ensemble_members_array shape (158, 54, 1, 1)
    all_ensemble_members_array = np.squeeze(all_ensemble_members_array)

    # Print the dimensions of the all_ensemble_members_array
    print("all_ensemble_members_array shape", np.shape(all_ensemble_members_array))

    # print the all_ensemble_members_array
    print("all_ensemble_members_array", all_ensemble_members_array)

    # Calculate the grand ensemble mean using the new method
    grand_ensemble_mean = np.mean(all_ensemble_members_array, axis=0)

    # print the grand ensemble mean shape
    print("grand ensemble mean shape", np.shape(grand_ensemble_mean))

    # print the grand ensemble mean
    print("grand ensemble mean", grand_ensemble_mean)

    # extract only the first axis of the grand ensemble mean
    # grand_ensemble_mean = grand_ensemble_mean[:, 0, 0]

    # Print the type of the times
    print("obs_time type:", type(obs_time))
    print("model_time type:", type(model_time))
    print("obs_time:", obs_time)
    print("model time:", list(model_times_by_model.values())[0])
    
    # print the shape of the obs time and model time
    print("obs time shape:", np.shape(obs_time))
    print("model time shape:", np.shape(list(model_times_by_model.values())[0]))

    # print the type of the obs nao anoms and the grand ensemble mean
    print("obs nao anoms type:", type(obs_nao_anom))
    print("grand ensemble mean type:", type(grand_ensemble_mean))

    # print the shape of the obs nao anoms and the grand ensemble mean
    print("obs nao anoms shape:", np.shape(obs_nao_anom))
    print("grand ensemble mean shape:", np.shape(grand_ensemble_mean))

    # print the type of the grand ensemble mean
    model_time = list(model_times_by_model.values())[0]

    # print the type of the model time
    print("model time type:", type(model_time))
    # print the shape of the model time
    print("model time shape:", np.shape(model_time))

    # print grand ensemble mean values
    print("grand ensemble mean values:", grand_ensemble_mean)


    # extract the first element of model time
    # first time array for BCC-CSM2-MR
    model_time = model_time[0]

    # print the type of the model time
    print("model time type:", type(model_time))

    # print the shape of the model time
    print("model time shape:", np.shape(model_time))

    # Calculate the ACC score for the \
    # short period using the function pearsonr_score
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, grand_ensemble_mean, model_time, obs_time, "1966-01-01","2010-12-31")

    # Calculate the ACC score for the \
    # long period using the function pearsonr_score
    # long period 1966 - 2019
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, grand_ensemble_mean, model_time, obs_time, "1966-01-01","2018-12-31")

    # Calculate the RPC score for the short period
    # using the function calculate_rpc_time
    # short period 1966 - 2010
    rpc_short = calculate_rpc_time(acc_score_short, all_ensemble_members_array, model_time, "1966-01-01","2010-12-31")

    # Calculate the RPC score for the long period
    # using the function calculate_rpc_time
    # long period 1966 - 2019
    rpc_long = calculate_rpc_time(acc_score_long, all_ensemble_members_array, model_time, "1966-01-01","2018-12-31")

    # Calculate the 5-95% confidence intervals using the two functions options
    # First calculate_confidence_intervals_sd
    # Then calculate_confidence_intervals
    conf_interval_lower, conf_interval_upper = calculate_confidence_intervals(all_ensemble_members_array)

    # print model time and grand ensemble mean shapes
    print("model time shape before plot", np.shape(model_time))
    print("grand ensemble mean shape before plot", np.shape(grand_ensemble_mean))

    # print the model time and grand ensemble mean
    print("model time before plot", model_time)

    # Plot the grand ensemble mean with the ACC score in the legend
    ax.plot(model_time, grand_ensemble_mean, color="red", label=f"DCPP-A")

    # Plot the 5-95% confidence intervals
    # different shading for the two different time periods
    # short period 1966 - 2010
    ax.fill_between(model_time[:-9], conf_interval_lower[:-9], conf_interval_upper[:-9], color="red", alpha=0.3)
    # for period 2010 - 2019
    ax.fill_between(model_time[-10:], conf_interval_lower[-10:], conf_interval_upper[-10:], color="red", alpha=0.2)

    # print obs time and obs nao anoms shapes
    print("obs time shape before plot", np.shape(obs_time))
    print("obs nao anoms shape before plot", np.shape(obs_nao_anom))

    # print obs time and obs nao anoms
    print("obs time before plot", obs_time)

    # Plot ERA5 data
    ax.plot(obs_time[2:], obs_nao_anom[2:], color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO anomalies (hPa)")

    # check if the p-value is les than 0.01
    # Check if the p_values are less than 0.01 and set the text accordingly
    if p_value_short < 0.01 and p_value_long < 0.01:
        p_value_text_short = '< 0.01'
        p_value_text_long = '< 0.01'
    elif p_value_short < 0.01:
        p_value_text_short = '< 0.01'
        p_value_text_long = f'= {p_value_long:.2f}'
    elif p_value_long < 0.01:
        p_value_text_short = f'= {p_value_short:.2f}'
        p_value_text_long = '< 0.01'
    else:
        p_value_text_short = f'= {p_value_short:.2f}'
        p_value_text_long = f'= {p_value_long:.2f}'
    
    # Set the title with the ACC and RPC scores
    # the title will be formatted like this:
    # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
    ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = +{rpc_short:.2f} (+{rpc_long:.2f}), N = {no_ensemble_members}")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Save the figure
    # In the plots_dir directory
    #fig.savefig(os.path.join(plots_dir, "nao_ensemble_mean_and_individual_members.png"), dpi=300)
    # include the number of ensemble members in the filename
    # and the current date
    fig.savefig(os.path.join(plots_dir, f"nao_ensemble_mean_and_individual_members_{no_ensemble_members}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    # Show the figure
    plt.show()

# Copied function from multi-model-jasmin for ensemble members and lagged adjusted mean
# Define a function for plotting the model subplots for the raw data
# TODO -- SORT OUT THIS FUNCTION AND ITS ASSOCIATED FUNCTIONS
    # -RPC SCORES TOO LOW
    # -EXTEND PERIOD FOR 2010-2019
    # IS IS SCALED TO THE RIGHT MAGNITUDE?
    # EMAIL DOUG POTENTIALLY
def plot_ensemble_members_and_lagged_adjusted_mean(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, lag=4):
    """
    Plot the ensemble mean of all members from all models and each of the ensemble members, with lagged and adjusted variance applied to the grand ensemble mean.

    Parameters
    ----------
    models : dict
        A dictionary containing a list of models.
    model_times_by_model : dict
        A dictionary containing model times for each model.
    model_nao_anoms_by_model : dict
        A dictionary containing model NAO anomalies for each model.
    obs_nao_anom : numpy.ndarray
        The observed NAO anomalies time series.
    obs_time : numpy.ndarray
        The observed time array.
    lag : int, optional, default: 4
        The number of years to lag the grand ensemble mean by.

    Returns
    -------
    None
    """

    # Create a figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize an empty list to store all ensemble members
    all_ensemble_members = []

    # Plot the ensemble members and calculate the ensemble mean for each model
    ensemble_means = []

    # Initialize a dictionary to store the count of ensemble members for each model
    ensemble_member_counts = {}

    # Iterate over the models
    for model_name in models:
        model_time = model_times_by_model[model_name]
        model_nao_anom = model_nao_anoms_by_model[model_name]

        # If the model_name is not in the dictionary, initialize its count to 0
        if model_name not in ensemble_member_counts:
            ensemble_member_counts[model_name] = 0

        # Plot ensemble members
        for member in model_nao_anom:
            #ax.plot(model_time, member, color="grey", alpha=0.1, linewidth=0.5)

            # Add each member to the list of all ensemble members
            all_ensemble_members.append(member)

            # Increment the count of ensemble members for the current model
            ensemble_member_counts[model_name] += 1

        # Calculate and store ensemble mean
        ensemble_means.append(ensemble_mean(model_nao_anom))

    # Convert the ensemble_member_counts dictionary to a list of tuples
    ensemble_member_counts_list = [(model, count) for model, count in ensemble_member_counts.items()]

    # Convert the list of all ensemble members to a NumPy array
    all_ensemble_members_array = np.array(all_ensemble_members)

    # Remove the extra dimension from the all_ensemble_members_array
    # all_ensemble_members_array shape (158, 54, 1, 1)
    all_ensemble_members_array = np.squeeze(all_ensemble_members_array)

    # print the dimensions of all ensemble members array
    print("shape of all ensemble members array", np.shape(all_ensemble_members_array))

    # Create an array for the nolag
    all_ensemble_members_mean_nolag = np.mean(all_ensemble_members_array, axis=0)

    # Set up the model time
    model_time = list(model_times_by_model.values())[0]

    # Extract the first element of model time
    # first time array for BCC-CSM2-MR
    model_time = model_time[0]

    # Call the lag_ensemble function from the NAO_matching.py file
    lagged_ensemble_members_array, lagged_ensemble_members_time = lag_ensemble(all_ensemble_members_array, model_time, lag=lag)

    # Calculate the NAO index for the full lagged ensemble
    lagged_ensemble_mean = np.mean(lagged_ensemble_members_array, axis=0)

    # Extract the number of ensemble members
    no_ensemble_members = lagged_ensemble_members_array.shape[0]


    # print the shape of the lagged ensemble mean
    print("shape of lagged ens mean", np.shape(lagged_ensemble_members_array))
    # also print the model times and its shape
    print("model time", (model_time))

    # # check the time output from this function
    # print("shape of model_time_lagged", np.shape(model_time_lagged))
    # print("model_time_lagged", model_time_lagged)
    # print("shape of lagged_grand_ensemble_mean", np.shape(lagged_grand_ensemble_mean))
    # print("lagged_grand_ensemble_mean", lagged_grand_ensemble_mean)

    # check the dimensions of lagged ensemble members time
    print("shape of lagged_ensemble_members_time", np.shape(lagged_ensemble_members_time))
    print("lagged_ensemble_members_time", lagged_ensemble_members_time)

    # calculate the ACC (short and long) for the lagged grand 
    # ensemble mean
    acc_score_short_lagged, _ = pearsonr_score(obs_nao_anom, lagged_ensemble_mean, lagged_ensemble_members_time, obs_time, "1969-01-01","2010-12-31")
    acc_score_long_lagged, _ = pearsonr_score(obs_nao_anom, lagged_ensemble_mean, lagged_ensemble_members_time, obs_time, "1969-01-01","2018-12-31")

    # Now use these ACC scores to calculate the RPC scores
    # For the short and long period
    rpc_short_lagged = calculate_rpc_time(acc_score_short_lagged, lagged_ensemble_members_array, lagged_ensemble_members_time, "1969-01-01","2010-12-31")
    rpc_long_lagged = calculate_rpc_time(acc_score_long_lagged, lagged_ensemble_members_array, lagged_ensemble_members_time, "1969-01-01","2018-12-31")

    # Now use the RPC scores to calculate the RPS
    # To be used in the variance adjustment
    rps_short_lagged = calculate_rps_time(rpc_short_lagged, obs_nao_anom, lagged_ensemble_members_array, lagged_ensemble_members_time, "1969-01-01","2010-12-31")
    rps_long_lagged = calculate_rps_time(rpc_long_lagged, obs_nao_anom, lagged_ensemble_members_array, lagged_ensemble_members_time, "1969-01-01","2018-12-31")

    # print these rpc scores
    print("RPC short lagged", rpc_short_lagged)
    print("RPC long lagged", rpc_long_lagged)

    # print these rps scores
    print("RPS short lagged", rps_short_lagged)
    print("RPS long lagged", rps_long_lagged)

    # apply the variance adjustment (via RPS scaling) to the 
    # lagged grand ensemble mean
    lagged_adjusted_ensemble_mean_short, lagged_adjusted_ensemble_mean_long = adjust_variance(lagged_ensemble_mean, rps_short_lagged, rps_long_lagged)

    # Also apply the adjustment (via RPS scaling) to the ensemble mean
    # NO LAGGING
    # For explanation purposes
    adjusted_ensemble_mean_short_nolag, adjusted_ensemble_mean_long_nolag = adjust_variance(all_ensemble_members_mean_nolag, rps_short_lagged, rps_long_lagged)

    # Print the shape and values of this to check whether it is realistic
    print("lagged adjusted ensemble mean short", np.shape(lagged_adjusted_ensemble_mean_short))
    print("lagged adjusted ensemble mean short", lagged_adjusted_ensemble_mean_short)
    print("lagged adjusted ensemble mean long", np.shape(lagged_adjusted_ensemble_mean_long))
    print("lagged adjusted ensemble mean long", lagged_adjusted_ensemble_mean_long)
    
    # Calculate the ACC scores for the lagged adjusted ensemble mean
    # for the short period and the long period
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_short, lagged_ensemble_members_time, obs_time, "1969-01-01","2010-12-31")
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_long, lagged_ensemble_members_time, obs_time, "1969-01-01","2018-12-31")

    # Calculate the 5-95% confidence intervals using compute_rmse_confidence_intervals
    conf_interval_lower_short, conf_interval_upper_short = compute_rmse_confidence_intervals(obs_nao_anom, lagged_adjusted_ensemble_mean_short, obs_time, lagged_ensemble_members_time)
    conf_interval_lower_long, conf_interval_upper_long = compute_rmse_confidence_intervals(obs_nao_anom, lagged_adjusted_ensemble_mean_long, obs_time, lagged_ensemble_members_time)

    # plot the RPS adjusted nolag ensemble mean
    # ax.plot(model_time, adjusted_ensemble_mean_short_nolag, color="red", alpha=0.8, linewidth=0.8)

    # plot the RPS adjusted lagged ensemble mean
    # for both the short period RPS adjust
    # and the long period RPS adjust
    # short period:
    ax.plot(lagged_ensemble_members_time, lagged_adjusted_ensemble_mean_short, color="red", label=f"DCPP-A (short RPS)")  
    # long period:
    ax.plot(lagged_ensemble_members_time, lagged_adjusted_ensemble_mean_long, color="red", alpha=0.3, label=f"DCPP-A (long RPS)")

    # ----TESTING----
    # try plotting the each individual lagged member
    # as thin grey lines
    # for member in lagged_ensemble_members_array:
        # ax.plot(lagged_ensemble_members_time, member, color="grey", alpha=0.1, linewidth=0.5)
    
    # Calculate the ACC for the short and long periods
    # Using the function pearsonr_score
    # For the lagged ensemble mean
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_short, lagged_ensemble_members_time, obs_time, "1969-01-01","2010-12-31")
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_long, lagged_ensemble_members_time, obs_time, "1969-01-01","2018-12-31")
    
    # check the dimensions of the ci's before plotting
    print("conf interval lower short", np.shape(conf_interval_lower_short))
    print("conf interval upper short", np.shape(conf_interval_upper_short))
    print("conf interval lower long", np.shape(conf_interval_lower_long))
    print("conf interval upper long", np.shape(conf_interval_upper_long))
    print("lagged ensemble members time", np.shape(lagged_ensemble_members_time))

    # Plot the confidence intervals for the short period
    ax.fill_between(lagged_ensemble_members_time, conf_interval_lower_short, conf_interval_upper_short, color="red", alpha=0.2)
    # for the long period
    #ax.fill_between(lagged_ensemble_members_time, conf_interval_lower_long, conf_interval_upper_long, color="red", alpha=0.25)

    # Plot ERA5 data
    ax.plot(obs_time[6:-4], obs_nao_anom[6:-4], color="black", label="ERA5")

    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlim([np.datetime64("1960"), np.datetime64("2020")])
    ax.set_ylim([-10, 10])
    ax.set_xlabel("Year")
    ax.set_ylabel("NAO (hPa)")

    # check if the p-value is les than 0.01
    # Check if the p_values are less than 0.01 and set the text accordingly
    if p_value_short < 0.01 and p_value_long < 0.01:
        p_value_text_short = '< 0.01'
        p_value_text_long = '< 0.01'
    elif p_value_short < 0.01:
        p_value_text_short = '< 0.01'
        p_value_text_long = f'= {p_value_long:.2f}'
    elif p_value_long < 0.01:
        p_value_text_short = f'= {p_value_short:.2f}'
        p_value_text_long = '< 0.01'
    else:
        p_value_text_short = f'= {p_value_short:.2f}'
        p_value_text_long = f'= {p_value_long:.2f}'
    
    # Set the title with the ACC and RPC scores
    # the title will be formatted like this:
    # "ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P = {p_value_short} ({p_value_long}), RPC = {rpc_short:.2f} ({rpc_long:.2f}), N = {no_ensemble_members}"
    ax.set_title(f"ACC = +{acc_score_short:.2f} (+{acc_score_long:.2f}), P {p_value_text_short} ({p_value_text_long}), RPC = {rpc_short_lagged:.2f} ({rpc_long_lagged:.2f}), N = {no_ensemble_members}")

    # Add a legend in the bottom right corner
    ax.legend(loc="lower right")

    # Save the figure
    # In the plots_dir directory
    # with the lag in the filename
    # and the current date
    # and the number of ensemble members#
    fig.savefig(os.path.join(plots_dir, f"nao_ensemble_mean_and_individual_members_lag_{lag}_{no_ensemble_members}_{datetime.now().strftime('%Y-%m-%d')}.png"), dpi=300)

    # Show the figure
    plt.show()

    # return lagged_ensemble_members_time, lagged_adjusted_ensemble_mean_short
    return lagged_ensemble_members_time, lagged_adjusted_ensemble_mean_short

# first we start a main function which will parse the arguments from the command line
# these arguments include the forecast range and season
def main():
    """
    This function parses the arguments from the command line
    and then calls the function to load the data
    """

    # create a usage statement for the script
    USAGE_STATEMENT = """python processing-NAO-data.py <forecast_range> <season>"""

    # check if the number of arguments is correct
    if len(sys.argv) != 3:
        print(USAGE_STATEMENT)
        sys.exit()

    # make the plots directory if it doesn't exist
    if not os.path.exists(dic.plots_dir):
        os.makedirs(dic.plots_dir)

    # parse the arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("forecast_range", help="forecast range", type=str)
    parser.add_argument("season", help="season", type=str)
    args = parser.parse_args()

    # print the arguments to the screen
    print("forecast range = ", args.forecast_range)
    print("season = ", args.season)

    # test run with the only model being
    test_model = [ "BCC-CSM2-MR" ]

    # call the function to load the data
    lagged_ensemble_members = load_lagged_ensemble_members(args.forecast_range, args.season, dic.models)

    # # print statements to check the dimensions of the data
    # print("lagged ensemble members", len(lagged_ensemble_members))
    # # print("values in lagged ensemble members", list(lagged_ensemble_members.values())[0])
    # same_init_data = lagged_ensemble_members['BCC-CSM2-MR']['same-init']
    # init_minus_1_data = lagged_ensemble_members['BCC-CSM2-MR']['init-minus-1']
    # init_minus_2_data = lagged_ensemble_members['BCC-CSM2-MR']['init-minus-2']
    # init_minus_3_data = lagged_ensemble_members['BCC-CSM2-MR']['init-minus-3']
    # # print("same init data", np.shape(same_init_data))
    # print("same init data", same_init_data)
    # print("init minus 1 data", init_minus_1_data)
    # print("init minus 2 data", init_minus_2_data)
    # print("init minus 3 data", init_minus_3_data)

    # call the function to process the data
    model_times, model_nao_anoms = process_model_datasets_by_init(lagged_ensemble_members)

    # prin statements to check the dimensions of the data
    # Access the processed data for a specific model and initialization scheme
    init_minus_1_times = model_times['BCC-CSM2-MR']['init-minus-1']
    init_minus_1_nao_anoms = model_nao_anoms['BCC-CSM2-MR']['init-minus-1']

    print("init minus 1 times", np.shape(init_minus_1_times))
    print("init minus 1 nao anoms", np.shape(init_minus_1_nao_anoms))
    print("init minus 1 times", init_minus_1_times)
    print("init minus 1 nao anoms", init_minus_1_nao_anoms)

    combined_model_data = combine_model_data(model_times, model_nao_anoms)

    # print statements to check the dimensions of the data
    print("combined model data", np.shape(combined_model_data))
    print("combined model data", combined_model_data['BCC-CSM2-MR']['time'].values)
    print("combined model data", combined_model_data['BCC-CSM2-MR'].values)
    print("combined model data shape", combined_model_data['BCC-CSM2-MR'].shape)

    # load the observations
    obs = xr.open_dataset(dic.obs_long, chunks={"time": 10})

    # call the function to process the observations
    obs_nao_anom, obs_time = process_observations(obs)

    # extract the model data from the combined model data
    extracted_model_data = extract_model_data(dic.models, combined_model_data)

    # print statements to check the dimensions of the data
    print("extracted model data", np.shape(extracted_model_data))
    print("extracted model data", extracted_model_data)
    print("extracted model data model", extracted_model_data.sel(model='BCC-CSM2-MR').values)

    extracted_model_data_mean = extracted_model_data.mean(dim=['model', 'init_scheme_member', 'lat', 'lon'])
    print("extracted model data mean shape", np.shape(extracted_model_data_mean))
    print("extracted model data mean", extracted_model_data_mean)

    # extract the time coordinates from the model data
    model_times = extracted_model_data_mean['time'].values
    print("model times shape", np.shape(model_times))
    print("model times", model_times)

    # create an array of the lag values
    extracted_model_data_array = np.array(extracted_model_data)
    print("extracted model data array", np.shape(extracted_model_data_array))
    print("extracted model data array", extracted_model_data_array[0,:,0,0,0])

    # # call the function to plot the data
    plot_ensemble_members_and_lagged_adjusted_mean(dic.models, extracted_model_data, obs_nao_anom, obs_time, args.forecast_range, args.season)

# if the script is called from the command line, then run the main function
if __name__ == "__main__":
    main()