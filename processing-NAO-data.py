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

# Third party imports
import xarray as xr
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import datetime

# Import from dictionaries
import dictionaries as dic

# set up the pattern for the directory where the files are stored
NAO_dir = "/work/scratch-nopw/benhutch/psl/"

# Set up a function to load the data
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

    # create an empty dictionary
    lagged_ensemble_members = {}

    # set up the argpath
    arg_path = f"/NAO/years_{forecast_range}/{season}/outputs"

    # loop over the models
    for model in models:

        # create the model path
        model_path = os.path.join(NAO_dir, model, arg_path)

        # set the files
        files = [f for f in os.listdir(model_path) if f.endswith(".nc")]

        # create an empty list
        datasets = [xr.open_dataset(os.path.join(model_path, file), chunks={"time": 10}) for file in files]

        # concatenate the datasets along the ensemble dimension
        lagged_ensemble_members[model] = xr.concat(datasets, dim="ensemble member")

    return lagged_ensemble_members


# Define a function to extract the psl and time variables from the .nc files
def process_ensemble_members(datasets_by_model):
    """
    Processes the ensemble members contained in the datasets_by_model dictionary
    by extracting the desired data, converting units, and setting the time variable's data type.

    Parameters:
    datasets_by_model (dict): Dictionary of datasets grouped by model

    Returns:
    model_times_by_model (dict): Dictionary of model times grouped by model
    model_nao_anoms_by_model (dict): Dictionary of model NAO anomalies grouped by model
    """

    def process_model_dataset(dataset):
        """
        Processes the dataset by extracting the desired data, converting units,
        and setting the time variable's data type.

        Parameters:
        dataset (xarray.Dataset): The input dataset

        Returns:
        numpy.ndarray: The model_time array
        numpy.ndarray: The model_nao_anom array
        """

        # Extract the data for the model
        # Extract the data based on the dataset type
        if "psl" in dataset:
            data_var = "psl"  # For the model dataset
        elif "var151" in dataset:
            data_var = "var151"  # For the observations dataset
        else:
            raise ValueError("Unknown dataset type. Cannot determine data variable.")

        model_data = dataset[data_var]
        model_time = model_data["time"].values

        # Set the type for the time variable
        model_time = model_time.astype("datetime64[Y]")

        # Process the model data from Pa to hPa
        if len(model_data.dims) == 4:
            model_nao_anom = model_data[:, :, 0, 0] / 100
        elif len(model_data.dims) == 3:
            model_nao_anom = model_data[:, 0, 0] / 100
        else:
            raise ValueError("Unexpected number of dimensions in the dataset.")

        return model_time, model_nao_anom

    # Create dictionaries to store the processed data
    model_times_by_model = {}
    model_nao_anoms_by_model = {}

    # Process each model's dataset
    for model, dataset in datasets_by_model.items():
        model_times, model_nao_anoms = process_model_dataset(dataset)
        model_times_by_model[model] = model_times
        model_nao_anoms_by_model[model] = model_nao_anoms

    return model_times_by_model, model_nao_anoms_by_model


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

    # debugging for NAO matching
    # print("model times", model_times)
    # print("model times shape", np.shape(model_times))
    # print("model times type", type(model_times))

    # Analyze dimensions of model_times and obs_times
    model_start_index = np.where(model_times == start_date)[0][0]
    model_end_index = np.where(model_times <= end_date)[0][-1]
    obs_start_index = np.where(obs_times >= start_date)[0][0]
    obs_end_index = np.where(obs_times <= end_date)[0][-1]

    # Filter the time series based on the analyzed dimensions
    filtered_time_series1 = time_series1[obs_start_index:obs_end_index+1]
    filtered_time_series2 = time_series2[model_start_index:model_end_index+1]

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

    # Convert the input arrays to numpy arrays
    obs = np.array(obs)
    forecast_members = np.array(forecast_members)

    # Convert the start_date and end_date to pandas Timestamp objects
    start_date = pd.Timestamp(start_date)
    end_date = pd.Timestamp(end_date)

    # Find the start and end indices of the time period for the model
    model_start_index = np.where(model_times >= start_date)[0][0]
    model_end_index = np.where(model_times <= end_date)[0][-1]

    # Filter the forecast members based on the start and end indices
    forecast_members = forecast_members[:, model_start_index:model_end_index+1]

    # Calculate the total variance of the observations
    variance_obs = np.std(obs)

    # Calculate the total variance of the forecast members
    variance_forecast_members = np.std(forecast_members)

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

    # print the obs time and model time
    print("shape of obs time", np.shape(obs_time))
    print("shape of model time", np.shape(model_time_lagged))
    print("obs_time", obs_time)
    print("model time", model_time_lagged)

    # Match the years in obs_time and model_time_lagged
    common_years = np.intersect1d(obs_time, model_time_lagged)

    # Match the years in obs_time and model_time_lagged
    common_years = np.intersect1d(obs_time, model_time_lagged)

    # Find the indices of the common years in both arrays
    obs_indices = np.where(np.isin(obs_time, common_years))[0]
    model_indices = np.where(np.isin(model_time_lagged, common_years))[0]

    print("model indices", model_indices)

    # Create new arrays with the corresponding values for the common years
    obs_nao_anoms_matched = obs_nao_anoms[obs_indices].values
    adjusted_lagged_model_nao_anoms_matched = adjusted_lagged_model_nao_anoms[model_indices]

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

# Define a plotting function that will plot the variance adjusted lag data
def plot_ensemble_members_and_lagged_adjusted_mean(models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom,
                                                   obs_time, forecast_range, season, lag=4):
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

    # Calculate the NAO index for the full lagged ensemble
    lagged_ensemble_mean = np.mean(all_ensemble_members_array, axis=0)

    # Extract the number of ensemble members
    no_ensemble_members = all_ensemble_members_array.shape[0]


    # calculate the ACC (short and long) for the lagged grand
    # ensemble mean
    acc_score_short_lagged, _ = pearsonr_score(obs_nao_anom, lagged_ensemble_mean, list(model_times_by_model.values())[0],
                                               obs_time, "1969-01-01", "2010-12-31")
    acc_score_long_lagged, _ = pearsonr_score(obs_nao_anom, lagged_ensemble_mean, list(model_times_by_model.values())[0],
                                              obs_time, "1969-01-01", "2019-12-31")

    # Now use these ACC scores to calculate the RPC scores
    # For the short and long period
    rpc_short_lagged = calculate_rpc_time(acc_score_short_lagged, all_ensemble_members_array,
                                          list(model_times_by_model.values())[0], "1960-01-01", "2010-12-31")
    rpc_long_lagged = calculate_rpc_time(acc_score_long_lagged, all_ensemble_members_array,
                                         list(model_times_by_model.values())[0], "1960-01-01", "2019-12-31")

    # Now use the RPC scores to calculate the RPS
    # To be used in the variance adjustment
    rps_short_lagged = calculate_rps_time(rpc_short_lagged, obs_nao_anom, all_ensemble_members_array,
                                          list(model_times_by_model.values())[0], "1960-01-01", "2010-12-31")
    rps_long_lagged = calculate_rps_time(rpc_long_lagged, obs_nao_anom, all_ensemble_members_array,
                                         list(model_times_by_model.values())[0], "1960-01-01", "2019-12-31")

    # print these rpc scores
    print("RPC short lagged", rpc_short_lagged)
    print("RPC long lagged", rpc_long_lagged)

    # print these rps scores
    print("RPS short lagged", rps_short_lagged)
    print("RPS long lagged", rps_long_lagged)

    # apply the variance adjustment (via RPS scaling) to the
    # lagged grand ensemble mean
    lagged_adjusted_ensemble_mean_short, lagged_adjusted_ensemble_mean_long = adjust_variance(lagged_ensemble_mean,
                                                                                              rps_short_lagged,
                                                                                              rps_long_lagged)

    # Calculate the ACC scores for the lagged adjusted ensemble mean
    # for the short period and the long period
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_short,
                                                    list(model_times_by_model.values())[0], obs_time, "1960-01-01", "2010-12-31")
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_long,
                                                  list(model_times_by_model.values())[0], obs_time, "1960-01-01", "2019-12-31")

    # Calculate the 5-95% confidence intervals using compute_rmse_confidence_intervals
    conf_interval_lower_short, conf_interval_upper_short = compute_rmse_confidence_intervals(obs_nao_anom,
                                                                                             lagged_adjusted_ensemble_mean_short,
                                                                                             obs_time,
                                                                                             list(model_times_by_model.values())[0])
    conf_interval_lower_long, conf_interval_upper_long = compute_rmse_confidence_intervals(obs_nao_anom,
                                                                                           lagged_adjusted_ensemble_mean_long,
                                                                                           obs_time,
                                                                                           list(model_times_by_model.values())[0])

    # plot the RPS adjusted lagged ensemble mean
    # for both the short period RPS adjust
    # and the long period RPS adjust
    # short period:
    ax.plot(list(model_times_by_model.values())[0], lagged_adjusted_ensemble_mean_short, color="red", label=f"DCPP-A")
    # long period:
    ax.plot(list(model_times_by_model.values())[0], lagged_adjusted_ensemble_mean_long, color="red")

    # Calculate the ACC for the short and long periods
    # Using the function pearsonr_score
    # For the lagged ensemble mean
    acc_score_short, p_value_short = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_short,
                                                    list(model_times_by_model.values())[0], obs_time, "1969-01-01", "2010-12-31")
    acc_score_long, p_value_long = pearsonr_score(obs_nao_anom, lagged_adjusted_ensemble_mean_long,
                                                  list(model_times_by_model.values())[0], obs_time, "1969-01-01", "2019-12-31")

    # # check the dimensions of the ci's before plotting
    # print("conf interval lower short", np.shape(conf_interval_lower_short))
    # print("conf interval upper short", np.shape(conf_interval_upper_short))
    # print("conf interval lower long", np.shape(conf_interval_lower_long))
    # print("conf interval upper long", np.shape(conf_interval_upper_long))
    # print("lagged ensemble members time", np.shape(lagged_ensemble_members_time))

    # Plot the confidence intervals for the short period
    ax.fill_between(list(model_times_by_model.values())[0], conf_interval_lower_short, conf_interval_upper_short,
                    color="red", alpha=0.2)
    # for the long period
    ax.fill_between(list(model_times_by_model.values())[0], conf_interval_lower_long, conf_interval_upper_long, color="red",
                    alpha=0.25)

    # Plot ERA5 data
    ax.plot(obs_time[2:], obs_nao_anom[2:], color="black", label="ERA5")

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


# first we start a main function which will parse the arguments from the command line
# these arguments include the forecast range and season
def main():
    """
    This function parses the arguments from the command line
    and then calls the function to load the data
    """

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
    test_model = "BCC-CSM2-MR"

    # call the function to load the data
    lagged_ensemble_members = load_lagged_ensemble_members(args.forecast_range, args.season, test_model)

    # call the function to process the data
    model_times_by_model, model_nao_anoms_by_model = process_ensemble_members(lagged_ensemble_members)

    # load the observations
    obs = xr.open_dataset(dic.obs_long, chunks={"time": 10})

    # call the function to process the observations
    obs_nao_anom, obs_time = process_observations(obs)

    # call the function to plot the data
    plot_ensemble_members_and_lagged_adjusted_mean(dic.models, model_times_by_model, model_nao_anoms_by_model, obs_nao_anom, obs_time, args.forecast_range, args.season)