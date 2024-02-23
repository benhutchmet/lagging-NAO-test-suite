#!/usr/bin/env python


"""
alternate_lag_functions.py
==========================

A script which processes the alternate lag data from the processed data (which
contains all forecast years). Loads all of the data into one large array with 
dimensions: (years, total nens, forecast_years, lats, lons). Then calculates
the alternate lag for each forecast range and lag.

Author: Ben Hutchins
Date: January 2024

Usage:
------

    $ python alternate_lag_functions.py <variable> <season> <region>
        <start_year> <end_year> <forecast_range> <lag>

    $ python alternate_lag_functions.py tas DJFM global 1961 2014 2-5 4

Parameters:
-----------

    variable: str
        The variable to load.
        E.g. "tas", "pr", "psl", "rsds"

    season: str
        The season to load.
        E.g. "DJFM", "SON", "JJA", "MAM"

    region: str
        The region to load.
        Default is "global".

    start_year: int
        The start year of the period to load.
        Default is 1961.

    end_year: int
        The end year of the period to load.
        Default is 2014.

    forecast_range: str
        The forecast range to take the alternate lag for.
        E.g. "1-5", "2-6", "3-6"

    lag: int
        The lag to take the alternate lag for.
        Default is 4.

Outputs:
--------

    A .npy file containing the full period array with dimensions:
    (years, total nens, forecast_years, lats, lons).

    A .npy file containing the lagged correlation array with dimensions:
    (years, total nens, lats, lons).
"""

# Functions for processing alternate lag data
# Import local modules
import sys
import os
import glob
import argparse
from time import time
import fnmatch

# Import third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# Import local modules
sys.path.append("/home/users/benhutch/lagging-NAO-test-suite/")

# Import dictionaries
import dictionaries as dicts

# Import the external function
sys.path.append("/home/users/benhutch/skill-maps/python")

# Import the functions
import functions as funcs

# Import the NAO matching functions
sys.path.append("/home/users/benhutch/skill-maps/rose-suite-matching")

# Import the NAO matching functions
import nao_matching_seasons as nms_funcs


# Define a function which loads the data
def load_data(
    variable: str,
    models_list: list,
    season: str,
    start_year: int = 1961,
    end_year: int = 2018,  # Last year of years 2-5
    forecast_range: str = "all_forecast_years",
    region: str = "global",
    base_dir: str = "/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data",
):
    """
    Functions which loads the processed full period data into an array of shape
    (years, total nens, forecast_years, lats, lons).

    Inputs:
    ------

    variable: str
        The variable to load.

    models_list: list
        The list of models to load for the variable.

    season: str
        The season to load.

    start_year: int
        The start year of the period to load.
        Default is 1961.

    end_year: int
        The end year of the period to load.
        Default is 2019.

    forecast_range: str
        The forecast range to load.
        Default is 'all_forecast_years'.

    region: str
        The region to load.
        Default is 'global'.

    base_dir: str
        The base directory to load the data from.
        Default is '/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data'.

    no_forecast_years: int
        The number of forecast years to load.
        Default is 10.

    Output:
    -------

    data: xarray DataArray
        The loaded data.
        With shape (years, total nens, forecast_years, lats, lons).
    """

    if season in ["DJFM", "DJF"]:
        # Hard code full forecast range
        full_forecast_years = 11
    else:
        # Hard code full forecast range
        full_forecast_years = 10

    if forecast_range not in ["1", "2", "3", "4", "5", "6"] and season not in [
        "DJFM",
        "DJF",
        "ONDJFM",
        "NDJFM",
    ]:
        # Set the number of forecast years
        no_forecast_years = 8
    elif forecast_range not in ["1", "2", "3", "4", "5", "6"] and season in [
        "DJFM",
        "DJF",
        "ONDJFM",
        "NDJFM",
    ]:
        # Set the number of forecast years
        no_forecast_years = 9
    elif forecast_range in ["1", "2", "3", "4", "5", "6"] and season not in [
        "DJFM",
        "DJF",
        "ONDJFM",
        "NDJFM",
    ]:
        # Set the number of forecast years
        no_forecast_years = 8
    elif forecast_range in ["1", "2", "3", "4", "5", "6"] and season in [
        "DJFM",
        "DJF",
        "ONDJFM",
        "NDJFM",
    ]:
        # Set the number of forecast years
        no_forecast_years = 9
    else:
        print("Forecast range not recognised")

    # extract the forecast range
    if "-" in forecast_range:
        # Extract the forecast range
        forecast_range_list = forecast_range.split("-")

        # Extract the start and end years
        first_year = int(forecast_range_list[0])
        last_year = int(forecast_range_list[1])
    else:
        # Set the first and last year to the start year
        first_year = int(forecast_range)

        # Set the last year to the end year
        last_year = int(forecast_range)

    # If the season is not NDJFM, DJF, DJFM, ONDJFM
    if season not in ["NDJFM", "DJF", "DJFM", "ONDJFM"]:
        # Modify the start year
        first_year = first_year + 1

        # Modify the end year
        last_year = last_year + 1

    # Generate the years array
    years = np.arange(start_year, end_year + 1)

    # Initialise a dictionary to store the list of files for each model
    files_dict = {}

    # Create a dictionary with model keys
    files_dict = {model: [] for model in models_list}

    # Loop over the models
    for model in models_list:
        # Form the directory path
        dir_path = os.path.join(
            base_dir, variable, model, region, forecast_range, season, "outputs"
        )

        # Asser that the directory exists
        assert os.path.isdir(dir_path), f"{dir_path} does not exist."

        # Get the full path of each file in the directory
        # Only want to import the anoms files
        # all-years-DJFM-global-psl_Amon_NorCPM1_dcppA-hindcast_s1961-r1i1p1f1_gn_196110-197112_years_2-5_start_1961_end_2014_anoms.nc
        file_path_list = [
            os.path.join(dir_path, file)
            for file in os.listdir(dir_path)
            if file.endswith(f"_start_{start_year}_end_{end_year}_anoms.nc")
        ]

        # Append the list of files to the dictionary
        files_dict[model].append(file_path_list)

    # Extract a single file from each files list for each model
    # and check the length of time
    for model in models_list:
        # Extract a single file
        file = files_dict[model][0][0]

        # Modify the file path
        file_path = os.path.join(
            base_dir, variable, model, region, forecast_range, season, "outputs", file
        )

        # Load in the file
        data = xr.open_dataset(file_path, chunks={"time": 10, "lat": 10, "lon": 10})

        # Assert that the model only has lon and lat dimensions
        assert (
            len(data.dims) == 3
        ), f"{model} has more than two dimensions. Check the file: {file_path}"

        # If the model is not MRI-ESM2-0
        if model != "MRI-ESM2-0":
            # Assert that the length of time is equal to the number of forecast years
            assert (
                len(data.time) >= full_forecast_years
            ), f"{model} does not have the correct number of forecast years. Check the file: {file_path}"
        else:
            # Assert that the length of time is equal to the number of forecast years
            assert (
                len(data.time) == 5
            ), f"{model} does not have the correct number of forecast years. Check the file: {file_path}"

    # Initialise total nens
    total_nens = 0

    # Initialise a dictionary to store the total nens for each model
    nens_dict = {}

    # Count the total nens for each model
    for model in models_list:
        # Extract the file list
        file_list = files_dict[model][0]

        # Find all of the unique combinations of r and i
        ens_list_pattern = np.unique([file.split("_")[4] for file in file_list])

        # Split by "-" and find all of the unique combinations of r and i
        ens_list = np.unique([ens.split("-")[1] for ens in ens_list_pattern])

        # Loop over the years
        for year in years:

            if "-" in forecast_range:
                # Assert that files exist for each ensemble member
                assert len(
                    [
                        file
                        for file in file_list
                        if fnmatch.fnmatch(
                            file, f"*s{year}*_years_{first_year}-{last_year}*_anoms.nc"
                        )
                    ]
                ) == len(
                    ens_list
                ), f"{model} does not have files for each ensemble member for year {year}"
            else:
                # Assert that files exist for each ensemble member
                assert len(
                    [
                        file
                        for file in file_list
                        if fnmatch.fnmatch(
                            file, f"*s{year}*_years_{first_year}-{last_year}*_anoms.nc"
                        )
                    ]
                ) == len(
                    ens_list
                ), f"{model} does not have files for each ensemble member for year {year}"
        # Append this list to the dictionary
        nens_dict[model] = ens_list

        # Add to total_nens
        total_nens += len(ens_list)

    # Print the total number of ensemble members
    print(f"Total number of ensemble members: {total_nens}")

    # Print the ensemble members for the models
    # print("Ensemble members for EC-Earth3: ", nens_dict["HadGEM3-GC31-MM"])

    # Extract the lats and lons
    nlats = data.lat.shape[0]
    nlons = data.lon.shape[0]

    # Initialise the data array
    data_arr = np.zeros([len(years), total_nens, no_forecast_years, nlats, nlons])

    # print the shape of the data array
    print(f"Shape of data array: {data_arr.shape}")

    # Initialise a counter for the ensemble members
    ens_counter = 0

    # Loop over the models
    for model in models_list:
        print("Extracting data into array for model: ", model)

        # Extract the file list
        file_list = files_dict[model][0]

        # Loop over the years
        for i, year in enumerate(years):
            print("Extracting data for year: ", year, "for model: ", model)
            print("Year index: ", i, "year: ", year)

            # Initialise the previous ensemble member
            prev_ens_member = 0 + ens_counter

            # Print
            print("resetting prev_ens_member from:", prev_ens_member, "to 0")

            # Loop over the ensemble members
            for j in range(len(nens_dict[model])):
                print(
                    "Extracting data for ensemble member: ", j + 1, "for model: ", model
                )

                # Extract the file containing f"s{year}" and f"r{j+1}i2"
                print(
                    "Only extracting data for ensemble member: ",
                    j + 1,
                    "for model: ",
                    model,
                    "for i1",
                )

                # Extract the unique ensemble member
                ens_member = nens_dict[model][j]

                # Print the year and the ensemble member
                print(f"Extracting year s{year} and ensemble member {ens_member}")

                # Extract the file containing f"s{year}"
                file = [
                    file
                    for file in file_list
                    if f"s{year}" in file and f"{ens_member}" in file
                ][0]

                # Extract the base file name
                base_file = os.path.basename(file)

                # Extract the init year e.g. "s1961" from the file name
                init_year_pattern = base_file.split("_")[4]

                # Print the init year pattern
                print("init_year_pattern: ", init_year_pattern)

                # Extract the init year e.g. "1961" from the init year pattern
                init_year = int(init_year_pattern.split("-")[0][1:])

                # Load the file using xarray
                data = xr.open_dataset(file, chunks={"time": 10, "lat": 10, "lon": 10})

                # If lagging necessary, then will have to modify
                # If forecast range does not contain a hyphen
                if forecast_range == "1" and season in [
                    "DJFM",
                    "DJF",
                    "NDJFM",
                    "ONDJFM",
                ]:
                    print("forecast range does not contain a hyphen")
                    print("Extracting first season: ", season)

                    # Set up the first year
                    first_year = init_year

                    # Set up the last year
                    last_year = init_year
                elif forecast_range == "1" and season not in [
                    "DJFM",
                    "DJF",
                    "NDJFM",
                    "ONDJFM",
                ]:

                    # Set up the first year
                    first_year = init_year + 1

                    # Set up the last year
                    last_year = init_year + 1
                elif forecast_range == "2" and season in [
                    "DJFM",
                    "DJF",
                    "NDJFM",
                    "ONDJFM",
                ]:
                    print("forecast range does not contain a hyphen")
                    print("Extracting second season: ", season)

                    # Set up the first year
                    first_year = init_year + 1

                    # Set up the last year
                    last_year = init_year + 9
                elif forecast_range == "2" and season not in [
                    "DJFM",
                    "DJF",
                    "NDJFM",
                    "ONDJFM",
                ]:
                    # Set up the first year
                    first_year = init_year + 2

                    # Set up the last year
                    last_year = init_year + 9
                elif season not in ["DJFM", "DJF", "NDJFM", "ONDJFM"]:
                    # Set up the first year
                    first_year = init_year + 2  # e.g. for s1961 would be 1963

                    # Set up the last year
                    last_year = init_year + 9  # e.g. for s1961 would be 1970
                else:
                    # Set up the first year
                    first_year = init_year + 1  # e.g. for s1961 would be 1962

                    # Set up the last year
                    last_year = init_year + 9  # e.g. for s1961 would be 1970

                # Print the period we are constraining the data to
                print(
                    f"Constraining the data from {first_year}-01-01 to {last_year + 1}-01-01"
                )

                # Constrain the data from init-year-01-01 to init-year+10-12-31
                data = data.sel(
                    time=slice(f"{first_year}-01-01", f"{last_year + 1}-01-01")
                )

                # Extract the data for the variable
                data = data[variable]

                # If the forecast range does not contain a hyphen
                # if "-" not in forecast_range:
                #     # Assert that there is only one forecast year
                #     assert (
                #         len(data.time) == 1
                #     ), f"forecast range does not contain a hyphen, but there is more than one forecast year for model: {model}, year: {year}, ensemble member: {j+1}"

                # Print the year and the ensemble member
                print("Year: ", year, "ensemble member: ", j + 1)
                print("ens counter: ", ens_counter)
                print("Appending i1 to index ", ens_counter + j)

                # Store the data in the array
                data_arr[i, ens_counter + j, :, :, :] = data

        # Increment the ensemble counter with the number of ensemble members
        # For the model
        ens_counter += len(nens_dict[model])

    # Print the shape of the data array
    print("Shape of data array: ", data_arr.shape)

    # Return the data array
    return data_arr


# TODO: Fix this function for the alternate lag 1 and 2 year case
# If the second year has some skill
# Write a function to calculate the lagged correlation
def alternate_lag(
    data: np.array, forecast_range: str, years: np.array, lag: int = 4
) -> np.array:
    """
    Calculate the lagged correlation for a given forecast range and lag.

    Parameters
    ----------
    data : np.array
        Array of data to calculate the lagged correlation for.
        Should have dimensions (num_years, nens, no_forecast_years, no_lats, no_lons).
    forecast_range : str
        The forecast range to calculate the lagged correlation for.
        This should be in the format "x-y" where x and y are integers.
    years : np.array
        Array of years to calculate the lagged correlation for.
        Should have dimensions (num_years,).
    lag : int
        The lag to calculate the lagged correlation for.
        The default is 4.

    Returns
    -------
    lagged_correlation : np.array
        Array of lagged correlation values with dimensions (num_years, nens, no_lats, no_lons).
    """

    # assert that forecast_range is not 1
    assert forecast_range != "1", "forecast_range should not be 1"

    # Set up the start index
    start_index = 2

    # Assert that the forecast range is in the correct format
    # assert (
    #     "-" in forecast_range
    # ), "forecast_range should be in the format 'x-y' where x and y are integers"

    if "-" in forecast_range:
        # Extract the forecast range
        forecast_range_list = forecast_range.split("-")

        # Extract the start and end years
        start_year = int(forecast_range_list[0])
        end_year = int(forecast_range_list[1])
    else:
        # Set the start year to the forecast range
        start_year = int(forecast_range)

        # Set the end year to the forecast range
        end_year = int(forecast_range)

    # Assert that end year is 6 or less than start year
    # assert (
    #     end_year <= 6
    # ), "end_year should be 6 or less to be valid for four year lagged correlation"

    if "-" in forecast_range:
        # Assert that end year is greater than start year
        assert end_year > start_year, "end_year should be greater than start_year"

    # Set up the number of lagged years
    no_lagged_years = data.shape[0] - lag + 1

    print("no_lagged_years: ", no_lagged_years)

    # Extract the lagged years
    lagged_years = years[lag - 1 :]

    # Print the first and last year of the lagged years
    print("First lagged year: ", lagged_years[0])
    print("Last lagged year: ", lagged_years[-1])

    # Create an empty array to store the lagged correlation
    lagged_correlation = np.zeros(
        [no_lagged_years, data.shape[1] * lag, data.shape[3], data.shape[4]]
    )

    # Loop over the years
    for i in range(no_lagged_years):
        print("Processing data for lag year index: ", i)
        # Loop over the lag
        for j in range(lag):
            print("Processing data for lag index: ", j)
            # Extract the data for the lagged year
            lagged_year_data = data[i + (lag - 1) - j, :, :, :, :]

            # Print which data we are extracting
            print("Extracting data for year index: ", (lag - 1) - j)
            print("Extracting data for year: ", years[i + (lag - 1) - j])
            print("For lag index: ", j)

            # Loop over the ensemble members
            for k in range(data.shape[1]):
                # Extract the data for the ensemble member
                ensemble_member_data = lagged_year_data[k, :, :, :]

                # print the years which we are taking the mean over
                print("start year: ", start_year + j, " end year: ", end_year + j)
                print(
                    f"starting at index {start_year - start_index + j}"
                    f"stoppping at index {end_year - start_index + j + 1}"
                )

                if forecast_range != "2-9" and forecast_range != "2":
                    # Take the mean over the forecast years
                    ensemble_member_data_mean = np.mean(
                        ensemble_member_data[
                            start_year
                            - start_index
                            + j : end_year
                            - start_index
                            + j
                            + 1,
                            :,
                            :,
                        ],
                        axis=0,
                    )
                elif forecast_range == "2":
                    print("forecast range is 2")
                    print("Taking the mean over year: ", start_year - start_index + j)
                    print("For lag index: ", j)
                    # Take the mean over the forecast years
                    ensemble_member_data_mean = np.mean(
                        ensemble_member_data[
                            start_year - start_index + j,
                            :,
                            :,
                        ],
                        axis=0,
                    )
                else:
                    # Take the mean over the forecast years
                    ensemble_member_data_mean = np.mean(
                        ensemble_member_data[
                            start_year - start_index : end_year - start_index + 1, :, :
                        ],
                        axis=0,
                    )

                # Print the year index, ensemble member index and lag index
                print(
                    "year index: ", i, " ensemble member index: ", k, " lag index: ", j
                )

                # Print which we are appending to
                print(
                    "Appending to: year index: ",
                    i,
                    " ensemble member index: ",
                    j + k * lag,
                )

                # Append the data to the array
                lagged_correlation[i, j + k * lag, :, :] = ensemble_member_data_mean

    # Return the lagged correlation
    return lagged_correlation


# Define a function to calculate the NAO index for the period specified
# And optionally plot this
def calculate_nao_index(
    season: str,
    forecast_range: str,
    start_year: int = 1961,
    end_year: int = 2014,
    variable: str = "psl",
    models_list: list = dicts.models,
    plot: bool = False,
    lag_var_adjust: bool = True,
    alt_lag: bool = False,
    winter_n_grid: dict = dicts.iceland_grid_corrected,
    winter_s_grid: dict = dicts.azores_grid_corrected,
    summer_n_grid: dict = dicts.snao_north_grid,
    summer_s_grid: dict = dicts.snao_south_grid,
    region: str = "global",
    base_dir: str = "/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data",
):
    """
    Calculate the NAO index for the period specified and optionally plot this.

    Parameters
    ----------

    season: str
        The season to calculate the NAO index for.

    forecast_range: str
        The forecast range to calculate the NAO index for.

    start_year: int
        The start year of the period to calculate the NAO index for.

    end_year: int
        The end year of the period to calculate the NAO index for.

    models_list: list
        The list of models to calculate the NAO index for.

    plot: bool
        Whether to plot the NAO index or not.

    winter_n_grid: dict
        The grid to calculate the NAO index for the winter northern pole.

    winter_s_grid: dict
        The grid to calculate the NAO index for the winter southern pole.

    summer_n_grid: dict
        The grid to calculate the NAO index for the summer northern pole.

    summer_s_grid: dict
        The grid to calculate the NAO index for the summer southern pole.

    region: str
        The region to calculate the NAO index for.
        Default is "global".

    base_dir: str
        The base directory to load the data from.
        Default is '/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data'.

    Returns
    -------

    NAO_index: xarray DataArray
        The NAO index for the period specified.

    """

    # Set up the north and south grid lats to be used
    if season in ["DJFM", "DJF", "ONDJFM", "NDJFM"]:
        # Set up the north and south grid lats
        n_grid = winter_n_grid
        s_grid = winter_s_grid
    else:
        # Set up the north and south grid lats
        n_grid = summer_n_grid
        s_grid = summer_s_grid

    # Extract the lats and lons for the north and south grids
    n_lat1, n_lat2 = n_grid["lat1"], n_grid["lat2"]
    s_lat1, s_lat2 = s_grid["lat1"], s_grid["lat2"]

    # Extract the lats and lons for the north and south grids
    n_lon1, n_lon2 = n_grid["lon1"], n_grid["lon2"]
    s_lon1, s_lon2 = s_grid["lon1"], s_grid["lon2"]

    # Set up the first and last years according to the forecast range
    if forecast_range == "2-9":
        # Set up the raw first and last years
        first_year_align = int(start_year) + 5
        last_year_align = int(end_year) + 5
    elif forecast_range == "2-5":
        # Set up the raw first and last years
        first_year_align = int(start_year) + 3
        last_year_align = int(end_year) + 3
    elif forecast_range == "2-3":
        # Set up the raw first and last years
        first_year_align = int(start_year) + 2
        last_year_align = int(end_year) + 2
    elif forecast_range == "2":
        # Set up the raw first and last years
        first_year_align = int(start_year) + 1
        last_year_align = int(end_year) + 1
    elif forecast_range == "1":
        # Set up the raw first and last years
        first_year_align = int(start_year)
        last_year_align = int(end_year)
    else:
        print("Forecast range not recognised")

    # If the season is not in the winter - i.e. data has not been shifted
    if season not in ["DJFM", "DJF", "ONDJFM", "NDJFM"]:
        # Add 1 to the first and last years
        first_year_align = first_year_align + 1
        last_year_align = last_year_align + 1

    # Set up the common years
    common_years = np.arange(first_year_align, last_year_align + 1)

    # First calculate the observed MSLP anomaly fields
    obs_psl_anomaly = funcs.read_obs(
        variable=variable,
        region=region,
        forecast_range=forecast_range,
        season=season,
        observations_path=nms_funcs.find_obs_path(match_var=variable),
        start_year=1960,
        end_year=2023,
    )

    # Constrain the obs_psl_anomaly to the common years
    obs_psl_anomaly = obs_psl_anomaly.sel(
        time=slice(f"{first_year_align}-01-01", f"{last_year_align}-12-31")
    )

    # Extract the data for the south grid
    obs_psl_anomaly_south = obs_psl_anomaly.sel(
        lat=slice(s_lat1, s_lat2), lon=slice(s_lon1, s_lon2)
    ).mean(dim=["lat", "lon"])

    # Extract the data for the north grid
    obs_psl_anomaly_north = obs_psl_anomaly.sel(
        lat=slice(n_lat1, n_lat2), lon=slice(n_lon1, n_lon2)
    ).mean(dim=["lat", "lon"])

    # Calculate the NAO index
    obs_nao_index = obs_psl_anomaly_south - obs_psl_anomaly_north

    # Loop over the obs_psl_anomaly field and remove any NaNs
    for year in obs_nao_index.time.dt.year.values:
        year_psl_anoms = obs_nao_index.sel(time=f"{year}")

        # If there are any NaNs in the data
        if np.isnan(year_psl_anoms).any():
            print("NaNs found in the data for year: ", year)
            if np.isnan(year_psl_anoms).all():
                print("All NaNs found in the data for year: ", year)
                print("Removing the year: ", year)
                obs_nao_index = obs_nao_index.sel(
                    time=obs_nao_index.time.dt.year != year
                )

    # Print the shape of the obs_nao_index
    print("Shape of obs_nao_index: ", obs_nao_index.shape)

    # Set up an empty list for all of the ensemble member paths
    variant_labels_models = {}

    # Loop over the models
    for model in models_list:
        # Set up the model path
        model_path = os.path.join(
            base_dir, variable, model, region, forecast_range, season, "outputs"
        )

        # Assert that the model path exists
        assert os.path.isdir(model_path), f"{model_path} does not exist."

        # Find the anoms files
        file_path_list = [
            os.path.join(model_path, file)
            for file in os.listdir(model_path)
            if file.endswith(f"_start_{start_year}_end_{end_year}_anoms.nc")
        ]

        # Extract the final string after th "/"
        file_path_list_split = [file.split("/")[-1] for file in file_path_list]

        # Split the final string by "_"
        file_path_list_split = [file.split("_")[4] for file in file_path_list_split]

        # Split this by "-"
        file_path_list_split = [file.split("-")[1] for file in file_path_list_split]

        # Find the unique combinations of r and i
        unique_members_model = np.unique(file_path_list_split)

        # Print the length of the unique members
        print(f"Length of unique members for {model}: ", len(unique_members_model))

        # Append the unique members to the dictionary
        variant_labels_models[model] = unique_members_model

    # Set up a list to store the member files
    member_files = []

    # Loop over the models
    for model in models_list:
        # Loop over the variant label
        for variant_label in variant_labels_models[model]:
            # Set up the variant label files
            variant_label_files = []

            # Loop over the years
            for year in range(start_year, end_year + 1):
                # Find the file for the given model,
                # year and member in the anoms_files list
                file_path = f"{base_dir}/{variable}/{model}/{region}/{forecast_range}/{season}/outputs/*s{year}-{variant_label}*years_{forecast_range}_start_{start_year}_end_{end_year}_anoms.nc"

                # Find the file for the given model,
                # year and member in the anoms_files list
                file = glob.glob(file_path)

                # Assert that there is only one file
                if len(file) != 1:
                    print("More than one file found for: ", file_path)
                    print("Files found: ", file[0].split("/")[-1])

                # Append the file to the variant label files
                variant_label_files.append(file[0])

            # Append the variant label files to the member files
            member_files.append(variant_label_files)

    # Assert that member_files is a list of lists
    assert isinstance(member_files, list), "member_files is not a list"

    # Assert that member_files is a list of lists
    assert isinstance(member_files[0], list), "member_files is not a list of lists"

    # Return the member files
    return member_files


# Write a function for preprocessing the data


# Define the main function
def main():
    # Define the hardcoded variables
    save_dir = "/gws/nopw/j04/canari/users/benhutch/alternate-lag-processed-data"

    # If the save directory does not exist
    if not os.path.isdir(save_dir):
        # Create the directory
        os.mkdir(save_dir)

    # Set up the parser for the CLAs
    parser = argparse.ArgumentParser()

    # Add the arguments
    parser.add_argument("variable", type=str, help="The variable to load.")
    parser.add_argument("season", type=str, help="The season to load.")
    parser.add_argument(
        "region", type=str, help="The region to load.", default="global"
    )
    parser.add_argument(
        "start_year",
        type=int,
        help="The start year of the period to load.",
        default=1961,
    )
    parser.add_argument(
        "end_year", type=int, help="The end year of the period to load.", default=2014
    )
    parser.add_argument(
        "forecast_range",
        type=str,
        help="The forecast range to take the alternate lag for.",
    )
    parser.add_argument(
        "lag", type=int, help="The lag to take the alternate lag for.", default=4
    )

    # Extract the CLAs
    args = parser.parse_args()

    # Extract the variables
    variable = args.variable
    season = args.season
    region = args.region
    start_year = args.start_year
    end_year = args.end_year
    forecast_range = args.forecast_range
    lag = args.lag

    # Print the variables
    print("variable: ", variable)
    print("season: ", season)
    print("region: ", region)
    print("start_year: ", start_year)
    print("end_year: ", end_year)
    print("forecast_range: ", forecast_range)
    print("lag: ", lag)

    # Extract the models for the given variable
    # Assuming that models initialised in November are the same for all variables
    if variable == "tas":
        models_list = dicts.models
        # models_list = dicts.nov_init_models_tas
    elif variable == "sfcWind":
        models_list = dicts.sfcWind_models
        # models_list = dicts.nov_init_models_sfcWind
    elif variable == "psl":
        models_list = dicts.psl_models
        # models_list = dicts.nov_init_models_psl
    elif variable == "rsds":
        models_list = dicts.rsds_models
        # models_list = dicts.nov_init_models_rsds
    elif variable == "pr":
        models_list = dicts.pr_models
    else:
        raise ValueError("variable not recognised")

    # If the forecast range contains a hyphen
    if "-" in forecast_range and "MRI-ESM2-0" in models_list:
        # Remove "MRI-ESM2-0" from the models list
        print(
            "Removing MRI-ESM2-0 from the models list as it only has 5 forecast years"
        )
        models_list = [model for model in models_list if model != "MRI-ESM2-0"]

    # Run the function to load the data
    data = load_data(
        variable=variable,
        models_list=models_list,
        season=season,
        start_year=start_year,
        end_year=end_year,
        forecast_range=forecast_range,
        region=region,
    )

    # Extract the current time
    current_time = time()

    # Set up the filename for saving the array
    filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}_{current_time}.npy"

    # Set up the full path for saving the array
    save_path = os.path.join(save_dir, filename)

    # Save the array
    np.save(save_path, data)

    # # If the forecas range does not contain a hyphen
    # if "-" not in forecast_range:
    #     # Print that we are not calculating the alternate lag
    #     print(
    #         "Not calculating the alternate lag for single year forecast range. Exiting."
    #     )

    #     # Exit the function
    #     return

    # TODO: If there is skill in the second year, then we can calculate the alternate lag
    # For year 1
    # Now process the alternate lag data
    data_alternate_lag = alternate_lag(
        data=data,
        forecast_range=forecast_range,
        years=np.arange(start_year, end_year + 1),
        lag=lag,
    )

    # Set up the lag start year
    lag_start_year = start_year + lag - 1

    # Set up the filename for saving the array
    filename = f"{variable}_{season}_{region}_{lag_start_year}_{end_year}_{forecast_range}_{lag}_{current_time}_alternate_lag.npy"

    # Set up the full path for saving the array
    save_path = os.path.join(save_dir, filename)

    # Save the array
    np.save(save_path, data_alternate_lag)


if __name__ == "__main__":
    # Execute the main function
    main()
