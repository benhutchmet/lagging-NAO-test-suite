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
import argparse
from time import time

# Import third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

# Import local modules
sys.path.append("/home/users/benhutch/skill-maps")

# Import dictionaries
import dictionaries as dicts


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
    no_forecast_years: int = 10,
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

    # Hard code full forecast range
    full_forecast_years = 11

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
        # TODO: Modify this for new processed data
        # /gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/psl/HadGEM3-GC31-MM/global/2-5/DJFM/outputs
        file_path = os.path.join(
            base_dir, variable, model, region, forecast_range, season, "outputs", file
        )

        # Load in the file
        data = xr.open_dataset(file_path, chunks={"time": 10, "lat": 10, "lon": 10})

        # Assert that the model only has lon and lat dimensions
        assert (
            len(data.dims) == 3
        ), f"{model} has more than two dimensions. Check the file: {file_path}"

        # Assert that the length of time is equal to the number of forecast years
        assert (
            len(data.time) == full_forecast_years
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
            # Assert that files exist for each ensemble member
            assert (len([file for file in file_list if f"s{year}" in file])) == len(
                ens_list
            ), f"{model} does not have files for each ensemble member for year {year}"  

        # Append this list to the dictionary
        nens_dict[model] = ens_list

        # Add to total_nens
        total_nens += len(ens_list)

    # Print the total number of ensemble members
    print(f"Total number of ensemble members: {total_nens}")

    # Print the ensemble members for the models
    print("Ensemble members for EC-Earth3: ", nens_dict["HadGEM3-GC31-MM"])

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
                data = xr.open_dataset(
                    file, chunks={"time": 10, "lat": 10, "lon": 10}
                )

                # Set up the final_init_year
                final_init_year = init_year + no_forecast_years - 1

                # Constrain the data from init-year-01-01 to init-year+10-12-31
                data = data.sel(time=slice(f"{init_year}-01-01", f"{final_init_year}-12-30"))

                # Extract the data for the variable
                data = data[variable]

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

    # Assert that the forecast range is in the correct format
    assert (
        "-" in forecast_range
    ), "forecast_range should be in the format 'x-y' where x and y are integers"

    # Extract the forecast range
    forecast_range_list = forecast_range.split("-")

    # Extract the start and end years
    start_year = int(forecast_range_list[0])
    end_year = int(forecast_range_list[1])

    # Assert that end year is 6 or less than start year
    assert (
        end_year <= 6
    ), "end_year should be 6 or less to be valid for four year lagged correlation"

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
                print(f"starting at index {start_year + j - 1}"
                      f"stoppping at index {end_year + j}")

                # Take the mean over the forecast years
                ensemble_member_data_mean = np.mean(
                    ensemble_member_data[start_year + j - 1 : end_year + j, :, :],
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
    if variable == "tas":
        models_list = dicts.tas_models
    elif variable == "sfcWind":
        models_list = dicts.sfcWind_models
    elif variable == "psl":
        models_list = dicts.psl_models
    elif variable == "rsds":
        models_list = dicts.rsds_models
    else:
        raise ValueError("variable not recognised")


    #FIXME: Remove this
    # Define a test models list
    models_list = dicts.psl_models_noCan

    # Run the function to load the data
    data = load_data(
        variable=variable,
        models_list=models_list,
        season=season,
        start_year=start_year,
        end_year=end_year,
        forecast_range=forecast_range,
        region=region,
        no_forecast_years=10,
    )

    # Extract the current time
    current_time = time()

    # Set up the filename for saving the array
    filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}_{current_time}.npy"

    # Set up the full path for saving the array
    save_path = os.path.join(save_dir, filename)

    # Save the array
    np.save(save_path, data)

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
