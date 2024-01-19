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
def load_data(variable: str,
              models_list: list,
              season: str,
              start_year: int = 1961,
              end_year: int = 2018, # Last year of years 2-5
              forecast_range: str = 'all_forecast_years',
              region: str = 'global',
              base_dir: str = '/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data',
              no_forecast_years: int = 11):
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
        Default is 11.

    Output:
    -------

    data: xarray DataArray
        The loaded data.
        With shape (years, total nens, forecast_years, lats, lons).
    """ 

    # Generate the years array
    years = np.arange(start_year, end_year+1)

    # Initialise a dictionary to store the list of files for each model
    files_dict = {}
    
    # Create a dictionary with model keys
    files_dict = {model: [] for model in models_list}

    # Loop over the models
    for model in models_list:
        # Form the directory path
        dir_path = os.path.join(base_dir, variable, model, region,
                                forecast_range, season,
                                "outputs", "anoms")
        
        # Asser that the directory exists
        assert os.path.isdir(dir_path), f"{dir_path} does not exist."

        # Get the full path of each file in the directory
        file_path_list = [os.path.join(dir_path, file) for file in os.listdir(dir_path) if file.endswith(".nc")]

        # Append the list of files to the dictionary
        files_dict[model].append(file_path_list)

    # Extract a single file from each files list for each model
    # and check the length of time
    for model in models_list:
        # Extract a single file
        file = files_dict[model][0][0]

        # Modify the file path
        file_path = os.path.join(base_dir, variable, model, region,
                                forecast_range, season,
                                "outputs", "anoms", file)

        # Load in the file
        data = xr.open_dataset(file_path, chunks={'time': 10,
                                                                     'lat': 10,
                                                                    'lon': 10})
        
        # Print the model and the length of time
        print(f"{model}: {len(data.time.dt.year)} years")

        # Assert that the length of time is correct - i.e. 11 years
        assert len(data.time.dt.year) == no_forecast_years, f"{model} has incorrect length of time."

    # Initialise total nens
    total_nens = 0

    # Initialise a dictionary to store the total nens for each model
    nens_dict = {}

    # Count the total nens for each model
    for model in models_list:
        # Extract the file list
        file_list = files_dict[model][0]

        # Find all of the files containing f"s{start_year}" and f"s{end_year}"
        # and add the length of the list to total_nens
        nens_model_start = len([file for file in file_list if f"s{start_year}" in file])

        # and for the end year
        nens_model_end = len([file for file in file_list if f"s{end_year}" in file])

        # Print the model and the number of ensemble members
        print(f"{model}: {nens_model_start} ensemble members: {start_year}")
        print(f"{model}: {nens_model_end} ensemble members: {end_year}")

        # include in the dictionary
        nens_dict[model] = nens_model_start

        # Assert that these are the same
        assert nens_model_start == nens_model_end, f"{model} has different number of ensemble members for start and end year."

        # Add to total_nens
        total_nens += nens_model_start

    # Print the total number of ensemble members
    print(f"Total number of ensemble members: {total_nens}")

    # Extract the lats and lons
    nlats = data.lat.shape[0] ; nlons = data.lon.shape[0]

    # Initialise the data array
    data = np.zeros([len(years), total_nens, no_forecast_years, nlats, nlons])

    # print the shape of the data array
    print(f"Shape of data array: {data.shape}")

    # Initialise a counter for the ensemble members
    ens_counter = 0

    # Previous step i2 bool
    prev_step_i2 = False

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

            # Set the previous step to i1
            prev_step_i2 = False

            # Print
            print("resetting prev_ens_member from:", prev_ens_member, "to 0")

            # Loop over the ensemble members
            for j in range(nens_dict[model]):
                print("Extracting data for ensemble member: ", j+1, "for model: ", model)

                # If the model is EC-Earth3 or NorCPM1
                if model == "EC-Earth3" or model == "NorCPM1":

                    # if j+1 is less than 10
                    if j+1 < 11:
                        print("j+1 is less than 11")
                        print("Extracting data for ensemble member: ", j+1, "for model: ", model)
                        print("for both i1 and i2")
                        # Extract the file containing f"s{year}"
                        i1_file = [file for file in file_list if f"s{year}" in file and f"r{j+1}i1" in file][0]

                        # If the file exists
                        i2_files = [file for file in file_list if f"s{year}" in file and f"r{j+1}i2" in file]

                        # Load the file using xarray
                        i1_data = xr.open_dataset(i1_file, chunks={'time': 10,
                                                                    'lat': 10,
                                                                    'lon': 10})

                        # Extract the data for the variable
                        i1_data = i1_data[variable]

                        # If the file exists
                        if len(i2_files) != 0:
                            print("i2 file exists for model: ", model, "for year: ", year, "for ensemble member: ", j+1)
                            i2_file = i2_files[0]

                            # Load the file using xarray
                            i2_data = xr.open_dataset(i2_file, chunks={'time': 10,
                                                                    'lat': 10,
                                                                    'lon': 10})
                        
                            # Extract the data for the variable
                            i2_data = i2_data[variable]

                            # if the previous step was i2
                            if prev_step_i2:
                                # Logging
                                print("Appending i1 to index ", prev_ens_member + 1)
                                print("Appending i2 to index ", prev_ens_member + 2)


                                # Store the data in the array
                                data[i, prev_ens_member + 1, :, :, :] = i1_data
                                data[i, prev_ens_member + 2, :, :, :] = i2_data

                                # Set the previous highest step to i2
                                prev_ens_member = prev_ens_member + 2

                            else:
                                # Logging
                                print("Appending i1 to index ", ens_counter + (j))
                                print("Appending i2 to index ", ens_counter + (j+1))

                                # Store the data in the array
                                data[i, ens_counter + (j), :, :, :] = i1_data

                                # Store the data in the array
                                data[i, ens_counter + (j+1), :, :, :] = i2_data

                                # Set the previous highest step to i2
                                prev_ens_member = ens_counter + (j+1)

                                # Set the previous step to i2
                                prev_step_i2 = True

                            # Increment the data counte
                        else:
                            print("i2 file does not exist for model: ", model, "for year: ", year, "for ensemble member: ", j+1)

                            # Append the data to the array
                            print("Appending i1 to index ", ens_counter + (j))

                            # Store the data in the array
                            data[i, ens_counter + (j), :, :, :] = i1_data

                            # Set the previous highest step to i1
                            prev_step_i2 = False
                    else:
                        print("j+1 is greater than 10")
                        print("files should not exist for i1 or i2")
                        # Assert that the file does not exist
                        assert len([file for file in file_list if f"s{year}" in file and f"r{j+1}i2" in file]) == 0, f"{model} has files for i2"

                        # And for i1
                        assert len([file for file in file_list if f"s{year}" in file and f"r{j+1}i1" in file]) == 0, f"{model} has files for i1"

                else:
                    print("Model is not EC-Earth3 or NorCPM1")
                    print("Only extracting data for ensemble member: ", j+1, "for model: ", model, "for i1")
                    # Extract the file containing f"s{year}"
                    i1_file = [file for file in file_list if f"s{year}" in file and f"r{j+1}i1" in file][0]

                    # Load the file using xarray
                    i1_data = xr.open_dataset(i1_file, chunks={'time': 10,
                                                                'lat': 10,
                                                                'lon': 10})
                    
                    # Extract the data for the variable
                    i1_data = i1_data[variable]

                    # Print the year and the ensemble member
                    print("Year: ", year, "ensemble member: ", j+1)
                    print("ens counter: ", ens_counter)
                    print("Appending i1 to index ", ens_counter + j)

                    # Store the data in the array
                    data[i, ens_counter + j, :, :, :] = i1_data

        # Increment the ensemble counter with the number of ensemble members
        # For the model
        ens_counter += nens_dict[model]

    # Print the shape of the data array
    print("Shape of data array: ", data.shape)

    # Return the data array
    return data

# Write a function to calculate the lagged correlation
def alternate_lag(data: np.array,
                  forecast_range: str,
                  years: np.array,
                  lag: int = 4) -> np.array:
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
    assert "-" in forecast_range, "forecast_range should be in the format 'x-y' where x and y are integers"

    # Extract the forecast range
    forecast_range_list = forecast_range.split("-")

    # Extract the start and end years
    start_year = int(forecast_range_list[0]) ; end_year = int(forecast_range_list[1])

    # Assert that end year is 6 or less than start year
    assert end_year <= 6, "end_year should be 6 or less to be valid for four year lagged correlation"

    # Assert that end year is greater than start year
    assert end_year > start_year, "end_year should be greater than start_year"

    # Set up the number of lagged years
    no_lagged_years = data.shape[0] - lag + 1

    print("no_lagged_years: ", no_lagged_years)

    # Extract the lagged years
    lagged_years = years[lag-1:]

    # Print the first and last year of the lagged years
    print("First lagged year: ", lagged_years[0])
    print("Last lagged year: ", lagged_years[-1])

    # Create an empty array to store the lagged correlation
    lagged_correlation = np.zeros([no_lagged_years, data.shape[1] * lag, data.shape[3], data.shape[4]])

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
                      f"stoppping at index {end_year+ j}")
                
                # Take the mean over the forecast years
                ensemble_member_data_mean = np.mean(ensemble_member_data[start_year + j - 1:end_year + j, :, :], axis=0)

                # Print the year index, ensemble member index and lag index
                print("year index: ", i, " ensemble member index: ", k, " lag index: ", j)

                # Print which we are appending to
                print("Appending to: year index: ", i, " ensemble member index: ", j + k * lag)

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
    parser.add_argument("variable", type=str,
                        help="The variable to load.")
    parser.add_argument("season", type=str,
                        help="The season to load.")
    parser.add_argument("region", type=str,
                        help="The region to load.",
                        default="global")
    parser.add_argument("start_year", type=int,
                        help="The start year of the period to load.",
                        default=1961)
    parser.add_argument("end_year", type=int,
                        help="The end year of the period to load.",
                        default=2014)
    parser.add_argument("forecast_range", type=str,
                        help="The forecast range to take the alternate lag for.")
    parser.add_argument("lag", type=int,
                        help="The lag to take the alternate lag for.",
                        default=4)
    
    # Extract the CLAs
    args = parser.parse_args()

    # Extract the variables
    variable = args.variable ; season = args.season ; region = args.region
    start_year = args.start_year ; end_year = args.end_year
    forecast_range = args.forecast_range ; lag = args.lag

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

    # Run the function to load the data
    data = load_data(variable=variable,
                     models_list=models_list,
                     season=season,
                     start_year=start_year,
                     end_year=end_year,
                     region=region)
    
    # Extract the current time
    current_time = time()

    # Set up the filename for saving the array
    filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}_{current_time}.npy"

    # Set up the full path for saving the array
    save_path = os.path.join(save_dir, filename)

    # Save the array
    np.save(save_path, data)

    # Now process the alternate lag data
    data_alternate_lag = alternate_lag(data=data,
                                       forecast_range=forecast_range,
                                       years=np.arange(start_year, end_year+1),
                                       lag=lag)
    
    # Set up the filename for saving the array
    filename = f"{variable}_{season}_{region}_{start_year}_{end_year}_{forecast_range}_{lag}_{current_time}_alternate_lag.npy"

    # Set up the full path for saving the array
    save_path = os.path.join(save_dir, filename)

    # Save the array
    np.save(save_path, data_alternate_lag)


if __name__ == "__main__":
    # Execute the main function
    main()