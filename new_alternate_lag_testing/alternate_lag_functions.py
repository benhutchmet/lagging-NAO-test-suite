# Functions for processing alternate lag data
# Import local modules
import sys
import os

# Import third-party modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr

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


    # Loop over the models
    for model in models_list:
        print("Extracting data into array for model: ", model)

        # Extract the file list
        file_list = files_dict[model][0]

        # Loop over the years
        for i, year in enumerate(years):
            print("Extracting data for year: ", year, "for model: ", model)
            print("Year index: ", i, "year: ", year)

            # Loop over the ensemble members
            for j in range(nens_dict[model]):
                print("Extracting data for ensemble member: ", j+1, "for model: ", model)

                # If the model is EC-Earth3 or NorCPM1
                if model == "EC-Earth3" or model == "NorCPM1":
                    
                    # if j+1 is less than 10
                    if j+1 < 10:
                        print("j+1 is less than 10")
                        print("Extracting data for ensemble member: ", j+1, "for model: ", model)
                        print("for both i1 and i2")
                        # Extract the file containing f"s{year}"
                        i1_file = [file for file in file_list if f"s{year}" in file and f"r{j+1}i1p1f1" in file][0]
                        i2_file = [file for file in file_list if f"s{year}" in file and f"r{j+1}i2p1f1" in file][0]

                        # Load the file using xarray
                        i1_data = xr.open_dataset(i1_file, chunks={'time': 10,
                                                                    'lat': 10,
                                                                    'lon': 10})
                        i2_data = xr.open_dataset(i2_file, chunks={'time': 10,
                                                                    'lat': 10,
                                                                    'lon': 10})
                        
                        # Extract the data for the variable
                        i1_data = i1_data[variable]
                        i2_data = i2_data[variable]

                        # Logging
                        print("Appending i1 to index ", ens_counter + (2*j))
                        print("Appending i2 to index ", ens_counter + (2*j + 1))

                        # Store the data in the array
                        data[i, ens_counter + (2*j), :, :, :] = i1_data

                        # # Increment the ensemble counter
                        # ens_counter += 1

                        # Store the data in the array
                        data[i, ens_counter + (2*j + 1), :, :, :] = i2_data

                        # # Increment the ensemble counter
                        # ens_counter += 1
                    else:
                        print("j+1 is greater than 10")
                        print("files should not exist for i1 or i2")
                        # Assert that the file does not exist
                        assert len([file for file in file_list if f"s{year}" in file and f"r{j+1}i2p1f1" in file]) == 0, f"{model} has files for i2"

                        # And for i1
                        assert len([file for file in file_list if f"s{year}" in file and f"r{j+1}i1p1f1" in file]) == 0, f"{model} has files for i1"

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

                    # Store the data in the array
                    data[i, ens_counter + j, :, :, :] = i1_data

        # Increment the ensemble counter with the number of ensemble members
        # For the model
        ens_counter += nens_dict[model]


    # Print the shape of the data array
    print("Shape of data array: ", data.shape)