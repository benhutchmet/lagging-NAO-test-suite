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
              base_dir: str = '/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data'):
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
        assert len(data.time.dt.year) == 11, f"{model} has incorrect length of time."

    