"""
Functions for selecting the season, taking the annual means and regridding the data in python.
"""

# Import local modules
import sys
import os
import glob

# Import third-party modules
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


# Define a function for loading the model data
def load_model_data(
    variable: str,
    model: str,
    experiment: str,
    start_year: int,
    end_year: int,
    csv_path: str = "/home/users/benhutch/lagging-NAO-test-suite/data_paths/paths_to_file_src.csv",
):
    """
    Loading the model data for a given variable, model, experiment and time period.

    Parameters
    ----------

    variable : str
        The variable of interest.

    model : str
        The model of interest.

    experiment : str
        The experiment of interest.

    start_year : int
        The start year of the time period.

    end_year : int
        The end year of the time period.

    Returns
    -------

        model_data : xarray DataArray
            The model data for the given variable, model, experiment and time period.
    """

    # Load the csv file
    csv_file = pd.read_csv(csv_path)

    # Try extracting the path for the given model, experiment and variable
    try:
        model_path = csv_file.loc[
            (csv_file["model"] == model)
            & (csv_file["experiment"] == experiment)
            & (csv_file["variable"] == variable),
            "path",
        ].values[0]
    except:
        print("The model, experiment or variable is not available in the csv file.")
        sys.exit()

    # Assert that the model path exists
    assert os.path.exists(model_path), "The model path does not exist."

    # Assert that the model path is not empty
    assert len(glob.glob(model_path)) > 0, "The model path is empty."

    # Extract the first part of the model path
    model_path_root = model_path.split("/")[1]

    # If the model path root is "gws"
    if model_path_root == "gws":
        print("The model path root is gws.")
    elif model_path_root == "badc":
        print("The model path root is badc.")
    else:
        print("The model path root is not gws or badc.")
        sys.exit()
