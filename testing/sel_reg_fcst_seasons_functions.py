"""
Functions for selecting the season, taking the annual means and regridding the data in python.
"""

# Import local modules
import sys
import os
import glob
import re

# Import third-party modules
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import cdo
from cdo import *

cdo = Cdo()


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

    # Set up an empty list for the file paths
    file_paths = []

    # If the model path root is "gws"
    if model_path_root == "gws":
        print("The model path root is gws.")

        # List the files at this location
        model_files = os.listdir(model_path)

        # Split the files by the "/" character
        model_files_split = [file.split("/")[-1] for file in model_files]

        # Split the filenames by the "_" character
        model_files_split = [file.split("_")[4] for file in model_files_split]

        # Split by the "-" character and extact the 1th element
        model_files_split = [file.split("-")[1] for file in model_files_split]

        # Find the unique elements
        unique_members = list(set(model_files_split))

        # Print the unique members
        print("The unique members are: ", unique_members)
        print("The length of the unique members is: ", len(unique_members))

        # Loop over the years
        for year in tqdm(range(start_year, end_year + 1)):
            # Loop over the unique members
            for member in unique_members:

                # Create the file path
                # psl_Amon_MPI-ESM1-2-LR_dcppA-hindcast_s2021-r9i1p1f1_gn_202111-203112.nc
                file_path = f"{model_path}/{variable}_?mon_{model}_{experiment}_s{year}-{member}_g?_*-*.nc"

                # # Print the file path
                # print("The file path is: ", file_path)

                # Assert that the file path exists
                assert (
                    len(glob.glob(file_path)) > 0
                ), f"The file path does not exist for year: {year} and member: {member}"

                # Assert that the size of the file path is greater than 100000 bytes
                assert (
                    os.path.getsize(glob.glob(file_path)[0]) > 100000
                ), "The file path is empty."

                # Append the file path to the list
                file_paths.append(glob.glob(file_path)[0])

    elif model_path_root == "badc":
        print("The model path root is badc.")

        # Lis the folders at this location
        model_folders = os.listdir(model_path)

        # Split the folders by the "/" character
        model_folders_split = [folder.split("/")[-1] for folder in model_folders]

        # Split the filenames by the "-" character
        model_folders_split = [folder.split("-")[1] for folder in model_folders_split]

        # Find the unique elements
        unique_members = list(set(model_folders_split))

        # Print the unique members
        print("The unique members are: ", unique_members)
        print("The length of the unique members is: ", len(unique_members))

        # Loop over the years
        for year in tqdm(range(start_year, end_year + 1)):
            # Loop over the unique members
            for member in unique_members:

                # Create the file path
                # psl_Amon_MPI-ESM1-2-LR_dcppA-hindcast_s2021-r9i1p1f1_gn_202111-203112.nc
                file_path = (
                    f"{model_path}/s{year}-{member}/Amon/{variable}/g?/files/d*/*.nc"
                )

                # Assert that there are files at this location
                assert (
                    len(glob.glob(file_path)) > 0
                ), "There are no files at this location."

                # Assrt that the size of the file path is greater than 100000 bytes
                assert (
                    os.path.getsize(glob.glob(file_path)[0]) > 100000
                ), "The file path is empty."

                # Append the file path to the list
                file_paths.append(glob.glob(file_path)[0])

    else:
        print("The model path root is not gws or badc.")
        sys.exit()

    # Return the length of the file paths
    print("The length of the file paths is: ", len(file_paths))

    # Retrun the file paths
    return file_paths


# Write a function which uses xarray to create the intermediate files
def sel_season_shift(
    file_paths: list,
    year: int,
    season: str,
    variable: str,
    model: str,
    output_dir: str = "/work/scratch-nopw2/benhutch",
):
    """
    Selects the valid months within the season, shifts the time axis back (if necessary) and takes the year mean, before saving the files to /work/scratch-nopw2/benhutch.

    Parameters
    ----------

    file_paths : list
        The list of file paths.

    year : int
        The year of interest.

    season : str
        The season of interest.

    variable : str
        The variable of interest.

    model : str
        The model of interest.

    output_dir : str
        The output directory.

    Returns
    -------

    int_file_paths : list
        The list of intermediate file paths. To be regridded and moved into gws/nopw/canari.
    """

    # Assert that file paths is not empty
    assert len(file_paths) > 0, "The file paths list is empty."

    # Find the file paths within file_paths containing the string
    # "s{year}"
    file_paths = [file for file in file_paths if f"s{year}" in file]

    # Print the length of the file paths
    print("The length of the year file paths is: ", len(file_paths))

    # Depending on the season, select the months
    if season == "DJF":
        months = [12, 1, 2]
    elif season == "MAM":
        months = [3, 4, 5]
    elif season == "JJA":
        months = [6, 7, 8]
    elif season == "JJAS":
        months = [6, 7, 8, 9]
    elif season == "SON":
        months = [9, 10, 11]
    elif season == "SOND":
        months = [9, 10, 11, 12]
    elif season == "NDJF":
        months = [11, 12, 1, 2]
    elif season == "DJFM":
        months = [12, 1, 2, 3]
    elif season == "ONDJFM":
        months = [10, 11, 12, 1, 2, 3]
    elif season == "AMJJAS":
        months = [4, 5, 6, 7, 8, 9]
    else:
        raise ValueError("Invalid season")

    # Set up an empty list for the intermediate file paths
    int_file_paths = []

    # Loop over the file paths
    for file in tqdm(file_paths):

        # Load the data
        data = xr.open_dataset(file)

        # # Print the data
        # print("The data is: ", data)

        # Select the months
        data = data.sel(time=data["time.month"].isin(months))

        # Shift the dataset if necessary
        # Shift the dataset if necessary
        if season in ["DJFM", "NDJFM", "ONDJFM"]:
            data = data.shift(time=-3)
        elif season in ["DJF", "NDJF", "ONDJF"]:
            data = data.shift(time=-2)
        elif season in ["NDJ", "ONDJ"]:
            data = data.shift(time=-1)
        else:
            data = data

        # Take the year mean
        data = data.resample(time="Y").mean("time")

        # Set up the output file dir
        output_file_dir = f"{output_dir}/{variable}/{model}/{season}/{year}/tmp"

        # If the output file dir does not exist, create it
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)

        # Set up the output file path
        org_file_name = file.split("/")[-1]

        # Replace the extension
        new_file_name = org_file_name.replace(".nc", f"_{season}_{year}_tmp.nc")

        # Set up the output file path
        output_file_path = f"{output_file_dir}/{new_file_name}"

        # if the output file path exists and has a file size greater than 10000 bytes, do not save the data
        if (
            os.path.exists(output_file_path)
            and os.path.getsize(output_file_path) > 10000
        ):
            print(
                f"The file {output_file_path} already exists and has a file size greater than 10000 bytes."
            )
            int_file_paths.append(output_file_path)
            continue

        # Save the data
        data.to_netcdf(output_file_path)

        # Append the output file path to the list
        int_file_paths.append(output_file_path)

    # asser that the length of the intermediate file paths is the same as the length of the file paths
    assert len(int_file_paths) == len(
        file_paths
    ), "The length of the intermediate file paths is not the same as the length of the file paths."

    # Return the intermediate file paths
    return int_file_paths


# Write a function to perform the regridding on the intermediate files
def regrid_int_files(
    int_file_paths: list,
    variable: str,
    model: str,
    season: str,
    region: str,
    gridspec_file: str = "/home/users/benhutch/gridspec/gridspec-global.txt",
    output_dir: str = "/work/scratch-nopw2/benhutch",
):
    """
    Regrid the intermediate files and save them to /work/scratch-nopw2/benhutch.

    Parameters
    ----------

    int_file_paths : list
        The list of intermediate file paths.

    variable : str
        The variable of interest.

    model : str
        The model of interest.

    season : str
        The season of interest.

    region : str
        The region of interest.

    gridspec_file : str
        The gridspec file.

    output_dir : str
        The output directory.

    Returns
    -------

    regrid_file_paths : list
        The list of regridded file paths. To be moved into gws/nopw/canari.
    """

    # Assert that the intermediate file paths is not empty
    assert len(int_file_paths) > 0, "The intermediate file paths list is empty."

    # Set up an empty list for the regridded file paths
    regrid_file_paths = []

    # Loop over the intermediate file paths
    for file in tqdm(int_file_paths):

        # Set up the output file dir
        output_file_dir = f"{output_dir}/{variable}/{model}/{region}/all_forecast_years/{season}/outputs"

        # Set up the output file name
        # extract the base file name
        base_file_name = file.split("/")[-1]

        # Replace the extension _{season}_????_tmp.nc
        # with .nc
        pattern = f"_{season}_...._tmp.nc"
        new_file_name = re.sub(pattern, ".nc", base_file_name)

        # Set up the output file name
        output_file_name = f"all-years-{season}-{region}-{new_file_name}"

        # Set up the output file path
        output_file_path = f"{output_file_dir}/{output_file_name}"

        # If the output file dir does not exist, create it
        if not os.path.exists(output_file_dir):
            os.makedirs(output_file_dir)

        # If the output file path exists and has a file size greater than 10000 bytes, do not regrid the file
        if (
            os.path.exists(output_file_path)
            and os.path.getsize(output_file_path) > 10000
        ):
            print(
                f"The file {output_file_path} already exists and has a file size greater than 10000 bytes."
            )
            regrid_file_paths.append(output_file_path)
            continue
        elif (
            os.path.exists(output_file_path)
            and os.path.getsize(output_file_path) < 10000
        ):
            print(
                f"The file {output_file_path} already exists but has a file size less than 10000 bytes."
            )
            os.remove(output_file_path)

            # Try regriding the file
            try:
                # Regrid the file
                cdo.remapbil(gridspec_file, input=file, output=output_file_path)
            except:
                print(f"The file {file} could not be regridded.")
                continue

        elif not os.path.exists(output_file_path):
            print(f"The file {output_file_path} does not exist.")

            # Try regriding the file
            try:
                # Regrid the file
                cdo.remapbil(gridspec_file, input=file, output=output_file_path)
            except:
                print(f"The file {file} could not be regridded.")
                continue

        # Append the output file path to the list
        regrid_file_paths.append(output_file_path)

    # Assert that the length of the regridded file paths is the same as the length of the intermediate file paths
    assert len(regrid_file_paths) == len(
        int_file_paths
    ), "The length of the regridded file paths is not the same as the length of the intermediate file paths."

    # Return the regridded file paths
    return regrid_file_paths


# Define a function to check whether the output files exist already
def check_regrid_files_exist(
    variable: str,
    model: str,
    season: str,
    experiment: str,
    region: str,
    start_year: int,
    end_year: int,
    files_dir: str = "/work/scratch-nopw2/benhutch",
):
    """
    A function to check whether all of the regridded file already exist.

    Parameters

    variable : str
        The variable of interest.

    model : str
        The model of interest.

    season : str
        The season of interest.

    experiment : str
        The experiment of interest.

    region : str
        The region of interest.

    start_year : int
        The start year of the time period.

    end_year : int
        The end year of the time period.

    files_dir : str
        The directory containing the files.
        Default is /work/scratch-nopw2/benhutch.

    Returns
    -------

    files: dataframe
        A dataframe containing the file names, paths and sizes.

    """

    # Set up an empty list for the file paths
    file_paths = []

    # Filenames
    filenames = []

    # Set up the file_sizes
    file_sizes = []

    # Set up the directory in which the data are stored
    files_dir = (
        f"{files_dir}/{variable}/{model}/{region}/all_forecast_years/{season}/outputs"
    )

    # List the files at this location
    model_files = os.listdir(files_dir)

    # Split the files by the "/" character
    model_files_split = [file.split("/")[-1] for file in model_files]

    # Split the filenames by the "_" character
    model_files_split = [file.split("_")[4] for file in model_files_split]

    # Split by the "-" character and extact the 1th element
    model_files_split = [file.split("-")[1] for file in model_files_split]

    # Find the unique elements
    unique_members = list(set(model_files_split))

    # Print the unique members
    print("The unique members are: ", unique_members)
    print("The length of the unique members is: ", len(unique_members))

    # Loop over the years
    for year in tqdm(range(start_year, end_year + 1)):
        for member in unique_members:
            # Create the file path
            file_path = f"{files_dir}/all-years-{season}-{region}-{variable}_?mon_{model}_{experiment}_s{year}-{member}_g?_*-*.nc"

            # Assert that the file path exists
            assert (
                len(glob.glob(file_path)) > 0
            ), f"The file path does not exist for year: {year} and member: {member}"

            # Assert that the filesize is greater than 10000 bytes
            assert (
                os.path.getsize(glob.glob(file_path)[0]) > 10000
            ), "The file path is empty."

            # Append the file path to the list
            file_paths.append(glob.glob(file_path)[0])

            # Append the filename to the list
            filenames.append(glob.glob(file_path)[0].split("/")[-1])

            # Append the file size to the list
            file_sizes.append(os.path.getsize(glob.glob(file_path)[0]))

    # Create a dataframe
    files = pd.DataFrame(
        {
            "file name": filenames,
            "file path": file_paths,
            "file size": file_sizes,
        }
    )

    # Return the dataframe
    return files