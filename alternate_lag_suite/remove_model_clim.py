#!/user/bin/env python

"""
remove_model_clim.py
====================

A script which removes the model climatology, for a specific season and range
of forecast years, from the model data.

Author: Ben Hutchins
Date: January 2024

Usage:

    $ python remove_model_clim.py <model> <variable> <season> <start_year> 
                                  <end_year> <region> <forecast_range>
    
    E.g. $ python remove_model_clim.py BCC-CSM2-MR psl DJFM 1961 2014 global
                                       2-9
    
    Parameters:
    ----------

    model : str
        The name of the model to be used.
        E.g. 'BCC-CSM2-MR'

    variable : str
        The name of the variable to be used.
        E.g. 'psl'

    season : str
        The season to be used.
        E.g. 'DJFM'

    start_year : int
        The start year of the forecast range.
        E.g. 1961

    end_year : int
        The end year of the forecast range.
        E.g. 2014

    region : str
        The region to be used.
        E.g. 'global'

    forecast_range : str
        The forecast range to be used.
        E.g. '2-9'

    Returns:

    A series of netCDF files, one for each forecast year and each ensemble
    member, containing the model data with the model climatology removed.
    """

# Import local modules
import os
import sys
import argparse
import glob

# Import third-party modules
from time import time
import numpy as np
import xarray as xr
import tqdm
import iris
import iris.coord_categorisation as icc
from iris.time import PartialDateTime

# FIXME: Testing for
# Import CDO
from cdo import *
cdo = Cdo()

# Import local modules
sys.path.append("/home/users/benhutch/lagging-NAO-test-suite/")

# Import dictionaries
import dictionaries as dicts


# Define a function to check whether all of the files exist
def check_files_exist(
    model: str,
    variable: str,
    season: str,
    start_year: int,
    end_year: int,
    region: str,
    forecast_range: str = "all_forecast_years",
    base_dir: str = "/work/scratch-nopw2/benhutch/",
) -> list:
    """
    Checks whether all of the files exist.

    Parameters:
    ----------

    model : str
        The name of the model to be used.
        E.g. 'BCC-CSM2-MR'

    variable : str
        The name of the variable to be used.
        E.g. 'psl'

    season : str
        The season to be used.
        E.g. 'DJFM'

    start_year : int
        The start year of the forecast range.
        E.g. 1961

    end_year : int
        The end year of the forecast range.
        E.g. 2014

    region : str
        The region to be used.
        E.g. 'global'

    forecast_range : str
        The forecast range to be used.
        E.g. '2-9'

    Returns:

    files:
        A list containing the paths to all of the files for the given
        parameters.
    """

    # Initialise a list to store the paths to the files
    files = []

    # Form the path
    path = os.path.join(
        base_dir, variable, model, region, forecast_range, season, "outputs"
    )

    # Assert that the path exists
    assert os.path.exists(path), "The path does not exist."

    # Find the files for the start year
    start_year_pattern = f"{path}/*s{start_year}*"

    # Find the len of the files which match the pattern
    start_year_len = len(glob.glob(start_year_pattern))

    # Find the number of uniqu "r*i?" values
    # in
    # all-years-DJFM-global-psl_Amon_HadGEM3-GC31-MM_dcppA-hindcast_s2018-r8i1_gn_201811-202903.nc
    no_ens = len(
        np.unique([file.split("_")[6] for file in glob.glob(start_year_pattern)])
    )

    # Print the number of ensemble members
    print(f"Number of ensemble members: {no_ens}")

    # Create a list of the unique combinations of "r*i?"
    ens_list = np.unique([file.split("_")[6] for file in glob.glob(start_year_pattern)])

    # Print the list
    print(f"Ensemble members: {ens_list}")

    # Split the ens_list by "-"
    ens_list = [ens.split("-")[1] for ens in ens_list]

    # Print the list
    print(f"Ensemble members: {ens_list}")

    # Loop over the forecast years
    for year in range(start_year, end_year + 1):
        # Find the files for the year
        year_pattern = f"{path}/*s{year}*"

        # Find the len of the files which match the pattern
        year_len = len(glob.glob(year_pattern))

        # Print
        print(f"Number of files for s{year}: {year_len}")

        # Assert that the number of files is the same as the number of
        # ensemble members
        assert (
            year_len == no_ens
        ), "The number of files does not match the number of ensemble members."

        # Loop over the ensemble members
        for ens in ens_list:
            # Find the file for the ensemble member
            file = f"{path}/*s{year}*{ens}*"

            # Find the full path
            file_path = glob.glob(file)

            # Assert that the file exists
            assert len(file_path) == 1, "The file does not exist."

            # Append the file to the list
            files.append(file_path[0])

    # Return the list
    return files


# Define a function to calculate and remove the model climatology
def extract_model_years(
    files: list,
    season: str,
    forecast_range: str,
    variable: str,
    model: str,
    region: str,
    start_year: int,
    end_year: int,
    output_dir: str = "/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/",
) -> None:
    """
    Calculates and removes the model climatology for a given set of files.
    Model climatology is calculated for files once the season (e.g DJFM) and
    forecast range (e.g. 2-9) have been selected.

    Parameters:

    files : list
        A list containing the paths to all of the files for the given
        parameters.

    season : str
        The season to be used.
        E.g. 'DJFM'

    forecast_range : str
        The forecast range to be used.
        E.g. '2-9'

    variable : str
        The name of the variable to be used.
        E.g. 'psl'

    model : str
        The name of the model to be used.
        E.g. 'BCC-CSM2-MR'

    region : str
        The region to be used.
        E.g. 'global'

    start_year : int
        The start year of the forecast range.
        E.g. 1961

    end_year : int
        The end year of the forecast range.
        E.g. 2014

    output_dir : str
        The directory in which to save the output files.
        E.g. '/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/'

    Returns:

    ens_list : list
        A list containing the ensemble members for which the model climatology
        E.g. ["r1i1", "r2i1", "r3i1", "r4i1", "r5i1", "r6i1", "r7i1", "r8i1"]

    """

    # Assert that files is a list and that it is not empty
    assert isinstance(files, list), "files must be a list."
    assert len(files) > 0, "files must not be empty."

    # Find all of the unique combinations of "r*i?"
    ens_list = np.unique([file.split("_")[6] for file in files])

    # Within ens_list, split by "-" and find all of the unique combinations
    ens_list = np.unique([ens.split("-")[1] for ens in ens_list])

    # Form the path
    path = os.path.join(
        output_dir, variable, model, region, forecast_range, season, "outputs"
    )

    # If the forecast range contains "-"
    if "-" in forecast_range:
        print("The forecast range contains a hyphen.")
        # Extract the forecast range
        forecast_range_split = forecast_range.split("-")

        # Extract the start and end years
        start_year = int(forecast_range_split[0])
        end_year = int(forecast_range_split[1])
    else:
        print("The forecast range does not contain a hyphen.")
        print("The forecast range is a single year.")
        # Set the start and end years to be the same
        start_year = int(forecast_range)
        end_year = int(forecast_range)

    # FIXME: Select the right years for months other than DJFM
    if season not in ["DJFM", "DJF", "NDJFM", "ONDJFM"]:
        print("The season is not DJFM.")
        print("Selecting the right years for the season.")

        # Add 1 to the start year
        start_year += 1

        # Add 1 to the end year
        end_year += 1

    # Print the start and end years
    print(f"Start year: {start_year}")
    print(f"End year: {end_year}")

    # Print the list
    print(f"Ensemble members: {ens_list}")

    # Print the path
    print(f"Path: {path}")

    # if the path does not exist
    if not os.path.exists(path):
        # Create the directory
        os.makedirs(path)

    # for each file
    for file in tqdm.tqdm(files):
        # Load the file
        ds = xr.open_dataset(file, chunks={"time": 10, "lat": 10, "lon": 10})

        # Extract the base name
        base_name = os.path.basename(file)

        # print the base name

        # Find the first year
        init_year = int(ds.time.dt.year.values[0])

        # Print the initialisation year
        print(f"Initialisation year: {init_year}")

        # Set the years to be extracted
        first_year = init_year + start_year - 1
        last_year = init_year + end_year - 1

        # Extract the years
        ds = ds.sel(time=slice(f"{first_year}-01-01", f"{last_year}-12-30"))

        # Take the time mean
        ds = ds.mean(dim="time")

        # Print the years
        print(f"First year: {first_year}")
        print(f"Last year: {last_year}")

        # Create the file name
        # cut the final .nc and replace with _years_2-9.nc
        filename = base_name[:-3] + f"_years_{start_year}-{end_year}.nc"

        # Form the path
        full_path = os.path.join(path, filename)

        # If the file exists
        if os.path.exists(full_path):
            # Print
            print(f"The file {filename} already exists.")

            # Find the size of the file
            size = os.path.getsize(full_path)

            # If the size is less than 1000 bytes
            if size < 1000:
                # Print that we are deleting the file
                print(f"Deleting empty file: {filename}")

                # Remove the file
                os.remove(full_path)

                # Save the file
                ds.to_netcdf(full_path)

                # Close the dataset
                ds.close()

                # Continue
                continue

            # Continue
            continue

        # Save the file
        ds.to_netcdf(full_path)

        # Close the dataset
        ds.close()

    # Loop over the new files
    # in this format:
    # filename = base_name[:-3] + f"_years_{start_year}-{end_year}.nc"
    for file in tqdm.tqdm(glob.glob(f"{path}/*_years_{start_year}-{end_year}.nc")):
        # Calculate the size of the file
        size = os.path.getsize(file)

        # Get the base name
        base_name = os.path.basename(file)

        # Find the pattern
        pattern = base_name.split("_")[4]

        # Extract the init year
        init_year = int(pattern.split("-")[0][1:])

        # Extract the variant_label
        variant_label = pattern.split("-")[1]

        # If the size is less than 10000 bytes
        if size < 20000:
            # Print that we are deleting the file
            print(f"Deleting empty file: {file}")

            # Remove the file
            os.remove(file)

            # Process this file again
            print(f"Processing file: {file} again.")

            # Convert init_year to a string
            init_year = str(init_year)

            # Print the types of the inputs
            print(f"Type of output_dir: {type(output_dir)}")
            print(f"Type of variable: {type(variable)}")
            print(f"Type of model: {type(model)}")
            print(f"Type of region: {type(region)}")
            print(f"Type of forecast_range: {type(forecast_range)}") # forecast range is a list?
            print(f"Type of season: {type(season)}")
            print(f"Type of init_year: {type(init_year)}")
            print(f"Type of variant_label: {type(variant_label)}")

            # Print the inputs
            print(f"output_dir: {output_dir}")
            print(f"variable: {variable}")
            print(f"model: {model}")
            print(f"region: {region}")
            print(f"forecast_range: {forecast_range}")
            print(f"season: {season}")
            print(f"init_year: {init_year}")
            print(f"variant_label: {variant_label}")

            # Assert that these are not a list
            assert not isinstance(output_dir, list), "output_dir is a list."
            assert not isinstance(variable, list), "variable is a list."
            assert not isinstance(model, list), "model is a list."
            assert not isinstance(region, list), "region is a list."
            assert not isinstance(forecast_range, list), "forecast_range is a list."
            assert not isinstance(season, list), "season is a list."
            assert not isinstance(init_year, list), "init_year is a list."
            assert not isinstance(variant_label, list), "variant_label is a list."

            # Form the path to the original file
            # original_file = os.path.join(
            #     output_dir,
            #     variable,
            #     model,
            #     region,
            #     forecast_range,
            #     season,
            #     "outputs",
            #     f"*s{init_year}*{variant_label}*"
            # )

            # form the original file path
            original_file = output_dir + "/" + variable + "/" + model + "/" + region + "/" + forecast_range + "/" + season + "/outputs/" + f"*s{init_year}*{variant_label}*_years_?-?.nc"

            # Print the original file
            print(f"Original file: {original_file}")

            # Make sure that only one file matches the pattern
            assert len(glob.glob(original_file)) == 1, "The file does not exist."

            # Load the file
            ds = xr.open_dataset(glob.glob(original_file)[0], chunks={"time": 10, "lat": 10, "lon": 10})

            # Set the years to be extracted
            first_year = init_year + start_year - 1 ; last_year = init_year + end_year - 1

            # Extract the years
            ds = ds.sel(time=slice(f"{first_year}-01-01", f"{last_year}-12-30"))

            # Take the time mean
            ds = ds.mean(dim="time")

            # Create the file name
            # cut the final .nc and replace with _years_2-9.nc
            filename = base_name[:-3] + f"_years_{start_year}-{end_year}.nc"

            # Form the path
            full_path = os.path.join(path, filename)

            # Assert that this file does not exist
            assert not os.path.exists(full_path), "The file already exists."

            # Save the file
            ds.to_netcdf(full_path)

            # Close the dataset
            ds.close()

    # Print
    print("Finished.")

    return ens_list


# Probably easiest to do this using CDO in python
def calculate_model_climatology(
    ens_list: list,
    season: str,
    forecast_range: str,
    variable: str,
    model: str,
    region: str,
    start_year: int,
    end_year: int,
    output_dir: str = "/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/",
) -> None:
    """
    Calculates the model climatology for a given set of files.

    Parameters:

    ens_list : list
        A list containing the ensemble members for which the model climatology

    season : str
        The season to be used.
        E.g. 'DJFM'

    forecast_range : str
        The forecast range to be used.
        E.g. '2-9'

    variable : str
        The name of the variable to be used.
        E.g. 'psl'

    model : str
        The name of the model to be used.
        E.g. 'BCC-CSM2-MR'

    region : str
        The region to be used.
        E.g. 'global'

    start_year : int
        The start year of the forecast range.
        E.g. 1961

    end_year : int
        The end year of the forecast range.
        E.g. 2014

    output_dir : str
        The directory in which to save the output files.
        E.g. '/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/'

    Returns:

    output_path : str
        The path to the output file.

    """

    # Assert that the directory exists
    assert os.path.exists(output_dir), "The directory does not exist."

    # Set up the path to the files
    path = os.path.join(
        output_dir, variable, model, region, forecast_range, season, "outputs"
    )

    if "-" in forecast_range:
        # extract the first year
        first_year = int(forecast_range.split("-")[0])

        # extract the last year
        last_year = int(forecast_range.split("-")[1])
    else:
        # Set the first and last years to be the same
        first_year = int(forecast_range)
        last_year = int(forecast_range)

    # if the season is not DJFM, DJF, NDJFM, or ONDJFM
    if season not in ["DJFM", "DJF", "NDJFM", "ONDJFM"]:
        # Print the season
        print(f"The season is {season}.")
        print(f"Adding 1 to the start and end years.")

        # Add 1 to the start year
        first_year += 1

        # Add 1 to the end year
        last_year += 1

    # Assert that the path exists
    assert os.path.exists(path), "The path does not exist."

    # Verify that there are len(ens_list) files for each year
    for year in range(start_year, end_year + 1):
        # Form the pattern
        pattern = f"{path}/*s{year}*_years_{first_year}-{last_year}.nc"

        # Find the len of the files which match the pattern
        year_len = len(glob.glob(pattern))

        # Print
        print(f"Number of files for s{year}: {year_len}")

        # Assert that the number of files is the same as the number of
        # ensemble members
        assert year_len == len(
            ens_list
        ), "The number of files does not match the number of ensemble members."

    # Verify that only the files for the years specified exist
    files = glob.glob(f"{path}/*years_{first_year}-{last_year}.nc")

    # Assert that there are len(ens_list) * len(range(start_year, end_year + 1))
    # files
    if len(files) != len(ens_list) * len(
        range(start_year, end_year + 1)
    ):
        # Print
        print("The number of files does not match the number of ensemble members.")

        # Print that we are deleting the files from outside the range
        print(f"Deleting files from outside the range specified by {start_year} and {end_year}.")

        # Loop over the files
        for file in files:
            # Extract the base name
            base_name = os.path.basename(file)

            # Extract the init year *e.g. s2018* from the file name
            pattern = base_name.split("_")[4]

            # Extract the init year
            init_year = int(pattern.split("-")[0][1:])

            # If the init year is not in the range
            if init_year not in range(start_year, end_year + 1):
                # Print
                print(f"Deleting file: {file}")

                # Delete the file
                os.remove(file)

    # Form the output path
    output_dir = os.path.join(path, "model_mean_state")

    # If the directory does not exist
    if not os.path.exists(output_dir):
        # Create the directory
        os.makedirs(output_dir)

    # Set the output filename
    output_fname = (
        f"{variable}_{model}_{region}_{season}"
        f"_years_{start_year}-{end_year}_{forecast_range}.nc"
    )

    # Form the output path
    output_path = os.path.join(output_dir, output_fname)

    # # If the file exists
    # if os.path.exists(output_path):
    #     # Print
    #     print(f"The file {output_path} already exists.")

    #     # Continue
    #     return output_path

    # Set up the paths
    paths = os.path.join(path, "*_years_?-?.nc")

    # Calculate the model climatology
    cdo.ensmean(input=paths, output=output_path)

    # Print
    print("Finished.")
    print(
        f"Output path: {output_path} for {variable} {model} {region} {season} "
        f"years {start_year}-{end_year} {forecast_range}."
    )

    return output_path


# Define a function which removes the climatology from the model data
def remove_model_climatology(
    climatology_path: str,
    ens_list: list,
    season: str,
    forecast_range: str,
    variable: str,
    model: str,
    region: str,
    start_year: int,
    end_year: int,
    base_dir: str = "/work/scratch-nopw2/benhutch/",
    output_dir: str = "/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/",
) -> list:
    """
    Removes the model climatology from the model data.
    NOTE: The climatology must be calculated before this function is called.

    Parameters:

    climatology_path : str
        The path to the model climatology.
        E.g. '/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/
        psl/BCC-CSM2-MR/global/2-9/DJFM/outputs/model_mean_state/
        psl_BCC-CSM2-MR_global_DJFM_years_1961-2014_2-9.nc'

    ens_list : list
        A list containing the ensemble members for which the model climatology
        E.g. ["r1i1", "r2i1", "r3i1", "r4i1", "r5i1", "r6i1", "r7i1", "r8i1"]

    season : str
        The season to be used.
        E.g. 'DJFM'

    forecast_range : str
        The forecast range to be used.
        E.g. '2-9'

    variable : str
        The name of the variable to be used.
        E.g. 'psl'

    model : str
        The name of the model to be used.
        E.g. 'BCC-CSM2-MR'

    region : str
        The region to be used.
        E.g. 'global'

    start_year : int
        The start year of the forecast range.
        E.g. 1961

    end_year : int
        The end year of the forecast range.
        E.g. 2014

    base_dir : str
        The base directory in which the model data is stored.
        E.g. '/work/scratch-nopw2/benhutch/'

    output_dir : str
        The directory in which to save the output files.
        E.g. '/gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/'

    Returns:

    output_files: list
        A list containing the paths to the output files.
    """

    # Set up the years
    valid_years = [int(year) for year in range(start_year, end_year + 1)]

    # form the path to the climatology
    climatology_dir = os.path.dirname(climatology_path)

    # Assert that the directory exists
    assert os.path.exists(climatology_dir), "The directory does not exist."

    # Assert that the file exists
    assert os.path.exists(climatology_path), "The file does not exist."

    # Assert that the *.nc file is not empty
    assert os.path.getsize(climatology_path) > 0, "The file is empty."

    # Create a copy of the forecast range
    forecast_range_copy = forecast_range

    if "-" in forecast_range:
        # split the forecast range
        forecast_range_copy = forecast_range_copy.split("-")

        # Extract the start and end years
        first_year = int(forecast_range_copy[0])
        last_year = int(forecast_range_copy[1])
    else:
        # Set the start and end years to be the same
        first_year = int(forecast_range_copy)
        last_year = int(forecast_range_copy)

    # Print the start and end years
    print(f"First year: {first_year}")
    print(f"Last year: {last_year}")

    # Depending on the season, set the forecast range
    if season not in ["DJFM", "DJF", "NDJFM", "ONDJFM"]:
        # Print the season
        print(f"The season is {season}.")
        print(f"Adding 1 to the start and end years.")

        # Add 1 to the start year
        first_year += 1

        # Add 1 to the end year
        last_year += 1

    # Load the climatology
    climatology = xr.open_dataset(climatology_path, chunks={"lat": 10, "lon": 10})

    # Set up the path to the files
    path = os.path.join(
        base_dir, variable, model, region, "all_forecast_years", season, "outputs"
    )

    # Assert that the path exists
    assert os.path.exists(path), "The path does not exist."

    # Verify that there are len(ens_list) files for each year
    for year in range(start_year, end_year + 1):
        # Form the pattern
        pattern = f"{path}/*s{year}*"

        # Find the len of the files which match the pattern
        year_len = len(glob.glob(pattern))

        # Print
        print(f"Number of files for s{year}: {year_len}")

        # Assert that the number of files is the same as the number of
        # ensemble members
        assert year_len == len(
            ens_list
        ), "The number of files does not match the number of ensemble members."

    # Set up an empty list of files
    files = []

    # Loop over the forecast years
    for year in range(start_year, end_year + 1):
        # Find the files for the year
        year_pattern = f"{path}/*s{year}*"

        # Append the files to the list
        files.extend(glob.glob(year_pattern))

    # Assert that there are len(ens_list) * len(range(start_year, end_year + 1))
    # files
    assert len(files) == len(ens_list) * len(
        range(start_year, end_year + 1)
    ), "The number of files does not match the number of ensemble members."

    # Initialise a list to store the paths to the output files
    output_files = []

    # Loop over the files
    for file in tqdm.tqdm(files):
        # Load the file
        ds = xr.open_dataset(file, chunks={"time": 10, "lat": 10, "lon": 10})

        # Extract the base name
        base_name = os.path.basename(file)

        # Extract the init year *e.g. s2018* from the file name
        pattern = base_name.split("_")[4]

        # Print the pattern
        print(f"Pattern: {pattern}")

        # Extract the init year
        init_year = int(pattern.split("-")[0][1:])

        # Print the initialisation year
        print(f"Initialisation year: {init_year}")

        # Verify that this is one of the valid years
        assert init_year in valid_years, "The initialisation year is not valid."

        # Print the initialisation year
        print(f"Initialisation year: {init_year}")

        # Remove the climatology
        ds = ds - climatology

        # Set up the previous filename
        prev_filename = base_name[:-3] + f"_years_{first_year}-{last_year}_anoms.nc"

        # Create the file name
        # cut the final .nc and replace with _anoms.nc
        filename = base_name[:-3] + f"_years_{first_year}-{last_year}_start_{start_year}_end_{end_year}_anoms.nc"

        # Set the path for the files
        path = os.path.join(
            output_dir,
            variable,
            model,
            region,
            forecast_range,
            season,
            "outputs",
            )

        # If the file exists and the model is not HadGEM3-GC31-MM
        if os.path.exists(os.path.join(path, prev_filename)) and model != "HadGEM3-GC31-MM":
            print(f"The file {prev_filename} already exists.")
            print(f"Renaming the file to {filename}.")
            print(f"As the correct climatology has already been removed.")

            # change the filename of the existing file
            os.rename(
                os.path.join(path, prev_filename), os.path.join(path, filename)
            )
        elif os.path.exists(os.path.join(path, prev_filename)) and model == "HadGEM3-GC31-MM":
            print("Deleting the previous file for HadGEM3-GC31-MM.")
            print(f"Deleting the file {prev_filename}.")
            print(f"As the incorrect climatology has been removed.")

            # Delete the previous file
            os.remove(os.path.join(path, prev_filename))


        # Form the path
        # /gws/nopw/j04/canari/users/benhutch/skill-maps-processed-data/psl/HadGEM3-GC31-MM/global/2-5/DJFM/outputs
        full_path = os.path.join(
            output_dir,
            variable,
            model,
            region,
            forecast_range,
            season,
            "outputs",
            filename,
        )

        # Print the climatology filename
        print(f"Removing climatology: {climatology_path}")

        # # If the file exists
        # if os.path.exists(full_path):
        #     # Print
        #     print(f"The file {filename} already exists.")

        #     # Print that we are deleting the file
        #     print(f"Deleting anoms file: {filename}")

        #     # Remove the file
        #     os.remove(full_path)

        # Save the file
        ds.to_netcdf(full_path)

        # Append the file to the list
        output_files.append(full_path)

        # # Remove the previous file
        # os.remove(file)

        # Close the dataset
        ds.close()

    # Print the output files
    print(output_files)

    # Print
    print("Finished.")

    return output_files


# Define a main function
def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("model", help="The name of the model to be used.")
    parser.add_argument("variable", help="The name of the variable to be used.")
    parser.add_argument("season", help="The season to be used.")
    parser.add_argument("start_year", help="The start year of the forecast range.")
    parser.add_argument("end_year", help="The end year of the forecast range.")
    parser.add_argument("region", help="The region to be used.")
    parser.add_argument(
        "forecast_range", help="The forecast range to be used.", type=str
    )

    # Parse the arguments
    args = parser.parse_args()

    # Extract the arguments
    model = args.model
    variable = args.variable
    season = args.season
    start_year = int(args.start_year)
    end_year = int(args.end_year)
    region = args.region
    forecast_range = args.forecast_range

    # Extract the models for the given variable
    if variable == "tas":
        models_list = dicts.models
    elif variable == "sfcWind":
        models_list = dicts.sfcWind_models
    elif variable == "psl":
        models_list = dicts.models
    elif variable == "rsds":
        models_list = dicts.rsds_models
    else:
        raise ValueError("variable not recognised")

    # if the model is a string containing an integer
    if model.isdigit():
        print("The model is an integer.")

        # Convert the model to an integer
        model = int(model)

        print(f"Model index: {model}")
        print("Extracting this model")

        # Verify that the length of the models_list is greater than the model
        if len(models_list) < model:
            # Raise an error
            print(f"The model index is not valid for variable {variable}.")
        else:
            # Extract the model
            model = models_list[model - 1]

    # Print the model
    print(f"Model: {model}")


    # Print the arguments
    print(f"Model: {model}")
    print(f"Variable: {variable}")
    print(f"Season: {season}")
    print(f"Start year: {start_year}")
    print(f"End year: {end_year}")
    print(f"Region: {region}")
    print(f"Forecast range: {forecast_range}")

    # Test the function
    files = check_files_exist(
        model=model,
        variable=variable,
        season=season,
        start_year=start_year,
        end_year=end_year,
        region=region,
    )

    # Print the files
    print(files)

    # Test the function
    ens_list = extract_model_years(
        files=files,
        season=season,
        forecast_range=forecast_range,
        variable=variable,
        model=model,
        region=region,
        start_year=start_year,
        end_year=end_year,
    )

    # Print the list
    print(ens_list)

    # Test the function
    output_path = calculate_model_climatology(
        ens_list=ens_list,
        season=season,
        forecast_range=forecast_range,
        variable=variable,
        model=model,
        region=region,
        start_year=start_year,
        end_year=end_year,
    )

    # Test the function
    output_files = remove_model_climatology(
        climatology_path=output_path,
        ens_list=ens_list,
        season=season,
        forecast_range=forecast_range,
        variable=variable,
        model=model,
        region=region,
        start_year=start_year,
        end_year=end_year,
    )

    # Print
    print("Finished.")


# Define behaviour when called from command line
if __name__ == "__main__":
    main()
