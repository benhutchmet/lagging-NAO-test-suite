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

# Import CDO
from cdo import *
cdo = Cdo()


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
# TODO: Would this function be different for JJA/MAM/SON?
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

    # Extract the forecast range
    forecast_range = forecast_range.split("-")

    # Extract the start and end years
    start_year = int(forecast_range[0])
    end_year = int(forecast_range[1])

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
        ds = xr.open_dataset(file)

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

            # Continue
            continue

        # Save the file
        ds.to_netcdf(full_path)

        # Close the dataset
        ds.close()

    # Print
    print("Finished.")

    return ens_list


# TODO: write a function for calculating the model climatology
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

    None

    """

    # Assert that the directory exists
    assert os.path.exists(output_dir), "The directory does not exist."

    # Set up the path to the files
    path = os.path.join(
        output_dir, variable, model, region, forecast_range, season, "outputs"
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

    # Verify that only the files for the years specified exist
    files = glob.glob(f"{path}/*.nc")

    # Assert that there are len(ens_list) * len(range(start_year, end_year + 1))
    # files
    assert len(files) == len(ens_list) * len(
        range(start_year, end_year + 1)
    ), "The number of files does not match the number of ensemble members."

    # Form the paths
    paths = f"{path}/*.nc"

    # Print the paths
    print(paths)

    # Form the output path
    output_dir = os.path.join(path, "model_mean_state")

    # If the directory does not exist
    if not os.path.exists(output_dir):
        # Create the directory
        os.makedirs(output_dir)
                               
    # Set the output filename
    output_fname = (f"{variable}_{model}_{region}_{season}"
                    f"_years_{start_year}-{end_year}_{forecast_range}.nc")

    # Form the output path
    output_path = os.path.join(output_dir, output_fname)
    
    # If the file exists
    if os.path.exists(output_path):
        # Print
        print(f"The file {output_path} already exists.")

        # Continue
        return None
    
    # Calculate the model climatology
    cdo.ensmean(input=paths, output=output_path)

    # Print
    print("Finished.")
    print(f"Output path: {output_path} for {variable} {model} {region} {season} "
          f"years {start_year}-{end_year} {forecast_range}.")

    return None


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
    calculate_model_climatology(
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
