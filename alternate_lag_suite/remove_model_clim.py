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
def remove_model_clim(
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

    None

    """

    # Assert that files is a list and that it is not empty
    assert isinstance(files, list), "files must be a list."
    assert len(files) > 0, "files must not be empty."

    # Print the number of files
    print(f"Number of files: {len(files)}")

    # Import multiple iris cubes
    cubes = iris.load(files)

    # Print the time axis of the first cube
    print(cubes[0].coord("time"))

    # Iterate over each cube in the cube list
    for cube in cubes:
        # Add the season membership to the cube
        icc.add_season_membership(cube, "time", f'{season}', name='sel_season')

    # Create the season constraint
    season_constraint = iris.Constraint(sel_season=True)

    # Apply the constraint to each cube in the cube list
    cubes = cubes.extract(season_constraint)

    # Print the time axis of the first cube
    print(cubes[0].coord("time"))

    # print the cubes
    print(cubes)

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
    remove_model_clim(
        files=files,
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
