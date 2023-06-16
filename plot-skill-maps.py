#!/usr/bin/env python

"""
plot-skill-maps.py
====================

Plots a map of the skill for model (or model combination) for a given time period, variable, forecast range and season. The plot is written to a PNG file.

Usage:

    $ python plot-skill-maps.py <model> <initial-year> <final-year> <variable> <region> <forecast-range> <season>

model:          Name of the model or model combination to plot
initial-year:   Initial year of the time period to plot
final-year:     Final year of the time period to plot
variable:       Variable to plot
region:         Region to plot
forecast-range: Forecast range to plot
season:         Season to plot

Example:

    $ python plot-skill-maps.py CMCC-CM2-SR5 1960 2019 psl north-atlantic 2-9 DJFM
    
"""

# Imports
import glob
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
import cartopy.crs as ccrs


def plot_time_series():
    """
    Reads in a set of CSV files and plots the time series for each county on a
    single line graph which is written to a PNG file.

    :return: None
    """
    dfs = []
    county_files = glob.glob(os.path.join(OUTPUT_DIR, '*.csv'))
    county_labels = [os.path.basename(_).split('.')[0].title() for _ in county_files]

    for fpath in county_files:
        _df = pd.read_csv(fpath, index_col=0)
        dfs.append(_df)

    df = pd.concat(dfs, axis=1).reindex()
    df.columns = county_labels

    title = 'Max Temp time series for 10 UK Counties'
    x_label = 'Time (year)'
    y_label = 'Annual maximum temperature (°C)'

    # Get the axes object and modify
    ax = df.plot(title=title)
    ax.set(xlabel=x_label, ylabel=y_label)
    ax.set_xticks([2000, 2004, 2008, 2012, 2016, 2020, 2024, 2028])
    ax.legend(fontsize='x-small')

    # Write the output file
    output_file = os.path.join(OUTPUT_DIR, 'annual-max-temp-time-series.png')
    plt.savefig(output_file)
    print('[INFO] Wrote: {}'.format(output_file))


# Define a main function which parses the arguments //
# and calls the plotting function
def main():
    """
    Main function which parses the command line arguments and calls the
    plotting function.

    :return: None
    """

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot skill maps')
    parser.add_argument('model', type=str, help='Model or model combination')
    parser.add_argument('initial_year', type=int, help='Initial year')
    parser.add_argument('final_year', type=int, help='Final year')
    parser.add_argument('variable', type=str, help='Variable')
    parser.add_argument('region', type=str, help='Region')
    parser.add_argument('forecast_range', type=str, help='Forecast range')
    parser.add_argument('season', type=str, help='Season')
    args = parser.parse_args()

    # Set up the output directory
    output_dir = f'/home/users/benhutch/skill-maps-processed-data/{args.variable}/{args.model}/{args.region}/years_{args.forecast_range}/{args.season}/outputs/mergetime'

    # Use the output directory path as needed
    print("Output directory: ", output_dir)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plot_time_series(args.model, args.initial_year, args.final_year,
                     args.variable, args.region, args.forecast_range,
                     args.season)

# The section below is run if the module is executed as a script
if __name__ == '__main__':

    plot_time_series()
