import os
import ipywidgets as widgets
from ipyleaflet import Map, Rectangle
import matplotlib.pyplot as plt
import xarray as xr
from scipy.interpolate import griddata
import numpy as np
import geopandas as gpd
import pandas as pd

def load_eval_df(file_path, min_lat=None, min_lon=None, max_lat=None, max_lon=None):
    eval_df = pd.read_csv(file_path) 
    eval_df = eval_df[eval_df['lat'] > min_lat] if min_lat is not None else eval_df
    eval_df = eval_df[eval_df['lat'] < max_lat] if max_lat is not None else eval_df
    eval_df = eval_df[eval_df['lon'] > min_lon] if min_lon is not None else eval_df
    eval_df = eval_df[eval_df['lon'] < max_lon] if max_lon is not None else eval_df
    return eval_df


# Define a function to plot data from a TIFF file within a bounding box
def plot_pred(file_path, date, min_lat=None, min_lon=None, max_lat=None, max_lon=None, gt_res=0.25):

    # Open data
    data = load_eval_df(file_path, min_lat, min_lon, max_lat, max_lon)

    grid_resolution = gt_res  # adjust based on your preference
    x_grid = np.arange(data['lon'].min(), data['lon'].max(), grid_resolution)
    y_grid = np.arange(data['lat'].min(), data['lat'].max(), grid_resolution)
    X, Y = np.meshgrid(x_grid, y_grid)

    # Function to interpolate values onto the regular grid
    def interpolate_values(timestamp):
        current_data = data[data['date'] == timestamp]
        values = current_data['prediction'].values
        points = current_data[['lon', 'lat']].values
        interpolated_values = griddata(points, values, (X, Y), method='linear')
        return interpolated_values


    # country border data
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    eu_countries = [
        'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czech Republic',
        'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary',
        'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta',
        'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia',
        'Spain', 'Sweden'
    ]
    eu_borders = world[world['name'].isin(eu_countries)]

    # figure and axis for animation
    fig, ax = plt.subplots()
    # borders
    eu_borders.plot(ax=ax, color='grey', linewidth=0.5)
    
    interpolated_values = interpolate_values(date)
    heatmap = ax.imshow(interpolated_values, extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
                    cmap='viridis', origin='lower', aspect='auto', animated=True, zorder=10, alpha=0.7)
    plt.colorbar(heatmap, ax=ax)
    ax.set_title(f'Date: {data.date.min()}')
    plt.show()
    

def sanity_check(min_lat, max_lat, min_lon, max_lon):
    if min_lat >= max_lat:
        print(
            "Error: Minimum Latitude cannot be greater than or equal to Maximum Latitude."
        )
        return False
    if min_lon >= max_lon:
        print(
            "Error: Minimum Longitude cannot be greater than or equal to Maximum Longitude."
        )
        return False
    return True
