import os
import argparse
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from netCDF4 import Dataset
from wrf import getvar, latlon_coords, to_np, smooth2d
import pandas as pd
from datetime import timedelta

def run_pressure_pipeline(wrf_path, output_dir):
    ncfile = Dataset(wrf_path)
    shapefile_path = "/home/erickwambugu/WEATHER/WORKFLOW/map_shapefile_kenya/ke_shp/ke.shp"
    counties = gpd.read_file(shapefile_path)
    
    # 1. EXTRACT SLP
    # 'slp' is a diagnostic variable in wrf-python that calculates MSLP
    slp = getvar(ncfile, "slp", timeidx=None) 
    lats, lons = latlon_coords(slp)
    extent = [33.5, 42.0, -4.8, 5.2]

    def format_eat_time(wrf_time_raw):
        ts = pd.to_datetime(str(wrf_time_raw))
        return (ts + timedelta(hours=3)).strftime('%d %b %Y | %H:%M EAT')

    def apply_map_features(axis):
        axis.set_extent(extent, crs=ccrs.PlateCarree())
        axis.add_feature(cfeature.COASTLINE, linewidth=1.2, zorder=5)
        axis.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=1.5, zorder=5)
        axis.add_feature(cfeature.LAKES, edgecolor="blue", facecolor="none", linewidth=0.5, zorder=5)
        counties.boundary.plot(ax=axis, edgecolor="black", linewidth=0.3, alpha=0.4, zorder=6)

        gl = axis.gridlines(draw_labels=True, linewidth=0.5, color='grey', alpha=0.2, linestyle='--')
        gl.top_labels = gl.right_labels = False
        gl.xformatter, gl.yformatter = LONGITUDE_FORMATTER, LATITUDE_FORMATTER

    # --- 1. STATIC MSLP ANALYSIS ---
    print("Generating Surface Pressure Analysis...")
    fig = plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    apply_map_features(ax)
    
    # Use the first time step for the summary or mean
    mean_slp = smooth2d(np.mean(slp, axis=0), 3) # Smooth slightly for cleaner contours
    
    # Plot Filled Contours (Colors)
    clevs = np.arange(1004, 1020, 1) # Typical range for East Africa
    cf = ax.contourf(to_np(lons), to_np(lats), to_np(mean_slp), 
                     levels=clevs, cmap="RdYlBu_r", extend="both", transform=ccrs.PlateCarree(), zorder=3)
    
    # Plot Contour Lines (Isobars)
    cl = ax.contour(to_np(lons), to_np(lats), to_np(mean_slp), 
                    levels=clevs, colors="black", linewidths=0.5, transform=ccrs.PlateCarree(), zorder=4)
    ax.clabel(cl, inline=1, fontsize=8, fmt="%1.0f") # Adds pressure values to lines

    plt.colorbar(cf, ax=ax, orientation='horizontal', pad=0.08, shrink=0.7, label='Sea Level Pressure (hPa)')

    start_t = format_eat_time(slp.Time.values[0])
    end_t = format_eat_time(slp.Time.values[-1])
    plt.title(f"24-Hour Mean Sea Level Pressure: {start_t} to {end_t}", fontsize=14, fontweight='bold')

    plt.savefig(os.path.join(output_dir, "mslp_analysis.png"), dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()
    run_pressure_pipeline(args.input, args.outdir)