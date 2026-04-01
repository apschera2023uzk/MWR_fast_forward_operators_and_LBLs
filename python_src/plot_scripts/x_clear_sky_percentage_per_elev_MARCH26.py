#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import xarray as xr
import pandas as pd
import numpy as np
import glob
from scipy.stats import pearsonr
import os
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.colors as colors

##############################################################################
# 1.5 Parameters:
##############################################################################
# Plotstyles:
fs = 20
plt.rc('font', size=fs) 
plt.style.use('seaborn-poster')
matplotlib.use("Qt5Agg")
grid_params = (-3,3.0001, 0.5)
ylims_bias = [-3, 3]
n_chans=14
elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])
label_colors = {'Dwdhat (MWR)': "blue",
        'Foghat (MWR)': "green",'RTTOV-gb (model)': "red",
        'ARMS-gb (model)': "orange",'Sunhat (MWR)': "blue",
        'Tophat (MWR)': "blue", 'Joyhat (MWR)': "green",
        'Hamhat (MWR)': "blue",
         "LABEL_9": "olive",
        "LABEL_10": "cyan"
}
thres_lwp=0.005 # kg m-2 fitting with Moritz' threshold of 5 g m-2

##############################################################################
# 2nd Used functions:
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scatter plots of TB MWR against sondes e.g.."
    )
    parser.add_argument(
        "--NetCDF", "-nc",
        type=str,
        default=os.path.expanduser("~/PhD_data/TB_preproc_and_proc_results/3campaigns_3models_all_results_and_stats.nc"),
        help="Input data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/PhD_plots/2026/"),
        help="Output plot directory"
    )
    return parser.parse_args()

##############################################################################

def get_deviation_variables(ds):
    """
    Finds all variables starting with 'Deviations_' in ds and returns
    lists of (var_name, var_label, ref_label) for looping.

    Example:
        "Deviations_RTTOV_R24"   → var_label="RTTOV",  ref_label="R24"
    """
    dev_vars    = []
    var_labels  = []
    ref_labels  = []

    for var in ds.data_vars:
        if var.startswith("Deviations_"):
            parts = var.split("_")   # e.g. ["Deviations", "RTTOV", "R24"]
            var_label = parts[1]
            ref_label = "_".join(parts[2:])   # handles multi-part names
            dev_vars.append(var)
            var_labels.append(var_label)
            ref_labels.append(ref_label)

    return dev_vars, var_labels, ref_labels

##############################################################################

def select_ds_camp_loc(ds, campaign, location):
    # Filter Datensatz nach Kampagne und Ort
    mask = (ds["Campaign"] == campaign) & (ds["Location"] == location)
    ds_sel = ds.sel(time=ds["time"].values[mask.values])
    return ds_sel

##############################################################################

def ensure_folder_exists(base_path, folder_name):
    # Join the base path with the folder name
    folder_path = os.path.join(base_path, folder_name)
    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    return os.path.abspath(folder_path)

##############################################################################

def plot_clear_sky_fraction(ds_sel, campaign, location, folder, elevations=elevations):
    """
    Computes and plots the percentage of clear-sky timesteps per elevation
    for a given campaign/location subset.

    Parameters
    ----------
    ds_sel     : xarray.Dataset filtered for campaign & location
    campaign   : str
    location   : str
    folder     : str, output directory
    elevations : array of elevation values
    """
    cf = ds_sel["cloud_flag"]  # (time, elevation)

    # Percentage of clear-sky (flag==0) per elevation
    n_total = cf.sizes["time"]
    n_clear = (cf == 0).sum(dim="time").values          # (elevation,)
    pct_clear = (n_clear / n_total) * 100.0             # (elevation,)

    elev_vals = ds_sel["elevation"].values               # actual elevation values
    elev_labels = [f"{e:.1f}°" for e in elev_vals]

    fig, ax = plt.subplots(figsize=(12, 7))
    bars = ax.bar(
        np.arange(len(elev_vals)),
        pct_clear,
        color="steelblue",
        edgecolor="black",
        alpha=0.8
    )

    # Write percentage value on top of each bar
    for bar, pct in zip(bars, pct_clear):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{pct:.1f}%",
            ha="center", va="bottom", fontsize=10
        )

    ax.set_xticks(np.arange(len(elev_vals)))
    ax.set_xticklabels(elev_labels, rotation=45, ha="right")
    ax.set_xlabel("Elevation angle", fontsize=12)
    ax.set_ylabel("Clear-sky fraction [%]", fontsize=12)
    ax.set_ylim(0, 110)
    ax.set_title(
        f"Clear-sky fraction per elevation\n"
        f"{campaign} — {location}  (N_total={n_total})",
        fontsize=13, fontweight="bold"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out_path = os.path.join(folder, f"clear_sky_fraction_{campaign}_{location}.png")
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    nc_out_path=args.NetCDF
    
    ###
    # 0th Open dataset and clear sky filtering
    ds0 = xr.open_dataset(nc_out_path)
    
    ###
    # 1st choose dataset (RAO / clear) & Make sure Output dirs exist:
    dev_vars, var_labels, ref_labels = get_deviation_variables(ds0) 
    campaigns = np.unique(ds0["Campaign"].values)
    locations = np.unique(ds0["Location"].values)
    for campaign in campaigns:
        print("Processing Campaign: ", campaign)
        for location in locations:
            print("Processing location:", location)
            ds_sel = select_ds_camp_loc(ds0, campaign, location)
            if len(ds_sel["time"]) == 0:
                continue
            folder = ensure_folder_exists(args.output, campaign + "_" + location)
            plot_clear_sky_fraction(ds_sel, campaign, location, folder)


