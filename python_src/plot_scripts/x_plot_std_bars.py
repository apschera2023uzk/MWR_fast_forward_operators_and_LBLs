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
fs = 14
plt.rc('font', size=fs) 
plt.style.use('seaborn-poster')
matplotlib.use("Qt5Agg")
grid_params = (-3,3.0001, 0.5)
ylims_bias = [-3, 3]
n_chans=14
elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])
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

def stats_by_channel(ds_sel, dev_var,i_elev, n_chans=n_chans):
    da = ds_sel[dev_var]
    dims = da.dims  
    if "azimuth" in dims:
        arr = da.mean(dim="azimuth").isel(elevation=i_elev).values      
    else:
        arr = da.isel(elevation=i_elev).values

    # ── Stats per channel ─────────────────────────────────────────────────────
    std_array     = np.full(n_chans, np.nan)
    bias_array    = np.full(n_chans, np.nan)
    rmse_array    = np.full(n_chans, np.nan)
    n_valid_array = np.zeros(n_chans, dtype=int)

    for i in range(n_chans):
        col     = arr[:, i]
        valid   = ~np.isnan(col)
        n_valid = int(np.sum(valid))
        n_valid_array[i] = n_valid
        if n_valid == 0:
            continue
        dev_valid      = col[valid]
        bias           = np.sum(dev_valid) / n_valid
        std            = np.sqrt(np.sum((dev_valid - bias) ** 2) / n_valid)
        rmse           = np.sqrt(np.sum(dev_valid ** 2) / n_valid)
        bias_array[i]  = bias
        std_array[i]   = std
        rmse_array[i]  = rmse

    return std_array, bias_array, rmse_array, n_valid_array

##############################################################################

def select_ds_camp_loc(ds, campaign, location):
    # Filter Datensatz nach Kampagne und Ort
    mask = (ds["Campaign"] == campaign) & (ds["Location"] == location)
    ds_sel = ds.sel(time=ds["time"].values[mask.values])
    return ds_sel

##############################################################################

def plot_std_bars(ds, stds, labels, channels, channel_labels, elev,
        n_valid_arr, save_path,campaign,location, ref_label,\
        elevations=elevations, thres_lwp=thres_lwp,\
        ylims_bias=ylims_bias, tag="any_tag"):
    
    fig, ax = plt.subplots(figsize=(14,9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}, \n\
        elevation: {elevations[elev]:.1f}°, (N: {n_valid_arr[0][0]} / {tag})")
    ax.set_title(f"Standard deviation of brightness temperature deviation (Reference: {ref_label})")
    
    width = 0.2  # Width of each bar
    offsets = np.linspace(-0.3,0.3,len(stds))
    
    for std, lbl, offset, n_valid in zip(stds, labels,offsets, n_valid_arr):
        if lbl.strip() != " (MWR)":  # Only plot if label is not empty
            lbl = lbl.strip()
            ax.bar(channels + offset, std, width, label=lbl, alpha=0.7)
    
    ax.set_yticks(np.arange(0,4.01,0.5)) 
    ax.set_xticks(channels)
    ax.set_xticklabels(channel_labels)
    ax.set_xlabel("Channels")
    # ax.set_ylim(0, 3)
    ax.set_ylabel(f"Standard Deviation of Brightness Temperature [K]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    
    plt.savefig(save_path+\
        f"/{tag}_{campaign}_{location}_{elevations[elev]:.1f}_bar_std.png",\
        dpi=300)
    plt.close("all")
    return 0

##############################################################################

def plot_rmse_bars(ds, rmses, labels, channels, channel_labels, elev,
        n_valid_arr, save_path, campaign, location, ref_label,
        elevations=elevations, thres_lwp=thres_lwp,
        ylims_bias=ylims_bias, tag="any_tag"):

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}, \n\
        elevation: {elevations[elev]:.1f}°, (N: {n_valid_arr[0][0]} / {tag})")
    ax.set_title(f"RMSE of brightness temperature deviation (Reference: {ref_label})")

    width   = 0.2
    offsets = np.linspace(-0.3, 0.3, len(rmses))

    for rmse, lbl, offset in zip(rmses, labels, offsets):
        lbl = lbl.strip()
        ax.bar(channels + offset, rmse, width, label=lbl, alpha=0.7)

    ax.set_yticks(np.arange(0, 4.01, 0.5))
    ax.set_xticks(channels)
    ax.set_xticklabels(channel_labels)
    ax.set_xlabel("Channels")
    # ax.set_ylim(0, 3)
    ax.set_ylabel(f"RMSE of Brightness Temperature [K]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path +
        f"/{tag}_{campaign}_{location}_{elevations[elev]:.1f}_bar_rmse.png",
        dpi=300)
    plt.close("all")
    return 0

##############################################################################

def plot_bias_lines(biases, labels, channels, channel_labels, elev,
        n_valid_arr, save_path, campaign, location, ref_label,
        elevations=elevations, thres_lwp=thres_lwp,
        ylims_bias=ylims_bias, grid_params=grid_params, tag="any_tag"):

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}, \n\
        elevation: {elevations[elev]:.1f}°, (N: {n_valid_arr[0][0]} / {tag})")
    ax.set_title(f"Bias of brightness temperature deviation (Reference: {ref_label})")

    ax.plot(channels, [0.] * len(channels), color="black",
            linewidth=2, label=ref_label)

    for bias, lbl in zip(biases, labels):
        lbl = lbl.strip()
        ax.bar(channels, bias, linewidth=2, alpha=0.8, label=lbl)

    ax.set_yticks(np.arange(*grid_params))
    ax.set_xticks(channels)
    ax.set_xticklabels(channel_labels)
    ax.set_xlabel("Channels")
    ax.set_xlim(channels.min() - 1, channels.max() + 1)
    # ax.set_ylim(ylims_bias)
    ax.set_ylabel(f"Bias of Brightness Temperature [K]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path +
        f"/{tag}_{campaign}_{location}_{elevations[elev]:.1f}_bias.png",
        dpi=300)
    plt.close("all")
    return 0

##############################################################################

def ensure_folder_exists(base_path, folder_name):
    # Join the base path with the folder name
    folder_path = os.path.join(base_path, folder_name)
    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    return os.path.abspath(folder_path)

##############################################################################

def apply_sky_mask(ds_sel, sky):
    """
    Masks data per (time, elevation) according to cloud_flag.

    Parameters
    ----------
    ds_sel : xarray.Dataset with 'cloud_flag' (time, elevation)
    sky    : str — "clear", "cloudy", or "all_sky"

    Returns
    -------
    ds_cf  : xarray.Dataset with cloudy/clear elevations set to NaN
    """
    if sky == "all_sky":
        return ds_sel

    cf = ds_sel["cloud_flag"]  # (time, elevation)
    bad_mask = (cf == 1) if sky == "clear" else (cf == 0)

    ds_cf = ds_sel.copy()
    for var in ds_cf.data_vars:
        if "elevation" not in ds_cf[var].dims:
            continue
        ds_cf[var] = ds_cf[var].where(~bad_mask.broadcast_like(ds_cf[var]))

    return ds_cf

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    nc_out_path=args.NetCDF
    
    ###
    # 0th Open dataset and clear sky filtering
    ds0 = xr.open_dataset(nc_out_path)
    keep_vars = [v for v in ds0.data_vars if 
                 v.startswith("Deviations_") or 
                 v.startswith("TBs_") or 
                 v == "cloud_flag" or 
                 v == "Campaign" or 
                 v == "Location"]
    ds0 = ds0[keep_vars]
    ds0 = ds0.mean(dim="azimuth", keep_attrs=True)
    
    ###
    # 1st choose dataset (RAO / clear) & Make sure Output dirs exist:
    dev_vars, var_labels, ref_labels = get_deviation_variables(ds0)
    channels = np.arange(n_chans)
    channel_labels = [f"Ch{i+1}" for i in range(n_chans)]  # Adjust as needed
    skies = ["clear", "cloudy", "all_sky"]
    campaigns = np.unique(ds0["Campaign"].values)
    locations = np.unique(ds0["Location"].values)
    for campaign in campaigns:
        print("Processing Campaign: ", campaign)
        for location in locations:
            ds_sel = select_ds_camp_loc(ds0, campaign, location)
            if len(ds_sel["time"]) == 0:
                continue
            else:
                print("Processing location:", location)
                folder = ensure_folder_exists(args.output, campaign+"_"+\
                    location+"/std/")
            for sky in skies:
                ds_cf = apply_sky_mask(ds_sel, sky)

                #######################################################

                for i_elev, elev in enumerate(ds_cf["elevation"].values):
                    stds, biases, rmses = [], [], []
                    stds_mwr, biases_mwr, rmses_mwr = [], [], []
                    labels, labels_mwr = [], []
                    n_valid_arr, n_valid_arr_mwr = [], []

                    for dev_var, var_label, ref_label in zip(dev_vars, var_labels,\
                            ref_labels):
                        std, bias, rmse, n_valid = stats_by_channel(ds_cf,\
                            dev_var, i_elev)
                        if np.all(np.isnan(std)):
                            continue
                        if "R24" in ref_label:
                            stds.append(std); biases.append(bias)
                            rmses.append(rmse); labels.append(var_label)
                            n_valid_arr.append(n_valid)
                        else:
                            stds_mwr.append(std); biases_mwr.append(bias)
                            rmses_mwr.append(rmse); labels_mwr.append(var_label)
                            n_valid_arr_mwr.append(n_valid)

                    for stds_i, biases_i, rmses_i, lbls, nvs, ref in [
                        (stds,     biases,     rmses,     labels,     n_valid_arr,     "R24"),
                        (stds_mwr, biases_mwr, rmses_mwr, labels_mwr, n_valid_arr_mwr, "MWR"),
                    ]:
                        if nvs == []:
                            continue
                        plot_std_bars(ds_cf, stds_i, lbls, channels, channel_labels,
                            i_elev, nvs, folder, campaign, location, ref, tag=sky)
                        plot_rmse_bars(ds_cf, rmses_i, lbls, channels, channel_labels,
                            i_elev, nvs, folder, campaign, location, ref, tag=sky)
                        plot_bias_lines(biases_i, lbls, channels, channel_labels,
                            i_elev, nvs, folder, campaign, location, ref, tag=sky)
                    del stds
                    del labels
                    del n_valid_arr
                    del stds_mwr
                    del labels_mwr
                    del n_valid_arr_mwr
                del ds_cf
            del ds_sel


    ###################
    # Biases scheinen sehr groß (vergleich mit CAO: 2 K...)
    # Grid anpassen
    





