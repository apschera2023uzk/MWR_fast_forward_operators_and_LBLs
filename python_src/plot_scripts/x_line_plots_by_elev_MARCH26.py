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

def get_deviation_variables_split(ds):
    """
    Finds all variables starting with 'Deviations_' in ds and returns
    two sets of lists depending on whether the reference is PyRTlib or not.

    Returns
    -------
    dev_vars_r24, var_labels_r24, ref_labels_r24   : PyRTlib as reference
    dev_vars_mwr, var_labels_mwr, ref_labels_mwr   : MWR as reference
    """
    dev_vars_r24,  var_labels_r24,  ref_labels_r24  = [], [], []
    dev_vars_mwr,  var_labels_mwr,  ref_labels_mwr  = [], [], []

    for var in ds.data_vars:
        if var.startswith("Deviations_"):
            parts     = var.split("_")
            var_label = parts[1]
            ref_label = "_".join(parts[2:])

            if "R24" in ref_label or "PyRTlib" in ref_label:
                dev_vars_r24.append(var)
                var_labels_r24.append(var_label)
                ref_labels_r24.append(ref_label)
            else:
                dev_vars_mwr.append(var)
                var_labels_mwr.append(var_label)
                ref_labels_mwr.append(ref_label)

    return dev_vars_r24, var_labels_r24, ref_labels_r24,\
            dev_vars_mwr, var_labels_mwr, ref_labels_mwr

##############################################################################

def stats_by_channel(ds_sel, dev_var,i_elev, n_chans=n_chans):
    da = ds_sel[dev_var]
    dims = da.dims  
    if "azimuth" in dims:
        print("Azi!")
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

def ensure_folder_exists(base_path, folder_name):
    # Join the base path with the folder name
    folder_path = os.path.join(base_path, folder_name)
    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    return os.path.abspath(folder_path)

##############################################################################

def apply_sky_mask(ds_sel, sky):
    if sky == "all_sky":
        return ds_sel

    cf = ds_sel["cloud_flag"]  # (time, elevation)

    if sky == "clear":
        bad_mask_elev = (cf == 1)   # (time, elevation): cloudy cells
    else:  # cloudy
        bad_mask_elev = (cf == 0)   # (time, elevation): clear cells

    # Timestep is dropped if ALL elevations are bad:
    bad_mask_time = bad_mask_elev.all(dim="elevation")  # (time,)
    good_time     = ~bad_mask_time

    # 1. Drop fully-bad timesteps from entire dataset (all vars, incl. no-elevation vars):
    ds_cf = ds_sel.isel(time=good_time.values)

    # 2. For vars with elevation dim: additionally NaN out bad (time, elevation) cells:
    bad_mask_elev_filtered = bad_mask_elev.isel(time=good_time.values)
    for var in ds_cf.data_vars:
        if "elevation" not in ds_cf[var].dims:
            continue
        ds_cf[var] = ds_cf[var].where(
            ~bad_mask_elev_filtered.broadcast_like(ds_cf[var]))

    return ds_cf

##############################################################################
    
def plot_bias_lines(all_biases, all_labels, channels, channel_labels,
                   elev_val, n_valid, save_path, campaign="unknown",
                   location="unknown", ref_label="PyRTlib R24 LBL",
                   ylims_bias=[-3,3],
                   grid_params=(-3, 3.0001, 0.5), tag="any_tag"):

    fig, ax = plt.subplots(figsize=(14, 9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}\n"
                 f"elevation: {elev_val:.1f}°  (N: {n_valid} / {tag})")
    ax.set_title("Bias of brightness temperature against "+ref_label)
    ax.plot(channels, [0.] * len(channels), label=ref_label,
            color="black", linewidth=2)

    for bias, label in zip(all_biases, all_labels):
        label = label.strip()
        ax.plot(channels, bias, label=label, alpha=0.8, linewidth=2)

    ax.set_yticks(np.arange(*grid_params))
    ax.set_xticks(channels)
    ax.set_xticklabels(channel_labels)
    ax.set_xlabel("Channels")
    ax.set_xlim(channels.min() - 1, channels.max() + 1)
    ax.set_ylabel("Bias of Brightness Temperature [K]")
    ax.set_ylim(ylims_bias)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

##############################################################################

def plot_bias_std_lines(all_biases, all_stds, all_labels, channels,
                        channel_labels, elev_val, n_valid, save_path,
                        campaign="unknown", location="unknown",
                        ref_label="PyRTlib R24 LBL", 
                        ylims_bias=[-3, 3], grid_params=grid_params,
                        tag="any_tag"):

    plt.figure(figsize=(14, 9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}\n"
                 f"elevation: {elev_val:.1f}°  (N: {n_valid}) [{tag}]")
    plt.title(f"Bias and standard deviation of brightness temperature error\n(reference: {ref_label})")
    plt.xlabel("Channels")
    plt.ylabel("Bias of Brightness Temperature [K]")
    plt.plot(channels, [0.] * len(channels), label=ref_label,
             color="black", linewidth=2)

    for bias, std, label in zip(all_biases, all_stds, all_labels):
        label = label.strip()
        plt.plot(channels, bias, label=label, alpha=0.8, linewidth=2)
        plt.fill_between(channels, bias - std, bias + std, alpha=0.2)

    plt.yticks(np.arange(*grid_params))
    plt.xticks(channels, channel_labels)
    plt.xlim(channels.min() - 1, channels.max() + 1)
    plt.ylim(ylims_bias)
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

##############################################################################

def derive_stats_per_chan_and_elev(ds_cf, dev_var, var_label, ref_label):

    biases = ds_cf[dev_var].mean(dim="time")
    stds = ds_cf[dev_var].std(dim="time")
    n_valid = ds_cf[dev_var].count(dim="time")  # shape: (N_Channels, elevation)
    rmses = np.sqrt((ds_cf[dev_var]**2).mean(dim="time"))  # (N_Channels, elevation)
    var_var = ds_cf[dev_var].attrs["var_label"]
    ref_var = ds_cf[dev_var].attrs["ref_label"]
    if "hat" in var_var: 
        pearsons_rs = xr.corr(ds_cf[var_var].transpose("time",\
            "N_Channels", "elevation", ...), ds_cf[ref_var], dim="time")[:,:,0]
    else:
        pearsons_rs = xr.corr(ds_cf[var_var], ds_cf[ref_var], dim="time")[:,:,0]

    return stds, rmses, biases, n_valid, var_label, ref_label, pearsons_rs

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    nc_out_path=args.NetCDF
    
    ###
    # 0th Open dataset and clear sky filtering
    ds0 = xr.open_dataset(nc_out_path)
    # ds0 = ds0.where(ds0["qual_flag"] == 0, drop=True)
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
    dev_vars_r24, var_labels_r24, ref_labels_r24, dev_vars_mwr, var_labels_mwr,\
         ref_labels_mwr = get_deviation_variables_split(ds0) 
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
            for sky in skies:
                ds_cf = apply_sky_mask(ds_sel, sky)
                print("Processing sky-tag:", sky)

                #####
                # PyRTlib: 
                # ── Statistiken einmal berechnen ──────────────────────────────
                stats_cache = {}
                for dev_var, var_label, ref_label in zip(dev_vars_r24, var_labels_r24, ref_labels_r24):
                    if ds_cf[dev_var].isnull().all():
                        continue
                    else:
                        print("Processing Variable:", dev_var)
                    stds, rmses, biases, n_valid, var_label, ref_label, pearsons_rs = \
                        derive_stats_per_chan_and_elev(ds_cf, dev_var, var_label, ref_label)
                    stats_cache[dev_var] = (stds, biases, n_valid, var_label)

                # ── Dann über Elevationen iterieren ───────────────────────────
                channels       = np.arange(n_chans)
                channel_labels = [f"Ch{i+1}" for i in range(n_chans)]
                bias_dir     = ensure_folder_exists(args.output, f"{campaign}_{location}/bias/")
                bias_std_dir = ensure_folder_exists(args.output, f"{campaign}_{location}/bias_std/")

                for i_elev, elev in enumerate(ds_cf["elevation"].values):
                    all_biases_ele = []
                    all_stds_ele   = []
                    all_labels_ele = []
                    n_valid_min    = None

                    for dev_var in stats_cache:
                        stds, biases, n_valid, var_label = stats_cache[dev_var]
                        bias_ele = biases.isel(elevation=i_elev).values
                        std_ele  = stds.isel(elevation=i_elev).values

                        if np.all(np.isnan(bias_ele)):
                            continue

                        all_biases_ele.append(bias_ele)
                        all_stds_ele.append(std_ele)
                        all_labels_ele.append(var_label)
                        n_valid_min = int(n_valid.isel(elevation=i_elev).min().values)
                        del stds, biases, n_valid

                    if not all_biases_ele:
                        continue

                    plot_bias_lines(
                        all_biases_ele, all_labels_ele, channels, channel_labels,
                        elev, n_valid_min,
                        save_path=os.path.join(bias_dir,
                            f"{sky}_elev{elev:.1f}_{campaign}_{location}_bias.png"),
                        campaign=campaign, location=location, tag=sky,
                ref_label=ref_label 
                    )  
                    plot_bias_std_lines(
                        all_biases_ele, all_stds_ele, all_labels_ele, channels, channel_labels,
                        elev, n_valid_min,
                        save_path=os.path.join(bias_std_dir,
                            f"{sky}_elev{elev:.1f}_{campaign}_{location}_bias_std.png"),
                        campaign=campaign, location=location, tag=sky,                
                        ref_label=ref_label 
                    )  

                #####
                # MWR:
                # ── Statistiken einmal berechnen ──────────────────────────────
                stats_cache = {}
                for dev_var, var_label, ref_label in zip(dev_vars_mwr, var_labels_mwr,ref_labels_mwr):
                    if ds_cf[dev_var].isnull().all():
                        continue
                    else:
                        print("Processing Variable:", dev_var)
                    stds, rmses, biases, n_valid, var_label, ref_label, pearsons_rs = \
                        derive_stats_per_chan_and_elev(ds_cf, dev_var, var_label, ref_label)
                    stats_cache[dev_var] = (stds, biases, n_valid, var_label)

                # ── Dann über Elevationen iterieren ───────────────────────────
                channels       = np.arange(n_chans)
                channel_labels = [f"Ch{i+1}" for i in range(n_chans)]
                bias_dir     = ensure_folder_exists(args.output, f"{campaign}_{location}/bias/")
                bias_std_dir = ensure_folder_exists(args.output, f"{campaign}_{location}/bias_std/")

                for i_elev, elev in enumerate(ds_cf["elevation"].values):
                    all_biases_ele = []
                    all_stds_ele   = []
                    all_labels_ele = []
                    n_valid_min    = None

                    for dev_var in stats_cache:
                        stds, biases, n_valid, var_label = stats_cache[dev_var]
                        bias_ele = biases.isel(elevation=i_elev).values
                        std_ele  = stds.isel(elevation=i_elev).values

                        if np.all(np.isnan(bias_ele)):
                            continue

                        all_biases_ele.append(bias_ele)
                        all_stds_ele.append(std_ele)
                        all_labels_ele.append(var_label)
                        n_valid_min = int(n_valid.isel(elevation=i_elev).min().values)
                        del stds, biases, n_valid

                    if not all_biases_ele:
                        continue

                    plot_bias_lines(
                        all_biases_ele, all_labels_ele, channels, channel_labels,
                        elev, n_valid_min,
                        save_path=os.path.join(bias_dir,
                            f"MWRREF_{sky}_elev{elev:.1f}_{campaign}_{location}_bias.png"),
                        campaign=campaign, location=location, tag=sky,
                    ylims_bias=[-15,15],
                   grid_params=(-15, 15.0001, 2),
                    ref_label=ref_label 
                    )
                    plot_bias_std_lines(
                        all_biases_ele, all_stds_ele, all_labels_ele, channels, channel_labels,
                        elev, n_valid_min,
                        save_path=os.path.join(bias_std_dir,
                            f"MWRREF_{sky}_elev{elev:.1f}_{campaign}_{location}_bias_std.png"),
                        campaign=campaign, location=location, tag=sky,
                    ylims_bias=[-15,15],
                   grid_params=(-15, 15.0001, 2),
                    ref_label=ref_label 
                    )        
            del ds_sel  # nach der sky-Schleife






