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
    bad_mask = (cf == 1) if sky == "clear" else (cf == 0)

    ds_cf = ds_sel.copy()
    for var in ds_cf.data_vars:
        if "elevation" not in ds_cf[var].dims:
            continue
        ds_cf[var] = ds_cf[var].where(~bad_mask.broadcast_like(ds_cf[var]))

    return ds_cf

##############################################################################

def create_plot_dirs(ds, args, campaign=None, location=None):

    if not campaign:
        campaign = ds["Campaign"].values[0]
        location = ds["Location"].values[0]   
    site_dir       = os.path.join(args.output, campaign+"_"+location)
    os.makedirs(site_dir, exist_ok=True)
        
    std_dir       = os.path.join(site_dir, "std")
    bias_dir      = os.path.join(site_dir, "bias")
    bias_std_dir  = os.path.join(site_dir, "bias_std")
    os.makedirs(std_dir, exist_ok=True)
    os.makedirs(bias_dir, exist_ok=True)
    os.makedirs(bias_std_dir, exist_ok=True)
    return std_dir, bias_dir, bias_std_dir

##############################################################################

def create_plot_by_chan_and_ele(ds, stds, rmses, biases, n_valid, label,\
        ref_label, pearsons_rs, elevations=elevations, campaign="any_campaign",\
        location="any_location",args=None, tag="any_tag"):
    ###
    # Outdirs:::
    std_dir, bias_dir, bias_std_dir = create_plot_dirs(ds, args,\
        campaign=campaign, location=location)
    channels = np.arange(14)+1
    elev_idcs = np.arange(8)
    valid_tag = (np.min(n_valid.values), np.max(n_valid.values))

    ###
    # Plot of Std:
    fig, ax = plt.subplots(figsize=(16, 9))
    norm = colors.LogNorm(vmin=0.25, vmax=10.) 
    c = ax.pcolormesh(
        channels,
        elev_idcs,
        stds[:,:8].T,          
        cmap="viridis",
        #################
        # vmin=0.0,
        # vmax=3.0,      
        norm=norm ,
        ###################
        shading="auto"
    )
    CS = ax.contour(
        channels,
        elev_idcs,
        stds[:,:8].T,     
        levels=[0.25, 0.5,1, 2, 3],
        colors=["white", "red", "black", "purple", "black"],
        linewidths=1.0
    )
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Elevation [deg]")
    elev_tags = [str(elev) for elev in elevations[:8]]
    ax.set_yticks(elev_idcs, elev_tags)
    title = f"Standard deviation of TB per Channel/Elevation\n\
        n_valid={valid_tag}, {location}, {campaign}, {label}-{ref_label}, {tag}"
    ax.set_title(title)
    cb = fig.colorbar(c, ax=ax)
    ticks = [0.25, 0.5, 1, 2, 3, 5,7.5, 10]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])  # explizit lineare Labels
    cb.set_label("Standard deviation TB [K]")
    out_path = f"{std_dir}/{tag}_std_chan_ele_{campaign}_{location}_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    ###
    # Plot of Bias:
    fig, ax = plt.subplots(figsize=(16, 9))
    norm = colors.SymLogNorm(linthresh=0.25, vmin=-15, vmax=15.) 
    c = ax.pcolormesh(
        channels,
        elev_idcs,
        biases[:,:8].T,          
        cmap="bwr",
        ##############
        #vmin=-2.0,
        #vmax=2.0,
        norm=norm,      
        ############## 
        shading="auto"
    )
    CS = ax.contour(
        channels,
        elev_idcs,
        biases[:,:8].T,     
        levels=[-2, -1,-0.5,-0.25,0.25, 0.5,1, 2],
        colors=["red", "black", "purple"],
        linewidths=1.0
    )
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Elevation [deg]")
    elev_tags = [str(elev) for elev in elevations[:8]]
    ax.set_yticks(elev_idcs, elev_tags)
    title = f"Bias of TB per Channel/Elevation\n\
        n_valid={valid_tag}, {location}, {campaign}, {label}-{ref_label}, {tag}"
    ax.set_title(title)
    cb = fig.colorbar(c, ax=ax)
    ticks = [-15,-7.5, -5,-3, -2,-1,-0.5,-0.25, 0.25, 0.5, 1, 2, 3,5,7.5,15]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])  # explizit lineare Labels
    cb.set_label("Bias TB [K]")
    out_path = f"{bias_dir}/{tag}_bias_chan_ele_{campaign}_{location}_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    ###
    # Plot of RMSE:
    fig, ax = plt.subplots(figsize=(16, 9))
    norm = colors.LogNorm(vmin=0.25, vmax=10.)
    c = ax.pcolormesh(
        channels,
        elev_idcs,
        rmses[:,:8].T,          
        cmap="viridis",
        ##########
        #vmin=0.0,
        # vmax=3.0,       
        norm = norm,
        shading="auto"
    )
    CS = ax.contour(
        channels,
        elev_idcs,
        rmses[:,:8].T,     
        levels=[0.25, 0.5,1, 2, 3],
        colors=["white", "red", "black", "purple", "black"],
        linewidths=1.0
    )
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Elevation [deg]")
    elev_tags = [str(elev) for elev in elevations[:8]]
    ax.set_yticks(elev_idcs, elev_tags)
    title = f"RMSE of TB per Channel/Elevation\n\
        n_valid={valid_tag}, {location}, {campaign}, {label}-{ref_label}, {tag}"
    ax.set_title(title)
    cb = fig.colorbar(c, ax=ax)
    ticks = [0.25, 0.5, 1, 2, 3, 5,7.5, 10]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])  # explizit lineare Labels
    cb.set_label("RMSE TB [K]")
    out_path = f"{bias_std_dir}/{tag}_RMSE_chan_ele_{campaign}_{location}_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    ###
    # Plot of Pearsons R:
    fig, ax = plt.subplots(figsize=(16, 9))
    c = ax.pcolormesh(
        channels,
        elev_idcs,
        pearsons_rs[:,:8].T,          
        cmap="viridis",
        vmin=-1,
        vmax=1,       
        shading="auto"
    )
    CS = ax.contour(
        channels,
        elev_idcs,
        pearsons_rs[:,:8].T,     
        levels=[-0.5, -0.25,0, 0.25, 0.5],
        colors=["white", "red", "black", "red", "white"],
        linewidths=1.0
    )
    ax.clabel(CS, inline=True, fontsize=8, fmt="%.2f")
    ax.set_xlabel("Channel")
    ax.set_ylabel("Elevation [deg]")
    elev_tags = [str(elev) for elev in elevations[:8]]
    ax.set_yticks(elev_idcs, elev_tags)
    title = f"Pearson's R of TB per Channel/Elevation\n\
        n_valid={valid_tag}, {location}, {campaign}, {label}-{ref_label}, {tag}"
    ax.set_title(title)
    cb = fig.colorbar(c, ax=ax)
    ticks = [-1,-0.5, -0.25,0, 0.25, 0.5, 1.]
    cb.set_ticks(ticks)
    cb.set_ticklabels([str(t) for t in ticks])  # explizit lineare Labels
    cb.set_label("correlation r")
    out_path = f"{bias_std_dir}/{tag}_Pearson_corr_chan_ele_{campaign}_{location}_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    ###
    # Plot of N_VALID: bar plot per elevation
    fig, ax = plt.subplots(figsize=(12, 7))

    # n_valid is (N_Channels, elevation) — take mean over channels (should be same per elev)
    n_valid_per_elev = n_valid[:, :8].mean(dim="N_Channels").values.astype(int)

    bars = ax.bar(
        np.arange(len(elev_idcs)),
        n_valid_per_elev,
        color="steelblue", edgecolor="black", alpha=0.8
    )

    # Write value on top of each bar
    for bar, val in zip(bars, n_valid_per_elev):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            str(val),
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_xticks(np.arange(len(elev_idcs)))
    ax.set_xticklabels(elev_tags, rotation=45, ha="right")
    ax.set_xlabel("Elevation [deg]")
    ax.set_ylabel("N valid timesteps")
    ax.set_title(
        f"Number of valid timesteps per Elevation\n"
        f"{location}, {campaign}, {label}-{ref_label}, [{tag}]"
    )
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    out_path = f"{bias_std_dir}/{tag}_nvalid_per_elev_{campaign}_{location}_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    return 0

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
                for dev_var, var_label, ref_label in zip(dev_vars, var_labels,\
                            ref_labels):
                    if ds_cf[dev_var].isna().all():
                        continue
                    else:
                        print("Processing Variable:", dev_var)
                    stds, rmses, biases, n_valid, var_label, ref_label,\
                        pearsons_rs = derive_stats_per_chan_and_elev(ds_cf,\
                        dev_var, var_label, ref_label)
                    create_plot_by_chan_and_ele(ds_cf, stds, rmses, biases,\
                        n_valid, var_label, ref_label, pearsons_rs,\
                        campaign=campaign,location=location ,args=args, tag=sky)
                    del stds, rmses, biases, n_valid, pearsons_rs
            del ds_sel  # nach der sky-Schleife


