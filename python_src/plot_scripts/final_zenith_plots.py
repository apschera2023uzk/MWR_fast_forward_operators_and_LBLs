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

# Plotstyles:
fs = 25
plt.rc('font', size=fs) 
plt.style.use('seaborn-poster')
matplotlib.use("Qt5Agg")


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
        default=os.path.expanduser("~/PhD_data/Vital_I_zenith_TBs.nc"),
        help="Input data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/PhD_plots/scatter/"),
        help="Output plot directory"
    )
    parser.add_argument(
        "--output2", "-o2",
        type=str,
        default=os.path.expanduser("~/PhD_plots/bias/"),
        help="Output plot directory"
    )

    return parser.parse_args()

##############################################################################

def derive_statistics(rs_vals, mwr_vals):
    # Statistiken
    mask = np.isfinite(mwr_vals) & np.isfinite(rs_vals)
    bias = np.mean(rs_vals[mask] - mwr_vals[mask])
    if np.sum(mask) > 1:
        corr, _ = pearsonr(mwr_vals[mask], rs_vals[mask])
    else:
        corr = np.nan
    rmse = np.sqrt(np.mean((rs_vals[mask] - mwr_vals[mask]) ** 2))
    return mask, bias, corr, rmse

##############################################################################

def divide2roof_and_yard_sets(ds):
    # , 
    yard_variables = ["TBs_RTTOV_gb", "TBs_R24",\
        "TBs_R03", "TBs_R16", "TBs_R19",\
        "TBs_R98", "TBs_R19SD",\
        "TBs_R20", "TBs_R20SD", "TBs_hamhat",\
        "TBs_RTTOV_gb_nc", "TBs_ARMS_gb", "TBs_R17"]
    # ,
    roof_variables = ["TBs_RTTOV_gb_cropped",\
        "TBs_R24_cropped", "TBs_R03_cropped", "TBs_R16_cropped",\
        "TBs_R19_cropped", "TBs_R98_cropped", "TBs_R19SD_cropped",\
        "TBs_R20_cropped", "TBs_R20SD_cropped", "TBs_joyhat",\
        "TBs_RTTOV_gb_nc_cropped", "TBs_ARMS_gb_cropped","TBs_R17_cropped"]
    
    ds_yard = ds[yard_variables]
    ds_roof = ds[roof_variables]
    return ds_yard, ds_roof
    
##############################################################################

def derive_mean_of_all_channels(ds_yard, ds_roof):

    # substract highest and lowest value (per channel)
    # and divide through i-1
    highs = [-10]*14
    lows = [500.]*14
    mean_by_channel_yard = np.array([0.]*14)
    for i, var in enumerate(ds_yard.data_vars):
        channel_mean_one_var = ds_yard.mean(dim="time", skipna=True)[var].values
        mean_by_channel_yard += channel_mean_one_var
        for j in range(14):
            if channel_mean_one_var[j]<lows[j]:
                lows[j] = channel_mean_one_var[j]
            if channel_mean_one_var[j]>highs[j]:
                highs[j] = channel_mean_one_var[j]
    mean_by_channel_yard = (mean_by_channel_yard-highs-lows)/(i-1)
    
    mean_by_channel_roof = np.array([0.]*14)
    highs = [-10]*14
    lows = [500.]*14
    for i, var in enumerate(ds_roof.data_vars):
        # if not np.isnan(ds_roof.mean(dim="time")[var].values)
        channel_mean_one_var = ds_roof.mean(dim="time", skipna=True)[var].values
        mean_by_channel_roof += channel_mean_one_var
        for j in range(14):
            if channel_mean_one_var[j]<lows[j]:
                lows[j] = channel_mean_one_var[j]
            if channel_mean_one_var[j]>highs[j]:
                highs[j] = channel_mean_one_var[j]
    mean_by_channel_roof = (mean_by_channel_roof-highs-lows)/(i-1)
    
    return mean_by_channel_roof, mean_by_channel_yard

##############################################################################

def bias_plot_by_R24(ds, tag="any tag", out=""):
    ds_yard, ds_roof = divide2roof_and_yard_sets(ds)
    
    # 1st derive mean TBs of all models per channel
    mean_by_channel_roof, mean_by_channel_yard =\
        derive_mean_of_all_channels(ds_yard, ds_roof)
    
    # 2nd derive difference of mean for single model from combined mean
    colors=["blue", "orange", "green", "purple", "brown", "pink",\
         "gray", "red","olive", "cyan", "indigo", "darkgreen", "coral", "black"]
                 
    yard_reference = "TBs_R24"
    yard_vars = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_ARMS_gb"]
    roof_reference = "TBs_R24_cropped"
    roof_vars = ["TBs_RTTOV_gb_cropped", "TBs_joyhat", "TBs_ARMS_gb_cropped"]

    # New bias plot yard:
    plt.figure()
    plt.title(f"HATPRO channels bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})")
    plt.plot(np.arange(1,15), [-0.5]*14, color="red", linestyle="dashed")
    plt.plot(np.arange(1,15), [0.5]*14, color="red", linestyle="dashed")
    plt.plot(np.arange(1,15), [0]*14, color="black")
    for i, var in enumerate(yard_vars):
            plt.scatter(np.arange(1,15), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[:],\
                label=f"Bias {var}", marker="X", color=colors[i])     
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"All_channels_yard.png")

    # New bias plot yard:
    plt.figure()
    plt.title(f"K-band channels bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})")
    plt.plot(np.arange(1,8), [-0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(1,8), [0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    for i, var in enumerate(yard_vars):
            plt.scatter(np.arange(1,8), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[:7],\
                label=f"Bias {var}", marker="X", color=colors[i])     
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K-band_yard.png")
    
    # New bias plot yard:
    plt.figure()
    plt.title(f"V-band channels bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})")
    plt.plot(np.arange(8,15), [-0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(8,15), [0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    for i, var in enumerate(yard_vars):
            plt.scatter(np.arange(8,15), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[7:],\
                label=f"Bias {var}", marker="X", color=colors[i])     
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V-Band_yard.png")
    
    # New bias plot roof all:
    plt.figure()
    plt.title(f"HATPRO channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})")
    plt.plot(np.arange(1,15), [-0.5]*14, color="red", linestyle="dashed")
    plt.plot(np.arange(1,15), [0.5]*14, color="red", linestyle="dashed")
    plt.plot(np.arange(1,15), [0]*14, color="black")
    for i, var in enumerate(roof_vars):
            plt.scatter(np.arange(1,15), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[:],\
                label=f"Bias {var}", marker="X", color=colors[i])     
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"All_channels_roof.png")

    # New bias plot roof k:
    plt.figure()
    plt.title(f"K-band channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})")
    plt.plot(np.arange(1,8), [-0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(1,8), [0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    for i, var in enumerate(roof_vars):
            plt.scatter(np.arange(1,8), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[:7],\
                label=f"Bias {var}", marker="X", color=colors[i])     
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K-band_roof.png")
    
    # New bias plot roof v:
    plt.figure()
    plt.title(f"V-band channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})")
    plt.plot(np.arange(8,15), [-0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(8,15), [0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    for i, var in enumerate(roof_vars):
            plt.scatter(np.arange(8,15), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[7:],\
                label=f"Bias {var}", marker="X", color=colors[i])     
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V-Band_roof.png")    
    plt.close("all")
    
    return 0

##############################################################################

def create_data_avail_plot(ds, tag="any tag", out=""):
    # make data availability plot for all dates:
    obs = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_joyhat", "TBs_R24", "TBs_ARMS_gb"]
    # prtlib; rttov-gb; hamhat; joyhat; potentially arms-gb
    # => cropped versions extra?
    
    n_obs = len(obs)
    n_time = len(ds["time"].values)
    availability = np.zeros([n_obs, n_time])
    ## Simulated availability (1 = available, 0 = missing)
    
    for i, variable in enumerate(obs):

        #############################
        # Zeros are not sorted out!!!
        # ARMS-gb is shown to be available!!!
        bool_indx = np.invert(np.isnan(ds[variable].mean(dim="frequency").values))
        availability[i,bool_indx] = 1.
        ##################
        # print("bool_indx: ", bool_indx) 
        # print("availability: ", availability)
    
    fig, ax = plt.subplots(figsize=(15, 8))
    # Plot colored grid
    cax = ax.imshow(availability, cmap="Greens",\
        aspect="auto", interpolation="nearest")
    # Add text labels
    for i in range(n_obs):
        for j in range(n_time):
            if availability[i, j] == 1:
                label = "Avail."
                ax.text(j, i, label, ha="center", va="center",\
                    color="black", fontsize=8)
            else:
                label= "X"
                ax.text(j, i, label, ha="center", va="center",\
                    color="red", fontsize=8)

    # Label axes
    ax.set_yticks(range(n_obs))
    ax.set_yticklabels([obs[i] for i in range(n_obs)])
    ax.set_xticks(range(0, n_time, 5))
    ax.set_xlabel("Radiosonde number")

    plt.title("Data availability Vital I zenith by sonde no ("+tag+")")
    plt.colorbar(cax, label="Availability")
    plt.tight_layout()

    plt.savefig(out+tag+"data_availability.png")
    plt.close("all")
    return

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    nc_out_path=args.NetCDF
    ds = xr.open_dataset(nc_out_path)
    channels = np.arange(1,15)
    output_dir = args.output

    # Filter for different issues:
    ds_zen_clear = ds.where(ds["cloud_flag"]==0).where(ds["elevation"]>89.5)\
    .where(ds["mean_rainfall"]<0.000001).where(ds["TBs_joyhat"]>0.000001)\
    .where(ds["elevation2"]>89.5)    
    print("Clear sky sondes: ",\
        ds_zen_clear["time"].values[np.invert(np.isnan(\
        ds_zen_clear["TBs_RTTOV_gb"].mean(dim="frequency").values))])
    
    ds_zen_all = ds.where(ds["elevation"]>89.5)\
    .where(ds["mean_rainfall"]<0.000001).where(ds["TBs_joyhat"]>0.000001)\
    .where(ds["elevation2"]>89.5)
    print("All sky sondes: ",\
        ds_zen_all["time"].values[np.invert(np.isnan(\
        ds_zen_all["TBs_RTTOV_gb"].mean(dim="frequency").values))])

    #############
        
    print("Processing clear sky zenith...")
    create_data_avail_plot(ds_zen_clear, tag="clear_sky",\
        out=os.path.expanduser("~/PhD_plots/availability/"))
    bias_plot_by_R24(ds_zen_clear, tag=" clear_sky ",\
        out=args.output2)

    print("Processing all sky zenith...")
    create_data_avail_plot(ds_zen_all, tag="all_sky",\
        out=os.path.expanduser("~/PhD_plots/availability/"))
    bias_plot_by_R24(ds_zen_all, tag=" all_sky ",\
        out=args.output2)



















