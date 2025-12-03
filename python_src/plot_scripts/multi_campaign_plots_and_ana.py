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


##############################################################################
# 1.5 Parameters:
##############################################################################
# Plotstyles:
fs = 20
plt.rc('font', size=fs) 
plt.style.use('seaborn-poster')
matplotlib.use("Qt5Agg")

# Clear sky LWP threshold
thres_lwp=0.04 # kg m-2
n_chans=14
model_tbs=["TBs_PyRTlib_R24",'TBs_RTTOV_gb', 'TBs_ARMS_gb']
mwr_vars = ['TBs_dwdhat', 'TBs_foghat', 'TBs_sunhat', 'TBs_tophat',\
        'TBs_joyhat', 'TBs_hamhat']
ref_label = "PyRTlib R24 LBL"
n_elev = 10
elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])

label_colors = {'Dwdhat (MWR)': "blue",
        'Foghat (MWR)': "green",'RTTOV-gb (model)': "red",
        'ARMS-gb (model)': "orange",'Sunhat (MWR)': "purple",
        'Tophat (MWR)': "brown", 'Joyhat (MWR)': "pink",
        'Hamhat (MWR)': "gray",
         "LABEL_9": "olive",
        "LABEL_10": "cyan"
}


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
        default=os.path.expanduser("~/PhD_data/TB_preproc_and_proc_results/3campaigns_3models_all_results.nc"),
        help="Input data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/PhD_plots/Nov_Dec_2025/"),
        help="Output plot directory"
    )

    return parser.parse_args()

##############################################################################

def clear_sky_dataset(ds, thres_lwp=thres_lwp):
    exclude =False
    exclude_times = []
    
    for i, timestamp in enumerate(ds["time"].values):
        
        water_sum = (
            np.nansum(np.nan_to_num(ds["Dwdhat_LWP"].values[i])) +
            np.nansum(np.nan_to_num(ds["Foghat_LWP"].values[i])) +
            np.nansum(np.nan_to_num(ds["Sunhat_LWP"].values[i])) +
            np.nansum(np.nan_to_num(ds["Tophat_LWP"].values[i])) +
            np.nansum(np.nan_to_num(ds["Joyhat_LWP"].values[i])) +
            np.nansum(np.nan_to_num(ds["Hamhat_LWP"].values[i]))
        )
        
        #######################################################
        # water_sum+=np.nansum(ds["LWP_radiosonde"].values[i,0])
        ############################################
        
        # water_sum in kg m-2
        # 15-40 g m-2 == 0.015-0.040
        if water_sum>thres_lwp:
            exclude=True
            # print("Water sum: ",water_sum)
        elif water_sum<=thres_lwp: # or np.isnan(water_sum):
            exclude=False

        if exclude:
            exclude_times.append(timestamp)
            exclude =False
    
    # Remove timesteps from ds:
    for timestamp in exclude_times:
         ds = ds.sel(time=ds.time != timestamp)
    
    return ds
    
##############################################################################

def stats_by_channel(values, references, n_chans=n_chans):
    # Equation taken from Shi et al. 2024 preprint / 2025
    # different sign than Shi et al!
    std_array = []
    bias_array = []
    rmse_array = []
    
    for i in range(n_chans):
        deviation = values[:,i] - references[:,i]
        avg = np.nansum((values[:,i] -references[:,i])/ len(ds["time"]))
        std = np.sqrt(np.nansum((deviation-avg)**2)/len(ds["time"]))
        rmse = np.sqrt(np.nansum((values[:,i] -references[:,i])**2/len(ds["time"])))
        std_array.append(std)
        bias_array.append(avg)
        rmse_array.append(rmse)
        
    return np.array(std_array), np.array(bias_array), rmse_array
    
##############################################################################

def select_ds_camp_loc(ds, campaign, location, crop_index=0):
    # Filter Datensatz nach Kampagne und Ort
    mask = (ds["Campaign"] == campaign) & (ds["Location"] == location)
    ds_sel = ds.sel(time=ds["time"].values[mask.values]).isel(Crop=crop_index)
    return ds_sel

##############################################################################

def plot_std_bars(ds, stds, labels, channels, channel_labels, elev,
        n_valid, save_path,elevations=elevations, thres_lwp=thres_lwp,\
        label_colors=label_colors):
        
    campaign = ds["Campaign"].values[0]
    location = ds["Location"].values[0]
    
    fig, ax = plt.subplots(figsize=(14,9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}, \nelevation: {elevations[elev]:.1f}°, (N: {n_valid} / LWP > {thres_lwp} kg m-2)")
    ax.set_title("Standard deviation of brightness temperature deviation")
    
    width = 0.2  # Width of each bar
    offsets = np.linspace(-width*1.5, width*1.5, len(stds))
    
    for std, lbl, offset in zip(stds, labels,offsets):
        if lbl.strip() != " (MWR)":  # Only plot if label is not empty
            lbl = lbl.strip()
            if lbl == "":
                continue  # Skip empty labels
            color = label_colors.get(lbl, "black")  
            ax.bar(channels + offset, std, width, label=lbl, color=color, alpha=0.7)
    
    ax.set_xticks(channels)
    ax.set_xticklabels(channel_labels)
    ax.set_xlabel("Channels")
    ax.set_xlim(0, 5)
    ax.set_ylabel("Standard Deviation of Brightness Temperature [K]")
    ax.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=600)
    plt.close()

##############################################################################
    
def plot_bias_lines(ds, all_biases, all_labels, channels, channel_labels, elev,
                   n_valid, save_path, elevations=elevations,\
                   thres_lwp=thres_lwp, ref_label=ref_label,\
                   label_colors=label_colors): 

    campaign = ds["Campaign"].values[0]
    location = ds["Location"].values[0]
    
    fig, ax = plt.subplots(figsize=(14, 9))
    elev_label = elevations[elev] if elevations is not None else elev
    
    title_campaign = campaign if campaign is not None else "Unknown Campaign"
    title_location = location if location is not None else "Unknown Location"
    
    plt.suptitle(f"Campaign: {title_campaign}, location: {title_location}, \n"
                 f"elevation: {elev_label:.1f}°, (N: {n_valid})")
    ax.set_title("Bias of brightness temperature (MWR-R24) / (model-R24)")
    
    ax.plot(channels, [0.] * len(channels), label=ref_label, color="black", linewidth=2)
    
    for bias, label in zip(all_biases, all_labels):
        label = label.strip()

        if label == "":
            continue  # Skip empty labels

        # Get color from dictionary, use fallback if missing
        color = label_colors.get(label, "black")  

        ax.plot(
            channels,
            bias,
            label=label,
            color=color,
            alpha=0.8,
            linewidth=2
        )
    
    ax.set_xticks(channels)
    ax.set_xticklabels(channel_labels)
    ax.set_xlabel("Channels")
    ax.set_xlim(channels.min() - 1, channels.max() + 1)
    ax.set_ylabel("Bias of Brightness Temperature [K]")
    ax.legend()
    ax.grid(True)
    ax.set_ylim([-5, 5])  # Adjust as needed for your data
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

##############################################################################

def plot_bias_std_lines(ds, all_biases, all_stds, all_labels, channels,\
        channel_labels, elev,n_valid, save_path, elevations=elevations,\
        thres_lwp=thres_lwp, ref_label=ref_label,\
        label_colors=label_colors):
 
    campaign = ds["Campaign"].values[0]
    location = ds["Location"].values[0]
    
    elev_label = elevations[elev] if elevations is not None else elev
    
    plt.figure(figsize=(14, 9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}, \n"
                 f"elevation: {elev_label:.1f}°, (N: {n_valid} / LWP > {thres_lwp} kg m-2)")
    
    plt.title("Bias and standard deviation of brightness temperature error")
    plt.xlabel("Channels")
    plt.ylabel("Bias of Brightness Temperature [K]")
    
    # Plot zero reference line
    plt.plot(channels, [0.] * len(channels), label=ref_label, color="black", linewidth=2)
    
    # Plot biases with lines and their std with fill_between shaded areas
    for bias, std, label in zip(all_biases, all_stds, all_labels):
        label = label.strip()
        if label == "":
            continue  # Skip empty labels
        # Get color from dictionary, use fallback if missing
        color = label_colors.get(label, "black")  
        plt.plot(channels, bias, label=label, color=color, alpha=0.8, linewidth=2)
        plt.fill_between(channels, bias - std, bias + std, color=color, alpha=0.2)
    
    plt.xticks(channels, channel_labels)
    plt.xlim(channels.min() - 1, channels.max() + 1)
    plt.ylim([-5, 5])  # Adjust as needed
    
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()
    
##############################################################################

def check_model_and_mwr_data_availability(ds, elev_idx, ref_mod,\
        model_tbs,azi_idx=0, mwr_vars=mwr_vars):
        
    # 1. Check MWRs:
    valid_mwrs = []
    mwr_labels = []
    
    print("Valid timesteps per instrument:")
    for var in mwr_vars:
        if var in ds:
            mwr_data = ds[var].values[:, elev_idx, azi_idx, :]  # Take first azimuth
            valid_times = np.sum(~np.isnan(mwr_data).all(axis=1))
            print(f"{var}: {valid_times} valid timesteps")
            if valid_times > 0:  # Include if any valid data
                valid_mwrs.append(var)
                # Extract instrument name from variable name
                name = var.replace('TBs_', '')
                mwr_labels.append(f"{name.title()} (MWR)")
    
    # 2. Check models
    valid_models = []
    model_labels = []
    for var in model_tbs:
        if var in ds:
            model_data = ds[var].values[:, :, elev_idx] 
            valid_times = np.sum(~np.isnan(model_data).all(axis=1))
            print(f"{var}: {valid_times} valid timesteps")
            if valid_times > 0:
                valid_models.append(var)
                model_labels.append(var.replace('TBs_', '').replace('_gb', '-gb') + " (model)")
    if ref_mod in valid_models:
        valid_models.remove(ref_mod)
    if ref_mod.replace('TBs_', '').replace('_gb', '-gb') + " (model)" in model_labels:
        model_labels.remove(ref_mod.replace('TBs_', '').replace('_gb', '-gb') + " (model)")
        
    return valid_mwrs, mwr_labels, valid_models, model_labels

##############################################################################

def valid_indices_and_count(ds, valid_mwrs, mwr_labels, valid_models,\
        model_labels,azi_idx=0):

    # Find common valid timesteps
    n_time = ds.sizes['time']
    common_valid_mask = np.ones(n_time, dtype=bool)
    for var in valid_mwrs + valid_models:
        if var in ds and 'TBs_' in var:
            if var in mwr_vars:
                data = ds[var].values[:, elev_idx, azi_idx, :]
            else:  # model
                data = ds[var].values[:, :, elev_idx]
            common_valid_mask &= ~np.isnan(data).all(axis=1)
    n_valid = np.sum(common_valid_mask)
    print(f"Common valid timesteps (n_valid): {n_valid}\n")

    return common_valid_mask, n_valid

##############################################################################

def get_all_TB_values_and_ref_TBs(ds, valid_mwrs, mwr_labels, valid_models,\
        model_labels,common_valid_mask,azi_idx=0, ref_mod=model_tbs[0]):
    all_values = []  # Model TBs
    
    # Stack valid MWRs as references
    for mwr_var in valid_mwrs:
        mwr_tb = ds[mwr_var].values[common_valid_mask, elev_idx, azi_idx, :]
        # Indices: 1. Chosen times, 2. one elev 3. one azimuth 4. all chans
        all_values.append(mwr_tb)
    
    # Models as values to compare against reference
    for i, model_var in enumerate(valid_models):
        model_tb = ds[model_var].values[common_valid_mask, :, elev_idx]
        all_values.append(model_tb)
            
    # Get reference TBs:
    reference_tb = ds[ref_mod].values[common_valid_mask, :, elev_idx]
            
    return all_values, reference_tb
    
##############################################################################

def create_plot_dirs(ds, args):

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

def analyze_mwr_model_stats(ds, args, elev_idx=0, model_tbs=model_tbs,\
        elevations=elevations, ref_mod=model_tbs[0],\
        n_chans=n_chans,mwr_vars=mwr_vars,azi_idx=0, thres_lwp=thres_lwp):
      
    # 1st Derive available MWRs and models:  
    valid_mwrs, mwr_labels, valid_models, model_labels =\
           check_model_and_mwr_data_availability(ds,\
           elev_idx, ref_mod, model_tbs)

    # 2nd Find number of valid timesteps and their indices:
    common_valid_mask, n_valid = valid_indices_and_count(ds, valid_mwrs,\
          mwr_labels, valid_models,model_labels)
    
    # 3rd Get all TB values of valid timestep and reference TBs:
    all_values, reference_tb = get_all_TB_values_and_ref_TBs(ds,\
        valid_mwrs, mwr_labels, valid_models, model_labels, common_valid_mask)
         
    ####   
    # 4th Calculate stats using loop
    all_stds = []
    all_biases = []
    all_rmses = []
    all_labels = []

    for values, label in zip(all_values, mwr_labels+model_labels):
    
        # print("Label: ", label)
        # print("value shape: ", np.shape(values))
    
        stds, biases, rmses = stats_by_channel(values, reference_tb, n_chans)
        all_stds.append(stds)
        all_labels.append(label)
        all_biases.append(biases)
        all_rmses.append(rmses)

    # Convert to numpy arrays for plotting (assuming all_stds is list of arrays)
    plot_stds = np.array(all_stds)  # Shape: (n_datasets, n_chans)
    plot_labels = all_labels
     ####  

    # 5th Create plots:
    channels = np.arange(n_chans)
    channel_labels = [f"Ch{i+1}" for i in range(n_chans)]  # Adjust as needed

    # Plot stds:
    std_dir, bias_dir, bias_std_dir = create_plot_dirs(ds, args)
    save_path = std_dir +"/" \
        f"{ds['Campaign'].values[0]}_{ds['Location'].values[0]}_elevation_{elevations[elev_idx]}_std_by_channel_{thres_lwp}.png"
    plot_std_bars(ds, plot_stds, plot_labels, channels, channel_labels,\
          elev_idx, n_valid, save_path)

    # Plot Bias:
    save_path = bias_dir+"/" \
        f"{ds['Campaign'].values[0]}_{ds['Location'].values[0]}_elevation_{elevations[elev_idx]}_bias_by_channel_{thres_lwp}.png"
    plot_bias_lines(ds, all_biases, all_labels, channels, channel_labels,\
         elev_idx, n_valid, save_path)
        
        
    # Plot bias and std:
    save_path = bias_std_dir+"/" \
        f"{ds['Campaign'].values[0]}_{ds['Location'].values[0]}_elevation_{elevations[elev_idx]}_bias_std_by_channel_{thres_lwp}.png"    
    plot_bias_std_lines(ds, all_biases, all_stds, all_labels, channels,\
        channel_labels, elev_idx,n_valid, save_path)

        
    return stds, biases, rmses, n_valid

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    nc_out_path=args.NetCDF
    
    # Open dataset and clear sky filtering
    ds = xr.open_dataset(nc_out_path)
    ds_clear = clear_sky_dataset(ds)
    
    for campaign in np.unique(ds['Campaign'].values):
        for location in np.unique(ds['Location'].values):
       
            # Filter data for campaign and location:
            ds_sel = select_ds_camp_loc(ds_clear, campaign, location)
            if len(ds_sel['time'])==0 or len(ds_sel["Campaign"].values)==0:
                continue
            print("\n\nCampaign / Location: ",campaign, "/", location)

            # 2nd Applying statistics to dataset:
            for elev_idx in range(n_elev):
                print("\nElevation index: ",elev_idx, " : ", elevations[elev_idx])
                stds, biases, rmses, n_valid = analyze_mwr_model_stats(ds_sel, args,\
                     elev_idx=elev_idx)

    
    # make some scatter plots - with RMSE std and bias

    







