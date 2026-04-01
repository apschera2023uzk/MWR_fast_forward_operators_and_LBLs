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

'''
# Clear sky LWP threshold


model_tbs=["TBs_PyRTlib_R24",'TBs_RTTOV_gb', 'TBs_ARMS_gb']
mwr_vars = ['TBs_dwdhat', 'TBs_sunhat', 'TBs_tophat',\
        'TBs_joyhat'] #, 'TBs_foghat', 'TBs_hamhat']
ref_label = "PyRTlib R24 LBL"
n_elev = 10


'''

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

def select_ds_camp_loc(ds, campaign, location, crop_index=0):
    # Filter Datensatz nach Kampagne und Ort
    mask = (ds["Campaign"] == campaign) & (ds["Location"] == location)
    ds_sel = ds.sel(time=ds["time"].values[mask.values]).isel(Crop=crop_index)
    return ds_sel

##############################################################################

def plot_std_bars(ds, stds, labels, channels, channel_labels, elev,
        n_valid, save_path,elevations=elevations, thres_lwp=thres_lwp,\
        label_colors=label_colors,\
        ylims_bias=ylims_bias, tag="any_tag"):
        
    campaign = ds["Campaign"].values[0]
    location = ds["Location"].values[0]
    
    fig, ax = plt.subplots(figsize=(14,9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}, \nelevation: {elevations[elev]:.1f}°, (N: {n_valid} / LWP < {thres_lwp} kg m-2) {tag}")
    ax.set_title("Standard deviation of brightness temperature deviation")
    
    width = 0.2  # Width of each bar
    offsets = np.linspace(-0.3,0.3,len(stds))
    
    for std, lbl, offset in zip(stds, labels,offsets):
        if lbl.strip() != " (MWR)":  # Only plot if label is not empty
            lbl = lbl.strip()
            if lbl == "":
                continue  # Skip empty labels
            color = label_colors.get(lbl, "black")  
            ax.bar(channels + offset, std, width, label=lbl, color=color, alpha=0.7)
    
    ax.set_yticks(np.arange(0,4.01,0.5)) 
    ax.set_xticks(channels)
    ax.set_xticklabels(channel_labels)
    ax.set_xlabel("Channels")
    ax.set_ylim(0, 3)
    ax.set_ylabel("Standard Deviation of Brightness Temperature [K]")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=600)
    plt.close()

##############################################################################

def ensure_folder_exists(base_path, folder_name):
    # Join the base path with the folder name
    folder_path = os.path.join(base_path, folder_name)
    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    return os.path.abspath(folder_path)

##############################################################################
##############################################################################




##############################################################################
##############################################################################
'''
def plot_bias_lines(ds, all_biases, all_labels, channels, channel_labels, elev,
                   n_valid, save_path, elevations=elevations,\
                   thres_lwp=thres_lwp, ref_label=ref_label,\
                   label_colors=label_colors, ylims_bias=ylims_bias,\
                    tag="any_tag"): 

    campaign = ds["Campaign"].values[0]
    location = ds["Location"].values[0]
    
    fig, ax = plt.subplots(figsize=(14, 9))
    elev_label = elevations[elev] if elevations is not None else elev
    
    title_campaign = campaign if campaign is not None else "Unknown Campaign"
    title_location = location if location is not None else "Unknown Location"
    
    plt.suptitle(f"Campaign: {title_campaign}, location: {title_location}, \n"
                 f"elevation: {elev_label:.1f}°, (N: {n_valid} / {tag})")
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
    
    ax.set_yticks(np.arange(*grid_params)) 
    ax.set_xticks(channels)
    ax.set_xticklabels(channel_labels)
    ax.set_xlabel("Channels")
    ax.set_xlim(channels.min() - 1, channels.max() + 1)
    ax.set_ylabel("Bias of Brightness Temperature [K]")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(ylims_bias)  # Adjust as needed for your data
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

##############################################################################

def plot_bias_std_lines(ds, all_biases, all_stds, all_labels, channels,\
        channel_labels, elev,n_valid, save_path, elevations=elevations,\
        thres_lwp=thres_lwp, ref_label=ref_label,\
        label_colors=label_colors, ylims_bias=ylims_bias, tag="any_tag"):
 
    campaign = ds["Campaign"].values[0]
    location = ds["Location"].values[0]
    
    elev_label = elevations[elev] if elevations is not None else elev
    
    plt.figure(figsize=(14, 9))
    plt.suptitle(f"Campaign: {campaign}, location: {location}, \n"
                 f"elevation: {elev_label:.1f}°, (N: {n_valid} / LWP < {thres_lwp} kg m-2){tag}")
    
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
    
    plt.yticks(np.arange(*grid_params)) 
    plt.xticks(channels, channel_labels)
    plt.xlim(channels.min() - 1, channels.max() + 1)
    plt.ylim(ylims_bias)  # Adjust as needed
    
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()
    
##############################################################################

def check_model_and_mwr_data_availability(ds, elev_idx, ref_mod,\
        model_tbs,azi_idx=0, mwr_vars=mwr_vars,
        old_valid_mwrs=[]):
        
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
    print("ref model: ", ref_mod)
    print("len common_valid and elev_index: ", len(common_valid_mask), elev_idx)
    reference_tb = ds[ref_mod].values[common_valid_mask, :, elev_idx]
    print("Shape of reference TBs before mean: ", np.shape(reference_tb))
            
    return all_values, reference_tb
    
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

def analyze_mwr_model_stats(ds, args, elev_idx=0, model_tbs=model_tbs,\
        elevations=elevations, ref_mod=model_tbs[0],\
        n_chans=n_chans,mwr_vars=mwr_vars,azi_idx=0, thres_lwp=thres_lwp,
        old_valid_mwrs=[], tag="any_tag"):
      
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
    
        print("***")
        print("Label: ", label)
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
        f"{tag}_{ds['Campaign'].values[0]}_{ds['Location'].values[0]}_elevation_{elevations[elev_idx]}_std_by_channel_{thres_lwp}.png"
    plot_std_bars(ds, plot_stds, plot_labels, channels, channel_labels,\
          elev_idx, n_valid, save_path, tag=tag)

    # Plot Bias:
    save_path = bias_dir+"/" \
        f"{tag}_{ds['Campaign'].values[0]}_{ds['Location'].values[0]}_elevation_{elevations[elev_idx]}_bias_by_channel_{thres_lwp}.png"
    plot_bias_lines(ds, all_biases, all_labels, channels, channel_labels,\
         elev_idx, n_valid, save_path, tag=tag)
        
        
    # Plot bias and std:
    save_path = bias_std_dir+"/" \
        f"{tag}_{ds['Campaign'].values[0]}_{ds['Location'].values[0]}_elevation_{elevations[elev_idx]}_bias_std_by_channel_{thres_lwp}.png"    
    plot_bias_std_lines(ds, all_biases, all_stds, all_labels, channels,\
        channel_labels, elev_idx,n_valid, save_path, tag=tag)

    plt.close("all")
    return plot_stds, all_biases, all_rmses, n_valid, plot_labels, valid_mwrs,\
        all_values, reference_tb, common_valid_mask

##############################################################################

def create_plot_by_chan_and_ele(camp_loc_stat_dict, elevations=elevations,\
        campaign="any_campaign", location="any_location",args=None,\
        tag="any_tag"):
    stds_array=camp_loc_stat_dict["stds"]
    rmses_array=camp_loc_stat_dict["rmses"]
    biases_array=camp_loc_stat_dict["biases"]
    n_valid=np.min(np.array(camp_loc_stat_dict["n_valid"]))
    labels=camp_loc_stat_dict["labels"]
    channels = np.arange(14)+1
    elev_idcs = np.arange(8)

    # print(labels); 8-times the same instruments / models
    std_dir, bias_dir, bias_std_dir = create_plot_dirs(ds, args,\
        campaign=campaign, location=location)

    for i, label in enumerate(labels[0]):
        
        stds = stds_array[i, :, :]
        rmses = rmses_array[i, :, :]
        biases = biases_array[i, :, :]

        ###
        # Plot of Std:
        fig, ax = plt.subplots(figsize=(16, 9))
        norm = colors.LogNorm(vmin=0.25, vmax=10.) 
        c = ax.pcolormesh(
            channels,
            elev_idcs,
            stds.T,          
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
            stds.T,     
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
            n_valid={n_valid}, {location}, {campaign}, {label}-R24 PyRTlib, {tag}"
        ax.set_title(title)
        cb = fig.colorbar(c, ax=ax)
        ticks = [0.25, 0.5, 1, 2, 3, 5,7.5, 10]
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])  # explizit lineare Labels
        cb.set_label("Standard deviation TB [K]")
        out_path = f"{std_dir}/{tag}_std_chan_ele_{campaign}_{location}_{label}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=600)
        plt.close(fig)

        ###
        # Plot of Bias:
        fig, ax = plt.subplots(figsize=(16, 9))
        norm = colors.SymLogNorm(linthresh=0.25, vmin=-15, vmax=15.) 
        c = ax.pcolormesh(
            channels,
            elev_idcs,
            biases.T,          
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
            biases.T,     
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
            n_valid={n_valid}, {location}, {campaign}, {label}-R24 PyRTlib, {tag}"
        ax.set_title(title)
        cb = fig.colorbar(c, ax=ax)
        ticks = [-15,-7.5, -5,-3, -2,-1,-0.5,-0.25, 0.25, 0.5, 1, 2, 3,5,7.5,15]
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])  # explizit lineare Labels
        cb.set_label("Bias TB [K]")
        out_path = f"{bias_dir}/{tag}_bias_chan_ele_{campaign}_{location}_{label}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=600)
        plt.close(fig)

        ###
        # Plot of RMSE:
        fig, ax = plt.subplots(figsize=(16, 9))
        norm = colors.LogNorm(vmin=0.25, vmax=10.)
        c = ax.pcolormesh(
            channels,
            elev_idcs,
            rmses.T,          
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
            rmses.T,     
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
            n_valid={n_valid}, {location}, {campaign}, {label}-R24 PyRTlib, {tag}"
        ax.set_title(title)
        cb = fig.colorbar(c, ax=ax)
        ticks = [0.25, 0.5, 1, 2, 3, 5,7.5, 10]
        cb.set_ticks(ticks)
        cb.set_ticklabels([str(t) for t in ticks])  # explizit lineare Labels
        cb.set_label("RMSE TB [K]")
        out_path = f"{bias_std_dir}/{tag}_RMSE_chan_ele_{campaign}_{location}_{label}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=600)
        plt.close(fig)

    plt.close("all")
    return 0

##############################################################################

def extract_IWV_timelines_of_camp_loc(ds_sel, campaign, location):
    
    if campaign == "FESSTVaL" and location == "RAO_Lindenberg":
        iwv_var1 = "Dwdhat_IWV"
        iwv_var2 = "Foghat_IWV"
        lwp_var1 = "Dwdhat_LWP"
        lwp_var2 = "Foghat_LWP"
        hua_var1 = "Dwdhat_hua"
        hua_var2 = "Foghat_hua"

        # DWDhat: BL-scans and zenith
        # Foghat: Only 30° or 90°

    elif campaign == "Vital I" and location == "JOYCE":
        iwv_var1 = "Joyhat_IWV"
        iwv_var2 = "Hamhat_IWV"
        lwp_var1 = "Joyhat_LWP"
        lwp_var2 = "Hamhat_LWP"
        hua_var1 = "Joyhat_hua"
        hua_var2 = "Hamhat_hua"

        # Joyhat: BL scans and 30° Azimuth and Zenith!!!
        # Hamhat: Only BL-scans and Zenith and loads of clutter!

    elif campaign == "FESSTVaL" and location == "Falkenberg":
        iwv_var1 = "Sunhat_IWV"
        lwp_var1 = "Sunhat_LWP"
        hua_var1 = "Sunhat_hua"
        iwv_var2 = None
        lwp_var2 = None
        hua_var2 = None

        # Sunhat: BL-scans and zenith

    elif campaign == "Socles" and location == "JOYCE":
        iwv_var1 = "Tophat_IWV"
        lwp_var1 = "Tophat_LWP"
        hua_var1 = "Tophat_hua"
        iwv_var2 = None
        lwp_var2 = None
        hua_var2 = None

        # Tophat: BL-scans and zenith

    return ds_sel[iwv_var1], ds_sel[lwp_var1], ds_sel[hua_var1]



##############################################################################

def plot_departures_vs_iwv(camp_loc_stat_dict, all_departures_arms_gb,\
        all_departures_rttov_gb, all_iwvs, elevations=None, channels=14,\
        campaign="any", location="any", tag="any_tag",
                          outpath="./"):

    ###
    # 1st plots:
    # RTTOV-gb vs. IWV:
    # all channels all elevations!   
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    # y_all = np.nanmean(all_departures_rttov_gb, axis=1)   # (288*14,)    
    y_all = (all_departures_rttov_gb).flatten() 
    x_all = np.tile(all_iwvs, 14) 
    scatter1 = ax1.scatter(x_all, y_all,marker="X",color="red")
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.set_ylabel('RTTOV-gb deviations from R24 [K]')
    ax1.set_xlabel('IWV [kg m-2]')
    ax1.set_xlim(0,45)
    if np.nanmax(np.abs(y_all))<14:
        ax1.set_ylim(-13,+13)
    ax1.set_title(f'RTTOV-gb deviations from R24 PyRTlib by IWV\n\
            ({campaign}_{location}_{tag})')
    plt.savefig(outpath+\
        f"{tag}_All_elevs_chans_RTTOV_by_IWV_{location}_{campaign}.png",\
         dpi=300, bbox_inches='tight')
    plt.close("all")

    ###
    # 1st plots:
    # ARMS-gb vs. IWV:
    # all channels all elevations!   
    fig1, ax1 = plt.subplots(figsize=(10, 10))
    # y_all = np.nanmean(all_departures_arms_gb, axis=1)   # (288*14,)  
    y_all = (all_departures_arms_gb).flatten()
    x_all = np.tile(all_iwvs, 14) 
    scatter1 = ax1.scatter(x_all, y_all,marker="X",color="red")
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.set_ylabel('ARMS-gb deviations from R24 [K]')
    ax1.set_xlabel('IWV [kg m-2]')
    ax1.set_xlim(0,45)
    if np.nanmax(np.abs(y_all))<14:
        ax1.set_ylim(-13,+13)
    ax1.set_title(f'ARMS-gb deviations from R24 PyRTlib by IWV\n\
            ({campaign}_{location}_{tag})')
    plt.savefig(outpath+\
        f"{tag}_All_elevs_chans_ARMS_by_IWV_{location}_{campaign}.png",\
         dpi=300, bbox_inches='tight')
    plt.close("all")

    for i in range(channels):

        ###
        # 1st plots:
        # RTTOV-gb vs. IWV:
        # Each channel all elevations!   
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        y_all = all_departures_rttov_gb[:,i]   
        scatter1 = ax1.scatter(all_iwvs, y_all,marker="X",color="red")
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax1.set_ylabel('RTTOV-gb deviations from R24 [K]')
        ax1.set_xlabel('IWV [kg m-2]')
        ax1.set_xlim(0,45)
        if np.nanmax(np.abs(y_all))<=4:
            axis_len = 4
        elif np.nanmax(np.abs(y_all))<=13:
            axis_len = 13
        if np.nanmax(np.abs(y_all))<=13:
            ax1.set_ylim(-axis_len,+axis_len)
        ax1.set_title(f'Channel {i+1} RTTOV-gb deviations from R24 PyRTlib by IWV\n\
                ({campaign}_{location}_{tag})')
        plt.savefig(outpath+\
            f"{tag}_ch{i+1}_RTTOV_by_IWV_{location}_{campaign}.png",\
             dpi=300, bbox_inches='tight')
        plt.close("all")

        ###
        # 1st plots:
        # ARMS-gb vs. IWV:
        # Each channel all elevations!     
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        y_all = all_departures_arms_gb[:,i]      
        scatter1 = ax1.scatter(all_iwvs, y_all,marker="X",color="red")
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax1.set_ylabel('ARMS-gb deviations from R24 [K]')
        ax1.set_xlabel('IWV [kg m-2]')
        ax1.set_xlim(0,45)
        if np.nanmax(np.abs(y_all))<=4:
            axis_len = 4
        elif np.nanmax(np.abs(y_all))<=13:
            axis_len = 13
        if np.nanmax(np.abs(y_all))<=13:
            ax1.set_ylim(-axis_len,+axis_len)
        ax1.set_title(f'Channel {i+1} ARMS-gb deviations from R24 PyRTlib by IWV\n\
                ({campaign}_{location}_{tag})')
        plt.savefig(outpath+\
            f"{tag}_ch{i+1}_ARMS_by_IWV_{location}_{campaign}.png",\
             dpi=300, bbox_inches='tight')
        plt.close("all")

    #############################################
    # Read variables form dictionary:
    stds_array=camp_loc_stat_dict["stds"]
    rmses_array=camp_loc_stat_dict["rmses"]
    biases_array=camp_loc_stat_dict["biases"]
    iwv_np_array = camp_loc_stat_dict["IWV"].values
    n_valid=np.min(np.array(camp_loc_stat_dict["n_valid"]))
    labels=camp_loc_stat_dict["labels"]
    non_ref_tbs = camp_loc_stat_dict["all_non_ref_tbs"]
    ref_tbs = camp_loc_stat_dict["all_ref_tbs"]
    com_val_mask = camp_loc_stat_dict["vald_masks"]

    for i, (labelset,ref_tbs, tbs, val_mask) in\
            enumerate(zip(labels,ref_tbs, non_ref_tbs, com_val_mask)):
        for j in range(len(labelset)):
            if "ARMS" in labelset[j]:
                arms_idx = j
            elif "RTTOV" in labelset[j]:
                rttov_idx = j
        if i==0:
            all_departures_rttov_gb = tbs[rttov_idx]-ref_tbs
            all_departures_arms_gb = tbs[arms_idx]-ref_tbs
            all_iwvs = iwv_np_array[val_mask]
        else:
            all_departures_arms_gb = np.concatenate(\
                    [all_departures_arms_gb, tbs[arms_idx]-ref_tbs], axis=0)
            all_departures_rttov_gb = np.concatenate(\
                    [all_departures_rttov_gb, tbs[rttov_idx]-ref_tbs], axis=0)
            all_iwvs = np.concatenate([all_iwvs,iwv_np_array[val_mask]], axis=0)

        ##################
        ###
        # 8 plots for each elevation and every channel
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        # y_all = np.nanmean((tbs[rttov_idx]-ref_tbs), axis=1)
        y_all = (tbs[rttov_idx]-ref_tbs).flatten()
        x_all = np.tile(iwv_np_array[val_mask], 14)
        scatter1 = ax1.scatter(x_all, y_all,marker="X",color="red")
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax1.set_xlabel('RTTOV-gb deviations from R24 [K]')
        ax1.set_xlabel('IWV [kg m-2]')
        ax1.set_xlim(0,45)
        if np.nanmax(np.abs(y_all))<14:
            ax1.set_ylim(-13,+13)
        ax1.set_title(f'RTTOV-gb deviations from R24 PyRTlib by IWV\n\
                Elevation: {elevations[i]}°_{campaign}_{location}_{tag})')
        plt.savefig(outpath+\
            f"{tag}_each_ele_RTTOV_by_IWV_{location}_{campaign}_{elevations[i]}.png",\
             dpi=300, bbox_inches='tight')

        ##################
        ###
        # 8 plots for each elevation and every channel
        fig1, ax1 = plt.subplots(figsize=(10, 10))
        # y_all = np.nanmean((tbs[arms_idx]-ref_tbs), axis=1)
        y_all = (tbs[arms_idx]-ref_tbs).flatten()
        x_all = np.tile(iwv_np_array[val_mask], 14)
        scatter1 = ax1.scatter(x_all, y_all,marker="X",color="red")
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax1.set_xlabel('ARMS-gb deviations from R24 [K]')
        ax1.set_xlabel('IWV [kg m-2]')
        if np.nanmax(np.abs(y_all))<14:
            ax1.set_ylim(-13,+13)
        ax1.set_xlim(0,45)
        ax1.set_title(f'ARMS-gb deviations from R24 PyRTlib by IWV\n\
                Elevation: {elevations[i]}°_{campaign}_{location}_{tag})')
        plt.savefig(outpath+\
            f"{tag}_each_ele_ARMS_by_IWV_{location}_{campaign}_{elevations[i]}.png",\
             dpi=300, bbox_inches='tight')
        plt.close("all")

    return 0

##############################################################################

def armsgb_vs_rttov_by_IWV(camp_loc_stat_dict, elevations=elevations,\
        campaign="any_campaign", location="any_location",args=None,\
        tag="any_tag"):

    # Read variables form dictionary:
    stds_array=camp_loc_stat_dict["stds"]
    rmses_array=camp_loc_stat_dict["rmses"]
    biases_array=camp_loc_stat_dict["biases"]
    iwv_np_array = camp_loc_stat_dict["IWV"].values
    n_valid=np.min(np.array(camp_loc_stat_dict["n_valid"]))
    labels=camp_loc_stat_dict["labels"]
    non_ref_tbs = camp_loc_stat_dict["all_non_ref_tbs"]
    ref_tbs = camp_loc_stat_dict["all_ref_tbs"]
    com_val_mask = camp_loc_stat_dict["vald_masks"]
    folder="ARMS_and_RTTOV_departures_against_IWV/"
    outpath = ensure_folder_exists(args.output+folder, campaign+"_"+location)+"/"
    ensure_folder_exists(outpath, "err_by_iwv")

    # First loop by elevation angle: 90° - 5.4°
    for i, (labelset,ref_tbs, tbs, val_mask) in\
            enumerate(zip(labels,ref_tbs, non_ref_tbs, com_val_mask)):
        for j in range(len(labelset)):
            if "ARMS" in labelset[j]:
                arms_idx = j
            elif "RTTOV" in labelset[j]:
                rttov_idx = j
        if i==0:
            all_departures_rttov_gb = tbs[rttov_idx]-ref_tbs
            all_departures_arms_gb = tbs[arms_idx]-ref_tbs
            all_iwvs = iwv_np_array[val_mask]
        else:
            all_departures_arms_gb = np.concatenate(\
                    [all_departures_arms_gb, tbs[arms_idx]-ref_tbs], axis=0)
            all_departures_rttov_gb = np.concatenate(\
                    [all_departures_rttov_gb, tbs[rttov_idx]-ref_tbs], axis=0)
            all_iwvs = np.concatenate([all_iwvs,iwv_np_array[val_mask]], axis=0)

        ##################
        ###
        # 8 plots for each elevation and every channel
        fig1, ax1 = plt.subplots(figsize=(12, 10))
        x_all = (tbs[arms_idx]-ref_tbs).flatten()
        y_all = (tbs[rttov_idx]-ref_tbs).flatten()
        colors_all = np.tile(iwv_np_array[val_mask], 14)
        scatter1 = ax1.scatter(x_all, y_all, c=colors_all, cmap='viridis',\
                s=40, alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ax1.set_xlabel('ARMS-gb deviations from R24 [K]')
        ax1.set_ylabel('RTTOV-gb deviations from R24 [K]')
        ax1.set_title(f'RTTOV-gb and ARMS-gb TB deviations from R24 PyRTlib by IWV\n\
                Elevation: {elevations[i]}°_{campaign}_{location}_{tag})')
        ############
        if np.nanmax(np.abs(y_all))<14:
            ax1.set_ylim(-13,+13)
        if np.nanmax(np.abs(x_all))<14:
            ax1.set_xlim(-13,+13)
        if np.nanmax(np.abs(x_all))<14 and np.nanmax(np.abs(y_all))<14:
            ax1.set_aspect('equal')
        ##############
        plt.colorbar(scatter1, ax=ax1, label='IWV [kg m-2]')
        plt.savefig(outpath+\
            f"{tag}_Each_elev_ARMS_RTTOV_by_IWV_{location}_{campaign}_{elevations[i]}.png",\
             dpi=300, bbox_inches='tight')

    ###
    # 1st plots:
    # Only ARMS-gb and RTTOV-gb minus reference;
    # all channels all elevations!   
    fig1, ax1 = plt.subplots(figsize=(12, 10))
    x_all = all_departures_arms_gb.flatten()    # (288*14,)
    y_all = all_departures_rttov_gb.flatten()    # (288*14,)
    colors_all = np.tile(all_iwvs, 14)            
    scatter1 = ax1.scatter(x_all, y_all, c=colors_all, cmap='viridis',\
            s=40, alpha=0.7)
    ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax1.set_xlabel('ARMS-gb deviations from R24 [K]')
    ax1.set_ylabel('RTTOV-gb deviations from R24 [K]')
    ############
    if np.nanmax(np.abs(y_all))<14:
        ax1.set_ylim(-13,+13)
    if np.nanmax(np.abs(x_all))<14:
        ax1.set_xlim(-13,+13)
    if np.nanmax(np.abs(x_all))<14 and np.nanmax(np.abs(y_all))<14:
        ax1.set_aspect('equal')
    ##############
    ax1.set_title(f'RTTOV-gb and ARMS-gb TB deviations from R24 PyRTlib by IWV\n\
            ({campaign}_{location}_{tag})')
    # ax1.set_aspect('equal')
    plt.colorbar(scatter1, ax=ax1, label='IWV [kg m-2]')
    plt.savefig(outpath+\
        f"{tag}_All_elevs_chans_ARMS_RTTOV_by_IWV_{location}_{campaign}.png",\
         dpi=300, bbox_inches='tight')
    plt.close("all")

    ########
    ###
    # 14 plots for each channel and every elevation!
    for idx in range(14):
        fig, ax = plt.subplots(figsize=(12, 10))  # Großes Format
        x_all = all_departures_arms_gb[:, idx]
        y_all = all_departures_rttov_gb[:, idx]
        colors_idx = all_iwvs[:]  # (288,)
        scatter = ax.scatter(x_all, y_all, c=colors_idx, cmap='viridis', 
                            s=40, alpha=0.7)
        ax.set_xlabel('ARMS-gb deviations from R24 [K]')
        ax.set_ylabel('RTTOV-gb deviations from R24 [K]')
        ax.set_title(f'TB deviations channel {idx+1} by IWV\n\
            ({campaign}_{location}_{tag})')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
        ############
        if idx>6:
            axis_len = 4
        elif idx<=6:
            axis_len = 13
        if np.nanmax(np.abs(y_all))<axis_len:
            ax.set_ylim(-axis_len,+axis_len)
        if np.nanmax(np.abs(x_all))<axis_len:
            ax.set_xlim(-axis_len,+axis_len)
        if np.nanmax(np.abs(x_all))<axis_len and np.nanmax(np.abs(y_all))<axis_len:
            ax.set_aspect('equal')
        ##############
        cbar = plt.colorbar(scatter, ax=ax, label='IWV [kg m$^{-2}$]')
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        # EINZELN SPEICHERN
        plt.savefig(outpath + 
                    f"{tag}_Channel_{idx+1}_ARMS_RTTOV_by_IWV_scatter_{location}_{campaign}.png", dpi=300, bbox_inches='tight')
        plt.close(fig)

    plot_departures_vs_iwv(camp_loc_stat_dict, all_departures_arms_gb,\
         all_departures_rttov_gb,\
        all_iwvs, elevations=elevations, channels=14, campaign=campaign,\
        location=location, tag=tag, outpath=outpath+"err_by_iwv/")

    plt.close("all")
    return 0
'''
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
    # campaign ="FESSTVaL"
    # location = "RAO_Lindenberg"
    # sky = "clear"  
    skies = ["clear", "cloudy", "all_sky"]

    for campaign in ds0["Campaign"].values:
        for location in ds0["Location"].values:  
            folder = ensure_folder_exists(args.output, campaign+"_"+\
                    location+"/std/")
            ds_sel = select_ds_camp_loc(ds0, campaign, location)
            for sky in skies:
                #########################################
                # 1.5 Determine clear / cloudy / complete!    
                if sky == "clear":
                    # Zeitschritte wo cloud_flag an ALLEN Elevationen 0 ist
                    time_mask = (ds_sel["cloud_flag"] == 0).all(dim="elevation")
                    ds_cf = ds_sel.isel(time=time_mask)
                elif sky == "cloudy":
                    time_mask = (ds_sel["cloud_flag"] == 1).any(dim="elevation")
                    ds_cf = ds_sel.isel(time=time_mask)
                elif sky == "all_sky":
                    ds_cf = ds_sel
                #############################################
                for dev_var, var_label, ref_label in zip(dev_vars, var_labels,\
                            ref_labels):
                    for i_elev, elev in enumerate(ds_cf["elevation"].values):
                        # ds_ele = ds_cf.sel(elevation=elevation)
                        
                        ###    
                        # Calculate stats and plot:
                        print(ds_cf)
                        stds, biass, rmses, n_valid = stats_by_channel(ds_cf,\
                            dev_var, i_elev)
                        plot_std_bars(ds_cf, stds, labels, channels,\
                            channel_labels, i_elev, n_valid, folder, tag=sky)


    ###################
    # Cloud flag does not take into account elevation cloud_flag!!!
    # Use Crop-Index 1 for Joyhat!!! somehow...!?!





