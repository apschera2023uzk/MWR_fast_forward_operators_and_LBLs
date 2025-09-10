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
        "TBs_RTTOV_gb_nc", "TBs_ARMS_gb", "TBs_R17", "IWV_hamhat",\
         "LWP_hamhat"]
    # ,
    roof_variables = ["TBs_RTTOV_gb_cropped",\
        "TBs_R24_cropped", "TBs_R03_cropped", "TBs_R16_cropped",\
        "TBs_R19_cropped", "TBs_R98_cropped", "TBs_R19SD_cropped",\
        "TBs_R20_cropped", "TBs_R20SD_cropped", "TBs_joyhat",\
        "TBs_RTTOV_gb_nc_cropped", "TBs_ARMS_gb_cropped",\
        "TBs_R17_cropped", "IWV_joyhat", "LWP_joyhat"]
        
    ds_yard = ds[yard_variables].where(ds["elevation2"]>89.5)
    ds_roof = ds[roof_variables].where(ds["elevation"]>89.5)\
        .where(ds["TBs_joyhat"]>0.000001)
    
    
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

    #################
    # New availability plots:
    n_roof = create_data_avail_plot(ds_roof, tag=tag+" roof ",\
        out=os.path.expanduser("~/PhD_plots/availability/"))
    n_yard = create_data_avail_plot(ds_yard, tag=tag+" yard ",\
        out=os.path.expanduser("~/PhD_plots/availability/"))
    #################
    
    # 1st derive mean TBs of all models per channel
    # mean_by_channel_roof, mean_by_channel_yard =\
    #    derive_mean_of_all_channels(ds_yard, ds_roof)
    
    # 2nd derive difference of mean for single model from combined mean
    colors=["blue", "orange", "green", "purple", "brown", "pink",\
         "gray", "red","olive", "cyan", "indigo", "darkgreen", "coral", "black"]
    markers = ["X", "o", "+", "<"]
                 
    yard_reference = "TBs_R24"
    yard_vars = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_ARMS_gb"]
    roof_reference = "TBs_R24_cropped"
    roof_vars = ["TBs_RTTOV_gb_cropped", "TBs_joyhat", "TBs_ARMS_gb_cropped"]
    
    # New bias plot yard:
    plt.figure()
    plt.title(f"HATPRO channels bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})\nnumber of sondes: "+str(n_yard))
    plt.plot(np.arange(1,15), [-0.5]*14, color="red", linestyle="dashed")
    plt.plot(np.arange(1,15), [0.5]*14, color="red", linestyle="dashed")
    plt.plot(np.arange(1,15), [0]*14, color="black")
    for i, var in enumerate(yard_vars):
            plt.scatter(np.arange(1,15), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[:],\
                label=f"Bias {var}", marker=markers[i], color=colors[i])     
            plt.plot(np.arange(1,15), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[:],\
                color=colors[i])                                                        
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"All_channels_yard.png", dpi=600 , bbox_inches="tight")

    # New bias plot yard:
    plt.figure()
    plt.title(f"K-band channels bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})\nnumber of sondes: "+str(n_yard))
    plt.plot(np.arange(1,8), [-0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(1,8), [0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    for i, var in enumerate(yard_vars):
            plt.scatter(np.arange(1,8), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[:7],\
                label=f"Bias {var}", marker=markers[i], color=colors[i])  
            plt.plot(np.arange(1,8), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[:7],\
                color=colors[i])                                                       
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K-band_yard.png", dpi=600 , bbox_inches="tight")
    
    # New bias plot yard:
    plt.figure()
    plt.title(f"V-band channels bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})\nnumber of sondes: "+str(n_yard))
    plt.plot(np.arange(8,15), [-0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(8,15), [0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    for i, var in enumerate(yard_vars):
            plt.scatter(np.arange(8,15), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[7:],\
                label=f"Bias {var}", marker=markers[i], color=colors[i])     
            plt.plot(np.arange(8,15), (ds_yard.mean(dim="time",\
                skipna=True)[var].values-ds_yard.mean(dim="time",\
                skipna=True)[yard_reference].values)[7:],\
                color=colors[i])
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V-Band_yard.png", dpi=600 , bbox_inches="tight")
    
    # New bias plot roof all:
    plt.figure()
    plt.title(f"HATPRO channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})\nnumber of sondes: "+str(n_roof))
    plt.plot(np.arange(1,15), [-0.5]*14, color="red", linestyle="dashed")
    plt.plot(np.arange(1,15), [0.5]*14, color="red", linestyle="dashed")
    plt.plot(np.arange(1,15), [0]*14, color="black")
    for i, var in enumerate(roof_vars):
            plt.scatter(np.arange(1,15), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[:],\
                label=f"Bias {var}", marker=markers[i], color=colors[i])    
            plt.plot(np.arange(1,15), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[:],\
                color=colors[i])                                                       
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"All_channels_roof.png", dpi=600 , bbox_inches="tight")

    # New bias plot roof k:
    plt.figure()
    plt.title(f"K-band channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})\nnumber of sondes: "+str(n_roof))
    plt.plot(np.arange(1,8), [-0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(1,8), [0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    for i, var in enumerate(roof_vars):
            plt.scatter(np.arange(1,8), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[:7],\
                label=f"Bias {var}", marker=markers[i], color=colors[i])  
            plt.plot(np.arange(1,8), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[:7],\
                color=colors[i])                                                         
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K-band_roof.png", dpi=600 , bbox_inches="tight")
    
    # New bias plot roof v:
    plt.figure()
    plt.title(f"V-band channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})\nnumber of sondes: "+str(n_roof))
    plt.plot(np.arange(8,15), [-0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(8,15), [0.5]*7, color="red", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    for i, var in enumerate(roof_vars):
            plt.scatter(np.arange(8,15), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[7:],\
                label=f"Bias {var}", marker=markers[i], color=colors[i])     
            plt.plot(np.arange(8,15), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-ds_roof.mean(dim="time",\
                skipna=True)[roof_reference].values)[7:],\
                color=colors[i])                                                        
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V-Band_roof.png", dpi=600 , bbox_inches="tight")
    plt.close("all")
    
    return 0

##############################################################################

def create_data_avail_plot(ds, tag="any tag", out=""):
    # make data availability plot for all dates:
    if "yard" in tag:
        obs = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_R24", "TBs_ARMS_gb"]
    elif "roof" in tag:
        obs = ["TBs_RTTOV_gb_cropped", "TBs_joyhat", "TBs_R24_cropped",\
            "TBs_ARMS_gb_cropped"]
    else:
        obs = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_joyhat", "TBs_R24", "TBs_ARMS_gb"]
    # prtlib; rttov-gb; hamhat; joyhat; potentially arms-gb
    # => cropped versions extra?
    
    print("Data availability for"+tag+": ",\
        ds["time"].values[np.invert(np.isnan(\
        ds[obs[0]].mean(dim="frequency").values))])
    df = create_statistics_dataframe(ds, tag=tag)
    ##########
    error_by_IWV_or_LWP(ds, tag =tag)
    
    n_obs = len(obs)
    n_time = len(ds["time"].values)
    availability = np.zeros([n_obs, n_time])
    ## Simulated availability (1 = available, 0 = missing)
    
    for i, variable in enumerate(obs):

        nan_indx = np.invert(np.isnan(ds[variable].mean(dim="frequency").values))
        availability[i,nan_indx] = 1.  
        zero_indx = ds[variable].mean(dim="frequency").values == 0.
        availability[i,zero_indx] = 0.   
    
    fig, ax = plt.subplots(figsize=(25, 8))
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

    
    # Gitterlinien setzen
    ax.set_xticks(np.arange(availability.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(availability.shape[0]+1)-0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    # Label axes
    ax.set_yticks(range(n_obs))
    ax.set_yticklabels([obs[i] for i in range(n_obs)])
    ax.set_xlabel("Radiosonde number")
    time_strings = pd.to_datetime(ds["time"].values).strftime('%d. Aug -  %H:%M')
    ax.set_xticks(ticks=range(len(time_strings)), labels=time_strings,\
        rotation=90)

    # caculate sonde number:
    n_sondes = np.count_nonzero(np.mean(availability, axis=0) == 1)
    
    plt.title("Data availability Vital I zenith by sonde no ("+tag+")\n\
        number of usable sondes: "+str(n_sondes))
    plt.colorbar(cax, label="Availability")
    # plt.tight_layout()

    plt.savefig(out+tag+"data_availability.png", dpi=600 ,\
        bbox_inches="tight")
    plt.close("all")
    return n_sondes

##############################################################################

def create_single_sonde_TSI_plot(ds, tag="any tag", out="",\
        picture_dir="/home/aki/PhD_data/TSI_42_Vital_I/used/"):
    ds_yard, ds_roof = divide2roof_and_yard_sets(ds)
    
    # 1st derive mean TBs of all models per channel
    # mean_by_channel_roof, mean_by_channel_yard =\
    #    derive_mean_of_all_channels(ds_yard, ds_roof)
    
    # 2nd derive difference of mean for single model from combined mean
    colors=["blue", "orange", "green", "purple", "brown", "pink",\
         "gray", "red","olive", "cyan", "indigo", "darkgreen", "coral", "black"]
    markers = ["X", "o", "+", "<"]
                 
    yard_reference = "TBs_R24"
    yard_vars = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_ARMS_gb"]
    roof_reference = "TBs_R24_cropped"
    roof_vars = ["TBs_RTTOV_gb_cropped", "TBs_joyhat", "TBs_ARMS_gb_cropped"]
    
    relevant_times = ds_zen_clear["time"].values[np.invert(np.isnan(\
        ds_zen_clear["TBs_RTTOV_gb"].mean(dim="frequency").values))]
    bool_array = np.invert(np.isnan(ds_zen_clear["TBs_RTTOV_gb"]\
                            .mean(dim="frequency").values))
    # n_sondes = len(relevant_times)
    indices = np.where(bool_array)[0].tolist()

    # Loop:
    for i, timestep in enumerate(relevant_times):
        # print(i, timestep)
        
        filestr = str(timestep).replace(":","").replace("-","")\
            .replace("T","_")[0:9]
        if len(glob.glob(picture_dir+filestr+"*")) ==1:
            img_path = glob.glob(picture_dir+filestr+"*")[0]
        else:
            numbers = []
            for j in range(len(glob.glob(picture_dir+filestr+"*"))):
                hhmm = int(glob.glob(picture_dir+filestr+"*")[j].split("/")[-1]\
                    .split("_")[-1][:4])
                numbers.append(hhmm)
            comp = int(str(timestep).replace(":","").replace("-","")\
                .replace("T","_")[9:13])
            index = np.argmin(abs(comp-np.array(numbers)))
            img_path = glob.glob(picture_dir+filestr+"*")[index]
        
        # print(img_path)
        img = Image.open(img_path)
        
        # 14 channels roof
        fig, axs = plt.subplots(1, 2, figsize=(13, 7))
        axs[0].set_title(f"All channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})")
        axs[0].plot(np.arange(1, 15), [-0.5]*14, color="red", linestyle="dashed")
        axs[0].plot(np.arange(1, 15), [0.5]*14, color="red", linestyle="dashed")
        axs[0].plot(np.arange(1, 15), [0]*14, color="black")
        for j, var in enumerate(roof_vars):
            bias = (ds_roof[var].values[indices[i],:] -
                    ds_roof[roof_reference].values[indices[i],:])
            axs[0].scatter(np.arange(1, 15), bias[:], label=f"Bias {var}",\
                           marker=markers[j], color=colors[j])
            axs[0].plot(np.arange(1, 15), bias[:], color=colors[j])
        axs[0].set_ylim(-3, 3)
        axs[0].legend(loc='lower right', fontsize=9)
        # --- Rechte Seite: das Bild
        axs[1].imshow(img)
        axs[1].axis('off')  # keine Achsen beim Bild anzeigen
        axs[1].set_title("TSI imager "+str(timestep)[0:16])
        # Layout anpassen
        plt.tight_layout()
        # Speichern
        # print(out+tag+str(timestep)[0:16]+"TSI_and_single_sonde.png")
        plt.savefig(out+"TSI_1_sonde/"+tag+str(timestep)[0:16]+\
                    "TSI_1sonde_allchans_roof.png",\
                    dpi=600 , bbox_inches="tight")
        plt.close("all")
        
        # K-band roof
        fig, axs = plt.subplots(1, 2, figsize=(13, 7))
        axs[0].set_title(f"K-band channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})")
        axs[0].plot(np.arange(1, 8), [-0.5]*7, color="red", linestyle="dashed")
        axs[0].plot(np.arange(1, 8), [0.5]*7, color="red", linestyle="dashed")
        axs[0].plot(np.arange(1, 8), [0]*7, color="black")
        for j, var in enumerate(roof_vars):
            bias = (ds_roof[var].values[indices[i],:] -
                    ds_roof[roof_reference].values[indices[i],:])
            axs[0].scatter(np.arange(1, 8), bias[:7], label=f"Bias {var}",\
                           marker=markers[j], color=colors[j])
            axs[0].plot(np.arange(1, 8), bias[:7], color=colors[j])
        axs[0].set_ylim(-3, 3)
        axs[0].legend(loc='lower right', fontsize=9)
        # --- Rechte Seite: das Bild
        axs[1].imshow(img)
        axs[1].axis('off')  # keine Achsen beim Bild anzeigen
        axs[1].set_title("TSI imager "+str(timestep)[0:16])
        # Layout anpassen
        plt.tight_layout()
        # Speichern
        # print(out+tag+str(timestep)[0:16]+"TSI_and_single_sonde.png")
        plt.savefig(out+"TSI_1_sonde/"+tag+str(timestep)[0:16]+\
                    "TSI_1sonde_Kband_roof.png",\
                    dpi=600 , bbox_inches="tight")
        plt.close("all")
        
        # V-band roof
        fig, axs = plt.subplots(1, 2, figsize=(13, 7))
        axs[0].set_title(f"V-band channels bias against Rosenkranz 24\nVital I (roof / Joyhat / {tag})")
        axs[0].plot(np.arange(8, 15), [-0.5]*7, color="red", linestyle="dashed")
        axs[0].plot(np.arange(8, 15), [0.5]*7, color="red", linestyle="dashed")
        axs[0].plot(np.arange(8, 15), [0]*7, color="black")
        for j, var in enumerate(roof_vars):
            bias = (ds_roof[var].values[indices[i],:] -
                    ds_roof[roof_reference].values[indices[i],:])
            axs[0].scatter(np.arange(8, 15), bias[7:], label=f"Bias {var}",\
                           marker=markers[j], color=colors[j])
            axs[0].plot(np.arange(8, 15), bias[7:], color=colors[j])
        axs[0].set_ylim(-3, 3)
        axs[0].legend(loc='lower right', fontsize=9)
        # --- Rechte Seite: das Bild
        axs[1].imshow(img)
        axs[1].axis('off')  # keine Achsen beim Bild anzeigen
        axs[1].set_title("TSI imager "+str(timestep)[0:16])
        # Layout anpassen
        plt.tight_layout()
        # Speichern
        # print(out+tag+str(timestep)[0:16]+"TSI_and_single_sonde.png")
        plt.savefig(out+"TSI_1_sonde/"+tag+str(timestep)[0:16]+\
                    "TSI_1sonde_Vband_roof.png",\
                    dpi=600 , bbox_inches="tight")
        plt.close("all")

        # 14 channels yard:
        fig, axs = plt.subplots(1, 2, figsize=(13, 7))
        axs[0].set_title(f"All channels bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})")
        axs[0].plot(np.arange(1, 15), [-0.5]*14, color="red", linestyle="dashed")
        axs[0].plot(np.arange(1, 15), [0.5]*14, color="red", linestyle="dashed")
        axs[0].plot(np.arange(1, 15), [0]*14, color="black")
        for j, var in enumerate(yard_vars):
            bias = (ds_yard[var].values[indices[i],:] -
                    ds_yard[yard_reference].values[indices[i],:])
            axs[0].scatter(np.arange(1, 15), bias[:], label=f"Bias {var}",\
                           marker=markers[j], color=colors[j])
            axs[0].plot(np.arange(1, 15), bias[:], color=colors[j])
        axs[0].set_ylim(-3, 3)
        axs[0].legend(loc='lower right', fontsize=9)
        # --- Rechte Seite: das Bild
        axs[1].imshow(img)
        axs[1].axis('off')  # keine Achsen beim Bild anzeigen
        axs[1].set_title("TSI imager "+str(timestep)[0:16])
        # Layout anpassen
        plt.tight_layout()
        # Speichern
        # print(out+tag+str(timestep)[0:16]+"TSI_and_single_sonde.png")
        plt.savefig(out+"TSI_1_sonde/"+tag+str(timestep)[0:16]+\
                    "TSI_1sonde_allchans_yard.png",\
                    dpi=600 , bbox_inches="tight")
        plt.close("all")
        
        # K-band yard:
        fig, axs = plt.subplots(1, 2, figsize=(13, 7))
        axs[0].set_title(f"K-band bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})")
        axs[0].plot(np.arange(1, 8), [-0.5]*7, color="red", linestyle="dashed")
        axs[0].plot(np.arange(1, 8), [0.5]*7, color="red", linestyle="dashed")
        axs[0].plot(np.arange(1, 8), [0]*7, color="black")
        for j, var in enumerate(yard_vars):
            bias = (ds_yard[var].values[indices[i],:] -
                    ds_yard[yard_reference].values[indices[i],:])
            axs[0].scatter(np.arange(1, 8), bias[:7], label=f"Bias {var}",\
                           marker=markers[j], color=colors[j])
            axs[0].plot(np.arange(1, 8), bias[:7], color=colors[j])
        axs[0].set_ylim(-3, 3)
        axs[0].legend(loc='lower right', fontsize=9)
        # --- Rechte Seite: das Bild
        axs[1].imshow(img)
        axs[1].axis('off')  # keine Achsen beim Bild anzeigen
        axs[1].set_title("TSI imager "+str(timestep)[0:16])
        # Layout anpassen
        plt.tight_layout()
        # Speichern
        # print(out+tag+str(timestep)[0:16]+"TSI_and_single_sonde.png")
        plt.savefig(out+"TSI_1_sonde/"+tag+str(timestep)[0:16]+\
                    "TSI_1sonde_K-band_yard.png",\
                    dpi=600 , bbox_inches="tight")
        plt.close("all")
        
        # V-band yard:
        fig, axs = plt.subplots(1, 2, figsize=(13, 7))
        axs[0].set_title(f"V-band bias against Rosenkranz 24\nVital I (yard / Hamhat / {tag})")
        axs[0].plot(np.arange(8, 15), [-0.5]*7, color="red", linestyle="dashed")
        axs[0].plot(np.arange(8, 15), [0.5]*7, color="red", linestyle="dashed")
        axs[0].plot(np.arange(8, 15), [0]*7, color="black")
        for j, var in enumerate(yard_vars):
            bias = (ds_yard[var].values[indices[i],:] -
                    ds_yard[yard_reference].values[indices[i],:])
            axs[0].scatter(np.arange(8, 15), bias[7:], label=f"Bias {var}",\
                           marker=markers[j], color=colors[j])
            axs[0].plot(np.arange(8, 15), bias[7:], color=colors[j])
        axs[0].set_ylim(-3, 3)
        axs[0].legend(loc='lower right', fontsize=9)
        # --- Rechte Seite: das Bild
        axs[1].imshow(img)
        axs[1].axis('off')  # keine Achsen beim Bild anzeigen
        axs[1].set_title("TSI imager "+str(timestep)[0:16])
        # Layout anpassen
        plt.tight_layout()
        # Speichern
        # print(out+tag+str(timestep)[0:16]+"TSI_and_single_sonde.png")
        plt.savefig(out+"TSI_1_sonde/"+tag+str(timestep)[0:16]+\
                    "TSI_1sonde_V-band_yard.png",\
                    dpi=600 , bbox_inches="tight")
        plt.close("all")
        
    return 0

##############################################################################

def bias_by_channel(ds, var, ref_var):
    bias_array = ds.mean(dim="time",\
        skipna=True)[ref_var].values[:]-ds.mean(dim="time",\
        skipna=True)[var].values
    return bias_array

##############################################################################

def std_by_channel(ds, var, ref_var):
    # Equation taken from SHi et al. 2024 preprint / 2025
    std_array = []
    for i in range(len(ds["frequency"].values)):
        deviation = ds[var].values[:, i] -ds[ref_var].values[:,i]
        avg = np.sqrt(np.nansum((ds[var].values[:, i] -\
            ds[ref_var].values[:,i])**2)/len(ds["time"]))
        std = np.sqrt(np.nansum((deviation-avg)**2)/len(ds["time"]))
        std_array.append(std)
    return np.array(std_array)
    
##############################################################################

def rmse_by_channel(ds, var, ref_var):
    rmse_array = []
    for i in range(len(ds["frequency"].values)):
        rmse_array.append(np.sqrt(np.nansum((ds[var].values[:, i] -\
            ds[ref_var].values[:,i])**2)/len(ds["time"])))
    return np.array(rmse_array)
    
##############################################################################

def bias_by_time(ds, var, ref_var):
    bias_array = ds.mean(dim="frequency",\
        skipna=True)[ref_var].values[:]-ds.mean(dim="frequency",\
        skipna=True)[var].values
    return bias_array

##############################################################################

def std_by_time(ds, var, ref_var):
    # Equation taken from SHi et al. 2024 preprint / 2025
    std_array = []
    for i in range(len(ds["time"].values)):
        deviation = ds[var].values[i,:] -ds[ref_var].values[i,:]
        avg = np.sqrt(np.nansum((ds[var].values[i,:] -\
            ds[ref_var].values[i,:])**2)/len(ds["frequency"]))
        std = np.sqrt(np.nansum((deviation-avg)**2)/len(ds["frequency"]))
        std_array.append(std)
    return np.array(std_array)
    
##############################################################################

def rmse_by_time(ds, var, ref_var):
    rmse_array = []
    for i in range(len(ds["time"].values)):
        rmse_array.append(np.sqrt(np.nansum((ds[var].values[i,:] -\
            ds[ref_var].values[i,:])**2)/len(ds["frequency"])))
    return np.array(rmse_array)

##############################################################################

def create_statistics_dataframe(ds, tag ="any tag", outdir="~/PhD_data/"):
    df_rmse = pd.DataFrame()
    df_std = pd.DataFrame()
    df_bias = pd.DataFrame()
    df_bias["frequency"] = ds["frequency"].values
    
    if "roof" in tag:
        ds_roof = ds
    elif "yard" in tag:
        ds_yard = ds
    else:
        ds_yard, ds_roof = divide2roof_and_yard_sets(ds)
        
    # 2nd derive difference of mean for single model from combined mean
    colors=["blue", "orange", "green", "purple", "brown", "pink",\
         "gray", "red","olive", "cyan", "indigo", "darkgreen", "coral", "black"]
    markers = ["X", "o", "+", "<"]
                 
    yard_reference = "TBs_R24"
    yard_vars = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_ARMS_gb"]
    roof_reference = "TBs_R24_cropped"
    roof_vars = ["TBs_RTTOV_gb_cropped", "TBs_joyhat", "TBs_ARMS_gb_cropped"]
    
    if "yard" in tag:
        for var in yard_vars :
            df_bias[var+"_bias"] = bias_by_channel(ds_yard, var, yard_reference)
            df_std[var+"_std"] = std_by_channel(ds_yard, var, yard_reference)
            df_rmse[var+"_rmse"] = rmse_by_channel(ds_yard, var, yard_reference)
    if "roof" in tag:
        for var in roof_vars:
            df_bias[var+"_bias"] = bias_by_channel(ds_roof, var, roof_reference)
            df_std[var+"_std"] = std_by_channel(ds_roof, var, roof_reference)
            df_rmse[var+"_rmse"] = rmse_by_channel(ds_roof, var, roof_reference)

    df = df_bias.join(df_std).join(df_rmse)
    df.to_csv(outdir+tag+"statistics_viatl_I.csv")
    
    return df
    
##############################################################################

def IWV_channel_plots(ds,\
        vartag="None", outdir="", tag=None):

    yard_reference = "TBs_R24"
    yard_vars = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_ARMS_gb"]
    roof_reference = "TBs_R24_cropped"
    roof_vars = ["TBs_RTTOV_gb_cropped", "TBs_joyhat", "TBs_ARMS_gb_cropped"]
        
    if "yard" in tag:
        bias_comp = bias_by_time(ds, vartag, yard_reference)
        std_comp = std_by_time(ds, vartag, yard_reference)
        rmse_comp = rmse_by_time(ds, vartag, yard_reference)
        lwp_var_np = ds["LWP_hamhat"].values
        iwv_var_np = ds["IWV_hamhat"].values
        '''
        print("yard shape: ", np.shape(iwv_var_np))
        print("yard shape: ", np.shape(lwp_var_np))
        print("bias shape: ", np.shape(bias_comp))
        print("std shape: ", np.shape(std_comp))
        print("rmse shape: ", np.shape(rmse_comp))
        '''
    if "roof" in tag:
        bias_comp = bias_by_time(ds, vartag, roof_reference)
        std_comp = std_by_time(ds, vartag, roof_reference)
        rmse_comp = rmse_by_time(ds, vartag, roof_reference)
        lwp_var_np = ds["LWP_joyhat"].values[:,0]
        iwv_var_np = ds["IWV_joyhat"].values[:,0]
        '''
        print("roof shape: ", np.shape(iwv_var_np)) # 30
        print("roof shape: ", np.shape(lwp_var_np))
        print("bias shape: ", np.shape(bias_comp))
        print("std shape: ", np.shape(std_comp))
        print("rmse shape: ", np.shape(rmse_comp))
        '''
        
    ############
    # Bias:
    # Plot IWV: 
    plt.figure(figsize=(10,10))
    plt.title("Bias of "+vartag+" by IWV - "+tag)
    plt.scatter(iwv_var_np, bias_comp, marker="X")
    plt.ylabel("Bias in K")
    plt.xlabel("IWV [kg m-2]")
    plt.savefig(outdir+vartag+"bias_by_IWV.png",dpi=600 , bbox_inches="tight")
    
    # Plot LWP: 
    plt.figure(figsize=(10,10))
    plt.title("Bias of "+vartag+" by LWP - "+tag)
    plt.scatter(lwp_var_np, bias_comp, marker="X")
    plt.ylabel("Bias in K")
    plt.xlabel("LWP [kg m-2]")
    plt.savefig(outdir+vartag+"bias_by_LWP.png",dpi=600 , bbox_inches="tight")

    ############
    # Std:
    # Plot IWV: 
    plt.figure(figsize=(10,10))
    plt.title("Standard deviation of "+vartag+" by IWV - "+tag)
    plt.scatter(iwv_var_np, std_comp)
    plt.ylabel("Std in K")
    plt.xlabel("IWV [kg m-2]")
    plt.savefig(outdir+vartag+"std_by_IWV.png",dpi=600 , bbox_inches="tight")
    
    # Plot LWP: 
    plt.figure(figsize=(10,10))
    plt.title("Standard deviation of "+vartag+" by LWP - "+tag)
    plt.scatter(lwp_var_np, std_comp)
    plt.ylabel("Std in K")
    plt.xlabel("LWP [kg m-2]")
    plt.savefig(outdir+vartag+"std_by_LWP.png",dpi=600 , bbox_inches="tight")
    
    ############
    # RMSE:
    # Plot IWV: 
    plt.figure(figsize=(10,10))
    plt.title("RMSE of "+vartag+" by IWV - "+tag)
    plt.scatter(iwv_var_np, rmse_comp)
    plt.ylabel("RMSE in K")
    plt.xlabel("IWV [kg m-2]")
    plt.savefig(outdir+vartag+"rmse_by_IWV.png",dpi=600 , bbox_inches="tight")
    
    # Plot LWP: 
    plt.figure(figsize=(10,10))
    plt.title("RMSE of "+vartag+" by LWP - "+tag)
    plt.scatter(lwp_var_np, rmse_comp)
    plt.ylabel("RMSE in K")
    plt.xlabel("LWP [kg m-2]")
    plt.savefig(outdir+vartag+"rmse_by_LWP.png",dpi=600 , bbox_inches="tight")

    ###########
    # But also by channel? slopes or correlations as one plot?
    ###########

    return 0
##############################################################################

def error_by_IWV_or_LWP(ds, tag =" any tag ", outdir="/home/aki/PhD_plots/IWV_LWP/"):
    
    if "yard" in tag:
        ds_yard = ds
    elif "roof" in tag:
        ds_roof = ds
    else:
        ds_yard, ds_roof = divide2roof_and_yard_sets(ds)
    
    # 1st derive mean TBs of all models per channel
    #mean_by_channel_roof, mean_by_channel_yard =\
    #    derive_mean_of_all_channels(ds_yard, ds_roof)
    
    # 2nd derive difference of mean for single model from combined mean
    colors=["blue", "orange", "green", "purple", "brown", "pink",\
         "gray", "red","olive", "cyan", "indigo", "darkgreen", "coral", "black"]
    markers = ["X", "o", "+", "<"]
                 
    yard_reference = "TBs_R24"
    yard_vars = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_ARMS_gb"]
    roof_reference = "TBs_R24_cropped"
    roof_vars = ["TBs_RTTOV_gb_cropped", "TBs_joyhat", "TBs_ARMS_gb_cropped"]
    
    relevant_times = ds_zen_clear["time"].values[np.invert(np.isnan(\
        ds_zen_clear["TBs_RTTOV_gb"].mean(dim="frequency").values))]
    bool_array = np.invert(np.isnan(ds_zen_clear["TBs_RTTOV_gb"]\
                            .mean(dim="frequency").values))
    indices = np.where(bool_array)[0].tolist()

    if "yard" in tag:
        for var in yard_vars:
            # print("VAR: ", var)
            IWV_channel_plots(ds_yard.isel(time=indices),\
	        vartag=var, outdir=outdir, tag=tag)
	 
    if "roof" in tag: 
        for var in roof_vars:
            # print("VAR: ", var)
            IWV_channel_plots(ds_roof.isel(time=indices),\
	        vartag=var, outdir=outdir, tag=tag)

    return 0

##############################################################################

def shi_plot_bias_std(df, tag =" any tag "):
    return 0

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
    ds_zen_all = ds.where(ds["mean_rainfall"]<0.000001)
    ds_zen_clear = ds_zen_all.where(ds["cloud_flag"]==0)\
    .where(ds["mean_rainfall"]<0.000001)
    
    '''
    #print("First filter: ")    
    #print("All sky sondes: ",\
    #    ds_zen_all["time"].values[np.invert(np.isnan(\
    #    ds_zen_all["TBs_RTTOV_gb"].mean(dim="frequency").values))])
    '''

    #############
    # Complete data availability:
    # create_data_avail_plot(ds, tag="all_data",\
    #     out=os.path.expanduser("~/PhD_plots/availability/"))
    create_data_avail_plot(ds_zen_clear, tag="clear_sky",\
        out=os.path.expanduser("~/PhD_plots/availability/"))
    create_data_avail_plot(ds_zen_all, tag="all_sky",\
        out=os.path.expanduser("~/PhD_plots/availability/"))
        
    print("Processing clear sky zenith...")
    # shi_plot_bias_std(df, tag =" clear_sky ")
    error_by_IWV_or_LWP(ds_zen_clear, tag =" clear_sky ")
    create_single_sonde_TSI_plot(ds_zen_clear, tag=" clear_sky ",\
        out=output_dir)
    bias_plot_by_R24(ds_zen_clear, tag=" clear_sky ",\
        out=args.output2)

    print("Processing all sky zenith...")#
    create_single_sonde_TSI_plot(ds_zen_all, tag=" all_sky ",\
        out=output_dir)#
    error_by_IWV_or_LWP(ds_zen_clear, tag =" clear_sky ")
    bias_plot_by_R24(ds_zen_all, tag=" all_sky ",\
        out=args.output2)



















