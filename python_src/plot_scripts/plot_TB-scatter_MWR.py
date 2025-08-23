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
        default=os.path.expanduser("~/PhD_data/combined_dataset.nc"),
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

def plot_tb_scatter_per_channel(rs_all, mwr_all, frequencies, output_dir, 
        label_rs="RTTOV-gb", label_mwr="Joyhat", campaign_name="Vital I",\
        fs=fs, tag="", folder=""):
    os.makedirs(output_dir, exist_ok=True)
    
    for ch in range(rs_all.shape[1]):
        rs_vals = rs_all[:, ch]
        mwr_vals = mwr_all[:, ch]

        mask, bias, corr, rmse = derive_statistics(rs_vals, mwr_vals)

        # Achsengrenzen
        all_vals = np.concatenate((rs_vals, mwr_vals))
        buffer = 1
        min_val = np.floor(np.nanmin(all_vals)- buffer)
        max_val = np.ceil(np.nanmax(all_vals) + buffer)

        # Plot
        plt.figure(figsize=(12, 12))
        plt.scatter(rs_vals, mwr_vals, alpha=0.7, marker="X")
        plt.plot([min_val, max_val], [min_val, max_val], 'k--',\
            label="1:1 line")
        plt.xlabel(f"{label_rs} Tb (K)", fontsize=fs)
        plt.ylabel(f"{label_mwr} Tb (K)", fontsize=fs)
        plt.title(f"{campaign_name}  ({tag})\nChannel {ch+1}: {frequencies[ch]:.2f} GHz",\
            fontsize=fs+3)

        # Textbox mit Kennzahlen
        plt.text(0.05, 0.95,
                 f"Bias: {bias:+.2f} K\nr = {corr:.2f}\nRMSE: {rmse:.2f} K",
                 transform=plt.gca().transAxes,
                 fontsize=fs, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.xlim(min_val, max_val)
        plt.ylim(min_val, max_val)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.grid(True)
        plt.tight_layout()

        filename = f"{output_dir}single_channel/{folder}scatter_{label_rs}_vs_{label_mwr}_ch{ch+1:02d}_{frequencies[ch]:.2f}GHz_{tag}.png"
        plt.savefig(filename)
        plt.close()

##############################################################################

def plot_tb_scatter_all_channels(rs_all, mwr_all, frequencies, output_dir,
        label_rs="RTTOV-gb", label_mwr="any_mwr",campaign_name="Vital I",\
        fs=fs, tag="", folder=""):
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 
          'brown', 'olive', 'pink', 'gray', 'teal', 'navy', 'gold']

    # Only K-Band
    plt.figure(figsize=(12, 12))
    combined = np.array([])
    for ch in range(7):
        rs_vals = rs_all[:, ch]
        mwr_vals = mwr_all[:, ch]
        mask, bias, corr, rmse = derive_statistics(rs_vals, mwr_vals)
        
        if np.any(mask):
            rs_valid = rs_vals[mask]
            mwr_valid = mwr_vals[mask]
            plt.scatter(mwr_valid, rs_valid,
                label=f"Ch {ch+1} ({frequencies[ch]:.2f} GHz)",
                alpha=0.7, marker="X",
                color=colors[ch])
            # Update Achsengrenzen
            combined = np.concatenate([combined, rs_valid, mwr_valid])


    # Referenzlinie (1:1)
    buffer = 1
    min_val = combined.min()
    max_val = combined.max()
    min_val = np.floor(min_val - buffer)
    max_val = np.ceil(max_val + buffer)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 line")

    plt.xlabel(f"{label_mwr} Tb (K)", fontsize=fs)
    plt.ylabel(f"{label_rs} Tb (K)", fontsize=fs)
    plt.title(f"{campaign_name}: Scatter of K-Band ({tag})\n({label_mwr} vs. {label_rs})",\
        fontsize=fs+3)
    plt.legend(ncol=2, fontsize=fs-8, loc="lower right")
    plt.grid(True)
    plt.xlim(min_val, max_val)
    # Textbox mit Kennzahlen
    plt.text(0.05, 0.95,
                 f"Bias: {bias:+.2f} K\nr = {corr:.2f}\nRMSE: {rmse:.2f} K",
                 transform=plt.gca().transAxes,
                 fontsize=fs, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    filename = f"{output_dir}{folder}scatter_{label_rs}_vs_{label_mwr}_K-Band_{frequencies[ch]:.2f}GHz_{tag}.png"
    plt.savefig(filename)
    plt.close()

    # Only V-Band
    plt.figure(figsize=(12, 12))
    combined = np.array([])
    for ch in range(7,14):
        rs_vals = rs_all[:, ch]
        mwr_vals = mwr_all[:, ch]
        mask, bias, corr, rmse = derive_statistics(rs_vals, mwr_vals)
        
        if np.any(mask):
            rs_valid = rs_vals[mask]
            mwr_valid = mwr_vals[mask]
            plt.scatter(mwr_valid, rs_valid, label=f"Ch {ch+1} ({frequencies[ch]:.2f} GHz)",\
                alpha=0.7, marker="X",
                color=colors[ch])

            # Update Achsengrenzen
            combined = np.concatenate([combined, rs_valid, mwr_valid])

    # Referenzlinie (1:1)
    buffer = 1
    min_val = combined.min()
    max_val = combined.max()
    min_val = np.floor(min_val - buffer)
    max_val = np.ceil(max_val + buffer)
    min_val = np.floor(min_val - buffer)
    max_val = np.ceil(max_val + buffer)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 line")

    plt.xlabel(f"{label_mwr} Tb (K)", fontsize=fs)
    plt.ylabel(f"{label_rs} Tb (K)", fontsize=fs)
    plt.title(f"{campaign_name}: Scatter of V-Band ({tag})\n({label_mwr} vs. {label_rs})",\
        fontsize=fs+3)
    plt.legend(ncol=2, fontsize=fs-8, loc="lower right")
    plt.grid(True)
    # Textbox mit Kennzahlen
    plt.text(0.05, 0.95,
                 f"Bias: {bias:+.2f} K\nr = {corr:.2f}\nRMSE: {rmse:.2f} K",
                 transform=plt.gca().transAxes,
                 fontsize=fs, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    filename = f"{output_dir}{folder}scatter_{label_rs}_vs_{label_mwr}_V-Band_{frequencies[ch]:.2f}GHz_{tag}.png"
    plt.savefig(filename)
    plt.close()

    # All channels
    plt.figure(figsize=(12, 12))
    combined = np.array([])
    for ch in range(14):
        rs_vals = rs_all[:, ch]
        mwr_vals = mwr_all[:, ch]
        mask, bias, corr, rmse = derive_statistics(rs_vals, mwr_vals)
        
        if np.any(mask):
            rs_valid = rs_vals[mask]
            mwr_valid = mwr_vals[mask]
            plt.scatter(mwr_valid, rs_valid, label=f"Ch {ch+1} ({frequencies[ch]:.2f} GHz)",\
                alpha=0.7, marker="X",
                color=colors[ch])

            # Update Achsengrenzen
            combined = np.concatenate([combined, rs_valid, mwr_valid])

    # Referenzlinie (1:1)
    buffer = 1
    min_val = combined.min()
    max_val = combined.max()
    min_val = np.floor(min_val - buffer)
    max_val = np.ceil(max_val + buffer)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 line")

    plt.xlabel(f"{label_mwr} Tb (K)", fontsize=fs)
    plt.ylabel(f"{label_rs} Tb (K)", fontsize=fs)
    plt.title(f"{campaign_name}: Scatter of all channels ({tag})\n({label_mwr} vs. {label_rs})",\
        fontsize=fs+3)
    plt.legend(ncol=2, fontsize=fs-8, loc="lower right")
    plt.grid(True)
    # Textbox mit Kennzahlen
    plt.text(0.05, 0.95,
                 f"Bias: {bias:+.2f} K\nr = {corr:.2f}\nRMSE: {rmse:.2f} K",
                 transform=plt.gca().transAxes,
                 fontsize=fs, verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    filename = f"{output_dir}{folder}scatter_{label_rs}_vs_{label_mwr}_all_chans_GHz_{tag}.png"
    plt.savefig(filename)
    plt.close()

##############################################################################

def all_plots_of_ds(reduced_ds, tag=""):

    # 1st Joyhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_cropped"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb cropped CSV", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_Joyhat/")
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_cropped"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb cropped CSV", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_Joyhat/")

    # 1st Joyhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb uncropped CSV", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag+"_uncropped")
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb uncropped CSV", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag+"_uncropped", folder="RTTOV_Joyhat/")

    # 2nd Hamhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb uncropped CSV", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb uncropped CSV", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_Hamhat/")

    # 3rd Hamhat against Joyhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_joyhat"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Joyhat'", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_joyhat"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Joyhat'", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag, folder="Hamhat_Joyhat/")

    # 5th Block all against LBL:
    # 5.1 Joyhat:
    plot_tb_scatter_per_channel(reduced_ds["TBs_joyhat"].values,\
        reduced_ds["TBs_R17_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Joyhat'", label_mwr="R17 cropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_joyhat"].values,\
        reduced_ds["TBs_R17_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Joyhat'", label_mwr="R17 cropped",\
        campaign_name="Vital I",tag=tag, folder="LBL_Joyhat/")

    # 5.2 Hamhat:
    plot_tb_scatter_per_channel(reduced_ds["TBs_hamhat"].values,\
        reduced_ds["TBs_R17_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Hamhat'", label_mwr="R17 cropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_hamhat"].values,\
        reduced_ds["TBs_R17_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Hamhat'", label_mwr="R17 cropped",\
        campaign_name="Vital I",tag=tag, folder="Hamhat_LBL/")

    # 5.3 RTTOV-gb
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_R17_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb uncropped CSV", label_mwr="R17 cropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_R17_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb uncropped CSV", label_mwr="R17 cropped",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_LBL/")
        
    # 6th Block all against LBL uncropped:
    # 5.1 Joyhat:
    plot_tb_scatter_per_channel(reduced_ds["TBs_joyhat"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Joyhat'", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_joyhat"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Joyhat'", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag, folder="LBL_Joyhat/")

    # 5.2 Hamhat:
    plot_tb_scatter_per_channel(reduced_ds["TBs_hamhat"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Hamhat'", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_hamhat"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RPG HATPRO 'Hamhat'", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag, folder="Hamhat_LBL/")

    # 5.3 RTTOV-gb
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb uncropped CSV", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb uncropped CSV", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_LBL/")
        
    # 7th block RTTOV-gb from NetCDF
    
    # 7.1 cropped
    # 7.1.1: Hamhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_nc_cropped"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (cropped_nc)", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_nc_cropped"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (cropped_nc)", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_Hamhat/")
    
    # 7.1.2: Joyhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_nc_cropped"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (cropped_nc)", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_Joyhat/")
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_nc_cropped"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (cropped_nc)", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_Joyhat/")
        
    # 7.1.3 R17:
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_nc_cropped"].values,\
        reduced_ds["TBs_R17_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (cropped_nc)", label_mwr="R17 cropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_nc_cropped"].values,\
        reduced_ds["TBs_R17_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (cropped_nc)", label_mwr="R17 cropped",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_LBL/")
    
    # 7.2 uncropped:
    # 7.2.1: Hamhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_Hamhat/")
    
    # 7.2.2: Joyhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag+"_uncropped", folder="RTTOV_Joyhat/")
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag+"_uncropped", folder="RTTOV_Joyhat/")
    
    # 7.2.3 R17:
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_LBL/")
        
    # 8. ARMS-gb uncropped:
    # 8.1: Hamhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_ARMS_gb"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (uncropped_nc)", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_ARMS_gb"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (uncropped_nc)", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I",tag=tag, folder="ARMS_Hamhat/") 
    # 8.2: Joyhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_ARMS_gb"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (uncropped_nc)", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag+"_uncropped", folder="ARMS_Joyhat/")
    plot_tb_scatter_all_channels(reduced_ds["TBs_ARMS_gb"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (uncropped_nc)", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I",tag=tag+"_uncropped", folder="ARMS_Joyhat/")
    # 8.3 R17:
    plot_tb_scatter_per_channel(reduced_ds["TBs_ARMS_gb"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (uncropped_nc)", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_ARMS_gb"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (uncropped_nc)", label_mwr="R17 uncropped",\
        campaign_name="Vital I",tag=tag, folder="ARMS_LBL/")
    # 8.4 RTTOV-gb:
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_ARMS_gb"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="ARMS-gb uncropped",\
        campaign_name="Vital I",tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_ARMS_gb"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="ARMS-gb uncropped",\
        campaign_name="Vital I",tag=tag, folder="RTTOV_ARMS/")

    # 9. ARMS-gb cropped:
    # 9.1: Hamhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_ARMS_gb_cropped"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (cropped_nc)", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I", tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_ARMS_gb_cropped"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (cropped_nc)", label_mwr="RPG HATPRO 'Hamhat'",\
        campaign_name="Vital I", tag=tag, folder="ARMS_Hamhat/") 
    # 9.2: Joyhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_ARMS_gb_cropped"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (cropped_nc)", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I", tag=tag+"_cropped", folder="ARMS_Joyhat/")
    plot_tb_scatter_all_channels(reduced_ds["TBs_ARMS_gb_cropped"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (cropped_nc)", label_mwr="RPG HATPRO 'Joyhat'",\
        campaign_name="Vital I", tag=tag+"_cropped", folder="ARMS_Joyhat/")
    # 9.3: R17
    plot_tb_scatter_per_channel(reduced_ds["TBs_ARMS_gb_cropped"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (cropped_nc)", label_mwr="R17 uncropped",\
        campaign_name="Vital I", tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_ARMS_gb_cropped"].values,\
        reduced_ds["TBs_R17"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="ARMS-gb (cropped_nc)", label_mwr="R17 uncropped",\
        campaign_name="Vital I", tag=tag, folder="ARMS_LBL/")
    # 9.4: RTTOV-gb
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_ARMS_gb_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="ARMS-gb cropped",\
        campaign_name="Vital I", tag=tag)
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb_nc"].values,\
        reduced_ds["TBs_ARMS_gb_cropped"].values,\
        ds["frequency"].values, output_dir, 
        label_rs="RTTOV-gb (uncropped_nc)", label_mwr="ARMS-gb cropped",\
        campaign_name="Vital I", tag=tag, folder="RTTOV_ARMS/")
        
    plt.close("all")
    return 0

##############################################################################

def divide2roof_and_yard_sets(ds):
    # , 
    yard_variables = ["TBs_RTTOV_gb", "TBs_R17",\
        "TBs_R03", "TBs_R16", "TBs_R19",\
        "TBs_R98", "TBs_R19SD",\
        "TBs_R20", "TBs_R20SD", "TBs_hamhat",\
        "TBs_RTTOV_gb_nc", "TBs_ARMS_gb"]
    # ,
    roof_variables = ["TBs_RTTOV_gb_cropped",\
        "TBs_R17_cropped", "TBs_R03_cropped", "TBs_R16_cropped",\
        "TBs_R19_cropped", "TBs_R98_cropped", "TBs_R19SD_cropped",\
        "TBs_R20_cropped", "TBs_R20SD_cropped", "TBs_joyhat",\
        "TBs_RTTOV_gb_nc_cropped", "TBs_ARMS_gb_cropped"]
    
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

def create_bias_plot_of_all_mods(ds, tag="any tag", out=""):
    ds_yard, ds_roof = divide2roof_and_yard_sets(ds)
    
    # 1st derive mean TBs of all models per channel
    mean_by_channel_roof, mean_by_channel_yard =\
        derive_mean_of_all_channels(ds_yard, ds_roof)
    
    # 2nd derive difference of mean for single model from combined mean
    colors=["blue", "orange", "green", "red","purple", "brown", "pink",\
         "gray", "olive", "cyan", "indigo", "darkgreen", "coral"]
    #2.1.1
    plt.figure()
    plt.title(f"K-Band channels Bias Vital I (roof / Joyhat / {tag})")
    plt.plot(np.arange(1,8), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    for i, var in enumerate(ds_roof.data_vars):
         if i==0 or i==1 or 9<=i<=11:
             plt.plot(np.arange(1,8), (ds_roof.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_roof)[:7],\
                 label=f"Bias {var}",  color=colors[i])
         elif i==10:
             break
         else:
            plt.scatter(np.arange(1,8), (ds_roof.mean(dim="time",\
                skipna=True)[var].values-mean_by_channel_roof)[:7],\
                label=f"Bias {var}", marker="X", color=colors[i])
         plt.text(1+0.5*i, 10, "All channels: "+str(np.nanmean(ds_roof.mean(dim="time",\
             skipna=True)[var].values-mean_by_channel_roof)))
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K_bias_roof.png")
    
    #2.1.2
    plt.figure()
    plt.title(f"V-Band Bias Vital I (roof / Joyhat / {tag})")
    plt.plot(np.arange(8,15), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    for i, var in enumerate(ds_roof.data_vars):
         if i==0 or i==1 or 9<=i<=11:
             plt.plot(np.arange(8,15), (ds_roof.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_roof)[7:],\
                 label=f"Bias {var}",  color=colors[i])
         elif i==10:
             break
         else:
             plt.scatter(np.arange(8,15), (ds_roof.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_roof)[7:],\
                 label=f"Bias {var}", marker="X", color=colors[i])
         plt.text(1+0.5*i, 10, "All channels: "+str(np.nanmean(ds_roof.mean(dim="time",\
             skipna=True)[var].values-mean_by_channel_roof)))
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V_bias_roof.png")
    plt.close("all")
    
    # 2.2.1
    plt.figure()
    plt.title(f"K-Band channels Bias Vital I (yard / Hamhat / {tag})")
    plt.plot(np.arange(1,8), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    for i, var in enumerate(ds_yard.data_vars):#
         if i==0 or i==1 or 9<=i<=11:
             plt.plot(np.arange(1,8), (ds_yard.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_yard)[:7],\
                 label=f"Bias {var}",  color=colors[i])
         elif i==10:
             break
         else:
             plt.scatter(np.arange(1,8), (ds_yard.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_yard)[:7],\
                 label=f"Bias {var}", marker="X", color=colors[i])
         plt.text(1+0.5*i, 10, "All channels: "+str(np.nanmean(ds_yard.mean(dim="time",\
             skipna=True)[var].values-mean_by_channel_yard)))
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K_bias_yard.png")

    # 2.2.2
    plt.figure()
    plt.title(f"V-Band Bias Vital I (yard / Hamhat / {tag})")
    plt.plot(np.arange(8,15), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    for i, var in enumerate(ds_yard.data_vars):
         if i==0 or i==1 or 9<=i<=11:
             plt.plot(np.arange(8,15), (ds_yard.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_yard)[7:],\
                 label=f"Bias {var}",  color=colors[i])
         elif i==10:
             break
         else:
             plt.scatter(np.arange(8,15), (ds_yard.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_yard)[7:],\
                 label=f"Bias {var}", marker="X", color=colors[i])
         plt.text(1+0.5*i, 10, "All channels: "+str(np.nanmean(ds_yard.mean(dim="time",\
             skipna=True)[var].values-mean_by_channel_yard)))
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V_bias_yard.png")
    plt.close("all")
    
    # 3.1 combined bias plot roof+yard K:
    diff_roof_yard=mean_by_channel_roof-mean_by_channel_yard
    Normalized_joyhat = (ds_roof.mean(dim="time",\
         skipna=True)["TBs_joyhat"].values-mean_by_channel_roof)-diff_roof_yard
    plt.figure()
    plt.title(f"K-Band channels Bias Vital I (Combined by offset / {tag})")
    plt.plot(np.arange(1,8), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    for i, var in enumerate(ds_yard.data_vars):#
         if i==0 or i==1 or 9<=i<=11:
             plt.plot(np.arange(1,8), (ds_yard.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_yard)[:7],\
                 label=f"Bias {var}",  color=colors[i])
         elif i==10:
             break
         else:
             plt.scatter(np.arange(1,8), (ds_yard.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_yard)[:7],\
                 label=f"Bias {var}", marker="X", color=colors[i])
         plt.text(1+0.5*i, 10, "All channels: "+str(np.nanmean(ds_yard.mean(dim="time",\
             skipna=True)[var].values-mean_by_channel_yard)))             
    # Add Joyhat here:  
    plt.plot(np.arange(1,8), Normalized_joyhat[:7],\
                 label=f"TBs_Joyhat_MWR",  color=colors[-1])          
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K_bias_combined.png")

    ##########################
    # Add anything on ARMS-gb here? Following lines?

    # 3.2 combined bias plot roof+yard V:
    plt.figure()
    plt.title(f"V-Band Bias Vital I (Combined by offset / {tag})")
    plt.plot(np.arange(8,15), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    for i, var in enumerate(ds_yard.data_vars):
         if i==0 or i==1 or 9<=i<=11:
             plt.plot(np.arange(8,15), (ds_yard.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_yard)[7:],\
                 label=f"Bias {var}",  color=colors[i])
         elif i==10:
             break
         else:
             plt.scatter(np.arange(8,15), (ds_yard.mean(dim="time",\
                 skipna=True)[var].values-mean_by_channel_yard)[7:],\
                 label=f"Bias {var}", marker="X", color=colors[i])
         plt.text(1+0.5*i, 10, "All channels: "+str(np.nanmean(ds_yard.mean(dim="time",\
             skipna=True)[var].values-mean_by_channel_yard)))
    # Add Joyhat here:  
    plt.plot(np.arange(8,15), Normalized_joyhat[7:],\
                 label=f"TBs_Joyhat_MWR",  color=colors[-1])
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V_bias_combined.png")
    
    # 3.3 Plot yard / roof difference:
    plt.figure()
    plt.title(f"K-Band channels: roof - yard difference by channel ({tag})")
    plt.plot(np.arange(1,8), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    plt.plot(np.arange(1,8), diff_roof_yard[:7],\
                 label=f"Yard roof difference",  color=colors[0])      
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K_bias_yard_roof_diff.png")
    
    # 3.4 Plot yard / roof difference:
    plt.figure()
    plt.title(f"V-Band channels: roof - yard difference by channel ({tag})")
    plt.plot(np.arange(8,15), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    plt.plot(np.arange(8,15), diff_roof_yard[7:],\
                 label=f"Yard roof difference",  color=colors[0])      
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V_bias_yard_roof_diff.png")
    
    # 4.1 RTTOV-gb_variants K:
    plt.figure()
    plt.title(f"K-Band Bias between different RTTOV-gb ({tag})")
    plt.plot(np.arange(1,8), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(1,8), [0]*7, color="black")
    plt.plot(np.arange(1,8), (ds_yard.mean(dim="time",\
        skipna=True)["TBs_RTTOV_gb_nc"].values-\
        mean_by_channel_yard)[:7],\
        label=f"RTTOV_gb_nc",  color=colors[0], linestyle="dashed")
    plt.scatter(np.arange(1,8), (ds_yard.mean(dim="time",\
        skipna=True)["TBs_RTTOV_gb"].values-\
        mean_by_channel_yard)[:7],\
        label=f"RTTOV_gb_csv",  color=colors[1], marker="X")
        
    # Hier wird nichts normalisiert!!!
    Normalized_nc_crop = ds_roof.mean(dim="time",\
        skipna=True)["TBs_RTTOV_gb_nc_cropped"].values #-diff_roof_yard
    Normalized_csv_crop = ds_roof.mean(dim="time",\
        skipna=True)["TBs_RTTOV_gb_cropped"].values #-diff_roof_yard
    plt.plot(np.arange(1,8), (Normalized_nc_crop-mean_by_channel_yard)[:7],\
       label=f"RTTOV-gb_crop_csv",  color=colors[2], linestyle="dashed")
    plt.scatter(np.arange(1,8), (Normalized_nc_crop-mean_by_channel_yard)[:7],\
       label=f"RTTOV-gb_crop_nc",  color=colors[3], marker="X") 
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"K_bias_combined_RTTOVs.png")
        
    # 4.2 RTTOV-gb_variants V:    
    plt.figure()
    plt.title(f"V-Band Bias between different RTTOV-gb ({tag})")
    plt.plot(np.arange(8,15), [-1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [1]*7, color="black", linestyle="dashed")
    plt.plot(np.arange(8,15), [0]*7, color="black")
    plt.plot(np.arange(8,15), (ds_yard.mean(dim="time",\
        skipna=True)["TBs_RTTOV_gb_nc"].values-\
        mean_by_channel_yard)[7:],\
        label=f"RTTOV_gb_nc",  color=colors[0], linestyle="dashed")
    plt.scatter(np.arange(8,15), (ds_yard.mean(dim="time",\
        skipna=True)["TBs_RTTOV_gb"].values-\
        mean_by_channel_yard)[7:],\
        label=f"RTTOV_gb_csv",  color=colors[1], marker="X")
    # Hier wird nichts normalisiert!!!
    Normalized_nc_crop = ds_roof.mean(dim="time",\
        skipna=True)["TBs_RTTOV_gb_nc_cropped"].values #-diff_roof_yard
    Normalized_csv_crop = ds_roof.mean(dim="time",\
        skipna=True)["TBs_RTTOV_gb_cropped"].values #-diff_roof_yard
    plt.plot(np.arange(8,15), (Normalized_nc_crop-mean_by_channel_yard)[7:],\
       label=f"RTTOV-gb_crop_csv",  color=colors[2], linestyle="dashed")
    plt.scatter(np.arange(8,15), (Normalized_nc_crop-mean_by_channel_yard)[7:],\
       label=f"RTTOV-gb_crop_nc",  color=colors[3], marker="X") 
    plt.ylim(-3,3)
    plt.legend(loc='lower right', fontsize=9)
    plt.savefig(out+tag+"V_bias_combined_RTTOVs.png")
    plt.close("all")
    
    return 0

##############################################################################

def create_data_avail_plot(ds, tag="any tag", out=""):
    # make data availability plot for all dates:
    obs = ["TBs_RTTOV_gb", "TBs_hamhat", "TBs_joyhat", "TBs_R17", "TBs_ARMS_gb"]
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
    # all_plots_of_ds(ds_zen_clear, tag=" clear_sky ")
    create_bias_plot_of_all_mods(ds_zen_clear, tag=" clear_sky ",\
        out=args.output2)

    print("Processing all sky zenith...")
    create_data_avail_plot(ds_zen_all, tag="all_sky",\
        out=os.path.expanduser("~/PhD_plots/availability/"))
    # all_plots_of_ds(ds_zen_all, tag=" all_sky ")
    create_bias_plot_of_all_mods(ds_zen_all, tag=" all_sky ",\
        out=args.output2)



















