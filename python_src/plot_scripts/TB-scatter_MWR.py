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

# Plotstyles:
fs = 25
plt.rc('font', size=fs) 
plt.style.use('seaborn-poster')


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
        default=os.path.expanduser("~/PhD_plots/"),
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
                    label_rs="RTTOV-gb", label_mwr="Joyhat", campaign_name="Vital I", fs=fs):
    """
    Erzeugt Scatterplots für jedes Frequenzkanalpaar zweier TB-Datensätze
    inklusive Bias, RMSE und Pearson-Korrelation.

    Parameter:
    - rs_all:     ndarray mit shape (channels, samples), Referenz (z.B. RTTOV)
    - mwr_all:    ndarray mit shape (channels, samples), Vergleich (z.B. MWR)
    - frequencies:Liste oder Array der Frequenzen pro Kanal (Länge = Kanäle)
    - output_dir: Zielordner für die Bilddateien
    - label_rs:   Label für Referenzdatensatz (x-Achse)
    - label_mwr:  Label für Vergleichsdatensatz (y-Achse)
    - campaign_name: Name der Kampagne (kommt in den Plot-Titel)

    Speichert PNG-Dateien im Output-Ordner.
    """
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
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 line")
        plt.xlabel(f"{label_rs} Tb (K)", fontsize=fs)
        plt.ylabel(f"{label_mwr} Tb (K)", fontsize=fs)
        plt.title(f"{campaign_name}\nChannel {ch+1}: {frequencies[ch]:.2f} GHz", fontsize=fs+3)

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

        filename = f"{output_dir}scatter_{label_rs}_vs_{label_mwr}_ch{ch+1:02d}_{frequencies[ch]:.2f}GHz.png"
        plt.savefig(filename)
        plt.close()

##############################################################################

def plot_tb_scatter_all_channels(rs_all, mwr_all, frequencies, output_dir,
                                 label_rs="RTTOV-gb", label_mwr="any_mwr", campaign_name="Vital I", fs=fs):
    """
    Erzeugt einen kombinierten Scatterplot aller TB-Kanäle
    für zwei Datensätze (z.B. Radiosonde vs. MWR).

    Parameter:
    - rs_all:       ndarray (channels, samples), Referenzdaten
    - mwr_all:      ndarray (channels, samples), Vergleichsdaten
    - frequencies:  Liste oder Array der Frequenzen pro Kanal
    - output_path:  Pfad zur Bilddatei (z.B. 'output/all_channels_scatter.png')
    - label_rs:     Label für Referenzdatensatz (x-Achse)
    - label_mwr:    Label für Vergleichsdatensatz (y-Achse)
    - campaign_name: Titelergänzung für die Kampagne
    """
    plt.figure(figsize=(12, 12))
    min_val, max_val = np.inf, -np.inf

    # Only K-Band
    for ch in range(7):
        rs_vals = rs_all[:, ch]
        mwr_vals = mwr_all[:, ch]
        mask, bias, corr, rmse = derive_statistics(rs_vals, mwr_vals)
        
        if np.any(mask):
            rs_valid = rs_vals[mask]
            mwr_valid = mwr_vals[mask]
            plt.scatter(mwr_valid, rs_valid, label=f"Ch {ch+1} ({frequencies[ch]:.2f} GHz)",\
                alpha=0.7, marker="X")

            # Update Achsengrenzen
            combined = np.concatenate([rs_valid, mwr_valid])
            min_val = combined.min()
            max_val = combined.max()

    # Referenzlinie (1:1)
    buffer = 1
    min_val = np.floor(min_val - buffer)
    max_val = np.ceil(max_val + buffer)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 line")

    plt.xlabel(f"{label_mwr} Tb (K)", fontsize=fs)
    plt.ylabel(f"{label_rs} Tb (K)", fontsize=fs)
    plt.title(f"{campaign_name}: Scatter of K-Band\n({label_mwr} vs. {label_rs})", fontsize=fs+3)
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
    filename = f"{output_dir}scatter_{label_rs}_vs_{label_mwr}_K-Band_{frequencies[ch]:.2f}GHz.png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(12, 12))
    min_val, max_val = np.inf, -np.inf
    # Only V-Band
    for ch in range(7,14):
        rs_vals = rs_all[:, ch]
        mwr_vals = mwr_all[:, ch]
        mask, bias, corr, rmse = derive_statistics(rs_vals, mwr_vals)
        
        if np.any(mask):
            rs_valid = rs_vals[mask]
            mwr_valid = mwr_vals[mask]
            plt.scatter(mwr_valid, rs_valid, label=f"Ch {ch+1} ({frequencies[ch]:.2f} GHz)",\
                alpha=0.7, marker="X")

            # Update Achsengrenzen
            combined = np.concatenate([rs_valid, mwr_valid])
            min_val = combined.min()
            max_val = combined.max()

    # Referenzlinie (1:1)
    buffer = 1
    min_val = np.floor(min_val - buffer)
    max_val = np.ceil(max_val + buffer)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 line")

    plt.xlabel(f"{label_mwr} Tb (K)", fontsize=fs)
    plt.ylabel(f"{label_rs} Tb (K)", fontsize=fs)
    plt.title(f"{campaign_name}: Scatter of V-Band\n({label_mwr} vs. {label_rs})", fontsize=fs+3)
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
    filename = f"{output_dir}scatter_{label_rs}_vs_{label_mwr}_V-Band_{frequencies[ch]:.2f}GHz.png"
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(12, 12))
    min_val, max_val = np.inf, -np.inf
    # Only V-Band
    for ch in range(14):
        rs_vals = rs_all[:, ch]
        mwr_vals = mwr_all[:, ch]
        mask, bias, corr, rmse = derive_statistics(rs_vals, mwr_vals)
        
        if np.any(mask):
            rs_valid = rs_vals[mask]
            mwr_valid = mwr_vals[mask]
            plt.scatter(mwr_valid, rs_valid, label=f"Ch {ch+1} ({frequencies[ch]:.2f} GHz)",\
                alpha=0.7, marker="X")

            # Update Achsengrenzen
            combined = np.concatenate([rs_valid, mwr_valid])
            min_val = combined.min()
            max_val = combined.max()

    # Referenzlinie (1:1)
    buffer = 1
    min_val = np.floor(min_val - buffer)
    max_val = np.ceil(max_val + buffer)
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="1:1 line")

    plt.xlabel(f"{label_mwr} Tb (K)", fontsize=fs)
    plt.ylabel(f"{label_rs} Tb (K)", fontsize=fs)
    plt.title(f"{campaign_name}: Scatter of all channels\n({label_mwr} vs. {label_rs})", fontsize=fs+3)
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
    filename = f"{output_dir}scatter_{label_rs}_vs_{label_mwr}_all_chans_GHz.png"
    plt.savefig(filename)
    plt.close()

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    nc_out_path=args.NetCDF
    ds = xr.open_dataset(nc_out_path)
    channels = np.arange(1,15)
    output_dir = args.output
    rs_tag ="Radiosondes via RTTOV-gb"
  

    # Filter for different issues:
    reduced_ds = ds.where(ds["cloud_flag"]==0).where(ds["elevation"]>89.5)\
    .where(ds["mean_rainfall"]<0.000001).where(ds["TBs_joyhat"]>0.000001)\
    .where(ds["elevation2"]>89.5)    

    # 1st Joyhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs=rs_tag, label_mwr="RPG HATPRO 'Joyhat'", campaign_name="Vital I")
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_joyhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs=rs_tag, label_mwr="RPG HATPRO 'Joyhat'", campaign_name="Vital I")

    # 2nd Hamhat
    plot_tb_scatter_per_channel(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs=rs_tag, label_mwr="RPG HATPRO 'Hamhat'", campaign_name="Vital I")
    plot_tb_scatter_all_channels(reduced_ds["TBs_RTTOV_gb"].values,\
        reduced_ds["TBs_hamhat"].values,\
        ds["frequency"].values, output_dir, 
        label_rs=rs_tag, label_mwr="RPG HATPRO 'Hamhat'", campaign_name="Vital I")
























