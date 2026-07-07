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
import sys
sys.path.append("./")
from m_mod_plot import select_ds_camp_loc, ensure_folder_exists, apply_sky_mask


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



##############################################################################


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
    # sky = "clear"  
    skies = ["clear", "cloudy", "all_sky"]

    for campaign in ds0["Campaign"].values:
        for location in ds0["Location"].values:  
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


                print(ds_cf)
                break


    ###################
    # Cloud flag does not take into account elevation cloud_flag!!!
    # Use Crop-Index 1 for Joyhat!!! somehow...!?!





