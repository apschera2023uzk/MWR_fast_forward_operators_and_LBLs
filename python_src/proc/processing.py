#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Describe this file...

##############################################################

import math
import argparse
import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import glob
import sys
import shutil
from datetime import datetime
import subprocess

##############################################################################
# 1.5: Parameter
##############################################################################

n_levels = 180

##############################################################################
# 2nd Used Functions
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.expanduser("/home/aki/PhD_data/armsgb_all_campaigns_zenith.nc"),
        help="NetCDF file with rs and MWR data"
    )
    parser.add_argument(
        "--rttov", "-r",
        type=str,
        default=os.path.expanduser("/home/aki/RTTOV-gb/rttov_test/test_example_k.1"),
        help="RTTOV-gb driectory to execute code!"
    )    
    parser.add_argument(
        "--script", "-s",
        type=str,
        default=os.path.expanduser("/home/aki/RTTOV-gb/rttov_test/run_apschera.sh"),
        help="Shell Script to run RTTOV-gb on profiles!"
    )        
    return parser.parse_args()

##############################################################################

def write1profile2str(t_array, ppmv_array,length_value,\
        p_array, liquid_array, height_in_km=0., deg_lat=50.,\
        zenith_angle=0.):
    string = ""
    for value in p_array:
        string+=f"{value:8.4f}\n"
    for value in t_array:
        string+=f"{value:6.3f}\n"
    for value in ppmv_array:
        string+=f"{value:9.4f}\n"
    for value in liquid_array:
        string+=f"{0.:12.6E}\n"
    string+=f"{t_array[-1]:10.4f}{p_array[-1]:10.2f}\n"
    string+=f"{height_in_km:6.1f}{deg_lat:6.1f}\n"
    string+=f"{zenith_angle:6.1f}\n"
        
    return string

##############################################################################

def check4NANs_in_time_crop_ele(ds, i,j,k):
    # Extract the relevant DataArrays for the given indices
    temp_arr = ds["Level_Temperature"].values[:, i, j]
    ppmvs_arr = ds["Level_ppmvs"].values[:, i, j]
    press_arr = ds["Level_Pressure"].values[:, i, j]
    liquid_arr = ds["Level_Liquid"].values[:, i, j]
    
    # Check each array for NaN values
    if np.any(np.isnan(temp_arr)) or \
            np.any(np.isnan(ppmvs_arr)) or \
            np.any(np.isnan(press_arr)) or \
            np.any(np.isnan(liquid_arr)):
        return True
    else:
        return False
        

##############################################################################

def create_RTTOV_gb_in_profiles(ds, args):
    profiles = ""
    valid_indices = []
    
    for i, timestep in enumerate(ds["time"].values):
         for j, crop in enumerate(ds["Crop"].values):
             for k, elevation in enumerate(ds["elevation"].values):
                 if check4NANs_in_time_crop_ele(ds, i,j,k):
                     pass
                 else:
                     profile1 = write1profile2str(\
                         ds["Level_Temperature"].values[:,i,j],\
                         ds["Level_ppmvs"].values[:,i,j],\
                         len(ds["Level_ppmvs"].values[:,i,j]),\
                         ds["Level_Pressure"].values[:,i,j],\
                         ds["Level_Liquid"].values[:,i,j],\
                         height_in_km=ds["Surface_Altitude"].values[i,j],\
                         deg_lat=ds["Latitude"].values[i],\
                         zenith_angle=90.-elevation)
                     profiles+=profile1
                     valid_indices.append((i,j,k))
         
         #################
         #if i==3:
         #    break
         ###########
                              
    # After loops - save results:
    dir_name = os.path.dirname(args.input)
    outfile = str(dir_name)+"/"+"prof_plev.dat"
    out = open(outfile, "w")
    out.write(profiles)
    out.close()
         
    return outfile, valid_indices

##############################################################################

def run_rttov_gb(outfile, valid_indices, args,nlevels = n_levels):

    # 1st Copy prof_plev.dat to inputs
    prof_file = outfile.split("/")[-1]
    where2 = args.rttov+"/"+prof_file
    shutil.copy(outfile, where2)
    
    # 2nd edit run script for level number:
    script_file = args.script
    nlevels = n_levels
    nprofs = len(valid_indices)
    with open(script_file, "r") as f:
        lines = f.readlines()
        # Zeile 30 (Index 29) ersetzen
    lines[28] = f"NPROF={nprofs}\n"    
    lines[29] = f"NLEVELS={nlevels}\n"
    with open(script_file, "w") as f:
        f.writelines(lines)    
        
    # 3rd run RTTOV-gb
    rttov_dir = os.path.dirname(args.script)
    subprocess.run(["bash", args.script, "ARCH=gfortran"], cwd=rttov_dir)    
    
    return 0
    
##############################################################################


def derive_TBs4RTTOV_gb(ds, args):

    # 1st Create prof_plev file from ds:
    outfile, valid_indices = create_RTTOV_gb_in_profiles(ds, args)
    
    # 2nd Run RTTOV-gb on them:
    run_rttov_gb(outfile, valid_indices, args)
        
    # 3rd Extract RTTOV-gb TBs:
    
    
    return 0
    
##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    ds = xr.open_dataset(args.input)
    
    ##################
    # print(ds.data_vars)
    # What the program should do:
    
    # 1 Derive TBs for all elevations  for RTTOV-gb
    derive_TBs4RTTOV_gb(ds, args)
    
    # 2 Derive TBs for all elevations  for pyrtlib

    # 2.5 Derive TBs for clear sky for ARMS-gb

    # 3 Append dataset

        

        

