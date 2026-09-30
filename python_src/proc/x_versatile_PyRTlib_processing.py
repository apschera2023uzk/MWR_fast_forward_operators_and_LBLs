#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Wrapper für PyRTlib processing of NetCDF file with radiosonde profiles. 
# Radiosonde profiles are read for every rs within the dataset and
# TBs are calculated according to these profiles and HATPRO channels.
# All of this is saved in a NetCDF file...

##############################################################

import math
import argparse
import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import glob
import shutil
from datetime import datetime
import subprocess
import sys
sys.path.append('/home/aki/pyrtlib')
from pyrtlib.climatology import AtmosphericProfiles as atmp
from pyrtlib.tb_spectrum import TbCloudRTE
from pyrtlib.utils import ppmv2gkg, mr2rh

##############################################################################
# 1.5: Parameter
##############################################################################

n_levels=180
batch_size=20
elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])

##############################################################################
# 2nd Used Functions
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="This script processes radiosondes into R24 TBs via PyRTlib"
    )
    
    # Define default output path and file:
    outpath = "~/PhD_data/TB_preproc_and_proc_results/"
    outfile = "4campaigns_PyRTlib_R24_processed_TBs_from_rs.nc"
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.expanduser(\
            outpath+"complete_Socles_FESSTVaL_vitI_vitII.nc"),
        help="NetCDF file with rs and MWR data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser(outpath+outfile),
        help="Where to save summarized NetCDF of Inputs and Output TBs"
    )     
    return parser.parse_args()
    
##############################################################################
# PyRTlib:
##############################################################################

def check_for_nans(z_in, p_in, t_in, rh_in, frqs, ang):
    return np.any([
        np.isnan(z_in).any(),
        np.isnan(p_in).any(),
        np.isnan(t_in).any(),
        np.isnan(rh_in).any(),
        np.isnan(frqs).any(),
        np.isnan(ang).any()
    ])
      
############################################################################## 

def tb_to_freq_ang(df, n_frq, n_ang):
    tb_flat = df["tbtotal"].values
    assert tb_flat.size == n_frq * n_ang, f"unexpected size {tb_flat.size}"
    return tb_flat.reshape(n_ang, n_frq).T   # adjust after the test below 

############################################################################## 

def derive_TBs4PyRTlib(ds, args):
    # Dieser Code ist sehr langsam...
    # Es liegt schon ein Unterschied zwischen 48 Sonden oder 521 Sonden à 10 Winkel...

    frqs = np.array([22.24,23.04,23.84,25.44,26.24,27.84,31.4,51.26,52.28,\
        53.86,54.94,56.66,57.3,58.])
    nf = len(frqs)    
    mdls = ["R17", "R03", "R16", "R19", "R98", "R19SD", "R20", "R20SD","R24"]
    tags = ["Rosenkranz 17", "Tretjakov 2003", "Rosenkranz 17 (2)",\
        "Rosenkranz + Cimini", "Rosenkranz + Cimini SD", "Makarov",\
         "Makarov SD", "Rosenkranz 24"]
    tbs = np.full((len(ds["time"].values), 14,len(ds["elevation"].values)), np.nan)
    tbs17 = np.full((len(ds["time"].values), 14,len(ds["elevation"].values)), np.nan)  
    tbs98 = np.full((len(ds["time"].values), 14,len(ds["elevation"].values)), np.nan)   
    tbs20 = np.full((len(ds["time"].values), 14,len(ds["elevation"].values)), np.nan)    

    #####################
    # Improved efficiency by only looping time and combining elevations:
    ds = ds.load()                     # avoid repeated lazy disk reads in the loop
    angs = ds["elevation"].values      # all elevation angles at once
    n_ang = len(angs)

    # (model, output array) pairs; list instead of dict to avoid key collisions
    model_outputs = [("R20", tbs20), (mdls[-1], tbs), (mdls[0], tbs17), (mdls[4], tbs98)]

    for i, timestep in enumerate(ds["time"].values):
        rh_in = ds["Level_RH"].isel(time=i).values / 100          # fraction
        z_in  = ds["Level_z"].isel(time=i).values / 1000          # km
        p_in  = ds["Level_Pressure"].isel(time=i).values          # hPa
        t_in  = ds["Level_Temperature"].isel(time=i).values       # K

        if check_for_nans(z_in, p_in, t_in, rh_in, frqs, angs):
            print(f"NaNs found in profile {i} — skipping")
            continue

        for mdl, out in model_outputs:
            rte = TbCloudRTE(z_in[::-1], p_in[::-1], t_in[::-1], rh_in[::-1], frqs, angs)
            rte.init_absmdl(mdl)
            rte.satellite = False   # downwelling
            df = rte.execute()
            out[i, :, :] = tb_to_freq_ang(df, len(frqs), n_ang)

    ds["TBs_PyRTlib_R24"] = (('time', 'N_Channels','elevation'), tbs)
    attributes = {
        'long_name': 'Brightness temperature modelled by R24',
        'units': 'K',
        'standard_name': 'brightness_temperature',
        'comments': 'Brightness temperatures modeled from radiosonde data for 14 channels of HATPRO radiometer',
    }
    ds["TBs_PyRTlib_R24"].attrs = attributes  
    
    ds["TBs_PyRTlib_R17"] = (('time', 'N_Channels','elevation'), tbs17)
    attributes = {
        'long_name': 'Brightness temperature modelled by R17',
        'units': 'K',
        'standard_name': 'brightness_temperature',
        'comments': 'Brightness temperatures modeled from radiosonde data for 14 channels of HATPRO radiometer',
    }
    ds["TBs_PyRTlib_R17"].attrs = attributes  
    
    ds["TBs_PyRTlib_R98"] = (('time', 'N_Channels','elevation'), tbs98)
    attributes = {
        'long_name': 'Brightness temperature modelled by R98',
        'units': 'K',
        'standard_name': 'brightness_temperature',
        'comments': 'Brightness temperatures modeled from radiosonde data for 14 channels of HATPRO radiometer',
    }
    ds["TBs_PyRTlib_R98"].attrs = attributes 
    
    ds["TBs_PyRTlib_R20"] = (('time', 'N_Channels','elevation'), tbs20)
    attributes = {
        'long_name': 'Brightness temperature modelled by R20',
        'units': 'K',
        'standard_name': 'brightness_temperature',
        'comments': 'Brightness temperatures modeled from radiosonde data for 14 channels of HATPRO radiometer',
    }
    ds["TBs_PyRTlib_R20"].attrs = attributes          
               
    return ds

##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    ds = xr.open_dataset(args.input)

    # 1st Derive TBs for all elevations  for pyrtlib
    ds = derive_TBs4PyRTlib(ds, args)

    # 2nd Print dataset to NetCDF
    ds.to_netcdf(args.output, format="NETCDF4_CLASSIC")
        

        

