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
    outfile = "3_campaigns_PyRTlib_R24_processed_TBs_from rs.nc"
    
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.expanduser(\
            outpath+"MWR_rs_FESSTVaLSoclesVital1_all_elevations.nc"),
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
    tbs = np.full((len(ds["time"].values), 14,10,2), np.nan)     
     
    for i, timestep in enumerate(ds["time"].values):
        for j, crop in enumerate(ds["Crop"].values):
            for k, elevation in enumerate(ds["elevation"].values):
                # print("Indices: ",i,j,k)
            
                #########
                # Put elevation in here:
                ang = np.array([elevation])
                
                # Which input variables do I need?
                rh_in = ds["Level_RH"].isel(time=i).isel(Crop=j).values/100
                # as fraction not %
                z_in =  ds["Level_z"].isel(time=i).isel(Crop=j).values/1000
                # m to km!
                p_in = ds["Level_Pressure"].isel(time=i).isel(Crop=j).values # hPa
                t_in =ds["Level_Temperature"].isel(time=i).isel(Crop=j).values  # K
                
                # Exclude profiles with any Nans again:
                nan_bool = check_for_nans(z_in, p_in, t_in, rh_in, frqs, ang)
                
                if not nan_bool:
                    # Run RTE model:
                    mdl = mdls[-1] # Rosenkranz 24
                    rte = TbCloudRTE(z_in[::-1], p_in[::-1], t_in[::-1], rh_in[::-1], frqs, ang)
                    rte.init_absmdl(mdl)
                    ####################
                    # Clear sky!!!
                    # rte.cloudy = False 
                    # False is default I guess...
                    print("RTE_cloudy: ", rte.cloudy)
                    ####################
                    rte.satellite = False # downwelling!!!
                    df_from_ground = rte.execute()                 
                    tbs[i,:,k,j] = df_from_ground["tbtotal"].values
                else:
                    print("NaNs found!!!!!!!")

    ds["TBs_PyRTlib_R24"] = (('time', 'N_Channels','elevation','Crop'), tbs)
    attributes = {
        'long_name': 'Brightness temperature modelled by R24',
        'units': 'K',
        'standard_name': 'brightness_temperature',
        'comments': 'Brightness temperatures modeled from radiosonde data for 14 channels of HATPRO radiometer',
    }
    ds["TBs_PyRTlib_R24"].attrs = attributes    
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
        

        

