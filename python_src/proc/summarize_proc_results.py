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
# 2nd Used Functions
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
    )
    # Define default output path and file:
    outpath = "~/PhD_data/TB_preproc_and_proc_results/"
    outfile_py = "3_campaigns_PyRTlib_R24_processed_TBs_from_rs.nc"    
    outfile_a = "3_campaigns_ARMS_gb_processed_TBs_from_rs.nc"    
    outfile_r = "3_campaigns_RTTOV_gb_processed_TBs_from_rs.nc"       
    outfile = "3campaigns_3models_all_results.nc"
    
    parser.add_argument(
        "--pyrtlib", "-p",
        type=str,
        default=os.path.expanduser(outpath+outfile_py),
        help="Where to find PyRTlib TBs"
    )  
    parser.add_argument(
        "--armsgb", "-a",
        type=str,
        default=os.path.expanduser(outpath+outfile_a),
        help="Where to find ARMS-gb TBs"
    )   
    parser.add_argument(
        "--rttovgb", "-r",
        type=str,
        default=os.path.expanduser(outpath+outfile_r),
        help="Where to find RTTOV-gb TBs"
    )      
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser(outpath+outfile),
        help="Where to save summarized NetCDF of Inputs and Output TBs"
    )                     
    return parser.parse_args()

##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    
    ds_py = xr.open_dataset(args.pyrtlib)
    ds_ar = xr.open_dataset(args.armsgb)
    ds_rt = xr.open_dataset(args.rttovgb)
    ds_new = ds_rt
    
    ###############
    # Maybe I will have to add other variables here:
    ds_new["TBs_ARMS_gb"] = ds_ar["TBs_ARMS_gb"]
    ds_new["TBs_PyRTlib_R24"] = ds_py["TBs_PyRTlib_R24"]
    ds_new["TBs_PyRTlib_R17"] = ds_py["TBs_PyRTlib_R17"]
    ds_new["TBs_PyRTlib_R98"] = ds_py["TBs_PyRTlib_R98"]
    ds_new["TBs_PyRTlib_R20"] = ds_py["TBs_PyRTlib_R20"]

    # 2nd Print dataset to NetCDF
    ds_new.to_netcdf(args.output, format="NETCDF4_CLASSIC")

   
        

        

