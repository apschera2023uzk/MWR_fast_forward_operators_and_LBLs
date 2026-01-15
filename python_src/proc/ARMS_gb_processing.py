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
from pathlib import Path
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
        description="This script processes radiosondes into TBs via ARMS-gb"
    )
    
    # Define default output path and file:
    outpath = "~/PhD_data/TB_preproc_and_proc_results/"
    outfile = "3_campaigns_ARMS_gb_processed_TBs_from rs.nc"
    
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
# ARMS-gb:
##############################################################################

def check4NANs_in_time_crop_ele(ds, t_index, crop_index):
    # Extract the relevant DataArrays for the given indices
    temp_arr = ds["Level_Temperature"].values[:, t_index, crop_index]
    ppmvs_arr = ds["Level_ppmvs"].values[:, t_index, crop_index]
    press_arr = ds["Level_Pressure"].values[:, t_index, crop_index]
    liquid_arr = ds["Level_Liquid"].values[:, t_index, crop_index]
    
    # Check each array for NaN values
    if np.any(np.isnan(temp_arr)) or \
            np.any(np.isnan(ppmvs_arr)) or \
            np.any(np.isnan(press_arr)) or \
            np.any(np.isnan(liquid_arr)):
        return True
    else:
        return False
        
####################################################################

def interp_any(x_array, y_array, x_new=None):
    interp_func = interp1d(x_array, y_array, kind='linear', fill_value="extrapolate")
    y_new = interp_func(x_new)   
    return x_new, y_new
    
##############################################################################

def get_O3_profile(x_new):
    z, p, _, t, md = atmp.gl_atm(atmp.MIDLATITUDE_SUMMER)
    # Ozone is usually identified by atmp.O3 index in molecular species array
    o3_profile_ppmv = md[:, atmp.O3]
    z_new, O3_new = interp_any(z*1000, o3_profile_ppmv, x_new=x_new)
    return z_new, O3_new
    
##############################################################################

def write_armsgb_input_nc(profile_indices, level_pressures,
        level_temperatures, level_wvs,level_ppmvs, level_liq,level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, ZA,\
        times ,outfile="", clear_sky_bool=True):
        
    if clear_sky_bool:
        level_liq[:,:] = 0.

    # Ermitteln der Dimensionen
    n_levels, n_profiles = level_pressures.shape

    # Optional: Konvertiere Inputs in float32 falls nötig
    level_pressures = np.array(level_pressures, dtype=np.float32)
    level_temperatures = np.array(level_temperatures, dtype=np.float32)
    level_wvs = np.array(level_wvs, dtype=np.float32)
    level_ppmvs = np.array(level_ppmvs, dtype=np.float32)
    level_liq = np.array(level_liq, dtype=np.float32)
    level_z = np.array(level_z, dtype=np.float32)
    level_rhs = np.array(level_rhs, dtype=np.float32)
    srf_pressures = np.array(srf_pressures, dtype=np.float32)
    srf_temps = np.array(srf_temps, dtype=np.float32)
    srf_wvs = np.array(srf_wvs, dtype=np.float32)
    srf_altitude = np.array(srf_altitude, dtype=np.float32)
    ZA = np.array(ZA, dtype=np.float32)
    profile_indices = np.array(profile_indices, dtype=np.int32)
    
    ##################
    # Static value of O3 0.06 ppmv is now replaced by climatology profile:
    level_o3s = np.empty(np.shape(level_wvs))
    z_new, O3_new = get_O3_profile(level_z)
    # level_o3s[:,:] = 0.06 
    level_o3s[:,:] = O3_new


    ####################

    # Setze Dummy-Werte für Dimensionsgrößen
    n_times = len(profile_indices)
    n_channels = 14  # Beispielwert
    any_obs = np.empty([14, n_times])
    #####################################
    # any_obs = np.full([14, n_times], np.nan)
    # => Lösungsversuch hat nicht geholfen!
    #############################
    n_data = 1       # Wird oft für Metadaten genutzt

    ds = xr.Dataset(
        data_vars={
            # Okay dieser Code-Block hat meinen Segfault error gelöst:
            "Times_Number": ("N_Data", np.array([n_times], dtype=np.int32)),
            "Levels_Number": ("N_Data", np.array([n_levels], dtype=np.int32)),
            "Profiles_Number": ("N_Data", np.array([n_profiles], dtype=np.int32)),
            
            # Profilebenen
            "Level_Pressure":       (("N_Levels", "N_Profiles"), level_pressures),
            "Level_Temperature":    (("N_Levels", "N_Profiles"), level_temperatures),
            "Level_H2O":            (("N_Levels", "N_Profiles"), level_wvs),
            "Level_ppmvs":          (("N_Levels", "N_Profiles"), level_ppmvs),
            "Level_Liquid":         (("N_Levels", "N_Profiles"), level_liq),
            "Level_z":              (("N_Levels", "N_Profiles"), level_z),
            'Level_O3':             (("N_Levels", "N_Profiles"), level_o3s),
            "Level_RH":              (("N_Levels", "N_Profiles"), level_rhs),

            # Oberflächenparameter
            "times":                (("N_Times"), times),
            "Obs_Surface_Pressure": (("N_Times",), srf_pressures),
            "Obs_Temperature_2M":   (("N_Times",), srf_temps),
            "Obs_H2O_2M":           (("N_Times",), srf_wvs),
            "Surface_Pressure":     (("N_Profiles",), srf_pressures),
            "Temperature_2M":       (("N_Profiles",), srf_temps),
            "H2O_2M":               (("N_Profiles",), srf_wvs),
            "Surface_Altitude":     (("N_Profiles",), srf_altitude),
            
            'Obs_BT':               (("N_Channels","N_Times",), np.array(any_obs)),
            'Sim_BT':               (("N_Channels","N_Times",), np.array(any_obs)),
            'OMB':                  (("N_Channels","N_Times",), np.array(any_obs)),
            
            'QC_Flag':              (("N_Times",), np.zeros([n_times])),

            # Zusätzliche Metadaten
            "Profile_Index":        (("N_Times",), profile_indices.astype(np.float64)),
            "GMRZenith":            (("N_Times",), 90-ZA), # Elevationswinkel!!!
            # Kein Zenitwinkel zsa:0; GMR: 90
            
        },
        
        coords={
            "N_Data":     np.arange(n_data),
            "N_Channels": np.arange(n_channels),
            "N_Times":    np.arange(n_times),
            "N_Levels":   np.arange(n_levels),
            "N_Profiles": profile_indices,
        }
    )

    # Add units:
    ds["Level_Pressure"].attrs["units"] = "hPa"
    ds["Level_Temperature"].attrs["units"] = "K"
    ds["Level_H2O"].attrs["units"] = "g/kg"
    ds["Level_ppmvs"].attrs["units"] = "ppmv"
    ds["Level_Liquid"].attrs["units"] = "kg/kg"
    ds["Level_z"].attrs["units"] = "m"
    ds["Level_RH"].attrs["units"] = "%"
    
    # Schreibe die NetCDF-Datei
    ds.to_netcdf(outfile, format="NETCDF4_CLASSIC")

    return 0

##############################################################################

def create_input4arms_gb_per_elevation(ds, args, elev_index,\
        outpath="~/PhD_data/TB_preproc_and_proc_results/arms/",\
         n_levels=n_levels, valid_indices =[]):

    elevation = ds["elevation"].values[elev_index]
    n_times = len(ds["time"].values)
    n_crops = len(ds["Crop"].values)      

    N_profiles = len(valid_indices)
    
    # Initialisiere Output-Arrays
    level_pressures    = np.full((n_levels, N_profiles), np.nan)
    level_temperatures = np.full((n_levels, N_profiles), np.nan)
    level_wvs          = np.full((n_levels, N_profiles), np.nan)
    level_ppmvs        = np.full((n_levels, N_profiles), np.nan)
    level_liq          = np.full((n_levels, N_profiles), np.nan)
    level_z            = np.full((n_levels, N_profiles), np.nan)
    level_rhs          = np.full((n_levels, N_profiles), np.nan)

    srf_pressures      = np.full((N_profiles), np.nan)
    srf_temperatures   = np.full((N_profiles), np.nan)
    srf_wvs            = np.full((N_profiles), np.nan)
    srf_altitude       = np.full((N_profiles), np.nan)
    if elev_index==0: 
        ZA             = np.full((N_profiles), 0.)
    else:
        ZA             = np.full((N_profiles), 90. - elevation)
    times              = np.full((N_profiles), np.nan)
    profile_indices    = []

    for total_index, (i, j, k) in enumerate(valid_indices):
        profile_indices.append(total_index)
        level_pressures[:, total_index]    = ds["Level_Pressure"].values[:, i, k]
        level_temperatures[:, total_index] = ds["Level_Temperature"].values[:, i, k]
        level_wvs[:, total_index]          = ds["Level_H2O"].values[:, i, k]
        level_ppmvs[:, total_index]        = ds["Level_ppmvs"].values[:, i, k]
        level_liq[:, total_index]          = ds["Level_Liquid"].values[:, i, k]
        level_rhs[:, total_index]          = ds["Level_RH"].values[:, i, k]
        level_z[:, total_index]            = ds["Level_z"].values[:, i, k]
        srf_pressures[total_index]         = ds["Surface_Pressure"].values[i, k]
        srf_temperatures[total_index]      = ds["Temperature_2M"].values[i, k]
        srf_wvs[total_index]               = ds["H2O_2M"].values[i, k]
        srf_altitude[total_index]          = ds["Surface_Altitude"].values[i, k]
        times[total_index]                 = ds["time"].values[i]

    ########################################
    # Was bewirkt dieser Absatz???? Auf einmal funktioniert 90° wieder...
    # Hat seine Magie noch immer nicht verloren...
    outfilename = outpath+f"elevation_{elev_index}.nc"
    write_armsgb_input_nc(
        profile_indices,
        level_pressures, level_temperatures, level_wvs, level_ppmvs,
        level_liq, level_z, level_rhs,
        srf_pressures, srf_temperatures, srf_wvs, srf_altitude, ZA,
        times, outfile=outfilename
    )
    ########################################

    outfilename = outpath+"arms_gb_inputs.nc"
    write_armsgb_input_nc(
        profile_indices,
        level_pressures, level_temperatures, level_wvs, level_ppmvs,
        level_liq, level_z, level_rhs,
        srf_pressures, srf_temperatures, srf_wvs, srf_altitude, ZA,
        times, outfile=outfilename
    )
    return outfilename
    
##############################################################################

def read_outputs_arms_gb(ds, infile, valid_indices):
    ds_arms = xr.open_dataset(infile)
    TBs1 = np.full((len(ds["time"].values),\
        len(ds["N_Channels"].values), len(ds["elevation"].values),\
        len(ds["Crop"].values)), np.nan)

    for total_index, indices in enumerate(valid_indices):
        i,j,k = indices          
        TBs1[i, :,j, k]=\
            ds_arms["Sim_BT"].isel(N_Times=total_index).values
            
    return TBs1
    
##############################################################################

def get_valid_profiles(ds):
    n_elevs = len(ds["elevation"].values)
    n_times = len(ds["time"].values)
    n_crops = len(ds["Crop"].values)
    
    list_of_valid_indices = []
    all_valid_indices = []
    
    for elev_index in range(n_elevs):
        valid_indices = []
        for i in range(n_times):
            for k in range(n_crops):
                if not check4NANs_in_time_crop_ele(ds, i, k):
                    valid_indices.append((i, elev_index, k))
        list_of_valid_indices.append(valid_indices)
        all_valid_indices.extend(valid_indices)
        
    return all_valid_indices, list_of_valid_indices


##############################################################################

def derive_TBs4ARMS_gb_per_elevation(ds, args, n_levels=n_levels,\
        outpath=""):

    all_tbs = np.full((len(ds["time"].values),\
        len(ds["N_Channels"].values), len(ds["elevation"].values),\
        len(ds["Crop"].values)), np.nan)
        
    # Brillante neue Bestimmung der validen Indices!::
    all_valid_indices, list_of_valid_lists = get_valid_profiles(ds)
    
    for elev_index in range(len(ds["elevation"].values)):
        elevation = ds["elevation"].values[elev_index]
        print(f"Processing elevation index {elev_index}; elevation: {elevation}")
        
        # 1st Für jede Elevation Inputdatei erstellen und ARMS-gb Modell laufen lassen
        infile = create_input4arms_gb_per_elevation(\
                ds, args, elev_index,outpath=outpath, n_levels=n_levels,\
                valid_indices=list_of_valid_lists[elev_index])
        
        # 2nd Modell ausführen
        ex_script = str(os.path.expanduser("~/MWR_fast_forward_operators_and_LBLs/python_src/proc/run_arms_gb.sh"))
        subprocess.run(ex_script)
        
        # 3rd Read TBs:
        tbs1 = read_outputs_arms_gb(ds, infile, list_of_valid_lists[elev_index])
        tbs1[tbs1 == 0] = np.nan
        all_tbs[:,:,elev_index,:] = tbs1[:,:,elev_index,:]
        
        #################################
        # Delete later:
        # if elev_index==4:
        #     break
        ################################
        
    # 4th append ds:
    # concatenated_tbs = np.concatenate(all_tbs, axis=2)
    TBs_ARMS_gb = xr.DataArray(
        all_tbs,
        dims=("time", "N_Channels", "elevation", "Crop"),
    )
    ds["TBs_ARMS_gb"] = TBs_ARMS_gb
        
    return ds

##############################################################################

def dir_of_file(string):
    pfad = Path(string)
    uebergeordnet = pfad.parent
    return str(uebergeordnet) + "/"

##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    ds = xr.open_dataset(args.input)

    # 1st Derive TBs for clear sky for ARMS-gb
    ds = derive_TBs4ARMS_gb_per_elevation(ds, args,\
        outpath=dir_of_file(args.output)+"arms/")

    # 2nd Print dataset to NetCDF
    ds.to_netcdf(args.output, format="NETCDF4_CLASSIC")
        

        

