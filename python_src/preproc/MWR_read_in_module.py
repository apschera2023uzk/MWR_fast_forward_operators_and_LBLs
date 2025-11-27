#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Author: Alexander Pschera (apscher1@uni-koeln.de / alexander.pschera@posteo.de)
# Takes a number of NetCDF files from radiosondes and MWR as input
# Creates one NetCDF that contains files in RT-model friendly structure. 

# Important limitations:
# => MWR and rs are matched if there is maximum a X minute difference between both measurement times.
# => LWC and LWP are derived via Nandan et al. 2022; Mixed phase clouds are assumed liquid; IWC is assumed same as LWC.

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
sys.path.append('/home/aki/pyrtlib')
from pyrtlib.climatology import AtmosphericProfiles as atmp
from pyrtlib.utils import ppmv2gkg, mr2rh
from datetime import datetime, timezone
from derive_cloud_water import derive_cloud_features
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

##############################################################################
# 1.5: Parameter
##############################################################################

elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])
azimuths = np.arange(0.,355.1,5.) # Interpoliere dazwischen!
min_time_diff_thres = 15 
n_levels=180
max_elev_azi_diff = 0.05 #°

dwdhat_pattern = "/home/aki/PhD_data/FESSTVaL_14GB/dwdhat/l*/*/*/*.nc"
foghat_pattern = "/home/aki/PhD_data/FESSTVaL_14GB/foghat/l*/*/*/*.nc"
sunhat_pattern = "/home/aki/PhD_data/FESSTVaL_14GB/sunhat/l*/*/*/*.nc"
tophat_pattern = "/home/aki/PhD_data/Socles/MWR_tophat/*.nc"
joyhat_pattern = "/home/aki/PhD_data/Vital_I/hatpro-joyhat/*.nc"
hamhat_pattern = "/home/aki/PhD_data/Vital_I/hamhat/*.nc"

##############################################################################
# 2nd Used Functions
##############################################################################

def nearest_ele4elevation(ele_values, azi_values, ele_times,\
        target_elevation,target_azi,datetime_np,\
        min_time_diff_thres=min_time_diff_thres,\
        max_elev_azi_diff=max_elev_azi_diff):
    # Boolean Maske für alle Stellen, an denen "ele" exakt target_elevation ist
    match_mask = (abs(ele_values-target_elevation)<max_elev_azi_diff)   
    if target_azi =="ANY":
        match_mask2 = [True]*len(ele_values)
    else:
        match_mask2 = (abs(azi_values-target_azi)<max_elev_azi_diff)
    final_mask = match_mask & match_mask2
    
    if not final_mask.any():
        nearest_idx = None
    else:
        # Zeitwerte, die zu den passenden Elevations gehören
        candidate_times = ele_times[final_mask]
        time_diffs = np.abs(candidate_times - datetime_np)
        min_idx = time_diffs.argmin()
        nearest_time = candidate_times[min_idx]
        
        # Derive time difference:
        mwr_time = candidate_times[min_idx].astype('M8[us]').astype(datetime)
        rs_time = datetime_np.astype('M8[us]').astype(datetime)
        delta = mwr_time - rs_time
        minutes_diff = delta.total_seconds() / 60
        if minutes_diff>min_time_diff_thres:
            print("Excluded due to huge time difference from scan!!")
            return None
            
        
        # Deterime nearest index:
        candidate_indices = np.where(final_mask)[0]
        nearest_idx = candidate_indices[min_idx]
        
        # Elevation check:
        nearest_value = ele_values[nearest_idx]
        nearest_value2 = azi_values[nearest_idx]
        if target_azi =="ANY":
            if abs(nearest_value-target_elevation)>max_elev_azi_diff:
                print("WARNING: Azimuth or Elevation does not agree, as expected!")
                print("Ele/Azi tagret values: ",target_elevation, target_azi)
                print("Ele/Azi found values: ",nearest_value,\
                    nearest_value2)            
        elif abs(nearest_value-target_elevation)>max_elev_azi_diff or\
                 (abs(nearest_value2-target_azi)>max_elev_azi_diff):
             print("WARNING: Azimuth or Elevation does not agree, as expected!")
             print("Ele/Azi tagret values: ",target_elevation, target_azi)
             print("Ele/Azi found values: ",nearest_value, nearest_value2)
                    
    return nearest_idx

##############################################################################

def nearest_ele4elevation_mean(ele_values, azi_values, ele_times,
        target_elevation, target_azi, datetime_np,\
        min_time_diff_thres=min_time_diff_thres,\
        max_elev_azi_diff=max_elev_azi_diff):
        
    # Boolean Maske für Elevation und Azimuth
    match_mask = (abs(ele_values - target_elevation)<max_elev_azi_diff)
    if target_azi == "ANY":
        match_mask2 = [True] * len(ele_values)
    else:
        match_mask2 = (abs(azi_values - target_azi)<max_elev_azi_diff)
    final_mask = match_mask & match_mask2
    
    if not final_mask.any():
        nearest_idx_list = None
    else:
        # Zeitdifferenzen zu allen passenden Zeitpunkten
        candidate_times = ele_times[final_mask]
        time_diffs = np.abs(candidate_times - datetime_np)
        minutes_diff_all = time_diffs.astype('timedelta64[s]').astype(float) / 60
        
        # Alle Indizes, die innerhalb des Zeitfensters liegen
        valid_mask = minutes_diff_all <= min_time_diff_thres
        if not valid_mask.any():
            print("Excluded: no measurements within time window.")
            nearest_idx_list = None
        else:
            nearest_idx_list = np.where(final_mask)[0][valid_mask]

    return nearest_idx_list

##############################################################################
    
def derive_elevation_index(ds_bl, elevation):
    for index, ele in enumerate(ds_bl["ele"].values):
        if abs(ele-elevation)<0.05:
            return index
    return None
    
##############################################################################

def time_indices_list4BL(ds_bl, datetime_np,\
        min_time_diff_thres=min_time_diff_thres):
        
    time_diffs = np.abs(ds_bl["time"].values - datetime_np)  
    time_diffs_seconds = time_diffs / np.timedelta64(1, 's') 
    indices_within_15min =\
        np.where(time_diffs_seconds <= min_time_diff_thres* 60)[0]  
    indices_list = indices_within_15min.tolist() 
    
    if indices_list:
        return indices_list
    else:
        return None

##############################################################################

def get_tbs_from_l1(l1_files, datetime_np, elevations=elevations,\
        azimuths=azimuths): 
    tbs = np.full((10, 72, 14), np.nan)
    lat = np.nan; lon = np.nan
    
    for file in l1_files:
        
        # BL-files:
        if "BL" in file:
            ds_bl = xr.open_dataset(file)
            for i,elevation in enumerate(elevations):
                ele_index = derive_elevation_index(ds_bl, elevation)
                if ele_index==None:
                    continue
                else:
                    idx_list = time_indices_list4BL(ds_bl, datetime_np)
                    if idx_list is not None and len(idx_list) > 0:
                        for ch in range(len(tbs[ele_index,0,:])):
                            tbs[ele_index,0, ch] = np.nanmean(ds_bl["tb"].values[idx_list,ele_index,ch])
                    
        # 1C01-files:
        elif "MWR_1C01" in file:
            ds_c1 = xr.open_dataset(file)            
            for i,elevation in enumerate(elevations):
                for j,azi in enumerate(azimuths):
                    time_idx_list = nearest_ele4elevation_mean(ds_c1["elevation_angle"].values,
                        ds_c1["azimuth_angle"].values,
                        ds_c1["time"].values, elevation, azi, datetime_np)
                    if time_idx_list is not None and len(time_idx_list) > 0:                       
                        for ch in range(len(tbs[i,j, :])):
                            tbs[i,j, ch] = np.nanmean(ds_c1["tb"].values[time_idx_list,ch])    
            if "latitude" in ds_c1.data_vars:
                lat = ds_c1["latitude"].values[0]
                lon = ds_c1["longitude"].values[0]
            elif "lat" in ds_c1.data_vars:
                lat = ds_c1["lat"].values
                lon = ds_c1["lon"].values
                
        # mwr-files:           
        else: 
            ###
            # Reason for double n_freq warning: 
            ds_mwr = xr.open_dataset(file)
            ###
            for i,elevation in enumerate(elevations):
                for j,azi in enumerate(azimuths):
                    time_idx_list = nearest_ele4elevation_mean(ds_mwr["ele"].values,\
                        ds_mwr["azi"].values,\
                        ds_mwr["time"].values, elevation,azi, datetime_np)
                        
                    if time_idx_list is not None and len(time_idx_list) > 0:
                        for ch in range(len(tbs[i,j, :])):
                            tbs[i,j, ch] = np.nanmean(ds_mwr["tb"].values[time_idx_list,ch])     
            if "latitude" in ds_mwr.data_vars:
                lat = ds_mwr["latitude"].values[0]
                lon = ds_mwr["longitude"].values[0]
            elif "lat" in ds_mwr.data_vars:
                lat = ds_mwr["lat"].values
                lon = ds_mwr["lon"].values 
                
    return tbs, lat, lon
   
##############################################################################

def interpolate_preserve_old_points_fix(x_old, new_length):
    total_new_points = new_length - len(x_old)
    n_intervals = len(x_old) - 1
    points_per_interval = total_new_points // n_intervals
    remainder = total_new_points % n_intervals

    x_new = []
    for i in range(n_intervals):
        
        if i==0:
            count = remainder + points_per_interval
        else:
            count = points_per_interval
        segment = np.linspace(x_old[i], x_old[i+1], count+2)
        
        
        # Punkte (bis auf letztes Intervall) ohne den letzten Punkt anhängen
        if i < n_intervals - 1:
            x_new.extend(segment[:-1])
        else:
            x_new.extend(segment)
    x_new = np.sort(np.array(x_new))        
    return np.array(x_new)

####################################################################

def interp2_180(x_array, y_array, x_new=None, n_levels=n_levels):
    # if x_new is None:
    x_new = interpolate_preserve_old_points_fix(x_array, n_levels)
    interp_func = interp1d(x_array, y_array, kind='linear', fill_value="extrapolate")
    y_new = interp_func(x_new)   
    return x_new, y_new
    
##############################################################################

def check_lwp_iwv(lwp, iwv):
    if isinstance(lwp, np.ndarray):
        lwp = np.nan    
    elif lwp<0:
        lwp = 0.
    if isinstance(iwv, np.ndarray):
        iwv = np.nan   
    elif iwv<0:
        iwv = 0.        
    return lwp, iwv

##############################################################################

def get_profs_from_l2(l2_files, datetime_np, n_levels = n_levels):
    data = np.full((4,n_levels), np.nan)
    lwp, iwv = np.nan, np.nan

    for file in l2_files:
    
        if "single" in file:
            ds = xr.open_dataset(file)    
            time_idx_list = nearest_ele4elevation_mean(\
                ds["elevation_angle"].values,\
                ds["azimuth_angle"].values,\
                ds["time"].values, 90., "ANY" , datetime_np)       
            if time_idx_list is not None and len(time_idx_list) > 0:   
                mean_temps = np.nanmean(\
                     ds["temperature"].values[time_idx_list, :], axis=0)
                x_new, y_new = interp2_180(ds["height"].values,\
		        mean_temps)
                data[0,:] = x_new     
                data[1,:] = y_new  
                mean_hums = np.nanmean(\
                    ds["absolute_humidity"].values[time_idx_list, :], axis=0)
                x_new, y_new = interp2_180(ds["height"].values,\
		        mean_hums)       
                data[3,:] = y_new  
                lwp = np.nanmean(ds["lwp"].values[time_idx_list])         
                iwv = np.nanmean(ds["iwv"].values[time_idx_list])
                     
        if "mwr0" in file and "_l2_ta_" in file:
            ds = xr.open_dataset(file)
            time_idx_list = nearest_ele4elevation_mean(\
                ds["ele"].values, ds["azi"].values,\
                ds["time"].values, 90., "ANY" , datetime_np)    
            if time_idx_list is not None and len(time_idx_list) > 0: 
                mean_temp = np.nanmean(ds["ta"].values[time_idx_list, :],axis=0)
                x_new, y_new = interp2_180(ds["height"].values,\
                     mean_temp)
                data[0,:] = x_new     
                data[1,:] = y_new
                
        elif "mwrBL0" in file and "_l2_ta_" in file:
            ds = xr.open_dataset(file)
            time_idx_list = time_indices_list4BL(ds, datetime_np)
            if time_idx_list is not None and len(time_idx_list) > 0:
                mean_BL = np.nanmean(ds["ta"].values[time_idx_list, :], axis=0)
                x_new1, y_new_after = interp2_180(
                    ds["height"].values,
                    mean_BL
                )
                data[2,:] = y_new_after                 
                        
        elif "_hua_" in file:
            ds = xr.open_dataset(file)
            time_idx_list = time_indices_list4BL(ds, datetime_np)
            if time_idx_list is not None and len(time_idx_list) > 0:
                mean_hua = np.nanmean(ds["hua"].values[time_idx_list, :], axis=0)
                x_new2, y_new_after = interp2_180(
                    ds["height"].values,
                    mean_hua
                )
                data[3,:] = y_new_after               
                
        elif "_prw_" in file:
            ds = xr.open_dataset(file)
            time_idx_list = time_indices_list4BL(ds, datetime_np)

            if time_idx_list is not None and len(time_idx_list) > 0:
                iwv_after = np.nanmean(ds["prw"].values[time_idx_list])
                iwv = iwv_after
            
        elif "_clwvi_" in file:
            ds = xr.open_dataset(file)
            time_idx_list = time_indices_list4BL(ds, datetime_np)
            if time_idx_list is not None and len(time_idx_list) > 0:
                lwp_after = np.nanmean(ds["clwvi"].values[time_idx_list])
                lwp = lwp_after        
            
    lwp, iwv = check_lwp_iwv(lwp, iwv)
    return data[:,::-1], lwp, iwv

##############################################################################

def get_mwr_data(datetime_np, mwrs,n_levels=n_levels,\
        dwdhat_pattern=dwdhat_pattern,foghat_pattern=foghat_pattern,\
        sunhat_pattern =sunhat_pattern, tophat_pattern =tophat_pattern,\
        joyhat_pattern =joyhat_pattern, hamhat_pattern =hamhat_pattern):
        
    datestring = str(datetime_np).replace("T","").replace(":","").replace("-","")[:8]
    
    if "dwdhat" in mwrs:
        dwd_files = glob.glob(dwdhat_pattern)   
        files = [file for file in dwd_files if datestring in file]   
        l1_files = [file for file in files if "l1" in file]   
        l2_files = [file for file in files if "l2" in file] 
        tbs_dwdhat, lat, lon = get_tbs_from_l1(l1_files, datetime_np)
        dwd_profiles, lwp_dwd, iwv_dwd = get_profs_from_l2(l2_files, datetime_np)
        dwd_profiles[0,:] = dwd_profiles[0,:] + 112
    else:
        tbs_dwdhat = np.full((10, 72, 14), np.nan)
        dwd_profiles = np.full((4,n_levels), np.nan)
        lwp_dwd, iwv_dwd = np.nan, np.nan
    if "foghat" in mwrs:
        fog_files = glob.glob(foghat_pattern)
        files = [file for file in fog_files if datestring in file]   
        l1_files = [file for file in files if "l1" in file]   
        l2_files = [file for file in files if "l2" in file] 
        tbs_foghat, lat, lon = get_tbs_from_l1(l1_files, datetime_np)
        fog_profiles, lwp_fog, iwv_fog = get_profs_from_l2(l2_files, datetime_np)
        fog_profiles[0,:] = fog_profiles[0,:] + 112
    else:
        tbs_foghat = np.full((10, 72, 14), np.nan)
        fog_profiles = np.full((4,n_levels), np.nan)
        lwp_fog, iwv_fog = np.nan, np.nan        
    if "sunhat" in mwrs:
        sun_files = glob.glob(sunhat_pattern)
        files = [file for file in sun_files if datestring in file]   
        l1_files = [file for file in files if "l1" in file]   
        l2_files = [file for file in files if "l2" in file] 
        tbs_sunhat, lat, lon = get_tbs_from_l1(l1_files, datetime_np)
        sun_profiles, lwp_sun, iwv_sun = get_profs_from_l2(l2_files, datetime_np)
        sun_profiles[0,:] = sun_profiles[0,:] + 74
    else:
        tbs_sunhat = np.full((10, 72, 14), np.nan)    
        sun_profiles = np.full((4,n_levels), np.nan)
        lwp_sun, iwv_sun = np.nan, np.nan            
    if "tophat" in mwrs:
        top_files = glob.glob(tophat_pattern)
        files = [file for file in top_files if datestring in file]   
        l1_files = [file for file in files if "l1" in file]   
        l2_files = [file for file in files if "l2" in file] 
        tbs_tophat, lat, lon = get_tbs_from_l1(l1_files, datetime_np)
        top_profiles, lwp_top, iwv_top = get_profs_from_l2(l2_files, datetime_np)
        top_profiles[0,:] = top_profiles[0,:] + 110
    else:
        tbs_tophat = np.full((10, 72, 14), np.nan)
        top_profiles = np.full((4,n_levels), np.nan)
        lwp_top, iwv_top = np.nan, np.nan        
    if "joyhat" in mwrs:
        joy_files = glob.glob(joyhat_pattern)
        files = [file for file in joy_files if datestring in file]   
        l1_files = [file for file in files if "1C01" in file]   
        l2_files = [file for file in files if "single" in file] 
        tbs_joyhat, lat, lon = get_tbs_from_l1(l1_files, datetime_np)
        joy_profiles, lwp_joy, iwv_joy = get_profs_from_l2(l2_files, datetime_np) 
    else:
        tbs_joyhat = np.full((10, 72, 14), np.nan)
        joy_profiles = np.full((4,n_levels), np.nan)
        lwp_joy, iwv_joy = np.nan, np.nan        
    if "hamhat" in mwrs:
        ham_files = glob.glob(hamhat_pattern)
        files = [file for file in ham_files if datestring in file]   
        l1_files = [file for file in files if "1C01" in file]   
        l2_files = [file for file in files if "single" in file] 
        tbs_hamhat, lat, lon = get_tbs_from_l1(l1_files, datetime_np)
        ham_profiles, lwp_ham, iwv_ham = get_profs_from_l2(l2_files, datetime_np)
    else:
        tbs_hamhat = np.full((10, 72, 14), np.nan)
        ham_profiles = np.full((4,n_levels), np.nan)
        lwp_ham, iwv_ham = np.nan, np.nan        
        
    #######################################
    # Then jsut read l1-files for now
    # TBs of shape: (elevation x azimuth x n_chans) , for one timestep
    integrals = [lwp_dwd, iwv_dwd, lwp_fog, iwv_fog, lwp_sun, iwv_sun,\
        lwp_top, iwv_top, lwp_joy, iwv_joy,lwp_ham, iwv_ham ]
    # [-0.001904392, 23.411406, -0.004118612, 31.603878, nan, nan, nan, nan, nan, nan, nan, nan]
    # => Warum habe ich denn für sunhat, tophat, joyhat, hamhat keine sinnvollen Werte???
    # Was mache ich mit negativen Werten???
    
    return tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles,fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, integrals, lat, lon

##############################################################################
        

