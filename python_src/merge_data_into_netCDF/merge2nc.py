#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import xarray as xr
import pandas as pd
import numpy as np
import glob
import os
from scipy.interpolate import interp1d
from Sc_module import clausius_clapeyron, rh2mixing_ratio
from datetime import datetime, timedelta

##############################################################################
# 2nd Used functions:
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
    )
    parser.add_argument(
        "--radiosondes", "-rs",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes/"),
        help="Pfad zum Verzeichnis mit den Radiosonden-Rohdaten and RTTOV-gb outputs"
    )
    parser.add_argument(
        "--mwr_path1", "-l1",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/hatpro-joyhat/"),
        help="MWR l1 data with Tbs"
    )
    parser.add_argument(
        "--mwr_path2", "-l2",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/hatpro-joyhat/"),
        help="MWR l2 data with atmospheric profiles"
    )
    parser.add_argument(
        "--mwrpy_ret", "-mp",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/mwrpy_ret_outputs/"),
        help="Simulated Brightness temperatures by LBL mwrpy_ret"
    )
    return parser.parse_args()

##############################################################################

def read_radiosonde_csv(file="anyfile"):
    dataframe = pd.read_csv(file,skiprows=9, encoding_errors='ignore',\
        header=None,names=["HeightMSL","Temp","Dewp","RH","P","Lat","Lon",\
        "AscRate","Dir","Speed","Elapsed time"])
    df_resampled =  pd.concat([dataframe.iloc[:100:15],\
        dataframe.iloc[100:500:15],dataframe.iloc[500:1000:20],\
        dataframe.iloc[1000:1500:25],dataframe.iloc[1500:3000:25],\
        dataframe.iloc[3000::50]])
    length_value = len(df_resampled)
    t_array = df_resampled["Temp"].values[::-1] + 273.15
    p_array = df_resampled["P"].values[::-1]

    m_array = []
    for rh_lev, t_lev, p_lev in zip (df_resampled["RH"].values[::-1],\
            t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)
    
    surf_height_in_km = df_resampled["HeightMSL"].values[0]/1000
    heights_in_m = df_resampled["HeightMSL"].values
    deg_lat = df_resampled["Lat"].values[0]
    
    return length_value, p_array, t_array, ppmv_array,\
        surf_height_in_km, deg_lat, m_array, heights_in_m

##############################################################################

def read_all_inputs(args):
    rs_files = glob.glob(args.radiosondes+"csvexport_*.txt")
    mwr_files_l1 = glob.glob(args.mwr_path1+"MWR_1C01_XXX_*.nc")
    mwr_files_l2 = glob.glob(args.mwr_path2+"MWR_single*.nc")
    rttov_files = glob.glob(args.radiosondes+"rttov-gb_*.txt")
    lbl_files = glob.glob(args.mwrpy_ret+"mwrpy_ret_out_rs_*.nc")
    return np.sort(rs_files), mwr_files_l1, mwr_files_l2, rttov_files, lbl_files

##############################################################################

def derive_datetime_of_sonde(file):
    datestring = (file.split("/")[-1]).split("_")[1]
    timestring = ((file.split("/")[-1]).split("_")[2]).split(".")[0]
    new_timestring = timestring
    datetime_of_sonde = np.datetime64(datestring[:4]+"-"+datestring[4:6]+\
        "-"+datestring[6:8]+"T"+new_timestring[:2]+\
        ":"+new_timestring[2:]+":00")
    return datetime_of_sonde

##############################################################################

def derive_datetime_of_mwr(file, timedelta_min=1): # _plus10min
    datestring = (file.split("/")[-1]).split("_")[1]
    timestring = ((file.split("/")[-1]).split("_")[2]).split(".")[0]

    # Shift time by 10 minutes
    dt = datetime.strptime(timestring, "%H%M")
    dt_new = dt + timedelta(minutes=timedelta_min)
    new_timestring = dt_new.strftime("%H%M")
    #new_timestring = timestring
    
    datetime_of_mwr_plus10 = np.datetime64(datestring[:4]+"-"+datestring[4:6]+\
        "-"+datestring[6:8]+"T"+new_timestring[:2]+\
        ":"+new_timestring[2:]+":00")
    datetime_of_mwr_orig = np.datetime64(datestring[:4]+"-"+\
        datestring[4:6]+"-"+datestring[6:8]+"T"+timestring[:2]+\
        ":"+timestring[2:]+":00")
    return datetime_of_mwr_plus10, datetime_of_mwr_orig

##############################################################################

def get_mwr_tbs_of_datetime(datetime_of_mwr_plus20, original_mwr_datetime, args):
    quality_flag = 0
    mwr_path = args.mwr_path1
    ds_c1 = xr.open_dataset(mwr_path+"/MWR_1C01_XXX_202408"+\
        str(datetime_of_mwr_plus20)[8:10]+".nc")
    t_index0 = np.argmin(\
        [abs(value) for value in [ds_c1["time"].values - original_mwr_datetime]] )
    t_index20 = np.argmin(\
        [abs(value) for value in [ds_c1["time"].values - datetime_of_mwr_plus20]] )
    time_diff = float(ds_c1["time"].values[t_index0]-original_mwr_datetime)

    # print("15min_diff: ", t_index0," bis ",t_index20)
    
    
    # Exclude certain data:
    # 1. No MWR measurement during radiosonde launch:
    if abs(time_diff)>100000000000: # 100 Sekunden
        quality_flag = 1
        print("Excluded because of time difference: ",\
            ds_c1["time"].values[t_index0],  " == ", original_mwr_datetime)
    
    # 2. Liquid cloud flag in MWR data:
    elif (ds_c1["liquid_cloud_flag"].values[t_index0-558:t_index0+558] !=0).any():
        # t_index0:t_index20
        quality_flag = 2
        print("Excluded because of liquid cloud: ", original_mwr_datetime)

    # 3. No zenith scan:
    elif (abs(ds_c1["elevation_angle"].values[t_index0:t_index20]-90)>0.5).any():
        quality_flag = 3
        print("Excluded because of slant path: ", original_mwr_datetime)

        '''
        # Shift time according to time difference rule:
        if (abs(ds_c1["elevation_angle"].values[t_index0+50:t_index20+50]-90)>0.5).any(): # + 5 Minuten == 187
            t_index0 = t_index0+50
            t_index20 = t_index20+50
            quality_flag = 0
        elif (abs(ds_c1["elevation_angle"].values[t_index0-50:t_index20-50]-90)>0.5).any(): # 74 == - 2 Minuten
            t_index0 = t_index0-50
            t_index20 = t_index20-50
            quality_flag = 0
        '''

    tbs_mwr = ds_c1["tb"][t_index0:t_index20, :].mean(dim="time", skipna=True).values
    mean_cloudflag = np.nanmean(ds_c1["liquid_cloud_flag"].values[t_index0-558:t_index0+558])
    if np.isnan(mean_cloudflag):
        mean_cloudflag = 0
    mean_rainfall = np.nanmean(ds_c1["rainfall_rate"].values[t_index0-558:t_index0+558])
    if np.isnan(mean_rainfall):
        mean_rainfall = 0
    elevation = np.nanmean(ds_c1["elevation_angle"].values[t_index0:t_index20])
    print("Bedenke das MWR TBs ein Mittelwert über 1 Minute oder ",\
        len((ds_c1["elevation_angle"].values[t_index0:t_index20])), " Datenpunkte sind.")
    print("Zeitfenster von: ", ds_c1["time"].values[t_index0], " bis ", ds_c1["time"].values[t_index20])
    frequency = ds_c1["frequency"].values
    std31 = np.nanstd(ds_c1["tb"][t_index0-1122:t_index0+1122, 6].values)
    print("31 GHz values: ", ds_c1["tb"][t_index0-1122:t_index0+1122, 6].values)
 
    return tbs_mwr, quality_flag, mean_cloudflag, frequency, mean_rainfall, std31, elevation

##############################################################################

def get_radisonde_tbs_of_file(rttovgb_outfile):
    # Diese Funktion läuft einwandfrei.
    switch = False
    switch_count = 0
    tb_string = ""
    file = open(rttovgb_outfile, "r")
    for i, line in enumerate(file.readlines()):
        if switch and switch_count<2:
            switch_count+= 1
            tb_string+= line
        elif "CALCULATED BRIGHTNESS TEMPERATURES (K):" in line:
            switch = True
    liste = tb_string.split(" ")
    tbs_rs = [float(s.strip("\n")) for s in liste if s.strip() != ""]
    
    file.close()
    return np.array(tbs_rs)

##############################################################################

def get_matching_rttov_and_lbl_files(datetime_mwr,rttov_files, lbl_files):
    rttov_file = None
    lbl_file = None
    for file in rttov_files:
        if str(datetime_mwr)[2:-2] in file:
            rttov_file = file
            # print("worked: ", file, str(datetime_mwr)[2:-2])
    lbl_pattern = str(datetime_mwr).replace("-","").replace(":", "").replace("T", "_")
    for file in lbl_files:
        if lbl_pattern[2:-2] in file:
            lbl_file = file
            # print("worked: ", file, lbl_pattern[2:-2])
    return lbl_file, rttov_file

##############################################################################

def get_tbs_of_all(index, nc_dict,rttov_files, lbl_files, mwr_files_l1,\
            datetime_mwr_plus, datetime_mwr, args):

    lbl_file, rttov_file = get_matching_rttov_and_lbl_files(datetime_mwr,\
        rttov_files, lbl_files)

    if lbl_file==None:
        lbl_tbs = np.array([np.nan]*14)
    else:
        ds_lbl = xr.open_dataset(lbl_file)
        lbl_tbs = np.squeeze(ds_lbl["tb"].values)
        if np.nanmean(lbl_tbs)<0:
            lbl_tbs = np.array([np.nan]*14)
    if rttov_file==None:
        TBs_rttov = np.array([np.nan]*14)
    else:
        TBs_rttov = get_radisonde_tbs_of_file(rttov_file)
    tbs_mwr, quality_flag, mean_cloudflag, frequency, mean_rainfall,\
        std31, elevation =\
        get_mwr_tbs_of_datetime(datetime_mwr_plus, datetime_mwr, args)
    nc_dict["cloud_flag"][index] = mean_cloudflag
    nc_dict["mean_rainfall"][index] = mean_rainfall
    nc_dict["std31"][index] = std31
    nc_dict["elevation"][index] = elevation
    # Let's just get more outputs from MWR also frquecncy!!!

    # print(tbs_mwr)
    nc_dict["TBs_RTTOV-gb"][index,:] = TBs_rttov
    nc_dict["TBs_mwrpy_ret"][index,:] = lbl_tbs
    nc_dict["TBs_HATPRO"][index,:] = tbs_mwr
    if index<2:
        nc_dict["frequency"] = frequency
    return nc_dict

##############################################################################

def dictionary2nc(nc_dict, nc_out_path="~/PhD_data/combined_dataset.nc"):
    # Erzeuge xarray Dataset
    ds = xr.Dataset(
    {
        "t_radiosonde": (("time", "height"), nc_dict["t_radiosonde"]),
        "mr_radiosonde": (("time", "height"), nc_dict["mr_radiosonde"]),
        "p_radiosonde": (("time", "height"), nc_dict["p_radiosonde"]),
        "TBs_RTTOV_gb": (("time", "frequency"), nc_dict["TBs_RTTOV-gb"]),
        "TBs_mwrpy_ret": (("time", "frequency"), nc_dict["TBs_mwrpy_ret"]),
        "TBs_HATPRO": (("time", "frequency"), nc_dict["TBs_HATPRO"]),
        "cloud_flag": (("time"), nc_dict["cloud_flag"]),
        "mean_rainfall": (("time"), nc_dict["mean_rainfall"]),
        "std31": (("time"), nc_dict["std31"]),
        "elevation": (("time"), nc_dict["elevation"]),
    },
    coords={
        "time": np.array(nc_dict["time"]),
        "height": nc_dict["height"],
        "frequency": nc_dict["frequency"],
    },
    attrs={
        "title": "Combined MWR, Radiosonde and Simulated Brightness Temperatures",
        "institution": "University of Colgne",
        "campaign": "Vital I campaign in summer 2024",
        "source": "Script combining radiosonde, MWR, RTTOV, and LBL results",
    }
    )
    # Speichere Dataset als NetCDF
    ds.to_netcdf(nc_out_path)

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    # 0th Get inputs:
    rs_files, mwr_files_l1, mwr_files_l2, rttov_files, lbl_files =\
        read_all_inputs(args)

    # Create preliminary dataset:
    nc_dict = {}
    nc_dict["time"] = []
    max_height = 9500
    nc_dict["height"] = np.linspace(112,max_height,195)
    nc_dict["t_radiosonde"] = np.zeros([len(rs_files), len(nc_dict["height"])])
    nc_dict["mr_radiosonde"] = np.zeros([len(rs_files), len(nc_dict["height"])])
    nc_dict["p_radiosonde"] = np.zeros([len(rs_files), len(nc_dict["height"])])
    nc_dict["TBs_RTTOV-gb"] = np.zeros([len(rs_files), 14])
    nc_dict["TBs_mwrpy_ret"] = np.zeros([len(rs_files), 14])
    nc_dict["TBs_HATPRO"] = np.zeros([len(rs_files), 14])
    nc_dict["frequency"] = np.zeros([14])
    nc_dict["cloud_flag"] = np.zeros([len(rs_files)])
    nc_dict["mean_rainfall"] = np.zeros([len(rs_files)])
    nc_dict["std31"] = np.zeros([len(rs_files)])
    nc_dict["elevation"] = np.zeros([len(rs_files)])
    # LWP (time)

    # 1st First read relevant data from all files in
    # 1.1 Read radiosondes - also to determine date:
    for i, file in enumerate(rs_files):
        print("Radiosonde no: ", i)
        datetime_np = derive_datetime_of_sonde(file)
        datetime_mwr_plus, datetime_mwr = derive_datetime_of_mwr(file, timedelta_min=1)
        nc_dict["time"].append(datetime_np)
        length_value, p_array, t_array, ppmv_array, surf_height_in_km,\
            deg_lat, m_array, heights_in_m = read_radiosonde_csv(file=file)

        # Interpolate mr:
        if max(heights_in_m)<max_height:
            print("Radiosonde excluded for to little max height: ",\
                max(heights_in_m), " m")
            continue
        interp_func = interp1d(heights_in_m, m_array)
        interpolated_mr = interp_func(nc_dict["height"])
        nc_dict["mr_radiosonde"][i,:] = interpolated_mr
        # Interpolate t:
        interp_func = interp1d(heights_in_m,t_array)
        interpolated_t = interp_func(nc_dict["height"])
        nc_dict["t_radiosonde"][i,:] = interpolated_t
        # Interpolate p:
        interp_func = interp1d(heights_in_m,p_array)
        interpolated_p = interp_func(nc_dict["height"])
        nc_dict["p_radiosonde"][i,:] = interpolated_p

        nc_dict = get_tbs_of_all(i, nc_dict,rttov_files, lbl_files, mwr_files_l1,\
            datetime_mwr_plus, datetime_mwr, args)

        # break

dictionary2nc(nc_dict, nc_out_path="~/PhD_data/combined_dataset.nc")

# Data:
# Radiosonde profiles: T, q, lat, lon?
# RTTOV: TBs (time, freqeuncy)
# LBL: TBs (time, freqeuncy)
# MWR: TBs (time, freequency); T, q (time, height), LWP (time); cloud_flag; rain_flag;
# also MWR: 31GHz Std 1 hour mean! for 48 timesteps!; Elevation of MWR at timepoint of RS flight






















