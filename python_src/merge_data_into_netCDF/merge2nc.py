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
    
    height_in_km = df_resampled["HeightMSL"].values[0]/1000
    deg_lat = df_resampled["Lat"].values[0]
    
    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat, m_array

##############################################################################

def read_all_inputs(args):
    rs_files = glob.glob(args.radiosondes+"csvexport_*.txt")
    mwr_files_l1 = glob.glob(args.mwr_path1+"MWR_1C01_XXX_*.nc")
    mwr_files_l2 = glob.glob(args.mwr_path2+"MWR_single*.nc")
    rttov_files = glob.glob(args.radiosondes+"rttov-gb_*.txt")
    lbl_files = glob.glob(args.mwrpy_ret+"mwrpy_ret_out_rs_*.nc")
    return rs_files, mwr_files_l1, mwr_files_l2, rttov_files, lbl_files

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
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    nc_dict = {}
    nc_dict["time"] = []
    nc_dict["height"] = np.linspace(100,9500,180)
    nc_dict["t_radiosonde"] = np.zeros(len(rs_files), 180)
    nc_dict["mr_radiosonde"] = np.zeros(len(rs_files), 180)

    # 0th Get inputs:
    rs_files, mwr_files_l1, mwr_files_l2, rttov_files, lbl_files =\
        read_all_inputs(args)

    # 1st First read relevant data from all files in
    # 1.1 Read radiosondes - also to determine date:
    for i, file in enumerate(rs_files):
        print("Radiosonde no: ", i)
        datetime_np = derive_datetime_of_sonde(file)
        nc_dict["time"].append(datetime_np)
        length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
             m_array = read_radiosonde_csv(file=file)
        # Interpoliere m_array und t_array und p_array auf Höhenkoordinate mit 180 Werten,
        # dann speichere sie ins  Feld.
        

        print(datetime_np)

# Oder sollte ich MWR interpolieren und lieber eine Höhenkoordinate mit 100-200 Werten nehmen,
# um die Höhere Auflössung der Radiosonden sichtbar zu machen...?
# Jetzt fehlt mir die Höhenkoordinate, um die Länge der rs Felder zu bestimmen - macht sind MWR Höhen zu nehmen...
# Wir haben Höhen der Länge: 43; 200; 6000 und quasi unendlich...
# Auf 100-200 interpolieren erscheint mir nicht das blödeste...




# 2nd decide for which coordinates (grid resolution)
# 3rd prepare dataArray after dataArray to fit into that dataset
# 4th write to NetCDF
    
# Data:
# Radiosonde profiles: T, q, lat, lon?
# RTTOV: TBs (time, freqeuncy)
# LBL: TBs (time, freqeuncy)
# MWR: TBs (time, freequency); T, q (time, height), LWP (time); cloud_flag; rain_flag;
# also MWR: 31GHz Std 1 hour mean! for 48 timesteps!; Elevation of MWR at timepoint of RS flight

























