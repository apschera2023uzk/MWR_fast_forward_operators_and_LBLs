#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import os
import glob
import pandas as pd
import xarray as xr
import numpy as np

##############################################################################
# 2nd Used functions:
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes/"),
        help="Pfad zum Verzeichnis mit den Radiosonden-Rohdaten (default: %(default)s)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes/"),
        help="Pfad zum Verzeichnis für RTTOV-gb Inputdateien (default: %(default)s)"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default=os.path.expanduser("csvexport_*.txt"),
        help="Name convention of radiosonde csv-files"
    )
    parser.add_argument(
        "--pattern2", "-p2",
        type=str,
        default=os.path.expanduser("202408*_*.nc"),
        help="Name convention of radiosonde nc-files"
    )
    return parser.parse_args()

##############################################################################

def clausius_clapeyron(temp_celsius):
    # Sättigungsdampfdruck für eine Temperatur in °C
    # es returned in Pa
    es = 610.78 * np.exp(2.5e6 / 462 * (1/273.15 - 1/(273.15+temp_celsius)))
    return es

##############################################################################

def rh2mixing_ratio(RH=70, abs_T=273.15+15, p=101325):
    es = clausius_clapeyron(abs_T-273.15)
    e = es * RH / 100
    mue = 0.622
    q = (mue*e) / (p-0.3777*e)
    m = q / (1-q)
    return m

##############################################################################

def read_radiosonde_csv(file=None, crop=0):
    dataframe = pd.read_csv(file,skiprows=9, encoding_errors='ignore',\
        header=None,names=["HeightMSL","Temp","Dewp","RH","P","Lat","Lon",\
        "AscRate","Dir","Speed","Elapsed time"])

    # Or just find 132 m height:
    if crop > 0:
        crop = np.nanargmin(abs(dataframe["HeightMSL"].values -132))
        print("Found crop in CSV: ", crop)
        
    # AccRate / Height change crop:
    if crop == 0:
        old_h = dataframe["HeightMSL"].values[0]
        for i in range(100):
            current_h = dataframe["HeightMSL"].values[i]
            current_AscRate = dataframe["AscRate"].values[i]
            if current_AscRate == 0. and current_h==old_h:
                if i!=0:
                    crop +=1
            else:
                break
            old_h = dataframe["HeightMSL"].values[i]
        if crop > 0:
            print("Crop due to height and AscRate in CSV: ", crop)
            
    if crop > 8:
        print("Unusually high crop value: ",crop)

    df_resampled =  pd.concat([dataframe.iloc[crop:100:15],\
        dataframe.iloc[100:500:15],dataframe.iloc[500:1000:20],\
        dataframe.iloc[1000:1500:25],dataframe.iloc[1500:3000:25],\
        dataframe.iloc[3000::50]])
    length_value = len(df_resampled)
    t_array = df_resampled["Temp"].values[::-1] + 273.15
    p_array = df_resampled["P"].values[::-1]

    m_array = []
    for rh_lev, t_lev, p_lev in zip (df_resampled["RH"].values[::-1], t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)
    
    height_in_km = df_resampled["HeightMSL"].values[0]/1000
    deg_lat = df_resampled["Lat"].values[0]

    # print("\n\nAll results CSV!: ")
    # print(length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat, m_array)
    
    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat, m_array

##############################################################################

def write_combined_input_prof_file(t_array, ppmv_array,length_value, p_array,height_in_km=0., deg_lat=50.,\
                                   filename="prof_plev.dat", zenith_angle=0.):
    with open(filename, "w") as file:
        # print("pressure levels: ", length_value)
        for value in p_array:
            file.write(f"{value:8.4f}\n")  # eingerückt, 4 Nachkommastellen
        for value in t_array:
            file.write(f"{value:6.3f}\n")  # eingerückt, 4 Nachkommastellen
        for value in ppmv_array:
            file.write(f"{value:9.4f}\n")  # eingerückt, 4 Nachkommastellen
        for value in range(len(ppmv_array)):
            file.write(f"{0.:12.6E}\n")  # eingerückt, 4 Nachkommastellen
        file.write(f"{t_array[-1]:10.4f}{p_array[-1]:10.2f}\n")
        file.write(f"{height_in_km:6.1f}{deg_lat:6.1f}\n")
        file.write(f"{zenith_angle:6.1f}\n")
        
        return 0

##############################################################################

def derive_datetime_of_sonde(file):
    datestring = (file.split("/")[-1]).split("_")[1]
    timestring = ((file.split("/")[-1]).split("_")[2]).split(".")[0]
    new_timestring = timestring
    datetime_of_sonde = np.datetime64(datestring[:4]+"-"+datestring[4:6]+"-"+datestring[6:8]+"T"+new_timestring[:2]+\
                                      ":"+new_timestring[2:]+":00")
    return datetime_of_sonde

##############################################################################

def read_radiosonde_nc(file=None, crop=0):
    if file==None:
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan
    ds = xr.open_dataset(file)
    max_index = np.nanargmax(ds["Height"].values)
    # Or just find 132 m height:
    if crop > 0:
        crop = np.nanargmin(abs(ds["Height"].values -132))
        print("Found crop in NetCDF: ", crop)
        # print("crop: ", crop)
        
    # AccRate / Height change crop:
    if crop == 0:
        old_h = ds["Height"].values[0]
        for i in range(100):
            current_h = ds["Height"].values[i]
            if abs(current_h-old_h)<0.3:
                if i!=0:
                    crop +=1
            else:
                break
            old_h = current_h
        if crop > 0:
            print("Crop due to same height in NC: ", crop)
        
    if crop > 8:
        print("Unusually high crop value: ",crop)
    if max_index<150:
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan
        
    # Thinning pattern:
    inds = np.r_[crop:100:15,
             100:500:15,
             500:1000:20,
             1000:1500:25,
             1500:2500:25,
             2500:max_index:50]
    inds = np.unique(inds)
        
    t_array = ds["Temperature"].isel(Time=inds).values[::-1]
    length_value = len(t_array )
    p_array = ds["Pressure"].isel(Time=inds).values[::-1]
    rh = ds["Humidity"].isel(Time=inds).values[::-1]

    m_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)
    
    height_in_km = ds["Height"].values[0]/1000
    deg_lat = ds["Latitude"].values[0]
    
    # print("\n\nAll results: ")
    # print(length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat, m_array)
    
    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat, m_array
    
##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    files_in = glob.glob(args.input+args.pattern)
    files_in2 = glob.glob(args.input+args.pattern2)
    h_km_vital = 0.092
    h_km_vital_crop = 0.112
    
    print("******* Started preprcoessing for RTTOV-gb***************")
    print("***Warning: Remember that radiosonde resolution dz is reduced here!\
        Could that have an effect on results?")
    print("Could I use full rs resolution with RTTOV-gb?\
        It had problems with that... (?)")

    for i, (file, file2) in enumerate(zip(files_in, files_in2)):
        print(i, file, file2)
        
        # Read inputfile and reduce vertical resolution dz for rttov-gb input:
        length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
            m_array = read_radiosonde_csv(file=file)
        if length_value<100:
            print("Length value: ", length_value)
            continue

        # Write output file:
        # 1st from CSV
        datetime_of_sonde = derive_datetime_of_sonde(file)
        rttovgb_infile = args.output+"prof_"+str(datetime_of_sonde)+".dat"
        write_combined_input_prof_file(t_array, ppmv_array,length_value,\
            p_array,height_in_km=h_km_vital, deg_lat=deg_lat,\
            filename=rttovgb_infile)
            
        # 2nd from nc:
        length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
            m_array = read_radiosonde_nc(file=file2)
        if length_value<100:
            print("Length value: ", length_value)
            continue
        rttovgb_infile = args.output+"prof_"+str(datetime_of_sonde)+"4nc.dat"
        write_combined_input_prof_file(t_array, ppmv_array,length_value,\
            p_array,height_in_km=h_km_vital, deg_lat=deg_lat,\
            filename=rttovgb_infile)

        # Write cropped output:        
        # 1st from CSV
        length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
            m_array = read_radiosonde_csv(file=file, crop=7)
        rttovgb_infile_crop = args.output+"prof_"+str(datetime_of_sonde)+"_crop.dat"
        write_combined_input_prof_file(t_array, ppmv_array,length_value,\
            p_array,height_in_km=h_km_vital_crop, deg_lat=deg_lat,\
            filename=rttovgb_infile_crop)

        # 2nd from nc
        length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
            m_array = read_radiosonde_nc(file=file2, crop=7)
        rttovgb_infile_crop = args.output+"prof_"+str(datetime_of_sonde)+"4nc_crop.dat"
        write_combined_input_prof_file(t_array, ppmv_array,length_value,\
            p_array,height_in_km=h_km_vital_crop, deg_lat=deg_lat,\
            filename=rttovgb_infile_crop)    


    print("******* Finished preprcoessing for RTTOV-gb***************")
























