#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import os
import glob
import pandas as pd
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
        # print("crop: ", crop)
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
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    files_in = glob.glob(args.input+args.pattern)
    
    print("***Warning: Remember that radiosonde resolution dz is reduced here!\
        Could that have an effect on results?")
    print("Could I use full rs resolution with RTTOV-gb?\
        It had problems with that... (?)")

    # print(args.input+args.pattern)
    # print(files_in)

    for i, file in enumerate(files_in):
        print(i, file)
        
        # Read inputfile and reduce vertical resolution dz for rttov-gb input:
        length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
            m_array = read_radiosonde_csv(file=file)
        if length_value<100:
            continue

        # Write output file:
        datetime_of_sonde = derive_datetime_of_sonde(file)
        rttovgb_infile = args.output+"prof_"+str(datetime_of_sonde)+".dat"
        write_combined_input_prof_file(t_array, ppmv_array,length_value,\
            p_array,height_in_km=height_in_km, deg_lat=deg_lat,\
            filename=rttovgb_infile)

        print("Pressurevergleich: ", p_array)

        # Write cropped output:
        length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
            m_array = read_radiosonde_csv(file=file, crop=7)
        rttovgb_infile_crop = args.output+"prof_"+str(datetime_of_sonde)+"_crop.dat"
        write_combined_input_prof_file(t_array, ppmv_array,length_value,\
            p_array,height_in_km=height_in_km, deg_lat=deg_lat,\
            filename=rttovgb_infile_crop)






























