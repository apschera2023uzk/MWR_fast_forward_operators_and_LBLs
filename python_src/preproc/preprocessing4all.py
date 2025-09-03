#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Takes a number of NetCDF files from rs as input
# Creates one NetCDF that contains files in ARMS-gb readable structure!

import argparse
import os
import xarray as xr
import numpy as np
import glob
import sys
sys.path.append('/home/aki/pyrtlib')
from pyrtlib.climatology import AtmosphericProfiles as atmp
from pyrtlib.utils import ppmv2gkg, mr2rh

##############################################################################
# 2nd Used Functions
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
    )
    parser.add_argument(
        "--output1", "-o1",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes/armsgb_vital_1_zenith.nc"),
        help="ARMS-gb summarized outputs!"
    )
    parser.add_argument(
        "--output2", "-o2",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes/armsgb_vital_1_zenith_cropped.nc"),
        help="ARMS-gb summarized CROPPED (!) outputs!"
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

def read_radiosonde_nc_arms(file="/home/aki/PhD_data/Vital_I/radiosondes/20240805_102936.nc", crop=0):
    # Bodenlevel ist bei Index -1 - Umkehr der Profile!
    
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
        
    t_array = ds["Temperature"].isel(Time=inds).values
    length_value = len(t_array )
    p_array = ds["Pressure"].isel(Time=inds).values
    rh = ds["Humidity"].isel(Time=inds).values

    m_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)
    
    height_in_km = ds["Height"].values[0]/1000
    deg_lat = ds["Latitude"].values[0]
    
    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat, m_array 

##############################################################################

def add_clim2profiles(p_array, t_array, ppmv_array, m_array):
    # 2nd Add upper extrapolation of profile by climatology:
    # Kombiniere beide ProfileXXX
    p_threshold = np.nanmin(p_array)
    z, p, d, t, md = atmp.gl_atm(atm=1) # midlatitude summer!
    gkg = ppmv2gkg(md[:, atmp.H2O], atmp.H2O)
    mask = p_threshold>np.array(p)
    
    p_array = np.concatenate([p_array, np.array(p)[mask]])
    t_array = np.concatenate([t_array ,np.array(t)[mask]])
    m_array = np.concatenate([m_array , np.array(gkg)[mask]])
    return p_array[::-1], t_array[::-1], ppmv_array[::-1], m_array[::-1]

##############################################################################

def derive_date_from_nc_name(file):
    string = file.split("/")[-1].split(".")[0]
    datestring = string[:4]+"-"+string[4:6]+"-"+string[6:8]+"T"+\
        string[9:11]+":"+string[11:13]+":"+string[13:15]
    datetime_np = np.datetime64(datestring)
    return datetime_np

##############################################################################

def summarize_many_profiles(pattern=\
                    "/home/aki/PhD_data/Vital_I/radiosondes/202408*_*.nc",\
                    crop=False):
    # Bodenlevel ist bei Index -1 - Umkehr der Profile!
    
    files = glob.glob(pattern)
    n = len(files)
    profile_indices = []
    srf_pressures = np.empty([n])
    srf_temps = np.empty([n])
    srf_wvs = np.empty([n])
    level_pressures = np.empty((137, n))
    level_temperatures = np.empty((137, n))
    level_wvs = np.empty((137, n))
    times = np.empty([n])
    srf_altitude = np.empty([n])
    sza = [0.]*n
    
    for i, file in enumerate(files):
        if crop:
            length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
                m_array = read_radiosonde_nc_arms(file=file, crop=7)
        else:
            length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
                m_array = read_radiosonde_nc_arms(file=file)
        profile_indices.append(i)
        if length_value<50:
            srf_temps[i] = 320
            level_pressures[:,i] = np.array([1000]*137)
            level_temperatures[:,i] = np.array([320]*137)
            level_wvs[:,i] = np.array([8.]*137)
            srf_pressures[i] = 1000
            srf_temps[i] = 320
            srf_wvs[i] = 8
            srf_altitude[i] = 0.5
            continue
        
        p_array, t_array, ppmv_array, m_array = add_clim2profiles(\
                                    p_array, t_array, ppmv_array, m_array)
        # p in hPa as in other inputs!
        # mixing ratio in g/kg
        level_pressures[:,i] = p_array[-137:]
        level_temperatures[:,i] = t_array[-137:]
        level_wvs[:,i] = m_array[-137:]*1000 # convert kg/kg to g/kg
        datetime_np = derive_date_from_nc_name(file)
        
        srf_pressures[i] = p_array[-1]
        srf_temps[i] = t_array[-1]
        srf_wvs[i] = m_array[-1]
        times[i] = datetime_np
        srf_altitude[i] = height_in_km
        
    return profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times

##############################################################################

def write_armsgb_input_nc(profile_indices, level_pressures,
        level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza,\
        times ,outifle="blub.nc"):

    # Ermitteln der Dimensionen
    n_levels, n_profiles = level_pressures.shape

    # Optional: Konvertiere Inputs in float32 falls nötig
    level_pressures = np.array(level_pressures, dtype=np.float32)
    level_temperatures = np.array(level_temperatures, dtype=np.float32)
    level_wvs = np.array(level_wvs, dtype=np.float32)
    srf_pressures = np.array(srf_pressures, dtype=np.float32)
    srf_temps = np.array(srf_temps, dtype=np.float32)
    srf_wvs = np.array(srf_wvs, dtype=np.float32)
    srf_altitude = np.array(srf_altitude, dtype=np.float32)
    sza = np.array(sza, dtype=np.float32)
    profile_indices = np.array(profile_indices, dtype=np.int32)
    level_o3s = np.empty(np.shape(level_wvs))

    # Setze Dummy-Werte für Dimensionsgrößen
    n_times = len(profile_indices)
    n_channels = 14  # Beispielwert
    any_obs = np.empty([14, n_times])
    n_data = 1       # Wird oft für Metadaten genutzt

    ds = xr.Dataset(
        data_vars={
            # Dimensionsgrößen
            # "channel_number":       (("n_Channels",), np.squeeze(np.array(np.arange(n_channels), dtype=np.int32))),
            # "Times_Number":         (("n_Times",), np.squeeze(np.array([times]))),
            # "Levels_Number":        (("n_Levels",), np.squeeze(np.array([range(n_levels)], dtype=np.int32))),
            # "Profiles_Number":      (("n_Profiles",), np.squeeze(np.array([profile_indices], dtype=np.int32)),
            
            ##########
            # Okay dieser Code-Block hat meinen Segfault error gelöst:
            "Times_Number": ("N_Data", np.array([n_times], dtype=np.int32)),
            "Levels_Number": ("N_Data", np.array([n_levels], dtype=np.int32)),
            "Profiles_Number": ("N_Data", np.array([n_profiles], dtype=np.int32)),
            
            
            # Profilebenen
            "Level_Pressure":       (("N_Levels", "N_Profiles"), level_pressures),
            # ValueError: conflicting sizes for dimension 'n_Levels': length 137 on 'Level_Pressure' and length 1 on
            
            "Level_Temperature":    (("N_Levels", "N_Profiles"), level_temperatures),
            "Level_H2O":            (("N_Levels", "N_Profiles"), level_wvs),
            'Level_O3':             (("N_Levels", "N_Profiles"), level_o3s),

            # Oberflächenparameter
            "times":                (("N_Times"), times),
            "Obs_Surface_Pressure": (("N_Times",), srf_pressures[profile_indices]),
            "Obs_Temperature_2M":   (("N_Times",), srf_temps[profile_indices]),
            "Obs_H2O_2M":           (("N_Times",), srf_wvs[profile_indices]),
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
            "GMRZenith":            (("N_Times",), 90-sza), # Elevationswinkel!!!
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

    # Schreibe die NetCDF-Datei
    ds.to_netcdf(outifle, format="NETCDF4_CLASSIC")

    return 0

##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    h_km_vital = 0.092
    # h_km_vital_crop = 0.112
    
    # Uncropped:
    profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times =\
        summarize_many_profiles()
    srf_altitude = np.array([h_km_vital]*len(srf_altitude,))
    write_armsgb_input_nc(profile_indices, level_pressures,\
        level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times ,\
        outifle=args.output1)
        
    # Cropped version:
    profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times =\
        summarize_many_profiles(crop=True)
    write_armsgb_input_nc(profile_indices, level_pressures,\
        level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times ,\
        outifle=args.output2)
