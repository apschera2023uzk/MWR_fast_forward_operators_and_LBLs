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

def clausius_clapeyron_liq(temp_celsius):
    # Sättigungsdampfdruck für eine Temperatur in °C
    # es returned in Pa
    # https://en.wikipedia.org/wiki/Latent_heat - enthalpiewerte
    L = 2.5e6
    esl = 610.78 * np.exp(L / 462 * (1/273.15 - 1/(273.15+temp_celsius)))
    return esl
    
##############################################################################

def clausius_clapeyron_ice(temp_celsius):
    # Sättigungsdampfdruck für eine Temperatur in °C
    # es returned in Pa
    L_s = 2.840e6
    esi = 610.78 * np.exp(L_s / 462 * (1/273.15 - 1/(273.15+temp_celsius)))
    return esi

##############################################################################

def rh2mixing_ratio(RH=70, abs_T=273.15+15, p=101325):
    es = clausius_clapeyron_liq(abs_T-273.15)
    e = es * RH / 100
    mue = 0.622
    q = (mue*e) / (p-0.3777*e)
    m = q / (1-q)
    return m

##############################################################################

def read_radiosonde_nc_arms(file=\
        "/home/aki/PhD_data/Vital_I/radiosondes/20240805_102936.nc",\
         crop=0):
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
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan
        
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
    z_array = ds["Height"].values
    
    height_in_km = ds["Height"].values[0]/1000
    deg_lat = ds["Latitude"].values[0]

    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
       m_array, z_array, rh

##############################################################################

def add_clim2profiles(p_array, t_array, ppmv_array, m_array, z_array, rh):
    # 2nd Add upper extrapolation of profile by climatology:
    # Kombiniere beide ProfileXXX
    p_threshold = np.nanmin(p_array)
    z, p, d, t, md = atmp.gl_atm(atm=1) # midlatitude summer!
    gkg = ppmv2gkg(md[:, atmp.H2O], atmp.H2O)
    rhs_clim = mr2rh(p, t, gkg)[0] / 100
    mask = p_threshold>np.array(p)
    
    p_array = np.concatenate([p_array, np.array(p)[mask]])
    t_array = np.concatenate([t_array ,np.array(t)[mask]])
    m_array = np.concatenate([m_array , np.array(gkg)[mask]])
    z_array = np.concatenate([z_array , np.array(z)[mask]])
    rh = np.concatenate([rh , np.array(rhs_clim)[mask]])
    ppmv_array = np.concatenate([ppmv_array , np.array(md[:, atmp.H2O])[mask]])
    return p_array[::-1], t_array[::-1], ppmv_array[::-1], m_array[::-1],\
        z_array[::-1], rh[::-1]

##############################################################################

def derive_date_from_nc_name(file):
    string = file.split("/")[-1].split(".")[0]
    datestring = string[:4]+"-"+string[4:6]+"-"+string[6:8]+"T"+\
        string[9:11]+":"+string[11:13]+":"+string[13:15]
    datetime_np = np.datetime64(datestring)
    return datetime_np

##############################################################################

def derive_cloud_features(p_array, t_array, ppmv_array, m_array,\
        z_array, rh):
    # Follow Nandan et al., 2022
    # min_rh, max_rh, inter_rh:
    below_2km = (92,95,84)
    two2sixkm = (90,93,82)
    six2twelvekm = (88,90,78)
    above_12km = (75,80,70)
    
    #######
    # 1) the conversion of RH with respect to liquid water to RH 
    # with respect to ice at temperatures below 0 ◦ C; 2)
    for i, temp in enumerate(t_array):
        if temp < 273.15:
            rh[i] = rh[i] *  clausius_clapeyron_liq(temp-273.15)/\
                clausius_clapeyron_ice(temp-273.15) # rhl * esl / esi
   
    ##########
    # 2) the base of the lowest moist layer is determined as 
    # the level at which RH exceeds the min-RH corresponding to this level;
    # base_index = 
    for i, temp in enumerate(t_array):
        print(i, temp)
    

    # 3) above the base of the moist layer continuous levels with 
    # RH over the corresponding min-RH are treated as the same layer;


    # LWC(z) kg/kg; and IWC probably different units for PyRTlib...
    # Column water and ice: g m-2
    # cloud base heights and cloud top heights...

    return 0

##############################################################################

def summarize_many_profiles(pattern=\
                    "/home/aki/PhD_data/Vital_I/radiosondes/202408*_*.nc",\
                    crop=False, sza_float =0.):
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
    level_ppmvs = np.empty((137, n))
    level_liq = np.empty((137, n))
    level_z = np.empty((137, n))
    level_rhs = np.empty((137, n))
    times = np.empty([n])
    srf_altitude = np.empty([n])
    sza = [sza_float]*n
    
    for i, file in enumerate(files):
        if crop:
            length_value, p_array, t_array, ppmv_array, height_in_km,\
                deg_lat, m_array, z_array, rh =\
                read_radiosonde_nc_arms(file=file, crop=7)
        else:
            length_value, p_array, t_array, ppmv_array, height_in_km,\
                deg_lat, m_array, z_array, rh =\
                read_radiosonde_nc_arms(file=file)
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
        
        p_array, t_array, ppmv_array, m_array, z_array, rh = add_clim2profiles(\
                                    p_array, t_array, ppmv_array,\
                                    m_array, z_array, rh)
                                    
        derive_cloud_features(p_array, t_array, ppmv_array, m_array, z_array, rh)
        
        #################
        break
        ##################
                                    
        # p in hPa as in other inputs!
        # mixing ratio in g/kg
        level_pressures[:,i] = p_array[-137:]
        level_temperatures[:,i] = t_array[-137:]
        level_wvs[:,i] = m_array[-137:]*1000 # convert kg/kg to g/kg
        datetime_np = derive_date_from_nc_name(file)
        level_ppmvs[:,i] =ppmv_array[-137:]
        level_liq[:,i] =np.array([0]*137)
        level_z[:,i] = z_array[-137:]
        level_rhs[:,i] = rh[-137:]
        
        srf_pressures[i] = p_array[-1]
        srf_temps[i] = t_array[-1]
        srf_wvs[i] = m_array[-1]
        times[i] = datetime_np
        srf_altitude[i] = height_in_km
        
    return profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs

##############################################################################

def write_armsgb_input_nc(profile_indices, level_pressures,
        level_temperatures, level_wvs,level_ppmvs, level_liq,level_z,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,outifle="blub.nc"):

    # Ermitteln der Dimensionen
    n_levels, n_profiles = level_pressures.shape

    # Optional: Konvertiere Inputs in float32 falls nötig
    level_pressures = np.array(level_pressures, dtype=np.float32)
    level_temperatures = np.array(level_temperatures, dtype=np.float32)
    level_wvs = np.array(level_wvs, dtype=np.float32)
    level_ppmvs = np.array(level_ppmvs, dtype=np.float32)
    level_liq = np.array(level_liq, dtype=np.float32)
    level_z = np.array(level_z, dtype=np.float32)
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

    # Add units:
    ds["Level_Pressure"].attrs["units"] = "hPa"
    ds["Level_Temperature"].attrs["units"] = "K"
    ds["Level_H2O"].attrs["units"] = "g/kg"
    ds["Level_ppmvs"].attrs["units"] = "ppmv"
    ###############
    # Füge auch relative humidity als Variable dieser Funktion ein!
    # Edit here:
    ds["Level_Liquid"].attrs["units"] = "None - kg/kg"
    ##################
    ds["Level_z"].attrs["units"] = "m"

    
    # Schreibe die NetCDF-Datei
    ds.to_netcdf(outifle, format="NETCDF4_CLASSIC")

    return 0

##############################################################################

def write_combined_input_prof_file(t_array, ppmv_array,length_value,\
        p_array, liquid_array, height_in_km=0., deg_lat=50.,\
        filename="prof_plev.dat", zenith_angle=0.):
    with open(filename, "w") as file:
        # print("pressure levels: ", length_value)
        for value in p_array:
            file.write(f"{value:8.4f}\n")  # eingerückt, 4 Nachkommastellen
        for value in t_array:
            file.write(f"{value:6.3f}\n")  # eingerückt, 4 Nachkommastellen
        for value in ppmv_array:
            file.write(f"{value:9.4f}\n")  # eingerückt, 4 Nachkommastellen
        for value in liquid_array:
            file.write(f"{0.:12.6E}\n")  # eingerückt, 4 Nachkommastellen
        file.write(f"{t_array[-1]:10.4f}{p_array[-1]:10.2f}\n")
        file.write(f"{height_in_km:6.1f}{deg_lat:6.1f}\n")
        file.write(f"{zenith_angle:6.1f}\n")
        
        return 0
        
##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    h_km_vital = 0.092
    sza_float = 0.
    # h_km_vital_crop = 0.112
    
    # Uncropped:
    profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs =\
        summarize_many_profiles(sza_float=sza_float)
    srf_altitude = np.array([h_km_vital]*len(srf_altitude,))
    write_armsgb_input_nc(profile_indices, level_pressures,\
        level_temperatures, level_wvs, level_ppmvs, level_liq,level_z,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times ,\
        outifle=args.output1)
        
    # Cropped version:
    profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs =\
        summarize_many_profiles(crop=True,sza_float=sza_float)
    write_armsgb_input_nc(profile_indices, level_pressures,\
        level_temperatures, level_wvs,level_ppmvs, level_liq,level_z,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times ,\
        outifle=args.output2)
