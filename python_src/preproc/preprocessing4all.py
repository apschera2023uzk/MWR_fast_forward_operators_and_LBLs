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
from MWR_read_in_module import get_mwr_data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

##############################################################################
# 1.5: Parameter
##############################################################################

elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])
azimuths = np.arange(0.,355.1,5.) # Interpoliere dazwischen!
n_levels=180
min_p = 100
# min_time_diff_thres = 30 # => Leads to 520 remaining sondes 
min_time_diff_thres = 15 

##############################################################################
# 2nd Used Functions
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/PhD_data/TB_preproc_and_proc_results/MWR_rs_FESSTVaLSoclesVital1_all_elevations.nc"),
        help="Where to save preprocessed radiosonde data NetCDF file."
    )
    ##########################
    # Add Input paths here!!!
    ############
    return parser.parse_args()

##############################################################################

def check_moisture_consistency(m_array, rh, ppmv_array, t_array, p_array, tol_m=0.2, tol_rh=3.0, tol_ppmv=100, tag=""):
    
    # Rückrechnungen (hier grobe Näherungen)
    m_from_rh = []
    for i, level_rh in enumerate(rh):        
        m_from_rh.append(rh2mixing_ratio(RH=rh[i], abs_T=t_array[i], p=p_array[i]*100))
    m_from_rh = np.array(m_from_rh)
        
    rh_from_ppmv = []
    for i, level_rh in enumerate(rh):
        rh_from_ppmv.append(ppmv2rh(ppmv_array[i], abs_T=t_array[i], p=p_array[i]*100))    
    rh_from_ppmv = np.array(rh_from_ppmv)
    
    
    # This one leads good results (ppmv and m are consistent):
    m_from_ppmv = []
    for i, level_rh in enumerate(rh):
        m_from_ppmv.append(ppmv2mr(ppmv=ppmv_array[i], abs_T=t_array[i], p=p_array[i]*100))                
    m_from_ppmv = np.array(m_from_ppmv)    

    # Abweichungen
    diff_m_rh = np.nanmax(np.abs(m_array - m_from_rh))
    diff_m_ppmv = np.nanmax(np.abs(m_array - m_from_ppmv))
    diff_rh_ppmv = np.nanmax(np.abs(rh - rh_from_ppmv))  # sehr grobe Umrechnung

    if diff_m_rh > tol_m:
        print(f"Warning ("+tag+"): Mixing ratio vs RH differ by {diff_m_rh:.3f} g/kg (tol {tol_m})")
    if diff_m_ppmv > tol_m:
        print(f"Warning ("+tag+"): Mixing ratio vs ppmv differ by {diff_m_ppmv:.3f} g/kg (tol {tol_m})")
    if diff_rh_ppmv > tol_rh:
        print(f"Warning ("+tag+"): RH vs ppmv differ by {diff_rh_ppmv:.3f} % (tol {tol_rh})")

##############################################################################

def clausius_clapeyron_liq(temp_celsius):
    # Sättigungsdampfdruck für eine Temperatur in °C
    # es returned in Pa
    # https://en.wikipedia.org/wiki/Latent_heat - enthalpiewerte
    L = 2.5e6
    esl = 610.78 * np.exp(L / 462 * (1/273.15 - 1/(273.15+temp_celsius)))
    return esl
    
##############################################################################

def rh2mixing_ratio(RH=70, abs_T=273.15+15, p=101325):
    es = clausius_clapeyron_liq(abs_T-273.15)
    e = es * RH / 100
    mue = 0.622
    q = (mue*e) / (p-0.3777*e)
    m = q / (1-q)
    return m
    
############################################################################

def rh2ppmv(RH=70, abs_T=273.15+15, p=101325):
    es = clausius_clapeyron_liq(abs_T-273.15)
    e = es * RH / 100
    ppmv = 1000000*e / p
    return ppmv
    
############################################################################

def ppmv2rh(ppmv, abs_T=273.15+15, p=101325):
    es = clausius_clapeyron_liq(abs_T-273.15)
    e = ppmv * p / 1e6
    RH = 100 * e / es
    return RH

############################################################################

def mixing_ratio2RH(mixing_ratio=7e-3, abs_T=273.15+15, p=101325):
    # mixing ratio in kg/kg; p in hPa
    e = mixing_ratio * p * 28.96e-3 / (18.02e-3)
    es = clausius_clapeyron_liq(abs_T-273.15)
    RH = e*100 / es 
    return RH

############################################################################

def ppmv2mr(ppmv=15e3, abs_T=273.15+15, p=101325):
    rh = ppmv2rh(ppmv, abs_T=abs_T, p=p)
    mr = rh2mixing_ratio(RH=rh, abs_T=abs_T, p=p)
    return mr    

##############################################################################

def running_mean_from_arrays(inds, z_array, any_array):
    new_array = []
    ind_max = len(inds)
    for i, ind in enumerate(inds):
        if i==0:
            new_array.append(any_array[inds[i]])
        elif i==(ind_max-1):
            new_array.append(any_array[inds[i]])
        else:
            lower = int((ind+inds[i-1])/2)
            upper = int((ind+inds[i+1])/2)
            val = np.nanmean(any_array[lower:upper])
            new_array.append(val)
    return np.array(new_array)

##############################################################################

def interpolation_test_plot(ds, length_value, p_array, t_array,\
       ppmv_array, height_in_km, deg_lat,\
       m_array, z_array, rh, deg_lon,height_var, t_var,\
       p_var, h_var, index):     
    
    if np.all(ds[h_var].values<1.5):
        rhs_before = ds[h_var].values*100
    else: 
        rhs_before = ds[h_var].values
    plt.figure(figsize=(15,15))
    plt.title("Interpolation of Radisonde profile")
    plt.plot(rhs_before,ds[height_var].values, label="Radiosonde profile")
    plt.plot(rh, z_array, label="Interpolated profile for RTE")
    plt.xlabel("Relative Humidity [%]")
    plt.ylabel("height [m]")
    plt.ylim(0,3000)
    plt.legend()
    plt.savefig("/home/aki/PhD_plots/After_oktober_25_1st_paper/\
RS_hum_interpolation/"+str(h_var)+"_"+str(index)+\
    "_interpolation.png")
    
    return 0
       
##############################################################################

def read_radiosonde_nc_arms(file=\
        "/home/aki/PhD_data/Vital_I/radiosondes/20240805_102936.nc",\
         crop=0, min_p=min_p, index=7):
    
    if file==None:
        return 0, [np.nan]*170, [np.nan]*170, [np.nan]*170, \
           float('nan'), float('nan'), [np.nan]*170, [np.nan]*170, \
           [np.nan]*170, float('nan')    
    ds = xr.open_dataset(file)
    
    if "Height" in ds.data_vars:
        height_var = "Height"
        height_in_km = ds[height_var].values[0]/1000
        t_var = "Temperature"
        p_var = "Pressure"
        h_var = "Humidity"
        p_factor = 1.
        deg_lat = ds["Latitude"].values[0]
        deg_lon = ds["Longitude"].values[0]
    elif "zg" in ds.data_vars:
        height_var = "zg"
        height_in_km = ds["zsl_start"].values/1000
        # Geopotential!!!
        # Z \approx z
        t_var = "ta"
        p_var = "pa"
        h_var = "hur"
        deg_lat = ds["lat"].values[0]
        deg_lon = ds["lon"].values[0]
        # g = gravity(deg_lat)
        p_factor = 100.
    elif "zsl" in ds.data_vars:
        height_var = "zsl"
        height_in_km = ds["zsl_start"].values/1000
        # Geopotential!!!
        # Z \approx z
        t_var = "ta"
        p_var = "pa"
        h_var = "hur"
        deg_lat = ds["lat"].values[0]
        deg_lon = ds["lon"].values[0]
        # g = gravity(deg_lat)
        p_factor = 100.
    
    # 3 Fangen Profile auch in Höhge an? 
    max_index = np.nanargmax(ds[height_var].values)
    if ds[p_var].values[max_index]/p_factor<min_p:
        max_index = np.nanargmin(np.abs(ds[p_var].values[:max_index]/p_factor-min_p))    
    index3000 = np.nanargmin(abs(ds[height_var].values[:max_index]-3000))
    
   # Or just find 132 m height:
    if crop > 0:
        crop = np.nanargmin(abs(ds[height_var].values -132))
        
    # AccRate / Height change crop:
    if crop == 0:
        old_h = ds[height_var].values[0]
        for i in range(1000):
            current_h = ds[height_var].values[i]
            if abs(current_h-old_h)<2.:
                if i!=0:
                    crop +=1
            else:
                break
            old_h = current_h
        if crop > 0:
            print("Crop due to same height in NC: ", crop)
        
    if crop > 8:
        print("Unusually high crop value: ",crop)
        
    if max_index<300:
        print("Low max index!!!")
        return 0, [np.nan]*180, [np.nan]*180, [np.nan]*180, \
           float('nan'), float('nan'), [np.nan]*180, [np.nan]*180, \
           [np.nan]*180, float('nan')    
    elif np.nanmax(ds[height_var].values)<10000:
        print("No 10000 m reached!")
        return 0, [np.nan]*180, [np.nan]*180, [np.nan]*180, \
           float('nan'), float('nan'), [np.nan]*180, [np.nan]*180, \
           [np.nan]*180, float('nan')    
        
    # Thinning pattern:
    datapoints_bl = 75
    datapoints_ft = 105
    increment_bl = int(np.ceil((index3000-crop)/datapoints_bl))
    increment_ft = int(np.ceil((max_index-index3000)/datapoints_ft))
    inds = np.r_[crop:index3000:increment_bl, index3000:max_index:increment_ft]
    inds = np.unique(inds)
    
    ####
    # Create running mean:
    if "Height" in ds.data_vars:
        z_array = ds[height_var].isel(Time=inds).values
    elif "zg" in ds.data_vars or "zsl" in ds.data_vars:
        z_array = ds[height_var].isel(time=inds).values
    t_array = running_mean_from_arrays(inds, ds[height_var].values,\
        ds[t_var].values)
    p_array = running_mean_from_arrays(inds, ds[height_var].values,\
        ds[p_var].values/p_factor)
    rh = running_mean_from_arrays(inds, ds[height_var].values,\
        ds[h_var].values)
    if np.all(np.array(rh<=1.5)):
        rh=rh*100
    length_value = len(t_array)
        
    # Humidity unit conversions:
    m_array = []
    ppmv_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
        ppmv_array.append(rh2ppmv(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    m_array = np.array(m_array)    
    ppmv_array = np.array(ppmv_array)
    
    # Addtional Check for z and p singularities:
    for i in range(1, int(len(z_array)/2)):
        if (abs(z_array[i] - z_array[i-1]) > 500) or (abs(p_array[i] - p_array[i-1]) > 50):
            print("Profile excluded due to huge p or z jump between 2 indices")
            print("zs: ", z_array[i],  z_array[i-1])
            print("ps: ", p_array[i],  p_array[i-1])
            return 0, \
       np.full_like(p_array, np.nan), \
       np.full_like(t_array, np.nan), \
       np.full_like(ppmv_array, np.nan), \
       float('nan'), \
       float('nan'), \
       [np.nan]*len(m_array), \
       np.full_like(z_array, np.nan), \
       np.full_like(rh, np.nan), \
       float('nan')

    ######################
    # Still might be needed to show interpolation in Paper:
    '''
    if index%50==0:
        interpolation_test_plot(ds, length_value, p_array, t_array,\
           ppmv_array, height_in_km, deg_lat,\
           m_array, z_array, rh, deg_lon,height_var, t_var, p_var, h_var,\
           index)
    '''     
    ###################### 

    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
       m_array, z_array, rh, deg_lon
       
##############################################################################

def read_radiosonde_txt(file=\
        "/home/aki/PhD_data/Socles/radiosondes/2021072106/'SOUNDING DATA'/20210721060020068041_Profile.txt",\
         crop=0,min_p=min_p):
    # Bodenlevel ist bei Index -1 - Umkehr der Profile!
    
    if file==None:
        return 0, [np.nan]*180, [np.nan]*180, [np.nan]*180, \
           float('nan'), float('nan'), [np.nan]*180, [np.nan]*180, \
           [np.nan]*180, float('nan')    
    df = pd.read_table(file, encoding_errors="ignore", engine='python',\
        skiprows=20,\
        skipfooter=10, header=None , names=["Time", "P", "T", "Hu", "Ws",\
        "Wd", "Long.", "Lat.", "Alt", "Geopot","Rs","Elevation",\
        "Azimuth", "Range"])
    max_index = np.nanargmax(df["Alt"].values)
    if df["P"].values[max_index]<min_p:
        max_index = np.nanargmin(np.abs(df["P"].values[:max_index]-min_p))       
    index3000 = np.nanargmin(abs(df["Alt"].values[:max_index]-3000))
                
    # AccRate / Height change crop:
    if crop == 0:
        old_h = df["Alt"].values[0]
        for i in range(1000):
            current_h = df["Alt"].values[i]
            if abs(current_h-old_h)<2.:
                if i!=0:
                    crop +=1
            else:
                break
            old_h = current_h
        if crop > 0:
            print("Crop due to same height in NC: ", crop)
        
    if crop > 8:
        print("Unusually high crop value: ",crop)
        
    if max_index<300:
        print("Low max index!!!")
        return 0, [np.nan]*180, [np.nan]*180, [np.nan]*180, \
           float('nan'), float('nan'), [np.nan]*180, [np.nan]*180, \
           [np.nan]*180, float('nan')    
    elif np.nanmax(df["Alt"].values)<10000:
        print("No 10000 m reached!")
        return 0, [np.nan]*180, [np.nan]*180, [np.nan]*180, \
           float('nan'), float('nan'), [np.nan]*180, [np.nan]*180, \
           [np.nan]*180, float('nan')    
        
    # Thinning pattern:
    datapoints_bl = 75
    datapoints_ft = 105
    increment_bl = int(np.ceil((index3000-crop)/datapoints_bl))
    increment_ft = int(np.ceil((max_index-index3000)/datapoints_ft))
    inds = np.r_[crop:index3000:increment_bl, index3000:max_index:increment_ft]
    inds = np.unique(inds)
    
    ####
    # Create running mean:
    z_array = df["Alt"].iloc[inds].values
    t_array = running_mean_from_arrays(inds,df["Alt"].values,\
        df["T"].values+ 273.15)
    p_array = running_mean_from_arrays(inds,df["Alt"].values,\
        df["P"].values)
    rh = running_mean_from_arrays(inds,df["Alt"].values,\
        df["Hu"].values)
    length_value = len(t_array )
    if np.all(np.array(rh<=1.5)):
        rh=rh*100

    # Humidity unit conversions:
    m_array = []
    ppmv_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
        ppmv_array.append(rh2ppmv(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    m_array = np.array(m_array)    
    ppmv_array = np.array(ppmv_array)

    # Old conversion to ppmv:
    '''    
    m_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)
    '''    
    
    height_in_km = df["Alt"].values[0]/1000
    deg_lat = df["Lat."].values[0]
    deg_lon = df["Long."].values[0]

    # Addtional Check for z and p singularities:
    for i in range(1, int(len(z_array)/200)):
        if (abs(z_array[i] - z_array[i-1]) > 500) or (abs(p_array[i] - p_array[i-1]) > 50):
            print("Profile excluded due to huge p or z jump between 2 indices")
            print("zs: ", z_array[i],  z_array[i-1])
            print("ps: ", p_array[i],  p_array[i-1])
            return 0, \
       np.full_like(p_array, np.nan), \
       np.full_like(t_array, np.nan), \
       np.full_like(ppmv_array, np.nan), \
       float('nan'), \
       float('nan'), \
       [np.nan]*len(m_array), \
       np.full_like(z_array, np.nan), \
       np.full_like(rh, np.nan), \
       float('nan')    

    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
       m_array, z_array, rh, deg_lon

##############################################################################

def add_clim2profiles(p_array, t_array, ppmv_array, m_array, z_array, rh):
    # 2nd Add upper extrapolation of profile by climatology:
    # Kombiniere beide ProfileXXX
    p_threshold = np.nanmin(p_array)
    if p_threshold<100:
        p_threshold = 100
    z, p, d, t, md = atmp.gl_atm(atm=1) # midlatitude summer!
    gkg = ppmv2gkg(md[:, atmp.H2O], atmp.H2O)
    rhs_clim = mr2rh(p, t, gkg)[0]
    
    if (np.all(np.isnan(p_array)) and np.all(np.isnan(t_array)) and
            np.all(np.isnan(ppmv_array)) and np.all(np.isnan(m_array))):
        mask_clim = np.zeros_like(p, dtype=bool)   # only False
        mask_rs   = np.ones_like(p_array, dtype=bool)  # only True
    else:
        mask_clim = p_threshold>np.array(p)
        mask_rs = np.array(p_array)>p_threshold
    
    p_array = np.concatenate([p_array[mask_rs], np.array(p)[mask_clim]])
    t_array = np.concatenate([t_array[mask_rs] ,np.array(t)[mask_clim]])
    m_array = np.concatenate([m_array[mask_rs], np.array(gkg)[mask_clim]/1000])
    z_array = np.concatenate([z_array[mask_rs], np.array(z*1000)[mask_clim]])
    rh = np.concatenate([rh[mask_rs], np.array(rhs_clim)[mask_clim]])
    # ppmv_array = np.concatenate([ppmv_array[mask_rs], np.array(md[:, atmp.H2O])[mask_clim]])
    ppmv_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        ppmv_array.append(rh2ppmv(RH=rh_lev, abs_T=t_lev, p=p_lev*100))  
    ppmv_array = np.array(ppmv_array)    
    
    return p_array[::-1], t_array[::-1], ppmv_array[::-1], m_array[::-1],\
        z_array[::-1], rh[::-1]

##############################################################################

def derive_date_from_file_name(file):
    if "sups_rao_sonde00" in file or "fval" in file:
        string = file.split("/")[-1].split(".")[0].split("_")[-1]
        datestring = string[:4]+"-"+string[4:6]+"-"+string[6:8]+"T"+\
            string[8:10]+":"+string[10:12]+":"+string[12:14]
    elif ".nc" in file:
        string = file.split("/")[-1].split(".")[0]
        datestring = string[:4]+"-"+string[4:6]+"-"+string[6:8]+"T"+\
            string[9:11]+":"+string[11:13]+":"+string[13:15]
    elif "Profile.txt" in file:
        string = file.split("/")[-1].split(".")[0]
        datestring = string[:4]+"-"+string[4:6]+"-"+string[6:8]+"T"+\
            string[8:10]+":"+string[10:12]+":"+string[12:14]        
    datetime_np = np.datetime64(datestring)
    return datetime_np

##############################################################################

def check_units_physical_realism(p_array, t_array, ppmv_array,\
                m_array, z_array, rh):
    if (np.array(p_array)>1100).any() or (np.array(p_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for p in hPa!!!")
    if abs(p_array[150]-p_array[164])<15:
        print("WARNING: Pressure gradient in lower levels is too little!")    
    if abs(z_array[-1]-z_array[-2])<2.:
        print("WARNING: Low differences between z-values - probably ground data in profile!!!")    
    if (np.array(t_array)>400).any() or (np.array(t_array)<0).any():   
        print("WARNING: Encoutered physically unrealistic value for T in K!!!")
    if np.any(np.array(rh)>110) or np.any(np.array(rh)<0) or\
            (not  np.any(np.array(rh)>1.5)):
        print("WARNING: Encoutered physically unrealistic value for RH in %!!!")
    if (np.array(ppmv_array)>40000).any() or (np.array(ppmv_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for WV in ppmv!!!")
    if (np.array(z_array)>130000).any() or (np.array(z_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for z in m!!!") 
    if (np.array(m_array)>20).any() or (np.array(m_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for mr in g/kg!!!")
    return 0  

##############################################################################

def check_rh_before_and_after(rh, rh_before):
    for i, rh_bef in enumerate(rh_before):
        if abs(rh_bef-rh[-(i+1)])>0.05 and i<160:
            print("Warning RH below index 160 has been altered during processing!")
    return 0

##############################################################################

def summarize_many_profiles(pattern=\
                    "/home/aki/PhD_data/Vital_I/radiosondes/202408*_*.nc",\
                    crop=False, sza_float =0., n_levels=n_levels, mwrs=""):
                         
    # Bodenlevel ist bei Index -1 - Umkehr der Profile!
    h_km_vital = 0.092
    h_km_vital_crop = 0.112   
    files = glob.glob(pattern)
    n = len(files)
    print("Number of files before filtering: ", n)
    profile_indices = []
    srf_pressures = np.empty([n,2])
    srf_temps = np.empty([n,2])
    srf_wvs = np.empty([n,2])
    tbs_dwdhat = np.full((n,10, 72, 14), np.nan)
    tbs_foghat = np.full((n, 10, 72, 14), np.nan)
    tbs_sunhat = np.full((n, 10, 72, 14), np.nan)
    tbs_tophat = np.full((n, 10, 72, 14), np.nan)
    tbs_joyhat = np.full((n, 10, 72, 14), np.nan)
    tbs_hamhat = np.full((n, 10, 72, 14), np.nan)    
    dwd_profiles = np.full((n, 4,n_levels), np.nan)
    fog_profiles= np.full((n, 4,n_levels), np.nan)
    sun_profiles= np.full((n, 4,n_levels), np.nan)
    top_profiles= np.full((n, 4,n_levels), np.nan)
    joy_profiles= np.full((n, 4,n_levels), np.nan)
    ham_profiles= np.full((n, 4,n_levels), np.nan)
    
    lwps_rs = np.full((n, 2), np.nan)
    lwps_dwd = np.full((n), np.nan)
    lwps_fog = np.full((n), np.nan)
    lwps_sun = np.full((n), np.nan)
    lwps_top = np.full((n), np.nan)
    lwps_joy = np.full((n), np.nan)
    lwps_ham = np.full((n), np.nan)
    iwvs_dwd = np.full((n), np.nan)
    iwvs_fog = np.full((n), np.nan)
    iwvs_sun = np.full((n), np.nan)
    iwvs_top = np.full((n), np.nan)
    iwvs_joy = np.full((n), np.nan)
    iwvs_ham = np.full((n), np.nan)

    level_pressures = np.empty((n_levels, n,2))
    level_temperatures = np.empty((n_levels, n,2))
    level_wvs = np.empty((n_levels, n,2))
    level_ppmvs = np.empty((n_levels, n,2))
    level_liq = np.empty((n_levels, n,2))
    level_ice = np.empty((n_levels, n,2))
    level_z = np.empty((n_levels, n,2))
    level_rhs = np.empty((n_levels, n,2))
    srf_altitude = np.empty([n,2])
    
    times = np.empty([n])
    lats = np.full((n), np.nan)
    lons = np.full((n), np.nan)
    sza = [sza_float]*n
    
    for i, file in enumerate(files):
        print(i, file)
        invalid_z = False
        
        profile_indices.append(i)
        datetime_np = derive_date_from_file_name(file)

        ###################
        tbs_dwdhat1, tbs_foghat1,tbs_sunhat1,tbs_tophat1, tbs_joyhat1,\
        tbs_hamhat1, dwd_profiles1,fog_profiles1, sun_profiles1,\
        top_profiles1, joy_profiles1, ham_profiles1, integrals,\
         lat, lon =\
            get_mwr_data(datetime_np, mwrs)
        times[i] = datetime_np
        if crop:
            if ".nc" in file:
                length_value, p_array, t_array, ppmv_array, height_in_km,\
                    deg_lat, m_array, z_array, rh, deg_lon =\
                    read_radiosonde_nc_arms(file=file, crop=7, index=i)
                rh_before = rh
            if length_value<150:
                invalid_z = True
                level_ppmvs[:,i,1] = np.array([np.nan]*n_levels)
                level_liq[:,i,1] = np.array([np.nan]*n_levels)
                level_ice[:,i,1] = np.array([np.nan]*n_levels)
                level_z[:,i,1] = np.array([np.nan]*n_levels)
                level_rhs[:,i,1] = np.array([np.nan]*n_levels)    
                srf_temps[i,1] = np.nan
                level_pressures[:,i, 1] =    np.array([np.nan]*n_levels)
                level_temperatures[:,i,1] = np.array([np.nan]*n_levels)
                level_wvs[:,i,1] = np.array([np.nan]*n_levels)
                srf_pressures[i,1] = np.nan
                srf_temps[i,1] = np.nan
                srf_wvs[i,1] = np.nan
                srf_altitude[i,1] = np.nan
                continue                       
            p_array, t_array, ppmv_array, m_array, z_array, rh = add_clim2profiles(\
                                    p_array, t_array, ppmv_array,\
                                    m_array, z_array, rh)  
            # derive liquid water content:
            lwc_kg_m3, lwc_kg_kg, lwp_kg_m2,iwc_kg_m3, iwc_kg_kg, iwp_kg_m2 =\
                derive_cloud_features(\
                p_array, t_array, ppmv_array, m_array, z_array, rh)   
            #################
            # p in hPa as in other inputs!
            # mixing ratio in g/kg
            lwps_rs[i, 1] = lwp_kg_m2
            level_pressures[:,i,1] = p_array[-n_levels:]
            level_temperatures[:,i,1] = t_array[-n_levels:]
            level_wvs[:,i,1] = m_array[-n_levels:]*1000 # convert kg/kg to g/kg
            level_ppmvs[:,i,1] =ppmv_array[-n_levels:]
            level_liq[:,i,1] = lwc_kg_kg[-n_levels:] # np.array([0]*n_levels)
            level_ice[:,i,1] = np.array([np.nan]*n_levels)
            level_z[:,i,1] = z_array[-n_levels:]
            level_rhs[:,i,1] = rh[-n_levels:]        
            srf_pressures[i,1] = p_array[-1]
            srf_temps[i,1] = t_array[-1]
            srf_wvs[i,1] = m_array[-1]        
            srf_altitude[i,1] = h_km_vital_crop
            tbs_dwdhat[i,:,:,:] = tbs_dwdhat1
            tbs_foghat[i,:,:,:] = tbs_foghat1
            tbs_sunhat[i,:,:,:] = tbs_sunhat1
            tbs_tophat[i,:,:,:] = tbs_tophat1
            tbs_joyhat[i,:,:,:] = tbs_joyhat1
            tbs_hamhat[i,:,:,:] = tbs_hamhat1      
            dwd_profiles[i,:,:] = dwd_profiles1
            fog_profiles[i,:,:] = fog_profiles1
            sun_profiles[i,:,:] = sun_profiles1
            top_profiles[i,:,:] = top_profiles1
            joy_profiles[i,:,:] = joy_profiles1
            ham_profiles[i,:,:] = ham_profiles1   
            
            lwps_dwd[i] = integrals[0]
            lwps_fog[i] = integrals[2]
            lwps_sun[i] = integrals[4]
            lwps_top[i] = integrals[6]
            lwps_joy[i] = integrals[8]
            lwps_ham[i] = integrals[10]
            iwvs_dwd[i] = integrals[1]
            iwvs_fog[i] = integrals[3]
            iwvs_sun[i] = integrals[5]
            iwvs_top[i] = integrals[7]
            iwvs_joy[i] = integrals[9]
            iwvs_ham[i] = integrals[11]
        else:
            #################
            # p in hPa as in other inputs!
            # mixing ratio in g/kg
            invalid_z = True
            level_pressures[:,i,1] = np.array([np.nan]*n_levels)
            level_temperatures[:,i,1] = np.array([np.nan]*n_levels)
            level_wvs[:,i,1] = np.array([np.nan]*n_levels)
            level_ppmvs[:,i,1] =np.array([np.nan]*n_levels)
            level_liq[:,i,1] = np.array([np.nan]*n_levels)
            level_ice[:,i,1] = np.array([np.nan]*n_levels)
            level_z[:,i,1] = np.array([np.nan]*n_levels)
            level_rhs[:,i,1] = np.array([np.nan]*n_levels)       
            srf_pressures[i,1] = np.nan
            srf_temps[i,1] = np.nan
            srf_wvs[i,1] = np.nan       
            srf_altitude[i,1] = np.nan
            ##########        
        if ".nc" in file:
            length_value, p_array, t_array, ppmv_array, height_in_km,\
                    deg_lat, m_array, z_array, rh, deg_lon =\
                    read_radiosonde_nc_arms(file=file, index=i)
            rh_before = rh
        elif "Profile.txt" in file:
            length_value, p_array, t_array, ppmv_array, height_in_km,\
                    deg_lat, m_array, z_array, rh, deg_lon =\
                    read_radiosonde_txt(file=file)
            rh_before = rh
        if length_value<150:
            invalid_z = True
            level_ppmvs[:,i,0] =    np.array([np.nan]*n_levels)
            level_liq[:,i,0] = np.array([np.nan]*n_levels)
            level_ice[:,i,0] = np.array([np.nan]*n_levels)
            level_z[:,i,0] = np.array([np.nan]*n_levels)
            level_rhs[:,i,0] = np.array([np.nan]*n_levels)    
            srf_temps[i,0] = np.nan
            level_pressures[:,i, 0] =    np.array([np.nan]*n_levels)
            level_temperatures[:,i,0] = np.array([np.nan]*n_levels)
            level_wvs[:,i,0] = np.array([np.nan]*n_levels)
            srf_pressures[i,0] = np.nan
            srf_temps[i,0] = np.nan
            srf_wvs[i,0] = np.nan
            srf_altitude[i,0] = np.nan
            continue 
        # Add climatology at top to fill profiles:
        p_array, t_array, ppmv_array, m_array, z_array, rh = add_clim2profiles(\
                                    p_array, t_array, ppmv_array,\
                                    m_array, z_array, rh)                                                       
        # derive liquid water content:
        lwc_kg_m3, lwc_kg_kg, lwp_kg_m2,iwc_kg_m3, iwc_kg_kg, iwp_kg_m2 =\
            derive_cloud_features(\
            p_array, t_array, ppmv_array, m_array, z_array, rh)                     
        #################
        # p in hPa as in other inputs!
        # mixing ratio in g/kg
        lwps_rs[i, 0] = lwp_kg_m2
        lats[i] = deg_lat
        lons[i] = deg_lon
        tbs_dwdhat[i,:,:,:] = tbs_dwdhat1
        tbs_foghat[i,:,:,:] = tbs_foghat1
        tbs_sunhat[i,:,:,:] = tbs_sunhat1
        tbs_tophat[i,:,:,:] = tbs_tophat1
        tbs_joyhat[i,:,:,:] = tbs_joyhat1
        tbs_hamhat[i,:,:,:] = tbs_hamhat1         
        dwd_profiles[i,:,:] = dwd_profiles1
        fog_profiles[i,:,:] = fog_profiles1
        sun_profiles[i,:,:] = sun_profiles1
        top_profiles[i,:,:] = top_profiles1
        joy_profiles[i,:,:] = joy_profiles1
        ham_profiles[i,:,:] = ham_profiles1    
        
        lwps_dwd[i] = integrals[0]
        lwps_fog[i] = integrals[2]
        lwps_sun[i] = integrals[4]
        lwps_top[i] = integrals[6]
        lwps_joy[i] = integrals[8]
        lwps_ham[i] = integrals[10]
        iwvs_dwd[i] = integrals[1]
        iwvs_fog[i] = integrals[3]
        iwvs_sun[i] = integrals[5]
        iwvs_top[i] = integrals[7]
        iwvs_joy[i] = integrals[9]
        iwvs_ham[i] = integrals[11]               
        level_pressures[:,i,0] = p_array[-n_levels:]
        level_temperatures[:,i,0] = t_array[-n_levels:]
        level_wvs[:,i,0] = 1000*m_array[-n_levels:] # convert kg/kg to g/kg
        level_ppmvs[:,i,0] =ppmv_array[-n_levels:]
        level_liq[:,i,0] = lwc_kg_kg[-n_levels:] # np.array([0]*n_levels)
        level_ice[:,i,0] = iwc_kg_kg[-n_levels:]
        level_z[:,i,0] = z_array[-n_levels:]
        level_rhs[:,i,0] = rh[-n_levels:]        
        srf_pressures[i,0] = p_array[-1]
        srf_temps[i,0] = t_array[-1]
        srf_wvs[i,0] = m_array[-1]      
        if crop:
            srf_altitude[i,0] = h_km_vital
        else:  
            srf_altitude[i,0] = height_in_km

        # Check profiles on physical consistency:
        check_units_physical_realism(p_array, t_array, ppmv_array,\
            m_array, z_array, rh)     
        check_moisture_consistency(m_array, rh, ppmv_array,\
            t_array, p_array, tag="after")
        check_rh_before_and_after(rh, rh_before)
          
        #######################    
        #if i==6:
        #   break    
        ############################
    
    return profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza,\
        times, level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham, lwps_rs, lats, lons

##############################################################################

def clean_dataset(ds):
    exclude_times = []
    
    print("************************************************")
    print("Campaign: ", ds["Campaign"].values[5])
    
    for i, timestamp in enumerate(ds["time"].values):
        if np.isnan(ds["Level_z"].values[:,i,0]).any() and\
                np.isnan(ds["Level_z"].values[:,i,1]).any():
            exclude_times.append(timestamp)

        # Check for TBs being only NaNs:
        all_nans = True
        for arr_name in ["TBs_dwdhat", "TBs_foghat", "TBs_sunhat",\
                "TBs_tophat", "TBs_joyhat", "TBs_hamhat"]:
            arr = ds[arr_name].values[i]
            if not np.isnan(arr).all():
                all_nans = False
                break
        if all_nans:
            exclude_times.append(timestamp)
    
    for timestamp in exclude_times:
         ds = ds.sel(time=ds.time != timestamp)
         
    print("Number of files after filtering: ", len(ds["time"].values))
    
    return ds

##############################################################################

def interpolate_azimuths(ds):    
    # Joyhat & Foghat je 30 °:
    
    ds["TBs_foghat"][:,1,:,:] = ds["TBs_foghat"].isel(elevation=1)\
        .interpolate_na(dim="azimuth", method="linear")
    ds["TBs_joyhat"][:,1,:,:] = ds["TBs_joyhat"].isel(elevation=1)\
        .interpolate_na(dim="azimuth", method="linear")
   
    return ds

##############################################################################

def replace_nan_lats_and_lons(ds):

    for i, lat in enumerate(ds["Latitude"].values):
        if np.isnan(lat):
            if (not np.isnan(ds["Latitude"].values[i-1])) and\
                    (ds["Location"].values[i-1]==ds["Location"].values[i]):
                ds["Latitude"].values[i] = ds["Latitude"].values[i-1]
            elif (not np.isnan(ds["Latitude"].values[i+1])) and\
                    (ds["Location"].values[i+1]==ds["Location"].values[i]):
                ds["Latitude"].values[i] = ds["Latitude"].values[i+1]      

    for i, lon in enumerate(ds["Longitude"].values):
        if np.isnan(lon):
            if (not np.isnan(ds["Longitude"].values[i-1])) and\
                    (ds["Location"].values[i-1]==ds["Location"].values[i]):
                ds["Longitude"].values[i] = ds["Longitude"].values[i-1]
            elif (not np.isnan(ds["Longitude"].values[i+1])) and\
                    (ds["Location"].values[i+1]==ds["Location"].values[i]):
                ds["Longitude"].values[i] =ds["Latitude"].values[i+1]         

    return ds 

##############################################################################

def add_attrs_CF_conform(ds):
    # === CF-COMPLIANT ATTRIBUTE ADDITIONS ===

    # --- Coordinate variables ---
    ds["time"].attrs.update({
        "standard_name": "time",
        "long_name": "Time of observation",
        "units": "seconds since 1970-01-01T00:00:00",
        "calendar": "gregorian",
        "axis": "T"
    })

    ds["Latitude"].attrs.update({
        "standard_name": "latitude",
        "long_name": "Latitude of observation",
        "units": "degree",
        "axis": "Y",
        "valid_range": (-90.0, 90.0)
    })

    ds["Longitude"].attrs.update({
        "standard_name": "longitude",
        "long_name": "Longitude of observation",
        "units": "degree",
        "axis": "X",
        "valid_range": (-180.0, 180.0)
    })

    ds["elevation"].attrs.update({
        "standard_name": "sensor_view_elevation_angle",
        "long_name": "Elevation angle of MWR observation",
        "units": "degree",
        "axis": "Z"
    })

    ds["azimuth"].attrs.update({
        "standard_name": "sensor_view_azimuth_angle",
        "long_name": "Azimuth angle of MWR observation",
        "units": "degree"
    })

    ds["Level_z"].attrs.update({
        "standard_name": "altitude",
        "long_name": "Geopotential height above mean sea level",
        "units": "m",
        "positive": "up",
        "axis": "Z"
    })

    # --- Level variables ---
    ds["Level_Pressure"].attrs.update({
        "standard_name": "air_pressure",
        "long_name": "Pressure profile from radiosonde",
        "units": "hPa",
        "_FillValue": np.nan,
        "coordinates": "time Latitude Longitude Level_z",
        "cell_methods": "time: mean"
    })

    ds["Level_Temperature"].attrs.update({
        "standard_name": "air_temperature",
        "long_name": "Temperature profile from radiosonde",
        "units": "K",
        "_FillValue": np.nan,
        "valid_range": (150.0, 330.0),
        "coordinates": "time Latitude Longitude Level_z",
        "cell_methods": "time: mean"
    })

    ds["Level_H2O"].attrs.update({
        "standard_name": "mixing ratio",
        "long_name": "Water vapor mixing ratio",
        "units": "g kg-1",
        "_FillValue": np.nan,
        "coordinates": "time Latitude Longitude Level_z"
    })

    ds["Level_Liquid"].attrs.update({
        "standard_name": "cloud_liquid_water_mixing_ratio",
        "long_name": "Liquid water mixing ratio",
        "units": "kg kg-1",
        "_FillValue": np.nan,
        "coordinates": "time Latitude Longitude Level_z"
    })

    ds["Level_Ice"].attrs.update({
        "standard_name": "cloud_ice_mixing_ratio",
        "long_name": "Ice water mixing ratio",
        "units": "kg kg-1",
        "_FillValue": np.nan,
        "coordinates": "time Latitude Longitude Level_z"
    })

    ds["Level_RH"].attrs.update({
        "standard_name": "relative_humidity",
        "long_name": "Relative humidity",
        "units": "%",
        "_FillValue": np.nan,
        "valid_range": (0.0, 100.0),
        "coordinates": "time Latitude Longitude Level_z"
    })

    # --- Integrated quantities ---
    for var in ["Dwdhat_IWV", "Foghat_IWV", "Sunhat_IWV", "Tophat_IWV", "Joyhat_IWV", "Hamhat_IWV"]:
        ds[var].attrs.update({
            "standard_name": "atmosphere_mass_content_of_water_vapor",
            "long_name": "Integrated water vapor",
            "units": "kg m-2",
            "_FillValue": np.nan,
            "coordinates": "time Latitude Longitude",
            "cell_methods": "altitude: sum"
        })

    for var in ["Dwdhat_LWP", "Foghat_LWP", "Sunhat_LWP", "Tophat_LWP", "Joyhat_LWP", "Hamhat_LWP", "LWP_radiosonde"]:
        ds[var].attrs.update({
            "standard_name": "atmosphere_cloud_liquid_water_content",
            "long_name": "Cloud liquid water path",
            "units": "kg m-2",
            "_FillValue": np.nan,
            "coordinates": "time Latitude Longitude",
            "cell_methods": "altitude: sum"
        })

    # --- Brightness temperatures ---
    for var in ["TBs_dwdhat", "TBs_foghat", "TBs_sunhat", "TBs_tophat", "TBs_joyhat", "TBs_hamhat"]:
        ds[var].attrs.update({
            "standard_name": "brightness_temperature",
            "long_name": "Brightness temperature measured by HATPRO",
            "units": "K",
            "_FillValue": np.nan,
            "coordinates": "time elevation azimuth Latitude Longitude N_Channels",
        })

    # --- Surface variables ---
    ds["Obs_Surface_Pressure"].attrs.update({
        "standard_name": "surface_air_pressure",
        "long_name": "Surface pressure",
        "units": "hPa",
        "_FillValue": np.nan
    })

    ds["Obs_Temperature_2M"].attrs.update({
        "standard_name": "air_temperature",
        "long_name": "2 m air temperature",
        "units": "K",
        "_FillValue": np.nan
    })

    ds["Obs_H2O_2M"].attrs.update({
        "standard_name": "specific_humidity",
        "long_name": "2 m specific humidity",
        "units": "kg kg-1",
        "_FillValue": np.nan
    })

    ds["Surface_Altitude"].attrs.update({
        "standard_name": "surface_altitude",
        "long_name": "Station altitude above mean sea level",
        "units": "m",
        "_FillValue": np.nan
    })

    # --- Metadata and index ---
    ds["Profile_Index"].attrs.update({
        "long_name": "Index number of radiosonde profile",
        "comment": "Unique identifier for each radiosonde profile in the dataset"
    })

    # --- Global attributes for CF compliance ---
    ds.attrs.update({
        "title": "Radiosonde and microwave radiometer brightness temperature dataset from FESSTVaL, SOCLES, and VITAL-I campaigns",
        "summary": "Co-located radiosonde profiles and HATPRO microwave radiometer brightness temperatures from field campaigns at JOYCE and RAO sites.",
        "keywords": "radiosonde, microwave radiometer, brightness temperature, water vapor, liquid water path, FESSTVaL, SOCLES, VITAL-I",
        "Conventions": "CF-1.8",
        "featureType": "profile",
        "institution": "University of Cologne, Institute for Geophysics and Meteorology",
        "source": "Vaisala RS41/GRAW DMF-09 radiosondes; RPG-HATPRO microwave radiometers",
        "references": "FESSTVaL: https://www.cen.uni-hamburg.de/icdc/data/atmosphere/samd-st-datasets/samd-st-fesstval.html; SOCLES: https://gepris.dfg.de/gepris/projekt/430226822; VITAL-I: https://www.herz.uni-bonn.de/wordpress/index.php/vital-campaigns/",
        "history": f"Created {datetime.now(timezone.utc).isoformat()}Z by {os.getlogin()} using xarray",
        "license": "CC BY 4.0",
        "creator_name": "Alexander Pschera",
        "creator_institution": "University of Cologne",
        "contact": "apscher1@uni-koeln.de",
        "geospatial_lat_min": float(np.nanmin(lats)),
        "geospatial_lat_max": float(np.nanmax(lats)),
        "geospatial_lon_min": float(np.nanmin(lons)),
        "geospatial_lon_max": float(np.nanmax(lons)),
        "time_coverage_start": str(np.min(times)),
        "time_coverage_end": str(np.max(times))
    })

    return ds

##############################################################################
    
def print_valid_timebins(ds, varname):
    # Anzahl der time-bins insgesamt:
    n_total = ds[varname].sizes['time']
    # Maske: Für jedes Zeitbin -> Gibt es irgendwo einen nicht-nan Wert?
    valid_times = (~ds[varname].isnull()).any(dim=("elevation", "azimuth", "N_Channels"))
    n_valid = valid_times.sum().item() 
    return n_valid
    
##############################################################################

def produce_dataset(profile_indices, level_pressures,
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat ,tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham,lwps_rs, lats, lons, \
        n_levels=137,\
        campaign="any_camp",\
        location="any_location", elevations=elevations, azimuths=azimuths):

    # Ermitteln der Dimensionen
    n_levels, n_profiles, n_crop = level_pressures.shape
    elevations = np.array(elevations, dtype=float)
    azimuths = np.array(azimuths, dtype=float)
    
    # Setze Dummy-Werte für Dimensionsgrößen
    n_times = len(profile_indices)
    n_channels = 14  # Beispielwert
 
    ds = xr.Dataset(
        data_vars={
            
            # Profilebenen
            "TBs_dwdhat":           (("time","elevation",\
                                        "azimuth","N_Channels"), tbs_dwdhat),
            "TBs_foghat":           (("time","elevation",\
                                     "azimuth","N_Channels"), tbs_foghat),
            "TBs_sunhat":           (("time","elevation",\
                                     "azimuth","N_Channels"), tbs_sunhat),
            "TBs_tophat":           (("time","elevation",\
                                     "azimuth","N_Channels"), tbs_tophat),
            "TBs_joyhat":           (("time","elevation",\
                                     "azimuth","N_Channels"), tbs_joyhat),
            "TBs_hamhat":           (("time","elevation",\
                                     "azimuth","N_Channels"), tbs_hamhat),
                                     
            "Dwdhat_z":           (( "time","N_Levels"), dwd_profiles[:,0,:]), 
            "Dwdhat_ta":           (( "time","N_Levels"), dwd_profiles[:,1,:]), 
            # "Dwdhat_taBL":           (( "time","N_Levels"), dwd_profiles[:,2,:]), 
            "Dwdhat_hua":           (( "time","N_Levels"), dwd_profiles[:,3,:]), 
            "Dwdhat_IWV":           (("time",), iwvs_dwd),
            "Dwdhat_LWP":           (("time",), lwps_dwd),
                         
            "Foghat_z":           (( "time","N_Levels"), fog_profiles[:,0,:]),    
            "Foghat_ta":           (( "time","N_Levels"), fog_profiles[:,1,:]), 
            # "Foghat_taBL":           (( "time","N_Levels"), fog_profiles[:,2,:]), 
            "Foghat_hua":           (( "time","N_Levels"), fog_profiles[:,3,:]),
            "Foghat_IWV":           (("time",), iwvs_fog),
            "Foghat_LWP":           (("time",), lwps_fog),
                        
            "Sunhat_z":           (( "time","N_Levels"), sun_profiles[:,0,:]),    
            "Sunhat_ta":           (( "time","N_Levels"), sun_profiles[:,1,:]), 
            # "Sunhat_taBL":           (( "time","N_Levels"), sun_profiles[:,2,:]), 
            "Sunhat_hua":           (( "time","N_Levels"), sun_profiles[:,3,:]),
            "Sunhat_IWV":           (("time",), iwvs_sun),
            "Sunhat_LWP":           (("time",), lwps_sun),
                        
            "Tophat_z":           (( "time","N_Levels"), top_profiles[:,0,:]),    
            "Tophat_ta":           (( "time","N_Levels"),top_profiles[:,1,:]), 
            # "Tophat_taBL":           (( "time","N_Levels"), top_profiles[:,2,:]), 
            "Tophat_hua":           (( "time","N_Levels"),top_profiles[:,3,:]),     
            "Tophat_IWV":           (("time",), iwvs_top),
            "Tophat_LWP":           (("time",), lwps_top),            
            
            "Joyhat_z":           (( "time","N_Levels"),joy_profiles[:,0,:]),    
            "Joyhat_ta":           (( "time","N_Levels"), joy_profiles[:,1,:]), 
            # "Joyhat_taBL":           (( "time","N_Levels"), joy_profiles[:,2,:]), 
            "Joyhat_hua":           (( "time","N_Levels"), joy_profiles[:,3,:]),  
            "Joyhat_IWV":           (("time",), iwvs_joy),
            "Joyhat_LWP":           (("time",), lwps_joy),               

            "Hamhat_z":           (( "time","N_Levels"), ham_profiles[:,0,:]),    
            "Hamhat_ta":           (( "time","N_Levels"), ham_profiles[:,1,:]), 
            # "Hamhat_taBL":           (( "time","N_Levels"),ham_profiles[:,2,:]), 
            "Hamhat_hua":           (( "time","N_Levels"),ham_profiles[:,3,:]),
            "Hamhat_IWV":           (("time",), iwvs_ham),
            "Hamhat_LWP":           (("time",), lwps_ham),
                        
            "Level_Pressure":       (("N_Levels", "time","Crop"), level_pressures),
            "Level_Temperature":    (("N_Levels", "time","Crop"), level_temperatures),
            "Level_H2O":            (("N_Levels", "time","Crop"), level_wvs),
            "Level_ppmvs":          (("N_Levels", "time","Crop"), level_ppmvs),
            "Level_Liquid":         (("N_Levels", "time","Crop"), level_liq),
            "Level_Ice":            (("N_Levels", "time","Crop"), level_ice),
            "Level_z":              (("N_Levels", "time","Crop"), level_z),
            # 'Level_O3':             (("N_Levels", "time","Crop"), level_o3s),
            "Level_RH":              (("N_Levels", "time","Crop"), level_rhs),

            # Oberflächenparameter
            # "times":                (("N_Times"), times),
            "LWP_radiosonde":       (("time","Crop"), lwps_rs),
            "Obs_Surface_Pressure": (("time","Crop"), srf_pressures),
            "Obs_Temperature_2M":   (("time","Crop"), srf_temps),
            "Obs_H2O_2M":           (("time","Crop"), srf_wvs),
            "Surface_Pressure":     (("time","Crop"), srf_pressures),
            "Temperature_2M":       (("time","Crop"), srf_temps),
            "H2O_2M":               (("time","Crop"), srf_wvs),
            "Surface_Altitude":     (("time","Crop"), srf_altitude),

            # Zusätzliche Metadaten
            "Profile_Index":        (("time",), profile_indices),
            "Campaign":        (("time",), [campaign]*len(times)),
            "Location":        (("time",), [location]*len(times)),
            "Latitude":        (("time",), lats),
            "Longitude":        (("time",), lons ),
        },
        
        # Ort / Kampagne - mit Dimension time!
        
        coords={
            "Crop":     np.array([False, True]),
            "N_Channels": np.arange(n_channels),
            "time":    times,
            "N_Levels":   np.arange(n_levels),
            "elevation":   elevations,
            "azimuth":   azimuths,
        }
    )

    for var in ["TBs_dwdhat", "TBs_foghat", "TBs_sunhat", "TBs_tophat", "TBs_joyhat", "TBs_hamhat"]:
        n = print_valid_timebins(ds, var)
        if n>0:
            print(var+" : "+str(n)+"/"+str(n_times))
    ds = add_attrs_CF_conform(ds)
    ds = clean_dataset(ds)
    ds = interpolate_azimuths(ds)
    ds = replace_nan_lats_and_lons(ds)
    
    return ds
     
##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    sza_float = 0.

    # Read FESSTVaL 1 RAO:
    print("Processing FESSTVaL RAO:")
    profile_indices1, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham, lwps_rs, lats, lons  =\
        summarize_many_profiles(pattern=\
        "/home/aki/PhD_data/FESSTVaL_14GB/radiosondes/RAO/sups_rao_sonde00_l1_any_v00_*.nc",\
             sza_float=sza_float,n_levels=n_levels, mwrs="dwdhat/foghat") 
    ds_fesst_rao = produce_dataset(profile_indices1, level_pressures,\
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat,tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham,lwps_rs, lats, lons,\
        n_levels=n_levels, campaign="FESSTVaL",\
        location="RAO_Lindenberg")

    # Read FESSTVaL 2 UHH:
    print("Processing FESSTVaL UHH:")
    profile_indices2, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham, lwps_rs, lats, lons  =\
        summarize_many_profiles(pattern=\
        "/home/aki/PhD_data/FESSTVaL_14GB/radiosondes/UHH/fval_uhh_sonde00_l1_any_v00_*.nc",\
             sza_float=sza_float,n_levels=n_levels, mwrs="dwdhat/foghat") 
    profile_indices2 = np.array(profile_indices2) + len(profile_indices1)  
    ds_fesst_uhh = produce_dataset(profile_indices2, level_pressures,\
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat,tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham,lwps_rs, lats, lons,\
        n_levels=n_levels, campaign="FESSTVaL",\
        location="RAO_Lindenberg")
            
    # Read FESSTVaL 2 UzK:
    print("Processing FESSTVaL UzK:")
    profile_indices3, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham,lwps_rs, lats, lons =\
        summarize_many_profiles(pattern=\
        "/home/aki/PhD_data/FESSTVaL_14GB/radiosondes/UzK/fval_uzk*.nc",\
             sza_float=sza_float,n_levels=n_levels, mwrs="sunhat") 
    profile_indices3 = np.array(profile_indices3) + len(profile_indices2)
    ds_fesst_uzk = produce_dataset(profile_indices3, level_pressures,\
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat,tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham,lwps_rs, lats, lons,\
        n_levels=n_levels, campaign="FESSTVaL",\
        location="Falkenberg")  
        
    # Read in Socles
    print("Processing Socles:")
    profile_indices4, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham, lwps_rs, lats, lons  =\
        summarize_many_profiles(pattern=\
        "/home/aki/PhD_data/Socles/radiosondes/202*/SOUNDING DATA/*_Profile.txt",\
             sza_float=sza_float,n_levels=n_levels, mwrs="tophat") 
    profile_indices4 = np.array(profile_indices4) + len(profile_indices3) 
    ds_socles =produce_dataset(profile_indices4, level_pressures,\
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat,tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham,lwps_rs, lats, lons,\
        n_levels=n_levels, campaign="Socles",\
        location="JOYCE")

    # Uncropped:
    print("Processing Vital I:")
    profile_indices5, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham, lwps_rs, lats, lons =\
        summarize_many_profiles(sza_float=sza_float,n_levels=n_levels,\
        crop=True, mwrs="hamhat/joyhat") 
    # srf_altitude = np.array([h_km_vital]*len(srf_altitude))
    profile_indices5 = np.array(profile_indices5) + len(profile_indices4)     
    ds_vital_uncrop = produce_dataset(profile_indices5, level_pressures,\
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat,tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham,lwps_rs, lats, lons,\
        n_levels=n_levels, campaign="Vital I",\
        location="JOYCE")
   
    ####
    # Create new dataset and save it:
    ds_list = [ds_fesst_rao, ds_fesst_uhh,ds_fesst_uzk,ds_socles,\
        ds_vital_uncrop]    
    new_ds = xr.concat(ds_list , dim="time")
    print(new_ds)
    new_ds.to_netcdf(args.output, format="NETCDF4_CLASSIC")
        

        

