#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Takes a number of NetCDF files from rs as input
# Creates one NetCDF that contains files in ARMS-gb readable structure!

# Important limitations:
# => MWR and rs are matched if there is maximum a 30 minute difference between both measurement times.
# => LWC and LWP are derived via Nandan et al. 2022; Mixed phase clouds are assumed liquid; IWC is assumed same as LWC.
# => 

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
from datetime import datetime

##############################################################################
# 1.5: Parameter
##############################################################################

elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])
azimuths = np.arange(0.,355.1,5.) # Interpoliere dazwischen!
n_levels=180

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
'''
def gravity(latitude_deg):
    phi = math.radians(latitude_deg)
    g = 9.780327 * (1 + 0.0053024 * math.sin(phi)**2 - 0.0000058 * math.sin(2*phi)**2)
    return g  # in m/s^2
'''
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

def read_radiosonde_nc_arms(file=\
        "/home/aki/PhD_data/Vital_I/radiosondes/20240805_102936.nc",\
         crop=0):
    
    if file==None:
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
    ds = xr.open_dataset(file)
    #######################
    if "Height" in ds.data_vars:
        height_var = "Height"
        height_in_km = ds[height_var].values[0]/1000
        t_var = "Temperature"
        p_var = "Pressure"
        h_var = "Humidity"
        p_factor = 1.
        deg_lat = ds["Latitude"].values[0]
    elif "zg" in ds.data_vars:
        height_var = "zg"
        height_in_km = ds["zsl_start"].values/1000
        # Geopotential!!!
        # Z \approx z
        t_var = "ta"
        p_var = "pa"
        h_var = "hur"
        deg_lat = ds["lat"].values[0]
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
        # g = gravity(deg_lat)
        p_factor = 100.

    #############
    # g0 = 9.80665
    # print("g value: ", g)
    # print("g_dev", g/g0)    
    #############
    
    # 3 Fangen Profile auch in Höhge an? 
    max_index = np.nanargmax(ds[height_var].values)
    index3000 = np.nanargmin(abs(ds[height_var].values[:max_index]-3000))
    
   # Or just find 132 m height:
    if crop > 0:
        crop = np.nanargmin(abs(ds[height_var].values -132))
        
    # AccRate / Height change crop:
    if crop == 0:
        old_h = ds[height_var].values[0]
        for i in range(100):
            current_h = ds[height_var].values[i]
            if abs(current_h-old_h)<0.3:
                if i!=0:
                    crop +=1
            else:
                break
            old_h = current_h
        if crop > 0:
            pass
            # print("Crop due to same height in NC: ", crop)
        
    if crop > 8:
        print("Unusually high crop value: ",crop)
        
    if max_index<1000:
        print("Low max index!!!")
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
    elif np.nanmax(ds[height_var].values)<10000:
        print("No 3000 m reached!")
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
        
    # Thinning pattern:
    datapoints_bl = 75
    datapoints_ft = 100
    increment_bl = int(np.ceil(index3000/datapoints_bl))
    increment_ft = int(np.ceil((max_index-index3000)/datapoints_ft))
    inds = np.r_[crop:index3000:increment_bl, index3000:max_index:increment_ft]
    inds = np.unique(inds)
    # print("dz: (BL)", 3000/datapoints_bl)
    # print("dz FT:", (ds["Height"].values[max_index]- 3000)/datapoints_ft)
    
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
    if np.array(rh<=1.5).all():
        rh=rh*100
    length_value = len(t_array)
        
    m_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)

    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat,\
       m_array, z_array, rh
       
##############################################################################

def read_radiosonde_txt(file=\
        "/home/aki/PhD_data/Socles/radiosondes/2021072106/'SOUNDING DATA'/20210721060020068041_Profile.txt",\
         crop=0):
    # Bodenlevel ist bei Index -1 - Umkehr der Profile!
    
    if file==None:
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan      
    df = pd.read_table(file, encoding_errors="ignore", engine='python',\
        skiprows=20,\
        skipfooter=10, header=None , names=["Time", "P", "T", "Hu", "Ws",\
        "Wd", "Long.", "Lat.", "Alt", "Geopot","Rs","Elevation",\
        "Azimuth", "Range"])
    max_index = np.nanargmax(df["Alt"].values)
    index3000 = np.nanargmin(abs(df["Alt"].values[:max_index]-3000))
                
    # AccRate / Height change crop:
    if crop == 0:
        old_h = df["Alt"].values[0]
        for i in range(100):
            current_h = df["Alt"].values[i]
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
        
    if max_index<1000:
        print("Low max index!!!")
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
    elif np.nanmax(df["Alt"].values)<10000:
        print("No 3000 value!")
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
        
    # Thinning pattern:
    datapoints_bl = 75
    datapoints_ft = 100
    increment_bl = int(np.ceil(index3000/datapoints_bl))
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

    m_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)
    
    height_in_km = df["Alt"].values[0]/1000
    deg_lat = df["Lat."].values[0]

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
    m_array = np.concatenate([m_array , np.array(gkg)[mask]/1000])
    z_array = np.concatenate([z_array , np.array(z*1000)[mask]])
    rh = np.concatenate([rh , np.array(rhs_clim)[mask]])
    # ppmv_array = np.concatenate([ppmv_array , np.array(md[:, atmp.H2O])[mask]])
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)
    
    # print("\n\nm_array: ",m_array)
    # print("\n\nppmv_array: ",ppmv_array)
    
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
    
def calc_lwc(tops, bases, p_array, t_array, ppmv_array, m_array,\
        z_array, rh, cloud_bools):
    ####
    # Chakraborty & Maitra 2011  / Nandan et al. 2022:
    cp = 1003.5 # J /kg / K == 1.003 J / g / K
    L = 334944 # J / kg == 80 cal / gm
    R_L = 287.06 # J / kg / K
    gamma_d = 9.76e-3 # K/m
    gamma_s = 6.5e-3 # K/m    
    lwc_kg_m3 = np.array([0.]*len(t_array))
    lwc_kg_kg = np.array([0.]*len(t_array))
    iwc_kg_m3 = np.array([0.]*len(t_array))
    iwc_kg_kg = np.array([0.]*len(t_array))  
    if len(tops) != len(bases):
        print("WARNING: base and top number are deviating!")    
        print("bases: " , bases)
        print("tops: ", tops) 
    for i, (base, top) in enumerate(zip(bases,tops)):
        z_index_top = np.nanargmin(abs(z_array-top))
        z_index_base = np.nanargmin(abs(z_array-base))
        if t_array[z_index_base]>273.15 and t_array[z_index_top]>273.15:
            # print("Water cloud")  
            for j in range(z_index_top,z_index_base):
                rho = p_array[j]*100 / R_L / t_array[j] # Ergebnis realistisch kg m-3
                dz = z_array[j-1] - z_array[j] # ergab soweit auch Sinn..
                lwc_ad = rho * cp / L * (gamma_d - gamma_s) * dz
                dh = z_array[j] - base # abs entfernt
                '''
                print("z_array[j]: ", z_array[j])
                print("base: ", base)
                print("dh: ", dh)           
                '''     
                lwc = lwc_ad * (1.239 - 0.145*np.log(dh)) # kg m-3
                if lwc<0:
                    lwc = 0
                lwc_kg_m3[j] = lwc          
                lwc_kg_kg[j] = lwc / rho
        elif t_array[z_index_base]<233.15 and t_array[z_index_top]<233.15:
            pass
            # print("Ice cloud")
            for j in range(z_index_top,z_index_base):
                rho = p_array[j]*100 / R_L / t_array[j] # Ergebnis realistisch kg m-3
                dz = z_array[j-1] - z_array[j] # ergab soweit auch Sinn..
                iwc_ad = rho * cp / L * (gamma_d - gamma_s) * dz
                dh = z_array[j] - base
                '''
                print("z_array[j]: ", z_array[j])
                print("base: ", base)
                print("dh: ", dh)
                '''
                iwc = iwc_ad * (1.239 - 0.145*np.log(dh)) # kg m-3
                if iwc<0:
                    iwc = 0
                iwc_kg_m3[j] = iwc          
                iwc_kg_kg[j] = iwc / rho  
        elif t_array[z_index_base]>233.15 and t_array[z_index_top]<273.15:
            # print("Mixed phase cloud")       
            for j in range(z_index_top,z_index_base):
                rho = p_array[j]*100 / R_L / t_array[j] # Ergebnis realistisch kg m-3
                dz = z_array[j-1] - z_array[j] # ergab soweit auch Sinn..
                lwc_ad = rho * cp / L * (gamma_d - gamma_s) * dz
                dh = z_array[j] - base
                '''
                print("z_array[j]: ", z_array[j])
                print("base: ", base)
                print("dh: ", dh)        
                '''        
                lwc = lwc_ad * (1.239 - 0.145*np.log(dh)) # kg m-3
                if lwc<0:
                    lwc = 0
                lwc_kg_m3[j] = lwc          
                lwc_kg_kg[j] = lwc / rho             
        else:
            print("Phase determination error!")

    ##################
    # Dafür wäre IWC profil noch nice...
    # Ich brauche IWP
    # Was macht man mit mixed phase???
    ###################
    
    lwp_kg_m2 = np.abs(np.sum(lwc_kg_m3 * np.gradient(z_array)))  # [kg/m²]
    iwp_kg_m2 = np.abs(np.sum(iwc_kg_m3 * np.gradient(z_array)))  # [kg/m²]
        
    return lwc_kg_m3, lwc_kg_kg, lwp_kg_m2, iwc_kg_m3, iwc_kg_kg, iwp_kg_m2

##############################################################################

def derive_cloud_features(p_array, t_array, ppmv_array, m_array,\
        z_array, rh):
    # Follow Nandan et al., 2022
    # min_rh, max_rh, inter_rh:
    below_2km = (92,95,84)
    two2sixkm = (90,93,82)
    six2twelvekm = (88,90,78)
    above_12km = (75,80,70)
    bases = []
    tops = []
    
    #######
    # 1) the conversion of RH with respect to liquid water to RH 
    # with respect to ice at temperatures below 0 ◦ C; 2)
    for i, temp in enumerate(t_array):
        if temp < 273.15:
            rh[i] = rh[i] *  clausius_clapeyron_liq(temp-273.15)/\
                clausius_clapeyron_ice(temp-273.15) # rhl * esl / esi
   
    ##########
    # 2-4: Find RHs above RH_min (preliminary layers):
    cloud_bools = np.array([False]*len(z_array))
    in_cloud = False
    for i, (temp, z) in enumerate(zip(t_array, z_array)):
        if z<2000:
            if rh[i]>below_2km[0]:
                cloud_bools[i] = True
                if not in_cloud:
                    tops.append(z)
                in_cloud = True
            else:
                if in_cloud:
                    bases.append(z)
                in_cloud = False
        elif 2000<z<6000:
            if rh[i]>two2sixkm[0]:
                cloud_bools[i] = True
                if not in_cloud:
                    tops.append(z)
                in_cloud = True
            else:
                if in_cloud:
                    bases.append(z)
                in_cloud = False
        elif 6000<z<12000:
            if rh[i]>six2twelvekm[0]:
                cloud_bools[i] = True
                if not in_cloud:
                    tops.append(z)
                in_cloud = True
            else:
                if in_cloud:
                    bases.append(z)
                in_cloud = False
        elif 12000<z:
            if rh[i]>above_12km[0]:
                cloud_bools[i] = True
                if not in_cloud:
                    tops.append(z)
                in_cloud = True
            else:
                if in_cloud:
                    bases.append(z)
                in_cloud = False         
    # print("after 4  tops: ", tops)    
    # print("after 4  bases: ", bases)
    ####################################
    if len(tops) == len(bases)+1:
        bases.append(z_array[-1])     
    elif len(tops) != len(bases):
        print("base and top number are deviating!")    
        print("bases: " , bases)
        print("tops: ", tops)        
    elif len(tops)>=1 and len(bases)>=1:
        if (0>(np.array(tops)-np.array(bases))).any():
            print("Warning! Top lower than cloud base... Why?")
            print("z_array:", z_array)
            print("rh: ", rh)     
    #################################
            
    ####
    # 5) Remove cloudbases below 500 m if thickness < 400 m:
    # New version:
    valid_indices = [i for i, (base, top) in enumerate(zip(bases, tops))\
        if not (base < 500 and abs(base-top) < 400)]
    new_bases = [bases[i] for i in valid_indices]
    new_tops = [tops[i] for i in valid_indices]
    for i in range(len(bases)):
        if i not in valid_indices:
            z_index_top = np.nanargmin(abs(z_array-tops[i]))
            z_index_base = np.nanargmin(abs(z_array-bases[i]))
            cloud_bools[z_index_top:z_index_base] = False
    # print("after 5  tops: ", new_tops)    
    # print("after 5  bases: ",new_bases)   

    
    ###
    # 6 ) RH_max reached within cloud layer? => discard else!
    to_remove_base = []
    to_remove_top = []      
    tops = new_tops
    bases = new_bases
    for i, (base, top) in enumerate(zip(bases,tops)):
        if base<2000:
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            if not np.any(below_2km[1] < rh[z_index_top:z_index_base]):
                cloud_bools[z_index_top:z_index_base] = False
                to_remove_base.append(i)
                to_remove_top.append(i)
        elif 2000<base<6000:
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            if not np.any(two2sixkm[1] < rh[z_index_top:z_index_base]):
                cloud_bools[z_index_top:z_index_base] = False
                to_remove_base.append(i)
                to_remove_top.append(i)
        elif 6000<base<12000:
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            if not np.any(six2twelvekm[1] < rh[z_index_top:z_index_base]):
                cloud_bools[z_index_top:z_index_base] = False
                to_remove_base.append(i)
                to_remove_top.append(i)      
        elif 12000<base:
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            if not np.any(above_12km[1]  < rh[z_index_top:z_index_base]):
                cloud_bools[z_index_top:z_index_base] = False
                to_remove_base.append(i)
                to_remove_top.append(i)      
    for i in sorted(to_remove_base, reverse=True):
        bases.pop(i)
    for i in sorted(to_remove_top, reverse=True):
        tops.pop(i)
    # print("after 6  tops: ", tops)    
    # print("after 6  bases: ",bases)   
        
    ###
    # 7) Connect layers, with a gap of less than 300 m:
    to_remove_base = []
    to_remove_top = []       
    for i, (base, top) in enumerate(zip(bases,tops)):      
        if base<2000:
            rh_inter = below_2km[2]
        elif 2000<base<6000:
            rh_inter = two2sixkm[2]
        elif 6000<base<12000:
            rh_inter = six2twelvekm[2]
        elif 12000<base:
            rh_inter = above_12km[2]           
        if i!=0:
           z_index_base = np.nanargmin(abs(z_array-bases[i-1]))
           z_index_top = np.nanargmin(abs(z_array-top))
           if 1==abs(z_index_base-z_index_top):
               if rh[z_index_base]>rh_inter:  
                   cloud_bools[z_index_base] = True
                   to_remove_base.append(i-1)
                   to_remove_top.append(i)                   
           elif abs(bases[i-1]-top)<300 or\
                   np.nanmin(rh[z_index_base:z_index_top-1])>rh_inter:    
               cloud_bools[z_index_base:z_index_top-1] = True
               to_remove_base.append(i-1)
               to_remove_top.append(i)               
    for i in sorted(to_remove_base, reverse=True):
        bases.pop(i)
    for i in sorted(to_remove_top, reverse=True):
        tops.pop(i)
    # print("after 7  tops: ", tops)    
    # print("after 7  bases: ",bases)   
            
    # step 8: 
    to_remove_base = []
    to_remove_top = []   
    for i, (base, top) in enumerate(zip(bases,tops)):               
        ####
        # 8) Cloud thickness below 100 m:
        if abs(base-top)<100:
            to_remove_base.append(i)
            to_remove_top.append(i)
            # new_bases.pop(i)
            # new_tops.pop(i)
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))     
            cloud_bools[z_index_top:z_index_base] = False
    for i in sorted(to_remove_base, reverse=True):
        bases.pop(i)
    for i in sorted(to_remove_top, reverse=True):
        tops.pop(i)
    # print("after 8  tops: ", tops)    
    # print("after 8  bases: ",bases)           

    ####
    # Lets get LWC and IWC:
    lwc_kg_m3, lwc_kg_kg, lwp_kg_m2,iwc_kg_m3, iwc_kg_kg, iwp_kg_m2 =\
        calc_lwc(tops, bases, p_array, t_array, ppmv_array, m_array,\
        z_array, rh, cloud_bools)

    # LWC(z) kg/kg; and IWC probably different units for PyRTlib...
    # Column water and ice: g m-2
    

    return lwc_kg_m3, lwc_kg_kg, lwp_kg_m2,iwc_kg_m3, iwc_kg_kg, iwp_kg_m2

##############################################################################

def nearest_ele4elevation(ele_values, azi_values, ele_times,\
        target_elevation,target_azi,datetime_np):
    # Boolean Maske für alle Stellen, an denen "ele" exakt target_elevation ist
    match_mask = (abs(ele_values-target_elevation)<0.05)   
    if target_azi =="ANY":
        match_mask2 = [True]*len(ele_values)
    else:
        match_mask2 = (abs(azi_values-target_azi)<0.05)
    final_mask = match_mask & match_mask2
    ####################################
    # print("len(match_mask):", len(match_mask)) 
    # print("len(match_mask2):", len(match_mask2))     
    # print("len(ele_values):", len(ele_values))         
    # print("len(ele_times):", len(ele_times))     
    
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
        if minutes_diff>30:
            print("Excluded due to huge time difference from scan!!")
            nearest_idx = None
        
        # Deterime nearest index:
        candidate_indices = np.where(final_mask)[0]
        nearest_idx = candidate_indices[min_idx]
        
        # Elevation check:
        nearest_value = ele_values[nearest_idx]
        nearest_value2 = azi_values[nearest_idx]
        if target_azi =="ANY":
            if abs(nearest_value-target_elevation)>0.05:
                print("WARNING: Azimuth or Elevation does not agree, as expected!")
                print("Ele/Azi tagret values: ",target_elevation, target_azi)
                print("Ele/Azi found values: ",nearest_value,\
                    nearest_value2)            
        elif abs(nearest_value-target_elevation)>0.05 or\
                 (abs(nearest_value2-target_azi)>0.05):
             print("WARNING: Azimuth or Elevation does not agree, as expected!")
             print("Ele/Azi tagret values: ",target_elevation, target_azi)
             print("Ele/Azi found values: ",nearest_value, nearest_value2)
    return nearest_idx

##############################################################################
    
def derive_elevation_index(ds_bl, elevation):
    # print("Input ele: ",elevation)
    for index, ele in enumerate(ds_bl["ele"].values):
        if abs(ele-elevation)<0.05:
            # print("Output ele: ", ele)
            # print("Output Index: ", index)
            return index
    return None

##############################################################################

def get_tbs_from_l1(l1_files, datetime_np, elevations=elevations,\
        azimuths=azimuths): 
    tbs = np.full((10, 72, 14), np.nan)
    lat = np.nan; lon = np.nan
    
    for file in l1_files:
        
        if "BL" in file:
            ds_bl = xr.open_dataset(file)
            # print("Opened BL for file: ", file)     
            for i,elevation in enumerate(elevations):
                ele_index = derive_elevation_index(ds_bl, elevation)
                if ele_index==None:
                    continue
                else:
                    time_diffs = np.abs(ds_bl["time"].values - datetime_np)
                    min_idx = time_diffs.argmin()
                    tbs[ele_index,0, :] = ds_bl["tb"].values[min_idx,ele_index,:]
                    # print("TBs found BL: ", ds_bl["tb"].values[min_idx,ele_index,:])
        elif "MWR_1C01" in file:
            # print("Found MWR_1C_File!", file)
            ds_c1 = xr.open_dataset(file)
            for i,elevation in enumerate(elevations):
                for j,azi in enumerate(azimuths):
                    time_idx = nearest_ele4elevation(ds_c1["elevation_angle"].values,\
                        ds_c1["azimuth_angle"].values,\
                        ds_c1["time"].values, elevation,azi, datetime_np)
                    if time_idx==None:
                        pass
                    else:
                        tbs[i,j, :] = ds_c1["tb"].values[time_idx,:]
            if "latitude" in ds_c1.data_vars:
                lat = ds_c1["latitude"].values[0]
                lon = ds_c1["longitude"].values[0]
            elif "lat" in ds_c1.data_vars:
                lat = ds_c1["lat"].values
                lon = ds_c1["lon"].values           
        else: 
            ds_mwr = xr.open_dataset(file)
            for i,elevation in enumerate(elevations):
                # 
                for j,azi in enumerate(azimuths):
                    time_idx = nearest_ele4elevation(ds_mwr["ele"].values,\
                        ds_mwr["azi"].values,\
                        ds_mwr["time"].values, elevation,azi, datetime_np)
                    if time_idx==None:
                        pass
                    else:
                        tbs[i,j, :] = ds_mwr["tb"].values[time_idx,:]
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

def get_profs_from_l2(l2_files, datetime_np, n_levels = n_levels):
    data = np.full((4,n_levels), np.nan)
    lwp, iwv = np.nan, np.nan

    for file in l2_files:
        if "single" in file:
            ds = xr.open_dataset(file)
            # time_diffs = np.abs(ds["time"].values - datetime_np)
            # min_idx = time_diffs.argmin()
            min_idx = nearest_ele4elevation(ds["elevation_angle"].values,\
                ds["azimuth_angle"].values,\
                ds["time"].values, 90., "ANY" , datetime_np)
            if min_idx is None:
                data[0,:] = [np.nan]*n_levels   
                data[1,:] = [np.nan]*n_levels   
                data[3,:] = [np.nan]*n_levels   
            else:                
                x_new, y_new = interp2_180(ds["height"].values,\
		        ds["temperature"].values[min_idx, :])
                data[0,:] = x_new     
                data[1,:] = y_new    
                x_new, y_new = interp2_180(ds["height"].values,\
		        ds["absolute_humidity"].values[min_idx, :])       
                data[3,:] = y_new   
                lwp = ds["lwp"].values[min_idx]         
                iwv = ds["iwv"].values[min_idx]       
        if "mwr00_l2_ta_" in file:
            ds = xr.open_dataset(file)
            # time_diffs = np.abs(ds["time"].values - datetime_np)
            # min_idx = time_diffs.argmin()
            min_idx = nearest_ele4elevation(ds["ele"].values, ds["azi"].values,\
                ds["time"].values, 90., "ANY" , datetime_np) 
            if min_idx is None:
                data[0,:] = [np.nan]*n_levels   
                data[1,:] = [np.nan]*n_levels   
            else:                            
                x_new, y_new = interp2_180(ds["height"].values,\
                     ds["ta"].values[min_idx, :])
                data[0,:] = x_new     
                data[1,:] = y_new
        elif "mwrBL00_l2_ta_" in file:
            ds = xr.open_dataset(file)
            time_diffs = np.abs(ds["time"].values - datetime_np)
            min_idx = time_diffs.argmin()
            if min_idx is None:
                data[2,:] = [np.nan]*n_levels     
            else: 
                x_new1, y_new = interp2_180(ds["height"].values,\
                     ds["ta"].values[min_idx, :]) #, x_new=x_new)
                data[2,:] = y_new            
        elif "_hua_" in file:
            ds = xr.open_dataset(file)
            min_idx = nearest_ele4elevation(ds["ele"].values, ds["azi"].values,\
                ds["time"].values, 90., "ANY" , datetime_np)  
            if min_idx is None:
                data[3,:] = [np.nan]*n_levels   
            else:                 
                x_new2, y_new = interp2_180(ds["height"].values,\
                     ds["hua"].values[min_idx, :]) #, x_new=x_new)
                data[3,:] = y_new                 
        elif "_prw_" in file:
            ds = xr.open_dataset(file)
            # time_diffs = np.abs(ds["time"].values - datetime_np)
            # min_idx = time_diffs.argmin()
            min_idx = nearest_ele4elevation(ds["ele"].values, ds["azi"].values,\
                ds["time"].values, 90., "ANY" , datetime_np)                 
            iwv = ds["prw"].values[min_idx] 
            # All Water Vapor
            # print(ds["prw"])             
        elif "_clwvi_" in file:
            ds = xr.open_dataset(file)
            min_idx = nearest_ele4elevation(ds["ele"].values, ds["azi"].values,\
                ds["time"].values, 90., "ANY" , datetime_np)  
            lwp = ds["clwvi"].values[min_idx]       
    
    ######################
    # Führt zu Fehler, wo keine Dateien vorhanden sind...
    # Lief im übrigen für meiste dwdhat files ohne Fehler durch...
    #if (x_new != x_new2).any():
    #    print("Warning: inconsistent interpolation!")
    
    return data[:,::-1], lwp, iwv

##############################################################################

def get_mwr_data(datetime_np, mwrs,n_levels=n_levels):
    dwdhat_pattern = "/home/aki/PhD_data/FESSTVaL_14GB/dwdhat/l*/*/*/*.nc"
    foghat_pattern = "/home/aki/PhD_data/FESSTVaL_14GB/foghat/l*/*/*/*.nc"
    sunhat_pattern = "/home/aki/PhD_data/FESSTVaL_14GB/sunhat/l*/*/*/*.nc"
    tophat_pattern = "/home/aki/PhD_data/Socles/MWR_tophat/*.nc"
    joyhat_pattern = "/home/aki/PhD_data/Vital_I/hatpro-joyhat/*.nc"
    hamhat_pattern = "/home/aki/PhD_data/Vital_I/hamhat/*.nc"
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
        
    # Then jsut read l1-files for now
    # TBs of shape: (elevation x azimuth x n_chans) , for one timestep
    integrals = [lwp_dwd, iwv_dwd, lwp_fog, iwv_fog, lwp_sun, iwv_sun,\
        lwp_top, iwv_top, lwp_joy, iwv_joy,lwp_ham, iwv_ham ]
    
    return tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles,fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, integrals, lat, lon

##############################################################################

def check_units_physical_realism(p_array, t_array, ppmv_array,\
                m_array, z_array, rh):
    if (np.array(p_array)>1100).any() or (np.array(p_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for p in hPa!!!")
    if (np.array(t_array)>400).any() or (np.array(t_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for T in K!!!")
    if (np.array(rh)>115).any() or (np.array(rh)<0).any() or\
            (not  (np.array(rh)>1.5).any()):
        print("WARNING: Encoutered physically unrealistic value for RH in %!!!")
    if (np.array(ppmv_array)>40000).any() or (np.array(ppmv_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for WV in ppmv!!!")
    if (np.array(z_array)>130000).any() or (np.array(z_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for z in m!!!") 
    if (np.array(m_array)>20).any() or (np.array(m_array)<0).any():
        print("WARNING: Encoutered physically unrealistic value for mr in g/kg!!!")
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
                    deg_lat, m_array, z_array, rh =\
                    read_radiosonde_nc_arms(file=file, crop=7)
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
            # Add climatology at top to fill profiles:
            check_units_physical_realism(p_array, t_array, ppmv_array,\
                m_array, z_array, rh)       
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
                    deg_lat, m_array, z_array, rh =\
                    read_radiosonde_nc_arms(file=file)
        elif "Profile.txt" in file:
            length_value, p_array, t_array, ppmv_array, height_in_km,\
                    deg_lat, m_array, z_array, rh =\
                    read_radiosonde_txt(file=file)
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
        lats[i] = lat
        lons[i] = lon
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
        level_wvs[:,i,0] = m_array[-n_levels:]*1000 # convert kg/kg to g/kg
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
         
        ##########   
        #if i==4:                                                                
         #   break 
        ##########                 
    
    return profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza,\
        times, level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat, dwd_profiles, fog_profiles, sun_profiles, top_profiles,\
        joy_profiles, ham_profiles, lwps_dwd, lwps_fog, lwps_sun, lwps_top,\
        lwps_joy, lwps_ham, iwvs_dwd, iwvs_fog, iwvs_sun, iwvs_top, iwvs_joy,\
        iwvs_ham, lwps_rs, lats, lons

##############################################################################

def write_armsgb_input_nc(profile_indices, level_pressures,
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,outifle="blub.nc", n_levels=137):

    # Ermitteln der Dimensionen
    n_levels, n_profiles = level_pressures.shape
    level_o3s = np.empty(np.shape(level_wvs))

    '''
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
    sza = np.array(sza, dtype=np.float32)
    profile_indices = np.array(profile_indices, dtype=np.int32)
    
    '''

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
    ds["Level_Liquid"].attrs["units"] = "kg/kg"
    ds["Level_RH"].attrs["units"] = "%"
    ds["Level_z"].attrs["units"] = "m"
    
    # Schreibe die NetCDF-Datei
    # ds.to_netcdf(outifle, format="NETCDF4_CLASSIC")

    return ds
    
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
    
    ds["TBs_foghat"] = ds["TBs_foghat"].isel(elevation=1)\
        .interpolate_na(dim="azimuth", method="linear")
    ds["TBs_joyhat"] = ds["TBs_joyhat"].isel(elevation=1)\
        .interpolate_na(dim="azimuth", method="linear")
    
    return ds
    
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
        outifle="blub.nc", n_levels=137,\
        campaign="any_camp",\
        location="any_location", elevations=elevations, azimuths=azimuths):

    # Ermitteln der Dimensionen
    n_levels, n_profiles, n_crop = level_pressures.shape
    
    # Setze Dummy-Werte für Dimensionsgrößen
    n_times = len(profile_indices)
    n_channels = 14  # Beispielwert
    any_obs = np.empty([14, n_times])
    # n_data = 1       # Wird oft für Metadaten genutzt

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

    # Add units:
    ds["Level_Pressure"].attrs["units"] = "hPa"
    ds["LWP_radiosonde"].attrs["units"] = "kg m-2"
    ds["elevation"].attrs["units"] = "degree"
    ds["azimuth"].attrs["units"] = "degree"
    ds["Level_Temperature"].attrs["units"] = "K"
    ds["Level_H2O"].attrs["units"] = "g/kg"
    ds["Level_ppmvs"].attrs["units"] = "ppmv"
    ds["Level_Liquid"].attrs["units"] = "kg/kg"
    ds["Level_Ice"].attrs["units"] = "kg/kg"
    ds["Level_RH"].attrs["units"] = "%"
    ds["Level_z"].attrs["units"] = "m"
    ds["Sunhat_hua"].attrs["units"] = "kg m-3"
    ds["Dwdhat_hua"].attrs["units"] = "kg m-3"
    ds["Foghat_hua"].attrs["units"] = "kg m-3"
    ds["Tophat_hua"].attrs["units"] = "kg m-3"
    ds["Joyhat_hua"].attrs["units"] = "kg m-3"
    ds["Hamhat_hua"].attrs["units"] = "kg m-3"
    ds["Sunhat_z"].attrs["units"] = "Height above mean sea level - not consistent for different instruments"
    ds["Dwdhat_z"].attrs["units"] = "Height above mean sea level - not consistent for different instruments"
    ds["Foghat_z"].attrs["units"] = "Height above mean sea level - not consistent for different instruments"
    ds["Tophat_z"].attrs["units"] = "Height above mean sea level - not consistent for different instruments"
    ds["Joyhat_z"].attrs["units"] = "Height above mean sea level - not consistent for different instruments"
    ds["Hamhat_z"].attrs["units"] = "Height above mean sea level - not consistent for different instruments"    
    ds["time"].attrs["units"] = "Seconds since 1970-01-01T00:00:00"
    for var in ["TBs_dwdhat", "TBs_foghat", "TBs_sunhat",\
              "TBs_tophat", "TBs_joyhat", "TBs_hamhat"]:
        ds[var].attrs["units"] = "K"
    for var in ["Dwdhat_ta", "Foghat_ta", "Sunhat_ta",\
                "Tophat_ta", "Joyhat_ta", "Hamhat_ta"]:
        ds[var].attrs["units"] = "K"
    for var in ["Dwdhat_IWV", "Dwdhat_LWP", "Foghat_IWV",\
           "Foghat_LWP","Sunhat_IWV", "Sunhat_LWP", "Tophat_IWV",\
           "Tophat_LWP","Joyhat_IWV", "Joyhat_LWP", "Hamhat_IWV",\
           "Hamhat_LWP"]:
        ds[var].attrs["units"] = "kg m-2"
    ds["Obs_Surface_Pressure"].attrs["units"] = "hPa"
    ds["Obs_Temperature_2M"].attrs["units"] = "K"
    ds["Obs_H2O_2M"].attrs["units"] = "kg/kg"
    ds["Surface_Pressure"].attrs["units"] = "hPa"
    ds["Temperature_2M"].attrs["units"] = "K"
    ds["H2O_2M"].attrs["units"] = "kg/kg"
    ds["Surface_Altitude"].attrs["units"] = "km"

    # Profile_Index: no unit, but add long_name
    ds["Profile_Index"].attrs["long_name"] = "Number of radiosonde in dataset"
    ds["Crop"].attrs["description"] = "Crop=1 contains shortened rs profiles for Processing with roof placed HATPRO Joyhat, as opposed to yard placed HATPRO Hamhat (Crop=0)."
    
    # Additional attrs:
    ds.attrs["author"] = "Alexander Pschera"
    ds.attrs["institution"] = "University of Cologne"    
    ds.attrs["description"] = "Radiosondes and associated MWR BT measurements from Vital I, Socles and FESSTVaL. Level variables are rs."
    
    ds.attrs.update({
        "title": "Radiosonde and microwave radiometer brightness temperature dataset from FESSTVaL, SOCLES, and VITAL-I campaigns",
        "summary": "This dataset contains co-located radiosonde profiles and HATPRO microwave radiometer brightness temperatures from three field campaigns at Jülich ObservatorY for Cloud Evolution (JOYCE) and Richard-Assmann-Observatory (RAO) in Lindenberg.",
        "keywords": "radiosonde, microwave radiometer, brightness temperature, atmospheric profiles, water vapor, liquid water path",
        "Conventions": "CF-1.8",  # WICHTIG für Interoperabilität!
        "history": f"Created on {datetime.utcnow().isoformat()}Z by {os.getlogin()}",
        "source": "Vaisala RS41/GRAW DMF-09 radiosondes and RPG-HATPRO microwave radiometers",
        "references": "FESSTVaL radiosondes and other data: https://www.cen.uni-hamburg.de/icdc/data/atmosphere/samd-st-datasets/samd-st-fesstval.html | https://www.fdr.uni-hamburg.de/record/10279; SOCLES: https://gepris.dfg.de/gepris/projekt/430226822?context=projekt&task=showDetail&id=430226822&; Vital I: https://www.herz.uni-bonn.de/wordpress/index.php/vital-campaigns/",
        "license": "CC BY 4.0",
        # "acknowledgement": "Data collected within the DFG-funded projects FESSTVaL, SOCLES, and VITAL-I.",
        "contact": "apscher1@uni-koeln.de",
        "geospatial_lat_min": str(np.nanmin(lats)),
        "geospatial_lat_max": str(np.nanmax(lats)),
        "geospatial_lon_min": str(np.nanmin(lons)),
        "geospatial_lon_max": str(np.nanmax(lons)),
        "time_coverage_start": str(times.min()),
        "time_coverage_end": str(times.max()),
     })
    
    
    ds = clean_dataset(ds)
    ds = interpolate_azimuths(ds)
    
    return ds

##############################################################################

'''
def write_combined_input_prof_file(t_array, ppmv_array,length_value,\
        p_array, liquid_array, height_in_km=0., deg_lat=50.,\
        filename="prof_plev.dat", zenith_angle=0.):
    with open(filename, "w") as file:
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
'''        
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
        outifle=args.output1,n_levels=n_levels, campaign="FESSTVaL_RAO",\
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
        outifle=args.output1,n_levels=n_levels, campaign="FESSTVaL_UHH",\
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
        outifle=args.output1,n_levels=n_levels, campaign="FESSTVaL_UzK",\
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
        outifle=args.output1 ,n_levels=n_levels, campaign="Socles",\
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
        outifle=args.output1,n_levels=n_levels, campaign="Vital I",\
        location="JOYCE")
   
    ####
    # New dataset:
    ds_list = [ds_fesst_rao, ds_fesst_uhh,ds_fesst_uzk,ds_socles,\
        ds_vital_uncrop]    
    new_ds = xr.concat(ds_list , dim="time")
    
    print(new_ds)
    
    # print(new_ds["time"].values)
    # print(len(new_ds["time"]))
    new_ds.to_netcdf("/home/aki/PhD_data/armsgb_all_campaigns_zenith.nc", format="NETCDF4_CLASSIC")
        

        

