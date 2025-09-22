#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Takes a number of NetCDF files from rs as input
# Creates one NetCDF that contains files in ARMS-gb readable structure!

import math
import argparse
import os
import xarray as xr
import numpy as np
import pandas as pd
import glob
import sys
sys.path.append('/home/aki/pyrtlib')
from pyrtlib.climatology import AtmosphericProfiles as atmp
from pyrtlib.utils import ppmv2gkg, mr2rh

elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])
azimuths = np.arange(0.,355.1,5.) # Interpoliere dazwischen!

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
    index3000 = np.nanargmin(abs(ds[height_var].values-3000))
    
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
        
    if max_index<150:
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
    elif ds[height_var].values[max_index]<3000:
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
        
    # Thinning pattern:
    datapoints_bl = 75
    datapoints_ft = 100
    increment_bl = int(index3000/datapoints_bl)
    increment_ft = int((max_index-index3000)/datapoints_ft)
    inds = np.r_[crop:index3000:increment_bl, index3000:max_index:increment_ft]
    inds = np.unique(inds)
    print("len(inds)", len(inds))
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
    length_value = len(t_array )
        
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
    index3000 = np.nanargmin(abs(df["Alt"].values-3000))
        
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
        
    if max_index<150:
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
    elif df["Alt"].values[max_index]<3000:
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
        
    # Thinning pattern:
    datapoints_bl = 75
    datapoints_ft = 100
    increment_bl = int(index3000/datapoints_bl)
    increment_ft = int((max_index-index3000)/datapoints_ft)
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
    m_array = np.concatenate([m_array , np.array(gkg)[mask]])
    z_array = np.concatenate([z_array , np.array(z*1000)[mask]])
    rh = np.concatenate([rh , np.array(rhs_clim)[mask]])
    ppmv_array = np.concatenate([ppmv_array , np.array(md[:, atmp.H2O])[mask]])
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
    for i, (base, top) in enumerate(zip(bases,tops)):
        z_index_top = np.nanargmin(abs(z_array-top))
        z_index_base = np.nanargmin(abs(z_array-base))
        if t_array[z_index_base]>273.15 and t_array[z_index_top]>273.15:
            for j in range(z_index_top,z_index_base):
                rho = p_array[j]*100 / R_L / t_array[j] # Ergebnis realistisch kg m-3
                dz = z_array[j-1] - z_array[j] # ergab soweit auch Sinn..
                lwc_ad = rho * cp / L * (gamma_d - gamma_s) * dz
                dh = z_array[j] - base # abs entfernt
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
    match_mask2 = (abs(azi_values-target_azi)<0.05)
    final_mask = match_mask & match_mask2

    if not final_mask.any():
        # Kein exakter Treffer
        # print("Kein Wert entspricht exakt Winkeln: ")
        # print("Ele/Azi values: ",target_elevation, target_azi)
        nearest_idx = None
    else:
        # Zeitwerte, die zu den passenden Elevations gehören
        candidate_times = ele_times[final_mask]
        time_diffs = np.abs(candidate_times - datetime_np)
        min_idx = time_diffs.argmin()
        nearest_time = candidate_times[min_idx]

        candidate_indices = np.where(match_mask)[0]
        nearest_idx = candidate_indices[min_idx]
        nearest_value = ele_values[nearest_idx]
        # print("Found TBs for angles: ")
        # print("Ele/Azi values: ",target_elevation, target_azi)
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
                        # print("TBs found MWR: ", ds_c1["tb"].values[time_idx,:])
        else: 
            ds_mwr = xr.open_dataset(file)
            # print("Opened MWR for file: ", file)
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
                        # print("TBs found MWR: ", ds_mwr["tb"].values[time_idx,:])
    return tbs

##############################################################################

def get_mwr_data(datetime_np, mwrs):
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
        tbs_dwdhat = get_tbs_from_l1(l1_files, datetime_np)
    else:
        tbs_dwdhat = np.full((10, 72, 14), np.nan)
    if "foghat" in mwrs:
        fog_files = glob.glob(foghat_pattern)
        files = [file for file in fog_files if datestring in file]   
        l1_files = [file for file in files if "l1" in file]   
        l2_files = [file for file in files if "l2" in file] 
        tbs_foghat = get_tbs_from_l1(l1_files, datetime_np)
    else:
        tbs_foghat = np.full((10, 72, 14), np.nan)
    if "sunhat" in mwrs:
        sun_files = glob.glob(sunhat_pattern)
        files = [file for file in sun_files if datestring in file]   
        l1_files = [file for file in files if "l1" in file]   
        l2_files = [file for file in files if "l2" in file] 
        tbs_sunhat = get_tbs_from_l1(l1_files, datetime_np)
    else:
        tbs_sunhat = np.full((10, 72, 14), np.nan)        
    if "tophat" in mwrs:
        top_files = glob.glob(tophat_pattern)
        files = [file for file in top_files if datestring in file]   
        l1_files = [file for file in files if "l1" in file]   
        l2_files = [file for file in files if "l2" in file] 
        tbs_tophat = get_tbs_from_l1(l1_files, datetime_np)
    else:
        tbs_tophat = np.full((10, 72, 14), np.nan)
    if "joyhat" in mwrs:
        joy_files = glob.glob(joyhat_pattern)
        files = [file for file in joy_files if datestring in file]   
        l1_files = [file for file in files if "1C01" in file]   
        l2_files = [file for file in files if "single" in file] 
        tbs_joyhat = get_tbs_from_l1(l1_files, datetime_np)
    else:
        tbs_joyhat = np.full((10, 72, 14), np.nan)
    if "hamhat" in mwrs:
        ham_files = glob.glob(hamhat_pattern)
        files = [file for file in ham_files if datestring in file]   
        l1_files = [file for file in files if "1C01" in file]   
        l2_files = [file for file in files if "single" in file] 
        tbs_hamhat = get_tbs_from_l1(l1_files, datetime_np)
    else:
        tbs_hamhat = np.full((10, 72, 14), np.nan)
        
    # Then jsut read l1-files for now
    # TBs of shape: (elevation x azimuth x n_chans) , for one timestep
    
    return tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat

##############################################################################

def summarize_many_profiles(pattern=\
                    "/home/aki/PhD_data/Vital_I/radiosondes/202408*_*.nc",\
                    crop=False, sza_float =0., n_levels=137, mwrs=""):
                         
    # Bodenlevel ist bei Index -1 - Umkehr der Profile!
    h_km_vital = 0.092
    h_km_vital_crop = 0.112   
    files = glob.glob(pattern)
    n = len(files)
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
    sza = [sza_float]*n
    
    for i, file in enumerate(files):
        # print(i, file)
        profile_indices.append(i)
        datetime_np = derive_date_from_file_name(file)
        ##################
        tbs_dwdhat1, tbs_foghat1,tbs_sunhat1,tbs_tophat1, tbs_joyhat1,\
        tbs_hamhat1 =\
            get_mwr_data(datetime_np, mwrs)
        times[i] = datetime_np
        if crop:
            if ".nc" in file:
                length_value, p_array, t_array, ppmv_array, height_in_km,\
                    deg_lat, m_array, z_array, rh =\
                    read_radiosonde_nc_arms(file=file, crop=7)
            if length_value<170: 
                level_ppmvs[:,i,1] =    np.array([np.nan]*n_levels)
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
            ##########
        else:
            #################
            # p in hPa as in other inputs!
            # mixing ratio in g/kg
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
        if length_value<170:
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
        tbs_dwdhat[i,:,:,:] = tbs_dwdhat1
        tbs_foghat[i,:,:,:] = tbs_foghat1
        tbs_sunhat[i,:,:,:] = tbs_sunhat1
        tbs_tophat[i,:,:,:] = tbs_tophat1
        tbs_joyhat[i,:,:,:] = tbs_joyhat1
        tbs_hamhat[i,:,:,:] = tbs_hamhat1         
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
        # break
    
    return profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza,\
        times, level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat

##############################################################################

def write_armsgb_input_nc(profile_indices, level_pressures,
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,outifle="blub.nc", n_levels=137):

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

def produce_dataset(profile_indices, level_pressures,
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        level_rhs,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat ,tbs_tophat, tbs_joyhat,\
        tbs_hamhat,\
        outifle="blub.nc", n_levels=137,\
        campaign="any_camp",\
        location="any_location", elevations=elevations, azimuths=azimuths):

    # Ermitteln der Dimensionen
    n_levels, n_profiles, n_crop = level_pressures.shape

    # Optional: Konvertiere Inputs in float32 falls nötig
    tbs_dwdhat = np.array(tbs_dwdhat, dtype=np.float32)
    tbs_foghat = np.array(tbs_foghat, dtype=np.float32)
    tbs_sunhat = np.array(tbs_sunhat, dtype=np.float32)
    tbs_tophat = np.array(tbs_tophat, dtype=np.float32)
    tbs_joyhat = np.array(tbs_joyhat, dtype=np.float32)
    tbs_hamhat = np.array(tbs_hamhat, dtype=np.float32)    
    level_pressures = np.array(level_pressures, dtype=np.float32)
    level_temperatures = np.array(level_temperatures, dtype=np.float32)
    level_wvs = np.array(level_wvs, dtype=np.float32)
    level_ppmvs = np.array(level_ppmvs, dtype=np.float32)
    level_liq = np.array(level_liq, dtype=np.float32)
    level_ice = np.array(level_ice, dtype=np.float32)
    level_z = np.array(level_z, dtype=np.float32)
    level_rhs = np.array(level_rhs, dtype=np.float32)
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
            "Level_Pressure":       (("N_Levels", "time","Crop"), level_pressures),
            "Level_Temperature":    (("N_Levels", "time","Crop"), level_temperatures),
            "Level_H2O":            (("N_Levels", "time","Crop"), level_wvs),
            "Level_ppmvs":          (("N_Levels", "time","Crop"), level_ppmvs),
            "Level_Liquid":         (("N_Levels", "time","Crop"), level_liq),
            "Level_Ice":            (("N_Levels", "time","Crop"), level_ice),
            "Level_z":              (("N_Levels", "time","Crop"), level_z),
            'Level_O3':             (("N_Levels", "time","Crop"), level_o3s),
            "Level_RH":              (("N_Levels", "time","Crop"), level_rhs),

            # Oberflächenparameter
            # "times":                (("N_Times"), times),
            "Obs_Surface_Pressure": (("time","Crop"), srf_pressures),
            "Obs_Temperature_2M":   (("time","Crop"), srf_temps),
            "Obs_H2O_2M":           (("time","Crop"), srf_wvs),
            "Surface_Pressure":     (("time","Crop"), srf_pressures),
            "Temperature_2M":       (("time","Crop"), srf_temps),
            "H2O_2M":               (("time","Crop"), srf_wvs),
            "Surface_Altitude":     (("time","Crop"), srf_altitude),

            # Zusätzliche Metadaten
            "Profile_Index":        (("time",), profile_indices.astype(np.float64)),
            "Campaign":        (("time",), [campaign]*len(times)),
            "Location":        (("time",), [location]*len(times)),   
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
    ds["elevation"].attrs["units"] = "degree"
    ds["azimuth"].attrs["units"] = "degree"
    ds["Level_Temperature"].attrs["units"] = "K"
    ds["Level_H2O"].attrs["units"] = "g/kg"
    ds["Level_ppmvs"].attrs["units"] = "ppmv"
    ds["Level_Liquid"].attrs["units"] = "kg/kg"
    ds["Level_Ice"].attrs["units"] = "kg/kg"
    ds["Level_RH"].attrs["units"] = "%"
    ds["Level_z"].attrs["units"] = "m"

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
    n_levels = 180


    # Read FESSTVaL 1 RAO:
    print("Processing FESSTVaL RAO:")
    profile_indices1, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat =\
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
        tbs_hamhat,\
        outifle=args.output1,n_levels=n_levels, campaign="FESSTVaL_RAO",\
        location="RAO_Lindenberg")

    # Read FESSTVaL 2 UHH:
    print("Processing FESSTVaL UHH:")
    profile_indices2, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat =\
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
        tbs_hamhat,\
        outifle=args.output1,n_levels=n_levels, campaign="FESSTVaL_UHH",\
        location="RAO_Lindenberg")
            
    # Read FESSTVaL 2 UzK:
    print("Processing FESSTVaL UzK:")
    profile_indices3, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat =\
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
        tbs_hamhat,\
        outifle=args.output1,n_levels=n_levels, campaign="FESSTVaL_UzK",\
        location="Falkenberg")  
        
    # Read in Socles
    print("Processing Socles:")
    profile_indices4, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat =\
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
        tbs_hamhat,\
        outifle=args.output1 ,n_levels=n_levels, campaign="Socles",\
        location="JOYCE")
        
    # Uncropped:
    print("Processing Vital I:")
    profile_indices5, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs, level_ice,\
        tbs_dwdhat, tbs_foghat, tbs_sunhat, tbs_tophat, tbs_joyhat,\
        tbs_hamhat =\
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
        tbs_hamhat,\
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
        

        

