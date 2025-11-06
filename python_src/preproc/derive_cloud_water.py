#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Only Cloudwater derival in this module

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
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

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
    rh = np.array(rh, copy=True)
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
############################################################################## 
