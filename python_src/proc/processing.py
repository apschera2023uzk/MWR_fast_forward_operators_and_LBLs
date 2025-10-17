#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

# Describe this file...

##############################################################

import math
import argparse
import os
import xarray as xr
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import glob
import shutil
from datetime import datetime
import subprocess
import sys
sys.path.append('/home/aki/pyrtlib')
from pyrtlib.climatology import AtmosphericProfiles as atmp
from pyrtlib.tb_spectrum import TbCloudRTE
from pyrtlib.utils import ppmv2gkg, mr2rh

##############################################################################
# 1.5: Parameter
##############################################################################

n_levels=180
batch_size=20
elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])

##############################################################################
# 2nd Used Functions
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.expanduser("/home/aki/PhD_data/MWR_rs_FESSTVaLSoclesVital1_all_elevations.nc"),
        help="NetCDF file with rs and MWR data"
    )
    parser.add_argument(
        "--rttov", "-r",
        type=str,
        default=os.path.expanduser("/home/aki/RTTOV-gb/rttov_test/test_example_k.1"),
        help="RTTOV-gb directory with input profile!"
    )    
    parser.add_argument(
        "--script", "-s",
        type=str,
        default=os.path.expanduser("/home/aki/RTTOV-gb/rttov_test/run_apschera.sh"),
        help="Shell Script to run RTTOV-gb on profiles!"
    )        
    return parser.parse_args()

##############################################################################

def write1profile2str(t_array, ppmv_array,length_value,\
        p_array, liquid_array, height_in_km=0., deg_lat=50.,\
        zenith_angle=0.):
    string = ""
    for value in p_array:
        string+=f"{value:8.4f}\n"
    for value in t_array:
        string+=f"{value:6.3f}\n"
    for value in ppmv_array:
        string+=f"{value:9.4f}\n"
    for value in liquid_array:
        string+=f"{value:12.6E}\n"
    string+=f"{t_array[-1]:10.4f}{p_array[-1]:10.2f}\n"
    string+=f"{height_in_km:6.3f}{deg_lat:6.1f}\n"
    string+=f"{zenith_angle:6.1f}\n"
        
    return string

##############################################################################

def check4NANs_in_time_crop_ele(ds, i,j,k):
    # Extract the relevant DataArrays for the given indices
    temp_arr = ds["Level_Temperature"].values[:, i, j]
    ppmvs_arr = ds["Level_ppmvs"].values[:, i, j]
    press_arr = ds["Level_Pressure"].values[:, i, j]
    liquid_arr = ds["Level_Liquid"].values[:, i, j]
    
    # Check each array for NaN values
    if np.any(np.isnan(temp_arr)) or \
            np.any(np.isnan(ppmvs_arr)) or \
            np.any(np.isnan(press_arr)) or \
            np.any(np.isnan(liquid_arr)):
        return True
    else:
        return False
        

##############################################################################

def create_RTTOV_gb_in_profiles(ds, args, batch):
    profiles = ""
    valid_indices = []

    print("len(batch): ", len(batch))
    print("len(ds[Crop].values): ", len(ds["Crop"].values))
    print("ds[elevation].values): ", len(ds["elevation"].values))

    for i in batch:    
        for j, crop in enumerate(ds["Crop"].values):
            for k, elevation in enumerate(ds["elevation"].values):
                if check4NANs_in_time_crop_ele(ds, i,j,k):
                    pass
                else:
                    profile1 = write1profile2str(\
                         ds["Level_Temperature"].values[:,i,j],\
                         ds["Level_ppmvs"].values[:,i,j],\
                         len(ds["Level_ppmvs"].values[:,i,j]),\
                         ds["Level_Pressure"].values[:,i,j],\
                         ds["Level_Liquid"].values[:,i,j],\
                         height_in_km=ds["Surface_Altitude"].values[i,j],\
                         deg_lat=ds["Latitude"].values[i],\
                         zenith_angle=90.-elevation)
                    profiles+=profile1
                    valid_indices.append((i,j,k))
                              
    # After loops - save results:
    dir_name = os.path.dirname(args.input)
    outfile = str(dir_name)+"/rttov_profs/prof_plev"+str(i)+".dat"
    out = open(outfile, "w")
    out.write(profiles)
    out.close()
         
    return outfile, valid_indices
    
##############################################################################

def batch_creator(array, batch_size):
    i=0
    while i*batch_size<len(array)-1:
        if i*batch_size+batch_size<len(array)-1:
            yield range(i*batch_size,i*batch_size+batch_size)
        else:
            yield range(i*batch_size,len(array))
        i+=1

##############################################################################

def run_rttov_gb(outfile, valid_indices, args,nlevels = n_levels):

    # 1st Copy prof_plev.dat to inputs
    where2 = args.rttov+"/prof_plev.dat"
    shutil.copy(outfile, where2)
    
    # 2nd edit run script for level number:
    script_file = args.script
    nlevels = n_levels
    nprofs = len(valid_indices)
    with open(script_file, "r") as f:
        lines = f.readlines()
        # Zeile 30 (Index 29) ersetzen
    lines[28] = f"NPROF={nprofs}\n"    
    lines[29] = f"NLEVELS={nlevels}\n"
    with open(script_file, "w") as f:
        f.writelines(lines)    
        
    # 3rd run RTTOV-gb
    rttov_dir = os.path.dirname(args.script)
    subprocess.run(["bash", args.script, "ARCH=gfortran"], cwd=rttov_dir)    
    
    return 0
    
##############################################################################

def get_rttov_outputs(valid_indices, rttovgb_outfile=\
        "/home/aki/RTTOV-gb/rttov_test/test_example_k.1/output_example_k.dat.gfortran",\
                   batch_size=batch_size,n_levels=n_levels):
    print("Reading in RTTOV-gb output from: ", rttovgb_outfile)
    
    tbs = np.full((batch_size, 14,10,2), np.nan)
    trans = np.full((batch_size, 14,10,2), np.nan)
    # time, channel, elevation, crop
    trans_by_lev=np.full((batch_size,n_levels, 14, 10,2 ), np.nan)
    # time, level, channel, elevation, crop
    jacs_by_lev=np.full((batch_size, n_levels, 14, 10,2, 4), np.nan)
    # time, level, channel, elevation, crop, variable (last one removed in ds)
    
    switch = False
    switch_t = False
    sw_trans_by_lev=False
    sw_jacs_by_lev=False
    switch_count = 0
    switcht_count = 0
    sw_c_trans_by_lev = 0
    sw_c_jacs_by_lev = 0
    tb_string = ""
    trans_string = ""
    string_jc_by_lev=""
    string_tr_by_lev = ""
    
    file = open(rttovgb_outfile, "r")
    
    # Read in TBs: 
    for i, line in enumerate(file.readlines()):
            
        if "Profile      " in line:
            prof_idx=int(line.split(" ")[-1])-1
            rs_time_idx, crop_idx, ele_idx = valid_indices[prof_idx] 
            rs_time_idx = rs_time_idx%20
            # print("i (profile in batch: ",  i )
            # print("rs_time_idx, crop_idx, ele_idx ::", rs_time_idx, crop_idx, ele_idx )
            # print("prof_idx: ", prof_idx)
        
        if switch and switch_count<2:
            switch_count+= 1
            tb_string+= line
        elif "CALCULATED BRIGHTNESS TEMPERATURES (K):" in line:
            switch = True
        elif switch:
            switch = False
            liste = tb_string.split(" ")
            tbs_rs = [float(s.strip("\n")) for s in liste if s.strip() != ""]
            tb_string = ""
            switch_count = 0
            tbs[rs_time_idx, :, ele_idx, crop_idx] = np.array(tbs_rs)
            
        # Read in Tot Transmittances:     
        if switch_t and switcht_count<2:
            switcht_count+= 1
            trans_string+= line
        elif "CALCULATED SURFACE TO SPACE TRANSMITTANCE:" in line:
            switch_t = True
        elif switch_t:
            switch_t = False
            liste = trans_string.split(" ")
            tot_trans_by_chan = [float(s.strip("\n")) for s in liste if s.strip() != ""]
            trans_string = ""
            switcht_count = 0
            trans[rs_time_idx, :, ele_idx, crop_idx]=np.array(tot_trans_by_chan)
          
        # Read in Lev Transmittances:   
        if sw_trans_by_lev and sw_c_trans_by_lev<n_levels:
            sw_c_trans_by_lev+= 1
            string_tr_by_lev+= line
        elif "Level to surface transmittances for channels" in line:
            sw_trans_by_lev = True
        elif sw_trans_by_lev:
            sw_trans_by_lev = False
            liste = string_tr_by_lev.split("\n")[1:]  #.split(" ")
            for j, line in enumerate(liste):
                list_of_numbers = line.split(" ")
                trans_by_lev1 = [float(s) for s in list_of_numbers if s.strip() != "" and s.strip() != "**"]
                if len(trans_by_lev1)<4:
                    break
                elif 4<=len(trans_by_lev1)<5:
                    trans_by_lev[rs_time_idx, j,10:, ele_idx, crop_idx]=\
                        np.array(trans_by_lev1)
                elif 5<=len(trans_by_lev1)<6:
                    trans_by_lev[rs_time_idx, j,10:, ele_idx, crop_idx] =\
                        np.array(trans_by_lev1[1:])                  
                elif j<99:
                    trans_by_lev[rs_time_idx, j,:10, ele_idx, crop_idx] =\
                        np.array(trans_by_lev1[1:])
                else:
                    trans_by_lev[rs_time_idx, j,:10, ele_idx, crop_idx] =\
                        np.array(trans_by_lev1)            
            string_tr_by_lev = ""
            sw_c_trans_by_lev = 0              
        
        # Read in Jacobians somehow:       
        if "Channel        " in line:
            sw_jacs_by_lev = True
            ch_idx = int(line.split("Channel")[-1])-1
        if sw_jacs_by_lev and sw_c_jacs_by_lev<n_levels+3:
            sw_c_jacs_by_lev+= 1
            string_jc_by_lev+= line
        elif sw_jacs_by_lev:
            liste = string_jc_by_lev.split("\n")[3:n_levels+3]
            for j, line in enumerate(liste):
                # print(j, line)
                werte = line.split()
                jacs_by_lev[rs_time_idx, j, ch_idx, ele_idx, crop_idx, :]=\
                    werte[1:]
            string_jc_by_lev=""
            sw_jacs_by_lev = False
            sw_c_jacs_by_lev= 0
            
    file.close()
    print("Finished reading RTTOV-gb output from: ", rttovgb_outfile)
    return tbs, trans, trans_by_lev, jacs_by_lev
    
##############################################################################

def write_jacobians_and_level_transmissions_to_file(trans_by_lev,\
        jacs_by_lev, batch, args,ds, valid_indices, elevations=elevations):
    outfile = os.path.dirname(args.input)+\
        "/RTTOV_jacs_and_trans/RTTOV-gb_"+str(batch)+\
        "_jacs_and_trans.nc"
    n_channels=14
    time_inds = np.unique(np.array([indices[0] for indices in valid_indices]))
    times=ds["time"].isel(time=time_inds).values
        
    ds = xr.Dataset(
        data_vars={
            'levtrans_RTTOV_gb':        (('time', 'N_Levels',"N_Channels",\
                'elevation','Crop'), trans_by_lev),
            'Jacobian_p_RTTOV_gb':      (('time', 'N_Levels','Crop'),\
                jacs_by_lev[:,:,0,0,:,0]),
            'Jacobian_T_RTTOV_gb':        (('time', 'N_Levels',"N_Channels",\
                'elevation','Crop'), jacs_by_lev[:,:,:,:,:,1]),    
            'Jacobian_ppmv_RTTOV_gb':    (('time', 'N_Levels',"N_Channels",\
                 'elevation','Crop'), jacs_by_lev[:,:,:,:,:,2]),    
            'Jacobian_liq_RTTOV_gb':     (('time', 'N_Levels',"N_Channels",\
                 'elevation','Crop'), jacs_by_lev[:,:,:,:,:,3]),                   
        },
        coords={
            "Crop":     np.array([False, True]),
            "N_Channels": np.arange(n_channels),
            "time":    times,
            "N_Levels":   np.arange(n_levels),
            "elevation":   elevations,
        }
    )
    
    ds['levtrans_RTTOV_gb'].attrs["units"]="dimensionless"
    ds['levtrans_RTTOV_gb'].attrs["long_name"]="Level Transmissivities by channel from RTTOV-gb"
    ds['levtrans_RTTOV_gb'].attrs["description"]="Transmissivities were derived with Fast RTE from radiosonde profiles." 
    
    ds['Jacobian_p_RTTOV_gb'].attrs["units"]="hPa"
    ds['Jacobian_p_RTTOV_gb'].attrs["long_name"]="Pressure levels for Sensitvities from RTTOV-gb"

    ds['Jacobian_T_RTTOV_gb'].attrs["units"]="K K-1"
    ds['Jacobian_T_RTTOV_gb'].attrs["long_name"]="TB Level Sensitivities to temperature by channel from RTTOV-gb"
    ds['Jacobian_T_RTTOV_gb'].attrs["description"]="Sensitivities were derived with Fast RTE from radiosonde profiles."   
    
    ds['Jacobian_ppmv_RTTOV_gb'].attrs["units"]="K ppmv-1"
    ds['Jacobian_ppmv_RTTOV_gb'].attrs["long_name"]="TB Level Sensitivities to WV by channel from RTTOV-gb"
    ds['Jacobian_ppmv_RTTOV_gb'].attrs["description"]="Sensitivities were derived with Fast RTE from radiosonde profiles." 

    ds['Jacobian_liq_RTTOV_gb'].attrs["units"]="K kg kg-1"
    ds['Jacobian_liq_RTTOV_gb'].attrs["long_name"]="TB Level Sensitivities to liquid water content by channel from RTTOV-gb"
    ds['Jacobian_liq_RTTOV_gb'].attrs["description"]="Sensitivities were derived with Fast RTE from radiosonde profiles."
    
    ds.to_netcdf(outfile, format="NETCDF4_CLASSIC")
         
    return 0

##############################################################################

def derive_TBs4RTTOV_gb(ds, args, batch_size=batch_size):
    all_valid_indices = []
    all_tbs = []
    all_trans = []
    all_trans_by_lev = []
    all_jacs_by_lev = []
    
    for m, batch in enumerate(batch_creator(ds["time"].values, batch_size)):
        print("batch: ", batch, " of total ",\
            len(ds["time"].values), " in processing")
        
        # 1st Create prof_plev file from ds:
        outfile, valid_indices = create_RTTOV_gb_in_profiles(ds, args, batch)
        all_valid_indices+=valid_indices
        
        # 2nd Run RTTOV-gb on them:
        run_rttov_gb(outfile, valid_indices, args)       

        # 3rd Extract RTTOV-gb TBs:
        tbs, trans, trans_by_lev, jacs_by_lev = get_rttov_outputs(valid_indices,\
            batch_size=len(batch),\
            rttovgb_outfile=\
            "/home/aki/RTTOV-gb/rttov_test/test_example_k.1/output_example_k.dat.gfortran")
            
        # Smaller Jacobian files:
        #write_jacobians_and_level_transmissions_to_file(trans_by_lev,\
        #    jacs_by_lev, batch, args,ds, valid_indices)
            
        all_tbs.append(tbs)
        all_trans.append(trans) 
        all_trans_by_lev.append(trans_by_lev)    
        all_jacs_by_lev.append(jacs_by_lev)   

    # Nach der Schleife alles zusammenfügen:
    tbs_concat = np.concatenate(all_tbs, axis=0)
    trans_concat = np.concatenate(all_trans, axis=0)
    trans_by_lev_concat = np.concatenate(all_trans_by_lev, axis=0)
    jacs_by_lev_concat = np.concatenate(all_jacs_by_lev, axis=0)
    ds['TBs_RTTOV_gb'] = (('time', 'N_Channels', 'elevation','Crop'), tbs_concat)
    ds['TBs_RTTOV_gb'].attrs["units"]="K"
    ds['TBs_RTTOV_gb'].attrs["long_name"]="Brightness temperatures from RTTOV-gb"
    ds['TBs_RTTOV_gb'].attrs["description"]="TBs were derived with Fast RTE from radiosonde profiles."
    
    ds['ttrans_RTTOV_gb'] = (('time', 'N_Channels', 'elevation','Crop'), trans_concat)
    ds['ttrans_RTTOV_gb'].attrs["units"]="dimensionless"
    ds['ttrans_RTTOV_gb'].attrs["long_name"]="Total Transmissivities by channel from RTTOV-gb"
    ds['ttrans_RTTOV_gb'].attrs["description"]="Transmissivities were derived with Fast RTE from radiosonde profiles."        

    # Complete Jacobians and transmissions:
    ds['levtrans_RTTOV_gb'] = (('time', 'N_Levels',"N_Channels",'elevation','Crop'), trans_by_lev_concat)
    ds['levtrans_RTTOV_gb'].attrs["units"]="dimensionless"
    ds['levtrans_RTTOV_gb'].attrs["long_name"]="Level Transmissivities by channel from RTTOV-gb"
    ds['levtrans_RTTOV_gb'].attrs["description"]="Transmissivities were derived with Fast RTE from radiosonde profiles." 
    
    ds['Jacobian_p_RTTOV_gb'] = (('time', 'N_Levels','Crop'),jacs_by_lev_concat[:,:,0,0,:,0])
    ds['Jacobian_p_RTTOV_gb'].attrs["units"]="hPa"
    ds['Jacobian_p_RTTOV_gb'].attrs["long_name"]="Pressure levels for Sensitvities from RTTOV-gb"

    ds['Jacobian_T_RTTOV_gb'] = (('time', 'N_Levels',"N_Channels",'elevation','Crop'), jacs_by_lev_concat[:,:,:,:,:,1])
    ds['Jacobian_T_RTTOV_gb'].attrs["units"]="K K-1"
    ds['Jacobian_T_RTTOV_gb'].attrs["long_name"]="TB Level Sensitivities to temperature by channel from RTTOV-gb"
    ds['Jacobian_T_RTTOV_gb'].attrs["description"]="Sensitivities were derived with Fast RTE from radiosonde profiles."   
    
    ds['Jacobian_ppmv_RTTOV_gb'] = (('time', 'N_Levels',"N_Channels", 'elevation','Crop'), jacs_by_lev_concat[:,:,:,:,:,2])
    ds['Jacobian_ppmv_RTTOV_gb'].attrs["units"]="K ppmv-1"
    ds['Jacobian_ppmv_RTTOV_gb'].attrs["long_name"]="TB Level Sensitivities to WV by channel from RTTOV-gb"
    ds['Jacobian_ppmv_RTTOV_gb'].attrs["description"]="Sensitivities were derived with Fast RTE from radiosonde profiles." 

    ds['Jacobian_liq_RTTOV_gb'] = (('time', 'N_Levels',"N_Channels",'elevation','Crop'), jacs_by_lev_concat[:,:,:,:,:,3])
    ds['Jacobian_liq_RTTOV_gb'].attrs["units"]="K kg kg-1"
    ds['Jacobian_liq_RTTOV_gb'].attrs["long_name"]="TB Level Sensitivities to liquid water content by channel from RTTOV-gb"
    ds['Jacobian_liq_RTTOV_gb'].attrs["description"]="Sensitivities were derived with Fast RTE from radiosonde profiles."
            
    return ds
    
##############################################################################  

def check_for_nans(z_in, p_in, t_in, rh_in, frqs, ang):
    return np.any([
        np.isnan(z_in).any(),
        np.isnan(p_in).any(),
        np.isnan(t_in).any(),
        np.isnan(rh_in).any(),
        np.isnan(frqs).any(),
        np.isnan(ang).any()
    ])
      
##############################################################################  

def derive_TBs4PyRTlib(ds, args):
    # Dieser Code ist sehr langsam...
    # Es liegt schon ein Unterschied zwischen 48 Sonden oder 521 Sonden à 10 Winkel...

    frqs = np.array([22.24,23.04,23.84,25.44,26.24,27.84,31.4,51.26,52.28,\
        53.86,54.94,56.66,57.3,58.])
    nf = len(frqs)    
    mdls = ["R17", "R03", "R16", "R19", "R98", "R19SD", "R20", "R20SD","R24"]
    tags = ["Rosenkranz 17", "Tretjakov 2003", "Rosenkranz 17 (2)",\
        "Rosenkranz + Cimini", "Rosenkranz + Cimini SD", "Makarov",\
         "Makarov SD", "Rosenkranz 24"]
    tbs = np.full((len(ds["time"].values), 14,10,2), np.nan)     
     
    for i, timestep in enumerate(ds["time"].values):
        for j, crop in enumerate(ds["Crop"].values):
            for k, elevation in enumerate(ds["elevation"].values):
                # print("Indices: ",i,j,k)
            
                #########
                # Put elevation in here:
                ang = np.array([elevation])
                
                # Which input variables do I need?
                rh_in = ds["Level_RH"].isel(time=i).isel(Crop=j).values/100
                # as fraction not %
                z_in =  ds["Level_z"].isel(time=i).isel(Crop=j).values/1000
                # m to km!
                p_in = ds["Level_Pressure"].isel(time=i).isel(Crop=j).values # hPa
                t_in =ds["Level_Temperature"].isel(time=i).isel(Crop=j).values  # K
                
                # Exclude profiles with any Nans again:
                nan_bool = check_for_nans(z_in, p_in, t_in, rh_in, frqs, ang)
                
                if not nan_bool:
                    # Run RTE model:
                    mdl = mdls[-1] # Rosenkranz 24
                    rte = TbCloudRTE(z_in[::-1], p_in[::-1], t_in[::-1], rh_in[::-1], frqs, ang)
                    rte.init_absmdl(mdl)
                    ####################
                    # Clear sky!!!
                    # rte.cloudy = False 
                    ####################
                    rte.satellite = False # downwelling!!!
                    df_from_ground = rte.execute()                 
                    tbs[i,:,k,j] = df_from_ground["tbtotal"].values
                else:
                    print("NaNs found!!!!!!!")

    ds["TBs_PyRTlib_R24"] = (('time', 'N_Channels','elevation','Crop'), tbs)
    return ds
    
##############################################################################
# 3rd: Main code:
##############################################################################

if __name__=="__main__":
    args = parse_arguments()
    ds = xr.open_dataset(args.input)
    
    ##################
    # print(ds.data_vars)
    # What the program should do:
    
    # 1 Derive TBs for all elevations  for RTTOV-gb
    ds = derive_TBs4RTTOV_gb(ds, args)

    # 2 Derive TBs for all elevations  for pyrtlib
    ds = derive_TBs4PyRTlib(ds, args)

    # 3. Derive TBs for clear sky for ARMS-gb

    # 3 Print dataset to NetCDF
    ds.to_netcdf(\
        "/home/aki/PhD_data/3campaigns_TBs_processed.nc",\
         format="NETCDF4_CLASSIC")
        

        

