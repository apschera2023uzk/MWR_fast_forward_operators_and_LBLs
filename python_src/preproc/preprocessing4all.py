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
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
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
        return 0, np.nan, np.nan, np.nan, np.nan,np.nan, np.nan, np.nan, np.nan
        
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
    z_array = ds["Height"].isel(Time=inds).values

    m_array = []
    for rh_lev, t_lev, p_lev in zip (rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    ppmv_array = np.array(m_array) * (28.9644e6 / 18.0153)
    
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
    z_array = np.concatenate([z_array , np.array(z*1000)[mask]])
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
    for i, (base, top) in enumerate(zip(bases,tops)):
        z_index_top = np.nanargmin(abs(z_array-top))
        z_index_base = np.nanargmin(abs(z_array-base))
        if t_array[z_index_base]>273.15 and t_array[z_index_top]>273.15:
            for j in range(z_index_top,z_index_base):
                rho = p_array[j]*100 / R_L / t_array[j] # Ergebnis realistisch kg m-3
                dz = z_array[j-1] - z_array[j] # ergab soweit auch Sinn..
                lwc_ad = rho * cp / L * (gamma_d - gamma_s) * dz
                dh = z_array[j] - base
                lwc = lwc_ad * (1.239 - 0.145*np.log(dh)) # kg m-3
                lwc_kg_m3[j] = lwc          
                lwc_kg_kg[j] = lwc / rho
        elif t_array[z_index_base]<233.15 and t_array[z_index_top]<233.15:
            print("Ice cloud")
        elif t_array[z_index_base]>233.15 and t_array[z_index_top]<273.15:
            print("Mixed phase cloud")                    
        else:
            print("Phase determination error!")

    ##################
    # Dafür wäre IWC profil noch nice...
    # Ich brauche IWP
    # Was macht man mit mixed phase???
    
    lwp = np.abs(np.sum(lwc_kg_m3 * np.gradient(z_array)))  # [kg/m²]
        
    return lwc_kg_m3, lwc_kg_kg, lwp_kg_m2

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
    
    #####################
    #rh[100:110] = 100
    #rh[110:114] = 100
    #rh[114:117] = 100
    #rh[149:152] = 100
    #######################
    
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
        # print(i,temp,z, rh[i])
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
          
    #print("after 4  tops: ", tops)    
    #print("after 4  bases: ", bases)     
    ####
    # 5) Remove cloudbases below 500 m if thickness < 400 m:
    new_tops = tops
    new_bases = bases
    for i, (base, top) in enumerate(zip(bases,tops)):
        if base<500 and abs(base-top)<400:
            new_bases.pop(i)
            new_tops.pop(i)
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            cloud_bools[z_index_top:z_index_base] = False
            
         
    ###
    # 6 ) RH_max reached within cloud layer? => discard else!
        if base<2000:
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            if not np.any(below_2km[1] < rh[z_index_top:z_index_base]):
                cloud_bools[z_index_top:z_index_base] = False
                new_bases.pop(i)
                new_tops.pop(i)
        elif 2000<base<6000:
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            if not np.any(two2sixkm[1] < rh[z_index_top:z_index_base]):
                cloud_bools[z_index_top:z_index_base] = False
                new_bases.pop(i)
                new_tops.pop(i)   
        elif 6000<base<12000:
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            if not np.any(six2twelvekm[1] < rh[z_index_top:z_index_base]):
                cloud_bools[z_index_top:z_index_base] = False
                new_bases.pop(i)
                new_tops.pop(i)         
        elif 12000<base:
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))
            if not np.any(above_12km[1]  < rh[z_index_top:z_index_base]):
                cloud_bools[z_index_top:z_index_base] = False
                new_bases.pop(i)
                new_tops.pop(i)           
    
        ###
        # 7) Connect layers, with a gap of less than 300 m:
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
           if abs(bases[i-1]-top)<300 or\
                   np.nanmin(rh[z_index_base:z_index_top-1])>rh_inter:               
               cloud_bools[z_index_base:z_index_top-1] = True
               new_bases.pop(i-1)
               new_tops.pop(i)
               
    #print("after 5, 6, 7  tops: ", new_tops)    
    #print("after 5, 6 ,7  bases: ", new_bases) 
                   
    tops = new_tops
    bases = new_bases
    for i, (base, top) in enumerate(zip(new_bases,new_tops)):               
        ####
        # 8) Cloud thickness below 100 m:
        if abs(base-top)<100:
            bases.pop(i)
            tops.pop(i)
            z_index_top = np.nanargmin(abs(z_array-top))
            z_index_base = np.nanargmin(abs(z_array-base))     
            cloud_bools[z_index_top:z_index_base] = False
                      
    # cloud base heights and cloud top heights...XXXX
    #print("after 8  tops: ", tops)    
    #print("after 8  bases: ", bases)    

    ####
    # Lets get LWC and IWC:
    lwc_kg_m3, lwc_kg_kg, lwp_kg_m2 =\
        calc_lwc(tops, bases, p_array, t_array, ppmv_array, m_array,\
        z_array, rh, cloud_bools)


    # LWC(z) kg/kg; and IWC probably different units for PyRTlib...
    # Column water and ice: g m-2
    

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
        # break
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
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
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
    write_armsgb_input_nc(profile_indices, level_pressures,
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,\
        outifle=args.output1)
        
    # Cropped version:
    profile_indices, level_pressures, level_temperatures, level_wvs,\
        srf_pressures, srf_temps, srf_wvs, srf_altitude, sza, times,\
        level_ppmvs, level_liq, level_z, level_rhs =\
        summarize_many_profiles(crop=True,sza_float=sza_float)
    write_armsgb_input_nc(profile_indices, level_pressures,
        level_temperatures, level_wvs,level_ppmvs, level_liq, level_z,\
        srf_pressures, srf_temps, srf_wvs,\
        srf_altitude, sza,\
        times ,\
        outifle=args.output2)
