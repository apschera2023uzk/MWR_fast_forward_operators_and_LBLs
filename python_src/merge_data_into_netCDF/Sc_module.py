#!/usr/bin/env python3

###########################################################################
# 1. Necessary modules:
###########################################################################

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from geopy.distance import geodesic

###########################################################################
# 2. Functions:
###########################################################################

def convert_single_timestamp(timestamp):
    day = str(timestamp)[:4]+"-"+str(timestamp)[4:6]+"-"+str(timestamp)[6:8]
    hour_decimal = timestamp % 1
    hour = int(hour_decimal * 24)
    minute_decimal = (hour_decimal * 24 - hour) * 60
    minute = int(minute_decimal)
    second_decimal = (minute_decimal - minute) * 60
    second = int(second_decimal)
    microsecond = int((second_decimal - second) * 1e6)
    
    time = np.datetime64(str(day)) + np.timedelta64(hour, 'h') + np.timedelta64(minute, 'm') + np.timedelta64(second, 's') + np.timedelta64(microsecond, 'us')
    
    return time

##############################################

def convert_to_float_timestamps(timestamps):
    float_timestamps = []
    for ts in timestamps:
        ts_str = str(ts)
        day_str = ts_str[:10]
        hour = int(ts_str[11:13])
        minute = int(ts_str[14:16])
        second = int(ts_str[17:19])
        try:
            microsecond = int("0."+ts_str[20:])
        except:
            microsecond = 0.
            
        # Berechnung des Float-Zeitstempels
        float_ts = float(day_str[:4])*10000+float(day_str[5:7])*100+float(day_str[8:10]) + \
                   (hour * 3600 + minute * 60 + second + microsecond) / 86400
        
        float_timestamps.append(float_ts)
    
    return np.array(float_timestamps)

##############################################

def find_nearest_gridbox(lat, lon, grid_lats, grid_lons):
    min_distance = float('inf')
    nearest_index = None
    
    for i, (grid_lat, grid_lon) in enumerate(zip(grid_lats, grid_lons)):
        grid_point = (np.rad2deg(grid_lat), np.rad2deg(grid_lon))
        target_point = (np.rad2deg(lat), np.rad2deg(lon))
        distance = geodesic(grid_point, target_point).kilometers
        
        if distance < min_distance:
            min_distance = distance
            nearest_index = i
    
    return nearest_index

##############################################

def calc_Ri_bulk(Theta_vsurface, Theta_vz, u_z, v_z, g=9.81):
    Ri_bulk = g/Theta_vsurface * (Theta_vz - Theta_vsurface) / (u_z**2 + v_z**2)
    return Ri_bulk

################################

def magnusformel(temp_celsius):
    # Sättigungsdampfdruck für eine Temperatur in °C
    # es returned in Pa
    es = 610.78 * np.exp((17.08085 * temp_celsius)/(243.175 + temp_celsius))
    return es

#################################

def clausius_clapeyron(temp_celsius):
    # Sättigungsdampfdruck für eine Temperatur in °C
    # es returned in Pa
    es = 610.78 * np.exp(2.5e6 / 462 * (1/273.15 - 1/(273.15+temp_celsius)))
    return es

#################################

def rh2mixing_ratio(RH=70, abs_T=273.15+15, p=101325):
    es = clausius_clapeyron(abs_T-273.15)
    e = es * RH / 100
    mue = 0.622
    q = (mue*e) / (p-0.3777*e)
    m = q / (1-q)
    return m

#################################

def absolute_moisture_rhov_or_f(vapour_pressure_e, temperature_abs):
    # calculates kg/m**3 from %
    # Takes in e_pressure in Pa
    Rv = 461.52
    rhov = vapour_pressure_e / (temperature_abs * Rv)
    return rhov

################################

def vapour_pressure_e_from_rh(relative_humidity=70, temperature_abs=273.15-15):
    # e returned in hPa
    vapour_pressure_e = clausius_clapeyron(temperature_abs-273.15) * relative_humidity / 100
        
    return vapour_pressure_e

################################

def abs_hum_from_rh(rhi, tempi):
    e_pressure = vapour_pressure_e_from_rh(rhi, tempi)
    abs_hum = absolute_moisture_rhov_or_f(e_pressure, tempi)
    return abs_hum

################################

def absolute_hum2specific_hum(rhov, T=15, p=1013.25):
    # T must be provided in °C (can easily be changed to Kelvin)
    # rhov must be provided in kg/m**3
    # p must be provided in hPa
    Rv=461.52
    K0=273.15
    mue=0.622
    e=rhov*Rv*(T+K0)
    q=(mue*e)/(p-0.3777*e)
    return q

################################

def relative2sepcific_humidity(relative_humidity=70, temperature_abs=273.15-15, pressure=1013.25):
    vapour_pressure_e = vapour_pressure_e_from_rh(relative_humidity , temperature_abs)
    rhov = absolute_moisture_rhov_or_f(vapour_pressure_e, temperature_abs)
    q = absolute_hum2specific_hum(rhov, T=temperature_abs-273.15, p=pressure)
    return q

#################################

def rh_aenderung_durch_temperatur(temp1=-15, temp2=20, rh1=70):
    # temperatures in °C:
    rh2 = (rh1*magnusformel(temp1))/ magnusformel(temp2)
    return rh2

################################

def absolute_moisture_rhov_or_f(vapour_pressure_e, temperature_abs):
    # calculates kg/m**3 from %
    Rv = 461.52
    rhov = vapour_pressure_e / (temperature_abs * Rv)
    return rhov

################################

def absolute_moisture2e_pressure(rhov, temperature_abs):
    # calculates kg/m**3 from %
    Rv = 461.52
    vapour_pressure_e = rhov * (temperature_abs * Rv)
    return vapour_pressure_e

################################

def vapour_pressure_e_from_rh(relative_humidity=70, temperature_abs=273.15-15):
    # e returned in hPa
    vapour_pressure_e = magnusformel(temperature_abs-273.15) * relative_humidity / 100
    return vapour_pressure_e

################################

def vapour_pressure_e_2rh(vapour_pressure_e, temperature_abs=273.15-15):
    # e returned in hPa
    relative_humidity = 100* vapour_pressure_e / magnusformel(temperature_abs-273.15)
    return relative_humidity

################################

def absolute_hum2specific_hum(rhov, T=15, p=1013.25):
    # T must be provided in °C (can easily be changed to Kelvin)
    # rhov must be provided in kg/m**3
    # p must be provided in hPa
    Rv=461.52
    K0=273.15
    mue=0.622
    e=rhov*Rv*(T+K0)
    q=(mue*e)/(p-0.3777*e)
    return q

################################

def specific_hum2absolute_hum(q, T=15, p=1013.25):
    # T must be provided in °C (can easily be changed to Kelvin)
    # rhov must be provided in kg/m**3
    # p must be provided in hPa
    Rv=461.52
    K0=273.15
    mue=0.622
    e = q*p / (mue+0.3777*q)
    rhov = e /Rv /(T+K0)
    # Output in g/kg
    return rhov

##################################

def calculate_IWV(rho_vs, heights):
    print("IWV ointegration is not ideal, maybe use trapez or simpson's rule.")
    # rho_vs array of absolute humidities kg/m**2
    # heights: array of heights in m
    IWV = 0
    for i, level in enumerate(heights[:-1]):
        IWV += rho_vs[i] * abs(heights[i]-heights[i+1])
    return IWV

##################################

def rhov2mixing(rhov, T=288.15, p=101325):
    # All inputs in SI units! (K, Pa, kg m-2 => kg/kg)
    # converts absolute humidity into mixing ratio
    Rv=461.52
    mue=0.622
    e=rhov*Rv*(T)
    q=(mue*e)/(p-0.3777*e)
    m = q / (1-q)
    return m
    
##################################

def mixing2specific(mixing):
    # All inputs in SI units! (K, Pa, kg m-2 => kg/kg)
    # converts absolute humidity into mixing ratio
    q = mixing / (1 +mixing)
    # m = q / (1-q)
    return q

################################

def relativey2sepcific_humidity(relative_humidity=70, temperature_abs=273.15-15, pressure=1013.25):
    vapour_pressure_e = vapour_pressure_e_from_rh(relative_humidity , temperature_abs)
    rhov = absolute_moisture_rhov_or_f(vapour_pressure_e, temperature_abs)
    q = absolute_hum2specific_hum(rhov, T=temperature_abs-273.15, p=pressure)
    return q

################################

def specific2relative_humidity(q=1, temperature_abs=273.15-15, pressure=1013.25):
    rhov = specific_hum2absolute_hum(q, T=temperature_abs-273.15, p=pressure)
    e_pressure = absolute_moisture2e_pressure(rhov, temperature_abs)
    rh = vapour_pressure_e_2rh(e_pressure, temperature_abs=temperature_abs)
    return rh

################################

def write_sounding_string(ps=1013.25, zs=np.array([500,1000,1500]), Thetas=np.array([288, 289, 300]),\
                          qs=np.array([10.1, 8, 9]), us=np.array([7, 6, 8]), vs=np.array([7, 6, 8]), zmax=5000,
                         filename = "sound_in_python"):    
    # Erste Zeile mit ps und den ersten Werten von Thetas, qs, us, vs
    text = f"  {ps:8.3f}{'':3}".zfill(8)
    for values in [Thetas[0], qs[0], us[0], vs[0]]:
        text += f"{values:7.3f}{'':3}".zfill(7)    # Zeilen 2 bis len(zs)+1
    for z, Theta, q, u, v in zip(zs, Thetas, qs, us, vs):
        ###########
        # Ist die Null in zweiter Zeile Grund fuer dne Abbruch?
        if z==0:
            continue
        ##########3
        text += f"\n  {z:8.3f}{'':3}".zfill(8)
        for values in [Theta, q, u, v]:
            text += f"{values:7.3f}{'':3}".zfill(7)        
        ########33
        # Abbruch in unterer Atmosphaere:
        if z>zmax:
            break
        ###########    

    new_file = open(filename,"w")    
    new_file.write(text)
    new_file.close()
    return text

################################

def ucla_time2useful_time(ucla_time):
    # <class 'numpy.datetime64'> angeblich hat es bereits diesen Datentypen...
    time_list = []
    switch = False
    for i, timestep in enumerate(ucla_time):
        if (i>2 and str(timestep)[10:19]=="T00:00:00") or switch:
            string = "2018-08-23"+str(timestep)[10:]
            switch = True
        else:
            string = "2018-08-22"+str(timestep)[10:]
        time_list.append(np.datetime64(string))
    new_time = np.array(time_list)
    return new_time

################################

def barometr(z, p0=1000, T=273.15+4):
    # 4°C erscheinen mir angemessen
    # STBL: Wolken -10 bis +10 °C
    # Wasseroberfläche um die 16°C
    hs = 0.02896*9.807 /8.314 /T # M*g/R /T
    p1 = p0*np.exp(- hs*z)
    return p1

################################
##############################################
