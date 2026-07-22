#!/usr/bin/env python3

##############################################################################
# Author: Alexander Pschera
#
# Versatile version: takes ONE radiosonde folder + ONE MWR folder + campaign/
# location labels, and produces (or appends to) a single NetCDF in RT-model
# friendly structure. Multiple campaigns/sites can be built up incrementally
# by re-running this script with --append pointing at the previous output.
##############################################################################

import argparse
import glob
import os
import sys
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime, timezone
from scipy.interpolate import interp1d

sys.path.append('/home/aki/pyrtlib')
from pyrtlib.climatology import AtmosphericProfiles as atmp
from pyrtlib.utils import ppmv2gkg, mr2rh
from derive_cloud_water import derive_cloud_features

##############################################################################
# Parameters (unchanged physics/geometry defaults)
##############################################################################

elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4, 6.6, 5.4, 4.8, 4.2])
azimuths   = np.arange(0., 355.1, 5.)
n_levels_default   = 180
min_p_default      = 137
datapoints_bl      = 80
datapoints_ft      = 120
min_time_diff_thres= 15
max_elev_azi_diff  = 0.05

##############################################################################
# Argument parsing
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Match radiosondes with MWR data (single site/campaign) "
                    "and (optionally) append to an existing preprocessed NetCDF."
    )
    parser.add_argument("--rs_dir", "-r", type=str,\
        default=os.path.expanduser("~/PhD_data/vitII_MWR_rs_comp/sondes_sin"),
        # default=os.path.expanduser("PhD_data/Socles/radiosondes/202*/SOUNDING DATA"),
        # default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes"),
        # default=os.path.expanduser("~/PhD_data/vitII_MWR_rs_comp/sondes_cgn"),
        help="Folder containing radiosonde NetCDF (or .txt) files for this site.")
    parser.add_argument("--mwr_dir", "-m", type=str,\
        default=os.path.expanduser("~/PhD_data/vitII_MWR_rs_comp/kithat_sin"),
        # default=os.path.expanduser("~/PhD_data/Socles/MWR_tophat"),
        # default=os.path.expanduser("~/PhD_data/Vital_I/hatpro-joyhat"),   
        # default=os.path.expanduser("~/PhD_data/vitII_MWR_rs_comp/foghat_cgn"),
        help="Folder containing MWR l1/l2 NetCDF files for this site.")
    parser.add_argument("--campaign", "-c", type=str, default="Vital_I",
        help="Campaign label to store in the output dataset.")
    parser.add_argument("--location", "-l", type=str, default="JOYCE",
        help="Site/location label to store in the output dataset.")
    parser.add_argument("--append", "-a", type=str,
        default=os.path.expanduser("~/PhD_data/TB_preproc_and_proc_results/combined.nc"),
        help="Path to an existing preprocessed NetCDF to append the new data to.")
    parser.add_argument("--output", "-o", type=str,
        default=os.path.expanduser("~/PhD_data/TB_preproc_and_proc_results/combined_o.nc"),
        help="Output NetCDF path.")
    parser.add_argument("--height_offset", type=float, default=0.0,
        help="Height offset [m] applied to MWR height profiles (e.g. instrument"
             " elevation above sea level, if the MWR reports height AGL).")
    parser.add_argument("--n_levels", type=int, default=n_levels_default)
    parser.add_argument("--min_p", type=float, default=min_p_default)
    return parser.parse_args()

##############################################################################
# Humidity / thermodynamics helpers (unverändert aus deinem Modul)
##############################################################################

def clausius_clapeyron_liq(temp_celsius):
    L = 2.5e6
    return 610.78 * np.exp(L / 462 * (1/273.15 - 1/(273.15+temp_celsius)))

##############################################################################

def rh2mixing_ratio(RH=70, abs_T=273.15+15, p=101325):
    es = clausius_clapeyron_liq(abs_T-273.15)
    e = es * RH / 100
    mue = 0.622
    q = (mue*e) / (p-0.3777*e)
    return q / (1-q)

##############################################################################

def rh2ppmv(RH=70, abs_T=273.15+15, p=101325):
    es = clausius_clapeyron_liq(abs_T-273.15)
    e = es * RH / 100
    return 1000000*e / p

##############################################################################

def running_mean_from_arrays(inds, z_array, any_array):
    new_array = []
    ind_max = len(inds)
    for i, ind in enumerate(inds):
        if i == 0 or i == ind_max - 1:
            new_array.append(any_array[ind])
        else:
            lower = int((ind+inds[i-1])/2)
            upper = int((ind+inds[i+1])/2)
            new_array.append(np.nanmean(any_array[lower:upper]))
    return np.array(new_array)

##############################################################################

def interpolate_preserve_old_points_fix(x_old, new_length):
    total_new_points = new_length - len(x_old)
    n_intervals = len(x_old) - 1
    points_per_interval = total_new_points // n_intervals
    remainder = total_new_points % n_intervals
    x_new = []
    for i in range(n_intervals):
        count = (remainder + points_per_interval) if i == 0 else points_per_interval
        segment = np.linspace(x_old[i], x_old[i+1], count+2)
        x_new.extend(segment[:-1] if i < n_intervals - 1 else segment)
    return np.sort(np.array(x_new))

##############################################################################

def interp2_180(x_array, y_array, n_levels):
    x_new = interpolate_preserve_old_points_fix(x_array, n_levels)
    f = interp1d(x_array, y_array, kind='linear', fill_value="extrapolate")
    return x_new, f(x_new)

##############################################################################

def check_lwp_iwv(lwp, iwv):
    if isinstance(lwp, np.ndarray): lwp = np.nan
    elif lwp < 0: lwp = 0.
    if isinstance(iwv, np.ndarray): iwv = np.nan
    elif iwv < 0: iwv = 0.
    return lwp, iwv

##############################################################################
# Radiosonde reading — auto-detect .nc vs .txt (unverändert aus deinem Modul,
# nur crop-Logik leicht generalisiert)
##############################################################################

def read_radiosonde_nc(file, min_p, n_levels, crop=0):
    ds = xr.open_dataset(file)

    if "Height" in ds.data_vars:
        height_var, t_var, p_var, h_var, p_factor = "Height", "Temperature", "Pressure", "Humidity", 1.
        deg_lat = ds["Latitude"].values[0]; deg_lon = ds["Longitude"].values[0]
        height_in_km = ds[height_var].values[0] / 1000
    else:
        height_var = "zg" if "zg" in ds.data_vars else "zsl"
        t_var, p_var, h_var, p_factor = "ta", "pa", "hur", 100.
        deg_lat = ds["lat"].values[0]; deg_lon = ds["lon"].values[0]
        if "zsl_start" in ds.data_vars:
            height_in_km = ds["zsl_start"].values / 1000
        else:
            height_in_km = ds["zsl"].values[0] / 1000

    max_index = np.nanargmax(ds[height_var].values)
    if ds[p_var].values[max_index] / p_factor < min_p:
        max_index = np.nanargmin(np.abs(ds[p_var].values[:max_index]/p_factor - min_p))
    index3000 = np.nanargmin(abs(ds[height_var].values[:max_index] - 3000))

    if crop > 0:
        crop = np.nanargmin(abs(ds[height_var].values - 132))
    else:
        old_h = ds[height_var].values[0]
        for i in range(1000):
            current_h = ds[height_var].values[i]
            if abs(current_h - old_h) < 2.:
                if i != 0: crop += 1
            else:
                break
            old_h = current_h

    if max_index < 300 or np.nanmax(ds[height_var].values) < 10000:
        return _nan_profile(n_levels)

    increment_bl = int(np.ceil((index3000-crop)/datapoints_bl))
    increment_ft = int(np.ceil((max_index-index3000)/datapoints_ft))
    inds = np.unique(np.r_[crop:index3000:increment_bl, index3000:max_index:increment_ft])

    time_dim = "Time" if "hur" in ds.data_vars else "time"
    z_array = ds[height_var].isel({time_dim: inds}).values
    t_array = running_mean_from_arrays(inds, ds[height_var].values, ds[t_var].values)
    p_array = running_mean_from_arrays(inds, ds[height_var].values, ds[p_var].values/p_factor)
    rh      = running_mean_from_arrays(inds, ds[height_var].values, ds[h_var].values)
    if np.all(rh <= 1.5):
        rh = rh * 100
    length_value = len(t_array)

    m_array, ppmv_array = [], []
    for rh_lev, t_lev, p_lev in zip(rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
        ppmv_array.append(rh2ppmv(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    m_array, ppmv_array = np.array(m_array), np.array(ppmv_array)

    return length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat, m_array, z_array, rh, deg_lon

##############################################################################

def read_radiosonde_txt(file, min_p, n_levels):
    df = pd.read_table(file, encoding_errors="ignore", engine='python', skiprows=20,
        skipfooter=10, header=None, names=["Time","P","T","Hu","Ws","Wd",
        "Long.","Lat.","Alt","Geopot","Rs","Elevation","Azimuth","Range"])

    max_index = np.nanargmax(df["Alt"].values)
    if df["P"].values[max_index] < min_p:
        max_index = np.nanargmin(np.abs(df["P"].values[:max_index] - min_p))
    index3000 = np.nanargmin(abs(df["Alt"].values[:max_index] - 3000))

    crop = 0
    old_h = df["Alt"].values[0]
    for i in range(1000):
        current_h = df["Alt"].values[i]
        if abs(current_h - old_h) < 2.:
            if i != 0: crop += 1
        else:
            break
        old_h = current_h

    if max_index < 300 or np.nanmax(df["Alt"].values) < 10000:
        return _nan_profile(n_levels)

    increment_bl = int(np.ceil((index3000-crop)/datapoints_bl))
    increment_ft = int(np.ceil((max_index-index3000)/datapoints_ft))
    inds = np.unique(np.r_[crop:index3000:increment_bl, index3000:max_index:increment_ft])

    z_array = df["Alt"].iloc[inds].values
    t_array = running_mean_from_arrays(inds, df["Alt"].values, df["T"].values + 273.15)
    p_array = running_mean_from_arrays(inds, df["Alt"].values, df["P"].values)
    rh      = running_mean_from_arrays(inds, df["Alt"].values, df["Hu"].values)
    length_value = len(t_array)
    if np.all(rh <= 1.5):
        rh = rh * 100

    m_array, ppmv_array = [], []
    for rh_lev, t_lev, p_lev in zip(rh, t_array, p_array):
        m_array.append(rh2mixing_ratio(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
        ppmv_array.append(rh2ppmv(RH=rh_lev, abs_T=t_lev, p=p_lev*100))
    m_array, ppmv_array = np.array(m_array), np.array(ppmv_array)

    height_in_km = df["Alt"].values[0] / 1000
    return (length_value, p_array, t_array, ppmv_array, height_in_km,
            df["Lat."].values[0], m_array, z_array, rh, df["Long."].values[0])

##############################################################################

def _nan_profile(n_levels):
    return (0, [np.nan]*n_levels, [np.nan]*n_levels, [np.nan]*n_levels,
            float('nan'), float('nan'), [np.nan]*n_levels, [np.nan]*n_levels,
            [np.nan]*n_levels, float('nan'))

##############################################################################

def read_radiosonde(file, min_p, n_levels):
    if file.endswith(".nc"):
        return read_radiosonde_nc(file, min_p, n_levels)
    elif "Profile.txt" in file or file.endswith(".txt"):
        return read_radiosonde_txt(file, min_p, n_levels)
    else:
        raise ValueError(f"Unrecognized radiosonde file format: {file}")

##############################################################################

def add_clim2profiles(p_array, t_array, ppmv_array, m_array, z_array, rh, min_p):
    p_index = np.nanargmin(p_array)
    wv_min = np.nanmin(ppmv_array)
    candidates = np.where(ppmv_array <= 2*wv_min)[0]
    wv_index = candidates[-1]
    z_index = np.nanargmax(z_array)
    thres_idx = np.nanmin([p_index, wv_index, z_index])
    p_threshold = max(p_array[thres_idx], min_p)
    if p_threshold > 200:
        p_threshold = 200

    z, p, d, t, md = atmp.gl_atm(atm=1)
    gkg = ppmv2gkg(md[:, atmp.H2O], atmp.H2O)
    rhs_clim = mr2rh(p, t, gkg)[0]

    if np.all(np.isnan(p_array)) and np.all(np.isnan(t_array)):
        mask_clim = np.zeros_like(p, dtype=bool)
        mask_rs   = np.ones_like(p_array, dtype=bool)
    else:
        mask_clim = p_threshold > np.array(p)
        mask_rs   = np.array(p_array) > p_threshold

    p_array = np.concatenate([p_array[mask_rs], np.array(p)[mask_clim]])
    t_array = np.concatenate([t_array[mask_rs], np.array(t)[mask_clim]])
    m_array = np.concatenate([m_array[mask_rs], np.array(gkg)[mask_clim]/1000])
    z_array = np.concatenate([z_array[mask_rs], np.array(z*1000)[mask_clim]])
    rh      = np.concatenate([rh[mask_rs], np.array(rhs_clim)[mask_clim]])

    ppmv_array = np.array([rh2ppmv(RH=r, abs_T=t, p=p*100)
                          for r, t, p in zip(rh, t_array, p_array)])

    return (p_array[::-1], t_array[::-1], ppmv_array[::-1],
            m_array[::-1], z_array[::-1], rh[::-1])

##############################################################################
# Generic MWR reading — auto-detects file naming convention in mwr_dir
##############################################################################

def nearest_ele4elevation_mean(ele_values, azi_values, ele_times, target_elevation,
                               target_azi, datetime_np,
                               min_time_diff_thres=min_time_diff_thres,
                               max_elev_azi_diff=max_elev_azi_diff):
    match_mask = (abs(ele_values - target_elevation) < max_elev_azi_diff)
    match_mask2 = ([True]*len(ele_values) if target_azi == "ANY"
                   else (abs(azi_values - target_azi) < max_elev_azi_diff))
    final_mask = match_mask & match_mask2
    if not final_mask.any():
        return None
    candidate_times = ele_times[final_mask]
    time_diffs = np.abs(candidate_times - datetime_np)
    minutes_diff_all = time_diffs.astype('timedelta64[s]').astype(float) / 60
    valid_mask = minutes_diff_all <= min_time_diff_thres
    if not valid_mask.any():
        return None
    return np.where(final_mask)[0][valid_mask]

##############################################################################

def time_indices_list4BL(ds_bl, datetime_np, min_time_diff_thres=min_time_diff_thres):
    time_diffs = np.abs(ds_bl["time"].values - datetime_np) / np.timedelta64(1, 's')
    idx = np.where(time_diffs <= min_time_diff_thres*60)[0]
    return idx.tolist() if idx.size else None

##############################################################################

def derive_elevation_index(ds_bl, elevation):
    for index, ele in enumerate(ds_bl["ele"].values):
        if abs(ele - elevation) < 0.05:
            return index
    return None

##############################################################################

def get_tbs_from_mwr_dir(mwr_dir, datestring, datetime_np,
                         elevations=elevations, azimuths=azimuths):
    """
    Auto-detects filenaming convention within mwr_dir and extracts TBs at
    the given date/time, matched by elevation/azimuth.
    Supported conventions (auto-detected by substring in filename):
      - '1C01'         : scanning TB dataset with 'elevation_angle'/'azimuth_angle'
      - 'BL' + 'flag'  : boundary-layer scan format with 'ele'/'azi'/'tb'/'flag'
      - fallback       : generic 'ele'/'azi'/'tb'/'flag' dataset
    """
    tbs = np.full((len(elevations), len(azimuths), 14), np.nan)
    lat, lon, qual_flag = np.nan, np.nan, 0.

    all_files = glob.glob(os.path.join(mwr_dir, "**", "*.nc"), recursive=True)
    files = [f for f in all_files if datestring in os.path.basename(f)]

    for file in files:
        fname = os.path.basename(file)

        if "BL" in fname:
            ds_bl = xr.open_dataset(file)
            for i, elevation in enumerate(elevations):
                ele_index = derive_elevation_index(ds_bl, elevation)
                if ele_index is None:
                    continue
                idx_list = time_indices_list4BL(ds_bl, datetime_np)
                if idx_list:
                    for ch in range(14):
                        tbs[i, 0, ch] = np.nanmean(ds_bl["tb"].values[idx_list, ele_index, ch])
                    qual_flag = np.nanmean(ds_bl["flag"].values[idx_list])

        elif "1C01" in fname:
            ds_c1 = xr.open_dataset(file)
            for i, elevation in enumerate(elevations):
                for j, azi in enumerate(azimuths):
                    idx_list = nearest_ele4elevation_mean(
                        ds_c1["elevation_angle"].values, ds_c1["azimuth_angle"].values,
                        ds_c1["time"].values, elevation, azi, datetime_np)
                    if idx_list is not None and len(idx_list) > 0:
                        for ch in range(14):
                            tbs[i, j, ch] = np.nanmean(ds_c1["tb"].values[idx_list, ch])
                        qual_flag = np.nanmean(ds_c1["quality_flag"].values[idx_list, :])
            if "latitude" in ds_c1.data_vars:
                lat, lon = ds_c1["latitude"].values[0], ds_c1["longitude"].values[0]
            elif "lat" in ds_c1.data_vars:
                lat, lon = ds_c1["lat"].values, ds_c1["lon"].values

        elif all(v in xr.open_dataset(file).data_vars for v in ["ele", "azi", "tb"]):
            ds_mwr = xr.open_dataset(file)
            for i, elevation in enumerate(elevations):
                for j, azi in enumerate(azimuths):
                    idx_list = nearest_ele4elevation_mean(
                        ds_mwr["ele"].values, ds_mwr["azi"].values,
                        ds_mwr["time"].values, elevation, azi, datetime_np)
                    if idx_list is not None and len(idx_list) > 0:
                        for ch in range(14):
                            tbs[i, j, ch] = np.nanmean(ds_mwr["tb"].values[idx_list, ch])
                        if "flag" in ds_mwr.data_vars:
                            qual_flag = np.nanmean(ds_mwr["flag"].values[idx_list])
            if "latitude" in ds_mwr.data_vars:
                lat, lon = ds_mwr["latitude"].values[0], ds_mwr["longitude"].values[0]
            elif "lat" in ds_mwr.data_vars:
                lat, lon = ds_mwr["lat"].values, ds_mwr["lon"].values

    return tbs, lat, lon, qual_flag

##############################################################################

def get_profile_from_mwr_dir(mwr_dir, datestring, datetime_np, n_levels, height_offset=0.0):
    """Generic L2 profile reader — same logic as get_profs_from_l2, single dir."""
    data = np.full((4, n_levels), np.nan)
    lwp, iwv = np.nan, np.nan

    all_files = glob.glob(os.path.join(mwr_dir, "**", "*.nc"), recursive=True)
    files = [f for f in all_files if datestring in os.path.basename(f)]

    for file in files:
        fname = os.path.basename(file)

        if "single" in fname:
            ds = xr.open_dataset(file)
            idx_list = nearest_ele4elevation_mean(
                ds["elevation_angle"].values, ds["azimuth_angle"].values,
                ds["time"].values, 90., "ANY", datetime_np)
            if idx_list is not None and len(idx_list) > 0:
                mean_t = np.nanmean(ds["temperature"].values[idx_list, :], axis=0)
                x_new, y_new = interp2_180(ds["height"].values, mean_t, n_levels)
                data[0, :], data[1, :] = x_new, y_new
                mean_h = np.nanmean(ds["absolute_humidity"].values[idx_list, :], axis=0)
                _, y_new = interp2_180(ds["height"].values, mean_h, n_levels)
                data[3, :] = y_new
                lwp = np.nanmean(ds["lwp"].values[idx_list])
                iwv = np.nanmean(ds["iwv"].values[idx_list])

        elif "mwr0" in fname and "_l2_ta_" in fname:
            ds = xr.open_dataset(file)
            idx_list = nearest_ele4elevation_mean(
                ds["ele"].values, ds["azi"].values, ds["time"].values,
                90., "ANY", datetime_np)
            if idx_list is not None and len(idx_list) > 0:
                mean_t = np.nanmean(ds["ta"].values[idx_list, :], axis=0)
                x_new, y_new = interp2_180(ds["height"].values, mean_t, n_levels)
                data[0, :], data[1, :] = x_new, y_new

        elif "mwrBL0" in fname and "_l2_ta_" in fname:
            ds = xr.open_dataset(file)
            idx_list = time_indices_list4BL(ds, datetime_np)
            if idx_list:
                mean_bl = np.nanmean(ds["ta"].values[idx_list, :], axis=0)
                _, y_new = interp2_180(ds["height"].values, mean_bl, n_levels)
                data[2, :] = y_new

        elif "_hua_" in fname:
            ds = xr.open_dataset(file)
            idx_list = time_indices_list4BL(ds, datetime_np)
            if idx_list:
                mean_hua = np.nanmean(ds["hua"].values[idx_list, :], axis=0)
                _, y_new = interp2_180(ds["height"].values, mean_hua, n_levels)
                data[3, :] = y_new

        elif "_prw_" in fname:
            ds = xr.open_dataset(file)
            idx_list = time_indices_list4BL(ds, datetime_np)
            if idx_list:
                iwv = np.nanmean(ds["prw"].values[idx_list])

        elif "_clwvi_" in fname:
            ds = xr.open_dataset(file)
            idx_list = time_indices_list4BL(ds, datetime_np)
            if idx_list:
                lwp = np.nanmean(ds["clwvi"].values[idx_list])

    data[0, :] = data[0, :] + height_offset
    lwp, iwv = check_lwp_iwv(lwp, iwv)
    return data[:, ::-1], lwp, iwv

##############################################################################
# Main processing loop for one site/campaign
##############################################################################

def derive_date_from_rs_filename(file):

    if "vitII" in file:
        string = os.path.basename(file).split(".")[0]
        return np.datetime64(f"{string[24:28]}-{string[28:30]}-{string[30:32]}T"
                             f"{string[32:34]}:{string[34:36]}:{string[36:38]}")
    elif ".nc" in file:
        string = os.path.basename(file).split(".")[0]
        return np.datetime64(f"{string[:4]}-{string[4:6]}-{string[6:8]}T"
                             f"{string[9:11]}:{string[11:13]}:{string[13:15]}")
    elif "Profile.txt" in file:
        string = os.path.basename(file).split(".")[0]
        return np.datetime64(f"{string[:4]}-{string[4:6]}-{string[6:8]}T"
                             f"{string[8:10]}:{string[10:12]}:{string[12:14]}")
    else:
        raise ValueError(f"Cannot derive date from filename: {file}")

##############################################################################

def process_site(rs_dir, mwr_dir, campaign, location, n_levels, min_p, height_offset):
    rs_files = sorted(glob.glob(os.path.join(rs_dir, "*.nc")) +
                      glob.glob(os.path.join(rs_dir, "**", "*Profile.txt"), recursive=True))
    n = len(rs_files)
    print(f"Found {n} radiosonde files in {rs_dir}")

    profile_indices = []
    srf_pressures, srf_temps, srf_wvs, srf_altitude = (np.full(n, np.nan) for _ in range(4))
    tbs_all       = np.full((n, len(elevations), len(azimuths), 14), np.nan)
    mwr_profiles  = np.full((n, 4, n_levels), np.nan)
    qual_flags    = np.full(n, np.nan)
    lwps_rs       = np.full(n, np.nan)
    lwps_mwr      = np.full(n, np.nan)
    iwvs_mwr      = np.full(n, np.nan)
    level_pressures    = np.full((n_levels, n), np.nan)
    level_temperatures = np.full((n_levels, n), np.nan)
    level_wvs          = np.full((n_levels, n), np.nan)
    level_ppmvs        = np.full((n_levels, n), np.nan)
    level_liq          = np.full((n_levels, n), np.nan)
    level_ice          = np.full((n_levels, n), np.nan)
    level_z            = np.full((n_levels, n), np.nan)
    level_rhs          = np.full((n_levels, n), np.nan)
    times = np.full(n, np.nan)
    lats  = np.full(n, np.nan)
    lons  = np.full(n, np.nan)

    for i, file in enumerate(rs_files):
        print(f"[{i+1}/{n}] {file}")
        profile_indices.append(i)
        datetime_np = derive_date_from_rs_filename(file)
        times[i] = datetime_np
        datestring = str(datetime_np).replace("T", "").replace(":", "").replace("-", "")[:8]

        tbs, lat, lon, qual_flag = get_tbs_from_mwr_dir(mwr_dir, datestring, datetime_np)
        mwr_prof, lwp_mwr, iwv_mwr = get_profile_from_mwr_dir(
            mwr_dir, datestring, datetime_np, n_levels, height_offset)

        tbs_all[i] = tbs
        mwr_profiles[i] = mwr_prof
        qual_flags[i] = qual_flag
        lwps_mwr[i] = lwp_mwr
        iwvs_mwr[i] = iwv_mwr

        length_value, p_array, t_array, ppmv_array, height_in_km, deg_lat, \
            m_array, z_array, rh, deg_lon = read_radiosonde(file, min_p, n_levels)

        if length_value < 150:
            continue

        p_array, t_array, ppmv_array, m_array, z_array, rh = add_clim2profiles(
            p_array, t_array, ppmv_array, m_array, z_array, rh, min_p)

        lwc_kg_m3, lwc_kg_kg, lwp_kg_m2, iwc_kg_m3, iwc_kg_kg, iwp_kg_m2 = \
            derive_cloud_features(p_array, t_array, ppmv_array, m_array, z_array, rh)

        lwps_rs[i] = lwp_kg_m2
        lats[i], lons[i] = deg_lat, deg_lon
        level_pressures[:, i]    = p_array[-n_levels:]
        level_temperatures[:, i] = t_array[-n_levels:]
        level_wvs[:, i]          = m_array[-n_levels:] * 1000
        level_ppmvs[:, i]        = ppmv_array[-n_levels:]
        level_liq[:, i]          = lwc_kg_kg[-n_levels:]
        level_ice[:, i]          = iwc_kg_kg[-n_levels:]
        level_z[:, i]            = z_array[-n_levels:]
        level_rhs[:, i]          = rh[-n_levels:]
        srf_pressures[i] = p_array[-1]
        srf_temps[i]     = t_array[-1]
        srf_wvs[i]       = m_array[-1]
        srf_altitude[i]  = height_in_km

    ds = xr.Dataset(
        data_vars={
            "TBs":              (("time","elevation","azimuth","N_Channels"), tbs_all),
            "MWR_z":            (("time","N_Levels"), mwr_profiles[:,0,:]),
            "MWR_ta":           (("time","N_Levels"), mwr_profiles[:,1,:]),
            "MWR_hua":          (("time","N_Levels"), mwr_profiles[:,3,:]),
            "MWR_IWV":          (("time",), iwvs_mwr),
            "MWR_LWP":          (("time",), lwps_mwr),
            "qual_flag":        (("time",), qual_flags),
            "Level_Pressure":    (("N_Levels","time"), level_pressures),
            "Level_Temperature": (("N_Levels","time"), level_temperatures),
            "Level_H2O":         (("N_Levels","time"), level_wvs),
            "Level_ppmvs":       (("N_Levels","time"), level_ppmvs),
            "Level_Liquid":      (("N_Levels","time"), level_liq),
            "Level_Ice":         (("N_Levels","time"), level_ice),
            "Level_z":           (("N_Levels","time"), level_z),
            "Level_RH":          (("N_Levels","time"), level_rhs),
            "LWP_radiosonde":    (("time",), lwps_rs),
            "Surface_Pressure":  (("time",), srf_pressures),
            "Temperature_2M":    (("time",), srf_temps),
            "H2O_2M":            (("time",), srf_wvs),
            "Surface_Altitude":  (("time",), srf_altitude),
            "Profile_Index":     (("time",), profile_indices),
            "Campaign":          (("time",), [campaign]*n),
            "Location":          (("time",), [location]*n),
            "Latitude":          (("time",), lats),
            "Longitude":         (("time",), lons),
        },
        coords={
            "N_Channels": np.arange(14),
            "time": times,
            "N_Levels": np.arange(n_levels),
            "elevation": elevations,
            "azimuth": azimuths,
        }
    )
    return ds

##############################################################################
# Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()

    ds_new = process_site(
        rs_dir=args.rs_dir,
        mwr_dir=args.mwr_dir,
        campaign=args.campaign,
        location=args.location,
        n_levels=args.n_levels,
        min_p=args.min_p,
        height_offset=args.height_offset,
    )

    if args.append is not None and os.path.exists(args.append):
        print(f"Appending to existing dataset: {args.append}")
        ds_old = xr.open_dataset(args.append)
        combined = xr.concat([ds_old, ds_new], dim="time")
        combined = combined.sortby("time")   # ← garantiert korrekte Zeitreihenfolge
        ds_old.close()
    else:
        combined = ds_new

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    combined.to_netcdf(args.output, format="NETCDF4")
    print(f"Saved: {args.output}  (total profiles: {combined.sizes['time']})")




#################################

#[1/14] /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_sin/vitII_sin_rs41_comp_v00_20260624043519.nc
#Traceback (most recent call last):
#  File "/usr/lib/python3/dist-packages/xarray/core/dataset.py", line 1446, in _construct_dataarray
#    variable = self._variables[name]
#KeyError: 'zsl_start'


############################
