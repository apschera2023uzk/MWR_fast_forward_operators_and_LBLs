#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import xarray as xr
import numpy as np
import glob
import os
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib

##############################################################################
# 1.5 Parameters:
##############################################################################
plt.style.use('seaborn-poster')
matplotlib.use("Qt5Agg")

# Clear sky LWP threshold
thres_lwp=0.005 # kg m-2 fitting with Moritz' threshold of 5 g m-2
n_chans=14
model_tbs=["TBs_PyRTlib_R24",'TBs_RTTOV_gb', 'TBs_ARMS_gb']
mwr_vars = ['TBs_dwdhat', 'TBs_sunhat', 'TBs_tophat',\
        'TBs_joyhat'] #, 'TBs_foghat', 'TBs_hamhat']
ref_label = "PyRTlib R24 LBL"
n_elev = 10
elevations = np.array([90., 30, 19.2, 14.4, 11.4, 8.4,  6.6,  5.4, 4.8,  4.2])

# Cloud_data NN by sites:
rao = "/home/aki/PhD_data/retrievals_final_ele/RAO_*.nc"
vit1_joy = "/home/aki//PhD_data/retrievals_final_ele/Vit1_JOY*.nc"

##############################################################################
# 2nd Used functions:
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Scatter plots of TB MWR against sondes e.g.."
    )
    parser.add_argument(
        "--NetCDF", "-nc",
        type=str,
        default=os.path.expanduser("~/PhD_data/TB_preproc_and_proc_results/3campaigns_3models_all_results.nc"),
        help="Input data"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/PhD_data/TB_preproc_and_proc_results/3campaigns_3models_all_results_and_stats.nc"),
        help="Output file with statistics!"
    )

    parser.add_argument(
        "--cloud", "-cl",
        type=str,
        default=os.path.expanduser("~/PhD_data/Alexander_Moritz_Exchange/retrievals_final_ele/*.nc"),
        help="Pattern of cloud_flag files!"
    )
    '''
    '''
    return parser.parse_args()

##############################################################################

def add_MLNN_cloud_info(ds, args, rao=rao, vit1=vit1_joy):

    # 1st for RAO:
    print("Rao: ",rao)
    cf_files = sorted(glob.glob(rao))
    ds_cloud = xr.open_mfdataset(cf_files)
    cf_da = ds_cloud["cloud_flag"].sel(time=slice("2021-05-01T00:00:00", "2021-08-31T00:00:00"))

    # Keep only cf_da times that fall within ds time range
    ds_tmin = ds["time"].values.min()
    ds_tmax = ds["time"].values.max()
    cf_da = cf_da.sel(time=slice(ds_tmin, ds_tmax))
    cf_da = cf_da.rename({"n_angle": "elevation"})

    # Interpolate cf_da onto ds time axis, fill outside range with NaN
    cf_interp = cf_da.reindex(time=ds["time"], method="nearest", tolerance="30min")
    ds["cloud_flag"] = cf_interp

    print("ds[location]", np.unique(ds["Location"].values))
    print("ds[location]", np.unique(ds["Campaign"].values))

    return ds

    # 1st just make a correction here, so that these cloud flag files are only applied to timesteps of the datasets where:
    # 1.1: Make sure campaign is: 'FESSTVaL'
    # 1.2: Make sure Location is: 'RAO_Lindenberg'

'''
def add_MLNN_cloud_info(ds, args, rao=rao, vit1_joy=vit1_joy):

    ###################
    # 1st RAO Lindenberg:
    cf_files = sorted(glob.glob(rao))
    ds_cloud = xr.open_mfdataset(cf_files)
    cf_da = ds_cloud["cloud_flag"].sel(time=slice("2021-05-01T00:00:00", "2021-08-31T00:00:00"))
    cf_da = cf_da.rename({"n_angle": "elevation"})

    # Interpoliere auf ds-Zeitachse — außerhalb des Bereichs → NaN:
    cf_interp = cf_da.reindex(time=ds["time"], method="nearest", tolerance="30min")

    # Maske: nur FESSTVaL & RAO_Lindenberg bekommt den MLNN-Flag
    valid_mask = (
        (ds["Campaign"].values == "FESSTVaL") &
        (ds["Location"].values == "RAO_Lindenberg")
    )

    # Set Cloudflag for RAO Lindenberg:
    ds["cloud_flag"] = cf_interp
    ds["cloud_flag"].values[:, ~valid_mask] = np.nan

    ###################
    # 2nd Vital I Joyhat:
    cf_files = sorted(glob.glob(vit1_joy))
    ds_cloud = xr.open_mfdataset(cf_files)

    print("ds_cloud: ", ds_cloud)

    # cf_da hat länge der Zeitdimension = 0
    #cf_da:  <xarray.DataArray 'cloud_flag' (elevation: 10, time: 0)> Size: 0B
    # WARUM????
    cf_da = ds_cloud["cloud_flag"].sel(time=slice("2024-08-01T00:00:00", "2024-08-31T00:00:00"))
    print("cf_da: ", cf_da)

    # Interpoliere auf ds-Zeitachse — außerhalb des Bereichs → NaN:
    cf_interp = cf_da.reindex(time=ds["time"], method="nearest", tolerance="30min")
    valid_mask = (
        (ds["Campaign"].values == 'Vital I') &
        (ds["Location"].values == 'JOYCE')
    )
    print("cf_interp: ", cf_interp)

    ds["cloud_flag"].values[:, valid_mask] = cf_interp.values[:, valid_mask]


    # Unfortunately this part still does not work, and I cannot repeat the trick from before...
    # Initialisiere cloud_flag komplett mit NaN:
    #cloud_flag_full = cf_interp.copy() * np.nan    
    #cloud_flag_full.values[:, valid_mask] = cf_interp.values[:, valid_mask]
    #ds["cloud_flag"] = cloud_flag_full


    ##################
    # Plots:
    plt.figure()
    plt.plot(cf_interp.values)
    plt.show()


    plt.figure()
    plt.plot(valid_mask)
    plt.show()

    plt.figure()
    plt.plot(ds["time"], ds["cloud_flag"].mean(dim="elevation"))
    plt.show()

    ###################
    # 3rd :
    #cf_files = sorted(glob.glob(vit1_joy))
    #ds_cloud = xr.open_mfdataset(cf_files)
    #cf_da = ds_cloud["cloud_flag"].sel(time=slice("2021-05-01T00:00:00", "2021-08-31T00:00:00"))

    # Interpoliere auf ds-Zeitachse — außerhalb des Bereichs → NaN:
    #cf_interp = cf_da.reindex(time=ds["time"], method="nearest", tolerance="30min")

    # Maske: nur FESSTVaL & RAO_Lindenberg bekommt den MLNN-Flag
    #valid_mask = (
    #    (ds["Campaign"].values == 'Vital I') &
    #    (ds["Location"].values == 'JOYCE')
    #)

    # Initialisiere cloud_flag komplett mit NaN:
    #cloud_flag_full = cf_interp.copy() * np.nan
    #cloud_flag_full.values[valid_mask] = cf_interp.values[valid_mask]
    #ds["cloud_flag"] = cloud_flag_full


# ds[location] ['Falkenberg' 'JOYCE' 'RAO_Lindenberg']
# ds[location] ['FESSTVaL' 'Socles' 'Vital I']

    return ds
'''
##############################################################################

def get_liquid_flag(ds, rs_lwp_thrs_kg_m2=0.2):
    """
    Returns a (time,) float array: 1.0=cloudy where
    LWP_radiosonde (Crop=0) > thres_liquid.
    Returns None if LWP_radiosonde not in ds.
    """
    if "LWP_radiosonde" not in ds:
        print("Info: LWP_radiosonde nicht im Dataset — kein Liquid-Filter.")
        return None

    lwp = ds["LWP_radiosonde"].isel(Crop=0)  # (time,)
    return (lwp.values > rs_lwp_thrs_kg_m2).astype(float)

##############################################################################

def add_cloud_flag(ds, args, thres_lwp=thres_lwp):
    """
    Adds 'cloud_flag' (1=cloudy, 0=clear, never NaN) to ds.
    Priority: MLNN cloud flag → LWP-based fallback for NaNs.
    """
    # ── 1. Get MLNN cloud flag (boolean, may contain NaN where not covered) ──
    ds = add_MLNN_cloud_info(ds, args)
    mlnn_flag = ds["cloud_flag"].values.astype(float)  # True→1.0, False→0.0, NaN→NaN

    # ── 2. Build LWP-based fallback flag for every timestep ──────────────────
    lwp_flag = np.zeros(len(ds["time"]), dtype=float)

    for i in range(len(ds["time"].values)):
        water_sum = np.nanmean(np.array([
            np.nansum(np.nan_to_num(ds["Dwdhat_LWP"].values[i])),
            np.nansum(np.nan_to_num(ds["Foghat_LWP"].values[i])),
            np.nansum(np.nan_to_num(ds["Sunhat_LWP"].values[i])),
            np.nansum(np.nan_to_num(ds["Tophat_LWP"].values[i])),
            np.nansum(np.nan_to_num(ds["Joyhat_LWP"].values[i])),
            np.nansum(np.nan_to_num(ds["Hamhat_LWP"].values[i]))
        ]))        
        lwp_flag[i] = 1.0 if water_sum > thres_lwp else 0.0
    lwp_flag_2d = np.tile(lwp_flag, (mlnn_flag.shape[0], 1))

    # ── 3. MLNN dominant; fill NaNs with LWP-based flag ──────────────────────
    nan_mask      = np.isnan(mlnn_flag)
    combined_flag = mlnn_flag.copy()
    combined_flag[nan_mask] = lwp_flag_2d[nan_mask]

    # ── 3b. Level_Liquid override: setze cloudy wo Liquid-Summe > threshold ──
    liquid_flag = get_liquid_flag(ds)
    if liquid_flag is not None:
        # liquid_flag ist (time,) → auf (elevation, time) broadcasten
        liquid_flag_2d = np.tile(liquid_flag, (combined_flag.shape[0], 1))
        combined_flag = np.where(liquid_flag_2d == 1.0, 1.0, combined_flag)

    # ── 4. Store as int (0/1, never NaN) and return ───────────────────────────
    ds["cloud_flag"] = xr.DataArray(
        np.transpose(combined_flag).astype(int),
        dims=["time", "elevation"],
        coords={
            "time":      ds["time"],
            "elevation": ds["elevation"],
        },
        attrs={
            "long_name": "Cloud flag (MLNN primary, LWP + Level_Liquid fallback)",
            "flag_values": "0, 1",
            "flag_meanings": "clear cloudy",
        }
    )

    return ds

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    nc_out_path=args.NetCDF
    
    # Open dataset and clear sky filtering
    ds = xr.open_dataset(nc_out_path)
    ds_write = add_cloud_flag(ds, args)
    
    # Add deviations:
    # 1st for models:
# Add deviations:
    # 1st for models:
    ds_write["Deviations_RTTOV_R24"] = (ds_write["TBs_RTTOV_gb"].isel(Crop=0) - ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze()
    ds_write["Deviations_RTTOV_R24"].attrs["var_label"] = "TBs_RTTOV_gb"
    ds_write["Deviations_RTTOV_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_ARMS_R24"]  = (ds_write["TBs_ARMS_gb"].isel(Crop=0)  - ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze()
    ds_write["Deviations_ARMS_R24"].attrs["var_label"] = "TBs_ARMS_gb"
    ds_write["Deviations_ARMS_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_ARMS2_R24"]  = (ds_write["TBs_ARMS_gb2"].isel(Crop=0)  - ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze()
    ds_write["Deviations_ARMS2_R24"].attrs["var_label"] = "TBs_ARMS_gb2"
    ds_write["Deviations_ARMS2_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_ARMS_ARMS2"]  = (ds_write["TBs_ARMS_gb"].isel(Crop=0)  - ds_write["TBs_ARMS_gb2"].isel(Crop=0)).squeeze()
    ds_write["Deviations_ARMS_ARMS2"].attrs["var_label"] = "TBs_ARMS_gb"
    ds_write["Deviations_ARMS_ARMS2"].attrs["ref_label"] = "TBs_ARMS_gb2"
    ds_write["Deviations_R17_R24"]  = (ds_write["TBs_PyRTlib_R17"].isel(Crop=0)  - ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze()
    ds_write["Deviations_R17_R24"].attrs["var_label"] = "TBs_PyRTlib_R17"
    ds_write["Deviations_R17_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_R98_R24"]  = (ds_write["TBs_PyRTlib_R98"].isel(Crop=0)  - ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze()
    ds_write["Deviations_R98_R24"].attrs["var_label"] = "TBs_PyRTlib_R98"
    ds_write["Deviations_R98_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"

    # 2nd for MWRs:
    ds_write["Deviations_dwdhat_R24"] = (ds_write["TBs_dwdhat"] -\
        ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze().transpose("time",\
        "N_Channels", "elevation", ...)
    ds_write["Deviations_dwdhat_R24"].attrs["var_label"] = "TBs_dwdhat"
    ds_write["Deviations_dwdhat_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_foghat_R24"] = (ds_write["TBs_foghat"] -\
        ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze().transpose("time",\
        "N_Channels", "elevation", "azimuth", ...)
    ds_write["Deviations_foghat_R24"].attrs["var_label"] = "TBs_foghat"
    ds_write["Deviations_foghat_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_sunhat_R24"] = (ds_write["TBs_sunhat"] -\
        ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze().transpose("time",\
        "N_Channels", "elevation", ...)
    ds_write["Deviations_sunhat_R24"].attrs["var_label"] = "TBs_sunhat"
    ds_write["Deviations_sunhat_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_tophat_R24"] = (ds_write["TBs_tophat"] -\
        ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze().transpose("time",\
        "N_Channels", "elevation", ...)
    ds_write["Deviations_tophat_R24"].attrs["var_label"] = "TBs_tophat"
    ds_write["Deviations_tophat_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_hamhat_R24"] = (ds_write["TBs_hamhat"] -\
        ds_write["TBs_PyRTlib_R24"].isel(Crop=0)).squeeze().transpose("time",\
        "N_Channels", "elevation", ...)
    ds_write["Deviations_hamhat_R24"].attrs["var_label"] = "TBs_hamhat"
    ds_write["Deviations_hamhat_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    ds_write["Deviations_joyhat_R24"] = (ds_write["TBs_joyhat"] -\
        ds_write["TBs_PyRTlib_R24"].isel(Crop=1)).squeeze().transpose("time",\
        "N_Channels", "elevation","azimuth", ...)
    ds_write["Deviations_joyhat_R24"].attrs["var_label"] = "TBs_joyhat"
    ds_write["Deviations_joyhat_R24"].attrs["ref_label"] = "TBs_PyRTlib_R24"
    # 3rd FRTMs against MWRs:
    ds_write["Deviations_RTTOV_dwdhat"] = (ds_write["TBs_RTTOV_gb"].isel(Crop=0) - ds_write["TBs_dwdhat"]).squeeze()
    ds_write["Deviations_RTTOV_dwdhat"].attrs["var_label"] = "TBs_RTTOV_gb"
    ds_write["Deviations_RTTOV_dwdhat"].attrs["ref_label"] = "TBs_dwdhat"
    ds_write["Deviations_ARMS_dwdhat"]  = (ds_write["TBs_ARMS_gb"].isel(Crop=0)  - ds_write["TBs_dwdhat"]).squeeze()
    ds_write["Deviations_ARMS_dwdhat"].attrs["var_label"] = "TBs_ARMS_gb"
    ds_write["Deviations_ARMS_dwdhat"].attrs["ref_label"] = "TBs_dwdhat"
    ds_write["Deviations_RTTOV_joyhat"] = (ds_write["TBs_RTTOV_gb"].isel(Crop=1) - ds_write["TBs_joyhat"]).squeeze()
    ds_write["Deviations_RTTOV_joyhat"].attrs["var_label"] = "TBs_RTTOV_gb"
    ds_write["Deviations_RTTOV_joyhat"].attrs["ref_label"] = "TBs_joyhat"
    ds_write["Deviations_ARMS_joyhat"]  = (ds_write["TBs_ARMS_gb"].isel(Crop=1)  - ds_write["TBs_joyhat"]).squeeze()
    ds_write["Deviations_ARMS_joyhat"].attrs["var_label"] = "TBs_ARMS_gb"
    ds_write["Deviations_ARMS_joyhat"].attrs["ref_label"] = "TBs_joyhat"
    print(ds_write.data_vars)
    
    ###
    # Write new file
    ds_write.to_netcdf(args.output)











