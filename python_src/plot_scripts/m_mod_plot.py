#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import xarray as xr
import pandas as pd
import numpy as np
import glob
from scipy.stats import pearsonr
import os
from scipy.interpolate import interp1d
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.image as mpimg
from PIL import Image
import matplotlib.colors as colors

##############################################################################

def select_ds_camp_loc(ds, campaign, location):
    # Filter Datensatz nach Kampagne und Ort
    mask = (ds["Campaign"] == campaign) & (ds["Location"] == location)
    ds_sel = ds.sel(time=ds["time"].values[mask.values])
    return ds_sel

##############################################################################

def ensure_folder_exists(base_path, folder_name):
    # Join the base path with the folder name
    folder_path = os.path.join(base_path, folder_name)
    # Create the directory if it does not exist
    os.makedirs(folder_path, exist_ok=True)

    return os.path.abspath(folder_path)

##############################################################################

def apply_sky_mask(ds_sel, sky):
    if sky == "all_sky":
        return ds_sel

    cf = ds_sel["cloud_flag"]  # (time, elevation)

    if sky == "clear":
        bad_mask_elev = (cf == 1)   # (time, elevation): cloudy cells
    else:  # cloudy
        bad_mask_elev = (cf == 0)   # (time, elevation): clear cells

    # Timestep is dropped if ALL elevations are bad:
    bad_mask_time = bad_mask_elev.all(dim="elevation")  # (time,)
    good_time     = ~bad_mask_time

    # 1. Drop fully-bad timesteps from entire dataset (all vars, incl. no-elevation vars):
    ds_cf = ds_sel.isel(time=good_time.values)

    # 2. For vars with elevation dim: additionally NaN out bad (time, elevation) cells:
    bad_mask_elev_filtered = bad_mask_elev.isel(time=good_time.values)
    for var in ds_cf.data_vars:
        if "elevation" not in ds_cf[var].dims:
            continue
        ds_cf[var] = ds_cf[var].where(
            ~bad_mask_elev_filtered.broadcast_like(ds_cf[var]))

    return ds_cf

##############################################################################






