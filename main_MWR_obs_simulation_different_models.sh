#!/bin/bash

./python_src/rttov-gb_wrapper/preprocessing4rttov-gb_4zen.py &&
./python_src/rttov-gb_wrapper/run_rttov-gb_on_rs4zen.py &&
./python_src/run_pyrtlib/run_pyrtlib_on_all.py &&
# Add ARMS-gb processing
./python_src/merge_data_into_netCDF/merge2nc.py &&
./python_src/plot_scripts/plot_TB-scatter_MWR.py





###############################

path_lbl="~/mwrpy_ret"
path_rttov_gb="~/RTTOV-gb"
path_radiosondes="~/atris/radiosondes/2024/08" # ganzer Monat in einem!
path_MWR="~/atris/hatpro-joyhat/2024/08" # plus dd Ordner!
path_output="~/PhD_data"

echo path_rttov_gb

