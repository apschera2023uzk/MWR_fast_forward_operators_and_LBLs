#!/bin/bash

# RTTOV-gb processing:
./python_src/rttov-gb_wrapper/preprocessing4rttov-gb_4zen.py &&
./python_src/rttov-gb_wrapper/run_rttov-gb_on_rs4zen.py &&

# Prtlib processing:
./python_src/run_pyrtlib/run_pyrtlib_on_all.py &&

# ARMS-gb processing:
./python_src/arms-gb-wrapper/preprocessing4armsgb.py &&
cd /home/aki/armsgb/Obs_Sim_armsgb &&
export FC=ifx &&
make clean &&
make &&
./FWD_Test &&
cd /home/aki/armsgb/Obs_Sim_armsgb_crop &&
export FC=ifx &&
make clean &&
make &&
./FWD_Test &&

# Summarize data:
cd ~/MWR_fast_forward_operators_and_LBLs &&
./python_src/merge_data_into_netCDF/merge2nc.py &&

# Plot data:
./python_src/plot_scripts/plot_TB-scatter_MWR.py &&
./python_src/plot_scripts/final_zenith_plots.py


###############################
# Maybe move up and make code dependent on folders:

# path_rttov_gb="~/RTTOV-gb"
# path_radiosondes="~/atris/radiosondes/2024/08" # ganzer Monat in einem!
# path_MWR="~/atris/hatpro-joyhat/2024/08" # plus dd Ordner!
# path_output="~/PhD_data"

# echo path_rttov_gb

