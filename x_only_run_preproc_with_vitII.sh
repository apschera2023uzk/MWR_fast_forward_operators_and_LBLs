#!/bin/bash

# 1st Process Sinthern sondes into one file:
./python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_sin -m /home/aki/PhD_data/vitII_MWR_rs_comp/kithat_sin -o /home/aki/PhD_data/TB_preproc_and_proc_results/combined_sinthern.nc -c Vital_II -l Sinthern

# 2nd Process Cologne sondes into one file (and add to Sinthern):
./python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_cgn -m /home/aki/PhD_data/vitII_MWR_rs_comp/foghat_cgn -c Vital_II -l Cologne -a /home/aki/PhD_data/TB_preproc_and_proc_results/combined_sinthern.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/combined_sinthern_and_cologne.nc
# 3rd Process Jülich sondes and add to others:
./python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_joy -m /home/aki/PhD_data/vitII_MWR_rs_comp/joyhat_joy -c Vital_II -l JOYCE -a /home/aki/PhD_data/TB_preproc_and_proc_results/combined_sinthern_and_cologne.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII.nc
# MWR_rs_FESSTVaL_Socles_vitI_vitII.nc is not the output files name as long as other inputs cannot be read...

# 4th process RAO data with DWDhat and add them to the dataset:
./python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/FESSTVaL_14GB/radiosondes/RAO -m /home/aki/PhD_data/FESSTVaL_14GB/dwdhat -m2 /home/aki/PhD_data/FESSTVaL_14GB/foghat -c FESSTVaL -l RAO -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO.nc

# 5th process RAO with UUH sondes:
./python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/FESSTVaL_14GB/radiosondes/UHH -m /home/aki/PhD_data/FESSTVaL_14GB/dwdhat -m2 /home/aki/PhD_data/FESSTVaL_14GB/foghat -c FESSTVaL -l RAO -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_v2.nc

# 6th Joyhat / Vital I
./python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/Vital_I/radiosondes -m /home/aki/PhD_data/Vital_I/hatpro-joyhat -m2 /home/aki/PhD_data/Vital_I/hamhat -c Vital_I -l JOYCE -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_v2.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_and_JoyhatvitI.nc

# 7th FESSTVaL/Falkenberg data!:
# Sunhat
./python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/FESSTVaL_14GB/radiosondes/UzK -m /home/aki/PhD_data/FESSTVaL_14GB/sunhat -c FESSTVaL -l Falkenberg -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_and_JoyhatvitI.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_and_JoyhatvitI_v2.nc

# 8th Socles / JOYCE:
./python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/Socles/radiosondes -m /home/aki/PhD_data/Socles/MWR_tophat -c Socles -l JOYCE -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_and_JoyhatvitI_v2.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_Socles_FESSTVaL_vitI_vitII.nc

##################
# Run models on data and summarize:

./python_src/proc/x_versatile_2ARMS_gb_processing.py  >> run_all.log &&
./python_src/proc/x_versatile_ARMS_gb_processing.py  >> run_all.log &&
./python_src/proc/x_versatile_RTTOV_gb_processing.py >> run_all.log



#Traceback (most recent call last):
#  File "././python_src/proc/x_2ARMS_gb_processing.py", line 383, in <module>
#    ds = derive_TBs4ARMS_gb_per_elevation(ds, args,\
#         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#  File "././python_src/proc/x_2ARMS_gb_processing.py", line 328, in derive_TBs4ARMS_gb_per_elevation
#    len(ds["Crop"].values)), np.nan)
#        ~~^^^^^^^^
#  File "/usr/lib/python3/dist-packages/xarray/core/dataset.py", line 1547, in __getitem__
#    raise KeyError(
#KeyError: "No variable named 'Crop'. Variables on the dataset include ['TBs', 'MWR_z', 'MWR_ta', 'MWR_hua', 'MWR_IWV', ..., 'N_Channels', 'time', 'N_Levels', #'elevation', 'azimuth']"


# 
#./python_src/proc/ARMS_gb_processing.py >> run_all.log &&
# ./python_src/proc/RTTOV_gb_processing.py >> run_all.log &&
# ./python_src/proc/PyRTlib_processing.py >> run_all.log &&

# ./python_src/proc/summarize_proc_results.py >> run_all.log &&












