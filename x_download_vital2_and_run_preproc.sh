#!/bin/bash

####################################################################
# Prepare sshpass:

PASSWORD_FILE="/home/aki/.akwfwkgl"  # Datei mit dem Passwort
# Überprüfen, ob die Passwort-Datei existiert
if [ ! -f "$PASSWORD_FILE" ]; then
    echo "Passwort-Datei $PASSWORD_FILE nicht gefunden!" >> update.log
    exit 1
fi
# Passwort aus der Datei lesen
PASSWORD=$(cat "$PASSWORD_FILE")

####################################################################
# Update MWR and Sonde data:

# 1.1 Cologne MWR:
cd /home/aki/PhD_data/vitII_MWR_rs_comp/foghat_cgn
sshpass -p "$PASSWORD" scp apscherra@secaire.meteo.uni-koeln.de:/data/obs/site/cgn/foghat/actris/level1/2026/08/*/MWR_1C01_*.nc .
sshpass -p "$PASSWORD" scp apscherra@secaire.meteo.uni-koeln.de:/data/obs/site/cgn/foghat/actris/level1/2026/09/*/MWR_1C01_*.nc .
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/site/cgn/foghat/actris/level2/2026/08/*/*.nc .
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/site/cgn/foghat/actris/level2/2026/09/*/*.nc .
# 1.2 Cologne sondes:
cd /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_cgn
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/campaigns/vital2/site/cologne/sounding/vitII*.nc .

# 2.1 Sinthern MWR:
cd /home/aki/PhD_data/vitII_MWR_rs_comp/kithat_sin
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/campaigns/vital2/site/sinthern/mwr/level1/2026/08/*/MWR_1C*.nc .
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/campaigns/vital2/site/sinthern/mwr/level1/2026/09/*/MWR_1C*.nc .
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/campaigns/vital2/site/sinthern/mwr/level2/2026/08/*/*.nc .
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/campaigns/vital2/site/sinthern/mwr/level2/2026/09/*/*.nc .
# 2.2 Sinthern sondes:
cd /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_sin
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/campaigns/vital2/site/sinthern/sounding/vitII*.nc .

# 3.1 Jülich MWR:
cd /home/aki/PhD_data/vitII_MWR_rs_comp/joyhat_joy
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/site/jue/joyhat/l1/2026/08/*/sups_joy_mwr00_l1_*.nc .
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/site/jue/joyhat/l1/2026/09/*/sups_joy_mwr00_l1_*.nc .
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/site/jue/joyhat/l2/2026/08/*/*_mwr00_*.nc .
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/site/jue/joyhat/l2/2026/09/*/*_mwr00_*.nc .
# 3.2 Jülich sondes:
cd /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_joy/
sshpass -p "$PASSWORD" scp apscherra@respos.meteo.uni-koeln.de:/data/obs/campaigns/vital2/site/juelich/sounding/vitII*.nc .

############################################################
# Automatize preproc of 3 Vital II sites:

# 1st Process Sinthern sondes into one file:
/home/aki/MWR_fast_forward_operators_and_LBLs/python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_sin -m /home/aki/PhD_data/vitII_MWR_rs_comp/kithat_sin -o /home/aki/PhD_data/TB_preproc_and_proc_results/combined_sinthern.nc -c Vital_II -l Sinthern

# 2nd Process Cologne sondes into one file (and add to Sinthern):
/home/aki/MWR_fast_forward_operators_and_LBLs/python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_cgn -m /home/aki/PhD_data/vitII_MWR_rs_comp/foghat_cgn -c Vital_II -l Cologne -a /home/aki/PhD_data/TB_preproc_and_proc_results/combined_sinthern.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/combined_sinthern_and_cologne.nc
# 3rd Process Jülich sondes and add to others:
/home/aki/MWR_fast_forward_operators_and_LBLs/python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/vitII_MWR_rs_comp/sondes_joy -m /home/aki/PhD_data/vitII_MWR_rs_comp/joyhat_joy -c Vital_II -l JOYCE -a /home/aki/PhD_data/TB_preproc_and_proc_results/combined_sinthern_and_cologne.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII.nc
# MWR_rs_FESSTVaL_Socles_vitI_vitII.nc is not the output files name as long as other inputs cannot be read...

# 4th process RAO data with DWDhat and add them to the dataset:
/home/aki/MWR_fast_forward_operators_and_LBLs/python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/FESSTVaL_14GB/radiosondes/RAO -m /home/aki/PhD_data/FESSTVaL_14GB/dwdhat -m2 /home/aki/PhD_data/FESSTVaL_14GB/foghat -c FESSTVaL -l RAO -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO.nc

# 5th process RAO with UUH sondes:
/home/aki/MWR_fast_forward_operators_and_LBLs/python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/FESSTVaL_14GB/radiosondes/UHH -m /home/aki/PhD_data/FESSTVaL_14GB/dwdhat -m2 /home/aki/PhD_data/FESSTVaL_14GB/foghat -c FESSTVaL -l RAO -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_v2.nc

# 6th Joyhat / Vital I
/home/aki/MWR_fast_forward_operators_and_LBLs/python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/Vital_I/radiosondes -m /home/aki/PhD_data/Vital_I/hatpro-joyhat -m2 /home/aki/PhD_data/Vital_I/hamhat -c Vital_I -l JOYCE -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_v2.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_and_JoyhatvitI.nc

# 7th FESSTVaL/Falkenberg data!:
# Sunhat
/home/aki/MWR_fast_forward_operators_and_LBLs/python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/FESSTVaL_14GB/radiosondes/UzK -m /home/aki/PhD_data/FESSTVaL_14GB/sunhat -c FESSTVaL -l Falkenberg -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_and_JoyhatvitI.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_and_JoyhatvitI_v2.nc

# 8th Socles / JOYCE:
/home/aki/MWR_fast_forward_operators_and_LBLs/python_src/preproc/x_versatile_modular_preproc.py -r /home/aki/PhD_data/Socles/radiosondes -m /home/aki/PhD_data/Socles/MWR_tophat -c Socles -l JOYCE -a /home/aki/PhD_data/TB_preproc_and_proc_results/complete_vitalII_and_RAO_and_JoyhatvitI_v2.nc -o /home/aki/PhD_data/TB_preproc_and_proc_results/complete_Socles_FESSTVaL_vitI_vitII.nc



##################
# => Make it in the long run work together with old dataset TB preproc from models and evaluation???












