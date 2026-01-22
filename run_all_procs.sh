#!/bin/bash
 
sudo echo "Started!" &&
touch run_all.log &&
# ./python_src/preproc/preprocessing4all.py >> run_all.log &&
# ./python_src/proc/ARMS_gb_processing.py >> run_all.log &&
# ./python_src/proc/RTTOV_gb_processing.py >> run_all.log &&
./python_src/proc/PyRTlib_processing.py >> run_all.log &&
./python_src/proc/summarize_proc_results.py >> run_all.log &&
./python_src/plot_scripts/multi_campaign_plots_and_ana.py >> run_all.log
sudo apt-get update -y && sudo apt-get upgrade -y
shutdown now
