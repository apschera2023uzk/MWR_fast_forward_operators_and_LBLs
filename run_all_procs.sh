#!/bin/bash
 
./python_src/preproc/preprocessing4all.py &&
./python_src/proc/ARMS_gb_processing.py && 
./python_src/proc/RTTOV_gb_processing.py && 
# ./python_src/proc/PyRTlib_processing.py > output.txt && 
./python_src/proc/summarize_proc_results.py &&
./python_src/plot_scripts/multi_campaign_plots_and_ana.py
# sudo apt-get update -y && sudo apt-get upgrade -y
# shutdown now
