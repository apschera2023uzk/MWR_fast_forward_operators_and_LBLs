#!/bin/bash
 
sudo echo "Started!" > run_all.log &&
# ./python_src/preproc/preprocessing4all.py > run_all.log &&
# ./python_src/proc/x_2ARMS_gb_processing.py >> run_all.log &&
#./python_src/proc/ARMS_gb_processing.py >> run_all.log &&
# ./python_src/proc/RTTOV_gb_processing.py >> run_all.log &&
# ./python_src/proc/PyRTlib_processing.py >> run_all.log &&
# ./python_src/proc/summarize_proc_results.py >> run_all.log &&
# ./python_src/plot_scripts/multi_campaign_plots_and_ana.py >> run_all.log &&
./python_src/plot_scripts/x_analysis_script_MARCH26.py >> run_all.log &&
# ./python_src/plot_scripts/x_clear_sky_percentage_per_elev_MARCH26.py &&
# ./python_src/plot_scripts/x_devs_by_IWV.py >> run_all.log &&
./python_src/plot_scripts/x_colorplot_by_elevs_and_chans_MARCH26.py >> run_all.log &&
./python_src/plot_scripts/x_line_plots_by_elev_MARCH26.py >> run_all.log &&
./python_src/plot_scripts/x_plot_std_bars.py >> run_all.log
# sudo apt-get update -y && sudo apt-get upgrade -y
# shutdown now
