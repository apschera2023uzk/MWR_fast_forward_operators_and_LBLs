#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import os
import subprocess
import glob
# import pandas as pd
import numpy as np
import shutil

##############################################################################
# 2nd Used functions:
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Wrapper for Radiosonde processing with mwrpy_ret."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes/"),
        help="Pfad zum Verzeichnis mit den Radiosonden-Rohdaten (default: %(default)s)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=os.path.expanduser("~/mwrpy_sim/tests/data/"),
        help="Pfad zum Verzeichnis für mwrpy_sim Output- und Inputdateien (default: %(default)s)"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default=os.path.expanduser("20*.nc"),
        help="Name convention of radiosonde files"
    )
    parser.add_argument(
        "--final_output", "-fo",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/mwrpy_sim_outputs/"),
        help="Where to store named mwrpy_sim outputs"
    )
    parser.add_argument(
        "--site", "-s",
        type=str,
        default=os.path.expanduser("juelich"),
        help="Measurement site: e.g. juelich"
    )
    return parser.parse_args()

##############################################################################
'''
def run_mwrpy_ret(workdir, site="juelich", date="2024-08-19"):
    # This function just runs the LBL model mwrpy_ret:
    os.chdir(workdir)
    venv_activate = os.path.join(workdir, "venv", "bin", "activate")
    subprocess.run(f"source {venv_activate} && pip3 install .",\
        shell=True, executable="/bin/bash", check=True)
    subprocess.run(f"source {venv_activate} && ./mwrpy_ret/cli.py -s "+site+" -d "+date+" radiosonde",\
        shell=True, executable="/bin/bash", check=True)
    return 0
'''
##############################################################################

def run_mwrpy_sim(workdir, site="juelich", date="2024-08-19"):
    # This function just runs the LBL model mwrpy_ret:
    os.chdir(workdir)
    venv_activate = os.path.join(workdir, "venv", "bin", "activate")
    subprocess.run(f"source {venv_activate} && pip3 install .",\
        shell=True, executable="/bin/bash", check=True)
    subprocess.run(f"source {venv_activate} && ./mwrpy_sim/cli.py -s "+site+" -d "+date+" radiosonde",\
        shell=True, executable="/bin/bash", check=True)
    return 0

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    files_in = glob.glob(args.input+args.pattern)

    for i, file in enumerate(files_in):
        
        # Copy radiosonde to mwrpy input according to date:
        yyyy = file.split("/")[-1][0:4]
        mm = file.split("/")[-1][4:6]
        dd = file.split("/")[-1][6:8]
        infile = args.output+yyyy+"/"+mm+"/"+dd+"/"+"v_radiosonde_"+file.split("/")[-1]
        shutil.copy(file, infile)

        # Run mwrpy_ret on one radiosonde:
        # INFO:root:Processing took 602.0 seconds == 10 minuten pro Profil.
        # INFO:root:Processing took 651.9 seconds
        workdir = (args.output).split("/mwrpy_sim/")[0]+"/mwrpy_sim"
        run_mwrpy_sim(workdir, site=args.site, date=yyyy+"-"+mm+"-"+dd)

        # Move result file:
        new_name = args.final_output+"mwrpy_sim_out_rs_"+file.split("/")[-1]
        old_file = args.output+"juelich_radiosonde_"+file.split("/")[-1][:8]+".nc"
        shutil.copy(old_file, new_name)

        # After all processing of this sonde is done:
        if os.path.exists(infile):
            os.remove(infile)
            print(f"Datei '{infile}' wurde gelöscht.")
        if os.path.exists(old_file):
            os.remove(old_file)
            print(f"Datei '{infile}' wurde gelöscht.")

        break
        ################
        # Remove this break for operational run!!
        # print("Remove this break for operational run!!")
        # break






























