#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import os
import glob
# import pandas as pd
import numpy as np
import shutil

##############################################################################
# 2nd Used functions:
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Preprocess radiosonde data for RTTOV-gb input format."
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
        default=os.path.expanduser("~/mwrpy_ret/tests/data/"),
        help="Pfad zum Verzeichnis für RTTOV-gb Inputdateien (default: %(default)s)"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default=os.path.expanduser("20*.nc"),
        help="Name convention of radiosonde csv-files"
    )
    return parser.parse_args()

##############################################################################


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
        shutil.copy(file, args.output+yyyy+"/"+mm+"/"+dd+"/"+"v_"+file.split("/")[-1])
           

   
    
    # Finde nc-Dateine
    # Kopiere an den Ausführungsort mit v_ umbenennung


    "v_radiosonde_20240819_190132.nc"


# ÖLass MWRpyret laufen:
# 1. source venv/bin/activate
# 2.   pip3 install .  # Re-build program
# 3. ./mwrpy_ret/cli.py -s juelich -d 2024-08-19 radiosonde
# => Probleme entstehen beim Schreiben des Outputs aufgrund falscher Dimensionen von data und dims in Datei # ret_mwr.py.

# data dims  (1, 6373)
# FO:root:Processing took 602.0 seconds == 10 minuten pro Profil.


     # lösche unnötige Funktionen von hier!!!






























