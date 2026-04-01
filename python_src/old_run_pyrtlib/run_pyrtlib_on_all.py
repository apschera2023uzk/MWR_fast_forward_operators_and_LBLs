#!/usr/bin/env python3

##############################################################################
# 1 Necessary modules
##############################################################################

import argparse
import os
import subprocess
import glob
import numpy as np

##############################################################################
# 2nd Used functions:
##############################################################################

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Wrapper for Radiosonde processing with pyrtlib."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes/"),
        help="Pfad zum Verzeichnis mit den Radiosonden-Rohdaten (default: %(default)s)"
    )
    parser.add_argument(
        "--pattern", "-p",
        type=str,
        default=os.path.expanduser("20*.nc"),
        help="Name convention of radiosonde files"
    )
    parser.add_argument(
        "--script", "-s",
        type=str,
        default=os.path.expanduser("~/pyrtlib/apschera_2025-07-28.py"),
        help="Pfad zum LBL script."
    )
    return parser.parse_args()

##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    files_in = glob.glob(args.input+args.pattern)
    script = args.script
    print("\n\nStart processing of all files via pyrtlib: ")

    for i, file in enumerate(files_in):
        print(i, file)
        try:
            subprocess.run(script+" -i "+file, shell=True, check=True)
        except:
            print("Could not process radiosonde: ", file)
            continue
    print("Finished processing of all files via pyrtlib\n\n")


######################
# What went here wrong: ???
# '/home/qwertz/pyrtlib/apschera_2025-07-28.py -i /home/qwertz/PhD_data/Vital_I/radiosondes/20240821_123404.nc


##############
# Maybe also solve:
# /usr/lib/python3/dist-packages/scipy/__init__.py:146: UserWarning: A NumPy version >=1.17.3 and <1.25.0 is required for this version of SciPy (detected version 1.26.4
#   warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
























