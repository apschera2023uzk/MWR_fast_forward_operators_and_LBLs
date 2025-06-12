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
import subprocess
import time

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
        default=os.path.expanduser("~/PhD_data/Vital_I/radiosondes/"),
        help="Pfad zum Verzeichnis für RTTOV-gb Inputdateien (default: %(default)s)"
    )
    parser.add_argument(
        "--rttov_script", "-rr",
        type=str,
        default=os.path.expanduser("~/RTTOV-gb/rttov_test/run_apschera.sh"),
        help="Path to RTTOV-gb run script for your use case"
    )
    parser.add_argument(
        "--infile_rttov_gb", "-ir",
        type=str,
        default=os.path.expanduser("~/RTTOV-gb/rttov_test/test_example_k.1/prof_plev.dat"),
        help="Path to RTTOV-gb input profile in RTTOV-gb package structure"
    )
    parser.add_argument(
        "--outfile_rttov_gb", "-or",
        type=str,
        default=os.path.expanduser("~/RTTOV-gb/rttov_test/test_example_k.1/output_example_k.dat.gfortran"),
        help="Path to RTTOV-gb output T_bs in RTTOV-gb package structure"
    )
    return parser.parse_args()

##############################################################################



##############################################################################
# 3 Main
##############################################################################

if __name__ == "__main__":
    args = parse_arguments()
    files_in = glob.glob(args.input+"prof_*.dat")
    dst_file = args.infile_rttov_gb
    script_file = args.rttov_script

    for i, file in enumerate(files_in):

        # Copy inputfile into RTTOV-gb inputs:
        shutil.copy(file, args.infile_rttov_gb)

        # Detect and replace number of levels in runscript:
        with open(dst_file, "r") as f:
            total_lines = sum(1 for _ in f)
        nlevels = total_lines // 4
        print("nlevels: ",nlevels)
        with open(script_file, "r") as f:
            lines = f.readlines()
        # Zeile 30 (Index 29) ersetzen
        lines[29] = f"NLEVELS={nlevels}\n"
        with open(script_file, "w") as f:
            f.writelines(lines)

        # Run RTTOV-gb:
        rttov_dir = os.path.dirname(args.rttov_script)
        subprocess.run(["bash", args.rttov_script, "ARCH=gfortran"], cwd=rttov_dir)
        time.sleep(2)

        # Copy result back to inputs:
        new_output_file=args.input+"rttov-gb_"+file.split("/")[-1][5:-4]+".txt"
        shutil.copy(args.outfile_rttov_gb, new_output_file)




























