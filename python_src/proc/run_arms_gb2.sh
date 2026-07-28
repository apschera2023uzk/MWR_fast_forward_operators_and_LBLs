#!/bin/bash

cd ~/armsgb2/Obs_Sim_armsgb &&
export FC=ifx &&
make clean &&
make &&
./FWD_Test
