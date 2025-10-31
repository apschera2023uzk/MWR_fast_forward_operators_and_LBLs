#!/bin/bash

cd ~/armsgb/Obs_Sim_armsgb &&
export FC=ifx &&
make clean &&
make &&
./FWD_Test
