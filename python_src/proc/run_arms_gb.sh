#!/bin/bash

cd /home/aki/armsgb/Obs_Sim_armsgb &&
export FC=ifx &&
make clean &&
make &&
./FWD_Test
