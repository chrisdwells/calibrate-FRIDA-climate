# calibrate-FRIDA-climate

Main FRIDA repository: https://github.com/metno/WorldTransFRIDA

This repository performs the climate calibration for FRIDA. 

## Requirements
- Stella
- anaconda python

## Reproduction
1. Create the conda environment with `conda env create -f environment.yml`
2. Activate the environment: `conda activate calibrate-FRIDA-climate`
3. run scripts/01
4. run Ocean_spinup_start.itmx
5. run scripts/02
6. run Ocean_spinup_end.itmx
7. The scripts should be ran in order from 03-10, in the scripts/ folder. 
8. TODO: make a script that makes the priors_output directory and a bunch of blank CSVs at place 11
9. then run temperature_and_ocean_from_1750 (and export data)
10. run the rest

Information on each is given in the code. Part of the process involves running versions of the climate model in Stella.
