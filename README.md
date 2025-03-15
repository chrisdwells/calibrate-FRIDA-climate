# calibrate-FRIDA-climate

Main FRIDA repository: https://github.com/metno/WorldTransFRIDA

This repository performs the climate calibration for FRIDA. 

## Requirements
- Stella
- anaconda python

## Reproduction
1. Create the conda environment with `conda env create -f environment.yml`
2. Activate the environment: `conda activate calibrate-FRIDA-climate`
3. move to the scripts/ folder
4. run 01
5. run Ocean_spinup_start.itmx (and export data)
6. run 02
7. run Ocean_spinup_end.itmx (and export data)
8. run 03-10
9. run temperature_and_ocean_from_1750.itmx (and export data)
10. run 11-13
11. run Temperature_and_Ocean_from_1750_100_members
12. run 14

Information on each is given in the code. Part of the process involves running versions of the climate model in Stella.