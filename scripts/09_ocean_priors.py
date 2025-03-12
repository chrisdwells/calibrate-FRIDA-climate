#!/usr/bin/env python
# coding: utf-8

# cribbed from fair calibrate
# this is for the ocean params which don't affect the spin-up; they only get
# varied in the prior ensemble. because of this, we use the full sample number 

# the salinity and alkalinity should change together, so we generate the 
# salinity parameters by scaling the alkalinity ones to match their range

import os

# import numpy as np
import pandas as pd
import scipy.stats
from dotenv import load_dotenv

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

ocean_variables = {
    "Ocean.Warm surface ocean alkalinity sensitivity to global T anomaly[1]":[-4e-6, -1e-6],
    "Ocean.Cold surface ocean alkalinity sensitivity to global T anomaly[1]":[-3e-5, -5e-6],
    "Ocean.High latitude carbon pump sensitivity to global T anomaly[1]":[-0.5,0],
    "Ocean.Warm surface ocean temperature sensitivity to global T anomaly[1]":[0.4, 1.0],
    "Ocean.Cold surface ocean temperature sensitivity to global T anomaly[1]":[0.3, 1.0],
    # "Ocean.Warm surface ocean salinity sensitivity to global T anomaly[1]":[-0.04, -0.01],
    # "Ocean.Cold surface ocean salinity sensitivity to global T anomaly[1]":[-0.3, -0.05],
    }


param_dict = {}

for o_i, ocean_var in enumerate(ocean_variables):
    
    param_dict[ocean_var] = scipy.stats.uniform.rvs(
        ocean_variables[ocean_var][0],
        ocean_variables[ocean_var][1] - ocean_variables[ocean_var][0],
        size=samples,
        random_state=3729329 + 1000*o_i,
    )
    
    
param_dict["Ocean.Warm surface ocean salinity sensitivity to global T anomaly[1]"
       ] = 10000*param_dict[
   "Ocean.Warm surface ocean alkalinity sensitivity to global T anomaly[1]"]

param_dict["Ocean.Cold surface ocean salinity sensitivity to global T anomaly[1]"
       ] = 10000*param_dict[
   "Ocean.Cold surface ocean alkalinity sensitivity to global T anomaly[1]"]

           
df = pd.DataFrame(param_dict, columns=param_dict.keys())

os.makedirs("../data/priors_input", exist_ok=True)
df.to_csv(
    f"../data/priors_input/ocean_priors_params_{samples}.csv",
    index=False,
)
