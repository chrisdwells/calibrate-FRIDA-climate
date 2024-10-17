#!/usr/bin/env python
# coding: utf-8

# cribbed from fair calibrate
# this is for the ocean params which don't affect the spin-up; they only get
# varied in the prior ensemble.

import os

import numpy as np
import pandas as pd
import scipy.stats
from dotenv import load_dotenv

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

ocean_variables = {
    "Ocean.Warm surface ocean alkalinity sensitivity to global T anomaly":[0.5,3.0],
    "Ocean.Cold surface ocean alkalinity sensitivity to global T anomaly":[0.5,3.0],
    "Ocean.High latitude carbon pump sensitivity to global T anomaly":[-5,0],
    }

param_dict = {}

for o_i, ocean_var in enumerate(ocean_variables):
    
    param_dict[ocean_var] = scipy.stats.uniform.rvs(
        ocean_variables[ocean_var][0],
        ocean_variables[ocean_var][1] - ocean_variables[ocean_var][0],
        size=samples,
        random_state=3729329 + 1000*o_i,
    )
    

df = pd.DataFrame(param_dict, columns=param_dict.keys())

df.to_csv(
    f"../data/priors_input/ocean_priors_params_{samples}.csv",
    index=False,
)
