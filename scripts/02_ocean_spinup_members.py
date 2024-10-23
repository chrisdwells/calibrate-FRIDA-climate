#!/usr/bin/env python
# coding: utf-8

# cribbed from fair calibrate. feed these into Ocean_spinup_start

import os

import numpy as np
import pandas as pd
import scipy.stats
from dotenv import load_dotenv

load_dotenv()

ocean_samples = int(os.getenv("OCEAN_SPINUP_SAMPLES"))

ocean_variables = {
    "Ocean.Depth of warm surface ocean layer":[50,500],
    "Ocean.Thickness of intermediate ocean layer":[300,1000],
    "Ocean.Depth of cold surface ocean layer":[50,500],
    "Ocean.Reference overturning strength in Sv":[10,30],
    "Ocean.Reference intermediate to warm surface ocean mixing strength":[50,90],
    "Ocean.Reference cold surface to deep ocean mixing strength":[10,30],
    "Ocean.Reference strength of biological carbon pump in low latitude ocean":[0,3],
    "Ocean.Reference strength of biological carbon pump in high latitude ocean":[4,12],
    "Ocean.High latitude carbon pump transfer efficiency":[0.1,0.5],
    }

param_dict = {}

for o_i, ocean_var in enumerate(ocean_variables):
    
    param_dict[ocean_var] = scipy.stats.uniform.rvs(
        ocean_variables[ocean_var][0],
        ocean_variables[ocean_var][1] - ocean_variables[ocean_var][0],
        size=ocean_samples,
        random_state=3729329 + 1000*o_i,
    )
    

df = pd.DataFrame(param_dict, columns=param_dict.keys())

df.to_csv(
    f"../data/spinup_input/ocean_spinup_params_{ocean_samples}.csv",
    index=False,
)
