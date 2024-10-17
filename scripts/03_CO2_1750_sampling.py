#!/usr/bin/env python
# coding: utf-8

# from fair calibrate - need 1750 CO2 to spin-up ocean

import os

import pandas as pd
import scipy.stats
from dotenv import load_dotenv

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

samples=1000

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
co2_1750_conc = scipy.stats.norm.rvs(
    size=samples, loc=278.3, scale=2.9 / NINETY_TO_ONESIGMA, random_state=1067061
)

df = pd.DataFrame({"Ocean.Atmospheric CO2 Concentration 1750": co2_1750_conc})

df.to_csv(
    f"../data/spinup_input/co2_1750_{samples}.csv",
    index=False,
)
