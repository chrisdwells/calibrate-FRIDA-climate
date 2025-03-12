import os
import numpy as np
import pandas as pd
import scipy.stats
from dotenv import load_dotenv

# Adapted from FaIR calibrate

# Based on AR6 Chapter 7 ERF uncertainty
#
# We do not modify forcing scale factors for ozone and aerosols, because we adjust the
# precursor species to span the forcing uncertainty this way.

# Update for reduced FaIR v1: we add an Ozone scaling factor, and remove for contrails
# Update for reduced FaIR v2: we remove the Ozone scaling factor as we are
# modelling the Ozone forcing components individually now. We replace land use
# with separate albedo and irrigation ones

load_dotenv()

print("Doing forcing uncertainty sampling...")


samples = int(os.getenv("PRIOR_SAMPLES"))

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
NINETY_TO_ONESIGMA

forcing_u90 = {
    #    'CO2': 0.12,      # CO2
    "CH4": 0.20,  # CH4: updated value from etminan 2016
    "N2O": 0.14,  # N2O
    "minorGHG": 0.19,  # other WMGHGs
    "Stratospheric water vapour": 1.00,
    # "Contrails": 0.70,  # contrails approx - half-normal
    "Light absorbing particles on snow and ice": 1.25,  # bc on snow - half-normal
    # "Land use": 0.50,  # land use change
    "Volcanic": 5.0 / 20.0,  # needs to be way bigger?
    "solar_amplitude": 0.50,
    "solar_trend": 0.07,
    "Albedo":0.1 / 0.15, # uncertainty in IPCC is -0.15 +- 0.1
}

seedgen = 380133900
scalings = {}
for forcer in forcing_u90:
    name = f"scale {forcer}"
    if forcer in ["solar_amplitude", "solar_trend"]:
        name = forcer

    scalings[name] = scipy.stats.norm.rvs(
        1, forcing_u90[forcer] / NINETY_TO_ONESIGMA, size=samples, random_state=seedgen
    )
    seedgen = seedgen + 112

# LAPSI is asymmetric Gaussian. We can just scale the half of the distribution
# above/below best estimate
scalings["scale Light absorbing particles on snow and ice"][
    scalings["scale Light absorbing particles on snow and ice"] < 1
] = (
    0.08
    / 0.1
    * (
        scalings["scale Light absorbing particles on snow and ice"][
            scalings["scale Light absorbing particles on snow and ice"] < 1
        ]
        - 1
    )
    + 1
)

# so is contrails - the benefits of doing this are tiny :)
# scalings["Contrails"][scalings["Contrails"] < 1] = (
#     0.0384 / 0.0406 * (scalings["Contrails"][scalings["Contrails"] < 1] - 1) + 1
# )

# Solar trend is absolute, not scaled
scalings["solar_trend"] = scalings["solar_trend"] - 1

# take CO2 scaling from 4xCO2 generated from the EBMs
df_ebm = pd.read_csv(
    f"../data/external/samples_for_priors/climate_response_ebm3_{samples}.csv"
)

scalings["scale CO2"] = np.array(
    1
    + 0.563 * (df_ebm["F_4xCO2"].mean() - df_ebm["F_4xCO2"]) / df_ebm["F_4xCO2"].mean()
)


# irrigation follows skewnorm distribution
def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


q05_in = -0.10
q50_in = -0.05
q95_in = 0.05

irr_params = scipy.optimize.root(opt, [1, 1, 1], args=(q05_in, q50_in, q95_in)).x

scalings["scale Irrigation"] = scipy.stats.skewnorm.rvs(irr_params[0], irr_params[1], irr_params[2], 
                              size=samples, random_state=seedgen)/q50_in


df_out = pd.DataFrame(scalings, columns=scalings.keys())

df_out.to_csv(
    f"../data/external/samples_for_priors/forcing_scaling_{samples}.csv",
    index=False,
)
