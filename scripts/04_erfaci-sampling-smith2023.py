#!/usr/bin/env python
# coding: utf-8

"""Sample aerosol indirect."""

# # Using the fair-2.1 pure log formula
#
# **Note**
# Estimating aerosol cloud interactions from 11 CMIP6 models was performed in Smith
#  et al. 2021: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD033622.
#
# The underlying APRP code was slightly wrong, and has been updated thanks to Mark
# Zelinka (released as climateforcing v0.3.0). Two more models are now available.
# Actually three are, but EC-Earth3 is unusable due to unphysical values of rsuscs and
# rsdscs leading to biased ERFaci estimates.
#
# \begin{equation}
# F = \beta \log \left( 1 + \sum_{i} n_i A_i \right)
# \end{equation}
#
# where
# - $A_i$ is the atmospheric input (concentrations or emissions of a specie),
# - $\beta_i$ is a scale factor,
# - $n_i$ dictates how much emissions of a specie contributes to CDNC.
#
# **Note also** the uniform prior from -2 to 0. A lot of the sublteties here might also
# want to go into the paper.


import glob
import os

import numpy as np
import pandas as pd
import pooch
import scipy.stats
from dotenv import load_dotenv
from scipy.optimize import curve_fit
from tqdm import tqdm

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

files = glob.glob("../data/external/smith2023aerosol/*.csv")

ari = {}
aci = {}
models = []
models_runs = {}
years = {}
for file in files:
    model = os.path.split(file)[1].split("_")[0]
    run = os.path.split(file)[1].split("_")[1]
    models.append(model)
    if run not in models_runs:
        models_runs[model] = []
    models_runs[model].append(run)

models = list(models_runs.keys())

for model in models:
    nruns = 0
    for run in models_runs[model]:
        file = f"../data/external/smith2023aerosol/{model}_{run}_aerosol_forcing.csv"
        df = pd.read_csv(file, index_col=0)
        if nruns == 0:
            ari_temp = df["ERFari"].values.squeeze()
            aci_temp = df["ERFaci"].values.squeeze()
        else:
            ari_temp = ari_temp + df["ERFari"].values.squeeze()
            aci_temp = aci_temp + df["ERFaci"].values.squeeze()
        years[model] = df.index + 0.5
        nruns = nruns + 1
    ari[model] = ari_temp / nruns
    aci[model] = aci_temp / nruns


rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

emis_df = pd.read_csv(rcmip_emissions_file)

so2 = (
    emis_df.loc[
        (emis_df["Scenario"] == "ssp245")
        & (emis_df["Region"] == "World")
        & (emis_df["Variable"] == "Emissions|Sulfur"),
        "1750":"2100",
    ]
    .interpolate(axis=1)
    .squeeze()
    .values
)


def aci_log(x, beta, n0):
    aci = beta * np.log(1 + x[0] * n0)
    aci_1850 = beta * np.log(1 + so2[100] * n0)
    return aci - aci_1850


param_fits = {}

for model in models:
    ist = int(np.floor(years[model][0] - 1750))
    ien = int(np.ceil(years[model][-1] - 1750))
    param_fits[model], cov = curve_fit(
        aci_log,
        [so2[ist:ien]],
        aci[model],
        bounds=((-np.inf, 0), (0, np.inf)),
        max_nfev=10000,
    )


def aci_log1750(x, beta, n0):
    aci = beta * np.log(1 + x[0] * n0)
    aci_1750 = beta * np.log(1 + so2[0] * n0)
    return aci - aci_1750


df_ar6 = pd.read_csv(
    "../data/external/forcing/table_A3.3_historical_ERF_1750-2019_best_estimate.csv"
)

params_ar6, cov = curve_fit(
    aci_log1750,
    [so2[:270]],
    df_ar6["aerosol-cloud_interactions"].values,
    bounds=((-np.inf, 0), (0, np.inf)),
    max_nfev=10000,
)

df_params = pd.DataFrame(param_fits, index=["aci_scale", "Sulfur"]).T

df_params.to_csv(
    "../data/external/forcing/aerosol_cloud.csv"
)

print("Correlation coefficients between aci parameters")
print(df_params.corr())

beta_samp = df_params["aci_scale"]
n0_samp = df_params["Sulfur"]

kde = scipy.stats.gaussian_kde([n0_samp])
aci_sample = kde.resample(size=samples * 4, seed=63648708)

aci_sample[0, aci_sample[0, :] < 0] = np.nan

mask = np.any(np.isnan(aci_sample), axis=0)
aci_sample = aci_sample[:, ~mask]

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
erfaci_sample = scipy.stats.uniform.rvs(
    size=samples, loc=-2.0, scale=2.0, random_state=71271
)

beta = np.zeros(samples)
erfaci = np.zeros((351, samples))
for i in tqdm(range(samples), desc="aci samples"):
    ts2010 = np.mean(
        aci_log(
            [so2[255:265]],
            1,
            aci_sample[0, i],
        )
    )
    ts1850 = aci_log(
        [so2[100]],
        1,
        aci_sample[0, i],
    )
    ts1750 = aci_log(
        [so2[0]],
        1,
        aci_sample[0, i],
    )
    erfaci[:, i] = (
        (
            aci_log(
                [so2],
                1,
                aci_sample[0, i],
            )
            - ts1750
        )
        / (ts2010 - ts1850)
        * (erfaci_sample[i])
    )
    beta[i] = erfaci_sample[i] / (ts2010 - ts1750)


df = pd.DataFrame(
    {
        "shape Sulfur": aci_sample[0, :samples],
        "beta": beta,
    }
)


df.to_csv(
    f"../data/external/samples_for_priors/aerosol_cloud_{samples}.csv",
    index=False,
)
