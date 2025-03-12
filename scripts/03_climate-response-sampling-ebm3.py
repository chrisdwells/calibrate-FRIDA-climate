import os
import numpy as np
import pandas as pd
import scipy.linalg
import scipy.stats
from dotenv import load_dotenv
from fair.energy_balance_model import EnergyBalanceModel
from tqdm import tqdm

from fair import FAIR
from fair.io import read_properties
from fair.interface import fill, initialise

from fair.energy_balance_model import (
    multi_ebm,
)

# Adapted from FaIR calibrate

# For FRIDA, get ECS and TCR from the FaIR EBM

# The purpose here is to provide correlated calibrations to the climate response in
# CMIP6 models.

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

df = pd.read_csv(
    os.path.join(
        "../data/external/4xCO2_cummins_ebm3_cmip6.csv"
    )
)
models = df["model"].unique()

for model in models:
    print(model, df.loc[df["model"] == model, "run"].values)


# Judgement time:
# - GISS-E2-1-G 'r1i1p1f1'
# - GISS-E2-1-H 'r1i1p3f1'  less wacky
# - MRI-ESM2-0 'r1i1p1f1'
# - EC-Earth3 'r3i1p1f1'  less wacky
# - FIO-ESM-2-0  'r1i1p1f1'
# - CanESM5  'r1i1p2f1'
# - FGOALS-f3-L 'r1i1p1f1'
# - CNRM-ESM2-1 'r1i1p1f2'

n_models = len(models)

multi_runs = {
    "GISS-E2-1-G": "r1i1p1f1",
    "GISS-E2-1-H": "r1i1p3f1",
    "MRI-ESM2-0": "r1i1p1f1",
    "EC-Earth3": "r3i1p1f1",
    "FIO-ESM-2-0": "r1i1p1f1",
    "CanESM5": "r1i1p2f1",
    "FGOALS-f3-L": "r1i1p1f1",
    "CNRM-ESM2-1": "r1i1p1f2",
}

params = {}

params[r"$\gamma$"] = np.ones(n_models) * np.nan
params["$c_1$"] = np.ones(n_models) * np.nan
params["$c_2$"] = np.ones(n_models) * np.nan
params["$c_3$"] = np.ones(n_models) * np.nan
params[r"$\kappa_1$"] = np.ones(n_models) * np.nan
params[r"$\kappa_2$"] = np.ones(n_models) * np.nan
params[r"$\kappa_3$"] = np.ones(n_models) * np.nan
params[r"$\epsilon$"] = np.ones(n_models) * np.nan
params[r"$\sigma_{\eta}$"] = np.ones(n_models) * np.nan
params[r"$\sigma_{\xi}$"] = np.ones(n_models) * np.nan
params[r"$F_{4\times}$"] = np.ones(n_models) * np.nan

for im, model in enumerate(models):
    if model in multi_runs:
        condition = (df["model"] == model) & (df["run"] == multi_runs[model])
    else:
        condition = df["model"] == model
    params[r"$\gamma$"][im] = df.loc[condition, "gamma"].values[0]
    params["$c_1$"][im], params["$c_2$"][im], params["$c_3$"][im] = df.loc[
        condition, "C1":"C3"
    ].values.squeeze()
    (
        params[r"$\kappa_1$"][im],
        params[r"$\kappa_2$"][im],
        params[r"$\kappa_3$"][im],
    ) = df.loc[condition, "kappa1":"kappa3"].values.squeeze()
    params[r"$\epsilon$"][im] = df.loc[condition, "epsilon"].values[0]
    params[r"$\sigma_{\eta}$"][im] = df.loc[condition, "sigma_eta"].values[0]
    params[r"$\sigma_{\xi}$"][im] = df.loc[condition, "sigma_xi"].values[0]
    params[r"$F_{4\times}$"][im] = df.loc[condition, "F_4xCO2"].values[0]
    
params = pd.DataFrame(params)
print(params.corr())

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

kde = scipy.stats.gaussian_kde(params.T)
ebm_sample = kde.resample(size=int(samples * 4), seed=2181882)

# remove unphysical combinations
for col in range(10):
    ebm_sample[:, ebm_sample[col, :] <= 0] = np.nan
ebm_sample[:, ebm_sample[0, :] <= 0.8] = np.nan  # gamma
ebm_sample[:, ebm_sample[1, :] <= 2] = np.nan  # C1
ebm_sample[:, ebm_sample[2, :] <= ebm_sample[1, :]] = np.nan  # C2
ebm_sample[:, ebm_sample[3, :] <= ebm_sample[2, :]] = np.nan  # C3
ebm_sample[:, ebm_sample[4, :] <= 0.3] = np.nan  # kappa1 = lambda

mask = np.all(np.isnan(ebm_sample), axis=0)
ebm_sample = ebm_sample[:, ~mask]

# check that covariance matrix is positive semidefinite and if not, remove param combo.
# to do: change away from sparse, once we move away from R
for isample in tqdm(range(len(ebm_sample.T))):
    ebm = EnergyBalanceModel(
        ocean_heat_capacity=ebm_sample[1:4, isample],
        ocean_heat_transfer=ebm_sample[4:7, isample],
        deep_ocean_efficacy=ebm_sample[7, isample],
        gamma_autocorrelation=ebm_sample[0, isample],
        sigma_xi=ebm_sample[9, isample],
        sigma_eta=ebm_sample[8, isample],
        forcing_4co2=ebm_sample[10, isample],
        stochastic_run=True,
    )
    eb_matrix = ebm._eb_matrix()
    q_mat = np.zeros((4, 4))
    q_mat[0, 0] = ebm.sigma_eta**2
    q_mat[1, 1] = (ebm.sigma_xi / ebm.ocean_heat_capacity[0]) ** 2
    h_mat = np.zeros((8, 8))
    h_mat[:4, :4] = -eb_matrix
    h_mat[:4, 4:] = q_mat
    h_mat[4:, 4:] = eb_matrix.T
    g_mat = scipy.sparse.linalg.expm(h_mat)
    q_mat_d = g_mat[4:, 4:].T @ g_mat[:4, 4:]
    q_mat_d = q_mat_d.astype(np.float64)

    # I can't work out exactly what checks scipy is doing to decide the param
    # set is a fail. Best to just let it tell me if it likes it or not.
    try:
        scipy.stats.multivariate_normal.rvs(size=1, mean=np.zeros(4), cov=q_mat_d)
    except:  # noqa: E722
        ebm_sample[:, isample] = np.nan

mask = np.all(np.isnan(ebm_sample), axis=0)
ebm_sample = ebm_sample[:, ~mask]

print("Total number of retained samples:", len(ebm_sample.T))

ebm_sample_df = pd.DataFrame(
    data=ebm_sample[:, :samples].T,
    columns=[
        "gamma",
        "c1",
        "c2",
        "c3",
        "kappa1",
        "kappa2",
        "kappa3",
        "epsilon",
        "sigma_eta",
        "sigma_xi",
        "F_4xCO2",
    ],
)

assert len(ebm_sample_df) >= samples


ebm_sample_df.to_csv(
    f"../data/external/samples_for_priors/climate_response_ebm3_{samples}.csv",
    index=False,
)

print(
    "CO2 scaling factor check:",
    np.percentile(
        1
        + 0.563
        * (ebm_sample_df["F_4xCO2"].mean() - ebm_sample_df["F_4xCO2"])
        / ebm_sample_df["F_4xCO2"].mean(),
        (5, 50, 95),
    ),
)

# what we do want to do is to scale the variability in 4xCO2 (correlated with the other
# EBM parameters)
# to feed into the effective radiative forcing scaling factor.

f = FAIR()


f.define_time(1750, 1752, 1)

scenarios = ['ssp119']
f.define_scenarios(scenarios)

f.define_configs(list(range(samples)))

species, properties = read_properties()

f.define_species(species, properties)

f.allocate()
f.fill_species_configs()
f.fill_from_rcmip()

initialise(f.concentration, f.species_configs['baseline_concentration'])
initialise(f.forcing, 0)
initialise(f.temperature, 0)
initialise(f.cumulative_emissions, 0)
initialise(f.airborne_emissions, 0)

capacities = [4.22335014, 16.5073541, 86.1841127]
kappas = [1.31180598, 2.61194068, 0.92986733]
epsilon = 1.29020599
fill(f.climate_configs['ocean_heat_capacity'], capacities)
fill(f.climate_configs['ocean_heat_transfer'], kappas)
fill(f.climate_configs['deep_ocean_efficacy'], epsilon)



fill(
    f.climate_configs["ocean_heat_capacity"],
    np.array([ebm_sample_df["c1"], ebm_sample_df["c2"], ebm_sample_df["c3"]]).T,
)
fill(
    f.climate_configs["ocean_heat_transfer"],
    np.array([ebm_sample_df["kappa1"], ebm_sample_df["kappa2"], ebm_sample_df["kappa3"]]).T,
)
fill(f.climate_configs["deep_ocean_efficacy"], ebm_sample_df["epsilon"])
fill(f.climate_configs["gamma_autocorrelation"], ebm_sample_df["gamma"])
# fill(f.climate_configs["sigma_eta"], ebm_sample_df["sigma_eta"])
# fill(f.climate_configs["sigma_xi"], ebm_sample_df["sigma_xi"])
# fill(f.climate_configs["seed"], ebm_sample_df["seed"])
# fill(f.climate_configs["stochastic_run"], True)
# fill(f.climate_configs["use_seed"], True)
fill(f.climate_configs["forcing_4co2"], ebm_sample_df["F_4xCO2"])

f.run()


ebms = multi_ebm(
    f.configs,
    ocean_heat_capacity=f.climate_configs["ocean_heat_capacity"],
    ocean_heat_transfer=f.climate_configs["ocean_heat_transfer"],
    deep_ocean_efficacy=f.climate_configs["deep_ocean_efficacy"],
    stochastic_run=f.climate_configs["stochastic_run"],
    sigma_eta=f.climate_configs["sigma_eta"],
    sigma_xi=f.climate_configs["sigma_xi"],
    gamma_autocorrelation=f.climate_configs["gamma_autocorrelation"],
    seed=f.climate_configs["seed"],
    use_seed=f.climate_configs["use_seed"],
    forcing_4co2=f.climate_configs["forcing_4co2"],
    timestep=f.timestep,
    timebounds=f.timebounds,
)

ecs_tcr_dict = {}
ecs_tcr_dict['ecs'] = ebms.ecs
ecs_tcr_dict['tcr'] = ebms.tcr

df_ecs_tcr = pd.DataFrame(data=ecs_tcr_dict, columns=ecs_tcr_dict.keys())

df_ecs_tcr.to_csv(
    f"../data/external/samples_for_priors/ecs_tcs_{samples}.csv",
    index=False,
)