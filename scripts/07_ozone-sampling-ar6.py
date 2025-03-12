import copy
import os
import numpy as np
import pandas as pd
import pooch
import scipy.stats
from dotenv import load_dotenv
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

# Adapted from FaIR calibrate

# out for FRIDAv1; back in for v2, with EESC brought in externally. 

# concentrations are midyear rather than endyear so are six months out, but it won't
# be a biggie.

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

# now include temperature feedback
Tobs = pd.read_csv("../data/external/forcing/AR6_GMST.csv", index_col=0).values

delta_gmst = [
    0,
    Tobs[65:76].mean(),
    Tobs[75:86].mean(),
    Tobs[85:96].mean(),
    Tobs[95:106].mean(),
    Tobs[105:116].mean(),
    Tobs[115:126].mean(),
    Tobs[125:136].mean(),
    Tobs[135:146].mean(),
    Tobs[145:156].mean(),
    Tobs[152:163].mean(),
    Tobs[155:166].mean(),
    Tobs[159:170].mean(),
    Tobs[167].mean(),  # we don't use this
    Tobs[168].mean(),
]
warming_pi_pd = Tobs[159:170].mean()

good_models = [
    "BCC-ESM1",
    "CESM2(WACCM6)",
    "GFDL-ESM4",
    "GISS-E2-1-H",
    "MRI-ESM2-0",
    "OsloCTM3",
]
skeie_trop = pd.read_csv(
    "../data/external/forcing/skeie_ozone_trop.csv", index_col=0
)
skeie_trop = skeie_trop.loc[good_models]
skeie_trop.insert(0, 1850, 0)
skeie_trop.columns = pd.to_numeric(skeie_trop.columns)
skeie_trop.interpolate(axis=1, method="values", limit_area="inside", inplace=True)

skeie_strat = pd.read_csv(
    "../data/external/forcing/skeie_ozone_strat.csv", index_col=0
)
skeie_strat = skeie_strat.loc[good_models]
skeie_strat.insert(0, 1850, 0)
skeie_strat.columns = pd.to_numeric(skeie_strat.columns)
skeie_strat.interpolate(axis=1, method="values", limit_area="inside", inplace=True)

skeie_total = skeie_trop + skeie_strat

coupled_models = copy.deepcopy(good_models)
coupled_models.remove("OsloCTM3")

skeie_total.loc[coupled_models] = skeie_total.loc[coupled_models] - (-0.037) * np.array(
    delta_gmst
)
skeie_ssp245 = skeie_total.mean()
skeie_ssp245[1750] = -0.03
skeie_ssp245.sort_index(inplace=True)
skeie_ssp245 = skeie_ssp245 + 0.03
skeie_ssp245.drop([2014, 2017, 2020], inplace=True)
skeie_ssp245 = skeie_ssp245._append(
    skeie_total.loc["OsloCTM3", 2014:]
    - skeie_total.loc["OsloCTM3", 2010]
    + skeie_ssp245[2010]
)

f = interp1d(
    skeie_ssp245.index, skeie_ssp245, bounds_error=False, fill_value="extrapolate"
)
years = np.arange(1750, 2021)
o3total = f(years)

print("2014-1750 ozone ERF from Skeie:", o3total[264])
print("2019-1750 ozone ERF from Skeie:", o3total[269])
print("2014-1850 ozone ERF from Skeie:", o3total[264] - o3total[100])

rcmip_emissions_file = pooch.retrieve(
    url="doi:10.5281/zenodo.4589756/rcmip-emissions-annual-means-v5-1-0.csv",
    known_hash="md5:4044106f55ca65b094670e7577eaf9b3",
)

rcmip_concentration_file = pooch.retrieve(
    url=("doi:10.5281/zenodo.4589756/" "rcmip-concentrations-annual-means-v5-1-0.csv"),
    known_hash="md5:0d82c3c3cdd4dd632b2bb9449a5c315f",
)

df_emis = pd.read_csv(rcmip_emissions_file)
df_conc = pd.read_csv(rcmip_concentration_file)

emitted_species = [
    "NOx",
    "VOC",
    "CO",
]

concentration_species = [
    "CH4",
    "N2O",
    ]
    
species_out = {}
for ispec, species in enumerate(emitted_species):
    emis_in = (
        df_emis.loc[
            (df_emis["Scenario"] == "ssp245")
            & (df_emis["Variable"].str.endswith("|" + species))
            & (df_emis["Region"] == "World"),
            "1750":"2020",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    species_out[species] = emis_in[:-1]

# Adjust NOx for units error in BB
gfed_sectors = [
    "Emissions|NOx|MAGICC AFOLU|Agricultural Waste Burning",
    "Emissions|NOx|MAGICC AFOLU|Forest Burning",
    "Emissions|NOx|MAGICC AFOLU|Grassland Burning",
    "Emissions|NOx|MAGICC AFOLU|Peat Burning",
]
species_out["NOx"] = (
    df_emis.loc[
        (df_emis["Scenario"] == "ssp245")
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"].isin(gfed_sectors)),
        "1750":"2020",
    ]
    .interpolate(axis=1)
    .values.squeeze()
    .sum(axis=0)
    * 46.006
    / 30.006
    + df_emis.loc[
        (df_emis["Scenario"] == "ssp245")
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"] == "Emissions|NOx|MAGICC AFOLU|Agriculture"),
        "1750":"2020",
    ]
    .interpolate(axis=1)
    .values.squeeze()
    + df_emis.loc[
        (df_emis["Scenario"] == "ssp245")
        & (df_emis["Region"] == "World")
        & (df_emis["Variable"] == "Emissions|NOx|MAGICC Fossil and Industrial"),
        "1750":"2020",
    ]
    .interpolate(axis=1)
    .values.squeeze()
)[:-1]


for ispec, species in enumerate(concentration_species):
    species_rcmip_name = species.replace("-", "")
    conc_in = (
        df_conc.loc[
            (df_conc["Scenario"] == "ssp245")
            & (df_conc["Variable"].str.endswith("|" + species_rcmip_name))
            & (df_conc["Region"] == "World"),
            "1750":"2020",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    species_out[species] = conc_in[:-1]


eesc_file = '../data/external/inputs_for_frida/EESC.csv'

df_eesc = pd.read_csv(eesc_file)

total_eesc = df_eesc[(years[0]<=df_eesc.Year) & (df_eesc.Year<=years[-2])]['EESC']


delta_Cch4 = species_out["CH4"][264] - species_out["CH4"][0]
delta_Cn2o = species_out["N2O"][264] - species_out["N2O"][0]
delta_Cods = total_eesc[264] - total_eesc[0]
delta_Eco = species_out["CO"][264] - species_out["CO"][0]
delta_Evoc = species_out["VOC"][264] - species_out["VOC"][0]
delta_Enox = species_out["NOx"][264] - species_out["NOx"][0]

# best estimate radiative efficiencies from 2014 - 1850 from AR6 here
radeff_ch4 = 0.14 / delta_Cch4
radeff_n2o = 0.03 / delta_Cn2o
radeff_ods = -0.11 / delta_Cods
radeff_co = 0.067 / delta_Eco  # stevenson CMIP5 scaled to CO + VOC total
radeff_voc = 0.043 / delta_Evoc  # stevenson CMIP5 scaled to CO + VOC total
radeff_nox = 0.20 / delta_Enox


fac_cmip6_skeie = (
    radeff_ch4 * delta_Cch4
    + radeff_n2o * delta_Cn2o
    + radeff_ods * delta_Cods
    + radeff_co * delta_Eco
    + radeff_voc * delta_Evoc
    + radeff_nox * delta_Enox
) / (o3total[264] - o3total[0])
ts = np.vstack(
    (
        species_out["CH4"],
        species_out["N2O"],
        total_eesc,
        species_out["CO"],
        species_out["VOC"],
        species_out["NOx"],
    )
).T


def fit_precursors(x, rch4, rn2o, rods, rco, rvoc, rnox):
    return (
        rch4 * x[0] + rn2o * x[1] + rods * x[2] + rco * x[3] + rvoc * x[4] + rnox * x[5]
    )


p, cov = curve_fit(
    fit_precursors,
    ts[:270, :].T - ts[0:1, :].T,
    o3total[:270] - o3total[0],
    bounds=(  # assumed likely range from Thornhill - maybe could be wider?
        (
            0.09 / delta_Cch4 / fac_cmip6_skeie,
            0.01 / delta_Cn2o / fac_cmip6_skeie,
            -0.21 / delta_Cods / fac_cmip6_skeie,
            0.010 / delta_Eco / fac_cmip6_skeie,
            0 / delta_Evoc / fac_cmip6_skeie,
            0.09 / delta_Enox / fac_cmip6_skeie,
        ),
        (
            0.19 / delta_Cch4 / fac_cmip6_skeie,
            0.05 / delta_Cn2o / fac_cmip6_skeie,
            -0.01 / delta_Cods / fac_cmip6_skeie,
            0.124 / delta_Eco / fac_cmip6_skeie,
            0.086 / delta_Evoc / fac_cmip6_skeie,
            0.31 / delta_Enox / fac_cmip6_skeie,
        ),
    ),
)

forcing = (
    p[0] * (species_out["CH4"] - species_out["CH4"][0])
    + p[1] * (species_out["N2O"] - species_out["N2O"][0])
    + p[2] * (total_eesc - total_eesc[0])
    + p[3] * (species_out["CO"] - species_out["CO"][0])
    + p[4] * (species_out["VOC"] - species_out["VOC"][0])
    + p[5] * (species_out["NOx"] - species_out["NOx"][0])
)


print(p)  # these coefficients we export to the ERF time series
# print(radeff_ch4, radeff_n2o, radeff_ods, radeff_co, radeff_voc, radeff_nox)

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

scalings = scipy.stats.norm.rvs(
    loc=np.array(p),
    scale=np.array(
        [
            0.05 / delta_Cch4 / fac_cmip6_skeie,
            0.02 / delta_Cn2o / fac_cmip6_skeie,
            0.10 / delta_Cods / fac_cmip6_skeie,
            0.057 / delta_Eco / fac_cmip6_skeie,
            0.043 / delta_Evoc / fac_cmip6_skeie,
            0.11 / delta_Enox / fac_cmip6_skeie,
        ]
    )
    # scale=np.array([0.000062, 0.000471, 0.000113, 0.000131, 0.000328, 0.000983])
    / NINETY_TO_ONESIGMA,
    size=(samples, 6),
    random_state=52,
)

df = pd.DataFrame(
    scalings,
    columns=[
        "o3 CH4",
        "o3 N2O",
        "o3 Equivalent effective stratospheric chlorine",
        "o3 CO",
        "o3 VOC",
        "o3 NOx",
    ],
)

df.to_csv(
    f"../data/external/samples_for_priors/ozone_{samples}.csv",
    index=False,
)

