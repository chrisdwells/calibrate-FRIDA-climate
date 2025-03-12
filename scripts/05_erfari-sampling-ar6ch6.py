import os
import numpy as np
import pandas as pd
import pooch
import scipy.stats
from dotenv import load_dotenv

# Adapted from FaIR calibrate

# Reduced FaIR: just SO2

# Use the AR6 per-species ERFari calibrations, from Chapter 6 Fig. 6.12. This includes
# contibutions from CH4, N2O and HCs.

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

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
    "Sulfur",
]

species_out = {}
for ispec, species in enumerate(emitted_species):
    species_rcmip_name = species.replace("-", "")
    emis_in = (
        df_emis.loc[
            (df_emis["Scenario"] == "ssp245")
            & (df_emis["Variable"].str.endswith("|" + species_rcmip_name))
            & (df_emis["Region"] == "World"),
            "1750":"2100",
        ]
        .interpolate(axis=1)
        .values.squeeze()
    )
    species_out[species] = emis_in

species_df = pd.DataFrame(species_out, index=range(1750, 2101))


# these come from AR6 WG1, yes?
erfari_emitted = pd.Series(
    {
        "Sulfur": -0.234228,
    }
)

# erfari radiative efficiency per Mt or ppb or ppt
re = erfari_emitted / (species_df.loc[2019, :] - species_df.loc[1750, :])
re.dropna(inplace=True)

scalings = scipy.stats.uniform.rvs(
    np.minimum(re * 2, 0),
    np.maximum(re * 2, 0) - np.minimum(re * 2, 0),
    size=(samples, 1),
    random_state=3729329,
)

df = pd.DataFrame(scalings, columns=['ari Sulfur'])

df.to_csv(
    f"../data/external/samples_for_priors/aerosol_radiation_{samples}.csv",
    index=False,
)

