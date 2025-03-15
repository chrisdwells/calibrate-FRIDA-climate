import os
import pandas as pd
import scipy.stats
from dotenv import load_dotenv
import numpy as np

# cribbed from fair calibrate. feed these into Ocean_spinup_start. includes
# atmos co2 1750 as this affects the ocean spinup

# Run this, and then Ocean_spinup_start 

load_dotenv()

ocean_samples = int(os.getenv("OCEAN_SPINUP_SAMPLES"))

ocean_variables = {
    "Ocean.Depth of warm surface ocean layer[1]":[50,500],
    "Ocean.Thickness of intermediate ocean layer[1]":[300,1000],
    "Ocean.Depth of cold surface ocean layer[1]":[50,500],
    "Ocean.Reference overturning strength in Sv[1]":[10,30],
    "Ocean.Reference intermediate to warm surface ocean mixing strength[1]":[50,90],
    "Ocean.Reference cold surface to deep ocean mixing strength[1]":[10,30],
    "Ocean.Reference strength of biological carbon pump in low latitude ocean[1]":[0,3],
    "Ocean.Reference strength of biological carbon pump in high latitude ocean[1]":[4,12],
    "Ocean.High latitude carbon pump transfer efficiency[1]":[0.1,0.5],
    }

param_dict = {}


run_list = []
for i in np.arange(ocean_samples):
    run_list.append(f'Run {i+1}')
param_dict['Run'] = run_list

for o_i, ocean_var in enumerate(ocean_variables):
    
    param_dict[ocean_var] = scipy.stats.uniform.rvs(
        ocean_variables[ocean_var][0],
        ocean_variables[ocean_var][1] - ocean_variables[ocean_var][0],
        size=ocean_samples,
        random_state=3729329 + 1000*o_i,
    )
    

df = pd.DataFrame(param_dict, columns=param_dict.keys())


NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)
co2_1750_conc = scipy.stats.norm.rvs(
    size=ocean_samples, loc=278.3, scale=2.9 / NINETY_TO_ONESIGMA, random_state=1067061
)

df_co2 = pd.DataFrame({"Ocean.Atmospheric CO2 Concentration 1750[1]": co2_1750_conc})

df = pd.concat([df, df_co2], axis=1)

os.makedirs("../data/spinup_input/", exist_ok=True)
df.to_csv(
    f"../data/spinup_input/ocean_spinup_params_{ocean_samples}.csv",
    index=False,
)


# make blank csvs if needed for output from spinup and priors

os.makedirs("../data/spinup_output/", exist_ok=True)
os.makedirs("../data/priors_output/", exist_ok=True)
os.makedirs("../data/posteriors_output/", exist_ok=True)


needed_csvs = [
    '../data/spinup_output/Ocean_spinup_output_end.csv',
    '../data/spinup_output/Ocean_spinup_output_end_tests.csv',
    '../data/spinup_output/Ocean_spinup_output_start.csv',

    '../data/priors_output/priors_1980_initials.csv',
    '../data/priors_output/priors_aerosols.csv',
    '../data/priors_output/priors_CO2.csv',
    '../data/priors_output/priors_ocean_CO2_flux.csv',
    '../data/priors_output/priors_ocean_heat_content.csv',
    '../data/priors_output/priors_temperature.csv',
    
    '../data/posteriors_output/posteriors_1980_initials.csv',
    '../data/posteriors_output/posteriors_aerosols.csv',
    '../data/posteriors_output/posteriors_CO2.csv',
    '../data/posteriors_output/posteriors_ocean_CO2_flux.csv',
    '../data/posteriors_output/posteriors_ocean_heat_content.csv',
    '../data/posteriors_output/posteriors_temperature.csv',
    
    
    ]

for csv in needed_csvs:
    if os.path.isfile(csv) == False:
        df_blank = pd.DataFrame(list())
        df_blank.to_csv(csv)

