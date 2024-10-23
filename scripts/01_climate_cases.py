import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np

# ocean_samples is the # of ocean members ran. This is a factor of the total 
# sample number of priors. This is done to reduce computation as the spinups are long

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
ocean_samples = int(os.getenv("OCEAN_SPINUP_SAMPLES"))

climate_case_data_full = {}

run_list = []
for i in np.arange(samples):
    run_list.append(f"Run {i}")

climate_case_data_full[''] = run_list
climate_case_data_full['Climate Units.selected climate case[1]'] = 1+np.arange(samples)

climate_case_data_full = pd.DataFrame(data=climate_case_data_full, 
                                      columns=climate_case_data_full.keys())

climate_case_data_full.to_csv(
    f"../data/priors_input/climate_cases_{samples}.csv",
    index=False,
)


climate_case_data_ocean_spinup = {}

run_list = []
for i in np.arange(ocean_samples):
    run_list.append(f"Run {i}")

climate_case_data_ocean_spinup[''] = run_list
climate_case_data_ocean_spinup['Ocean.selected climate case[1]'] = 1+np.arange(ocean_samples)

climate_case_data_ocean_spinup = pd.DataFrame(data=climate_case_data_ocean_spinup, 
                                      columns=climate_case_data_ocean_spinup.keys())

climate_case_data_ocean_spinup.to_csv(
    f"../data/spinup_input/climate_cases_{ocean_samples}.csv",
    index=False,
)

