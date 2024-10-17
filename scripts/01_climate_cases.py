import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))

samples = 1000

climate_case_data = {}

run_list = []
for i in np.arange(samples):
    run_list.append(f"Run {i}")

climate_case_data[''] = run_list
climate_case_data['Ocean.selected climate case[1]'] = 1+np.arange(samples)

df_climate_case = pd.DataFrame(data=climate_case_data, columns=climate_case_data.keys())

df_climate_case.to_csv(
    f"../data/spinup_input/climate_cases_{samples}.csv",
    index=False,
)
