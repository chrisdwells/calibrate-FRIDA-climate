import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np


load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))


csv_list = ['aerosol_cloud', 'aerosol_radiation', 'carbon_cycle', 
           'climate_response_ebm3', 'forcing_scaling', 'ozone']

df_in = pd.DataFrame()

for csv in csv_list:
    
    df_csv = pd.read_csv(f"../data/external/samples_for_priors/{csv}_{samples}.csv")
    
    df_in = pd.concat([df_in, df_csv], axis=1)
    


# CO2 stored in spinup as used in ocean spinup
df_csv = pd.read_csv(f"../data/spinup_input/co2_1750_{samples}.csv")

df_in = pd.concat([df_in, df_csv], axis=1)


median_data = {}
full_data = {}

fair_vars_to_frida = {
'c1':'Energy Balance Model.Heat Capacity of Land & Ocean Surface',
'c2':'Energy Balance Model.Heat Capacity of Thermocline Ocean',
'c3':'Energy Balance Model.Heat Capacity of Deep Ocean',
'Ocean.Atmospheric CO2 Concentration 1750':'CO2 Forcing.Atmospheric CO2 Concentration 1750',
'epsilon':'Energy Balance Model.Deep Ocean Heat Uptake Efficacy Factor',
'kappa1':'Energy Balance Model.Heat Transfer Coefficient between Land & Ocean Surface and Space',
'kappa2':'Energy Balance Model.Heat Transfer Coefficient between Surface and Thermocline Ocean',
'kappa3':'Energy Balance Model.Heat Transfer Coefficient between Thermocline Ocean and Deep Ocean',
'beta':'Aerosol Forcing.Scaling Aerosol Cloud Interactions Effective Radiative Forcing scaling factor',
'shape Sulfur':'Aerosol Forcing.Logarithmic Aerosol Cloud Interactions Effective Radiative Forcing scaling factor',
'ari Sulfur':'Aerosol Forcing.Effective Radiative Forcing from Aerosol Radiation Interactions per unit SO2 Emissions',
'rA':'CO2 Forcing.Effect of atmospheric CO2 on CO2 lifetime parameter',
'rT':'CO2 Forcing.Effect of temperature on CO2 lifetime parameter',
'rU':'CO2 Forcing.Effect of CO2 uptake on CO2 lifetime parameter',
'r0':'CO2 Forcing.Baseline CO2 lifetime parameter',
'scale CH4':'CH4 Forcing.Calibration scaling of CH4 forcing',
'scale N2O':'N2O Forcing.Calibration scaling of N2O forcing',
'scale minorGHG':'Minor GHGs Forcing.Calibration scaling of Minor GHG forcing',
'scale Stratospheric water vapour':'Stratospheric Water Vapour Forcing.Calibration scaling of Stratospheric H2O forcing',
'scale Light absorbing particles on snow and ice':'BC on Snow Forcing.Calibration scaling of Black Carbon on Snow forcing',
'scale Albedo':'Land Use Forcing.Calibration scaling of Albedo forcing',
'scale Irrigation':'Land Use Forcing.Calibration scaling of Irrigation forcing',
'scale Volcanic':'Natural Forcing.Calibration scaling of Volcano forcing',
'scale CO2':'CO2 Forcing.Calibration scaling of CO2 forcing',
'solar_amplitude':'Natural Forcing.Amplitude of Effective Radiative Forcing from Solar Output Variations',
'solar_trend':'Natural Forcing.Linear trend in Effective Radiative Forcing from Solar Output Variations',
'o3 CH4':'Ozone Forcing.Ozone forcing per unit CH4 concentration change',
'o3 N2O':'Ozone Forcing.Ozone forcing per unit N2O concentration change',
'o3 Equivalent effective stratospheric chlorine':'Ozone Forcing.Ozone forcing per unit Montreal gases equivalent effective stratospheric chlorine concentration change',
'o3 CO':'Ozone Forcing.Ozone forcing per unit CO emissions change',
'o3 VOC':'Ozone Forcing.Ozone forcing per unit VOC emissions change',
'o3 NOx':'Ozone Forcing.Ozone forcing per unit NOx emissions change',
        }

for var in fair_vars_to_frida.keys():
    median_data[fair_vars_to_frida[var]] = np.percentile(df_in[var],50)
    print(f'{var}: {median_data[fair_vars_to_frida[var]]}')
    full_data[fair_vars_to_frida[var]] = df_in[var]

df_all = pd.DataFrame(data=full_data, columns=full_data.keys())


df_all.to_csv(
    f"../data/priors_input/temperature_parameters_{samples}.csv",
    index=False,
)

climate_case_data = {}

run_list = []
for i in np.arange(samples):
    run_list.append(f"Run {i}")

climate_case_data[''] = run_list
climate_case_data['Climate Units.selected climate case[1]'] = 1+np.arange(samples)

df_climate_case = pd.DataFrame(data=climate_case_data, columns=climate_case_data.keys())

df_climate_case.to_csv(
    f"../data/priors_input/climate_cases_{samples}.csv",
    index=False,
)
