import pandas as pd
import os
from dotenv import load_dotenv
import numpy as np

# this collates the input parameters for the priors - so we take the ocean
# parameters (on ocean_samples) and the FaIR ones (on samples), and combine
# to the full set, on samples

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
ocean_samples = int(os.getenv("OCEAN_SPINUP_SAMPLES"))


# load FaIR parameters - simple as these are just  on samples

csv_list = ['aerosol_cloud', 'aerosol_radiation', 'carbon_cycle', 
           'climate_response_ebm3', 'forcing_scaling', 'ozone']

df_in = pd.DataFrame()

for csv in csv_list:
    
    df_csv = pd.read_csv(f"../data/external/samples_for_priors/{csv}_{samples}.csv")
    
    df_in = pd.concat([df_in, df_csv], axis=1)
    

fair_data_dict = {}

fair_vars_to_frida = {
'c1':'Energy Balance Model.Heat Capacity of Land & Ocean Surface',
'c2':'Energy Balance Model.Heat Capacity of Thermocline Ocean',
'c3':'Energy Balance Model.Heat Capacity of Deep Ocean',
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
    fair_data_dict[fair_vars_to_frida[var]] = df_in[var]

df_fair_data = pd.DataFrame(data=fair_data_dict, columns=fair_data_dict.keys())


#%%

# get valid members of ocean spinup

df_ocean_spinup_tests = pd.read_csv("../data/spinup_output/Ocean_spinup_output_end_tests.csv")

idxs = np.full(ocean_samples, np.nan)

for i in np.arange(ocean_samples):
    if np.mean(np.abs(df_ocean_spinup_tests[f'="Run {i+1}: Ocean.Air sea co2 flux[1]"'])) < 0.01:
        idxs[i] = i

idxs = idxs[~np.isnan(idxs)]
        
n_kept = idxs.shape[0]

n_repeats = int(np.ceil(samples/n_kept))

#%%


# import matplotlib.pyplot as plt

# variable_stock_list = ['Ocean.Air sea co2 flux[1]',
#                         'Ocean.Deep ocean ocean carbon reservoir[1]',
#                         ]

# df_ocean_spinup_tests = pd.read_csv("../data/spinup_output/Ocean_spinup_output_end_tests.csv")


# ocean_stocks_dict = {}

# for var in variable_stock_list:
#     ocean_stocks_dict[var] = np.full((101, ocean_samples), np.nan)
#     for n_i in np.arange(ocean_samples):
#         data_in = df_ocean_spinup_tests[f'="Run {n_i+1}: {var}"'].values
#         ocean_stocks_dict[var][:,n_i] = data_in
        
# fig = plt.figure()

# ax = fig.add_subplot(121)
# ax.plot(np.arange(101), ocean_stocks_dict['Ocean.Air sea co2 flux[1]'])
# ax.set_xlim([95,100])

# ax = fig.add_subplot(122)
# ax.plot(np.arange(101), ocean_stocks_dict['Ocean.Deep ocean ocean carbon reservoir[1]'])
# ax.set_xlim([80,100])

# plt.tight_layout()

# fig = plt.figure(figsize=(8, 8))

# for yr in np.arange(3):
    
#     flux_data = ocean_stocks_dict['Ocean.Air sea co2 flux[1]'][yr,:]
#     flux_data.sort()
    
#     plt.plot(np.arange(1000), flux_data, label=yr)
    
# # plt.legend()

# plt.ylabel('Ocean.Air sea co2 flux[1]')
# plt.xlabel('1000 members')

# plt.ylim([-0.2, 0.2])
    
    
    
    


        
        #%%

# pull in ocean data

df_ocean_data = pd.DataFrame()


# stocks - output from spinup

variable_stock_list = ['Ocean.Cold surface ocean pH[1]', 
                    'Ocean.Warm surface ocean pH[1]',
                    'Ocean.Cold surface ocean carbon reservoir[1]', 
                    'Ocean.Warm surface ocean carbon reservoir[1]', 
                    'Ocean.Intermediate depth ocean carbon reservoir[1]',
                    'Ocean.Deep ocean ocean carbon reservoir[1]']

df_ocean_spinup_output = pd.read_csv("../data/spinup_output/Ocean_spinup_output_end.csv")


ocean_stocks_dict = {}


variable_stock_list_frida = []
for variable_stock in variable_stock_list:
    variable_stock_list_frida.append(variable_stock.split(".")[0
                       ] + '.Initial ' + variable_stock.split(".")[1][:-3])

df_out = pd.DataFrame(columns=[variable_stock_list_frida])


for var in variable_stock_list:
    var_init = var.split(".")[0] + '.Initial ' + var.split(".")[1][:-3]
    
    ocean_stocks_dict[var_init] = np.full((ocean_samples), np.nan)
    for n_i in np.arange(ocean_samples):
        data_in = df_ocean_spinup_output[f'="Run {n_i+1}: {var}"'].values[0]
        ocean_stocks_dict[var_init][n_i] = data_in

df_ocean_stocks = pd.DataFrame(data=ocean_stocks_dict, columns=ocean_stocks_dict.keys())

df_ocean_data = pd.concat([df_ocean_data, df_ocean_stocks], axis=1)


# original ocean params

df_ocean_params = pd.read_csv(f"../data/spinup_input/ocean_spinup_params_{ocean_samples}.csv")

df_ocean_data = pd.concat([df_ocean_data, df_ocean_params], axis=1)

# 1750 co2


df_1750_co2 = pd.read_csv(f"../data/spinup_input/co2_1750_{ocean_samples}.csv")

df_1750_co2 = df_1750_co2.rename(columns={"Ocean.Atmospheric CO2 Concentration 1750": 
                            "CO2 Forcing.Atmospheric CO2 Concentration 1750"})

df_ocean_data = pd.concat([df_ocean_data, df_1750_co2], axis=1)

#%%

# filter by idx 

df_ocean_data_filtered = df_ocean_data.iloc[idxs]

# repeat along index as needed, then crop down to match samples

df_ocean_data_out = pd.concat([df_ocean_data_filtered]*n_repeats, ignore_index=True)

df_ocean_data_out = df_ocean_data_out.iloc[:samples]

#%%

# ocean params which aren't varied in spinup (as are Temperature sensitivities)
# ie these are just on samples

df_ocean_temp_params = pd.read_csv(f"../data/priors_input/ocean_priors_params_{samples}.csv")

df_ocean_data_out = pd.concat([df_ocean_data_out, df_ocean_temp_params], axis=1)




#%%

# combine fair and ocean data

df_combined = pd.concat([df_fair_data, df_ocean_data_out], axis=1)


#%%

df_combined.to_csv(
    f"../data/priors_input/priors_inputs_{samples}.csv",
    index=False,
)
