import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

runids = np.loadtxt(
    "../data/constraining/runids_rmse_reweighted_pass.csv",
).astype(np.int64)

df_prior_params = pd.read_csv(f"../data/priors_input/priors_inputs_{samples}.csv")



# don't want the Initial stocks for the historical runs, but want the parameters
df_prior_params = df_prior_params.loc[
    :,~df_prior_params.columns.str.contains('Ocean.Initial')]

run_params = df_prior_params.values[runids]

columns_params = []
for col_in in df_prior_params.keys():
    columns_params.append(col_in[:-3])
    
df_run_params = pd.DataFrame(data=run_params, columns=columns_params)




df_1980_inits = pd.read_csv("../data/priors_output/priors_1980_initials.csv")

#%%

# Forgot to include CO2 stock in outputs from priors. Have fudged by running the 
# posteriors back through and pulling it out for just those members, but 
# now is set up so it should work if ran again, so this bit could be removed.

if '="Run 1: CO2 Forcing.Atmospheric CO2 mass anomaly since 1750[1]"' in df_1980_inits.keys():
    print('now time to remove this part')

# TO DO remove this fudge
df_1980_inits_co2 = pd.read_csv("../data/priors_output/priors_1980_initials_100_members.csv")


df_1980_inits_co2_out = pd.DataFrame(columns=['CO2 Forcing.Initial Atmospheric CO2 mass anomaly since 1750'])


for n_i in np.arange(output_ensemble_size):
    row = []
    
    row.append(df_1980_inits_co2[
        f'="Run {n_i+1}: CO2 Forcing.Atmospheric CO2 mass anomaly since 1750[1]"'].values[0])

    df_1980_inits_co2_out.loc[n_i] = row
   


#%%

df_temperature = pd.read_csv("../data/priors_output/priors_temperature.csv")


variable_stock_list = ['Ocean.Cold surface ocean pH[1]', 
                    'Ocean.Warm surface ocean pH[1]',
                    'Ocean.Cold surface ocean carbon reservoir[1]', 
                    'Ocean.Warm surface ocean carbon reservoir[1]', 
                    'Ocean.Intermediate depth ocean carbon reservoir[1]',
                    'Ocean.Deep ocean ocean carbon reservoir[1]',
                                    
                    'Energy Balance Model.Land & Ocean Surface Temperature[1]', 
                    'Energy Balance Model.Thermocline Ocean Temperature[1]', 
                    'Energy Balance Model.Deep Ocean Temperature[1]', 
                    
                    'CH4 Forcing.CH4 in atmosphere[1]',
                    ]

variable_stock_list_frida = []
for variable_stock in variable_stock_list:
    variable_stock_list_frida.append(variable_stock.split(".")[0
                       ] + '.Initial ' + variable_stock.split(".")[1][:-3])

column_list = variable_stock_list_frida + ['Energy Balance Model.Surface Temperature 1850 to 1900 offset relative to 1750']

df_inits_out = pd.DataFrame(columns=column_list)


for n_i in np.arange(samples):
    row = []
    
    for stock in variable_stock_list:
         row.append(df_1980_inits[f'="Run {n_i+1}: {stock}"'].values[0])
    
    
    row.append(np.mean(df_temperature[f'="Run {n_i+1}: Energy Balance Model.Land & Ocean Surface Temperature[1]"'
                 ].values[:51]))

    df_inits_out.loc[n_i] = row



run_inits = df_inits_out.values[runids]
df_run_inits = pd.DataFrame(data=run_inits, columns=df_inits_out.keys())


df_run_inits = pd.concat([df_run_inits, df_1980_inits_co2_out], axis=1)



df_combined = pd.concat([df_run_params, df_run_inits], axis=1)


df_combined.to_csv(
    f"../data/constraining/frida_climate_inputs_{output_ensemble_size}_from_{samples}.csv",
    index=False,
)

#%%

fixed_stock_list = [
    # 'CO2 Forcing.Cumulative CO2 emissions[1]', 
    # 'N2O Forcing.Cumulative N2O emissions[1]', 
    'N2O Forcing.N2O in atmosphere[1]',
    'Minor GHGs Forcing.HFC134a eq in atmosphere[1]']

fixed_stock_list_frida = []
for fixed_stock in fixed_stock_list:
    fixed_stock_list_frida.append(fixed_stock.split(".")[0
                       ] + '.Initial ' + fixed_stock.split(".")[1][:-3])

df_out_fixed = pd.DataFrame(columns=fixed_stock_list_frida)
row = []
for stock in fixed_stock_list:
     row.append(df_1980_inits[f'="Run 1: {stock}"'].values[0])


df_out_fixed.loc[0] = row

# need to internalise this when emissions change
df_out_fixed['CO2 Forcing.Initial Cumulative CO2 emissions'] = 1128941.22245
df_out_fixed['N2O Forcing.Initial Cumulative N2O emissions'] = 369316.190142

df_out_fixed.to_csv('../data/constraining/frida_climate_inputs_constant_stocks.csv', index=False)

#%%

passed_temperatures = df_temperature.loc[df_temperature['Year'] > 1979].values[:,1:][:,runids]

passed_temperatures_1850_1900 = np.mean(df_temperature.loc[(df_temperature[
        'Year'] > 1850) & (df_temperature['Year'] < 1900)], axis=0).values[1:][runids]

passed_temperatures_offset = passed_temperatures - passed_temperatures_1850_1900


#%%

# find closest member

# This is the temperature timeseries of the prior version of FRIDA. So this
# finds the member of the new ensemble which is closest to the old default,
# to allow max continuity in testing.

df_old_T1 = pd.read_csv('../data/external/temp_before_changes.csv')

df_old_T1_hist = df_old_T1.loc[df_old_T1['Year'] < 2023]

target_T1 = df_old_T1_hist['Energy Balance Model.Surface Temperature Anomaly[1]']

idxs_closest_to_old = []
for n_i in np.arange(output_ensemble_size):
    rmse_in = np.sqrt(((passed_temperatures_offset[:,n_i]-target_T1)**2).mean())
    if n_i == 0:
        rmse = rmse_in
        idx = n_i + 1 # in FRIDA they start from 1
    else:
        if rmse_in < rmse:
            rmse = rmse_in
            idx = n_i + 1 # in FRIDA they start from 1
idxs_closest_to_old.append(idx)

#%%

# find percentile members

percs = np.linspace(0, 100, 15)

perc_ts = np.percentile(passed_temperatures_offset, percs, axis=1)

idxs_percs = []
for p_i, perc in enumerate(percs):
    for n_i in np.arange(output_ensemble_size):
        rmse_in = np.sqrt(((passed_temperatures_offset[:,n_i]-perc_ts[p_i])**2).mean())
        if n_i == 0:
            rmse = rmse_in
            idx = n_i + 1
        else:
            if rmse_in < rmse:
                if n_i + 1 not in idxs_percs:
                    rmse = rmse_in
                    idx = n_i + 1
    idxs_percs.append(idx)


#%%

# find median member

t_means = np.mean(passed_temperatures_offset, axis=0)

idx_med = np.argsort(t_means)[len(t_means)//2] + 1

#%%

plt.plot(np.arange(43), passed_temperatures_offset, color='grey')

plt.plot(np.arange(43), passed_temperatures_offset[:,1], color='grey', label='New ensemble')

plt.plot(np.arange(43), passed_temperatures_offset[:,np.asarray(idxs_percs) - 1], color='orange')

plt.plot(np.arange(43), target_T1, color='red', label='Old median member')

plt.plot(np.arange(43), passed_temperatures_offset[:,idxs_closest_to_old[0]-1], color='blue', label='New closest member')

plt.plot(np.arange(43), passed_temperatures_offset[:,idx_med - 1], color='green', label='New median member')

plt.legend()

#%%

perc_runs_data = df_combined.values[np.asarray(idxs_percs) - 1,:]

df_perc_runs_data = pd.DataFrame(data=perc_runs_data, columns=df_combined.keys())

# df_perc_runs_data.to_csv(
#     f"../data/constraining/frida_climate_inputs_{output_ensemble_size}_from_{samples}_{len(percs)}_percs.csv",
#     index=False,
# )

