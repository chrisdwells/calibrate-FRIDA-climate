import pandas as pd
from dotenv import load_dotenv
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
import scipy.optimize
import scipy.stats

# Check distribution of 100 members when ran in FRIDA climate matches
# the calibration result

load_dotenv()

output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

draws_in = pd.read_csv(f'../data/constraining/draws_{output_ensemble_size}.csv')

with open('../data/constraining/distributions.pickle', 'rb') as handle:
    dict_distributions = pickle.load(handle)

temp_posteriors = pd.read_csv('../data/posteriors_output/posteriors_temperature.csv')


temp_pi = np.average(temp_posteriors.loc[(temp_posteriors['Year']>=1850) & (temp_posteriors['Year']<=1900)].drop(columns='Year').values, axis=0)
temp_pd = np.average(temp_posteriors.loc[(temp_posteriors['Year']>=2003) & (temp_posteriors['Year']<=2022)].drop(columns='Year').values, axis=0)

temp_in = temp_pd - temp_pi


#%%
df_ohc = pd.read_csv('../data/posteriors_output/posteriors_ocean_heat_content.csv')

ohc_data = df_ohc.drop(columns='Year').values

ohc_in = (ohc_data[1,:] - ohc_data[0,:])*1000 # units


df_aer = pd.read_csv("../data/posteriors_output/posteriors_aerosols.csv")

faci_in = np.full(output_ensemble_size, np.nan)
fari_in = np.full(output_ensemble_size, np.nan)

for i in np.arange(output_ensemble_size):
    faci_in[i] = np.mean(df_aer[
    f'="Run {i+1}: Aerosol Forcing.Effective Radiative Forcing from Aerosol-Cloud Interactions[1]"'])
    
    fari_in[i] = np.mean(df_aer[
    f'="Run {i+1}: Aerosol Forcing.Effective Radiative Forcing from Aerosol-Radiation Interactions[1]"'])

faer_in = fari_in + faci_in

df_co2 = pd.read_csv("../data/posteriors_output/posteriors_CO2.csv")
co2_in = df_co2.drop(columns='Year').values[0,:]


colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black", "post check": "orange"}

fig, ax = plt.subplots(3, 3, figsize=(10, 10))

# post_check_ecs = scipy.stats.gaussian_kde(draws_in["ECS"])

ax[0, 0].plot(
    dict_distributions['ECS']['Xs'],
    dict_distributions['ECS']['Priors'],
    color=colors["prior"],
    label="Prior",
)
ax[0, 0].plot(
    dict_distributions['ECS']['Xs'],
    dict_distributions['ECS']['Post1'],
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[0, 0].plot(
    dict_distributions['ECS']['Xs'],
    dict_distributions['ECS']['Post2'],
    color=colors["post2"],
    label="All constraints",
)
ax[0, 0].plot(
    dict_distributions['ECS']['Xs'],
    dict_distributions['ECS']['Target'],
    color=colors["target"],
    label="Target",
)
ax[0, 0].set_xlim(dict_distributions['ECS']['xlim'])
ax[0, 0].set_ylim(0, 0.5)
ax[0, 0].set_title("ECS")
ax[0, 0].set_yticklabels([])
ax[0, 0].set_xlabel("°C")




ax[0, 1].plot(
    dict_distributions['TCR']['Xs'],
    dict_distributions['TCR']['Priors'],
    color=colors["prior"],
    label="Prior",
)
ax[0, 1].plot(
    dict_distributions['TCR']['Xs'],
    dict_distributions['TCR']['Post1'],
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[0, 1].plot(
    dict_distributions['TCR']['Xs'],
    dict_distributions['TCR']['Post2'],
    color=colors["post2"],
    label="All constraints",
)
ax[0, 1].plot(
    dict_distributions['TCR']['Xs'],
    dict_distributions['TCR']['Target'],
    color=colors["target"],
    label="Target",
)
ax[0, 1].set_xlim(dict_distributions['TCR']['xlim'])
ax[0, 1].set_ylim(0, 1.4)
ax[0, 1].set_title("TCR")
ax[0, 1].set_yticklabels([])
ax[0, 1].set_xlabel("°C")



post_check_temp = scipy.stats.gaussian_kde(temp_in)

ax[0, 2].plot(
    dict_distributions['Temp']['Xs'],
    dict_distributions['Temp']['Priors'],
    color=colors["prior"],
    label="Prior",
)
ax[0, 2].plot(
    dict_distributions['Temp']['Xs'],
    dict_distributions['Temp']['Post1'],
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[0, 2].plot(
    dict_distributions['Temp']['Xs'],
    dict_distributions['Temp']['Post2'],
    color=colors["post2"],
    label="All constraints",
)
ax[0, 2].plot(
    dict_distributions['Temp']['Xs'],
    dict_distributions['Temp']['Target'],
    color=colors["target"],
    label="Target",
)
ax[0, 2].plot(
    dict_distributions['Temp']['Xs'],
    post_check_temp(dict_distributions['Temp']['Xs']),
    color=colors["post check"],
    label="Posteriors",
    linestyle='--',
)
ax[0, 2].set_xlim(dict_distributions['Temp']['xlim'])
ax[0, 2].set_ylim(0, 5)
ax[0, 2].set_title("Temperature anomaly")
ax[0, 2].set_yticklabels([])
ax[0, 2].set_xlabel("°C, 2003-2022 minus 1850-1900")


post_check_aer = scipy.stats.gaussian_kde(faer_in)

ax[1, 2].plot(
    dict_distributions['Aerosol']['Xs'],
    dict_distributions['Aerosol']['Priors'],
    color=colors["prior"],
    label="Prior",
)
ax[1, 2].plot(
    dict_distributions['Aerosol']['Xs'],
    dict_distributions['Aerosol']['Post1'],
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[1, 2].plot(
    dict_distributions['Aerosol']['Xs'],
    dict_distributions['Aerosol']['Post2'],
    color=colors["post2"],
    label="All constraints",
)
ax[1, 2].plot(
    dict_distributions['Aerosol']['Xs'],
    dict_distributions['Aerosol']['Target'],
    color=colors["target"],
    label="Target",
)
ax[1, 2].plot(
    dict_distributions['Aerosol']['Xs'],
    post_check_aer(dict_distributions['Aerosol']['Xs']),
    color=colors["post check"],
    label="Posteriors",
    linestyle='--',
)
ax[1, 2].set_xlim(dict_distributions['Aerosol']['xlim'])
ax[1, 2].set_ylim(0, 1.6)
ax[1, 2].set_title("Aerosol ERF")
ax[1, 2].legend(frameon=False, loc="upper left")
ax[1, 2].set_yticklabels([])
ax[1, 2].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")



post_check_co2 = scipy.stats.gaussian_kde(co2_in)

ax[2, 0].plot(
    dict_distributions['CO2']['Xs'],
    dict_distributions['CO2']['Priors'],
    color=colors["prior"],
    label="Prior",
)
ax[2, 0].plot(
    dict_distributions['CO2']['Xs'],
    dict_distributions['CO2']['Post1'],
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[2, 0].plot(
    dict_distributions['CO2']['Xs'],
    dict_distributions['CO2']['Post2'],
    color=colors["post2"],
    label="All constraints",
)
ax[2, 0].plot(
    dict_distributions['CO2']['Xs'],
    dict_distributions['CO2']['Target'],
    color=colors["target"],
    label="Target",
)
ax[2, 0].plot(
    dict_distributions['CO2']['Xs'],
    post_check_co2(dict_distributions['CO2']['Xs']),
    color=colors["post check"],
    label="Posteriors",
    linestyle='--',
)
ax[2, 0].set_xlim(dict_distributions['CO2']['xlim'])
ax[2, 0].set_ylim(0, 1.2)
ax[2, 0].set_title("CO$_2$ concentration")
ax[2, 0].set_yticklabels([])
ax[2, 0].set_xlabel("ppm, 2022")



post_check_ohc = scipy.stats.gaussian_kde(ohc_in)

ax[2, 1].plot(
    dict_distributions['OHC']['Xs'],
    dict_distributions['OHC']['Priors'],
    color=colors["prior"],
    label="Prior",
)
ax[2, 1].plot(
    dict_distributions['OHC']['Xs'],
    dict_distributions['OHC']['Post1'],
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[2, 1].plot(
    dict_distributions['OHC']['Xs'],
    dict_distributions['OHC']['Post2'],
    color=colors["post2"],
    label="All constraints",
)
ax[2, 1].plot(
    dict_distributions['OHC']['Xs'],
    dict_distributions['OHC']['Target'],
    color=colors["target"],
    label="Target",
)
ax[2, 1].plot(
    dict_distributions['OHC']['Xs'],
    post_check_ohc(dict_distributions['OHC']['Xs']),
    color=colors["post check"],
    label="Posteriors",
    linestyle='--',
)
ax[2, 1].set_xlim(dict_distributions['OHC']['xlim'])
ax[2, 1].set_ylim(0, 0.006)
ax[2, 1].set_title("Ocean heat content change")
ax[2, 1].set_yticklabels([])
ax[2, 1].set_xlabel("ZJ, 2020 minus 1971")


plt.tight_layout()

plt.savefig(
    "../plots/check_posteriors.png"
)
