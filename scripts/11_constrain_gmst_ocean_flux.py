#!/usr/bin/env python
# coding: utf-8

"""First constraint: RMSE < 0.16 K"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

print("Doing RMSE constraint...")

samples = int(os.getenv("PRIOR_SAMPLES"))

def rmse(obs, mod):
    return np.sqrt(np.sum((obs - mod) ** 2) / len(obs))


weights = np.ones(52)
weights[0] = 0.5
weights[-1] = 0.5

df_temp = pd.read_csv("../data/priors_output/priors_temperature.csv")

temp_hist = df_temp.loc[(df_temp['Year']>=1850) & (df_temp['Year']<=2022)].drop(columns='Year').values
temp_hist_offset = temp_hist - np.average(temp_hist[:52, :], weights=weights, axis=0)


df_temp_obs = pd.read_csv("../data/external/forcing/annual_averages.csv")
gmst = df_temp_obs["gmst"].loc[(df_temp_obs['time'] > 1850) 
                               & (df_temp_obs['time'] < 2023)].values

time = df_temp_obs["time"].loc[(df_temp_obs['time'] > 1850) 
                               & (df_temp_obs['time'] < 2023)].values

#%%

fig, ax = plt.subplots(figsize=(5, 5))

ax.fill_between(time, np.min(temp_hist_offset, axis=1), 
                 np.max(temp_hist_offset, axis=1), color="#000000", alpha=0.2)

ax.fill_between(time, np.percentile(temp_hist_offset, 95, axis=1), 
              np.percentile(temp_hist_offset, 5, axis=1), color="#000000", alpha=0.2)

ax.fill_between(time, np.percentile(temp_hist_offset, 84, axis=1), 
              np.percentile(temp_hist_offset, 16, axis=1), color="#000000", alpha=0.2)

ax.plot(time, np.median(temp_hist_offset, axis=1), 
                  color="#000000", alpha=0.2)


ax.plot(time, gmst, label='AR6 obs')

ax.set_xlim(1850, 2025)
ax.set_ylim(-1, 5)
ax.set_ylabel("°C relative to 1850-1900")
ax.axhline(0, color="k", ls=":", lw=0.5)
plt.title("Temperature anomaly priors")
plt.legend()
plt.tight_layout()


#%%


df_flux = pd.read_csv("../data/priors_output/priors_ocean_CO2_flux.csv")

flux_hist = df_flux.loc[(df_flux['Year']>=1781) & (df_flux['Year']<=2022)].drop(columns='Year').values
flux_hist_for_rmse = df_flux.loc[(df_flux['Year']>=1960) & (df_flux['Year']<=2022)].drop(columns='Year').values


df_ocean = pd.read_csv("../data/external/GCB_historical_budget.csv")
df_ocean_hist = df_ocean.loc[(df_ocean['Year']>=1781) & (df_ocean['Year']<=2022)]

flux = df_ocean_hist["ocean sink"].values

df_ocean_hist_crop = df_ocean.loc[(df_ocean['Year']>=1960) & (df_ocean['Year']<=2022)]
flux_for_rmse = df_ocean_hist_crop["ocean sink"].values


#%%

fig, ax = plt.subplots(figsize=(5, 5))

ax.fill_between(df_ocean_hist["Year"], np.min(flux_hist, axis=1), 
                 np.max(flux_hist, axis=1), color="#000000", alpha=0.2)

ax.fill_between(df_ocean_hist["Year"], np.percentile(flux_hist, 95, axis=1), 
              np.percentile(flux_hist, 5, axis=1), color="#000000", alpha=0.2)

ax.fill_between(df_ocean_hist["Year"], np.percentile(flux_hist, 84, axis=1), 
              np.percentile(flux_hist, 16, axis=1), color="#000000", alpha=0.2)

ax.plot(df_ocean_hist["Year"], np.median(flux_hist, axis=1), 
                  color="#000000", alpha=0.2)

ax.plot(df_ocean_hist["Year"], flux, label='GCB obs')
ax.plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs for RMSE')

ax.set_xlim(1780, 2025)
# ax.set_ylim(-1, 5)
ax.set_ylabel("Ocean-air CO2 flux")
ax.axhline(0, color="k", ls=":", lw=0.5)
plt.title("CO2 flux priors")
plt.legend()
plt.tight_layout()

#%%

rmse_temp = np.zeros((samples))

for i in range(samples):
    rmse_temp[i] = rmse(
        gmst,
        temp_hist_offset[:, i],
    )
    
accept_temp = rmse_temp < 0.16

n_pass_temp = np.sum(accept_temp)

print("Passing Temperature constraint:", n_pass_temp)
valid_temp = np.arange(samples, dtype=int)[accept_temp]


#%%

flux_constraint = 0.2*np.mean(flux_for_rmse)

rmse_flux = np.zeros((samples))

for i in range(samples):
    rmse_flux[i] = rmse(
        flux_for_rmse[:170],
        flux_hist_for_rmse[:170, i],
    )
    

accept_flux = rmse_flux < flux_constraint

n_pass_flux = np.sum(accept_flux)

print("Passing Flux constraint:",n_pass_flux)
valid_flux = np.arange(samples, dtype=int)[accept_flux]


#%%
valid_both = np.intersect1d(valid_temp,valid_flux)

n_pass_both = valid_both.shape[0]

print("Passing both constraints:",n_pass_both)

accept_both = np.logical_and(accept_temp, accept_flux)



#%%

priors_temp_2005_14 = np.mean(df_temp.loc[(df_temp['Year']>=2005) & (
                df_temp['Year']<=2014)].drop(columns='Year').values - np.average(temp_hist[:52, :], 
                     weights=weights, axis=0), axis=0)

priors_flux_2005_14 = np.mean(df_flux.loc[(df_flux['Year']>=2005) & (
                df_flux['Year']<=2014)].drop(columns='Year').values, axis=0)

priors_flux_2005_14_obs = np.mean(df_ocean["ocean sink"].loc[(df_ocean['Year']>=2005) & (
                df_ocean['Year']<=2014)].drop(columns='Year').values, axis=0)


priors_temp_2005_14_obs = np.mean(df_temp_obs["gmst"].values[-16:-6])


plt.scatter(priors_temp_2005_14, priors_flux_2005_14, color='grey')

plt.axhline(y = priors_flux_2005_14_obs)
plt.axhline(y = priors_flux_2005_14_obs+flux_constraint)
plt.axhline(y = priors_flux_2005_14_obs-flux_constraint)

plt.axvline(x = priors_temp_2005_14_obs)
plt.axvline(x = priors_temp_2005_14_obs+0.16)
plt.axvline(x = priors_temp_2005_14_obs-0.16)


plt.xlabel('Temp 2005-14')
plt.ylabel('Flux 2005-14')



#%%

fig, axs = plt.subplots(4, 2, figsize=(12, 12))

axs[0,0].fill_between(time, np.percentile(temp_hist_offset, 84, axis=1), 
              np.percentile(temp_hist_offset, 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[0,0].plot(time, np.median(temp_hist_offset, axis=1), 
              color="#000000", label='Median')

axs[0,0].plot(time, gmst, label='AR6 obs')

axs[0,0].legend()
axs[0,0].set_ylabel('deg C')
axs[0,0].set_title(f'All priors: {samples}')



axs[0,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist, 84, axis=1), 
              np.percentile(flux_hist, 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[0,1].plot(df_ocean_hist["Year"], np.median(flux_hist, axis=1), 
              color="#000000", label='Median')

axs[0,1].plot(df_ocean_hist["Year"], flux)
axs[0,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[0,1].legend()
axs[0,1].set_ylabel('GtC/yr')
axs[0,1].set_title(f'All priors: {samples}')





axs[1,0].fill_between(time, np.percentile(temp_hist_offset[:, accept_temp], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_temp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[1,0].plot(time, np.median(temp_hist_offset[:, accept_temp], axis=1), 
              color="#000000", label='Median')

axs[1,0].plot(time, gmst, label='AR6 obs')

axs[1,0].legend()
axs[1,0].set_ylabel('deg C')
axs[1,0].set_title(f'Passing temp: {n_pass_temp}')



axs[1,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist[:, accept_temp], 84, axis=1), 
              np.percentile(flux_hist[:, accept_temp], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[1,1].plot(df_ocean_hist["Year"], np.median(flux_hist[:, accept_temp], axis=1), 
              color="#000000", label='Median')

axs[1,1].plot(df_ocean_hist["Year"], flux)
axs[1,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[1,1].legend()
axs[1,1].set_ylabel('GtC/yr')
axs[1,1].set_title(f'Passing temp: {n_pass_temp}')




axs[2,0].fill_between(time, np.percentile(temp_hist_offset[:, accept_flux], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_flux], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[2,0].plot(time, np.median(temp_hist_offset[:, accept_flux], axis=1), 
              color="#000000", label='Median')

axs[2,0].plot(time, gmst, label='AR6 obs')

axs[2,0].legend()
axs[2,0].set_ylabel('deg C')
axs[2,0].set_title(f'Passing flux: {n_pass_flux}')



axs[2,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist[:, accept_flux], 84, axis=1), 
              np.percentile(flux_hist[:, accept_flux], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[2,1].plot(df_ocean_hist["Year"], np.median(flux_hist[:, accept_flux], axis=1), 
              color="#000000", label='Median')

axs[2,1].plot(df_ocean_hist["Year"], flux)
axs[2,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[2,1].legend()
axs[2,1].set_ylabel('GtC/yr')
axs[2,1].set_title(f'Passing flux: {n_pass_flux}')





axs[3,0].fill_between(time, np.percentile(temp_hist_offset[:, accept_both], 84, axis=1), 
              np.percentile(temp_hist_offset[:, accept_both], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[3,0].plot(time, np.median(temp_hist_offset[:, accept_both], axis=1), 
              color="#000000", label='Median')

axs[3,0].plot(time, gmst, label='AR6 obs')

axs[3,0].legend()
axs[3,0].set_ylabel('deg C')
axs[3,0].set_title(f'Passing both: {n_pass_both}')



axs[3,1].fill_between(df_ocean_hist["Year"], np.percentile(flux_hist[:, accept_both], 84, axis=1), 
              np.percentile(flux_hist[:, accept_both], 16, axis=1), color="#000000", alpha=0.2,
              label = '16-84 %ile')

axs[3,1].plot(df_ocean_hist["Year"], np.median(flux_hist[:, accept_both], axis=1), 
              color="#000000", label='Median')

axs[3,1].plot(df_ocean_hist["Year"], flux)
axs[3,1].plot(df_ocean_hist_crop["Year"], flux_for_rmse, label='GCB obs')

axs[3,1].legend()
axs[3,1].set_ylabel('GtC/yr')
axs[3,1].set_title(f'Passing both: {n_pass_both}')


plt.tight_layout()

plt.savefig(
    "../plots/rmse_constrained.png"
)
#%%
np.savetxt(
    "../data/constraining/runids_rmse_pass.csv",
    valid_both.astype(int),
    fmt="%d",
)
