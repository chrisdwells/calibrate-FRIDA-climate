import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
from dotenv import load_dotenv

# Adapted from FaIR calibrate
# for FRIDA, calculate ECS from the parameters.

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))
output_ensemble_size = int(os.getenv("POSTERIOR_SAMPLES"))

NINETY_TO_ONESIGMA = scipy.stats.norm.ppf(0.95)

valid_temp_flux = np.loadtxt(
    "../data/constraining/runids_rmse_pass.csv",
).astype(np.int64)

input_ensemble_size = len(valid_temp_flux)

assert input_ensemble_size > output_ensemble_size


weights_20yr = np.ones(21)
weights_20yr[0] = 0.5
weights_20yr[-1] = 0.5
weights_51yr = np.ones(52)
weights_51yr[0] = 0.5
weights_51yr[-1] = 0.5

df_temp = pd.read_csv("../data/priors_output/priors_temperature.csv")

temp_hist = df_temp.loc[(df_temp['Year']>=1850) & (df_temp['Year']<=2020)].drop(columns='Year').values[:,1:]
temp_in = np.average(temp_hist[145:166, :], weights=weights_20yr, axis=0
             ) - np.average(temp_hist[:52, :], weights=weights_51yr, axis=0)

temp_hist_offset = temp_hist - np.average(temp_hist[:52, :], weights=weights_51yr, axis=0)


df_ohc = pd.read_csv("../data/priors_output/priors_ocean_heat_content.csv")

ohc_data = df_ohc.drop(columns='Year').values

ohc_in = (ohc_data[1,:] - ohc_data[0,:])*1000 # units


df_aer = pd.read_csv("../data/priors_output/priors_aerosols.csv")

faci_in = np.full(samples, np.nan)
fari_in = np.full(samples, np.nan)

for i in np.arange(samples):
    
    faci_in[i] = np.mean(df_aer[
    f'="Run {i+1}: Aerosol Forcing.Effective Radiative Forcing from Aerosol-Cloud Interactions[1]"'])
    
    fari_in[i] = np.mean(df_aer[
    f'="Run {i+1}: Aerosol Forcing.Effective Radiative Forcing from Aerosol-Radiation Interactions[1]"'])
    

df_co2 = pd.read_csv("../data/priors_output/priors_CO2.csv")
co2_in = df_co2.drop(columns='Year').values[0,:]


df_ecs_tcr = pd.read_csv(f"../data/external/samples_for_priors/ecs_tcs_{samples}.csv")

ecs_in = df_ecs_tcr['ecs']
tcr_in = df_ecs_tcr['tcr']

faer_in = fari_in + faci_in

#%%
def opt(x, q05_desired, q50_desired, q95_desired):
    "x is (a, loc, scale) in that order."
    q05, q50, q95 = scipy.stats.skewnorm.ppf(
        (0.05, 0.50, 0.95), x[0], loc=x[1], scale=x[2]
    )
    # print(q05, q50, q95, x)
    return (q05 - q05_desired, q50 - q50_desired, q95 - q95_desired)


ecs_params = scipy.optimize.root(opt, [1, 1, 1], args=(2, 3, 5)).x
gsat_params = scipy.optimize.root(opt, [1, 1, 1], args=(0.87, 1.03, 1.13)).x

samples_dict = {}
samples_dict["ECS"] = scipy.stats.skewnorm.rvs(
    ecs_params[0],
    loc=ecs_params[1],
    scale=ecs_params[2],
    size=10**5,
    random_state=91603,
)
samples_dict["TCR"] = scipy.stats.norm.rvs(
    loc=1.8, scale=0.6 / NINETY_TO_ONESIGMA, size=10**5, random_state=18196
)
samples_dict["temperature 2003-2022"] = scipy.stats.skewnorm.rvs(
    gsat_params[0],
    loc=gsat_params[1],
    scale=gsat_params[2],
    size=10**5,
    random_state=19387,
)
samples_dict["OHC"] = scipy.stats.norm.rvs(
    loc=465.3, scale=108.5 / NINETY_TO_ONESIGMA, size=10**5, random_state=43178
)
# samples_dict["ERFari"] = scipy.stats.norm.rvs(
#     loc=-0.3, scale=0.3 / NINETY_TO_ONESIGMA, size=10**5, random_state=70173
# )
# samples_dict["ERFaci"] = scipy.stats.norm.rvs(
#     loc=-1.0, scale=0.7 / NINETY_TO_ONESIGMA, size=10**5, random_state=91123
# )
samples_dict["ERFaer"] = scipy.stats.norm.rvs(
    loc=-1.3,
    scale=np.sqrt(0.7**2 + 0.3**2) / NINETY_TO_ONESIGMA,
    size=10**5,
    random_state=3916153,
)
samples_dict["CO2 concentration"] = scipy.stats.norm.rvs(
    loc=417.0, scale=0.5, size=10**5, random_state=81693
)


ar_distributions = {}
for constraint in [
    "ECS",
    "TCR",
    "OHC",
    "temperature 2003-2022",
    # "ERFari",
    # "ERFaci",
    "ERFaer",
    "CO2 concentration",
]:
    ar_distributions[constraint] = {}
    ar_distributions[constraint]["bins"] = np.histogram(
        samples_dict[constraint], bins=100, density=True
    )[1]
    ar_distributions[constraint]["values"] = samples_dict[constraint]

accepted = pd.DataFrame(
    {
        "ECS": ecs_in[valid_temp_flux],
        "TCR": tcr_in[valid_temp_flux],
        "OHC": ohc_in[valid_temp_flux],
        "temperature 2003-2022": temp_in[valid_temp_flux],
        # "ERFari": fari_in[valid_temp_flux],
        # "ERFaci": faci_in[valid_temp_flux],
        "ERFaer": faer_in[valid_temp_flux],
        "CO2 concentration": co2_in[valid_temp_flux],
    },
    index=valid_temp_flux,
)


def calculate_sample_weights(distributions, samples, niterations=50):
    weights = np.ones(samples.shape[0])
    gofs = []
    gofs_full = []

    unique_codes = list(distributions.keys())  # [::-1]

    for k in range(niterations):
        gofs.append([])
        if k == (niterations - 1):
            weights_second_last_iteration = weights.copy()
            weights_to_average = []

        for j, unique_code in enumerate(unique_codes):
            unique_code_weights, our_values_bin_idx = get_unique_code_weights(
                unique_code, distributions, samples, weights, j, k
            )
            if k == (niterations - 1):
                weights_to_average.append(unique_code_weights[our_values_bin_idx])

            weights *= unique_code_weights[our_values_bin_idx]

            gof = ((unique_code_weights[1:-1] - 1) ** 2).sum()
            gofs[-1].append(gof)

            gofs_full.append([unique_code])
            for unique_code_check in unique_codes:
                unique_code_check_weights, _ = get_unique_code_weights(
                    unique_code_check, distributions, samples, weights, 1, 1
                )
                gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
                gofs_full[-1].append(gof)

    weights_stacked = np.vstack(weights_to_average).mean(axis=0)
    weights_final = weights_stacked * weights_second_last_iteration

    gofs_full.append(["Final iteration"])
    for unique_code_check in unique_codes:
        unique_code_check_weights, _ = get_unique_code_weights(
            unique_code_check, distributions, samples, weights_final, 1, 1
        )
        gof = ((unique_code_check_weights[1:-1] - 1) ** 2).sum()
        gofs_full[-1].append(gof)

    return (
        weights_final,
        pd.DataFrame(np.array(gofs), columns=unique_codes),
        pd.DataFrame(np.array(gofs_full), columns=["Target marginal"] + unique_codes),
    )


def get_unique_code_weights(unique_code, distributions, samples, weights, j, k):
    bin_edges = distributions[unique_code]["bins"]
    our_values = samples[unique_code].copy()

    our_values_bin_counts, bin_edges_np = np.histogram(our_values, bins=bin_edges)
    np.testing.assert_allclose(bin_edges, bin_edges_np)
    assessed_ranges_bin_counts, _ = np.histogram(
        distributions[unique_code]["values"], bins=bin_edges
    )

    our_values_bin_idx = np.digitize(our_values, bins=bin_edges)

    existing_weighted_bin_counts = np.nan * np.zeros(our_values_bin_counts.shape[0])
    for i in range(existing_weighted_bin_counts.shape[0]):
        existing_weighted_bin_counts[i] = weights[(our_values_bin_idx == i + 1)].sum()

    if np.equal(j, 0) and np.equal(k, 0):
        np.testing.assert_equal(
            existing_weighted_bin_counts.sum(), our_values_bin_counts.sum()
        )

    unique_code_weights = np.nan * np.zeros(bin_edges.shape[0] + 1)

    # existing_weighted_bin_counts[0] refers to samples outside the
    # assessed range's lower bound. Accordingly, if `our_values` was
    # digitized into a bin idx of zero, it should get a weight of zero.
    unique_code_weights[0] = 0
    # Similarly, if `our_values` was digitized into a bin idx greater
    # than the number of bins then it was outside the assessed range
    # so get a weight of zero.
    unique_code_weights[-1] = 0

    for i in range(1, our_values_bin_counts.shape[0] + 1):
        # the histogram idx is one less because digitize gives values in the
        # range bin_edges[0] <= x < bin_edges[1] a digitized index of 1
        histogram_idx = i - 1
        if np.equal(assessed_ranges_bin_counts[histogram_idx], 0):
            unique_code_weights[i] = 0
        elif np.equal(existing_weighted_bin_counts[histogram_idx], 0):
            # other variables force this box to be zero so just fill it with
            # one
            unique_code_weights[i] = 1
        else:
            unique_code_weights[i] = (
                assessed_ranges_bin_counts[histogram_idx]
                / existing_weighted_bin_counts[histogram_idx]
            )

    return unique_code_weights, our_values_bin_idx


weights, gofs, gofs_full = calculate_sample_weights(
    ar_distributions, accepted, niterations=30
)

effective_samples = int(np.floor(np.sum(np.minimum(weights, 1))))
print("Number of effective samples:", effective_samples)

assert effective_samples >= output_ensemble_size

draws = []
drawn_samples = accepted.sample(
    n=output_ensemble_size, replace=False, weights=weights, random_state=10099
)
draws.append((drawn_samples))

#%%
target_ecs = scipy.stats.gaussian_kde(samples_dict["ECS"])
prior_ecs = scipy.stats.gaussian_kde(ecs_in)
post1_ecs = scipy.stats.gaussian_kde(ecs_in[valid_temp_flux])
post2_ecs = scipy.stats.gaussian_kde(draws[0]["ECS"])

target_tcr = scipy.stats.gaussian_kde(samples_dict["TCR"])
prior_tcr = scipy.stats.gaussian_kde(tcr_in)
post1_tcr = scipy.stats.gaussian_kde(tcr_in[valid_temp_flux])
post2_tcr = scipy.stats.gaussian_kde(draws[0]["TCR"])

target_temp = scipy.stats.gaussian_kde(samples_dict["temperature 2003-2022"])
prior_temp = scipy.stats.gaussian_kde(temp_in)
post1_temp = scipy.stats.gaussian_kde(temp_in[valid_temp_flux])
post2_temp = scipy.stats.gaussian_kde(draws[0]["temperature 2003-2022"])

target_ohc = scipy.stats.gaussian_kde(samples_dict["OHC"])
prior_ohc = scipy.stats.gaussian_kde(ohc_in)
post1_ohc = scipy.stats.gaussian_kde(ohc_in[valid_temp_flux])
post2_ohc = scipy.stats.gaussian_kde(draws[0]["OHC"])

target_aer = scipy.stats.gaussian_kde(samples_dict["ERFaer"])
prior_aer = scipy.stats.gaussian_kde(faer_in)
post1_aer = scipy.stats.gaussian_kde(faer_in[valid_temp_flux])
post2_aer = scipy.stats.gaussian_kde(draws[0]["ERFaer"])

# target_aci = scipy.stats.gaussian_kde(samples_dict["ERFaci"])
# prior_aci = scipy.stats.gaussian_kde(faci_in)
# post1_aci = scipy.stats.gaussian_kde(faci_in[valid_temp_flux])
# post2_aci = scipy.stats.gaussian_kde(draws[0]["ERFaci"])

# target_ari = scipy.stats.gaussian_kde(samples_dict["ERFari"])
# prior_ari = scipy.stats.gaussian_kde(fari_in)
# post1_ari = scipy.stats.gaussian_kde(fari_in[valid_temp_flux])
# post2_ari = scipy.stats.gaussian_kde(draws[0]["ERFari"])

target_co2 = scipy.stats.gaussian_kde(samples_dict["CO2 concentration"])
prior_co2 = scipy.stats.gaussian_kde(co2_in)
post1_co2 = scipy.stats.gaussian_kde(co2_in[valid_temp_flux])
post2_co2 = scipy.stats.gaussian_kde(draws[0]["CO2 concentration"])

colors = {"prior": "#207F6E", "post1": "#684C94", "post2": "#EE696B", "target": "black"}



fig, ax = plt.subplots(3, 3, figsize=(10, 10))
start = 0
stop = 8
ax[0, 0].plot(
    np.linspace(start, stop, 1000),
    prior_ecs(np.linspace(start, stop, 1000)),
    color=colors["prior"],
    label="Prior",
)
ax[0, 0].plot(
    np.linspace(start, stop, 1000),
    post1_ecs(np.linspace(start, stop, 1000)),
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[0, 0].plot(
    np.linspace(start, stop, 1000),
    post2_ecs(np.linspace(start, stop, 1000)),
    color=colors["post2"],
    label="All constraints",
)
ax[0, 0].plot(
    np.linspace(start, stop, 1000),
    target_ecs(np.linspace(start, stop, 1000)),
    color=colors["target"],
    label="Target",
)
ax[0, 0].set_xlim(start, stop)
ax[0, 0].set_ylim(0, 0.5)
ax[0, 0].set_title("ECS")
ax[0, 0].set_yticklabels([])
ax[0, 0].set_xlabel("°C")

start = 0
stop = 4
ax[0, 1].plot(
    np.linspace(start, stop, 1000),
    prior_tcr(np.linspace(start, stop, 1000)),
    color=colors["prior"],
    label="Prior",
)
ax[0, 1].plot(
    np.linspace(start, stop, 1000),
    post1_tcr(np.linspace(start, stop, 1000)),
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[0, 1].plot(
    np.linspace(start, stop, 1000),
    post2_tcr(np.linspace(start, stop, 1000)),
    color=colors["post2"],
    label="All constraints",
)
ax[0, 1].plot(
    np.linspace(start, stop, 1000),
    target_tcr(np.linspace(start, stop, 1000)),
    color=colors["target"],
    label="Target",
)
ax[0, 1].set_xlim(start, stop)
ax[0, 1].set_ylim(0, 1.4)
ax[0, 1].set_title("TCR")
ax[0, 1].set_yticklabels([])
ax[0, 1].set_xlabel("°C")

start = 0.5
stop = 1.3
ax[0, 2].plot(
    np.linspace(start, stop, 1000),
    target_temp(np.linspace(start, stop, 1000)),
    color=colors["target"],
    label="Target",
)
ax[0, 2].plot(
    np.linspace(start, stop, 1000),
    prior_temp(np.linspace(start, stop, 1000)),
    color=colors["prior"],
    label="Prior",
)
ax[0, 2].plot(
    np.linspace(start, stop, 1000),
    post1_temp(np.linspace(start, stop, 1000)),
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[0, 2].plot(
    np.linspace(start, stop, 1000),
    post2_temp(np.linspace(start, stop, 1000)),
    color=colors["post2"],
    label="All constraints",
)
ax[0, 2].set_xlim(start, stop)
ax[0, 2].set_ylim(0, 5)
ax[0, 2].set_title("Temperature anomaly")
ax[0, 2].set_yticklabels([])
ax[0, 2].set_xlabel("°C, 2003-2022 minus 1850-1900")

# start = -1.0
# stop = 0.3
# ax[1, 0].plot(
#     np.linspace(start, stop, 1000),
#     target_ari(np.linspace(start, stop, 1000)),
#     color=colors["target"],
#     label="Target",
# )
# ax[1, 0].plot(
#     np.linspace(start, stop, 1000),
#     prior_ari(np.linspace(start, stop, 1000)),
#     color=colors["prior"],
#     label="Prior",
# )
# ax[1, 0].plot(
#     np.linspace(start, stop, 1000),
#     post1_ari(np.linspace(start, stop, 1000)),
#     color=colors["post1"],
#     label="Temp+Flux RMSE",
# )
# ax[1, 0].plot(
#     np.linspace(start, stop, 1000),
#     post2_ari(np.linspace(start, stop, 1000)),
#     color=colors["post2"],
#     label="All constraints",
# )
# ax[1, 0].set_xlim(start, stop)
# ax[1, 0].set_ylim(0, 3)
# ax[1, 0].set_title("Aerosol ERFari")
# ax[1, 0].set_yticklabels([])
# ax[1, 0].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

# start = -2.25
# stop = 0.25
# ax[1, 1].plot(
#     np.linspace(start, stop, 1000),
#     target_aci(np.linspace(start, stop, 1000)),
#     color=colors["target"],
#     label="Target",
# )
# ax[1, 1].plot(
#     np.linspace(start, stop, 1000),
#     prior_aci(np.linspace(start, stop, 1000)),
#     color=colors["prior"],
#     label="Prior",
# )
# ax[1, 1].plot(
#     np.linspace(start, stop, 1000),
#     post1_aci(np.linspace(start, stop, 1000)),
#     color=colors["post1"],
#     label="Temp+Flux RMSE",
# )
# ax[1, 1].plot(
#     np.linspace(start, stop, 1000),
#     post2_aci(np.linspace(start, stop, 1000)),
#     color=colors["post2"],
#     label="All constraints",
# )
# ax[1, 1].set_xlim(start, stop)
# ax[1, 1].set_ylim(0, 1.6)
# ax[1, 1].set_title("Aerosol ERFaci")
# ax[1, 1].set_yticklabels([])
# ax[1, 1].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

start = -3
stop = 0
ax[1, 2].plot(
    np.linspace(start, stop, 1000),
    target_aer(np.linspace(start, stop, 1000)),
    color=colors["target"],
    label="Target",
)
ax[1, 2].plot(
    np.linspace(start, stop, 1000),
    prior_aer(np.linspace(start, stop, 1000)),
    color=colors["prior"],
    label="Prior",
)
ax[1, 2].plot(
    np.linspace(start, stop, 1000),
    post1_aer(np.linspace(start, stop, 1000)),
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[1, 2].plot(
    np.linspace(start, stop, 1000),
    post2_aer(np.linspace(start, stop, 1000)),
    color=colors["post2"],
    label="All constraints",
)
ax[1, 2].set_xlim(start, stop)
ax[1, 2].set_ylim(0, 1.6)
ax[1, 2].set_title("Aerosol ERF")
ax[1, 2].legend(frameon=False, loc="upper left")
ax[1, 2].set_yticklabels([])
ax[1, 2].set_xlabel("W m$^{-2}$, 2005-2014 minus 1750")

start = 413
stop = 421
ax[2, 0].plot(
    np.linspace(start, stop, 1000),
    target_co2(np.linspace(start, stop, 1000)),
    color=colors["target"],
    label="Target",
)
ax[2, 0].plot(
    np.linspace(start, stop, 1000),
    prior_co2(np.linspace(start, stop, 1000)),
    color=colors["prior"],
    label="Prior",
)
ax[2, 0].plot(
    np.linspace(start, stop, 1000),
    post1_co2(np.linspace(start, stop, 1000)),
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[2, 0].plot(
    np.linspace(start, stop, 1000),
    post2_co2(np.linspace(start, stop, 1000)),
    color=colors["post2"],
    label="All constraints",
)
ax[2, 0].set_xlim(start, stop)
ax[2, 0].set_ylim(0, 1.2)
ax[2, 0].set_title("CO$_2$ concentration")
ax[2, 0].set_yticklabels([])
ax[2, 0].set_xlabel("ppm, 2022")

start = 0
stop = 800
ax[2, 1].plot(
    np.linspace(start, stop),
    target_ohc(np.linspace(start, stop)),
    color=colors["target"],
    label="Target",
)
ax[2, 1].plot(
    np.linspace(start, stop),
    prior_ohc(np.linspace(start, stop)),
    color=colors["prior"],
    label="Prior",
)
ax[2, 1].plot(
    np.linspace(start, stop),
    post1_ohc(np.linspace(start, stop)),
    color=colors["post1"],
    label="Temp+Flux RMSE",
)
ax[2, 1].plot(
    np.linspace(start, stop),
    post2_ohc(np.linspace(start, stop)),
    color=colors["post2"],
    label="All constraints",
)
ax[2, 1].set_xlim(start, stop)
ax[2, 1].set_ylim(0, 0.006)
ax[2, 1].set_title("Ocean heat content change")
ax[2, 1].set_yticklabels([])
ax[2, 1].set_xlabel("ZJ, 2020 minus 1971")


fig.tight_layout()
plt.savefig(
    "../plots/constraints.png"
)
# plt.close()



# move these to the validation script
print("Constrained, reweighted parameters:")
print("ECS:", np.percentile(draws[0]["ECS"], (5, 50, 95)))
print(
    "CO2 concentration 2022:", np.percentile(draws[0]["CO2 concentration"], (5, 50, 95))
)
print(
    "temperature 2003-2022 rel. 1850-1900:",
    np.percentile(draws[0]["temperature 2003-2022"], (5, 50, 95)),
)
# print(
#     "Aerosol ERFari 2005-2014 rel. 1750:",
#     np.percentile(draws[0]["ERFari"], (5, 50, 95)),
# )
# print(
#     "Aerosol ERFaci 2005-2014 rel. 1750:",
#     np.percentile(draws[0]["ERFaci"], (5, 50, 95)),
# )
print(
    "Aerosol ERF 2005-2014 rel. 1750:",
    np.percentile(draws[0]["ERFaer"], (5, 50, 95)),
)
print(
    "OHC change 2020 rel. 1971*:", np.percentile(draws[0]["OHC"] * 0.91, (16, 50, 84))
)

print("*likely range")

#%%
df_temp_obs = pd.read_csv("../data/external/forcing/annual_averages.csv")
gmst = df_temp_obs["gmst"].loc[(df_temp_obs['time'] > 1850) 
                               & (df_temp_obs['time'] < 2023)].values

fig, ax = plt.subplots(1, 2, figsize=(10, 6))

ax[0].fill_between(
    np.arange(1850, 2021),
    np.min(temp_hist_offset[:, draws[0].index], axis=1),
    np.max(temp_hist_offset[:, draws[0].index], axis=1),
    color="#000000",
    alpha=0.2,
)
ax[0].fill_between(
    np.arange(1850, 2021),
    np.percentile(temp_hist_offset[:, draws[0].index], 5, axis=1,),
    np.percentile(temp_hist_offset[:, draws[0].index], 95, axis=1,),
    color="#000000",
    alpha=0.2,
)
ax[0].fill_between(
    np.arange(1850, 2021),
    np.percentile(temp_hist_offset[:, draws[0].index], 16, axis=1,),
    np.percentile(temp_hist_offset[:, draws[0].index], 84, axis=1,),
    color="#000000",
    alpha=0.2,
)
ax[0].plot(
    np.arange(1850, 2021),
    np.median(temp_hist_offset[:, draws[0].index], axis=1,),
    color="#000000",
)

ax[0].plot(np.arange(1850.5, 2023), gmst, color="b", label="Observations")

ax[0].legend(frameon=False, loc="upper left")

ax[0].set_xlim(1850, 2025)
ax[0].set_ylim(-1, 5)
ax[0].set_ylabel("°C relative to 1850-1900")
ax[0].axhline(0, color="k", ls=":", lw=0.5)
ax[0].set_title("Temperature anomaly: posterior")



plt.tight_layout()
plt.savefig(
    "../plots/final_reweighted_temp.png"
)



# plt.close()


np.savetxt(
    "../data/constraining/runids_rmse_reweighted_pass.csv",
    sorted(draws[0].index),
    fmt="%d",
)
