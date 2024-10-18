import numpy as np
import pandas as pd
import os

from dotenv import load_dotenv

load_dotenv()

samples = int(os.getenv("PRIOR_SAMPLES"))


# adapted from stocks output, for the ocean initialisation, to do the
# ocean & temperature calibration. Need to run Ocean_spinup_start before this.

# Ocean_spinup_start runs for 100 years at dt=1/8, to avoid initial shocks.
# This script converts the output stocks from those into the format that 
# can be fed into Ocean_spinup_end.


# this was 03 in the prior version; take in the output from the 100 member
# frida ensemble; output the initialisations.

#ie need to run FRIDA Temperature_from_1750_100_members before this

df_inits = pd.read_csv('../data/spinup_output/Ocean_spinup_output_start.csv')

variable_stock_list = [
    'Ocean.Cold surface ocean pH[1]', 
    'Ocean.Warm surface ocean pH[1]',
    'Ocean.Cold surface ocean carbon reservoir[1]', 
    'Ocean.Warm surface ocean carbon reservoir[1]', 
    'Ocean.Intermediate depth ocean carbon reservoir[1]',
    'Ocean.Deep ocean ocean carbon reservoir[1]']

variable_stock_list_frida = []
for variable_stock in variable_stock_list:
    variable_stock_list_frida.append(variable_stock.split(".")[0
                       ] + '.Initial ' + variable_stock.split(".")[1][:-3])

df_out = pd.DataFrame(columns=[variable_stock_list_frida])

for n_i in np.arange(samples):
    row = []    
    for stock in variable_stock_list:
         row.append(df_inits[f'="Run {n_i+1}: {stock}"'].values[0])

    df_out.loc[n_i] = row

df_out.to_csv('../data/spinup_input/ocean_input_end.csv', index=False)
