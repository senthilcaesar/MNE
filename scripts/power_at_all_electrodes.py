import mne
import numpy as np
import pickle

participant = 'PD'
condition = 'open'
freq_bands = ['theta', 'alpha', 'lowerbeta', 'higherbeta', 'gamma', 'allbands']
band = freq_bands[5]

for band in freq_bands:
    tfr_avg = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_{participant}_eyes{condition}_tfr_avg_allsubejcts.h5'
    power_all_electrocode = mne.time_frequency.read_tfrs(tfr_avg)
    mean_power = power_all_electrocode[0].data.mean(1).mean(1)
    mean_power = list(mean_power)
    datafile = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_{participant}_{condition}_meanpower.pkl'
    F = open(datafile, 'wb')
    pickle.dump(mean_power, F)
    F.close()


