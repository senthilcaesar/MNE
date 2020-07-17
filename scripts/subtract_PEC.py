import mne
import numpy as np
import matplotlib.pyplot as plt


freq_bands = ['theta', 'alpha', 'lowerbeta', 'higherbeta', 'gamma', 'allbands']
band = freq_bands[4]


theta_CTL_EO = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PEC/{band}_CTL_EO_epochs.txt'
with open(theta_CTL_EO) as f:
    theta_CTL_EO_list = f.read().splitlines()

theta_PD_EO = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PEC/{band}_PD_EO_epochs.txt'
with open(theta_PD_EO) as f:
    theta_PD_EO_list = f.read().splitlines() 

data_CTL = np.zeros(np.load(theta_CTL_EO_list[0]).shape)
for data_npy_CTL in theta_CTL_EO_list:
    data_CTL = np.add(data_CTL, np.load(data_npy_CTL))

data_PD = np.zeros(np.load(theta_PD_EO_list[0]).shape)
for data_npy_PD in theta_PD_EO_list:
    data_PD = np.add(data_PD, np.load(data_npy_PD))

CTL_minus_PD = data_CTL - data_PD
corr = mne.connectivity.envelope_correlation(CTL_minus_PD, combine='mean', verbose=True)
np.save(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PEC/{band}_CTL_minus_PD_EO.npy', corr)