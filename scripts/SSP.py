import os
import numpy as np 
import matplotlib.pyplot as plt 
import mne
from mne.preprocessing import (create_eog_epochs, create_ecg_epochs,
                                compute_proj_ecg, compute_proj_eog)

''' Signal Space Projection is a technique for removing
noise from EEG and MEG signals by projecting the signal onto a 
lower-dimensional subspace'''
sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
system_projs = raw.info['projs'] 
'''Remove system-provided SSP projectors, SSP projectors for environmental noise removal is stored in raw.info['projs']'''
raw.del_proj()
empty_room_file = '/Users/senthilp/Desktop/mne_tutorial/ernoise_raw.fif' 
'''Empty room recording ( more accrurate estimate of environmental noise than the projectors stored with the system )'''
empty_room_raw = mne.io.read_raw_fif(empty_room_file)
empty_room_raw.del_proj()

# Visualizing the empty-room noise
for average in (False, True):
    empty_room_raw.plot_psd(average=average, dB=False, xscale='log')


