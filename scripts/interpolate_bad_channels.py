import os
from copy import deepcopy
import numpy as np 
import mne 

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)

# List of bad channels stored in Info object
print(raw.info['bads']) # ['MEG 2443', 'EEG 053']

picks = mne.pick_channels_regexp(raw.ch_names, regexp='EEG 05.')
# raw.plot(order=picks, n_channels=len(picks), block=True)

picks = mne.pick_channels_regexp(raw.ch_names, regexp='MEG 2..3')
# raw.plot(order=picks, n_channels=len(picks), block=True)

# Interpolating bad channels
raw.crop(tmin=0, tmax=3).load_data()
''' interpolate_bads() functions will clean out raw.info['bads']
after interpolation
'''

eeg_data = raw.copy().pick_types(meg=False, eeg=True, exclude=[])
eeg_data_interpolate = eeg_data.copy().interpolate_bads(reset_bads=False)

lst1 = ['orig.', 'interp.']
lst2 = [eeg_data, eeg_data_interpolate]

for title, data in zip(lst1, lst2):
    fig = data.plot(butterfly=True, color='#00000022', bad_color='r', block=True)
    fig.subplots_adjust(top=9.0)
    fig.suptitle(title, size='xx-large', weight='bold')

''' Also consider ever more automated approach to bad channel detection
and interpolation using autoreject package ( http://autoreject.github.io/ )
which interfaces well with MNE-Python-based pipeline'''
