import mne
import os
import numpy as np 
# pylint: disable=E1101

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
info = raw.info

# Querying the info object
print(info.keys())
print()
print(f"Total number of channel {info['nchan']}")
print(info['ch_names'])
print(type(info['chs'])) # Contain list of dictionary, one per channel
print(info['chs'][2].keys())

# Pick EEG channel indicies from info object
eeg_indices = mne.pick_types(info, meg=False, eeg=True, exclude=[])
print(eeg_indices)
print(f"Channel type on index 315 is {mne.channel_type(info, 315)}")