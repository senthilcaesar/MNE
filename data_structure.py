import os
import numpy as np 
import mne
# pylint: disable=E1101

sample_data_raw_file = 'sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file)
info = mne.io.read_info(sample_data_raw_file)

print(info.keys())
print(info['chs'][0])

raw.crop(tmax=60).load_data()
