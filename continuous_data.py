import os
import numpy as np 
import matplotlib.pyplot as plt
import mne 
# pylint: disable=E1101

filename = '/Users/senthilp/Desktop/mne_tutorial/BrainVision_data/Control1129.vhdr'
raw = mne.io.read_raw_brainvision(filename)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
print(raw.info)
raw.crop(tmax=60).load_data()

n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)

print(f"The cropped sample data object has {n_time_samps} time samples and {n_chan} channels.")
print(f"The last time sample at {time_secs[-1]} seconds.")
print(f"The first few channel names are {ch_names[:3]}")
print(f"Bad channles marked during data acquisition {raw.info['bads']}")
print(f"Sampling Frequency {raw.info['sfreq']} Hz")
print(f"Miscellaneous acquisition info {raw.info['description']}")
print(raw.time_as_index(60)) # Convert time to indices

# From the raw object select only the EEG channels
eeg = raw.copy().pick_types(eeg=True, meg=False, eog=False)
print(f"The number of EEG channels {len(eeg.ch_names)}")

# From the EEG data pick channels by name
#print(eeg.ch_names)
eeg.pick_channels(['P1', 'P2', 'P3'])

# Extracting data by index
sampling_freq = raw.info['sfreq']
start_stop_seconds = np.array([11, 13])
start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
channel_index = 2
eeg_data, eeg_times = eeg[channel_index, start_sample:stop_sample]
plt.plot(eeg_times, eeg_data.T)
plt.show()
