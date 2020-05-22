import os
import numpy as np 
import matplotlib.pyplot as plt
import mne 
# pylint: disable=E1101

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
#print(raw.info)
raw.crop(tmax=60).load_data()

n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)

print(f"The cropped sample data object has {n_time_samps} time samples and {n_chan} channels.")
print(f"The last time sample at {time_secs[-1]} seconds.")
print(f"The first few channel names are {ch_names[:3]}")
print(f"Bad channels marked during data acquisition {raw.info['bads']}")
print(f"Sampling Frequency {raw.info['sfreq']} Hz")
print(f"Miscellaneous acquisition info {raw.info['description']}")
print(f"Convert time in sec ( 60s ) to ingeter index {raw.time_as_index(60)}") # Convert time to indices

# From the raw object select only the EEG channels
eeg = raw.copy().pick_types(eeg=True, meg=False, eog=False)
print(f"The number of EEG channels {len(eeg.ch_names)}")

# From the EEG data pick channels by name
#print(eeg.ch_names)
eeg.pick_channels(['EEG 037', 'EEG 059'])

# Extracting data by index into Numpy Array for analysis or plotting
sampling_freq = raw.info['sfreq']
start_stop_seconds = np.array([11, 13])
start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
channel_index = 0
data_arr, times_arr = raw[channel_index, start_sample:stop_sample]
#plt.plot(times_arr, data_arr.T)
#plt.show()

# Extracting data by channel name
channel_names = ['MEG 0712', 'MEG 1022']
data_array, time_array = raw[channel_names, start_sample:stop_sample]
y_offset = np.array([5e-11, 0])  # just enough to separate the channel traces
x = time_array
y = data_array.T + y_offset
lines = plt.plot(x, y)
#plt.legend(lines, channel_names)
#plt.show()

# Extracting channel by types
eeg_channel_indices = mne.pick_types(raw.info, meg=False, eeg=True)
eeg_data, eeg_times = raw[eeg_channel_indices]
print(f"Shape of EEG data {eeg_data.shape}")

# Get the entire dataset
data, times = raw.get_data(return_times=True)
print(f"Entire continous data shape {data.shape}")
print(f"Entire continous time shape {times.shape}")

# Custom pick data
first_channel_data = raw.get_data(picks=0)
eeg_and_eog_data = raw.get_data(picks=['eeg', 'eog'])
two_meg_chans_data = raw.get_data(picks=['MEG 0712', 'MEG 1022'], start=1000, stop=2000)
print(first_channel_data.shape)
print(eeg_and_eog_data.shape)
print(two_meg_chans_data.shape)