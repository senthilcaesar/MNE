from pymatreader import read_mat
import mne
import numpy as np 
import os

'''
CTL do session 1    ( data from 28 healthy age-matched control CTL subjects  )
No CTL do session 2 ( data from 28 Parkinsons’s disease PD patients )
Referenced to CPz
'''

filename = '/Users/senthilp/Desktop/PD/801_1_PD_REST.mat'
data = read_mat(filename)
raw_dict = data['EEG']
data_eeg, sfreq, n_pnts, max_time = (raw_dict['data'], raw_dict['srate'], raw_dict['pnts'], raw_dict['xmax'])
channel_name = raw_dict['chanlocs']['labels']
events_type = raw_dict['event']['type']
sample_times = raw_dict['event']['latency']
durations = raw_dict['event']['duration']
markers = list(zip(events_type,sample_times))

print(f"Shape of EEG data array {np.shape(data_eeg)}")
print(f"Sampling rate {sfreq} hz")
print(f"Total duration in sec {max_time} sec")
print(f"Time points per sec {sfreq}")
print(f"Total number of points {n_pnts}")
print(f"First 5 channel names {channel_name[:5]}")
print(f"Unique event type {np.unique(events_type)}")

# Definition of channel types and names.
ch_types = 67 * ['eeg']
ch_names = channel_name.copy()

# Create info object
info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

# Create MNE raw object
raw = mne.io.RawArray(data_eeg, info, verbose=False)

# Plot the raw data
#raw.plot(n_channels=4, scalings='auto', title='Data from arrays',
#         show=True, block=True)

# Adding annotations to a Raw object
'''Annotations are a way of storing short strings of information about temporal spans of a raw object'''
onset_sec = [x+3.698 for x in range(120)]
my_annot = mne.Annotations(onset=onset_sec, duration=durations[1:], description=events_type[1:])
time_of_first_sample = 25800 / sfreq
my_annot.onset = my_annot.onset + time_of_first_sample
raw.set_annotations(my_annot)

'''
55.298 sec to 115.298 sec ( 1 min of eyes closed rest )
trigger s3 happens every 2 sec
trigger s4 happens every 2 sec

115.298 sec to 175.298 sec ( 1 min of eyes open rest )
trigger s1 happens every 2 sec
trigger s2 happens every 2 sec
'''

# Plot the annotation alongside raw data
fig = raw.plot(n_channels=6, start=50, duration=6, scalings='auto', show=True, block=True)
#print(raw.info['chs'])
print(f"Time in 51.6 sec to integer index of the sample occuring {raw.time_as_index(51.6)}")
print(f"Time in 55.298 sec to integer index of the sample occuring {raw.time_as_index(55.298)}")

# Iterate over annotation
for ann in raw.annotations:
    descr = ann['description']
    start = ann['onset']
    end = ann['onset'] + ann['duration']
    print(f"{descr} goes from {start} to {end}")
