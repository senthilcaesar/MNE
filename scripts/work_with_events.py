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
''' STIM channels are often called triggers and are used in the dataset
to mark experimental events such as stimulus onset, stimulus type and
participant response'''
events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
print(f"Shape of STIM events numpy array {np.shape(events)}")
print("3 columns in the event array (Index of event, Length of the event, Event type)")
print(f"STIM Event IDs: {np.unique(events[:,2])}")

# Mapping Event IDs to trail descriptor
'''Keeping track of which Event ID corresponds to which experimental condition is important
Event dictionaries are used when extracting epochs from continuous data and the resulting
epoch object allows pooling by requesting partial trail descriptor...
"auditory" trails will select all epochs with Event IDs 1 and 2
"left" trails will select all epochs with Event IDs 1 and 3'''

event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'smilley': 5, 'buttonpress': 32}

# Plotting events
#fig = mne.viz.plot_events(events, sfreq=raw.info['sfreq'], first_samp=raw.first_samp, event_id=event_dict)
#fig.subplots_adjust(right=0.7)

# Plotting events and raw data together
event_color = {1: 'r', 2: 'g', 3: 'b', 4: 'm', 5: 'y', 32: 'k'}
raw.plot(events=events, start=5, duration=10, color='gray', event_color=event_color, block=True)
