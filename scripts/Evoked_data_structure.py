import os
import mne 
import numpy as np 

'''
Evoked objects typically store an EEG signal that has been averaged over
multiple epochs, which is a common techinque for estimating
stimulus-evoked activity
evoked array = (n_channels, n_times)

averaging together all epochs from one condition
'''
sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4}
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True, verbose=False)
evoked = epochs['auditory/left'].average()
del raw 

# Basic visualization of Evoked objects
evoked.plot(picks='eeg')


