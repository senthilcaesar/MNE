import mne
import os
import numpy as np 
# pylint: disable=E1101

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
raw.crop(tmax=60).load_data()
n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)
sample_freq =raw.info['sfreq'] 
print(f"The cropped sample data object has {n_time_samps} time samples and {n_chan} channels.")
print(f"The last time sample at {time_secs[-1]} seconds.")
print(f"The first few channel names are {ch_names[:3]}")
print(f"Bad channles marked during data acquisition {raw.info['bads']}")
print(f"Sampling Frequency {sample_freq} Hz")
print(f"Miscellaneous acquisition info {raw.info['description']}")

'''Annoatations / Events provide a mapping between times during
and EEG recording and a description of what happened at those times

                                   |  Event Data Structure    |     Annotation Data Structure
---------------------------------------------------------------------------------------------
UNITS ( when )                     |  samples                 |     seconds
LIMITS ON THE DESCRIPTION ( what ) |  Event ID                |     String
HOW DURATION IS ENCODED            |  No duration             |     Include a duration
INTERNAL REPRESENTATION            |  Numpy array             |     list 
'''

# STIM Channel ( stimulus channel )
# raw.copy().pick_types(meg=False, stim=True).plot(start=4, duration=6, block=True)
# Convert STIM Channel to events array
events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
print(f"Shape of STIM events numpy array {np.shape(events)}")
print("3 columns in the event array (Index of event, Length of the event, Event type)")
print(f"STIM Event IDs: {np.unique(events[:,2])}")

# Events array to annotations object
mapping = {1: 'auditory/left', 2: 'auditory/right', 3: 'visual/left',
           4: 'visual/right', 5: 'smiley', 32: 'buttonpress'}
onsets = events[:,0] / sample_freq
durations = np.zeros_like(onsets)
descriptions = [ mapping[event_id] for event_id in events[:,2] ]
annot_from_events = mne.Annotations(onset=onsets, duration=durations,
                                    description=descriptions,
                                    orig_time=raw.info['meas_date'])
raw.set_annotations(annot_from_events)
raw.plot(start=5, duration=5, block=True)