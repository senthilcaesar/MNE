from pymatreader import read_mat
import mne
import numpy as np 
import os

'''
CTL do session 1    ( data from 28 age-matched control CTL subjects  )
No CTL do session 2 ( data from 28 Parkinsons’s disease PD patients )

Referenced to CPz

( 1 min of eyes closed rest )
trigger s3 happens every 2 sec
trigger s4 happens every 2 sec

( 1 min of eyes open rest )
trigger s1 happens every 2 sec
trigger s2 happens every 2 sec

'''

def make_montage(data_eeg: dict, ch_names: list) -> dict:
    """
    Args:
        data_eeg: dictionary of key value pairs
        ch_names: list of electrode channels names
    Returns:
        Dictionary of channel positions. Keys are channel names and values are 3D coordinates
    """
    locations = ['X', 'Y', 'Z']
    # Ignore last 3 channels
    x, y, z = [ data_eeg['chanlocs'][k][:-3] for k in locations ]
    ch_coords = np.column_stack([y, x, z])
    ch_pos = dict(zip(ch_names[:-3], ch_coords))
    montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')
    print(f"Created {len(ch_names[:-3])} channel positions")
    return montage

def iterate_over_annotation(raw):
    for ann in raw.annotations:
        descr = ann['description']
        start = ann['onset']
        end = ann['onset'] + ann['duration']
        print(f"{descr} goes from {start} to {end}")

subject, session = (1, 1)
filename = f"/Users/senthilp/Desktop/PD/80{subject}_{session}_PD_REST.mat"
data = read_mat(filename)
raw_dict = data['EEG']
(data_eeg, sfreq,
n_pnts, nbchan, max_time) = (raw_dict['data'][:64,:], raw_dict['srate'], raw_dict['pnts'], 
                            raw_dict['nbchan'], raw_dict['xmax'])
channel_name = raw_dict['chanlocs']['labels']
events_type = raw_dict['event']['type']
events_type = [x.replace(" ", "") for x in events_type]
sample_times = raw_dict['event']['latency']
sample_times_sec = [(x-1)/sfreq for x in sample_times]
durations = raw_dict['event']['duration']
event_length = len(events_type)

print(f"Shape of EEG data array {np.shape(data_eeg)}")
print(f"Sampling rate {sfreq} hz")
print(f"Total duration in sec {max_time} sec")
print(f"Time points per sec {sfreq}")
print(f"Total number of points {n_pnts}")
print(f"First 5 channel names {channel_name[:5]}")
print(f"Unique event type {np.unique(events_type)}")

# Definition of channel types and names.
ch_types = (nbchan-3) * ['eeg']
ch_names = channel_name.copy()

# Create info object
info = mne.create_info(ch_names=ch_names[:-3], sfreq=sfreq, ch_types=ch_types)

# Set Channel location ( Montage for digitized electrode and headshape position data. )
my_montage = make_montage(raw_dict, ch_names)

# Create MNE raw object
raw = mne.io.RawArray(data_eeg, info, verbose=False)
raw.set_montage(my_montage)
#raw.plot_sensors(block=True, show_names=True)
#layout_from_raw = mne.channels.make_eeg_layout(raw.info)
#layout_from_raw.plot()

# Plot the raw data
# raw.plot(n_channels=10, scalings='auto', title='Data from arrays',
#          show=True, block=True)

# Adding annotations to a Raw object
'''Annotations are a way of storing short strings of information about temporal spans of a raw object'''
my_annot = mne.Annotations(onset=sample_times_sec[1:], duration=durations[1:], description=events_type[1:])
raw.set_annotations(my_annot)

# Plot the annotation alongside raw data
fig = raw.plot(n_channels=10, start=20, duration=6, scalings='auto', show=True, block=True)
print(f"Time in 51.6 sec to integer index of the sample occuring {raw.time_as_index(51.6)}")
print(f"Time in 55.298 sec to integer index of the sample occuring {raw.time_as_index(55.298)}")