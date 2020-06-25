from pymatreader import read_mat
import mne
import numpy as np 
import os
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.time_frequency import tfr_morlet
from mne.preprocessing import create_eog_epochs

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

def make_montage_cap385():
    """
    Returns:
        <class 'mne.channels.montage.DigMontage'>
    """
    montage_file = '/Users/senthilp/Downloads/standard-10-5-cap385.elp'
    montage = mne.channels.read_custom_montage(fname=montage_file, coord_frame='head')
    print(f"Created {len(ch_names[:-3])} channel positions")
    return montage


def make_montage_from_array(data_eeg: dict, ch_names: list) -> dict:
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
    ch_coords = np.column_stack([x, y, z])
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


def ica_apply(raw, n_component=15, l_freq=0.5):
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=l_freq, h_freq=None)
    ica = ICA(n_components=n_component, random_state=97)
    ica.fit(filt_raw)
    return ica


def plot_psd(epochs):
    # Plot power spectral density
    # Exploring frequency content of our epochs
    epochs.plot_psd(average=True, spatial_colors=False, fmin=0, fmax=50)
    epochs.plot_psd_topomap(normalize=True)


def single_trail_TFR(power=None, channel='AFz', segment=0):

    if (segment > 59):
        raise IndexError('Out of range, should be between 0-59')

    ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 
    'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 
    'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 
    'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 
    'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 'F6', 'F2', 'AF4', 'AF8']

    ch_index = np.arange(0, len(ch_names), 1)
    ch_dict = dict(zip(ch_names, ch_index))
    single_trail = power.data[:, ch_dict[channel], :, :] # only 1 channel as 3D matrix
    data = single_trail[segment,:,:] # Only 1 epoch as 2D matrix
    times = np.arange(-2,5,1)
    freqs = [2,3,4,5,6,7,8,9,11,13,15,17,20,23,26,30,35,40,45,50,60]

    '''
    extent=[horizontal_min,horizontal_max,vertical_min,vertical_max]
    set_xticks() on the axes will set the locations
    set_xticklabels() will set the displayed text.
    '''
    fig, ax = plt.subplots(1,1)
    im = ax.imshow(data[::-1], extent=[0, len(times)-1, 0, len(freqs)-1], 
               aspect='auto')
    x = np.arange(len(times))
    y = np.arange(len(freqs))
    ax.set_xlabel('Time')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Single trail power {channel} | segment {segment}')
    ax.set_xticks(x)
    ax.set_xticklabels(times)
    ax.set_yticks(y)
    ax.set_yticklabels(freqs)
    fig.colorbar(im)
    plt.show()

subject, session = (1, 1)
do_ica = False
filename = f"/Users/senthilp/Desktop/PD/80{subject}_{session}_PD_REST.mat"
data = read_mat(filename)
raw_dict = data['EEG']
(data_eeg, sfreq,
n_pnts, nbchan, max_time) = (raw_dict['data'][:63,:], raw_dict['srate'], raw_dict['pnts'], 
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
ch_types = (nbchan-4) * ['eeg']
ch_names = channel_name.copy()

# Create info object
info = mne.create_info(ch_names=ch_names[:-4], sfreq=sfreq, ch_types=ch_types)

# Set Channel location ( Montage for digitized electrode and headshape position data. )
my_montage = make_montage_cap385()

# Create MNE raw object
raw = mne.io.RawArray(data_eeg, info, verbose=False)
raw.set_montage(my_montage)
# raw.plot_sensors(block=True, show_names=True, title='Standard-10-5-cap385', show=True)

# Plot the raw data
# raw.plot(n_channels=10, scalings='auto', title='Data from arrays', show=True, block=True)

# Adding annotations to a Raw object
'''Annotations are a way of storing short strings of information about temporal spans of a raw object'''
my_annot = mne.Annotations(onset=sample_times_sec[1:], duration=durations[1:], description=events_type[1:])
raw.set_annotations(my_annot)
     
# Plot the annotation alongside raw data
# unfiltered_data = raw.copy()
# filtered_data = raw.copy()
# filtered_data.filter(0.5, None)
# raw.filter(l_freq=10.5, h_freq=None)
# raw.plot_psd(area_mode='range', tmax=10.0, average=True)
# fig1 = unfiltered_data.plot(n_channels=10, start=20, duration=10, scalings='auto', show=True, block=True)
# fig2 = filtered_data.plot(n_channels=10, start=20, duration=10, scalings='auto', show=True, block=True)


if do_ica:
    ica = ica_apply(raw, n_component=20, l_freq=0.5)
    ica.exclude = [3]
    ica.apply(raw)

# raw.plot(n_channels=10, scalings='auto', title='Data from arrays', show=True, block=True)
#reconst_raw.plot(n_channels=10, scalings='auto', title='Data from arrays', show=True, block=True)

print(f"Time in 51.6 sec to integer index of the sample occuring {raw.time_as_index(51.6)}")
print(f"Time in 55.298 sec to integer index of the sample occuring {raw.time_as_index(55.298)}")
print(f"Time in 183.624 sec to integer index of the sample occuring {raw.time_as_index(183.624)}")

print(f"Length of annotation {len(raw.annotations)} sec")
print(f"Duration of annotation in sec {set(raw.annotations.duration)}")
print(f"Annotation descriotion {set(raw.annotations.description)}")
print(f"Annotation onset {raw.annotations.onset[0]}\n")

# Event array is needed for epoching continuous data
events, event_dict = mne.events_from_annotations(raw)
print(f"Shape of STIM events numpy array {np.shape(events)}")
print("event array = (Index of event, Length of the event, Event type)")
print(f"STIM Event IDs: {np.unique(events[:,2])}\n")

# Creating epochs
tmin = -1.0 # start of each epoch ( 1 sec before the trigger )
tmax = 3.0 # end of each epoch ( 3 sec after the trigger )

# Load condition 1 eyes closed
event_id_eyes_closed = dict(S3=3, S4=4)
epochs_eyes_closed = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, event_id=event_id_eyes_closed, 
                    preload=True, verbose=True)
epochs_arr_eyes_closed = epochs_eyes_closed.get_data() # Get all epochs as a 3D array
print(f"Shape of eyes closed epochs array {np.shape(epochs_arr_eyes_closed)}")
print(f"Size of 1st epoch {np.shape(epochs_arr_eyes_closed[0,:,:])}")
# epochs_eyes_closed.plot(n_epochs=2, n_channels=3, block=True, scalings='auto') # scalings is Y limits for plots
# plot_psd(epochs_eyes_closed)
# single_trail_TFR(epochs_arr_eyes_closed)


# Load condition 2 eyes open
event_id_eyes_open = dict(S1=1, S2=2)
epochs_eyes_open = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, event_id=event_id_eyes_open, 
                    preload=True, verbose=True)
epochs_arr_eyes_open = epochs_eyes_open.get_data() # Get all epochs as a 3D array
print(f"Shape of eyes open epochs array {np.shape(epochs_arr_eyes_open)}")
print(f"Size of 1st epoch {np.shape(epochs_arr_eyes_open[0,:,:])}")
# print(epochs_eyes_open.drop_log[-1])
# epochs_eyes_open.plot(n_epochs=2, n_channels=3, block=True, scalings='auto') # scalings is Y limits for plots
# plot_psd(epochs_eyes_open)

# Plot epoch as Image map
# print(epochs_eyes_closed.ch_names)
# print(epochs.metadata)
# epochs_eyes_closed.plot_image(picks=['Fz'])

# Epochs to dataframe
#df = epochs.to_data_frame(time_format=None, index='condition', long_format=False)
#df.to_csv('epochs_org.csv')

# Time Frequency Analysis
tfr = False
if (tfr == True):
    freqs = [2,3,4,5,6,7,8,9,11,13,15,17,20,23,26,30,35,40,45,50]       # linearly spaced
    # freqs = np.logspace(*np.log10([4, 60]), num=21)                   # logarithmic spaced
    n_cycles = [3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7]
    power_eyes_closed = tfr_morlet(epochs_eyes_closed, freqs=freqs, n_cycles=n_cycles, 
                         use_fft=True, return_itc=False, decim=3, 
                         n_jobs=4, average=False)
    power_eyes_open = tfr_morlet(epochs_eyes_open, freqs=freqs, n_cycles=n_cycles, 
                         use_fft=True, return_itc=False, decim=3, 
                         n_jobs=4, average=False)

    power_eyes_closed_avg = power_eyes_closed.average()
    power_eyes_open_avg = power_eyes_open.average()
    power_eyes_closed_avg.save('eyesClosed-tfr.h5', overwrite=True)
    power_eyes_open_avg.save('eyesOpen-tfr.h5', overwrite=True)

# power_eyes_closed_avg = mne.time_frequency.read_tfrs('eyesClosed-tfr.h5')
# power_eyes_open_avg = mne.time_frequency.read_tfrs('eyesOpen-tfr.h5')

# vmin, vmax = (0, 60)
# np.save('vol.npy', power_eyes_closed.average().data)
# single_trail_TFR(power=power_eyes_closed, channel='AFz', segment=30)
# np.save('check.npy', power_eyes_closed_avg[0].data)

# Optional: convert power to decibels (dB = 10 * log10(power))
# power_eyes_closed_avg[0].data = 10 * np.log10(power_eyes_closed_avg[0].data)
# power_eyes_open_avg[0].data = 10 * np.log10(power_eyes_open_avg[0].data)

# pick_channel = 'POz'
# power_eyes_closed_avg[0].plot([pick_channel],
#                          title=f'Using Morlet wavelets and EpochsTFR - Eyes closed {pick_channel}', show=True)
# power_eyes_open_avg[0].plot([pick_channel],
#                         title=f'Using Morlet wavelets and EpochsTFR - Eyes open {pick_channel}', show=True)