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


def GFP(epochs_eyes_closed1, epochs_eyes_closed2, session):
    eyesClosed_S3 = epochs_eyes_closed1.average()
    eyesClosed_S4 = epochs_eyes_closed2.average()
    mne.viz.plot_compare_evokeds(dict(S3=eyesClosed_S3, S4=eyesClosed_S4),
                             legend='upper left', show_sensors='lower right',
                             title='PD patient 1 - GFP S3 versus S4 (EYES CLOSED) \n Epochs averaged together in each condition')


def plot_psd(epochs):
    # Plot power spectral density
    # Exploring frequency content of our epochs
    epochs.plot_psd(fmax=60., average=True, spatial_colors=False)
    epochs.plot_psd_topomap(normalize=True)


def welch_PSD(epochs_eyes_closed):
    from mne.time_frequency import psd_welch
    fig, ax = plt.subplots(1, 1, figsize=(8,5))
    kwargs = dict(fmin=0, fmax=50, n_jobs=4)
    CTL_psds_welch_mean, freqs_mean = psd_welch(epochs_eyes_closed, average='mean', **kwargs)
    CTL_psds_welch_mean = 10 * np.log10(CTL_psds_welch_mean)
    CTL_psds_welch_mean = CTL_psds_welch_mean.mean(0).mean(0)
    #np.save('PD_PSD_mean_eyesclosed.npy', CTL_psds_welch_mean)
    PD_psds_welch_mean = np.load('PD_PSD_mean_eyesclosed.npy')
    ax.plot(freqs_mean, CTL_psds_welch_mean, color='red', ls='-', label='CTL mean of segments')
    ax.plot(freqs_mean, PD_psds_welch_mean, color='green', ls='-', label='PD mean of segments')
    y_location = np.round(np.arange(-15,15,5), 2)
    ax.set_yticks(y_location)
    y_labelname = y_location
    ax.set_yticklabels(y_labelname)
    ax.set(title='Welch PSD - Eyes closed', xlabel='Frequncy (Hz)', ylabel='Power Spectral Density (dB)')
    ax.legend(loc='upper right')
    plt.show()


def plot_topomap_power(power_avg):
    fig, axis = plt.subplots(1, 4, figsize=(7, 4))
    power_avg[0].plot_topomap(ch_type='eeg', fmin=12, fmax=30,axes=axis[0],
                   title='Beta', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=8, fmax=12,axes=axis[1],
                   title='Alpha', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=4, fmax=8,axes=axis[2],
                   title='Theta', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=1, fmax=4,axes=axis[3],
                   title='Delta', show=False)
    mne.viz.tight_layout()
    plt.show()

subject, session = (1, 1)
do_ica = False
do_tfr = False
dict_session = {1:'CTL', 2:'PD'}
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

# Adding annotations to a Raw object
'''Annotations are a way of storing short strings of information about temporal spans of a raw object'''
my_annot = mne.Annotations(onset=sample_times_sec[1:], duration=durations[1:], description=events_type[1:])
raw.set_annotations(my_annot)
     
print(f"Time in 51.6 sec to integer index of the sample occuring {raw.time_as_index(51.6)}")
print(f"Time in 55.298 sec to integer index of the sample occuring {raw.time_as_index(55.298)}")
print(f"Time in 183.624 sec to integer index of the sample occuring {raw.time_as_index(183.624)}")

print(f"Length of annotation {len(raw.annotations)} sec")
print(f"Duration of annotation in sec {set(raw.annotations.duration)}")
print(f"Annotation descriotion {set(raw.annotations.description)}")
print(f"Annotation onset {raw.annotations.onset[0]}\n")

electrodes = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 
    'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 
    'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 
    'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 
    'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 
    'F6', 'F2', 'AF4', 'AF8']

# Event array is needed for epoching continuous data
events, event_dict = mne.events_from_annotations(raw)
#print(events)
print(f"Shape of STIM events numpy array {np.shape(events)}")
print("event array = (Index of event, Length of the event, Event type)")
print(f"STIM Event IDs: {np.unique(events[:,2])}\n")

if do_ica:
    ica = ica_apply(raw, n_component=20, l_freq=0.5)
    ica.exclude = [3]
    ica.apply(raw)

# Creating epochs
tmin = -1.0 # start of each epoch ( 2 sec before the trigger )
tmax = 3.0 # end of each epoch ( 4 sec after the trigger )

# Load conditio eyes closed
event_id_eyes_closed = dict(S3=3, S4=4)
epochs_eyes_closed = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, event_id=event_id_eyes_closed, 
                    preload=True, verbose=True)
epochs_arr_eyes_closed = epochs_eyes_closed.get_data() # Get all epochs as a 3D array
print(f"Shape of eyes closed epochs array {np.shape(epochs_arr_eyes_closed)}")
print(f"Size of 1st epoch {np.shape(epochs_arr_eyes_closed[0,:,:])}")
welch_PSD(epochs_eyes_closed)
# epochs_eyes_closed.plot(n_channels=10, n_epochs=10, block=True, scalings='auto') # scalings is Y limits for plots

# Time Frequency Analysis
if (do_tfr):
    freqs = [2,3,4,5,6,7,8,9,11,13,15,17,20,23,26,30,35,40,45,50]       # linearly spaced
    # freqs = np.logspace(*np.log10([4, 60]), num=21)                   # logarithmic spaced
    n_cycles = [3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7]
    power_eyes_closed = tfr_morlet(epochs_eyes_closed, freqs=freqs, n_cycles=n_cycles, 
                         use_fft=True, return_itc=False, decim=3, 
                         n_jobs=4, average=False)
    power_eyes_closed_avg = power_eyes_closed.average()
    power_eyes_closed_avg.save(f'S3_S4-{dict_session[session]}-tfr.h5', overwrite=True)

#power_eyes_closed_avg = mne.time_frequency.read_tfrs(f'S3_S4-{dict_session[session]}-tfr.h5')

#print(type(power_eyes_closed_avg[0]))
# power_eyes_closed_avg.plot_topo(vmin=vmin, vmax=vmax, title='Using Morlet wavelets and EpochsTFR', show=True)

# Optional: convert power to decibels (dB = 10 * log10(power))
#power_eyes_closed_avg[0].data = 10 * np.log10(power_eyes_closed_avg[0].data)

#style = dict(sensors=True, image_interp='sinc')
#power_eyes_closed_avg[0].plot_joint(mode=None, timefreqs=[(0.5, 10), (1.3, 20)], topomap_args=style)

# for pick_channel in electrodes:
#     fig, axis = plt.subplots(1, 1, figsize=(8,5))
#     power_eyes_closed_avg[0].plot([pick_channel], show=False, axes=axis, colorbar=True)
#     fig.suptitle(f'{dict_session[session]} eyes closed {pick_channel}', fontsize=12)
#     fig.savefig(f'/Users/senthilp/Desktop/{dict_session[session]}_S3_and_S4/{pick_channel}.png')

# plot_topomap_power(power_eyes_closed_avg)