from pymatreader import read_mat
import mne
import numpy as np 
import os
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne.time_frequency import psd_welch
from mne.time_frequency import tfr_morlet
from mne.time_frequency import psd_multitaper
from mne.preprocessing import create_eog_epochs

'''
CTL do session 1    ( data from 28 age-matched control CTL subjects  )
No CTL do session 2 ( data from 28 Parkinsons’s disease PD patients )

Referenced to CPz

( 1 min of eyes open rest )
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


def plot_topomap_power(power_avg):
    fig, axis = plt.subplots(1, 4, figsize=(7, 4))
    power_avg[0].plot_topomap(ch_type='eeg', fmin=13, fmax=30,axes=axis[0],
                   title='Beta', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=8, fmax=13,axes=axis[1],
                   title='Alpha', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=4, fmax=8,axes=axis[2],
                   title='Theta', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=1, fmax=4,axes=axis[3],
                   title='Delta', show=False)
    plt.text(1,35,'Note: 2.4e+01 = 2.4 x 10^1')
    mne.viz.tight_layout()
    plt.show()


def welch_PSD(epochs_eyes_open, subject, participant):
    kwargs = dict(fmin=0, fmax=50, n_jobs=4)
    psds_welch_mean, freqs_mean = psd_welch(epochs_eyes_open, **kwargs)
    psds_welch_mean = 10 * np.log10(psds_welch_mean)
    psds_welch_mean = psds_welch_mean.mean(0).mean(0)
    psd_mean_subjs = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{subject}_{participant}_PSD_mean_EO.npy'
    freq_subjs = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{subject}_{participant}_freq_EO.npy'
    np.save(psd_mean_subjs, psds_welch_mean)
    np.save(freq_subjs, freqs_mean)
    return (psd_mean_subjs, freq_subjs)


def mean_welch_psd(welch_subjs_mean, freqs_mean, participant):
    PSD_mean_arr = np.zeros(len(np.load(welch_subjs_mean[0])))
    for i, wl_mean in enumerate(welch_subjs_mean):
        freq = np.load(freqs_mean[i])
        welch = np.load(wl_mean)
        PSD_mean_arr = np.add(PSD_mean_arr, welch)
    PSD_mean_arr = PSD_mean_arr / len(welch_subjs_mean)
    freq_mean_arr = freq
    np.save(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{participant}_psd_mean_EO.npy', PSD_mean_arr)
    np.save(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{participant}_freq_mean_EO.npy', freq_mean_arr)


def GFP(epochs_eyes_open1, epochs_eyes_open2, session):
    eyesOpen_S1 = epochs_eyes_open1.average()
    eyesOpen_S2 = epochs_eyes_open2.average()
    mne.viz.plot_compare_evokeds(dict(S1=eyesOpen_S1, S2=eyesOpen_S2),
                              legend='upper left', show_sensors='lower right',
                              title=f'{dict_session[session]} 1 - S1 versus S2 (EYES OPEN) \n Epochs averaged together in each condition')


def ica_apply(raw, n_component=15, l_freq=0.5):
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=l_freq, h_freq=None)
    ica = ICA(n_components=n_component, random_state=97)
    ica.fit(filt_raw)
    return ica


def plot_psd(epochs):
    # Plot power spectral density
    # Exploring frequency content of our epochs
    epochs.plot_psd(fmin=0, fmax=50., average=True, spatial_colors=False)
    #epochs.plot_psd_topomap(normalize=True)


PDsx = [801, 802, 804, 805, 806, 807, 808, 809, 810, 811, 813, 814, 815, 816, 817, 818, 
        819, 820, 821, 822, 823, 824, 825, 826, 827, 828, 829]
CTLsx = [894, 908, 890, 891, 892, 893, 895, 896, 897, 898, 899, 900, 901, 902, 903, 904, 
        905, 906, 907, 909, 910, 911, 912, 913, 914, 8060, 8070]
electrodes = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 
    'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 
    'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 
    'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 
    'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 
    'F6', 'F2', 'AF4', 'AF8']
ica_bool_PD = [None, None, 1, 2, None, None, None, None, 1, None, None, 1, None, None, None
            ,None, 3, 3, 1, 5, None, 2, None, 2, 1, None, 5]
ica_bool_CTL = [3, 2, None, None, None, 2, None, 1, 1, 1, None, 1, None, None, 1, 1, None,
                1, None, None, None, None, None, 1, 1, None, None]
participant = 'PD'
session = 1
freq_bands = ['theta', 'alpha', 'lowerbeta', 'higherbeta', 'gamma', 'allbands']
band = freq_bands[0]
ica_dict = {participant:ica_bool_PD, participant:ica_bool_CTL}

i = 0
PSD_sub_list = []
PSD_freq_list = []
my_variable = PDsx if participant == 'PD' else CTLsx
for id in my_variable:
    subject, session = (id, session)
    do_ica = False
    do_tfr = False
    dict_session = {1:'ON', 2:'OFF'}
    filename = f"/Users/senthilp/Desktop/PD_REST/{subject}_{session}_PD_REST.mat"
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

    # print(f"Shape of EEG data array {np.shape(data_eeg)}")
    # print(f"Sampling rate {sfreq} hz")
    # print(f"Total duration in sec {max_time} sec")
    # print(f"Time points per sec {sfreq}")
    # print(f"Total number of points {n_pnts}")
    # print(f"First 5 channel names {channel_name[:5]}")
    # print(f"Unique event type {np.unique(events_type)}")

    # Definition of channel types and names.
    ch_types = (nbchan-4) * ['eeg']
    ch_names = channel_name.copy()

    # Create info object
    info = mne.create_info(ch_names=ch_names[:-4], sfreq=sfreq, ch_types=ch_types, verbose=False)

    # Set Channel location ( Montage for digitized electrode and headshape position data. )
    my_montage = make_montage_cap385()

    # Create MNE raw object
    raw = mne.io.RawArray(data_eeg, info, verbose=False)
    raw.set_montage(my_montage)

    # Adding annotations to a Raw object
    '''Annotations are a way of storing short strings of information about temporal spans of a raw object'''
    my_annot = mne.Annotations(onset=sample_times_sec[1:], duration=durations[1:], description=events_type[1:])
    raw.set_annotations(my_annot)

    raw.plot(n_channels=10, start=20, duration=10, scalings='auto', show=True, block=True)
     
    # print(f"Time in 51.6 sec to integer index of the sample occuring {raw.time_as_index(51.6)}")
    # print(f"Time in 55.298 sec to integer index of the sample occuring {raw.time_as_index(55.298)}")
    # print(f"Time in 183.624 sec to integer index of the sample occuring {raw.time_as_index(183.624)}")

    # print(f"Length of annotation {len(raw.annotations)} sec")
    # print(f"Duration of annotation in sec {set(raw.annotations.duration)}")
    # print(f"Annotation descriotion {set(raw.annotations.description)}")
    # print(f"Annotation onset {raw.annotations.onset[0]}\n")

    # Event array is needed for epoching continuous data
    events, event_dict = mne.events_from_annotations(raw, verbose=False)
    # print(events)
    # print(f"Shape of STIM events numpy array {np.shape(events)}")
    # print("event array = (Index of event, Length of the event, Event type)")
    # print(f"STIM Event IDs: {np.unique(events[:,2])}\n")

    if ica_dict[participant][i] != None:
        if do_ica:
            ica = ica_apply(raw, n_component=20, l_freq=0.5)
            #ica.plot_components()
            print("Peforming ICA")
            ica.exclude = [ica_dict[participant][i]]
            ica.apply(raw)
    i += 1

    # Creating epochs
    tmin =  -2.0 # start of each epoch ( 2 sec before the trigger )
    tmax = 4.0 # end of each epoch ( 4 sec after the trigger )

    # Load condition eyes open
    event_id_eyes_open = dict(S1=1, S2=2)
    epochs_eyes_open = mne.Epochs(raw, events, tmin=tmin, tmax=tmax, event_id=event_id_eyes_open, 
                        preload=True, verbose=True)
    epochs_arr_eyes_open = epochs_eyes_open.get_data() # Get all epochs as a 3D array
    np.save('test.npy', epochs_arr_eyes_open)
    print(f"Shape of eyes open epochs array {np.shape(epochs_arr_eyes_open)}")
    print(f"Size of 1st epoch {np.shape(epochs_arr_eyes_open[0,:,:])}")
    #subject_mean, freq = welch_PSD(epochs_eyes_open, subject, participant)
    #PSD_sub_list.append(subject_mean)
    #PSD_freq_list.append(freq)

    #epochs_eyes_open.plot(n_channels=10, n_epochs=10, block=True, scalings='auto') # scalings is Y limits for plots

    # Time Frequency Analysis
    if (do_tfr):
        freqs = [2,3,4,5,6,7,8,9,11,13,15,18,21,23,26,30,35,40,45,50]    # linearly spaced
        n_cycles = [3, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 7]
        if band == freq_bands[0]:
            freqs = freqs[2:7]
            n_cycles = n_cycles[2:7]
        elif band == freq_bands[1]:
            freqs = freqs[6:10]
            n_cycles = n_cycles[6:10]
        elif band == freq_bands[2]:
            freqs = freqs[9:13]
            n_cycles = n_cycles[9:13]
        elif band == freq_bands[3]:
            freqs = freqs[12:16]
            n_cycles = n_cycles[12:16]
        elif band == freq_bands[4]:
            freqs = freqs[15:]
            n_cycles = n_cycles[15:]
        else:
             freqs = freqs[:]
             n_cycles = n_cycles[:]

        # freqs = np.logspace(*np.log10([4, 60]), num=21)                # logarithmic spaced
        print(freqs)
        power_eyes_open = tfr_morlet(epochs_eyes_open, freqs=freqs, n_cycles=n_cycles, 
                            use_fft=True, return_itc=False, decim=3, 
                            n_jobs=4, average=False)
        power_eyes_open_avg = power_eyes_open.average()
        power_eyes_open_avg.save(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_{subject}_{participant}_{dict_session[session]}_EO-tfr.h5', overwrite=True)

#mean_welch_psd(PSD_sub_list, PSD_freq_list, participant)

# power_eyes_open_avg = mne.time_frequency.read_tfrs(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_{subject}_{participant}_{dict_session[session]}_EO-tfr.h5')

# print(power_eyes_open_avg1)
# #power_eyes_open_avg.plot_topo(vmin=vmin, vmax=vmax, title='Using Morlet wavelets and EpochsTFR', show=True)

# mne.viz.plot_compare_evokeds(dict(auditory=power_eyes_open_avg1, visual=power_eyes_open_avg2),
#                              legend='upper left', show_sensors='upper right')

# Optional: convert power to decibels (dB = 10 * log10(power))
# power_eyes_open_avg[0].data = 10 * np.log10(power_eyes_open_avg[0].data)

#style = dict(sensors=True, image_interp='sinc')
#power_eyes_open_avg[0].plot_joint(mode=None, timefreqs=[(0.5, 10), (1.3, 20)], topomap_args=style)

# for pick_channel in electrodes:
#     fig, axis = plt.subplots(1, 1, figsize=(8,5))
#     power_eyes_open_avg[0].plot([pick_channel], show=False, axes=axis, colorbar=True)
#     fig.suptitle(f'{dict_session[session]} eyes open {pick_channel}', fontsize=12)
#     fig.savefig(f'/Users/senthilp/Desktop/{dict_session[session]}_S1_and_S2/{pick_channel}.png')

# plot_topomap_power(power_eyes_open_avg)