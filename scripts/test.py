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
    do_ica = True
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

    # Event array is needed for epoching continuous data
    events, event_dict = mne.events_from_annotations(raw, verbose=False)

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

    freqs = [2,3,4,5,6,7,8,9,11,13,15,18,21,23,26,30,35,40,45,50]
    if band == freq_bands[0]:
        freqs = freqs[2:7]
    elif band == freq_bands[1]:
        freqs = freqs[6:10]
    elif band == freq_bands[2]:
        freqs = freqs[9:13]
    elif band == freq_bands[3]:
        freqs = freqs[12:16]
    elif band == freq_bands[4]:
        freqs = freqs[15:]
    else:
        freqs = freqs[:]

    high_pass = freqs[0]
    low_pass = freqs[-1]
    epochs_eyes_open.filter(high_pass, low_pass, n_jobs=4)
    epochs_arr_eyes_open = epochs_eyes_open.get_data() # Get all epochs as a 3D array
    print(np.shape(epochs_arr_eyes_open), subject)
    np.save(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PEC/{band}_{subject}_{participant}_EO_epochs.npy', epochs_arr_eyes_open)
    # Correlation values are statistically masked
    corr = mne.connectivity.envelope_correlation(epochs_eyes_open, combine='mean', verbose=True, orthogonalize=False)
    np.save(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PEC/{band}_{subject}_{participant}_EO.npy', corr)