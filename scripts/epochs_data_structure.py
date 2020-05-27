import os
import mne
import numpy as np 

'''
epochs array = (n_epochs, n_channels, n_times)
'''
sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
raw.crop(tmax=60).load_data()

n_time_samps = raw.n_times
time_secs = raw.times
ch_names = raw.ch_names
n_chan = len(ch_names)
sample_freq =raw.info['sfreq'] 
print(f"The cropped sample data object has {n_time_samps} time samples and {n_chan} channels.")
print(f"The last time sample at {time_secs[-1]} seconds.")
print(f"The first few channel names are {ch_names[:3]}")
print(f"Bad channels marked during data acquisition {raw.info['bads']}")
print(f"Sampling Frequency {sample_freq} Hz")
print(f"Miscellaneous acquisition info {raw.info['description']}")

# Extract events from raw object
events = mne.find_events(raw, stim_channel='STI 014', verbose=False)
print(f"Shape of STIM events numpy array {np.shape(events)}")
print("event array = (Index of event, Length of the event, Event type)")
print(f"STIM Event IDs: {np.unique(events[:,2])}")

# The raw object and the events array are the bare minimum to create an epochs object
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
'''
-0.3  -0.2  -0.1  0.0  0.1  0.2  0.3  0.4  0.5  0.6  0.7   # time in sec

    60    60    60   60   60   60   60   60   60   60      # time points ( 601 )

'''
epochs = mne.Epochs(raw, events, tmin=-0.3, tmax=0.7, event_id=event_dict, preload=True, verbose=False)
epochs_arr = epochs.get_data() # Get all epochs as a 3D array
print(f"Shape of epochs array {np.shape(epochs_arr)}")
print(f"Size of 1st epoch {np.shape(epochs_arr[0,:,:])}")
print(epochs.event_id)
del raw 

# Basic visualization of Epochs object
# epochs.plot(n_epochs=10, block=True)

# Pool across conditions
print(f"All trails where the stimulus was a face {np.shape(epochs['face'])}")
print(f"All trails where the stimulus was a visual {np.shape(epochs['visual'])}")
print(f"All trails where the stimulus was a auditory {np.shape(epochs['auditory'])}")
print(f"All trails where the stimulus was a buttonpress {np.shape(epochs['buttonpress'])}")
# epochs['face'].plot(n_epochs=10, block=True)

# Selecting epochs by index
print(f"Epochs 0-9 of epoch object {np.shape(epochs[:10])}")
print(f"Epochs 0-9 of epoch numpy array {np.shape(epochs_arr[:10,:,:])}")
print(f"Epochs 1, 3, 5, 7 of epoch object {np.shape(epochs[1:8:2])}")

# Selecting, dropping and reordering channels
epochs_eeg = epochs.copy().pick_types(meg=False, eeg=True)
print(epochs_eeg.ch_names)

new_order = ['EEG 002', 'STI 014', 'EOG 061', 'MEG 2521']
epochs_subset = epochs.copy().reorder_channels(new_order)
print(epochs_subset.ch_names)

# Changing channel name and type
epochs.rename_channels({'EOG 061': 'BlinkChannel'})
epochs.set_channel_types({'EEG 060': 'ecg'})
print(list(zip(epochs.ch_names, epochs.get_channel_types()))[-4:])
epochs.rename_channels({'BlinkChannel': 'EOG 061'})
epochs.set_channel_types({'EEG 060': 'eeg'})

# Selection in the time domain
shorter_epochs = epochs.copy().crop(tmin=-0.1, tmax=0.1) # 120 time points
# shorter_epochs.plot(block=True, n_epochs=5)
for name, obj in dict(Original=epochs, Cropped=shorter_epochs).items():
    print(f"{name} data has {obj.get_data().shape[-1]} time samples")

# Get single epoch from epochs object
single_epoch = epochs.get_data(item=2)
print(np.shape(single_epoch))

# Export data to Pandas DataFrames
df = epochs.to_data_frame(index=['condition', 'epoch', 'time'])
df.to_csv('file1.csv')

# Loading and saving Epochs objects to disk
epochs.save('saved-audiovisual-epo.fif', overwrite=True)
epochs_from_file = mne.read_epochs('saved-audiovisual-epo.fif', preload=True)

# Interating over Epochs
for epoch in epochs[:3]:
    print(type(epoch))