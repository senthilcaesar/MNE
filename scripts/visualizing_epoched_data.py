import os
import mne
import numpy as np 

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
raw.crop(tmax=120).load_data()

# To create the epochs data structure, we will extract the event IDs stored in STIM channel
# map those integers event IDs to more descriptive condition label using an event dictionary
# and pass those to the Epochs constructor, along with the raw data and the desired temporal limits of your epochs

events = mne.find_events(raw, stim_channel='STI 014')
event_dict = {'auditory/left': 1, 'auditory/right': 2, 'visual/left': 3,
              'visual/right': 4, 'face': 5, 'buttonpress': 32}
epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, event_id=event_dict, preload=True, verbose=False)
del raw 

catch_trials_and_buttonpress = mne.pick_events(events, include=[5, 32])
print(f"Shape of face and buttonpress epochs {np.shape(catch_trials_and_buttonpress)}")
# epochs['face'].plot(events=catch_trials_and_buttonpress, event_id=event_dict,
                    # event_colors=dict(buttonpress='red', face='blue'), block=True)

# Use projectors to clean heartbeat artifacts
ecg_proj_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_ecg-proj.fif'
ecg_projs = mne.read_proj(ecg_proj_file)
epochs.add_proj(ecg_projs)
epochs.apply_proj()
# epochs.plot_projs_topomap(vlim='joint')

# Check whether the projector is active
'''The all() function returns True if all items in an iterable are true, otherwise it returns False...'''
check_active = [ proj['active'] for proj in epochs.info['projs'] ]
# Note if your use () instead of [] the list comprehension retruns a generator
print(all(check_active))

# Plotting sensor location
# epochs.plot_sensors(kind='3d', ch_type='all', block=True)
# epochs.plot_sensors(kind='topomap', ch_type='all', block=True)

# Plotting the power spectrum of Epochs
# epochs['auditory'].plot_psd(picks='eeg')

# Plotting Epochs as an image map
# epochs['auditory'].plot_image(picks='mag', combine='mean') # Mean across magnetometers for all epochs with an auditory stimulus

# Plotting the images for individual sensor
# epochs['auditory'].plot_image(picks=['MEG 0242', 'MEG 0243'])
# epochs['auditory'].plot_image(picks=['MEG 0242', 'MEG 0243'], combine='gfp')

# Plot image for all sensors
reject_criteria = dict(mag=3000e-15,
                        grad=3000e-13,
                        eeg=150e-6)
epochs.drop_bad(reject=reject_criteria)

for ch_type, title in dict(mag='Magnetometers', grad='Gradiometers').items():
    layout = mne.channels.find_layout(epochs.info, ch_type=ch_type)
    epochs['auditory/left'].plot_topo_image(layout=layout, fig_facecolor='w',
                                            font_color='k', title=title)

# Plot maps for all EEG sensors
layout = mne.channels.find_layout(epochs.info, ch_type='eeg')
epochs['auditory/left'].plot_topo_image(layout=layout, fig_facecolor='w',
                                        font_color='k', sigma=1)

