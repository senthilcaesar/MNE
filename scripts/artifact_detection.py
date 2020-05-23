import os
import mne
import numpy as np
# pylint: disable=E1101

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
#print(raw.info)
raw.crop(tmax=60).load_data()

# Set aside SSP ( Signal Space Projection ) for later analysis
ssp_projectors = raw.info['projs']
raw.del_proj()

# Low Frequency drifts
mag_channel = mne.pick_types(raw.info, meg='mag')
#raw.plot(duration=60, order=mag_channel, n_channels=len(mag_channel), remove_dc=False, block=True)

# Power line noise ( You can see narrow peaks at 60,120,180 and 240 Hz )
# There are some peaks around 25 to 30 Hz are probably related to the heartbeat
#fig = raw.plot_psd(tmax=np.inf, fmax=250, average=True)

# Heartbeat artifacts ( ECG )
''' Creates epochs from ECG events array'''
ecg_epochs = mne.preprocessing.create_ecg_epochs(raw)
#ecg_epochs.plot_image(combine='mean') # Mean across all the channel

# Average ECG epochs
avg_ecg_epochs = ecg_epochs.average()
avg_ecg_epochs.plot_topomap(times=np.linspace(-0.05, 0.05, 11))
avg_ecg_epochs.plot_joint()

# Ocular Artifacts ( EOG )
''' Ocular Artifacts are usually most prominent in the EEG Channel '''
eog_epochs = mne.preprocessing.create_eog_epochs(raw, baseline=(-0.5, -0.2))
eog_epochs.plot_image(combine='mean')
eog_epochs.average().plot_joint()