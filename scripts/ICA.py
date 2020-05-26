import os
from datetime import timedelta
import mne
from mne.preprocessing import *
# pylint: disable=E1101

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
#print(raw.info)
raw.crop(tmax=60).load_data()

# Visualizing the artifacts
regexp = r'(MEG [12][45][123]1|EEG 00.)'
artifact_picks = mne.pick_channels_regexp(raw.ch_names, regexp=regexp)
#raw.plot(order=artifact_picks, n_channels=len(artifact_picks), block=True)

# Blink Artifacts
eog_evoked = mne.preprocessing.create_eog_epochs(raw).average()
eog_evoked.apply_baseline(baseline=(None, -0.2))
# eog_evoked.plot_joint()

# Heart Beat Artifacts
ecg_evoked = mne.preprocessing.create_ecg_epochs(raw).average()
ecg_evoked.apply_baseline(baseline=(None, -0.2))
# ecg_evoked.plot_joint()

# Filtering to remove lower frequency drifts
filt_raw = raw.copy()
filt_raw.load_data().filter(l_freq=1.0, h_freq=None)

# Independent Component Analysis
ica = ICA(n_components=15, random_state=97)
ica.fit(filt_raw)
raw.load_data()
# ica.plot_sources(raw, block=True)
# ica.plot_components()
ica.exclude = [0, 1]
reconst_raw = raw.copy()
ica.apply(reconst_raw)
# raw.plot(order=artifact_picks, n_channels=len(artifact_picks), block=True)
# reconst_raw.plot(order=artifact_picks, n_channels=len(artifact_picks), block=True)





