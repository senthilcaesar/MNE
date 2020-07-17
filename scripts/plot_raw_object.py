import os
import mne
# pylint: disable=E1101

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
#print(raw.info)
raw.crop(tmax=60).load_data()

# Plot the raw continous data
raw.plot(block=True)

# Frequency content of each EEG channel
raw.plot_psd(picks='eeg')

# Frequenct content over all EEG channel averaged
raw.plot_psd(average=True, picks='eeg')

# Plotting Projectors from raw object
raw.plot_projs_topomap(colorbar=True)
