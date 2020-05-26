import os
import mne

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_filt-0-40_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)

events_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_filt-0-40_raw-eve.fif'
events = mne.read_events(events_file)

# fig = raw.plot(block=True)

# How annotations are added non-interactively
eog_events = mne.preprocessing.find_eog_events(raw)
onsets = eog_events[:,0] / raw.info['sfreq'] - 0.25
durations = [0.5] * len(eog_events)
descriptions = ['bad blick'] * len(eog_events)
blink_annot = mne.Annotations(onsets, durations, descriptions, orig_time=raw.info['meas_date'])
raw.set_annotations(blink_annot)

# Blinks are usually easiest to see in the EEG channels
eeg_picks = mne.pick_types(raw.info, meg=False, eeg=True)
# raw.plot(events=eog_events, order=eeg_picks, block=True)

# Rejecting epochs based on channel amplitude thresholds
''' Setting maximum acceptable peak-to-peak amplitudes for each channel type'''
reject_criteria = dict(mag=3000e-15,     # 3000 fT 
                       grad=3000e-13,    # 3000 fT/cm
                       eeg=100e-6,       # 100 micro V
                       eog=200e-6)       # 200 micro V
'''Setting minimum acceptable peak-to-peak amplitudes for each channel type'''
flat_criteria = dict(mag=1e-15,          # 1 fT
                     grad=1e-13,         # 1 fT/cm
                     eeg=1e-6)           # 1 micro V

epochs = mne.Epochs(raw, events, tmin=-0.2, tmax=0.5, reject_tmax=0,
                    reject=reject_criteria, flat=flat_criteria,
                    reject_by_annotation=False, preload=True)
epochs.plot_drop_log()
print(epochs.drop_log)
