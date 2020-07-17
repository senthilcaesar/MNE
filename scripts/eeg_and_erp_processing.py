import mne

raw_fname = 'sample_audvis_filt-0-40_raw.fif'
event_fname = 'sample_audvis_filt-0-40_raw-eve.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True, verbose=False)

# This particular dataset already has an average reference projection added
# that we want to remove it
raw.set_eeg_reference([])
raw.pick_types(meg=False, eeg=True, eog=True)
print(raw.info)

raw.plot_sensors('3d', block=True)