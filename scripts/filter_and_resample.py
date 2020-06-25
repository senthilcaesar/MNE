import os
from datetime import timedelta
import mne
# pylint: disable=E1101

sample_data_raw_file = '/Users/senthilp/Desktop/mne_tutorial/sample_audvis_raw.fif'
raw = mne.io.read_raw_fif(sample_data_raw_file, verbose=False)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
#print(raw.info)
raw.crop(tmax=60).load_data()

''' Background on filtering
1) A filter removes or attenuates parts of a signal
2) filters act on sepcific frequency ranges of a signal
3) Artifacts that are restricted to a narrow frequency range can sometimes be repaired by filtering the data
4) Slow drifts and Power line noise ( frequency-restricted artifacts )
'''

# Slow drifts
print(f"Sampling frequency {raw.info['sfreq']}")
picks = mne.pick_types(raw.info, meg='mag')

#Construct Epochs
events = mne.find_events(raw, stim_channel='STI 014')
event_id, tmin, tmax = 1, -1., 3.
baseline = (None, 0)
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=baseline,preload=True)

#epochs.resample(150., npad='auto')
epochs.plot_psd()

#raw.plot(duration=1, order=picks, proj=False, n_channels=1, remove_dc=False, block=True)
#raw.filter(l_freq=0.2, h_freq=None)
#raw.plot(duration=60, order=picks, proj=False, n_channels=10, remove_dc=False, block=True)

# for cutoff in (0.1, 0,2):
#     raw_highpass = raw.copy().filter(l_freq=cutoff, h_freq=None)
#     fig = raw_highpass.plot(duration=60, order=mag_channels, proj=False,
#                             n_channels=10, remove_dc=False, block=True)
#     fig.subplots_adjust(top=0.9)
#     fig.suptitle('High-pass filtered at {} Hz'.format(cutoff), size='xx-large', weight='bold')
