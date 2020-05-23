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

''' Annotations in MNE-Python are a way of storing short strings of information
about temporal spans of Raw object
Annotation data structure = [onset time in seconds, duration in seconds, description a text string]
orig_time = real-world time to which the annotation onsets should be interpreted
'''

# Creating annotations programmatically
my_annot = mne.Annotations(onset=[3, 5 , 7],
                           duration=[1, 0.5, 0.25],
                           description=['AAA', 'BBB', 'CCC'])
print(my_annot.duration)
raw.set_annotations(my_annot)

# orig_time will be set to match the recording measurement date raw.info['meas_date']
meas_date = raw.info['meas_date']
orig_time = raw.annotations.orig_time
print(meas_date == orig_time)

print(f"Sampling Frequency (No of time points/sec) {raw.info['sfreq']} Hz")
# first_samp integer is the start of the data recorded at time 0 sec
print(f"{raw.first_samp} time samples have passed between the onset of the hardware \
acquistion system and the time when data started to be recorded to disk")
time_of_first_sample = raw.first_samp / raw.info['sfreq']
print(f"Time of first sample {time_of_first_sample} sec")
print(f"first sample ( 42.95597082905339 ) * sampling rate ( 600.614990234375 ) = first_sampl_integer ( 25800 )")
my_annot.onset + time_of_first_sample
print(raw.annotations.onset)

fig = raw.plot(start=2, duration=6, block=True)

# Iterating over annotations
for ann in raw.annotations:
    descr = ann['description']
    start = ann['onset']
    end = ann['onset'] + ann['duration']
    print(f"{descr} goes from {start} to {end}")

# Write and read annotations to/from CSV file
raw.annotations.save('saved-annotations.csv')
ann_from_csv_file = mne.read_annotations('saved-annotations.csv')
print(ann_from_csv_file)

