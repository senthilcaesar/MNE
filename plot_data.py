import os
import numpy as np 
import matplotlib.pyplot as plt
import mne 
# pylint: disable=E1101

filename = '/Users/senthilp/Desktop/mne_tutorial/BrainVision_data/Control1129.vhdr'
raw = mne.io.read_raw_brainvision(filename)
'''Most of the fields of raw.info reflect metadata recorder at 
    acquisition time and should not be changed by the user'''
print(raw.info)
raw.crop(tmax=60).load_data()

#raw.plot(block=True)
#raw.plot_psd() # Frequency content of each channel
#raw.plot_psd(average=True) # Frequency content of all channel averaged
#raw.plot_sensors(ch_type='eeg', block=True)
