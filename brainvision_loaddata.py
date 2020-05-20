import mne

filename = '/Users/senthilp/Desktop/mne_tutorial/BrainVision_data/Control1129.vhdr'
raw_data = mne.io.read_raw_brainvision(filename)
info = raw_data.info