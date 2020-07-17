import mne
import numpy as np
import matplotlib.pyplot as plt

def plot_topomap_power(power_avg, participant, condition):
    fig, axis = plt.subplots(2, 3, figsize=(7, 4))
    power_avg[0].plot_topomap(ch_type='eeg', fmin=30, fmax=50,axes=axis[0][0],
                   title='Gamma 30-50 Hz', show=False, vmin=0.0, vmax=40.0)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=21, fmax=30,axes=axis[0][1],
                   title='Higher Beta 21-30 Hz', show=False, vmin=0.0, vmax=40.0)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=13, fmax=21,axes=axis[0][2],
                   title='Lower Beta 13-21 Hz', show=False, vmin=0.0, vmax=40.0)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=8, fmax=13,axes=axis[1][0],
                   title='Alpha 8-13 Hz', show=False, vmin=0.0, vmax=40.0)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=4, fmax=8,axes=axis[1][1],
                   title='Theta 4-8 Hz', show=False, vmin=0.0, vmax=40.0)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=1, fmax=4,axes=axis[1][2],
                   title='Delta 1-4 Hz', show=False, vmin=0.0, vmax=40.0)
    fig.suptitle(f'{participant} Eyes {condition}', fontsize=16)
    mne.viz.tight_layout()
    plt.show()

participant = 'CTL'
condition = 'closed'
freq_bands = ['theta', 'alpha', 'lowerbeta', 'higherbeta', 'gamma', 'allbands']
band = freq_bands[5]
do_avg = True
plot_topo = True

filename = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_{participant}_eyes{condition}_tfr_avg_allsubejcts.h5'
tfr = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_{participant}_eyes{condition}.txt'
with open(tfr) as f:
    tfr = f.read().splitlines()

if do_avg:
    tfr_avg = mne.time_frequency.read_tfrs(tfr[0])
    for tfr in tfr[1:]:
        tfr_sub = mne.time_frequency.read_tfrs(tfr)
        tfr_avg[0].data = tfr_avg[0].data + tfr_sub[0].data
    tfr_avg[0].data = tfr_avg[0].data / len(tfr)
    tfr_avg[0].data = 10 * np.log10(tfr_avg[0].data)
    tfr_avg[0].save(filename,overwrite=True)
    #style = dict(sensors=True, image_interp='sinc')
    #tfr_avg[0].plot_joint(mode=None, timefreqs=[(0.5, 10), (1.3, 20)], topomap_args=style, title=f'{participant} Mean of 27 participants')

if plot_topo:
    tfr_avg = mne.time_frequency.read_tfrs(filename)
    plot_topomap_power(tfr_avg, participant, condition)