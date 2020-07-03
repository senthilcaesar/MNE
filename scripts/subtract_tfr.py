import mne
import numpy as np
import matplotlib.pyplot as plt

def plot_topomap_power(power_avg, condition):
    fig, axis = plt.subplots(2, 3, figsize=(7, 4))
    power_avg[0].plot_topomap(ch_type='eeg', fmin=30, fmax=50,axes=axis[0][0],
                   title='Gamma 30-50 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=21, fmax=30,axes=axis[0][1],
                   title='Higher Beta 21-30 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=13, fmax=21,axes=axis[0][2],
                   title='Lower Beta 13-21 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=8, fmax=12,axes=axis[1][0],
                   title='Alpha 8-12 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=4, fmax=8,axes=axis[1][1],
                   title='Theta 4-8 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=1, fmax=4,axes=axis[1][2],
                   title='Delta 1-4 Hz', show=False)
    fig.suptitle(f'CTL minus PD Eyes {condition}', fontsize=16)
    mne.viz.tight_layout()
    plt.show()

condition = 'open'
PD_tfr = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PD_eyes{condition}_tfr_avg_allsubejcts.h5'
CTL_tfr = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/CTL_eyes{condition}_tfr_avg_allsubejcts.h5'

PD_power = mne.time_frequency.read_tfrs(PD_tfr)
CTL_power = mne.time_frequency.read_tfrs(CTL_tfr)
CTL_power[0].data = CTL_power[0].data - PD_power[0].data

#CTL_power[0].save(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/CTL-PD_eyes{condition}_tfr_avg_allsubejcts.h5', overwrite=True)
#style = dict(sensors=True, image_interp='sinc')
#CTL_power[0].plot_joint(mode=None, timefreqs=[(0.5, 10), (1.3, 20)], topomap_args=style, title=f'CTL minus PD')
plot_topomap_power(CTL_power, condition)