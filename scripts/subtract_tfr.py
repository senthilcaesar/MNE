import mne
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'xx-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'xx-large',
         'ytick.labelsize':'xx-large'}
pylab.rcParams.update(params)

def plot_topomap_power(power_avg, condition):
    fig, axis = plt.subplots(2, 3, figsize=(7, 4))
    power_avg[0].plot_topomap(ch_type='eeg', fmin=30, fmax=50,axes=axis[0][0],
                   title='Gamma 30-50 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=21, fmax=30,axes=axis[0][1],
                   title='Higher Beta 21-30 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=13, fmax=21,axes=axis[0][2],
                   title='Lower Beta 13-21 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=8, fmax=13,axes=axis[1][0],
                   title='Alpha 8-13 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=4, fmax=8,axes=axis[1][1],
                   title='Theta 4-8 Hz', show=False)
    power_avg[0].plot_topomap(ch_type='eeg', fmin=1, fmax=4,axes=axis[1][2],
                   title='Delta 1-4 Hz', show=False)
    fig.suptitle(f'CTL minus PD Eyes {condition}', fontsize=16)
    mne.viz.tight_layout()
    plt.show()


condition = 'closed'
freq_bands = ['theta', 'alpha', 'lowerbeta', 'higherbeta', 'gamma', 'allbands']
band = freq_bands[5]

PD_tfr = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_PD_eyes{condition}_tfr_avg_allsubejcts.h5'
CTL_tfr = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_CTL_eyes{condition}_tfr_avg_allsubejcts.h5'

PD_power = mne.time_frequency.read_tfrs(PD_tfr)
CTL_power = mne.time_frequency.read_tfrs(CTL_tfr)


# fig, ax1 = plt.subplots(1, 1, figsize=(30,8))
# CTL_power[0].plot(picks="eeg", combine="mean", yscale='log', title=None,
# vmin=-30, vmax=30, axes=ax1, show=False)
# ax1.set_label('Time (s)')
# ax1.set_ylabel('Frequency (Hz)')
# ax1.set_title('CTL mean of 27 participants', fontsize=26)
# plt.savefig('/Users/senthilp/Desktop/1.png', dpi=100)

# fig, ax2 = plt.subplots(1, 1, figsize=(30,8))
# PD_power[0].plot(picks="eeg", combine="mean", yscale='log', title=None,
# vmin=-30, vmax=30, axes=ax2, show=False)
# ax2.set_label('Time (s)')
# ax2.set_ylabel('Frequency (Hz)')
# ax2.set_title('PD mean of 27 participants', fontsize=26)
# plt.savefig('/Users/senthilp/Desktop/2.png', dpi=100)

#CTL_power[0].data = CTL_power[0].data - PD_power[0].data

print(CTL_power[0].data.shape)
dlist = []
some = CTL_power[0].data
#for i in range(20):
print(f"Shape of frequency {some[:,0,:].shape}")
dlist.append(some[:,10,:])
#dlist.append(some[:,10,:])

print(len(dlist))
coor = mne.connectivity.envelope_correlation(dlist, combine='mean', orthogonalize='pairwise', verbose=True)
print(coor.shape)

np.save('/Users/senthilp/Desktop/test.npy', coor)

# fig, ax3 = plt.subplots(1, 1, figsize=(30,8))
# CTL_power[0].plot(picks="eeg", combine="mean", yscale='log', axes=ax3, show=False, title=None)
# ax3.set_label('Time (s)')
# ax3.set_ylabel('Frequency (Hz)')
# ax3.set_title('CTL minus PD', fontsize=26)
# plt.savefig('/Users/senthilp/Desktop/3.png', dpi=100)












#CTL_power[0].save(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/CTL-PD_eyes{condition}_tfr_avg_allsubejcts.h5', overwrite=True)
# style = dict(sensors=True, image_interp='sinc')
# CTL_power[0].plot_joint(yscale='log', mode=None, timefreqs=[(0.5, 10), (1.3, 20)], topomap_args=style, title=f'CTL minus PD')
#plot_topomap_power(CTL_power, condition)