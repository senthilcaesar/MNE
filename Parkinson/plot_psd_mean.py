import matplotlib.pyplot as plt
import numpy as np

# plt.subplots(n_rows, n_cols, figsize=(width, height))
fig, ax = plt.subplots(1, 2, figsize=(12,4))
print(f"Shape of axes {ax.shape}")

x1_EO = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PD_freq_mean_EO.npy'
y1_EO = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PD_psd_mean_EO.npy'
y2_EO = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/CTL_psd_mean_EO.npy'

x1_EO = np.load(x1_EO)
y1_EO = np.load(y1_EO)
y2_EO = np.load(y2_EO)

ax[0].set_xlabel('Frequency (Hz)')
ax[0].set_ylabel('Power Spectral Density (dB)')
ax[0].set_title('Welch PSD - Eyes open')
ax[0].set_xlim([0,50])
ax[0].set_ylim([-15,10])
ax[0].plot(x1_EO, y1_EO, marker='o', color='r',ls='--', label='PD mean of 27 subjects')
ax[0].plot(x1_EO, y2_EO, marker='o', color='b', ls='--', label='CTL mean of 27 subjects')
ax[0].legend(loc='upper right')

x1_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PD_freq_mean_EC.npy'
y1_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PD_psd_mean_EC.npy'
y2_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/CTL_psd_mean_EC.npy'

x1_EC = np.load(x1_EC)
y1_EC = np.load(y1_EC)
y2_EC = np.load(y2_EC)

ax[1].set_xlabel('Frequency (Hz)')
ax[1].set_ylabel('Power Spectral Density (dB)')
ax[1].set_title('Welch PSD - Eyes closed')
ax[1].set_xlim([0,50])
ax[1].set_ylim([-15,10])
ax[1].plot(x1_EC, y1_EC, marker='o', color='r',ls='--', label='PD mean of 27 subjects')
ax[1].plot(x1_EC, y2_EC, marker='o', color='b', ls='--', label='CTL mean of 27 subjects')
ax[1].legend(loc='upper right')

plt.savefig('/Users/senthilp/Desktop/filename.png', dpi=300)
