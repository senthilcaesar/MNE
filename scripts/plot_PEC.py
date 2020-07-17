import numpy as np
import matplotlib.pyplot as plt


group = ['PD', 'CTL']
freq = ['theta', 'alpha', 'lowerbeta', 'higherbeta', 'gamma']
band = freq[0]
parc = group[0]

EO = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PEC/{band}_{parc}_EO.txt'
with open(EO) as f:
    EO_list = f.read().splitlines() 


data = np.zeros(np.load(EO_list[0]).shape)
for data_npy in EO_list:
    data = np.add(data, np.load(data_npy))

    
data = data / len(EO_list)
# let's plot this matrix
corr = data #np.load(f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/PEC/{band}_CTL_minus_PD_EO.npy')
fig, ax = plt.subplots(figsize=(5, 5))
im = ax.imshow(corr, cmap='viridis', clim=np.percentile(corr, [5, 95]))
ax.set_xlabel('Power at 63 electrodes')
ax.set_ylabel('Power at 63 electrodes')
ax.set_title(f'Power envelope correlation - {band} (CTL minus PD) EO', fontsize=10)
fig.colorbar(im)
fig.tight_layout()
plt.savefig(f'/Users/senthilp/Desktop/{band}.png', dpi=300)




