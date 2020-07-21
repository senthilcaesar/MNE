import numpy as np
import matplotlib.pyplot as plt

electrodes = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 
    'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 
    'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 
    'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 
    'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 
    'F6', 'F2', 'AF4', 'AF8']

condition = ['EO', 'EC']
group = ['PD', 'CTL']
freq = ['theta', 'alpha', 'lowerbeta', 'higherbeta', 'gamma']
band = freq[4]
cond = condition[1]
parc = group[1]

value = ''
if band == 'theta':
    value = '4-8 Hz'
elif band == 'alpha':
    value = '8-13 Hz'
elif band =='lowerbeta':
    value = '13-21 Hz'
elif band == 'higherbeta':
    value = '21-30 Hz'
elif band == 'gamma':
    value = '30-50 Hz'
else:
    value = '4-50 Hz'
    
data_f = f'/home/senthil/caesar/MNE/scripts/data/PEC/{band}_{parc}_{cond}.txt'
print(data_f)
with open(data_f) as f:
    data_f_list = f.read().splitlines() 

data = np.zeros(np.load(data_f_list[0]).shape)
for data_npy in data_f_list:
    data = np.add(data, np.load(data_npy))

data = data / len(data_f_list)
# let's plot this matrix
corr = data
fig = plt.figure()
ax = fig.add_axes([6., 6., 6., 6., ])
im = ax.imshow(corr, cmap='viridis', 
               clim=np.percentile(corr, [5, 95]), vmin=0, vmax=1)
ax.set_xlabel('Electrodes', fontsize=20)
ax.set_ylabel('Electrodes', fontsize=20)
ax.set_xticks(np.arange(0, len(electrodes)))
ax.set_xticklabels(electrodes)
ax.set_yticks(np.arange(0, len(electrodes)))
ax.set_yticklabels(electrodes)
ax.set_title(f'{band} ({value}) {parc} {cond}', fontsize=20)
fig.colorbar(im)
#fig.tight_layout()
#fig.savefig(f'/home/senthil/Desktop/{band}_{parc}_{cond}.svg', dpi=200)
#plt.show()