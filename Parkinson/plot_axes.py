import matplotlib.pyplot as plt
import numpy as np


epoch1 = np.load('/Users/senthilp/Desktop/mne_tutorial/scripts/epochs1.npy')
#times = np.load('/Users/senthilp/Desktop/mne_tutorial/scripts/time.npy')

power = np.arange(0,190,10)
freqs = [2,3,4,5,6,7,8,9,11,13,15,17,20,23,26,30,35,40,45,50]

'''
extent=[horizontal_min,horizontal_max,vertical_min,vertical_max]
set_xticks() on the axes will set the locations
set_xticklabels() will set the displayed text.
'''
epoch1_mean = np.mean(epoch1, axis=0)
fig, ax = plt.subplots(1,1)
im = ax.imshow(epoch1_mean, extent=[0, len(power), 0, len(freqs)-1], 
               aspect='auto')

x = np.arange(len(power))
y = np.arange(len(freqs))

ax.set_xlabel('Time')
ax.set_ylabel('Frequency')
ax.set_title('Single trail power')

# Location for setting the labels
# control the ticks position
ax.set_xticks(y) 
ax.set_xticklabels(freqs)

ax.set_yticks(x)
ax.set_yticklabels(power)

fig.colorbar(im)
plt.show()