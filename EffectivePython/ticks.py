import matplotlib.pyplot as plt
import numpy as np

x = np.arange(1, 11)
y = np.arange(1, 11)

fig, ax = plt.subplots(1, 2, figsize=(12,6))

ax[0].plot(x, y, color='g', marker='p')
ax[0].set_xlabel('X')
ax[0].set_ylabel('Y')
ax[0].set_title('Axes 1')
ax[0].set_xticks([1,3,5,7,9,11])
ax[0].set_xticklabels(['one','three','five','seven',
                       'nine','eleven'])