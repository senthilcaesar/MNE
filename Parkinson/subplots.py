import matplotlib.pyplot as plt
import numpy as np

# plt.subplots(n_rows, n_cols, figsize=(width, height))
fig, ax = plt.subplots(2, 2, figsize=(12,10))
print(f"Shape of axes {ax.shape}")

ax[0][0].set_xlabel('X')
ax[0][0].set_ylabel('Y')
ax[0][0].set_title('Subplot 1')
ax[0][0].plot([1,2,3,4,5], [1,4,9,16,25])

ax[0][1].set_xlabel('X')
ax[0][1].set_ylabel('Y')
ax[0][1].set_title('Subplot 2')
ax[0][1].scatter([1,2,3,4,5], [1,4,9,16,25])

x = np.arange(1,11)
y = np.arange(1,11)
z = np.random.randn(10)

ax[1][0].plot(x, y, color='g', marker='p')
ax[1][0].set_xlabel('X')
ax[1][0].set_ylabel('Y')
ax[1][0].set_title('Subplot 3')

ax[1][1].plot(x, z, color='b', marker='o')
ax[1][1].set_xlabel('X')
ax[1][1].set_ylabel('Y')
ax[1][1].set_title('Subplot 4')


# Adding extra axes to the figure
# add_axes([left, bottom, width, height])
# ax10 = fig.add_axes([1,0,1,1])
# ax11 = fig.add_axes([1,1,1,1])
# ax10.plot(x, y)
# ax10.set_xlabel('X')
# ax10.set_ylabel('Y')
# ax10.set_title('Linear')


plt.tight_layout()
