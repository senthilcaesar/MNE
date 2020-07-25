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


def make_montage_cap385(montage_file='default'):
    """
    Returns:
        <class 'mne.channels.montage.DigMontage'>
    """
    montage = mne.channels.read_custom_montage(fname=montage_file, coord_frame='head')
    print(f"Created {len(ch_names[:-3])} channel positions")
    return montage


def make_montage_from_array(data_eeg: dict, ch_names: list) -> dict:
    """
    Args:
        data_eeg: dictionary of key value pairs
        ch_names: list of electrode channels names
    Returns:
        Dictionary of channel positions. Keys are channel names and values are 3D coordinates
    """
    locations = ['X', 'Y', 'Z']
    # Ignore last 3 channels
    x, y, z = [ data_eeg['chanlocs'][k][:-3] for k in locations ]
    ch_coords = np.column_stack([x, y, z])
    ch_pos = dict(zip(ch_names[:-3], ch_coords))
    montage = mne.channels.make_dig_montage(ch_pos, coord_frame='head')
    print(f"Created {len(ch_names[:-3])} channel positions")
    return montage

    
def iterate_over_annotation(raw):
    for ann in raw.annotations:
        descr = ann['description']
        start = ann['onset']
        end = ann['onset'] + ann['duration']
        print(f"{descr} goes from {start} to {end}")


def ica_apply(raw, n_component=15, l_freq=0.5):
    filt_raw = raw.copy()
    filt_raw.load_data().filter(l_freq=l_freq, h_freq=None)
    ica = ICA(n_components=n_component, random_state=97)
    ica.fit(filt_raw)
    return ica


def plot_psd(epochs):
    # Plot power spectral density
    # Exploring frequency content of our epochs
    epochs.plot_psd(fmin=0, fmax=50., average=True, spatial_colors=False)
    #epochs.plot_psd_topomap(normalize=True)