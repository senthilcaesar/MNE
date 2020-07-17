import os
import numpy as np 
import pandas as pd 
import mne 

'''
The dataset contain EEG data averaged across 75 subjects who were performing
a lexical decision ( word / non-word ) task . The data is in Epochs format,
with each epoch representing the response to different stimulus ( word )
'''
kiloword_data_file = '/Users/senthilp/Desktop/mne_tutorial/kword_metadata-epo.fif'
epochs = mne.read_epochs(kiloword_data_file, verbose=False)
print(f"Shape of epochs array {np.shape(epochs)}")

# View epoch metadata ( the metada is in pandas.DataFrame object )
df = epochs.metadata
print(df.dtypes)
print("Index based selection with .iloc")
print(df.iloc[2:4])

# Modifying the metadata
print(df['NumberOfLetters'].dtypes)
df['NumberOfLetters'] = df['NumberOfLetters'].map(int)
print(df['NumberOfLetters'].dtypes)
df['HighComplexity'] = df['VisualComplexity'] > 65
print(df.head())

# Selecting epochs using metadata queries
print(epochs['WORD.str.startswith("dis")'])
print(epochs['Concreteness > 6 and WordFrequency < 1'])

# Selecting epochs based on condition
# epochs['solenoid'].plot_psd()

# Select specific words for plotting
words = ['typhoon', 'bungalow', 'colossus', 'drudgery', 'linguist', 'solenoid']
# epochs[ f"WORD in {words}" ].plot(n_channels=29, block=True)

# Sorting epochs in an image plot based on word frequency
sort_order = np.argsort(df['WordFrequency'])
epochs.plot_image(order=sort_order, picks='Pz')