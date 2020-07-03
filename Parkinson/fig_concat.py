import numpy as np
import PIL
from PIL import Image

ch_names = ['Fp1', 'Fz', 'F3', 'F7', 'FT9', 'FC5', 'FC1', 'C3', 'T7', 'TP9', 'CP5', 
    'CP1', 'Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'TP10', 'CP6', 'CP2', 'Cz', 
    'C4', 'T8', 'FT10', 'FC6', 'FC2', 'F4', 'F8', 'Fp2', 'AF7', 'AF3', 'AFz', 'F1', 
    'F5', 'FT7', 'FC3', 'FCz', 'C1', 'C5', 'TP7', 'CP3', 'P1', 'P5', 'PO7', 'PO3', 
    'POz', 'PO4', 'PO8', 'P6', 'P2', 'CP4', 'TP8', 'C6', 'C2', 'FC4', 'FT8', 
    'F6', 'F2', 'AF4', 'AF8']

ch_names.sort()

ch_index = np.arange(0, len(ch_names), 1)
ch_dict = dict(zip(ch_names, ch_index))

CTL = '/Users/senthilp/Desktop/CTL_S1_and_S2/one.txt'
PD = '/Users/senthilp/Desktop/PD_S1_and_S2/two.txt'
CTL_minus_PD = '/Users/senthilp/Desktop/CTL_minus_PD_eyesopen/three.txt'

with open(CTL) as f:
    CTL_list = f.read().splitlines()  
with open(PD) as f:
    PD_list = f.read().splitlines() 
with open(CTL_minus_PD) as f:
    CTL_minus_PD_list = f.read().splitlines()
    
    
combined = list(zip(CTL_list, PD_list, CTL_minus_PD_list))


for i, (fig1, fig2, fig3) in enumerate(combined):
    list_im = [fig1, fig2, fig3]
    imgs    = [ PIL.Image.open(i) for i in list_im ]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs])[0][1]
    imgs_comb = np.hstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    # save that beautiful picture
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    #imgs_comb.save( 'Trifecta.png' )    
    # for a vertical stacking it is simple: use vstack
    imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs ) )
    imgs_comb = PIL.Image.fromarray( imgs_comb)
    imgs_comb.save(f'/Users/senthilp/Desktop/eyesopen/{ch_names[i]}.png' )