import numpy as np


'''MNI 152 to MNI 305
I have an RAS from a voxel in MNI152 space 
which I want to convert to MNI305 space (fsaverage space)'''

trans = np.array([[1.0022,0.0071,-0.0177,0.0528], 
                  [-0.0146,0.9990,0.0027,-1.5519], 
                  [0.0129,0.0094,1.0027,-1.2012]])

coords = np.loadtxt('/Users/senthilp/Desktop/coords.txt')
new_col = np.ones((131326,1))
coords = np.append(coords, new_col, 1)

f = open("/Users/senthilp/Desktop/demofile2.txt", "a")
for val in coords:
    affine = np.dot(trans, val)
    x, y, z = str(affine[0]), str(affine[1]), str(affine[2])
    out_str = f'{x} {y} {z}\n'
    f.write(out_str)
f.close()


import nibabel as nib
MNI152_vol = '/Users/senthilp/Desktop/SATA/Roast/example/MNI152_T1_1mm.nii'
t1 = nib.load(MNI152_vol)
CD = np.loadtxt('/Users/senthilp/Desktop/exp_val.txt', dtype='float64')
coords = np.loadtxt('/Users/senthilp/Desktop/coords_exp.txt')
img = np.zeros([182,218,182])

count = 0
for idx, val in enumerate(coords):
    i, j, k = int(val[0]), int(val[1]), int(val[2])
    img[i][j][k] = CD[idx]
    count += 1
            
affine = t1.affine
hdr = t1.header
result_img = nib.Nifti1Image(img, affine, header=hdr)

output = '/Users/senthilp/Desktop/exp.nii.gz'
result_img.to_filename(output)
