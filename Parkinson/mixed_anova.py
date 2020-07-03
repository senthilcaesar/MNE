import pickle
import pandas as pd
import numpy as np
import seaborn as sns
import pingouin as pg
import matplotlib.pyplot as plt
'''
ANOVA summary:

'Source': Factor names
'SS': Sums of squares
'DF': Degrees of freedom
'MS': Mean squares
'F': F-values
'p-unc': uncorrected p-values
'np2': Partial eta-square effect sizes

'''
band = 'gamma'
CTL_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_CTL_closed_meanpower.pkl'
CTL_EO = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_CTL_open_meanpower.pkl'
PD_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_PD_closed_meanpower.pkl'
PD_EO = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band}_PD_open_meanpower.pkl'

freq = dict(theta='4-8 Hz', alpha='8-13 Hz', lowerbeta='13-21 Hz',
            higherbeta='21-30 Hz', gamma='30-50 Hz')

F_open = open(CTL_EC, 'rb')
CTL_Eyesclosed = pickle.load(F_open)
F_open = open(CTL_EO, 'rb')
CTL_Eyesopen = pickle.load(F_open)
F_open = open(PD_EC, 'rb')
PD_Eyesclosed = pickle.load(F_open)
F_open = open(PD_EO, 'rb')
PD_Eyesopen = pickle.load(F_open)

n = 63
condition = ['EC', 'EO']
group = ['CTL', 'PD']

df_tfr = pd.DataFrame({'Power': np.r_[CTL_Eyesclosed, CTL_Eyesopen,
                                       PD_Eyesclosed, PD_Eyesopen],
                       'Condition': np.r_[np.repeat(condition, n), 
                                          np.repeat(condition, n)],
                       'Group': np.repeat(group, len(condition) * n),
                       'Electrode': np.r_[np.tile(np.arange(n), 2),
                                    np.tile(np.arange(n, n+n), 2)],
                       })

sns.set()
fig_name = f'{band}.png'
fig, ax = plt.subplots(1, 1, figsize=(12,10))
sns.pointplot(data=df_tfr, x='Condition', y='Power', hue='Group', dodge=True,
              markers=['o', 's'], capsize=1, errwidth=1, palette='colorblind',
              ax=ax)
ax.set_title(f'Mixed ANOVA for {band} {freq[band]}')
filename = f'/Users/senthilp/Desktop/{band}_Mixed_ANOVA'
plt.savefig(filename, dpi=300)
sd = df_tfr.groupby(['Condition', 'Group'])['Power'].agg(['mean', 'std']).round(2)

aov = pg.mixed_anova(dv='Power', within='Condition', between='Group', 
                     subject='Electrode', data=df_tfr)

posthocs = pg.pairwise_ttests(dv='Power', within='Condition', between='Group',
                              subject='Electrode', data=df_tfr)