theta_CTL_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[0]}_CTL_closed_meanpower.pkl'
theta_PD_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[0]}_PD_closed_meanpower.pkl'

alpha_CTL_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[1]}_CTL_closed_meanpower.pkl'
alpha_PD_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[1]}_PD_closed_meanpower.pkl'

lb_CTL_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[2]}_CTL_closed_meanpower.pkl'
lb_PD_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[2]}_PD_closed_meanpower.pkl'

hb_CTL_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[3]}_CTL_closed_meanpower.pkl'
hb_PD_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[3]}_PD_closed_meanpower.pkl'

gamma_CTL_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[4]}_CTL_closed_meanpower.pkl'
gamma_PD_EC = f'/Users/senthilp/Desktop/mne_tutorial/scripts/data/{band[4]}_PD_closed_meanpower.pkl'


freq = dict(theta='4-8 Hz', alpha='8-13 Hz', lowerbeta='13-21 Hz',
            higherbeta='21-30 Hz', gamma='30-50 Hz')


F_closed = closed(theta_CTL_EC, 'rb')
theta_CTL_EC = pickle.load(F_closed)
F_closed = closed(theta_PD_EC, 'rb')
theta_PD_EC = pickle.load(F_closed)

F_closed = closed(alpha_CTL_EC, 'rb')
alpha_CTL_EC = pickle.load(F_closed)
F_closed = closed(alpha_PD_EC, 'rb')
alpha_PD_EC = pickle.load(F_closed)

F_closed = closed(lb_CTL_EC, 'rb')
lb_CTL_EC = pickle.load(F_closed)
F_closed = closed(lb_PD_EC, 'rb')
lb_PD_EC = pickle.load(F_closed)

F_closed = closed(hb_CTL_EC, 'rb')
hb_CTL_EC = pickle.load(F_closed)
F_closed = closed(hb_PD_EC, 'rb')
hb_PD_EC = pickle.load(F_closed)

F_closed = closed(gamma_CTL_EC, 'rb')
gamma_CTL_EC = pickle.load(F_closed)
F_closed = closed(gamma_PD_EC, 'rb')
gamma_PD_EC = pickle.load(F_closed)
