#!/bin/bash

ls -l | awk '{print $NF}' | sed -e '/theta/!d' -e '/PD/!d' -e '/EO/!d' -e '/h5/!d' > theta_PD_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/theta/!d' -e '/CTL/!d' -e '/EO/!d' -e '/h5/!d' > theta_CTL_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/alpha/!d' -e '/PD/!d' -e '/EO/!d' -e '/h5/!d' > alpha_PD_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/alpha/!d' -e '/CTL/!d' -e '/EO/!d' -e '/h5/!d' > alpha_CTL_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/lowerbeta/!d' -e '/PD/!d' -e '/EO/!d' -e '/h5/!d' > lowerbeta_PD_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/lowerbeta/!d' -e '/CTL/!d' -e '/EO/!d' -e '/h5/!d' > lowerbeta_CTL_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/higherbeta/!d' -e '/PD/!d' -e '/EO/!d' -e '/h5/!d' > higherbeta_PD_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/higherbeta/!d' -e '/CTL/!d' -e '/EO/!d' -e '/h5/!d' > higherbeta_CTL_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/gamma/!d' -e '/PD/!d' -e '/EO/!d' -e '/h5/!d' > gamma_PD_eyesopen.txt 2>&1
ls -l | awk '{print $NF}' | sed -e '/gamma/!d' -e '/CTL/!d' -e '/EO/!d' -e '/h5/!d' > gamma_CTL_eyesopen.txt 2>&1

sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' theta_PD_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' theta_CTL_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' alpha_PD_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' alpha_CTL_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' lowerbeta_PD_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' lowerbeta_CTL_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' higherbeta_PD_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' higherbeta_CTL_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' gamma_PD_eyesopen.txt
sed -i ' ' 's/^/\/Users\/senthilp\/Desktop\/mne_tutorial\/scripts\/data\//' gamma_CTL_eyesopen.txt

cp theta_PD_eyesopen.txt theta_PD_eyesclosed.txt
cp theta_CTL_eyesopen.txt theta_CTL_eyesclosed.txt
cp alpha_PD_eyesopen.txt alpha_PD_eyesclosed.txt
cp alpha_CTL_eyesopen.txt alpha_CTL_eyesclosed.txt
cp lowerbeta_PD_eyesopen.txt lowerbeta_PD_eyesclosed.txt
cp lowerbeta_CTL_eyesopen.txt lowerbeta_CTL_eyesclosed.txt
cp higherbeta_PD_eyesopen.txt higherbeta_PD_eyesclosed.txt
cp higherbeta_CTL_eyesopen.txt higherbeta_CTL_eyesclosed.txt
cp gamma_PD_eyesopen.txt gamma_PD_eyesclosed.txt
cp gamma_CTL_eyesopen.txt gamma_CTL_eyesclosed.txt

sed -i ' ' 's/EO/EC/g' theta_PD_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' theta_CTL_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' alpha_PD_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' alpha_CTL_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' lowerbeta_PD_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' lowerbeta_CTL_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' higherbeta_PD_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' higherbeta_CTL_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' gamma_PD_eyesclosed.txt
sed -i ' ' 's/EO/EC/g' gamma_CTL_eyesclosed.txt

