# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 12:04:32 2020

@author: ancrebo
"""
import pickle
import matplotlib.pyplot as plt                                                
import pandas as pd                                                                    
import numpy as np 
import os

unitslstm = 50
numepochlstm = 50000
unitsfnn =50
numepochfnn = 5000

with open('data/workspace/lstm'+str(unitslstm)+'Nepoch'+str(numepochlstm)+'N_FNN'+str(unitsfnn)+'Nepoch'+str(numepochfnn)+'articleb.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    data_sum,timevec,data_kref,data_kref_st,data_kref_fnn,datacfd_kref,kref_vec,torsionmean,torsionmin,torsionmax,bendingmean,bendingmin,bendingmax,kref_vec_fnn,torsionmean_fnn,torsionmin_fnn,torsionmax_fnn,bendingmean_fnn,bendingmin_fnn,bendingmax_fnn,kref_vec_st,torsionmean_st,torsionmin_st,torsionmax_st,bendingmean_st,bendingmin_st,bendingmax_st,kadimcfd,mean_theta_cfd,min_theta_cfd,max_theta_cfd,mean_w_cfd,min_w_cfd,max_w_cfd = pickle.load(f)
try:   
    os.mkdir("save/"+'lstm'+str(unitslstm)+'epoch'+str(numepochlstm)+'N_FNN'+str(unitsfnn)+'Nepoch'+str(numepochfnn)+'_article')
except: 
    print("Directorio existente")
#%%
kadim_vec =  np.array([3,5.5,6.5,8.5,12.5,20,50,90])
ii = 0
for data in datacfd_kref:
    ktheta2d_adim = kadim_vec[ii]
    ii+= 1
    intpart = str(int(np.floor(ktheta2d_adim)))
    decpart = str(int(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000)))
    name_kref = "save/"+'lstm'+str(unitslstm)+'epoch'+str(numepochlstm)+'N_FNN'+str(unitsfnn)+'Nepoch'+str(numepochfnn)+"_article/data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4)+ "_cfd.csv"
    data.to_csv(name_kref)
ii = 0
print('ok!')
for data in data_kref_st:
    ktheta2d_adim = kadim_vec[ii]
    ii+= 1
    intpart = str(int(np.floor(ktheta2d_adim)))
    decpart = str(int(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000)))
    name_kref = "save/"+'lstm'+str(unitslstm)+'epoch'+str(numepochlstm)+'N_FNN'+str(unitsfnn)+'Nepoch'+str(numepochfnn)+"_article/data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4)+ "_st.csv"
    data.to_csv(name_kref)
ii = 0
print('ok!')
for data in data_kref_fnn:
    ktheta2d_adim = kadim_vec[ii]
    ii+= 1
    intpart = str(int(np.floor(ktheta2d_adim)))
    decpart = str(int(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000)))
    name_kref = "save/"+'lstm'+str(unitslstm)+'epoch'+str(numepochlstm)+'N_FNN'+str(unitsfnn)+'Nepoch'+str(numepochfnn)+"_article/data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4)+ "_FNN_"+ str(unitsfnn)+"_Epoch_" + str(numepochfnn) +".csv"
    data.to_csv(name_kref)
ii = 0
print('ok!')
for data in data_kref:
    ktheta2d_adim = kadim_vec[ii]
    ii+= 1
    intpart = str(int(np.floor(ktheta2d_adim)))
    decpart = str(int(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000)))
    name_kref = "save/"+'lstm'+str(unitslstm)+'epoch'+str(numepochlstm)+'N_FNN'+str(unitsfnn)+'Nepoch'+str(numepochfnn)+"_article/data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4)+ "_LSTM_"+ str(unitslstm)+"_Epoch_" + str(numepochlstm) +".csv"
    data.to_csv(name_kref)
print('ok!')
    
data_sum["wmean_cfd"] = - data_sum["wmean_cfd"]
data_sum.to_csv("save/"+'lstm'+str(unitslstm)+'epoch'+str(numepochlstm)+'N_FNN'+str(unitsfnn)+'Nepoch'+str(numepochfnn)+"_article/sum_aeroelastic.csv")       
#%%
plt.figure(0)
plt.plot(kadimcfd,mean_theta_cfd,label='CFD')
plt.plot(kref_vec,torsionmean*180/np.pi,label='LSTM+Aeroelastic')
plt.plot(kref_vec_st,torsionmean_st*180/np.pi,label='St')
plt.plot(kref_vec_fnn,torsionmean_fnn*180/np.pi,label='FNN+Aeroelastic')
plt.title('Mean torsion')
plt.xlabel('$k_{\Theta}^*$')
plt.ylabel('$\Theta_{mean}^*$')
plt.legend()
plt.ylim((2.5,3.5))
plt.grid()    
    
plt.figure(10)
plt.plot(kadimcfd,abs(max_theta_cfd-min_theta_cfd),label='CFD')
plt.plot(kref_vec,(torsionmax-torsionmin)*180/np.pi,label='LSTM+Aeroelastic')
plt.plot(kref_vec_st,(torsionmax_st-torsionmin_st)*180/np.pi,label='St')
plt.plot(kref_vec_fnn,(torsionmax_fnn-torsionmin_fnn)*180/np.pi,label='FNN+Aeroelastic')
plt.title('Mean torsion')
plt.xlabel('$k_{\Theta}^*$')
plt.ylabel('$\Theta_{mean}^*$')
plt.legend()
plt.ylim((0,10))
plt.grid() 
    
    
#%%
# _____________________________________________________________________________
plta=plt.figure(1)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[0]["time"].values,datacfd_kref[0]["theta"].values,label="$CFD$")
pltb.plot(timevec,data_kref[0]["theta"].values*180/np.pi,label="$LSTM$")
pltb.plot(timevec,data_kref_st[0]["theta"].values*180/np.pi,label="$St$")
pltb.plot(timevec,data_kref_fnn[0]["theta"].values*180/np.pi,label="$FNN$")
pltb.set_title('k=3')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
# pltb.set_ylim(2,3)
pltb.grid()
pltb.legend(loc='upper right')
# pltb.set_xlim(0,0.2)

plta=plt.figure(2)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[1]["time"].values,datacfd_kref[1]["theta"].values,label="$CFD$")
pltb.plot(timevec,data_kref[1]["theta"].values*180/np.pi,label="$LSTM$")
pltb.plot(timevec,data_kref_st[1]["theta"].values*180/np.pi,label="$St$")
pltb.plot(timevec,data_kref_fnn[1]["theta"].values*180/np.pi,label="$FNN$")
pltb.set_title('k=5.5')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
# pltb.set_ylim(2,3)
pltb.grid()
pltb.legend(loc='upper right')
# pltb.set_xlim(0,0.2)

plta=plt.figure(3)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[2]["time"].values,datacfd_kref[2]["theta"].values,label="$CFD$")
pltb.plot(timevec,data_kref[2]["theta"].values*180/np.pi,label="$LSTM$")
pltb.plot(timevec,data_kref_st[2]["theta"].values*180/np.pi,label="$St$")
pltb.plot(timevec,data_kref_fnn[2]["theta"].values*180/np.pi,label="$FNN$")
pltb.set_title('k=6.5')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
# pltb.set_ylim(2,3)
pltb.grid()
pltb.legend(loc='upper right')
# pltb.set_xlim(0,0.2)

plta=plt.figure(4)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[3]["time"].values,datacfd_kref[3]["theta"].values,label="$CFD$")
pltb.plot(timevec,data_kref[3]["theta"].values*180/np.pi,label="$LSTM$")
pltb.plot(timevec,data_kref_st[3]["theta"].values*180/np.pi,label="$St$")
pltb.plot(timevec,data_kref_fnn[3]["theta"].values*180/np.pi,label="$FNN$")
pltb.set_title('k=8.5')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
# pltb.set_ylim(2,3)
pltb.grid()
pltb.legend(loc='upper right')
# pltb.set_xlim(0,0.2)

plta=plt.figure(5)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[4]["time"].values,datacfd_kref[4]["theta"].values,label="$CFD$")
pltb.plot(timevec,data_kref[4]["theta"].values*180/np.pi,label="$LSTM$")
pltb.plot(timevec,data_kref_st[4]["theta"].values*180/np.pi,label="$St$")
pltb.plot(timevec,data_kref_fnn[4]["theta"].values*180/np.pi,label="$FNN$")
pltb.set_title('k=12.5')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
# pltb.set_ylim(2,3)
pltb.grid()
pltb.legend(loc='upper right')
# pltb.set_xlim(0,0.2)

plta=plt.figure(6)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[5]["time"].values,datacfd_kref[5]["theta"].values,label="$CFD$")
pltb.plot(timevec,data_kref[5]["theta"].values*180/np.pi,label="$LSTM$")
pltb.plot(timevec,data_kref_st[5]["theta"].values*180/np.pi,label="$St$")
pltb.plot(timevec,data_kref_fnn[5]["theta"].values*180/np.pi,label="$FNN$")
pltb.set_title('k=20')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
# pltb.set_ylim(2,3)
pltb.grid()
pltb.legend(loc='upper right')
# pltb.set_xlim(0,0.2)

plta=plt.figure(7)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[6]["time"].values,datacfd_kref[6]["theta"].values,label="$CFD$")
pltb.plot(timevec,data_kref[6]["theta"].values*180/np.pi,label="$LSTM$")
pltb.plot(timevec,data_kref_st[6]["theta"].values*180/np.pi,label="$St$")
pltb.plot(timevec,data_kref_fnn[6]["theta"].values*180/np.pi,label="$FNN$")
pltb.set_title('k=50')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
# pltb.set_ylim(2,3)
pltb.grid()
pltb.legend(loc='upper right')
# pltb.set_xlim(0,0.2)

plta=plt.figure(8)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[7]["time"].values,datacfd_kref[7]["theta"].values,label="$CFD$")
pltb.plot(timevec,data_kref[7]["theta"].values*180/np.pi,label="$LSTM$")
pltb.plot(timevec,data_kref_st[7]["theta"].values*180/np.pi,label="$St$")
pltb.plot(timevec,data_kref_fnn[7]["theta"].values*180/np.pi,label="$FNN$")
pltb.set_title('k=90')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
# pltb.set_ylim(2,3)
pltb.grid()
pltb.legend(loc='upper right')
# pltb.set_xlim(0,0.2)