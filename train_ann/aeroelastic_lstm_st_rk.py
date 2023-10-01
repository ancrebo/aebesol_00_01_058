# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 12:56:41 2020

@author: ancrebo
"""


# Aeroelastic motion code
import matplotlib.pyplot as plt                                                
import pandas as pd                                                            
import seaborn as sns                                                          
import math                                                                    
import os                                                                    
import numpy as np  
import scipy                                                           

import copy

import tensorflow as tf

from tensorflow import keras                                                   
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error                                      
from time import time as Tic
from scipy.interpolate import interp1d

import pickle

print(tf.__version__) 



#vinf = 20
chord = 0.1
#%% Static polar
polar = pd.read_csv('data/polar/polar.csv')
f_cl_st = interp1d(polar["aoa"].values,polar["cl"].values)


# %%
# Se lee la polar estacionaria del perfil
pathpolarbase = 'data/polares_w' 
polar = []
for f in [f_ for f_ in os.listdir(pathpolarbase) if 'a0' in f_]:
 polaruni = pd.read_csv(pathpolarbase+f)
 polar = pd.concat([polar, polaruni])
dataset_path_base = '../cycle_bbdd3/'
ii = 0
# %%
# Se leen todos los datos temporales

data_tot = pd.DataFrame()
data_tot1 = pd.DataFrame()
for f in [f_ for f_ in os.listdir(dataset_path_base) if 'a0' in f_]:
    print(f)
    data = pd.read_csv(dataset_path_base + f, sep=",")
    try:
        data.index += data_tot.index.values[-1]
    except IndexError:
        pass
    # -----------------------------------------------------------------
    alpha = (-data["rotation Monitor: Expression (deg)"].values
             + data["Alpha0 Monitor: Expression"].values * 180 / np.pi)
    f_cl_st_w = interp2d(polar["w"].values,polar["aoa"].values,polar["cl"].values)
    cl_st = f_cl_st(alpha)
    data["cl_st"] = cl_st
    f_cm_st_w = interp2d(polar["w"].values,polar["aoa"].values,polar["cm "].values)
    cm_st = f_cm_st(alpha)
    data["cm_st"] = cm_st
    
    freq = data["Frequency Monitor: Expression"].values[-1]      
    time = data["Physical Time: Physical Time (s)"].values                          
    delta_time = time[-1]  - time[-2]               
    point_period = round(1 / (freq * delta_time))  
                                                              
    alpha_p = np.gradient(alpha,delta_time, edge_order=2)                      
    data['Alpha prima'] = alpha_p                                             
    alpha_pp = np.gradient(alpha_p, delta_time, edge_order=2)                                      
    data['Alpha prima prima'] = alpha_pp                                                                                    # The data is concatenated in a global data frame
    tcar = 0.1/20
    npoint_past = int(np.floor(tcar/delta_time))
    void = np.nan*np.ones(npoint_past)
    alpha_past = np.array([*void, *alpha.tolist()[0:len(alpha)-npoint_past]])   
    alpha_past =  alpha_past.astype(np.float)                          
    data['Alpha past'] = alpha_past                                    
    alphaprima_past = np.array([*void, *alpha_p.tolist()[0:len(alpha)-npoint_past]])   
    alphaprima_past =  alphaprima_past.astype(np.float)                          
    data['Alpha prima past'] = alphaprima_past                                   
    alphaprimaprima_past = np.array([*void, *alpha_pp.tolist()[0:len(alpha)-npoint_past]])    
    alphaprimaprima_past =  alphaprimaprima_past.astype(np.float)                         
    data['Alpha prima prima past'] = alphaprimaprima_past            
    dataperiod = data.tail(int(point_period))
    dataperiod['deltaCL'] = dataperiod["CL Monitor: Force Coefficient"] - dataperiod["cl_st"]
    dataperiod['deltaCM'] = dataperiod["CM Monitor: Moment Coefficient"]/chord - dataperiod["cm_st"]
    dataperiod = dataperiod.dropna()
    #plt.figure()
    #plt.plot(dataperiod["CL Monitor: Force Coefficient"].values)
    datautil = pd.DataFrame()
    
    datautil["CM"] = dataperiod["CM Monitor: Moment Coefficient"].values/chord
    datautil["CM_st"] = dataperiod["cm_st"].values
    datautil["Delta CM"] =  dataperiod["deltaCM"].values    
    datautil["CL"] = dataperiod["CL Monitor: Force Coefficient"].values
    datautil["CL_st"] = dataperiod["cl_st"].values
    datautil["Delta CL"] =  dataperiod["deltaCL"].values
    datautil["Alpha0"] = dataperiod["Alpha0 Monitor: Expression"].values*180/np.pi
    datautil["Delta Alpha"] = -dataperiod["rotation Monitor: Expression (deg)"].values
    datautil["Alpha prima"] = dataperiod["Alpha prima"].values
    datautil["Alpha prima prima"] = dataperiod["Alpha prima prima"].values
    datautil["Flag"] = np.concatenate(([1E-0],np.zeros(int(point_period)-1)))
    
    if  data["Amplitude Monitor: Expression"].values[-1]*180/np.pi<5 and  data["Alpha0 Monitor: Expression"].values[-1]*180/np.pi<7 and freq>10 and freq<60:
    
        if np.mod(ii,10)==0:
            data_tot = pd.concat([data_tot, datautil],axis=0)
        else:
            data_tot1 = pd.concat([data_tot1, datautil],axis=0)
        ii=ii+1 
data_tot1 = data_tot1.dropna()
data_tot = data_tot.dropna()
data_tot1fnn = data_tot1.copy()
data_totfnn = data_tot.copy()
#%%
def norm(x,y):
    return (x-np.amin(y))/(np.amax(y)-np.amin(y))

def desnorm(x,y):
    return x*(np.amax(y)-np.amin(y))+np.amin(y)

def sigm(x,y):
    return 1/(1+math.exp(-(x-y)))

def negsigm(x,y):
    return 1/(1+math.exp(x-y))

deltacl_train = data_tot["Delta CL"].values
deltacm_train = data_tot["Delta CM"].values
alpha0_train = data_tot["Alpha0"].values
deltaalpha_train = data_tot["Delta Alpha"].values
alphap_train = data_tot["Alpha prima"].values
alphapp_train = data_tot["Alpha prima prima"].values

deltacl_train1 = data_tot1["Delta CL"].values
deltacm_train1 = data_tot1["Delta CM"].values
alpha0_train1 = data_tot1["Alpha0"].values
deltaalpha_train1 = data_tot1["Delta Alpha"].values
alphap_train1 = data_tot1["Alpha prima"].values
alphapp_train1 = data_tot1["Alpha prima prima"].values

deltacl_trainfnn = data_totfnn["Delta CL"].values
deltacm_trainfnn = data_totfnn["Delta CM"].values
alpha0_trainfnn = data_totfnn["Alpha0"].values
deltaalpha_trainfnn = data_totfnn["Delta Alpha"].values
alphap_trainfnn = data_totfnn["Alpha prima"].values
alphapp_trainfnn = data_totfnn["Alpha prima prima"].values

deltacl_train1fnn = data_tot1fnn["Delta CL"].values
deltacm_train1fnn = data_tot1fnn["Delta CM"].values
alpha0_train1fnn = data_tot1fnn["Alpha0"].values
deltaalpha_train1fnn = data_tot1fnn["Delta Alpha"].values
alphap_train1fnn = data_tot1fnn["Alpha prima"].values
alphapp_train1fnn = data_tot1fnn["Alpha prima prima"].values
#%%
# Simulation
deltatime = 0.5
totaltime = 2000
numtime = int(totaltime/deltatime)
timevec = np.linspace(0,totaltime,numtime)
#%% LOAD NEURAL NETWORK MODEL LSTM 20-1
unitslstm1 = 10
numepoch = 50000
namesave1 = 'LSTM_'+str(unitslstm1)+'N_'+str(numepoch)+'epoch'
namesavered1 = 'data/redes/'+namesave1
model1 = keras.models.load_model(namesavered1 + 'model1.h5')
#%%
weights0 = model1.layers[0].get_weights()
weights1 = model1.layers[1].get_weights()
dimvar = 5
#dimtemp = 1
model1b = Sequential()
model1b.add(LSTM(unitslstm1,  batch_input_shape=(numtime,1, dimvar)))
model1b.add(Dense(2))
model1b.layers[0].set_weights(weights0)
model1b.layers[1].set_weights(weights1)
optimizer = tf.keras.optimizers.RMSprop(0.001) 
model1b.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse']) 

#%% 
def eqsys(xvec,Amat,bmat):
    return np.matmul(Amat,xvec)+bmat


#%% Structural Model
# kadim_vec_st = np.array([3,4,5,6,7,12,21,80])#[5,7,20,80]np.array([470,700,1000,1300,1600,1920,2000,3300,13480])*1e6
# vinf_vec_st = np.zeros(len(kadim_vec_st),)
# torsionmean_st = np.zeros(len(kadim_vec_st),) #np.zeros(7,)
# bendingmean_st = np.zeros(len(kadim_vec_st),) 
# kref_vec_st = np.zeros(len(kadim_vec_st),) # np.zeros(7,)
# data_kref_st = []
# torsionmin_st = np.zeros(len(kadim_vec_st),) #np.zeros(7,)
# bendingmin_st = np.zeros(len(kadim_vec_st),) 
# torsionmax_st = np.zeros(len(kadim_vec_st),) #np.zeros(7,)
# bendingmax_st = np.zeros(len(kadim_vec_st),) 
# for jj in np.arange(len(kadim_vec_st)):
#     data_vec_krefuni = pd.DataFrame()
#     rhoinf =  1.18415
#     rhos = 1180 
#     thickness = 4e-3
#     young_E = 3300*1e6
#     inertia_area = 1/12*chord*thickness**3
#     length = 0.37
#     poisson = 0.35
#     young_G = young_E/(2*(1+poisson))
#     inertia_torsion = 1/3*chord*thickness**3
#     M11theta = 0.455 
#     A11theta = 0.455 
#     K11theta = 1.2005
#     M11w = 0.247 
#     A11thetaw = 0.325 
#     K11w = 3.189

#     I2d = 1/12*rhos*chord**3*thickness*M11theta/A11theta
#     ktheta2d = young_G*inertia_torsion/(length**2)*K11theta/A11theta
#     m2d = rhos*chord*thickness*M11w/A11thetaw
#     kw2d = young_E*inertia_area/(length**4)*K11w/A11thetaw

    
#     ktheta2d_adim = kadim_vec_st[jj]
#     vinf = (ktheta2d/(0.5*rhoinf*ktheta2d_adim*chord**2))**0.5
#     I2d_adim =  I2d/(0.5*rhoinf*chord**4)
#     m2d_adim = m2d/(0.5*rhoinf*chord**2)
#     kw2d_adim = kw2d/(0.5*rhoinf*vinf**2)
    
#     kref_vec_st[jj] = ktheta2d_adim
#     vinf_vec_st[jj] = vinf
    
#     theta_vec = np.zeros((numtime,))
#     w_vec = np.zeros((numtime,))
#     aoa_vec = np.zeros((numtime,))
#     thetad_vec = np.zeros((numtime,))
#     wd_vec = np.zeros((numtime,))
#     aoad_vec = np.zeros((numtime,))
#     thetadd_vec = np.zeros((numtime,))
#     wdd_vec = np.zeros((numtime,))
#     aoadd_vec = np.zeros((numtime,))
#     wddd_vec = np.zeros((numtime,))
#     flag_vec = np.zeros((numtime,))
#     delta_aoa_vec = np.zeros((numtime,))
#     aoa_mean_vec = np.zeros((numtime,))
#     cl = np.zeros((numtime,))
#     cl1 = np.zeros((numtime,))
#     cl_st = np.zeros((numtime,))
#     cl_dyn1 = np.zeros((numtime,))
#     cm = np.zeros((numtime,))
#     cm1 = np.zeros((numtime,))
#     cm_st = np.zeros((numtime,))
#     cm_dyn1 = np.zeros((numtime,))
#     theta_vec[0] = 2.5*np.pi/180
#     thetad_vec[0] = 0
#     thetadd_vec[0] = 0
#     w_vec[0] = 0
#     wd_vec[0] = 0
#     wdd_vec[0] = 0
#     wddd_vec[0] = 0
#     aoa_vec[0] = (theta_vec[0]+wd_vec[0])*180/np.pi
#     aoad_vec[0] = vinf/chord*(thetad_vec[0]+wdd_vec[0])*180/np.pi
#     aoadd_vec[0] = vinf**2/chord**2*(thetadd_vec[0]+wddd_vec[0])*180/np.pi
#     delta_aoa_vec[0] = 0
#     aoa_mean_vec[0] = aoa_vec[0]
#     datalstm1 = np.zeros((numtime,5))
#     datalstm1[:,0] = norm(aoa_mean_vec,alpha0_train1)
#     datalstm1[:,1] = norm(delta_aoa_vec,deltaalpha_train1)
#     datalstm1[:,2] = norm(aoad_vec,alphap_train1)
#     datalstm1[:,3] = norm(aoadd_vec,alphapp_train1)
#     data_lstm1 = np.reshape(datalstm1,(numtime,1,4))
#     flag_in = 0

#     freq = (ktheta2d_adim/I2d_adim)**(0.5)/(2*np.pi)
#     freqdim = (ktheta2d_adim*0.5*rhoinf*vinf**2*chord**2/(I2d_adim*rhoinf*chord**4))**(0.5)/(2*np.pi)
#     period = 1/freq
#     nmean = 2*int(period/deltatime)

#     ii_ini = nmean + 100

#     for ii in np.arange(numtime-1):
#         if flag_in == 0:
#             if np.abs(aoa_vec[ii]) < 60:
#                 cl_st[ii] = f_cl_st_w(w_vec[ii],aoa_vec[ii])
#                 cm_st[ii] = -f_cm_st_w(w_vec[ii],aoa_vec[ii])
#                 # data_pred1 = model1b.predict(data_lstm1)
#                 # cl_dyn1b = data_pred1[:,0]
#                 # cm_dyn1b = data_pred1[:,1]
#                 # cl_dyn1[ii] = desnorm(cl_dyn1b[ii],deltacl_train1)
#                 cl1[ii] = float(cl_st[ii])
#                 # cm_dyn1[ii] = -desnorm(cm_dyn1b[ii],deltacm_train1)
#                 cm1[ii] = float(cm_st[ii])
                
                
#                 cl = cl1
#                 cm = cm1
#                 print(jj,ii/(numtime-1),aoa_mean_vec[ii],aoa_vec[ii],aoad_vec[ii],aoadd_vec[ii])
#                 # print(ii/(numtime-1),cl1[ii],cl2[ii],cl3[ii],cl[ii])
                
#                 if ii>ii_ini:
#                     Amat_theta = np.array([[0,1],[-ktheta2d_adim/I2d_adim,0]])
#                     bmat_theta = np.array([[0],[cm[ii]/I2d_adim]])
#                     xvec_theta = np.array([[theta_vec[ii]-theta_vec[0]],[thetad_vec[ii]]])
#                     k1_theta = deltatime*eqsys(xvec_theta,Amat_theta,bmat_theta)
#                     k2_theta = deltatime*eqsys(xvec_theta+0.5*k1_theta,Amat_theta,bmat_theta)
#                     k3_theta = deltatime*eqsys(xvec_theta+0.5*k2_theta,Amat_theta,bmat_theta)
#                     k4_theta = deltatime*eqsys(xvec_theta+k3_theta,Amat_theta,bmat_theta)
#                     xvec_theta = xvec_theta+1/6*(k1_theta+2*k2_theta+2*k3_theta+k4_theta)
#                     theta_vec[ii+1] = theta_vec[0]+xvec_theta[0][0]
#                     thetad_vec[ii+1] = xvec_theta[1][0]
#                     thetadd_vec[ii+1] = (cm[ii]-ktheta2d_adim*(theta_vec[ii+1]-theta_vec[0]))/I2d_adim
#                     wdd2 =wdd_vec[ii]
#                     Amat_w = np.array([[0,1],[-kw2d_adim/m2d_adim,0]])
#                     bmat_w = np.array([[0],[cl[ii]/m2d_adim]])
#                     xvec_w = np.array([[w_vec[ii]-w_vec[0]],[wd_vec[ii]]])
#                     k1_w = deltatime*eqsys(xvec_w,Amat_w,bmat_w)
#                     k2_w = deltatime*eqsys(xvec_w+0.5*k1_w,Amat_w,bmat_w)
#                     k3_w = deltatime*eqsys(xvec_w+0.5*k2_w,Amat_w,bmat_w)
#                     k4_w = deltatime*eqsys(xvec_w+k3_w,Amat_w,bmat_w)
#                     xvec_w = xvec_w+1/6*(k1_w+2*k2_w+2*k3_w+k4_w)
#                     w_vec[ii+1] = w_vec[0]+xvec_w[0][0]
#                     wd_vec[ii+1] = xvec_w[1][0]
#                     wdd_vec[ii+1] =  (cl[ii]-kw2d_adim*(w_vec[ii+1]-w_vec[0]))/m2d_adim
#                     wddd_vec[ii+1] = (wdd_vec[ii+1] - wdd2)/deltatime
#                 else:
#                     thetadd_vec[ii+1] = thetadd_vec[ii]
#                     wdd2 =wdd_vec[ii]
#                     wdd_vec[ii+1] =  wdd_vec[ii]
#                     wddd_vec[ii+1] =  wddd_vec[ii]
#                     thetad_vec[ii+1] = thetad_vec[ii]
#                     wd_vec[ii+1] = wd_vec[ii]
#                     theta_vec[ii+1] = theta_vec[ii]
#                     w_vec[ii+1] = w_vec[ii]
        
#                 aoa_vec[ii+1] = (theta_vec[ii+1]-wd_vec[ii+1])*180/np.pi
#                 aoad_vec[ii+1] = vinf/chord*(thetad_vec[ii+1]-wdd_vec[ii+1])*180/np.pi
#                 aoadd_vec[ii+1] = vinf**2/chord**2*(thetadd_vec[ii+1]-wddd_vec[ii+1])*180/np.pi
#                 if ii>ii_ini:
#                     aoa_mean_vec[ii+1] = np.mean(aoa_vec[ii-nmean:ii])
#                 else: 
#                     aoa_mean_vec[ii+1] = aoa_vec[0]
#                 delta_aoa_vec[ii+1] = aoa_vec[ii+1]-aoa_mean_vec[ii+1]
#                 datalstm1[:,0] = norm(aoa_mean_vec,alpha0_train1)
#                 datalstm1[:,1] = norm(delta_aoa_vec,deltaalpha_train1)
#                 datalstm1[:,2] = norm(aoad_vec,alphap_train1)
#                 datalstm1[:,3] = norm(aoadd_vec,alphapp_train1)
#                 data_lstm1 = np.reshape(datalstm1,(numtime,1,4))
#             else: 
#                 flag_in = 1
#     # plt.figure(3*jj)
#     # plt.plot(cm)
#     # plt.figure(3*jj+1)
#     # plt.plot(cm)
#     # plt.figure(0)
#     # plt.plot(cm)
#     # plt.figure(3*jj+2)
#     # plt.plot( norm(aoa_vec[0]*np.ones((numtime,)),alpha0_train))
#     # plt.plot( norm(delta_aoa_vec,deltaalpha_train))
#     # plt.plot( norm(aoad_vec,alphap_train))
#     # plt.plot( norm(aoadd_vec,alphapp_train))
#     data_vec_krefuni["theta"] = theta_vec
#     data_vec_krefuni["thetad"] = thetad_vec
#     data_vec_krefuni["thetadd"] = thetadd_vec
#     data_vec_krefuni["w"] = w_vec
#     data_vec_krefuni["wd"] = wd_vec
#     data_vec_krefuni["wdd"] = wdd_vec
#     data_vec_krefuni["wddd"] = wddd_vec
#     data_vec_krefuni["aoa"] = aoa_vec
#     data_vec_krefuni["aoad"] = aoad_vec
#     data_vec_krefuni["aoadd"] = aoadd_vec
#     data_vec_krefuni["cl"] = cl
#     data_vec_krefuni["cm"] = cm
#     intpart = str(np.floor(ktheta2d_adim))
#     decpart = str(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000))
#     name_kref = "data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4) +".csv"
#     data_vec_krefuni.to_csv(name_kref)
#     data_kref_st.append(data_vec_krefuni)
        
#     torsionmean_st[jj] = np.mean(theta_vec[-2000:])
#     bendingmean_st[jj] = np.mean(w_vec[-2000:])
#     torsionmax_st[jj] = np.max(theta_vec[-2000:])
#     bendingmax_st[jj] = np.max(w_vec[-2000:])
#     torsionmin_st[jj] = np.min(theta_vec[-2000:])
#     bendingmin_st[jj] = np.min(w_vec[-2000:])
    


#%% Structural Model
kadim_vec = np.array([3,4,5,6,7,12,21,80])#[5,7,20,80]np.array([470,700,1000,1300,1600,1920,2000,3300,13480])*1e6
vinf_vec = np.zeros(len(kadim_vec),)
torsionmean = np.zeros(len(kadim_vec),) #np.zeros(7,)
bendingmean = np.zeros(len(kadim_vec),) 
torsionmax = np.zeros(len(kadim_vec),) #np.zeros(7,)
bendingmax = np.zeros(len(kadim_vec),)
kref_vec = np.zeros(len(kadim_vec),) # np.zeros(7,)
torsionmin = np.zeros(len(kadim_vec),) #np.zeros(7,)
bendingmin = np.zeros(len(kadim_vec),) 
data_kref = []
for jj in np.arange(len(kadim_vec)):
    data_vec_krefuni = pd.DataFrame()
    rhoinf =  1.18415
    rhos = 1180 
    thickness = 4e-3
    young_E = 3300*1e6
    inertia_area = 1/12*chord*thickness**3
    length = 0.37
    poisson = 0.35
    young_G = young_E/(2*(1+poisson))
    inertia_torsion = 1/3*chord*thickness**3
    M11theta = 0.455 
    A11theta = 0.455 
    K11theta = 1.2005
    M11w = 0.247 
    A11thetaw = 0.325 
    K11w = 3.189

    I2d = 1/12*rhos*chord**3*thickness*M11theta/A11theta
    ktheta2d = young_G*inertia_torsion/(length**2)*K11theta/A11theta
    m2d = rhos*chord*thickness*M11w/A11thetaw
    kw2d = young_E*inertia_area/(length**4)*K11w/A11thetaw

    
    ktheta2d_adim = kadim_vec[jj]
    vinf = (ktheta2d/(0.5*rhoinf*ktheta2d_adim*chord**2))**0.5
    I2d_adim =  I2d/(0.5*rhoinf*chord**4)
    m2d_adim = m2d/(0.5*rhoinf*chord**2)
    kw2d_adim = kw2d/(0.5*rhoinf*vinf**2)
    
    kref_vec[jj] = ktheta2d_adim
    vinf_vec[jj] = vinf
    
    theta_vec = np.zeros((numtime,))
    w_vec = np.zeros((numtime,))
    aoa_vec = np.zeros((numtime,))
    thetad_vec = np.zeros((numtime,))
    wd_vec = np.zeros((numtime,))
    aoad_vec = np.zeros((numtime,))
    thetadd_vec = np.zeros((numtime,))
    wdd_vec = np.zeros((numtime,))
    aoadd_vec = np.zeros((numtime,))
    wddd_vec = np.zeros((numtime,))
    delta_aoa_vec = np.zeros((numtime,))
    aoa_mean_vec = np.zeros((numtime,))
    cl = np.zeros((numtime,))
    cl1 = np.zeros((numtime,))
    cl_st = np.zeros((numtime,))
    cl_dyn1 = np.zeros((numtime,))
    cm = np.zeros((numtime,))
    cm1 = np.zeros((numtime,))
    cm_st = np.zeros((numtime,))
    cm_dyn1 = np.zeros((numtime,))
    theta_vec[0] = 2.5*np.pi/180
    thetad_vec[0] = 0
    thetadd_vec[0] = 0
    w_vec[0] = 0
    wd_vec[0] = 0
    wdd_vec[0] = 0
    wddd_vec[0] = 0
    aoa_vec[0] = (theta_vec[0]+wd_vec[0])*180/np.pi
    aoad_vec[0] = vinf/chord*(thetad_vec[0]+wdd_vec[0])*180/np.pi
    aoadd_vec[0] = vinf**2/chord**2*(thetadd_vec[0]+wddd_vec[0])*180/np.pi
    delta_aoa_vec[0] = 0
    aoa_mean_vec[0] = aoa_vec[0]
    datalstm1 = np.zeros((numtime,5))
    datalstm1[:,0] = norm(aoa_mean_vec,alpha0_train1)
    datalstm1[:,1] = norm(delta_aoa_vec,deltaalpha_train1)
    datalstm1[:,2] = norm(aoad_vec,alphap_train1)
    datalstm1[:,3] = norm(aoadd_vec,alphapp_train1)
    data_lstm1 = np.reshape(datalstm1,(numtime,1,4))
    flag_in = 0

    freq = (ktheta2d_adim/I2d_adim)**(0.5)/(2*np.pi)
    freqdim = (ktheta2d_adim*0.5*rhoinf*vinf**2*chord**2/(I2d_adim*rhoinf*chord**4))**(0.5)/(2*np.pi)
    period = 1/freq
    nmean = 2*int(period/deltatime)

    ii_ini = nmean 

    for ii in np.arange(numtime-1):
        if flag_in == 0:
            if np.abs(aoa_vec[ii]) < 60:
                cl_st[ii] = f_cl_st_w(w_vec[ii],aoa_vec[ii])
                cm_st[ii] = -f_cm_st_w(w_vec[ii],aoa_vec[ii])
                data_pred1 = model1b.predict(data_lstm1)
                cl_dyn1b = data_pred1[:,0]
                cm_dyn1b = data_pred1[:,1]
                cl_dyn1[ii] = desnorm(cl_dyn1b[ii],deltacl_train1)
                cl1[ii] = float(cl_st[ii]+cl_dyn1[ii])
                cm_dyn1[ii] = -desnorm(cm_dyn1b[ii],deltacm_train1)
                cm1[ii] = float(cm_st[ii]+cm_dyn1[ii])
                
                
                cl = cl1
                cm = cm1
                print(jj,ii/(numtime-1),aoa_mean_vec[ii],aoa_vec[ii],aoad_vec[ii],aoadd_vec[ii])
                # print(ii/(numtime-1),cl1[ii],cl2[ii],cl3[ii],cl[ii])
                
                if ii>ii_ini:
                    Amat_theta = np.array([[0,1],[-ktheta2d_adim/I2d_adim,0]])
                    bmat_theta = np.array([[0],[cm[ii]/I2d_adim]])
                    xvec_theta = np.array([[theta_vec[ii]-theta_vec[0]],[thetad_vec[ii]]])
                    k1_theta = deltatime*eqsys(xvec_theta,Amat_theta,bmat_theta)
                    k2_theta = deltatime*eqsys(xvec_theta+0.5*k1_theta,Amat_theta,bmat_theta)
                    k3_theta = deltatime*eqsys(xvec_theta+0.5*k2_theta,Amat_theta,bmat_theta)
                    k4_theta = deltatime*eqsys(xvec_theta+k3_theta,Amat_theta,bmat_theta)
                    xvec_theta = xvec_theta+1/6*(k1_theta+2*k2_theta+2*k3_theta+k4_theta)
                    theta_vec[ii+1] = theta_vec[0]+xvec_theta[0][0]
                    thetad_vec[ii+1] = xvec_theta[1][0]
                    thetadd_vec[ii+1] = (cm[ii]-ktheta2d_adim*(theta_vec[ii+1]-theta_vec[0]))/I2d_adim
                    wdd2 =wdd_vec[ii]
                    Amat_w = np.array([[0,1],[-kw2d_adim/m2d_adim,0]])
                    bmat_w = np.array([[0],[cl[ii]/m2d_adim]])
                    xvec_w = np.array([[w_vec[ii]-w_vec[0]],[wd_vec[ii]]])
                    k1_w = deltatime*eqsys(xvec_w,Amat_w,bmat_w)
                    k2_w = deltatime*eqsys(xvec_w+0.5*k1_w,Amat_w,bmat_w)
                    k3_w = deltatime*eqsys(xvec_w+0.5*k2_w,Amat_w,bmat_w)
                    k4_w = deltatime*eqsys(xvec_w+k3_w,Amat_w,bmat_w)
                    xvec_w = xvec_w+1/6*(k1_w+2*k2_w+2*k3_w+k4_w)
                    w_vec[ii+1] = w_vec[0]+xvec_w[0][0]
                    wd_vec[ii+1] = xvec_w[1][0]
                    wdd_vec[ii+1] =  (cl[ii]-kw2d_adim*(w_vec[ii+1]-w_vec[0]))/m2d_adim
                    wddd_vec[ii+1] = (wdd_vec[ii+1] - wdd2)/deltatime
                else:
                    thetadd_vec[ii+1] = thetadd_vec[ii]
                    wdd2 =wdd_vec[ii]
                    wdd_vec[ii+1] =  wdd_vec[ii]
                    wddd_vec[ii+1] =  wddd_vec[ii]
                    thetad_vec[ii+1] = thetad_vec[ii]
                    wd_vec[ii+1] = wd_vec[ii]
                    theta_vec[ii+1] = theta_vec[ii]
                    w_vec[ii+1] = w_vec[ii]
        
                aoa_vec[ii+1] = (theta_vec[ii+1]-wd_vec[ii+1])*180/np.pi
                aoad_vec[ii+1] = vinf/chord*(thetad_vec[ii+1]-wdd_vec[ii+1])*180/np.pi
                aoadd_vec[ii+1] = vinf**2/chord**2*(thetadd_vec[ii+1]-wddd_vec[ii+1])*180/np.pi
                if ii>ii_ini:
                    aoa_mean_vec[ii+1] = np.mean(aoa_vec[ii-nmean:ii])
                else: 
                    aoa_mean_vec[ii+1] = aoa_vec[0]
                delta_aoa_vec[ii+1] = aoa_vec[ii+1]-aoa_mean_vec[ii+1]
                datalstm1[:,0] = norm(aoa_mean_vec,alpha0_train1)
                datalstm1[:,1] = norm(delta_aoa_vec,deltaalpha_train1)
                datalstm1[:,2] = norm(aoad_vec,alphap_train1)
                datalstm1[:,3] = norm(aoadd_vec,alphapp_train1)
                data_lstm1 = np.reshape(datalstm1,(numtime,1,4))
            else: 
                flag_in = 1
    # plt.figure(3*jj)
    # plt.plot(cm)
    # plt.figure(3*jj+1)
    # plt.plot(cm)
    # plt.figure(0)
    # plt.plot(cm)
    # plt.figure(3*jj+2)
    # plt.plot( norm(aoa_vec[0]*np.ones((numtime,)),alpha0_train))
    # plt.plot( norm(delta_aoa_vec,deltaalpha_train))
    # plt.plot( norm(aoad_vec,alphap_train))
    # plt.plot( norm(aoadd_vec,alphapp_train))
    data_vec_krefuni["theta"] = theta_vec
    data_vec_krefuni["thetad"] = thetad_vec
    data_vec_krefuni["thetadd"] = thetadd_vec
    data_vec_krefuni["w"] = w_vec
    data_vec_krefuni["wd"] = wd_vec
    data_vec_krefuni["wdd"] = wdd_vec
    data_vec_krefuni["wddd"] = wddd_vec
    data_vec_krefuni["aoa"] = aoa_vec
    data_vec_krefuni["aoad"] = aoad_vec
    data_vec_krefuni["aoadd"] = aoadd_vec
    data_vec_krefuni["cl"] = cl
    data_vec_krefuni["cm"] = cm
    intpart = str(np.floor(ktheta2d_adim))
    decpart = str(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000))
    name_kref = "data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4) +".csv"
    data_vec_krefuni.to_csv(name_kref)
    data_kref.append(data_vec_krefuni)
        
    torsionmean[jj] = np.mean(theta_vec[-2000:])
    bendingmean[jj] = np.mean(w_vec[-2000:])
    torsionmax[jj] = np.max(theta_vec[-2000:])
    bendingmax[jj] = np.max(w_vec[-2000:])
    torsionmin[jj] = np.min(theta_vec[-2000:])
    bendingmin[jj] = np.min(w_vec[-2000:])
    
# %%
ii = 0
dataset_path_base = 'data/comparar/';
kadimcfd = [3,5.25,6.44,7,8,9,12.35,12.88,21.25,86.78];
mean_theta_cfd = np.zeros(len(kadimcfd),)
max_theta_cfd = np.zeros(len(kadimcfd),)
min_theta_cfd = np.zeros(len(kadimcfd),)
datacfd_kref = []
for f in [f_ for f_ in os.listdir(dataset_path_base) if 'displacement' in f_]:
    print(f)
    datacfd_vec_krefuni = pd.DataFrame()
    data_cfd = pd.read_csv(dataset_path_base + f, sep=",")
    try:
        data_cfd.index += data_cfd.index.values[-1]
    except IndexError:
        pass    
    timecfd = data_cfd["Physical Time: Physical Time (s)"]
    thetacfd = data_cfd["6-DOF Body Rotation Angle 1 Monitor: Rigid Body 1-DOF Angle (deg)"]
    wcfd = data_cfd["6-DOF Body Translation 1 Monitor: Rigid Body Translation (m)"]
    mean_theta_cfd[ii] = -np.mean(thetacfd.tail(2000).values)+2.5
    max_theta_cfd[ii] = -np.max(thetacfd.tail(2000).values)+2.5
    min_theta_cfd[ii] = -np.min(thetacfd.tail(2000).values)+2.5
    print(mean_theta_cfd[ii])
    datacfd_vec_krefuni["time"] = timecfd 
    datacfd_vec_krefuni["theta"] = -thetacfd+2.5
    datacfd_vec_krefuni["w"] = wcfd
    datacfd_kref.append(datacfd_vec_krefuni)
    ii = ii+1
    # plt.figure(26+ii)
    # plt.plot(timecfd,thetacfd)
    # plt.title('Torsion')
    # plt.xlabel('$k_{\Theta}^*$')
    # plt.ylabel('$\Theta_{mean}^*$')
    # plt.legend()
    # plt.grid()
#%%    


data_sum = pd.DataFrame()
data_sum["kref"] = kref_vec
data_sum["thetamean"] = torsionmean
data_sum["wmean"] = bendingmean
data_sum.to_csv("sum_aeroelastic.csv")      

plt.figure(3)
plt.plot(kadimcfd,mean_theta_cfd,label='CFD')
plt.plot(kref_vec,torsionmean*180/np.pi,label='LSTM+Aeroelastic')
plt.plot(kref_vec_st,torsionmean_st*180/np.pi,label='st+Aeroelastic')
plt.title('Mean torsion')
plt.xlabel('$k_{\Theta}^*$')
plt.ylabel('$\Theta_{mean}^*$')
plt.legend()
plt.ylim((2.5,3.5))
plt.grid()

plt.figure(3)
plt.fill_between(kadimcfd,min_theta_cfd,max_theta_cfd,alpha=0.5,label='CFD')
plt.fill_between(kref_vec,torsionmin*180/np.pi,torsionmax*180/np.pi,alpha=0.5,label='LSTM+Aeroelastic')
plt.fill_between(kref_vec_st,torsionmin_st*180/np.pi,torsionmax_st*180/np.pi,alpha=0.5,label='st+Aeroelastic')
plt.title('Mean torsion')
plt.xlabel('$k_{\Theta}^*$')
plt.ylabel('$\Theta_{mean}^*$')
plt.legend()
plt.ylim((2.5,3.5))
plt.grid()

#%%☻



with open('data/workspace/2020_04_21_22_11.pkl', 'wb') as f:
    pickle.dump([data_kref,datacfd_kref,kref_vec,torsionmean,kadimcfd,mean_theta_cfd], f)
    
    #%%
    
plta=plt.figure(0)
gs = plta.add_gridspec(1,3)
pltb = plta.add_subplot(gs[0,0:2])
pltb.plot(timevec*chord/vinf_vec_st[0]*(kref_vec_st[0]/I2d_adim)**0.5,data_kref[0]["theta"].values*180/np.pi,label="$k^*=3$")#*chord/vinf*(kref_vec[2]/I2d_adim)**0.5
pltb.plot(timevec*chord/vinf_vec_st[1]*(kref_vec_st[1]/I2d_adim)**0.5,data_kref_st[1]["theta"].values*180/np.pi,label="$k^*=4$")
pltb.plot(timevec*chord/vinf_vec_st[2]*(kref_vec_st[2]/I2d_adim)**0.5,data_kref_st[2]["theta"].values*180/np.pi,label="$k^*=5$")
pltb.plot(timevec*chord/vinf_vec_st[3]*(kref_vec_st[3]/I2d_adim)**0.5,data_kref_st[3]["theta"].values*180/np.pi,label="$k^*=6$")
pltb.plot(timevec*chord/vinf_vec_st[4]*(kref_vec_st[4]/I2d_adim)**0.5,data_kref_st[4]["theta"].values*180/np.pi,label="$k^*=7$")
pltb.plot(timevec*chord/vinf_vec_st[5]*(kref_vec_st[5]/I2d_adim)**0.5,data_kref_st[5]["theta"].values*180/np.pi,label="$k^*=12$")
pltb.plot(timevec*chord/vinf_vec_st[6]*(kref_vec_st[6]/I2d_adim)**0.5,data_kref_st[6]["theta"].values*180/np.pi,label="$k^*=21$")
pltb.plot(timevec*chord/vinf_vec_st[7]*(kref_vec_st[7]/I2d_adim)**0.5,data_kref_st[7]["theta"].values*180/np.pi,label="$k^*=80$")
# pltb.plot(timevec*chord/vinf*(kref_vec[8]/I2d_adim)**0.5,data_kref[8]["theta"].values*180/np.pi,label="$k^*=32.1$")
# pltb.plot(timevec*chord/vinf*(kref_vec[9]/I2d_adim)**0.5,data_kref[9]["theta"].values*180/np.pi,label="$k^*=86.7$")
pltb.set_title('Torsion st')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
pltb.set_ylim(-5,10)
pltb.grid()
pltb.legend(loc='upper right',bbox_to_anchor=(1.5, 1.0), borderaxespad=0.)
pltb.set_xlim(0,10)
plta=plt.figure(1)
gs = plta.add_gridspec(1,3)
pltb = plta.add_subplot(gs[0,0:2])
pltb.plot(datacfd_kref[0]["time"].values*(kadimcfd[0]/I2d_adim)**0.5,datacfd_kref[0]["theta"].values,label="$k^*=3$")
pltb.plot(datacfd_kref[1]["time"].values*(kadimcfd[1]/I2d_adim)**0.5,datacfd_kref[1]["theta"].values,label="$k^*=5.25$")
pltb.plot(datacfd_kref[2]["time"].values*(kadimcfd[2]/I2d_adim)**0.5,datacfd_kref[2]["theta"].values,label="$k^*=6.4$")
pltb.plot(datacfd_kref[3]["time"].values*(kadimcfd[3]/I2d_adim)**0.5,datacfd_kref[3]["theta"].values,label="$k^*=7$")
pltb.plot(datacfd_kref[4]["time"].values*(kadimcfd[4]/I2d_adim)**0.5,datacfd_kref[4]["theta"].values,label="$k^*=8$")
pltb.plot(datacfd_kref[5]["time"].values*(kadimcfd[5]/I2d_adim)**0.5,datacfd_kref[5]["theta"].values,label="$k^*=9$")
pltb.plot(datacfd_kref[6]["time"].values*(kadimcfd[6]/I2d_adim)**0.5,datacfd_kref[6]["theta"].values,label="$k^*=12.4$")
pltb.plot(datacfd_kref[7]["time"].values*(kadimcfd[7]/I2d_adim)**0.5,datacfd_kref[7]["theta"].values,label="$k^*=12.9$")
pltb.plot(datacfd_kref[8]["time"].values*(kadimcfd[8]/I2d_adim)**0.5,datacfd_kref[8]["theta"].values,label="$k^*=21.3$")
pltb.plot(datacfd_kref[9]["time"].values*(kadimcfd[9]/I2d_adim)**0.5,datacfd_kref[9]["theta"].values,label="$k^*=86.8$")
pltb.set_title('Torsion CFD')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
pltb.grid()
pltb.legend(loc='upper right',bbox_to_anchor=(1.5, 1.0), borderaxespad=0.)
pltb.set_xlim(0,10)
pltb.set_ylim(-5,10)

plta=plt.figure(2)
gs = plta.add_gridspec(1,3)
pltb = plta.add_subplot(gs[0,0:2])
pltb.plot(timevec*chord/vinf_vec[0]*(kref_vec[0]/I2d_adim)**0.5,data_kref[0]["theta"].values*180/np.pi,label="$k^*=3$")#*chord/vinf*(kref_vec[2]/I2d_adim)**0.5
pltb.plot(timevec*chord/vinf_vec[1]*(kref_vec[1]/I2d_adim)**0.5,data_kref[1]["theta"].values*180/np.pi,label="$k^*=4$")
pltb.plot(timevec*chord/vinf_vec[2]*(kref_vec[2]/I2d_adim)**0.5,data_kref[2]["theta"].values*180/np.pi,label="$k^*=5$")
pltb.plot(timevec*chord/vinf_vec[3]*(kref_vec[3]/I2d_adim)**0.5,data_kref[3]["theta"].values*180/np.pi,label="$k^*=6$")
pltb.plot(timevec*chord/vinf_vec[4]*(kref_vec[4]/I2d_adim)**0.5,data_kref[4]["theta"].values*180/np.pi,label="$k^*=7$")
pltb.plot(timevec*chord/vinf_vec[5]*(kref_vec[5]/I2d_adim)**0.5,data_kref[5]["theta"].values*180/np.pi,label="$k^*=12$")
pltb.plot(timevec*chord/vinf_vec[6]*(kref_vec[6]/I2d_adim)**0.5,data_kref[6]["theta"].values*180/np.pi,label="$k^*=21$")
pltb.plot(timevec*chord/vinf_vec[7]*(kref_vec[7]/I2d_adim)**0.5,data_kref[7]["theta"].values*180/np.pi,label="$k^*=80$")
# pltb.plot(timevec*chord/vinf*(kref_vec[8]/I2d_adim)**0.5,data_kref[8]["theta"].values*180/np.pi,label="$k^*=32.1$")
# pltb.plot(timevec*chord/vinf*(kref_vec[9]/I2d_adim)**0.5,data_kref[9]["theta"].values*180/np.pi,label="$k^*=86.7$")
pltb.set_title('Torsion LSTM')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
pltb.set_ylim(-5,10)
pltb.grid()
pltb.legend(loc='upper right',bbox_to_anchor=(1.5, 1.0), borderaxespad=0.)
pltb.set_xlim(0,10)
plta=plt.figure(1)
gs = plta.add_gridspec(1,3)
pltb = plta.add_subplot(gs[0,0:2])
pltb.plot(datacfd_kref[0]["time"].values*(kadimcfd[0]/I2d_adim)**0.5,datacfd_kref[0]["theta"].values,label="$k^*=3$")
pltb.plot(datacfd_kref[1]["time"].values*(kadimcfd[1]/I2d_adim)**0.5,datacfd_kref[1]["theta"].values,label="$k^*=5.25$")
pltb.plot(datacfd_kref[2]["time"].values*(kadimcfd[2]/I2d_adim)**0.5,datacfd_kref[2]["theta"].values,label="$k^*=6.4$")
pltb.plot(datacfd_kref[3]["time"].values*(kadimcfd[3]/I2d_adim)**0.5,datacfd_kref[3]["theta"].values,label="$k^*=7$")
pltb.plot(datacfd_kref[4]["time"].values*(kadimcfd[4]/I2d_adim)**0.5,datacfd_kref[4]["theta"].values,label="$k^*=8$")
pltb.plot(datacfd_kref[5]["time"].values*(kadimcfd[5]/I2d_adim)**0.5,datacfd_kref[5]["theta"].values,label="$k^*=9$")
pltb.plot(datacfd_kref[6]["time"].values*(kadimcfd[6]/I2d_adim)**0.5,datacfd_kref[6]["theta"].values,label="$k^*=12.4$")
pltb.plot(datacfd_kref[7]["time"].values*(kadimcfd[7]/I2d_adim)**0.5,datacfd_kref[7]["theta"].values,label="$k^*=12.9$")
pltb.plot(datacfd_kref[8]["time"].values*(kadimcfd[8]/I2d_adim)**0.5,datacfd_kref[8]["theta"].values,label="$k^*=21.3$")
pltb.plot(datacfd_kref[9]["time"].values*(kadimcfd[9]/I2d_adim)**0.5,datacfd_kref[9]["theta"].values,label="$k^*=86.8$")
pltb.set_title('Torsion CFD')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$\Theta^*$')
pltb.grid()
pltb.legend(loc='upper right',bbox_to_anchor=(1.5, 1.0), borderaxespad=0.)
pltb.set_xlim(0,10)
pltb.set_ylim(-5,10)
#%%
    
# plt.figure(0)
# #plt.plot(cm)
# plt.plot( norm(aoa_vec[0]*np.ones((numtime,)),alpha0_train))
# plt.plot( norm(delta_aoa_vec,deltaalpha_train))
# plt.plot( norm(aoad_vec,alphap_train))
# plt.plot( norm(aoadd_vec,alphapp_train))
# plt.figure(1)
# plt.plot(cm)
#%%
# plt.figure(21)

# plt.title('Static CM')
# plt.xlabel('$\Theta$ (deg)')
# plt.ylabel('CM')
# plt.grid()
# alpha2 = np.linspace(-60,60,200)
# cm_st2 = f_cm_st(alpha2)
# plt.plot(alpha2,cm_st2)