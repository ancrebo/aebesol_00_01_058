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
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.signal import savgol_filter


import pickle

print(tf.__version__) 



#vinf = 20
chord = 0.1
#%% Static polar
#%% Static polar
pathpolarbase = 'data/polares_w/' 
polar = pd.DataFrame()
pointgridcl = np.zeros((55,6))
pointgridcm = np.zeros((55,6))
kk = 0
for f in [f_ for f_ in os.listdir(pathpolarbase) if 'polar' in f_]:
    polaruni = pd.read_csv(pathpolarbase+f,sep=';')
    polar = pd.concat([polar, polaruni])
    pointgridcl[:,kk] = polaruni["cl"].values
    pointgridcm[:,kk] = polaruni["cm"].values
    kk = kk +1
f_cl_st_w = RegularGridInterpolator((polaruni["aoa"].values,[0,0.01,0.02,0.03,0.04,0.05]),pointgridcl)
f_cm_st_w = RegularGridInterpolator((polaruni["aoa"].values,[0,0.01,0.02,0.03,0.04,0.05]),pointgridcm)
# %%
# Se lee la polar estacionaria del perfil

pathpolarbase = 'data/polares_w/' 
polar = pd.DataFrame()
pointgridcl = np.zeros((55,6))
pointgridcm = np.zeros((55,6))
kk = 0
for f in [f_ for f_ in os.listdir(pathpolarbase) if 'polar' in f_]:
    polaruni = pd.read_csv(pathpolarbase+f,sep=';')
    polar = pd.concat([polar, polaruni])
    pointgridcl[:,kk] = polaruni["cl"].values
    pointgridcm[:,kk] = polaruni["cm"].values
    kk = kk +1
f_cl_st_w = RegularGridInterpolator((polaruni["aoa"].values,[0,0.01,0.02,0.03,0.04,0.05]),pointgridcl)
f_cm_st_w = RegularGridInterpolator((polaruni["aoa"].values,[0,0.01,0.02,0.03,0.04,0.05]),pointgridcm)
ii = 0



# %%
# Se leen todos los datos temporales

data_tot = pd.DataFrame()
data_tot1 = pd.DataFrame()
data_tot2 = pd.DataFrame()
data_tot3 = pd.DataFrame()

polar1d = pd.read_csv('data/polar/polar.csv')
dataset_path_base = '../cycle_bbdd3/'
for f in [f_ for f_ in os.listdir(dataset_path_base) if 'a0' in f_]:
    print(f)
    data = pd.read_csv(dataset_path_base + f, sep=",")
    try:
        data.index += data_tot.index.values[-1]
    except IndexError:
        pass
    # -----------------------------------------------------------------
    alpha = (-data["rotation Monitor: Expression (deg)"].values)
    f_cl_st = interp1d(polar1d["aoa"].values,polar1d["cl"].values)
    cl_st = f_cl_st(alpha)
    data["cl_st"] = cl_st
    f_cm_st = interp1d(polar1d["aoa"].values,polar1d["cm "].values)
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
    datautil["Delta Alpha"] = (-dataperiod["rotation Monitor: Expression (deg)"].values-dataperiod["Alpha0 Monitor: Expression"].values * 180 / np.pi)#-dataperiod["rotation Monitor: Expression (deg)"].values#( -dataperiod["rotation Monitor: Expression (deg)"].values-dataperiod["Alpha0 Monitor: Expression"].values * 180 / np.pi)
    datautil["Alpha prima"] = dataperiod["Alpha prima"].values
    datautil["Alpha prima prima"] = dataperiod["Alpha prima prima"].values
    datautil["Flag"] = np.concatenate(([1E-0],np.zeros(int(point_period)-1)))
    
    if  data["Amplitude Monitor: Expression"].values[-1]*180/np.pi<5.5 and  data["Alpha0 Monitor: Expression"].values[-1]*180/np.pi<7 and freq>20 and freq<70:
    
        if np.mod(ii,10)==0:
            data_tot = pd.concat([data_tot, datautil],axis=0)
            data_tot2 = pd.concat([data_tot, datautil],axis=0)
        else:
            data_tot1 = pd.concat([data_tot1, datautil],axis=0)
            data_tot3 = pd.concat([data_tot1, datautil],axis=0)
        ii=ii+1 
data_tot1 = data_tot1.dropna()
data_tot2 = data_tot2.dropna()
data_tot3 = data_tot3.dropna()
data_tot = data_tot.dropna()
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

deltacl_train2 = data_tot2["Delta CL"].values
deltacm_train2 = data_tot2["Delta CM"].values
alpha0_train2 = data_tot2["Alpha0"].values
deltaalpha_train2 = data_tot2["Delta Alpha"].values
alphap_train2 = data_tot2["Alpha prima"].values
alphapp_train2 = data_tot2["Alpha prima prima"].values

deltacl_train3 = data_tot3["Delta CL"].values
deltacm_train3 = data_tot3["Delta CM"].values
alpha0_train3 = data_tot3["Alpha0"].values
deltaalpha_train3 = data_tot3["Delta Alpha"].values
alphap_train3 = data_tot3["Alpha prima"].values
alphapp_train3 = data_tot3["Alpha prima prima"].values
#%%
# Simulation
deltatime = 0.0001
totaltime = 5 
numtime = int(totaltime/deltatime)
timevec = np.linspace(0,totaltime,numtime)
#%% LOAD NEURAL NETWORK MODEL LSTM 20-1
unitslstm = 50
numepochlstm = 50000
namesave = 'LSTM_'+str(unitslstm)+'N_'+str(numepochlstm)+'epoch'
namesavered = 'data/redes/'+namesave
namesave1 = namesavered + 'model1'
model1 = keras.models.load_model(namesave1 + '.h5')
unitsfnn =50
numepochfnn = 5000
namesavefnn = 'FNN_'+str(unitsfnn)+'N'+str(numepochfnn)+'epoch'
namesavefnnred = 'data/redes/'+namesavefnn
namesave1fnn = namesavefnnred + 'model1'
model1fnn = keras.models.load_model(namesave1fnn + '.h5')
#%%
weights0 = model1.layers[0].get_weights()
weights1 = model1.layers[1].get_weights()
dimvar = 5
#dimtemp = 1
model1b = Sequential()
model1b.add(LSTM(unitslstm,  batch_input_shape=(numtime,1, dimvar)))
model1b.add(Dense(2))
model1b.layers[0].set_weights(weights0)
model1b.layers[1].set_weights(weights1)
optimizer = tf.keras.optimizers.RMSprop(0.001) 
model1b.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse']) 
#%%
weights0 = model1fnn.layers[0].get_weights()
weights1 = model1fnn.layers[1].get_weights()
dimvar = 4
#dimtemp = 1
model1fnnb = Sequential()
model1fnnb.add(Dense(unitsfnn, activation='sigmoid', input_shape=[dimvar]))
model1fnnb.add(Dense(2))
model1fnnb.layers[0].set_weights(weights0)
model1fnnb.layers[1].set_weights(weights1)
optimizer = tf.keras.optimizers.RMSprop(0.001) 
model1fnnb.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse']) 

#%% 
def eqsys(xvec,Amat,bmat):
    return np.matmul(Amat,xvec)+bmat

def func_dyn(Amat,bmat,xvec):
    return np.matmul(Amat,xvec)+bmat

def rk4_explicit(Amat,bmat,xvec,delta_t):
    rk1 = func_dyn(Amat,bmat,xvec)
    rk2 = func_dyn(Amat,bmat,xvec+0.5*rk1*delta_t)
    rk3 = func_dyn(Amat,bmat,xvec+0.5*rk2*delta_t)
    rk4 = func_dyn(Amat,bmat,xvec+rk3*delta_t)
    slope = 1/6*(rk1+2*rk2+2*rk3+rk4)
    xvec_1 = xvec+delta_t*slope
    return xvec_1,slope


def pred_corr(Amat,bmat,xvec,xini,delta_t):
    xvecpred = xvec[:]+delta_t/24*(55*func_dyn(Amat,bmat,xvec[:])-59*func_dyn(Amat,bmat,xini[:,[2]])+37*func_dyn(Amat,bmat,xini[:,[1]])-9*func_dyn(Amat,bmat,xini[:,[0]])) #xvec[:]+delta_t/12*(23*func_dyn(Amat,bmat,xvec[:])-16*func_dyn(Amat,bmat,xini[:,1])+5*func_dyn(Amat,bmat,xini[:,0]))
    func_vecpred = func_dyn(Amat,bmat,xvecpred)
    slope = 1/720*(251*func_vecpred+646*func_dyn(Amat,bmat,xvec[:])-264*func_dyn(Amat,bmat,xini[:,[2]])+106*func_dyn(Amat,bmat,xini[:,[1]])-19*func_dyn(Amat,bmat,xini[:,[0]]))#1/24*(9*func_vecpred+19*func_dyn(Amat,bmat,xvec[:])-5*func_dyn(Amat,bmat,xini[:,1])+1*func_dyn(Amat,bmat,xini[:,0]))
    xveccorr = xvec[:]+delta_t*slope
    return xveccorr,slope

#%% Structural Model
kadim_vec_st = np.array([3,5.5,6.5,8.5,12.5,20,50,90])#3,4,5,6,7,8,9,12,21,80[5,7,20,80]np.array([470,700,1000,1300,1600,1920,2000,3300,13480])*1e6
vinf_vec_st = np.zeros(len(kadim_vec_st),)
torsionmean_st = np.zeros(len(kadim_vec_st),) #np.zeros(7,)
bendingmean_st = np.zeros(len(kadim_vec_st),) 
torsionmax_st = np.zeros(len(kadim_vec_st),) #np.zeros(7,)
bendingmax_st = np.zeros(len(kadim_vec_st),)
kref_vec_st = np.zeros(len(kadim_vec_st),) # np.zeros(7,)
torsionmin_st = np.zeros(len(kadim_vec_st),) #np.zeros(7,)
bendingmin_st = np.zeros(len(kadim_vec_st),) 
data_kref_st = []
for jj in np.arange(len(kadim_vec_st)):
    data_vec_krefuni = pd.DataFrame()
    rhoinf =  1.18415
    rhos = 1180 
    thickness = 4e-3
    young_E = 2000*1e6
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

    
    ktheta2d_adim = kadim_vec_st[jj]
    vinf = (ktheta2d/(0.5*rhoinf*ktheta2d_adim*chord**2))**0.5
    # I2d_adim =  I2d/(0.5*rhoinf*chord**4)
    # m2d_adim = m2d/(0.5*rhoinf*chord**2)
    # kw2d_adim = kw2d/(0.5*rhoinf*vinf**2)
    
    kref_vec_st[jj] = ktheta2d_adim
    vinf_vec_st[jj] = vinf
    
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
    wdd_vec_filt = np.zeros((numtime,))
    flag_vec = np.zeros((numtime,))
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
    flag_vec[0] = 1
    aoa_vec[0] = (theta_vec[0]+wd_vec[0]/vinf)*180/np.pi
    aoad_vec[0] =(thetad_vec[0]+wdd_vec[0]/vinf)*180/np.pi
    aoadd_vec[0] = (thetadd_vec[0]+wddd_vec[0]/vinf)*180/np.pi
    delta_aoa_vec[0] = 0
    aoa_mean_vec[0] = aoa_vec[0]
    flag_in = 0

    freq =(ktheta2d/I2d)**(0.5)/(2*np.pi)
    period = 1/freq
    nmean = 2*int(period/deltatime)

    ii_ini = nmean 

    for ii in np.arange(numtime-1):
        if flag_in == 0:
            if np.abs(aoa_vec[ii]) < 60:
                if w_vec[ii] > 0.05:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],0.05]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],0.05]))
                elif w_vec[ii] <0:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],0]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],0]))
                else:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],w_vec[ii]]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],w_vec[ii]]))
                
                cl = cl_st
                cm = cm_st
                if ii%1000==0:
                    print('st',jj,ii/(numtime-1),aoa_mean_vec[ii],aoa_vec[ii],aoad_vec[ii],aoadd_vec[ii])
                # print(ii/(numtime-1),cl1[ii],cl2[ii],cl3[ii],cl[ii])
                
                
                
                if ii>ii_ini and ii<ii_ini+5:
                    Amat_theta = np.array([[0,1],[-ktheta2d/I2d,0]])
                    bmat_theta = np.array([[0],[0.5*rhoinf*chord**2*vinf**2*cm[ii]/I2d]])
                    xvec_theta = np.array([[theta_vec[ii]-theta_vec[0]],[thetad_vec[ii]]])
                    xvec_theta, xvecd_theta =  rk4_explicit(Amat_theta,bmat_theta,xvec_theta,deltatime)
                    # xvec_theta, xvecd_theta = pred_corr(Amat_theta,bmat_theta,xvec_theta,xini_theta,deltatime)
                    theta_vec[ii+1] = theta_vec[0]+xvec_theta[0][0]
                    thetad_vec[ii+1] = xvec_theta[1][0]
                    thetadd_vec[ii+1] = xvecd_theta[1][0]#(0.5*rhoinf*chord**2*vinf**2*cm[ii]-ktheta2d*(theta_vec[ii+1]-theta_vec[0]))/I2d
                    wdd2 =wdd_vec[ii]
                    Amat_w = np.array([[0,1],[-kw2d/m2d,0]])
                    bmat_w = np.array([[0],[0.5*rhoinf*chord*vinf**2*cl[ii]/m2d]])
                    xvec_w = np.array([[w_vec[ii]-w_vec[0]],[wd_vec[ii]]])
                    xvec_w, xvecd_w =  rk4_explicit(Amat_w,bmat_w,xvec_w,deltatime)
                    # xvec_w, xvecd_w = pred_corr(Amat_w,bmat_w,xvec_w,xini_w,deltatime)
                    w_vec[ii+1] = w_vec[0]+xvec_w[0][0]
                    wd_vec[ii+1] = xvec_w[1][0]
                    wdd_vec[ii+1] =  xvecd_w[1][0]#(0.5*rhoinf*chord*vinf**2*cl[ii]-kw2d*(w_vec[ii+1]-w_vec[0]))/m2d
                    wdd_vec_filt[ii+1] = savgol_filter(wdd_vec[:ii+2],ii_ini+1,3)[-1]
                    wddd_vec[ii+1] = (wdd_vec_filt[ii+1] - wdd_vec_filt[ii])/(deltatime)
                elif ii> ii_ini+4:
                    Amat_theta = np.array([[0,1],[-ktheta2d/I2d,0]])
                    bmat_theta = np.array([[0],[0.5*rhoinf*chord**2*vinf**2*cm[ii]/I2d]])
                    xvec_theta = np.array([[theta_vec[ii]-theta_vec[0]],[thetad_vec[ii]]])
                    xini_theta = np.array([theta_vec[ii-3:ii]-theta_vec[0],thetad_vec[ii-3:ii]])
                    # xvec_theta, xvecd_theta =  rk4_explicit(Amat_theta,bmat_theta,xvec_theta,deltatime)
                    xvec_theta, xvecd_theta = pred_corr(Amat_theta,bmat_theta,xvec_theta,xini_theta,deltatime)
                    theta_vec[ii+1] = theta_vec[0]+xvec_theta[0][0]
                    thetad_vec[ii+1] = xvec_theta[1][0]
                    thetadd_vec[ii+1] = xvecd_theta[1][0]#(0.5*rhoinf*chord**2*vinf**2*cm[ii]-ktheta2d*(theta_vec[ii+1]-theta_vec[0]))/I2d
                    wdd2 =wdd_vec[ii]
                    Amat_w = np.array([[0,1],[-kw2d/m2d,0]])
                    bmat_w = np.array([[0],[0.5*rhoinf*chord*vinf**2*cl[ii]/m2d]])
                    xvec_w = np.array([[w_vec[ii]-w_vec[0]],[wd_vec[ii]]])
                    xini_w = np.array([w_vec[ii-3:ii]-w_vec[0],wd_vec[ii-3:ii]])
                    # xvec_w, xvecd_w =  rk4_explicit(Amat_w,bmat_w,xvec_w,deltatime)
                    xvec_w, xvecd_w = pred_corr(Amat_w,bmat_w,xvec_w,xini_w,deltatime)
                    w_vec[ii+1] = w_vec[0]+xvec_w[0][0]
                    wd_vec[ii+1] = xvec_w[1][0]
                    wdd_vec[ii+1] =  xvecd_w[1][0]#(0.5*rhoinf*chord*vinf**2*cl[ii]-kw2d*(w_vec[ii+1]-w_vec[0]))/m2d
                    wdd_vec_filt[ii+1] = savgol_filter(wdd_vec[:ii+2],ii_ini+1,3)[-1]
                    wddd_vec[ii+1] = (wdd_vec_filt[ii+1] - wdd_vec_filt[ii])/(deltatime)
                else:
                    thetadd_vec[ii+1] = thetadd_vec[ii]
                    wdd2 =wdd_vec[ii]
                    wdd_vec[ii+1] =  wdd_vec[ii]
                    wddd_vec[ii+1] =  wddd_vec[ii]
                    thetad_vec[ii+1] = thetad_vec[ii]
                    wd_vec[ii+1] = wd_vec[ii]
                    theta_vec[ii+1] = theta_vec[ii]
                    w_vec[ii+1] = w_vec[ii]
        
                aoa_vec[ii+1] = (theta_vec[ii+1]-wd_vec[ii+1]/vinf)*180/np.pi
                aoad_vec[ii+1] =(thetad_vec[ii+1]-wdd_vec[ii+1]/vinf)*180/np.pi
                aoadd_vec[ii+1] = (thetadd_vec[ii+1]-wddd_vec[ii+1]/vinf)*180/np.pi
                flag_vec[ii+1] = 0
                if ii>10*ii_ini:
                    aoa_mean_vec[ii+1] = np.mean(aoa_vec[9*ii_ini:ii])
                elif ii>ii_ini:
                    aoa_mean_vec[ii+1] = np.mean(aoa_vec[ii-ii_ini:ii])
                else: 
                    aoa_mean_vec[ii+1] = aoa_vec[0]
                delta_aoa_vec[ii+1] = aoa_vec[ii+1]-aoa_mean_vec[ii+1]
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
    data_vec_krefuni["time"] = timevec
    intpart = str(np.floor(ktheta2d_adim))
    decpart = str(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000))
    name_kref = "data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4) +"mma_1ms_3s.csv"
    data_vec_krefuni.to_csv(name_kref)
    data_kref_st.append(data_vec_krefuni)
        
    torsionmean_st[jj] = np.mean(theta_vec[10*nmean:])
    bendingmean_st[jj] = np.mean(w_vec[10*nmean:])
    torsionmax_st[jj] = np.max(theta_vec[10*nmean:])
    bendingmax_st[jj] = np.max(w_vec[10*nmean:])
    torsionmin_st[jj] = np.min(theta_vec[10*nmean:])
    bendingmin_st[jj] = np.min(w_vec[10*nmean:])



#%% Structural Model
kadim_vec_fnn = np.array([3,5.5,6.5,8.5,12.5,20,50,90])#3,4,5,6,7,8,9,12,21,80[5,7,20,80]np.array([470,700,1000,1300,1600,1920,2000,3300,13480])*1e6
vinf_vec_fnn = np.zeros(len(kadim_vec_fnn),)
torsionmean_fnn = np.zeros(len(kadim_vec_fnn),) #np.zeros(7,)
bendingmean_fnn = np.zeros(len(kadim_vec_fnn),) 
torsionmax_fnn = np.zeros(len(kadim_vec_fnn),) #np.zeros(7,)
bendingmax_fnn = np.zeros(len(kadim_vec_fnn),)
kref_vec_fnn = np.zeros(len(kadim_vec_fnn),) # np.zeros(7,)
torsionmin_fnn = np.zeros(len(kadim_vec_fnn),) #np.zeros(7,)
bendingmin_fnn = np.zeros(len(kadim_vec_fnn),) 
data_kref_fnn = []
for jj in np.arange(len(kadim_vec_fnn)):
    data_vec_krefuni = pd.DataFrame()
    rhoinf =  1.18415
    rhos = 1180 
    thickness = 4e-3
    young_E = 2000*1e6
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

    
    ktheta2d_adim = kadim_vec_fnn[jj]
    vinf = (ktheta2d/(0.5*rhoinf*ktheta2d_adim*chord**2))**0.5
    # I2d_adim =  I2d/(0.5*rhoinf*chord**4)
    # m2d_adim = m2d/(0.5*rhoinf*chord**2)
    # kw2d_adim = kw2d/(0.5*rhoinf*vinf**2)
    
    kref_vec_fnn[jj] = ktheta2d_adim
    vinf_vec_fnn[jj] = vinf
    
    theta_vec = np.zeros((numtime,))
    w_vec = np.zeros((numtime,))
    aoa_vec = np.zeros((numtime,))
    thetad_vec = np.zeros((numtime,))
    wd_vec = np.zeros((numtime,))
    aoad_vec = np.zeros((numtime,))
    thetadd_vec = np.zeros((numtime,))
    wdd_vec = np.zeros((numtime,))
    wdd_vec_filt = np.zeros((numtime,))
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
    aoa_vec[0] = (theta_vec[0]+wd_vec[0]/vinf)*180/np.pi
    aoad_vec[0] =(thetad_vec[0]+wdd_vec[0]/vinf)*180/np.pi
    aoadd_vec[0] = (thetadd_vec[0]+wddd_vec[0]/vinf)*180/np.pi
    delta_aoa_vec[0] = 0
    aoa_mean_vec[0] = aoa_vec[0]
    datalstm1 = np.zeros((numtime,4))
    datalstm1[:,0] = norm(aoa_mean_vec,alpha0_train1)
    datalstm1[:,1] = norm(delta_aoa_vec,deltaalpha_train1)
    datalstm1[:,2] = norm(aoad_vec,alphap_train1)
    datalstm1[:,3] = norm(aoadd_vec,alphapp_train1)
    flag_in = 0

    freq = (ktheta2d/I2d)**(0.5)/(2*np.pi)
    period = 1/freq
    nmean = 2*int(period/deltatime)

    ii_ini = nmean 

    for ii in np.arange(numtime-1):
        if flag_in == 0:
            if np.abs(aoa_vec[ii]) < 60:
                if w_vec[ii] > 0.05:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],0.05]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],0.05]))
                elif w_vec[ii] <0:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],0]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],0]))
                else:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],w_vec[ii]]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],w_vec[ii]]))
                data_pred1 = model1fnnb.predict(datalstm1[ii:ii+1,:]).flatten()
                cl_dyn1b = data_pred1[0]
                cm_dyn1b = data_pred1[1]
                cl_dyn1[ii] = desnorm(cl_dyn1b,deltacl_train1)
                cl1[ii] = float(cl_st[ii]+cl_dyn1[ii])
                cm_dyn1[ii] = -desnorm(cm_dyn1b,deltacm_train1)
                cm1[ii] = float(cm_st[ii]+cm_dyn1[ii])
                
                
                cl = cl1
                cm = cm1
                if ii%1000==0:
                    print('fnn',jj,ii/(numtime-1),aoa_mean_vec[ii],aoa_vec[ii],aoad_vec[ii],aoadd_vec[ii])
                # print(ii/(numtime-1),cl1[ii],cl2[ii],cl3[ii],cl[ii])
                
                if ii>ii_ini and ii<ii_ini+5:
                    Amat_theta = np.array([[0,1],[-ktheta2d/I2d,0]])
                    bmat_theta = np.array([[0],[0.5*rhoinf*chord**2*vinf**2*cm[ii]/I2d]])
                    xvec_theta = np.array([[theta_vec[ii]-theta_vec[0]],[thetad_vec[ii]]])
                    xvec_theta, xvecd_theta =  rk4_explicit(Amat_theta,bmat_theta,xvec_theta,deltatime)
                    # xvec_theta, xvecd_theta = pred_corr(Amat_theta,bmat_theta,xvec_theta,xini_theta,deltatime)
                    theta_vec[ii+1] = theta_vec[0]+xvec_theta[0][0]
                    thetad_vec[ii+1] = xvec_theta[1][0]
                    thetadd_vec[ii+1] = xvecd_theta[1][0]#(0.5*rhoinf*chord**2*vinf**2*cm[ii]-ktheta2d*(theta_vec[ii+1]-theta_vec[0]))/I2d
                    wdd2 =wdd_vec[ii]
                    Amat_w = np.array([[0,1],[-kw2d/m2d,0]])
                    bmat_w = np.array([[0],[0.5*rhoinf*chord*vinf**2*cl[ii]/m2d]])
                    xvec_w = np.array([[w_vec[ii]-w_vec[0]],[wd_vec[ii]]])
                    xvec_w, xvecd_w =  rk4_explicit(Amat_w,bmat_w,xvec_w,deltatime)
                    # xvec_w, xvecd_w = pred_corr(Amat_w,bmat_w,xvec_w,xini_w,deltatime)
                    w_vec[ii+1] = w_vec[0]+xvec_w[0][0]
                    wd_vec[ii+1] = xvec_w[1][0]
                    wdd_vec[ii+1] =  xvecd_w[1][0]#(0.5*rhoinf*chord*vinf**2*cl[ii]-kw2d*(w_vec[ii+1]-w_vec[0]))/m2d
                    wdd_vec_filt[ii+1] = savgol_filter(wdd_vec[:ii+2],ii_ini+1,3)[-1]
                    wddd_vec[ii+1] = (wdd_vec_filt[ii+1] - wdd_vec_filt[ii])/(deltatime)
                elif ii> ii_ini+4:
                    Amat_theta = np.array([[0,1],[-ktheta2d/I2d,0]])
                    bmat_theta = np.array([[0],[0.5*rhoinf*chord**2*vinf**2*cm[ii]/I2d]])
                    xvec_theta = np.array([[theta_vec[ii]-theta_vec[0]],[thetad_vec[ii]]])
                    xini_theta = np.array([theta_vec[ii-3:ii]-theta_vec[0],thetad_vec[ii-3:ii]])
                    # xvec_theta, xvecd_theta =  rk4_explicit(Amat_theta,bmat_theta,xvec_theta,deltatime)
                    xvec_theta, xvecd_theta = pred_corr(Amat_theta,bmat_theta,xvec_theta,xini_theta,deltatime)
                    theta_vec[ii+1] = theta_vec[0]+xvec_theta[0][0]
                    thetad_vec[ii+1] = xvec_theta[1][0]
                    thetadd_vec[ii+1] = xvecd_theta[1][0]#(0.5*rhoinf*chord**2*vinf**2*cm[ii]-ktheta2d*(theta_vec[ii+1]-theta_vec[0]))/I2d
                    wdd2 =wdd_vec[ii]
                    Amat_w = np.array([[0,1],[-kw2d/m2d,0]])
                    bmat_w = np.array([[0],[0.5*rhoinf*chord*vinf**2*cl[ii]/m2d]])
                    xvec_w = np.array([[w_vec[ii]-w_vec[0]],[wd_vec[ii]]])
                    xini_w = np.array([w_vec[ii-3:ii]-w_vec[0],wd_vec[ii-3:ii]])
                    # xvec_w, xvecd_w =  rk4_explicit(Amat_w,bmat_w,xvec_w,deltatime)
                    xvec_w, xvecd_w = pred_corr(Amat_w,bmat_w,xvec_w,xini_w,deltatime)
                    w_vec[ii+1] = w_vec[0]+xvec_w[0][0]
                    wd_vec[ii+1] = xvec_w[1][0]
                    wdd_vec[ii+1] =  xvecd_w[1][0]#(0.5*rhoinf*chord*vinf**2*cl[ii]-kw2d*(w_vec[ii+1]-w_vec[0]))/m2d
                    wdd_vec_filt[ii+1] = savgol_filter(wdd_vec[:ii+2],ii_ini+1,3)[-1]
                    wddd_vec[ii+1] = (wdd_vec_filt[ii+1] - wdd_vec_filt[ii])/(deltatime)
                else:
                    thetadd_vec[ii+1] = thetadd_vec[ii]
                    wdd2 =wdd_vec[ii]
                    wdd_vec[ii+1] =  wdd_vec[ii]
                    wddd_vec[ii+1] =  wddd_vec[ii]
                    thetad_vec[ii+1] = thetad_vec[ii]
                    wd_vec[ii+1] = wd_vec[ii]
                    theta_vec[ii+1] = theta_vec[ii]
                    w_vec[ii+1] = w_vec[ii]
        
                aoa_vec[ii+1] = (theta_vec[ii+1]-wd_vec[ii+1]/vinf)*180/np.pi
                aoad_vec[ii+1] =(thetad_vec[ii+1]-wdd_vec[ii+1]/vinf)*180/np.pi
                aoadd_vec[ii+1] = (thetadd_vec[ii+1]-wddd_vec[ii+1]/vinf)*180/np.pi
                if ii>10*ii_ini:
                    aoa_mean_vec[ii+1] = np.mean(aoa_vec[9*ii_ini:ii])
                elif ii>ii_ini:
                    aoa_mean_vec[ii+1] = np.mean(aoa_vec[ii-ii_ini:ii])
                else: 
                    aoa_mean_vec[ii+1] = aoa_vec[0]
                delta_aoa_vec[ii+1] = aoa_vec[ii+1]-aoa_mean_vec[ii+1]
                datalstm1[:,0] = norm(aoa_mean_vec,alpha0_train1)
                datalstm1[:,1] = norm(delta_aoa_vec,deltaalpha_train1)
                datalstm1[:,2] = norm(aoad_vec,alphap_train1)
                datalstm1[:,3] = norm(aoadd_vec,alphapp_train1)
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
    data_vec_krefuni["time"] = timevec
    intpart = str(np.floor(ktheta2d_adim))
    decpart = str(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000))
    name_kref = "data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4)+ "_LSTM_"+ str(unitsfnn)+"_Epoch_" + str(numepochfnn) +"mma_1ms_3s.csv"
    data_vec_krefuni.to_csv(name_kref)
    data_kref_fnn.append(data_vec_krefuni)
        
    torsionmean_fnn[jj] = np.mean(theta_vec[10*nmean:])
    bendingmean_fnn[jj] = np.mean(w_vec[10*nmean:])
    torsionmax_fnn[jj] = np.max(theta_vec[10*nmean:])
    bendingmax_fnn[jj] = np.max(w_vec[10*nmean:])
    torsionmin_fnn[jj] = np.min(theta_vec[10*nmean:])
    bendingmin_fnn[jj] = np.min(w_vec[10*nmean:])

#%% Structural Model
kadim_vec =  np.array([3,5.5,6.5,8.5,12.5,20,50,90])#3,4,5,6,7,12,21,80[5,7,20,80]np.array([470,700,1000,1300,1600,1920,2000,3300,13480])*1e6
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
    young_E = 2000*1e6
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
    # I2d_adim =  I2d/(0.5*rhoinf*chord**4)
    # m2d_adim = m2d/(0.5*rhoinf*chord**2)
    # kw2d_adim = kw2d/(0.5*rhoinf*vinf**2)
    
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
    wdd_vec_filt = np.zeros((numtime,))
    aoadd_vec = np.zeros((numtime,))
    wddd_vec = np.zeros((numtime,))
    flag_vec = np.zeros((numtime,))
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
    flag_vec[0] = 1
    aoa_vec[0] = (theta_vec[0]+wd_vec[0]/vinf)*180/np.pi
    aoad_vec[0] =(thetad_vec[0]+wdd_vec[0]/vinf)*180/np.pi
    aoadd_vec[0] = (thetadd_vec[0]+wddd_vec[0]/vinf)*180/np.pi
    delta_aoa_vec[0] = 0
    aoa_mean_vec[0] = aoa_vec[0]
    datalstm1 = np.zeros((numtime,5))
    datalstm1[:,0] = norm(aoa_mean_vec,alpha0_train1)
    datalstm1[:,1] = norm(delta_aoa_vec,deltaalpha_train1)
    datalstm1[:,2] = norm(aoad_vec,alphap_train1)
    datalstm1[:,3] = norm(aoadd_vec,alphapp_train1)
    datalstm1[:,4] = flag_vec
    data_lstm1 = np.reshape(datalstm1,(numtime,1,5))
    flag_in = 0

    freq = (ktheta2d/I2d)**(0.5)/(2*np.pi)
    period = 1/freq
    nmean = 2*int(period/deltatime)

    ii_ini = nmean 

    for ii in np.arange(numtime-1):
        if flag_in == 0:
            if np.abs(aoa_vec[ii]) < 60:
                if w_vec[ii] > 0.05:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],0.05]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],0.05]))
                elif w_vec[ii] <0:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],0]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],0]))
                else:
                    cl_st[ii] = f_cl_st_w(np.array([aoa_vec[ii],w_vec[ii]]))
                    cm_st[ii] = -f_cm_st_w(np.array([aoa_vec[ii],w_vec[ii]]))
                data_pred1 = model1b.predict(data_lstm1)
                cl_dyn1b = data_pred1[:,0]
                cm_dyn1b = data_pred1[:,1]
                cl_dyn1[ii] = desnorm(cl_dyn1b[ii],deltacl_train1)
                cl1[ii] = float(cl_st[ii]+cl_dyn1[ii])
                cm_dyn1[ii] = -desnorm(cm_dyn1b[ii],deltacm_train1)
                cm1[ii] = float(cm_st[ii]+cm_dyn1[ii])
                
                
                cl = cl1
                cm = cm1
                if ii%1000==0:
                    print('lstm',jj,ii/(numtime-1),aoa_mean_vec[ii],aoa_vec[ii],aoad_vec[ii],aoadd_vec[ii])
                # print(ii/(numtime-1),cl1[ii],cl2[ii],cl3[ii],cl[ii])
                
                if ii>ii_ini and ii<ii_ini+5:
                    Amat_theta = np.array([[0,1],[-ktheta2d/I2d,0]])
                    bmat_theta = np.array([[0],[0.5*rhoinf*chord**2*vinf**2*cm[ii]/I2d]])
                    xvec_theta = np.array([[theta_vec[ii]-theta_vec[0]],[thetad_vec[ii]]])
                    xvec_theta, xvecd_theta =  rk4_explicit(Amat_theta,bmat_theta,xvec_theta,deltatime)
                    # xvec_theta, xvecd_theta = pred_corr(Amat_theta,bmat_theta,xvec_theta,xini_theta,deltatime)
                    theta_vec[ii+1] = theta_vec[0]+xvec_theta[0][0]
                    thetad_vec[ii+1] = xvec_theta[1][0]
                    thetadd_vec[ii+1] = xvecd_theta[1][0]#(0.5*rhoinf*chord**2*vinf**2*cm[ii]-ktheta2d*(theta_vec[ii+1]-theta_vec[0]))/I2d
                    wdd2 =wdd_vec[ii]
                    Amat_w = np.array([[0,1],[-kw2d/m2d,0]])
                    bmat_w = np.array([[0],[0.5*rhoinf*chord*vinf**2*cl[ii]/m2d]])
                    xvec_w = np.array([[w_vec[ii]-w_vec[0]],[wd_vec[ii]]])
                    xvec_w, xvecd_w =  rk4_explicit(Amat_w,bmat_w,xvec_w,deltatime)
                    # xvec_w, xvecd_w = pred_corr(Amat_w,bmat_w,xvec_w,xini_w,deltatime)
                    w_vec[ii+1] = w_vec[0]+xvec_w[0][0]
                    wd_vec[ii+1] = xvec_w[1][0]
                    wdd_vec[ii+1] =  xvecd_w[1][0]#(0.5*rhoinf*chord*vinf**2*cl[ii]-kw2d*(w_vec[ii+1]-w_vec[0]))/m2d
                    wdd_vec_filt[ii+1] = savgol_filter(wdd_vec[:ii+2],ii_ini+1,3)[-1]
                    wddd_vec[ii+1] = (wdd_vec_filt[ii+1] - wdd_vec_filt[ii])/(deltatime)
                elif ii> ii_ini+4:
                    Amat_theta = np.array([[0,1],[-ktheta2d/I2d,0]])
                    bmat_theta = np.array([[0],[0.5*rhoinf*chord**2*vinf**2*cm[ii]/I2d]])
                    xvec_theta = np.array([[theta_vec[ii]-theta_vec[0]],[thetad_vec[ii]]])
                    xini_theta = np.array([theta_vec[ii-3:ii]-theta_vec[0],thetad_vec[ii-3:ii]])
                    # xvec_theta, xvecd_theta =  rk4_explicit(Amat_theta,bmat_theta,xvec_theta,deltatime)
                    xvec_theta, xvecd_theta = pred_corr(Amat_theta,bmat_theta,xvec_theta,xini_theta,deltatime)
                    theta_vec[ii+1] = theta_vec[0]+xvec_theta[0][0]
                    thetad_vec[ii+1] = xvec_theta[1][0]
                    thetadd_vec[ii+1] = xvecd_theta[1][0]#(0.5*rhoinf*chord**2*vinf**2*cm[ii]-ktheta2d*(theta_vec[ii+1]-theta_vec[0]))/I2d
                    wdd2 =wdd_vec[ii]
                    Amat_w = np.array([[0,1],[-kw2d/m2d,0]])
                    bmat_w = np.array([[0],[0.5*rhoinf*chord*vinf**2*cl[ii]/m2d]])
                    xvec_w = np.array([[w_vec[ii]-w_vec[0]],[wd_vec[ii]]])
                    xini_w = np.array([w_vec[ii-3:ii]-w_vec[0],wd_vec[ii-3:ii]])
                    # xvec_w, xvecd_w =  rk4_explicit(Amat_w,bmat_w,xvec_w,deltatime)
                    xvec_w, xvecd_w = pred_corr(Amat_w,bmat_w,xvec_w,xini_w,deltatime)
                    w_vec[ii+1] = w_vec[0]+xvec_w[0][0]
                    wd_vec[ii+1] = xvec_w[1][0]
                    wdd_vec[ii+1] =  xvecd_w[1][0]#(0.5*rhoinf*chord*vinf**2*cl[ii]-kw2d*(w_vec[ii+1]-w_vec[0]))/m2d
                    wdd_vec_filt[ii+1] = savgol_filter(wdd_vec[:ii+2],ii_ini+1,3)[-1]
                    wddd_vec[ii+1] = (wdd_vec_filt[ii+1] - wdd_vec_filt[ii])/(deltatime)
                else:
                    thetadd_vec[ii+1] = thetadd_vec[ii]
                    wdd2 =wdd_vec[ii]
                    wdd_vec[ii+1] =  wdd_vec[ii]
                    wddd_vec[ii+1] =  wddd_vec[ii]
                    thetad_vec[ii+1] = thetad_vec[ii]
                    wd_vec[ii+1] = wd_vec[ii]
                    theta_vec[ii+1] = theta_vec[ii]
                    w_vec[ii+1] = w_vec[ii]
        
                aoa_vec[ii+1] = (theta_vec[ii+1]-wd_vec[ii+1]/vinf)*180/np.pi
                aoad_vec[ii+1] =(thetad_vec[ii+1]-wdd_vec[ii+1]/vinf)*180/np.pi
                aoadd_vec[ii+1] = (thetadd_vec[ii+1]-wddd_vec[ii+1]/vinf)*180/np.pi
                flag_vec[ii+1] = 0
                if ii>10*ii_ini:
                    aoa_mean_vec[ii+1] = np.mean(aoa_vec[9*ii_ini:ii])
                elif ii>ii_ini:
                    aoa_mean_vec[ii+1] = np.mean(aoa_vec[ii-ii_ini:ii])
                else: 
                    aoa_mean_vec[ii+1] = aoa_vec[0]
                delta_aoa_vec[ii+1] = aoa_vec[ii+1]-aoa_mean_vec[ii+1]
                datalstm1[:,0] = norm(aoa_mean_vec,alpha0_train1)
                datalstm1[:,1] = norm(delta_aoa_vec,deltaalpha_train1)
                datalstm1[:,2] = norm(aoad_vec,alphap_train1)
                datalstm1[:,3] = norm(aoadd_vec,alphapp_train1)
                datalstm1[:,4] = flag_vec
                data_lstm1 = np.reshape(datalstm1,(numtime,1,5))
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
    data_vec_krefuni["time"] = timevec
    intpart = str(np.floor(ktheta2d_adim))
    decpart = str(np.floor((ktheta2d_adim-(np.floor(ktheta2d_adim)))*10000))
    name_kref = "data_aeroelastic_"+ intpart.zfill(3) + "_" +decpart.zfill(4)+ "_LSTM_"+ str(unitslstm)+"_Epoch_" + str(numepochlstm) +"mma_1ms_3s.csv"
    data_vec_krefuni.to_csv(name_kref)
    data_kref.append(data_vec_krefuni)
        
    torsionmean[jj] = np.mean(theta_vec[10*nmean:])
    bendingmean[jj] = np.mean(w_vec[10*nmean:])
    torsionmax[jj] = np.max(theta_vec[10*nmean:])
    bendingmax[jj] = np.max(w_vec[10*nmean:])
    torsionmin[jj] = np.min(theta_vec[10*nmean:])
    bendingmin[jj] = np.min(w_vec[10*nmean:])
    
# %%
ii = 0
dataset_path_base = 'data/comparar/';
kadimcfd =[3,5.5,6.5,8.5,12.5,20,50,90]#[3,5.25,6.44,7,8,9,12.35,12.88,21.25,86.78];
mean_theta_cfd = np.zeros(len(kadimcfd),)
max_theta_cfd = np.zeros(len(kadimcfd),)
min_theta_cfd = np.zeros(len(kadimcfd),)
mean_w_cfd = np.zeros(len(kadimcfd),)
max_w_cfd = np.zeros(len(kadimcfd),)
min_w_cfd = np.zeros(len(kadimcfd),)
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
    mean_w_cfd[ii] = np.mean(wcfd.tail(2000).values)
    max_w_cfd[ii] = np.max(wcfd.tail(2000).values)
    min_w_cfd[ii] = np.min(wcfd.tail(2000).values)
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
data_sum["kref_st"] = kref_vec_st
data_sum["thetamean_st"] = torsionmean_st
data_sum["wmean_st"] = bendingmean_st
data_sum["kref_fnn"] = kref_vec_fnn
data_sum["thetamean_fnn"] = torsionmean_fnn
data_sum["wmean_fnn"] = bendingmean_fnn
data_sum["kref_lstm"] = kref_vec
data_sum["thetamean_lstm"] = torsionmean
data_sum["wmean_lstm"] = bendingmean
data_sum["kref_cfd"] = kadimcfd
data_sum["thetamean_cfd"] = mean_theta_cfd
data_sum["wmean_cfd"] = mean_w_cfd

data_sum.to_csv("sum_aeroelasticmma_1ms_3s.csv")      

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
plt.ylim((2.5,4.5))
plt.grid()

# plt.figure(0)
# plt.fill_between(kref_vec,torsionmin*180/np.pi,torsionmax*180/np.pi,alpha=0.7,label='LSTM+Aeroelastic')
# plt.fill_between(kref_vec_fnn,torsionmin_fnn*180/np.pi,torsionmax_fnn*180/np.pi,alpha=0.3,label='FNN+Aeroelastic')
# plt.fill_between(kadimcfd,min_theta_cfd,max_theta_cfd,alpha=0.3,label='CFD')
# plt.fill_between(kref_vec_st,torsionmin_st*180/np.pi,torsionmax_st*180/np.pi,alpha=0.1,label='St')
# # plt.title('Mean torsion')
# # plt.xlabel('$k_{\Theta}^*$')
# # plt.ylabel('$\Theta_{mean}^*$')
# plt.legend()
# # plt.ylim((2.5,3.5))
# # plt.grid()


# _____________________________________________________________________________

#%%

# with open('data/workspace/2020_06_10_10_07.pkl', 'wb') as f:
    # pickle.dump([data_kref,data_kref_st,data_kref_fnn,datacfd_kref,kref_vec,torsionmean,kadimcfd,mean_theta_cfd], f)
#%%
#     # Getting back the objects:
# with open('data/workspace/2020_06_10_10_07.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#     data_kref,datacfd_kref,kref_vec,torsionmean,kadimcfd,mean_theta_cfd = pickle.load(f)
    
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
# plta=plt.figure(100)
# gs = plta.add_gridspec(1,1)
# pltb = plta.add_subplot(gs[0,0])
# pltb.plot(datacfd_kref[2]["time"].values,datacfd_kref[2]["w"].values,label="$CFD$")
# pltb.plot(timevec,data_kref[0]["w"].values,label="$LSTM$")
# pltb.plot(timevec,data_kref_st[0]["w"].values,label="$St$")
# pltb.plot(timevec,data_kref_fnn[0]["w"].values,label="$FNN$")
# pltb.set_title('k=3')
# pltb.set_xlabel('$t_{ref}$')
# pltb.set_ylabel('$w/c$')
# # pltb.set_ylim(2,3)
# pltb.grid()
# pltb.legend(loc='upper right')
# # pltb.set_xlim(0,0.2)
#%%
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

plta=plt.figure(104)
gs = plta.add_gridspec(1,1)
pltb = plta.add_subplot(gs[0,0])
pltb.plot(datacfd_kref[3]["time"].values,datacfd_kref[3]["w"].values,label="$CFD$")
pltb.plot(timevec,data_kref[3]["w"].values,label="$LSTM$")
pltb.plot(timevec,data_kref_st[3]["w"].values,label="$St$")
pltb.plot(timevec,data_kref_fnn[3]["w"].values,label="$FNN$")
pltb.set_title('k=3')
pltb.set_xlabel('$t_{ref}$')
pltb.set_ylabel('$w/c$')
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

#%%
# import dill                            
# filepath = 'data/workspace/lstm100N_FNN100N.pkl'
# dill.dump(data_kref,filepath) # Save the session
# #dill.load_session(filepath) # Load the session

with open('data/workspace/lstm'+str(unitslstm)+'Nepoch'+str(numepochlstm)+'N_FNN'+str(unitsfnn)+'Nepoch'+str(numepochfnn)+'articleb.pkl', 'wb') as f:
    pickle.dump([data_sum,timevec,data_kref,data_kref_st,data_kref_fnn,datacfd_kref,
                 kref_vec,torsionmean,torsionmin,torsionmax,bendingmean,bendingmin,bendingmax,
                 kref_vec_fnn,torsionmean_fnn,torsionmin_fnn,torsionmax_fnn,bendingmean_fnn,bendingmin_fnn,bendingmax_fnn,
                 kref_vec_st,torsionmean_st,torsionmin_st,torsionmax_st,bendingmean_st,bendingmin_st,bendingmax_st,
                 kadimcfd,mean_theta_cfd,min_theta_cfd,max_theta_cfd,mean_w_cfd,min_w_cfd,max_w_cfd], f)