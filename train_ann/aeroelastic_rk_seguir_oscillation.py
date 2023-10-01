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
from scipy.interpolate import interp1d,interp2d

import pickle

print(tf.__version__) 



#vinf = 20
chord = 0.1


# %%
# Se lee la polar estacionaria del perfil

pathpolarbase = 'data/polares_w/' 
polar = pd.DataFrame()
for f in [f_ for f_ in os.listdir(pathpolarbase) if 'polar' in f_]:
    polaruni = pd.read_csv(pathpolarbase+f,sep=';')
    polar = pd.concat([polar, polaruni])
dataset_path_base = '../cycle_bbdd3/'
f_cl_st_w = interp2d(polar["w"].values,polar["aoa"].values,polar["cl"].values)
f_cm_st_w = interp2d(polar["w"].values,polar["aoa"].values,polar["cm"].values)
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
    alpha = (-data["rotation Monitor: Expression (deg)"].values
             + data["Alpha0 Monitor: Expression"].values * 180 / np.pi)
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
    datautil["Delta Alpha"] = -dataperiod["rotation Monitor: Expression (deg)"].values
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
totaltime = 2
numtime = int(totaltime/deltatime)
timevec = np.linspace(0,totaltime,numtime)
#%% LOAD NEURAL NETWORK MODEL LSTM 20-1
unitslstm = 50
numepochlstm = 50000
namesave = 'LSTM_'+str(unitslstm)+'N_'+str(numepochlstm)+'epoch'
namesavered = 'data/redes/'+namesave
namesave1 = namesavered + 'model1'
model1 = keras.models.load_model(namesave1 + '.h5')
unitsfnn = 50
numepochfnn = 5000
namesavefnn = 'FNN_'+str(unitsfnn)+'N'+str(numepochfnn)+'epoch'
namesavefnnred = 'data/redes/'+namesavefnn
namesave1fnn = namesavefnnred + 'model1'
model1fnn = keras.models.load_model(namesave1fnn + '.h5')

#%% 
rhoinf =  1.18415
vinf = 20
c = 0.1
cfdseguir = pd.DataFrame()
cfdmotion = pd.read_csv('data/seguirmotion/postproc_a0_03_0_A_01_00_f_45_00.csv')
cfdseguir["theta"] = (-data["rotation Monitor: Expression (deg)"].values + data["Alpha0 Monitor: Expression"].values * 180 / np.pi)
cfdseguir["w"] = np.zeros(len(cfdseguir["theta"]))
time_motion = cfdmotion["Physical Time: Physical Time (s)"].values
delta_t = time_motion[-1]-time_motion[-2]
cfdseguir["theta dot"] = np.gradient(cfdseguir["theta"],delta_t, edge_order=2)
cfdseguir["theta dotdot"] = np.gradient(cfdseguir["theta dot"],delta_t, edge_order=2)
cfdseguir["w dot"] = np.gradient(cfdseguir["w"],delta_t, edge_order=2)
cfdseguir["w dotdot"] = np.gradient(cfdseguir["w dot"],delta_t, edge_order=2)
cfdseguir["w dotdotdot"] = np.gradient(cfdseguir["w dotdot"],delta_t, edge_order=2)
cfdseguir["alpha"] = cfdseguir["theta"]-cfdseguir["w dot"]/vinf
cfdseguir["alpha dot"] = cfdseguir["theta dot"]-cfdseguir["w dotdot"]/vinf
cfdseguir["alpha dotdot"] = cfdseguir["theta dotdot"]-cfdseguir["w dotdotdot"]/vinf
alphamed = np.zeros((len(cfdseguir),))
deltaalpha = np.zeros((len(cfdseguir),))
tperiod = 1/((18.9464/3.933333e-04)**0.5/2/np.pi)
num = int(tperiod/delta_t)
print("aoamed")
#%%
for ii in np.arange(len(cfdseguir)):
    if ii> 10*num:
        alphamed[ii] = np.mean(cfdseguir["alpha"][ii-10*num:ii])
    else:
        alphamed[ii] = np.mean(cfdseguir["alpha"][:ii])
    deltaalpha[ii] = cfdseguir["alpha"][ii]-alphamed[ii]
cfdseguir["alpha mean"] = alphamed
cfdseguir["delta alpha"] = deltaalpha
cfdseguir["cl"] = cfdmotion["CL Monitor: Force Coefficient"]
cfdseguir["cm"] = -cfdmotion["CM Monitor: Moment Coefficient"]/chord
cfdseguir["flag"] = np.zeros((len(cfdseguir),))
cfdseguir["flag"][0] = 1
data2fnn = np.zeros((len(cfdseguir),4))
data2fnn[:,0] = norm(cfdseguir["alpha mean"].values,alpha0_train1)
data2fnn[:,1] = norm(cfdseguir["delta alpha"].values,deltaalpha_train1)
data2fnn[:,2] = norm(cfdseguir["alpha dot"].values,alphap_train1)
data2fnn[:,3] = norm(cfdseguir["alpha dotdot"].values,alphapp_train1)
data2lstm = np.zeros((len(cfdseguir),5))
data2lstm[:,0] = norm(cfdseguir["alpha mean"].values,alpha0_train1)
data2lstm[:,1] = norm(cfdseguir["delta alpha"].values,deltaalpha_train1)
data2lstm[:,2] = norm(cfdseguir["alpha dot"].values,alphap_train1)
data2lstm[:,3] = norm(cfdseguir["alpha dotdot"].values,alphapp_train1)
data2lstm[:,4] = cfdseguir["flag"]

data2lstm = np.reshape(data2lstm,(data2lstm.shape[0],1,data2lstm.shape[1]))


#%%
numtime = len(data2lstm)
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
# train_clst1 = np.zeros(len(cfdseguir["w"].values))
# train_cmst1 = np.zeros(len(cfdseguir["w"].values))
# for ii_aux in np.arange(len(cfdseguir["w"].values)):
#     train_clst1[ii_aux] = f_cl_st_w(cfdseguir["w"].values[ii_aux],cfdseguir["alpha"].values[ii_aux])
#     train_cmst1[ii_aux] = -f_cm_st_w(cfdseguir["w"].values[ii_aux],cfdseguir["alpha"].values[ii_aux])
train_clst1 = f_cl_st(cfdseguir["alpha"].values)
train_cmst1 = -f_cm_st(cfdseguir["alpha"].values)

#%%
trainPredict1 = model1b.predict(data2lstm)
trainPredict_cl1 = trainPredict1[:,0]
trainPredict_cm1 = trainPredict1[:,1]
#%%
CL_cfd1 = cfdseguir["cl"]
CL_st1 = train_clst1
CL_dyn1 = trainPredict_cl1*(np.amax(data_tot3["Delta CL"])-np.amin(data_tot3["Delta CL"]))+np.amin(data_tot3["Delta CL"])
CM_cfd1 = cfdseguir["cm"]
CM_st1 = train_cmst1
CM_dyn1 = trainPredict_cm1*(np.amax(data_tot3["Delta CM"])-np.amin(data_tot3["Delta CM"]))+np.amin(data_tot3["Delta CM"])
CL_nn1 = CL_st1+CL_dyn1
CM_nn1 = CM_st1-CM_dyn1
tiempo1 = delta_time*(np.arange(0,len(CL_cfd1),1))
#%%
plt.figure(16)
plt.plot(tiempo1,CL_cfd1,label="$C_L$ (CFD)")
plt.plot(tiempo1,CL_nn1,label="$C_L$ (NN)")
plt.plot(tiempo1,CL_st1,label="$C_L^{st}$")
plt.title('CL prediction prove 1')
plt.xlabel('Time (s)')
plt.ylabel('CL')
plt.legend()
plt.grid()
plt.figure(17)
plt.plot(tiempo1,CM_cfd1,label="$C_M$ (CFD)")
plt.plot(tiempo1,CM_nn1,label="$C_M$ (NN)")
plt.plot(tiempo1,CM_st1,label="$C_M^{st}$")
plt.title('CM prediction prove 1')
plt.xlabel('Time (s)')
plt.ylabel('CM')
plt.legend()
plt.grid()

