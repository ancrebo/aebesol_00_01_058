# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 11:43:46 2020

@author: ancrebo
"""

# Load packages

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

print(tf.__version__) 
#%% Se carga la informacion basica, epoch, nombrered, etc
numepoch = 5000
unitsfnn1 = 50
namesave = 'FNN_'+str(unitsfnn1)+'N'+str(numepoch)+'epochmodel1'
namesavered = 'redes/'+namesave
#model = keras.models.load_model(namesave + '.h5')
polar = pd.read_csv('polar/polar.csv')
# %%
# Se leen todos los datos temporales
chord = 0.1
vinf = 20
dataset_path_base = '../../cycle_bbdd3/'
ii=0

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
    alpha = (-data["rotation Monitor: Expression (deg)"].values)
    f_cl_st = interp1d(polar["aoa"].values,polar["cl"].values)
    cl_st = f_cl_st(alpha)
    data["cl_st"] = cl_st
    f_cm_st = interp1d(polar["aoa"].values,polar["cm "].values)
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
    
    datautil["CL"] = dataperiod["CL Monitor: Force Coefficient"].values
    datautil["CL_st"] = dataperiod["cl_st"].values
    datautil["Delta CL"] =  dataperiod["deltaCL"].values
    datautil["CM"] = dataperiod["CM Monitor: Moment Coefficient"].values/chord
    datautil["CM_st"] = dataperiod["cm_st"].values
    datautil["Delta CM"] =  dataperiod["deltaCM"].values
    datautil["Alpha0"] = dataperiod["Alpha0 Monitor: Expression"].values*180/np.pi
    datautil["Delta Alpha"] = (-dataperiod["rotation Monitor: Expression (deg)"].values-dataperiod["Alpha0 Monitor: Expression"].values * 180 / np.pi)
    datautil["Alpha prima"] = dataperiod["Alpha prima"].values
    datautil["Alpha prima prima"] = dataperiod["Alpha prima prima"].values
    datautil["Flag"] = np.concatenate(([1E-10],np.zeros(int(point_period)-1)))
    datautil["freq"] = freq
    
    if  data["Amplitude Monitor: Expression"].values[-1]*180/np.pi<5.5 and  data["Alpha0 Monitor: Expression"].values[-1]*180/np.pi<7 and freq>20 and freq<70:
    
        if np.mod(ii,10)==0:
            data_tot = pd.concat([data_tot, datautil],axis=0)
        else:
            data_tot1 = pd.concat([data_tot1, datautil],axis=0)
        ii=ii+1 
data_tot1 = data_tot1.dropna()
data_tot = data_tot.dropna()

#%% Funciones para normalizar desnormalizar datos
def norm(x,y):
    return (x-np.amin(y))/(np.amax(y)-np.amin(y))
def desnorm(x,y):
    return x*(np.amax(y)-np.amin(y))+np.amin(y)


#%% Se normalizan los datos de entrenamiento y se preparan para la lstm
freq1 = data_tot1.pop("freq")  
flagout1 = data_tot1.pop("Flag")       
data_tot1b = data_tot1.copy()
data_norm1 = norm(data_tot1,data_tot1b)
data_norm1 = data_norm1.fillna(0)
dataval_tot1 = data_norm1.values


trainx1 = dataval_tot1[0:len(dataval_tot1):1,6:]
train_cl1 = dataval_tot1[0:len(dataval_tot1):1,0]
train_clst1 = dataval_tot1[0:len(dataval_tot1):1,1]
train_deltacl1 = dataval_tot1[0:len(dataval_tot1):1,2]
train_cm1 = dataval_tot1[0:len(dataval_tot1):1,3]
train_cmst1 = dataval_tot1[0:len(dataval_tot1):1,4]
train_deltacm1 = dataval_tot1[0:len(dataval_tot1):1,5]


mat_train1 = np.zeros((len(train_deltacl1),2))
mat_train1[:,0] = train_deltacl1
mat_train1[:,1] = train_deltacm1


dimtemp1 = trainx1.shape[0]
dimvar1 = trainx1.shape[1]
# trainx1 = np.reshape(trainx1,(trainx1.shape[0],1,trainx1.shape[1]))
# testx1 = np.reshape(testx1,(testx1.shape[0],1,testx1.shape[1]))


#%%


freq = data_tot.pop("freq")    
flagout = data_tot.pop("Flag")    

column_names = ["CL","CL_st","Delta CL","CM","CM_st","Delta CM","Alpha0","Delta Alpha","Alpha prima","Alpha prima prima"]
data_norm1t = norm(data_tot,data_tot1b)
data_norm1t = data_norm1t.fillna(0)
data_norm1t = data_norm1t.reindex(columns=column_names)
dataval_tot1t = data_norm1t.values
     

#%%
trainx1t = dataval_tot1t[0:len(dataval_tot1t):1,6:]
train_cl1t = dataval_tot1t[0:len(dataval_tot1t):1,0]
train_clst1t = dataval_tot1t[0:len(dataval_tot1t):1,1]
train_deltacl1t = dataval_tot1t[0:len(dataval_tot1t):1,2]
train_cm1t = dataval_tot1t[0:len(dataval_tot1t):1,3]
train_cmst1t = dataval_tot1t[0:len(dataval_tot1t):1,4]
train_deltacm1t = dataval_tot1t[0:len(dataval_tot1t):1,5]


mat_train1t = np.zeros((len(train_deltacl1t),2))
mat_train1t[:,0] = train_deltacl1t
mat_train1t[:,1] = train_deltacm1t


dimtemp1t = trainx1t.shape[0]
dimvar1t = trainx1t.shape[1]
#%%


trainPredict_cl1t = np.zeros((dimtemp1t,1))
trainPredict_cm1t = np.zeros((dimtemp1t,1))

#%%
val_trainx1t = np.zeros((dimtemp1,dimvar1))
val_trainx1t[:dimtemp1t,:] =trainx1t
val_mat_train1t = np.zeros((dimtemp1,2))
val_mat_train1t[:dimtemp1t,:] = mat_train1t

#%% 
model1 = Sequential()
model1.add(Dense(unitsfnn1, activation='sigmoid', input_shape=[dimvar1]))
model1.add(Dense(2))
optimizer = tf.keras.optimizers.RMSprop(0.001) 
model1.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])  


#%%
#loss_train = np.zeros((numepoch,)) #np.concatenate((loss_train,np.zeros((int(numepoch/2),))))#
#loss_test =np.zeros((numepoch,)) # np.concatenate((loss_test,np.zeros((int(numepoch/2),))))
#for ii in range(numepoch): #range(int(numepoch/2),numepoch):#
#    print('Epoch: '+str(ii)+'/'+str(numepoch))
#    train_clcm = model1.predict(trainx1)
#    test_clcm = model1.predict(trainx1t)
#    test_clcm = test_clcm[:dimtemp1t,:]
#    test_clcmreal = mat_train1t
#    loss_train[ii] = np.mean(sum((mat_train1-train_clcm)**2)/dimtemp1)
#    loss_test[ii] = np.mean(sum((test_clcmreal-test_clcm)**2)/dimtemp1t)
#    history1 = model1.fit(trainx1, mat_train1, epochs=1,verbose=0)
#    print('loss ='+"{:.4f}".format(loss_train[ii])+' -- loss_val = ',"{:.4f}".format(loss_test[ii]))
#    
    
#%%  
#plt.figure(0)
#plt.plot(np.arange(numepoch),loss_train)
#plt.plot(np.arange(numepoch),loss_test)
#plt.xlabel('epoch')
#plt.ylabel('MSE')
#plt.grid()
#plt.yscale("log")


#%%

model1 = keras.models.load_model(namesavered + '.h5')
trainPredict1 = model1.predict(trainx1)
trainPredict_cl1 = trainPredict1[:,0]
trainPredict_cm1 = trainPredict1[:,1]

#%%
CL_cfd1 = train_cl1*(np.amax(data_tot1b["CL"])-np.amin(data_tot1b["CL"]))+np.amin(data_tot1b["CL"])
CL_st1 = train_clst1*(np.amax(data_tot1b["CL_st"])-np.amin(data_tot1b["CL_st"]))+np.amin(data_tot1b["CL_st"])
CL_dyn1 = trainPredict_cl1*(np.amax(data_tot1b["Delta CL"])-np.amin(data_tot1b["Delta CL"]))+np.amin(data_tot1b["Delta CL"])
CM_cfd1 = train_cm1*(np.amax(data_tot1b["CM"])-np.amin(data_tot1b["CM"]))+np.amin(data_tot1b["CM"])
CM_st1 = train_cmst1*(np.amax(data_tot1b["CM_st"])-np.amin(data_tot1b["CM_st"]))+np.amin(data_tot1b["CM_st"])
CM_dyn1 = trainPredict_cm1*(np.amax(data_tot1b["Delta CM"])-np.amin(data_tot1b["Delta CM"]))+np.amin(data_tot1b["Delta CM"])
alpha0_case1 = trainx1[:,0]*(np.amax(data_tot1b["Alpha0"])-np.amin(data_tot1b["Alpha0"]))+np.amin(data_tot1b["Alpha0"])
deltaalpha_case1 = trainx1[:,1]*(np.amax(data_tot1b["Delta Alpha"])-np.amin(data_tot1b["Delta Alpha"]))+np.amin(data_tot1b["Delta Alpha"])
alpha_case1 = alpha0_case1+deltaalpha_case1
CL_nn1 = CL_st1+CL_dyn1
CM_nn1 = CM_st1+CM_dyn1
tiempo1 = delta_time*(np.arange(0,len(CL_cfd1),1))
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

#%%
#model1.save(namesavered + 'model1.h5')


#%%
weights0 = model1.layers[0].get_weights()
weights1 = model1.layers[1].get_weights()
model1t = Sequential()
model1t.add(Dense(unitsfnn1, activation='sigmoid', input_shape=[dimvar1]))
model1t.add(Dense(2))
model1t.layers[0].set_weights(weights0)
model1t.layers[1].set_weights(weights1)
optimizer = tf.keras.optimizers.RMSprop(0.001) 
model1t.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse']) 




#%%
trainPredict1t = model1t.predict(trainx1t)
trainPredict_cl1t = trainPredict1t[:,0]
trainPredict_cm1t = trainPredict1t[:,1]

#%%
CL_cfd = train_cl1t*(np.amax(data_tot1b["CL"])-np.amin(data_tot1b["CL"]))+np.amin(data_tot1b["CL"])
CL_st = train_clst1t*(np.amax(data_tot1b["CL_st"])-np.amin(data_tot1b["CL_st"]))+np.amin(data_tot1b["CL_st"])
CM_cfd = train_cm1t*(np.amax(data_tot1b["CM"])-np.amin(data_tot1b["CM"]))+np.amin(data_tot1b["CM"])
CM_st = train_cmst1t*(np.amax(data_tot1b["CM_st"])-np.amin(data_tot1b["CM_st"]))+np.amin(data_tot1b["CM_st"])
CL_dyn1t = trainPredict_cl1t*(np.amax(data_tot1b["Delta CL"])-np.amin(data_tot1b["Delta CL"]))+np.amin(data_tot1b["Delta CL"])
CM_dyn1t = trainPredict_cm1t*(np.amax(data_tot1b["Delta CM"])-np.amin(data_tot1b["Delta CM"]))+np.amin(data_tot1b["Delta CM"])
CL_nn1t = CL_st+CL_dyn1t
CM_nn1t = CM_st+CM_dyn1t



CL_nn = np.zeros((dimtemp1t,1))
CM_nn = np.zeros((dimtemp1t,1))
freq_vec = freq.values
CL_nn = CL_nn1t
CM_nn = CM_nn1t
CL_dyn = CL_dyn1t
CM_dyn = CM_dyn1t
# %%
plt.figure(25)
plt.plot(CL_cfd,label="$C_L$ (CFD)")
plt.plot(CL_nn,label="$C_L$ (NN)")
plt.plot(CL_st,label="$C_L$ (St)")
plt.title('CL prediction')
plt.xlabel('Data')
plt.ylabel('CL')
plt.legend()
plt.grid()
plt.figure(26)
plt.plot(CM_cfd,label="$C_M$ (CFD)")
plt.plot(CM_nn,label="$C_M$ (NN)")
plt.plot(CM_st,label="$C_M$ (St)")
plt.title('CM prediction')
plt.xlabel('Data')
plt.ylabel('CM')
plt.legend()
plt.grid()

#%% Export data#%% Export data
Exportdata1 = pd.DataFrame()
Exportdata1["DeltaCL_pred"] = CL_dyn1[:,]
Exportdata1["DeltaCL_real"] = data_tot1b["Delta CL"].values
Exportdata1["CL_st"] = CL_st1[:,]
Exportdata1["CL_pred"] = CL_nn1[:,]
Exportdata1["CL_real"] = data_tot1b["CL"].values
Exportdata1["Error_deltacl"] = (Exportdata1["DeltaCL_pred"]-Exportdata1["DeltaCL_real"])
Exportdata1["Error_cl"] = (Exportdata1["CL_pred"]-Exportdata1["CL_real"])
Exportdata1["DeltaCM_pred"] = CM_dyn1[:,]
Exportdata1["DeltaCM_real"] = data_tot1b["Delta CM"].values
Exportdata1["CM_st"] = CM_st1[:,]
Exportdata1["CM_pred"] = CM_nn1[:,]
Exportdata1["CM_real"] = data_tot1b["CM"].values
Exportdata1["Error_deltacm"] = (Exportdata1["DeltaCM_pred"]-Exportdata1["DeltaCM_real"])
Exportdata1["Error_cm"] = (Exportdata1["CM_pred"]-Exportdata1["CM_real"])
Exportdata1["aoa_0"] = trainx1[:,0]*(np.amax(data_tot1b["Alpha0"])-np.amin(data_tot1b["Alpha0"]))+np.amin(data_tot1b["Alpha0"])
Exportdata1["delta_aoa"] = trainx1[:,1]*(np.amax(data_tot1b["Delta Alpha"])-np.amin(data_tot1b["Delta Alpha"]))+np.amin(data_tot1b["Delta Alpha"])
Exportdata1["aoa_d"] = trainx1[:,2]*(np.amax(data_tot1b["Alpha prima"])-np.amin(data_tot1b["Alpha prima"]))+np.amin(data_tot1b["Alpha prima"])
Exportdata1["aoa_dd"] = trainx1[:,3]*(np.amax(data_tot1b["Alpha prima prima"])-np.amin(data_tot1b["Alpha prima prima"]))+np.amin(data_tot1b["Alpha prima prima"])
Exportdata1.to_csv("cl_cm_error/"+namesave+"_train.csv")
Exportdata = pd.DataFrame()
Exportdata["DeltaCL_pred"] = CL_dyn[:,]
Exportdata["DeltaCL_real"] = data_tot["Delta CL"].values
Exportdata["CL_st"] = CL_st[:,]
Exportdata["CL_pred"] = CL_nn[:,]
Exportdata["CL_real"] = data_tot["CL"].values
Exportdata["Error_deltacl"] = (Exportdata["DeltaCL_pred"]-Exportdata["DeltaCL_real"])
Exportdata["Error_cl"] = (Exportdata["CL_pred"]-Exportdata["CL_real"])
Exportdata["DeltaCM_pred"] = CM_dyn[:,]
Exportdata["DeltaCM_real"] = data_tot["Delta CM"].values
Exportdata["CM_st"] = CM_st[:,]
Exportdata["CM_pred"] = CM_nn[:,]
Exportdata["CM_real"] = data_tot["CM"].values
Exportdata["Error_deltacm"] = (Exportdata["DeltaCM_pred"]-Exportdata["DeltaCM_real"])
Exportdata["Error_cm"] = (Exportdata["CM_pred"]-Exportdata["CM_real"])
Exportdata["aoa_0"] = trainx1t[:,0]*(np.amax(data_tot1b["Alpha0"])-np.amin(data_tot1b["Alpha0"]))+np.amin(data_tot1b["Alpha0"])
Exportdata["delta_aoa"] = trainx1t[:,1]*(np.amax(data_tot1b["Delta Alpha"])-np.amin(data_tot1b["Delta Alpha"]))+np.amin(data_tot1b["Delta Alpha"])
Exportdata["aoa_d"] = trainx1t[:,2]*(np.amax(data_tot1b["Alpha prima"])-np.amin(data_tot1b["Alpha prima"]))+np.amin(data_tot1b["Alpha prima"])
Exportdata["aoa_dd"] = trainx1t[:,3]*(np.amax(data_tot1b["Alpha prima prima"])-np.amin(data_tot1b["Alpha prima prima"]))+np.amin(data_tot1b["Alpha prima prima"])
Exportdata.to_csv("cl_cm_error/"+namesave+"_test.csv")
#%% error energetico
def integration(x,y):
    x = np.append(x,x[0])
    y = np.append(y,y[0])
    deltax = np.diff(x)
    deltay = np.diff(y)
    intvalue = np.sum(deltax*y[:-1]+deltax*deltay/2)
    return intvalue

int_data1 = pd.DataFrame()
data_tot1c = data_tot1b.reset_index()
flagout2 = flagout1.values
flagout3 = flagout1.reset_index()
index_fin1 = flagout3["index"][(flagout3["Flag"]>0)== True].index.values
index_fin1 = np.append(index_fin1,len(data_tot1c)-1)
LL = len(index_fin1)-1
intdatapre1=np.zeros((LL,1))
jj=0
for ii in index_fin1:
    int_dataprev1 = pd.DataFrame()
    if ii == 0:
        ii2 = ii
    elif ii == len(data_tot1c)-1:
        integerrcl1 = integration(data_tot1c["Delta Alpha"].values[ii2:ii],Exportdata1["Error_deltacm"].values[ii2:ii])
        intdatapre1[jj,:] = integerrcl1
        print(jj,intdatapre1[jj,:])
        jj=jj+1
    else:
        integerrcl1 = integration(data_tot1c["Delta Alpha"].values[ii2:ii],Exportdata1["Error_deltacm"].values[ii2:ii])
        intdatapre1[jj,:] = integerrcl1
        print(jj,intdatapre1[jj,:])
        
        ii2 = ii
        jj=jj+1
    int_data1["Ecy_CM"] = intdatapre1[:,0]

int_data1.to_csv("energy_error/"+namesave+"_cycle_train.csv")
int_data = pd.DataFrame()
data_totc = data_tot.reset_index()
flagout2 = flagout.values
flagout3 = flagout.reset_index()
index_fin = flagout3["index"][(flagout3["Flag"]>0)== True].index.values
index_fin = np.append(index_fin,len(data_totc)-1)
LL = len(index_fin)-1
intdatapre=np.zeros((LL,1))
jj=0
for ii in index_fin:
    int_dataprev = pd.DataFrame()
    if ii == 0:
        ii2 = ii
    elif ii == len(data_totc)-1:
        integerrcl = integration(data_totc["Delta Alpha"].values[ii2:ii],Exportdata["Error_deltacm"].values[ii2:ii])
        intdatapre[jj,:] = integerrcl
        print(jj,intdatapre[jj,:])
        jj=jj+1
    else:
        integerrcl = integration(data_totc["Delta Alpha"].values[ii2:ii],Exportdata["Error_deltacm"].values[ii2:ii])
        intdatapre[jj,:] = integerrcl
        print(jj,intdatapre[jj,:])
        
        ii2 = ii
        jj=jj+1
    int_data["Ecy_CM"] = intdatapre[:,0]

int_data.to_csv("energy_error/"+namesave+"_cycle_test.csv")

#%%
#data_train = pd.DataFrame()
#data_train["epoch"] = np.arange(numepoch)+1
#data_train["losstrain"] = loss_train
#data_train["losstest"] = loss_test
#data_train.to_csv("training_loss/"+namesave+"_datatrain.csv")