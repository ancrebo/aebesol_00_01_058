# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:04:34 2020

@author: ancrebo
"""


import pandas as pd
import numpy as np
import os

from scipy.interpolate import interp1d


#model = keras.models.load_model(namesave + '.h5')
polar = pd.read_csv('data/polar/polar.csv')

# %% Lectura de datos temporales
# %% se carga la info geométrica básica
chord = 0.1
vinf = 20
dataset_path_base = '../cycle_bbdd3/'
ii=0

data_tot = pd.DataFrame()
data_tota1 = pd.DataFrame()
data_tota2 = pd.DataFrame()
data_tota1a = pd.DataFrame()
data_tota2a = pd.DataFrame()
data_tota1b = pd.DataFrame()
data_tota2b = pd.DataFrame()
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
    datautil = pd.DataFrame()
    datautil_a1 = pd.DataFrame()
    datautil_a2 = pd.DataFrame()
    datautil_a1b = pd.DataFrame()
    datautil_a2b = pd.DataFrame()
    
    datautil["CL"] = dataperiod["CL Monitor: Force Coefficient"].values
    datautil["CL_st"] = dataperiod["cl_st"].values
    datautil["Delta CL"] =  dataperiod["deltaCL"].values
    datautil["CM"] = -dataperiod["CM Monitor: Moment Coefficient"].values/chord
    datautil["CM_st"] = -dataperiod["cm_st"].values
    datautil["Delta CM"] =  -dataperiod["deltaCM"].values
    datautil["Alpha0"] = dataperiod["Alpha0 Monitor: Expression"].values*180/np.pi
    datautil["Delta Alpha"] = (-dataperiod["rotation Monitor: Expression (deg)"].values-dataperiod["Alpha0 Monitor: Expression"].values * 180 / np.pi)
    datautil["Alpha prima"] = dataperiod["Alpha prima"].values
    datautil["Alpha prima prima"] = dataperiod["Alpha prima prima"].values
    datautil["Flag"] = np.concatenate(([1E-10],np.zeros(int(point_period)-1)))
    datautil["freq"] = freq
    

    
    if np.mod(ii,10)==0:
        data_tot = pd.concat([data_tot, datautil],axis=0)
        datautil_a1b["alpha (st)"] = datautil["Alpha0"]+datautil["Delta Alpha"]
        datautil_a1b["CL (st)"] = datautil["CL_st"]
        datautil_a1b["CM (st)"] = datautil["CM_st"]
        datautil_a1b["alpha mean (dyn)"] = datautil["Alpha0"]
        datautil_a1b["delta alpha (dyn)"] = datautil["Delta Alpha"]
        datautil_a1b["alpha dot (dyn)"] = datautil["Alpha prima"]
        datautil_a1b["alpha ddot (dyn)"] = datautil["Alpha prima prima"]
        datautil_a1b["delta CL (dyn)"] = datautil["Delta CL"]
        datautil_a1b["delta CM (dyn)"] = datautil["Delta CM"]
        datautil_a1b["inicio caso"] = datautil["Flag"]*2
        
        
        datautil_a2b["alpha"] = datautil["Alpha0"]+datautil["Delta Alpha"]
        datautil_a2b["alpha dot"] = datautil["Alpha prima"]
        datautil_a2b["alpha ddot"] = datautil["Alpha prima prima"]
        datautil_a2b["CL"] = datautil["CL"]
        datautil_a2b["CM"] = datautil["CM"]
        datautil_a2b["inicio caso"] = datautil["Flag"]*2
        datautil_a1b = datautil_a1b[::10]
        datautil_a2b = datautil_a2b[::10]    
        data_tota1b = pd.concat([data_tota1b,datautil_a1b],axis=0)
        data_tota2b = pd.concat([data_tota2b,datautil_a2b],axis=0)
        
    else:
        data_tot1 = pd.concat([data_tot1, datautil],axis=0)
        datautil_a1["alpha (st)"] = datautil["Alpha0"]+datautil["Delta Alpha"]
        datautil_a1["CL (st)"] = datautil["CL_st"]
        datautil_a1["CM (st)"] = datautil["CM_st"]
        datautil_a1["alpha mean (dyn)"] = datautil["Alpha0"]
        datautil_a1["delta alpha (dyn)"] = datautil["Delta Alpha"]
        datautil_a1["alpha dot (dyn)"] = datautil["Alpha prima"]
        datautil_a1["alpha ddot (dyn)"] = datautil["Alpha prima prima"]
        datautil_a1["delta CL (dyn)"] = datautil["Delta CL"]
        datautil_a1["delta CM (dyn)"] = datautil["Delta CM"]
        datautil_a1["inicio caso"] = datautil["Flag"]
        
        
        datautil_a2["alpha"] = datautil["Alpha0"]+datautil["Delta Alpha"]
        datautil_a2["alpha dot"] = datautil["Alpha prima"]
        datautil_a2["alpha ddot"] = datautil["Alpha prima prima"]
        datautil_a2["CL"] = datautil["CL"]
        datautil_a2["CM"] = datautil["CM"]
        datautil_a2["inicio caso"] = datautil["Flag"]
        datautil_a1 = datautil_a1[::10]
        datautil_a2 = datautil_a2[::10]
        data_tota1a = pd.concat([data_tota1a,datautil_a1],axis=0)
        data_tota2a = pd.concat([data_tota2a,datautil_a2],axis=0)
    ii=ii+1 
data_tota1 = pd.concat([data_tota1b,data_tota1a])
data_tota2 = pd.concat([data_tota2b,data_tota2a])
data_tot1 = data_tot1.dropna()
data_tot = data_tot.dropna()

#%%
data_tota1.to_excel("exportdata/data_analis1.xlsx")
data_tota2.to_excel("exportdata/data_analis2.xlsx")

#%%

dataset_path_base = 'data/datacheck/'
data_tot = pd.DataFrame()
data_tota1 = pd.DataFrame()
data_tota2 = pd.DataFrame()
data_tot1 = pd.DataFrame()
for f in [f_ for f_ in os.listdir(dataset_path_base) if 'postproc_a0_02_5_A1_05_00_f1_25_00_A2_01_0_f2_60_00' in f_]:
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
    datautil = pd.DataFrame()
    datautil_a1 = pd.DataFrame()
    datautil_a2 = pd.DataFrame()
    
    datautil["CL"] = dataperiod["CL Monitor: Force Coefficient"].values
    datautil["CL_st"] = dataperiod["cl_st"].values
    datautil["Delta CL"] =  dataperiod["deltaCL"].values
    datautil["CM"] = -dataperiod["CM Monitor: Moment Coefficient"].values/chord
    datautil["CM_st"] = -dataperiod["cm_st"].values
    datautil["Delta CM"] =  -dataperiod["deltaCM"].values
    datautil["Alpha0"] = dataperiod["Alpha0 Monitor: Expression"].values*180/np.pi
    datautil["Delta Alpha"] = (-dataperiod["rotation Monitor: Expression (deg)"].values-dataperiod["Alpha0 Monitor: Expression"].values * 180 / np.pi)
    datautil["Alpha prima"] = dataperiod["Alpha prima"].values
    datautil["Alpha prima prima"] = dataperiod["Alpha prima prima"].values
    datautil["Flag"] = np.concatenate(([1E-10],np.zeros(int(point_period)-1)))
    datautil["freq"] = freq
    
    datautil_a1["alpha (st)"] = datautil["Alpha0"]+datautil["Delta Alpha"]
    datautil_a1["CL (st)"] = datautil["CL_st"]
    datautil_a1["CM (st)"] = datautil["CM_st"]
    datautil_a1["alpha mean (dyn)"] = datautil["Alpha0"]
    datautil_a1["delta alpha (dyn)"] = datautil["Delta Alpha"]
    datautil_a1["alpha dot (dyn)"] = datautil["Alpha prima"]
    datautil_a1["alpha ddot (dyn)"] = datautil["Alpha prima prima"]
    datautil_a1["delta CL (dyn)"] = datautil["Delta CL"]
    datautil_a1["delta CM (dyn)"] = datautil["Delta CM"]
    datautil_a1["inicio caso"] = datautil["Flag"]
    
    
    datautil_a2["alpha"] = datautil["Alpha0"]+datautil["Delta Alpha"]
    datautil_a2["alpha dot"] = datautil["Alpha prima"]
    datautil_a2["alpha ddot"] = datautil["Alpha prima prima"]
    datautil_a2["CL"] = datautil["CL"]
    datautil_a2["CM"] = datautil["CM"]
    datautil_a2["inicio caso"] = datautil["Flag"]
    
    data_tota1 = pd.concat([data_tota1,datautil_a1],axis=0)
    data_tota2 = pd.concat([data_tota2,datautil_a2],axis=0)
    if  data["Amplitude Monitor: Expression"].values[-1]*180/np.pi<5.5 and  data["Alpha0 Monitor: Expression"].values[-1]*180/np.pi<7 and freq>20 and freq<70:
    
        if np.mod(ii,10)==0:
            data_tot = pd.concat([data_tot, datautil],axis=0)
        else:
            data_tot1 = pd.concat([data_tot1, datautil],axis=0)
        ii=ii+1 
data_tot1 = data_tot1.dropna()
data_tot = data_tot.dropna()


data_tota1.to_excel("exportdata/combsin_analis1.xlsx")
data_tota2.to_excel("exportdata/combsin_analis2.xlsx")
