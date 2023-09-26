# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:38:26 2020

@author: ancrebo
"""
import os
import numpy as np
import re
current_path = os.getcwd()
import pandas as pd
#os.chdir('../../')
# os.path.join('../../')
from launch_abesol import launch_abesol
from scipy import interpolate
import pandas as pd
# from launch_abesol import launch_abesol
os.chdir(current_path)
slope0 = np.nan
slope_min = 1e6
per_slope = 0.03
lambda_opt = 7 # np.linspace(1,10,10)
E_young = 141.96e9
rho_inf = 1.225
delta_v = 0.2
tol = 1e-4
vinf = 1
omega_max = 15
flagerror = 0
per_slope = 0.05
hist_v = []
hist_P = []
hist_vinf_vec=[]
hist_omega_vec=[]
hist_omega_vec=[]
hist_P=[]
hist_T=[]
hist_lambda=[]
flagsave_vec = []
lambda_val = lambda_opt
ct = []
cp = []
T = []
P = [] 
omega_vec = []
Eadim_vec = []
lambda_vec = []
vinf_vec = []
v_max = 25
lambda_min = 1
max_power = 19.8e3
casebeam2 = '25r'
while flagerror == 0: 
    flagsavecase = 0 
    flag = 1
    while flag == 1:
        print('Velocity = ' + str(vinf) + 'm/s')
        Eadim_val = E_young/(1/2*rho_inf*vinf**2*(1+(lambda_val)**2))
        Eadim_vec.append(Eadim_val)
        lambda_vec.append(lambda_val)
        if vinf > v_max or lambda_val<1:
            flag = 0
            Power_val = P[-1]
            print(Power_val)
        with open('D:/Documentos/Doctorado/turbine_NREL_phaseiv/NREL_S809_structure'+casebeam2+'/operation/case_base.abe', 'r') as file:
            folder = 'D:/Documentos/Doctorado/turbine_NREL_phaseiv/NREL_S809_structure'+casebeam2+'/operation/vinf_'+str(int(vinf*100))
            # data = file.read().replace('$FOLDER$', folder)
            data = file.read()
#            pattern = '(?<=BC_VINF=)(.*)'
#            result = re.search(pattern, data)
#            vinf = float(result.groups(0)[0])
            pattern = '(?<=BC_RADIUS=)(.*)'
            result = re.search(pattern, data)
            rad = float(result.groups(0)[0])
#            pattern = '(?<=BC_RHO=)(.*)'
#            result = re.search(pattern, data)
#            rho_inf = float(result.groups(0)[0])
            omega_val = lambda_val/rad*vinf
            data = data.replace('$OMEGA$', str(omega_val))
            data = data.replace('$VINF$',str(vinf))
            data = data.replace('$RHO$',str(rho_inf))
        try:
            os.mkdir(folder)
            os.chdir(folder)
            os.mkdir('results')
            os.mkdir('vibration')
            os.chdir(current_path)
        except:
            pass
        file = "aerogenerador_"+str(int(vinf*100))+".abe"
        text_file = open(folder+'/'+file, "wt")
        n = text_file.write(data)
        text_file.close()
        solution = launch_abesol(folder,file,current_path)
        ct.append(solution.CT*(np.pi*rad**2*(omega_val*rad)**2*rho_inf)/(1/2*np.pi*rad**2*(vinf)**2*rho_inf))
        cp.append(-solution.CP*(np.pi*rad**2*(omega_val*rad)**3*rho_inf)/(1/2*np.pi*rad**2*(vinf)**3*rho_inf)) #
        T.append(solution.CT*(rho_inf*(rad*omega_val)**2*np.pi*rad**2))
        P.append(-solution.CP*(rho_inf*(rad*omega_val)**3*np.pi*rad**2))
        try:
            slope_test = (P[-1]-P[-2])/delta_v
            if np.isnan(slope0):
                slope0 = slope_test
            elif abs(slope_test) > abs(slope0):
                slope0 = slope_test
        except:
            slope_test = 1e6
        if abs(slope_test) < abs(slope_min) and lambda_val < lambda_opt:
            slope_min = slope_test
        if abs(slope_min) < abs(per_slope*slope0):
            Power_val = P[-1]
            print(Power_val)
            if abs(Power_val-max_power)/max_power <= 1e-2:
                flagerror = 1
        omega_vec.append(omega_val)
        print(omega_vec)
        vinf_vec.append(vinf)
        vinf += delta_v
        omega_val = vinf*lambda_val/rad
        if omega_val > omega_max:
            omega_val = omega_max
            lambda_val = omega_val*rad/vinf
        if lambda_val < 1:
            break
#        print('lambda = ' + str(lambda_val))
#        print('power = ' + str(-solution.CP*(rho_inf*(rad*omega_val)**3*np.pi*rad**2)), ' W')
    if flagerror == 1:
        hist_vinf_vec.append(vinf_vec)
        hist_omega_vec.append(omega_vec)
        hist_P.append(P)
        hist_T.append(T)
        hist_lambda.append(lambda_vec)
    if flagerror == 0:
        if Power_val < max_power :
            iilist = []
            for ii in np.arange(len(omega_vec)):
                if omega_vec[ii] < omega_max:
                    iilist.append(int(ii))
            print(iilist)
            omega_vec = np.array(omega_vec)[iilist].tolist()
            vinf_vec = np.array(vinf_vec)[iilist].tolist()
            P = np.array(P)[iilist].tolist()
            T = np.array(T)[iilist].tolist()
            ct = np.array(ct)[iilist].tolist()
            cp = np.array(cp)[iilist].tolist()
            lambda_vec = np.array(lambda_vec)[iilist].tolist()
            Eadim_vec = np.array(Eadim_vec)[iilist].tolist()
            vinf = vinf_vec[-1]+delta_v
            lambda_val = lambda_opt
            omega_val = vinf*lambda_val/rad
            if omega_val > omega_max:
                omega_val = omega_max
                lambda_val = omega_val*rad/vinf
            omega_max *= 1.1
        else:
            omega_max *= 0.9
            iilist = []
            for ii in np.arange(len(omega_vec)):
                if omega_vec[ii] < omega_max:
                    iilist.append(int(ii))
            print(iilist)
            omega_vec = np.array(omega_vec)[iilist].tolist()
            vinf_vec = np.array(vinf_vec)[iilist].tolist()
            P = np.array(P)[iilist].tolist()
            T = np.array(T)[iilist].tolist()
            ct = np.array(ct)[iilist].tolist()
            cp = np.array(cp)[iilist].tolist()
            lambda_vec = np.array(lambda_vec)[iilist].tolist()
            Eadim_vec = np.array(Eadim_vec)[iilist].tolist()
            vinf = vinf_vec[-1]+delta_v
            lambda_val = lambda_opt
            omega_val = vinf*lambda_val/rad
            if omega_val > omega_max:
                omega_val = omega_max
                lambda_val = omega_val*rad/vinf
    hist_v.append(vinf_vec)
    hist_P.append(P)
    
#%%

datlim = pd.read_excel('D:/Documentos/Doctorado/CFD_calc_post/postprocesar/art4_rotatewindturbine/lim_structure.xlsx')
lamlim1 = datlim['Str1_lambda'].values
Eadlim1 = datlim['Str1_Ezero'].values*10**6
lamlim2 = datlim['Str2_lambda'].values
Eadlim2 = datlim['Str2_Ezero'].values*10**6
vinf_vec2 = vinf_vec.copy()
omega_vec2 = omega_vec.copy()
ct_vec2 = ct.copy()
cp_vec2 = cp.copy()
T_vec2 = T.copy()
P_vec2 = P.copy()
Eadim_vec2 = Eadim_vec.copy()
lambda_vec2 = lambda_vec.copy()
#lambda_max = [1,2,3,4,5,6,7,8,9,10]
#%%
casebeamvec = ['25r','26r']# 
for casebeam in casebeamvec:
    if casebeam == '25r':
        lambda_max = lamlim1
        Eadim_max = Eadlim1 #[894.8705e6,338.1456e6,251.5132e6,389.1034e6,107.0775e6,27.0949e6,137.7559e6,88.1539e6,85.5356e6,84.6802e6] #[500e6,250e6,125e6,250e6,125e6,31e6,31e6,31e6,31e6,31e6] 
    elif casebeam == '26r':
        lambda_max = lamlim2
        Eadim_max = Eadlim2 #[72.4780e6,47.1354e6,47.7413e6,48.7196e6,46.9352e6,11.6180e6,10.0096e6,34.6737e6,22.0196e6,32.8561e6] #[62e6,62e6,62e6,62e6,62e6,15e6,15e6,31e6,31e6,31e6] #
    f_maxval = interpolate.interp1d(lambda_max,Eadim_max)
    for iiEE in np.arange(len(Eadim_vec2)):
        EE = Eadim_vec2[iiEE]
        Eadim_maxval = f_maxval(lambda_vec2[iiEE]) 
        print([EE,Eadim_maxval])  
        if EE > Eadim_maxval:
            vinf_vec = vinf_vec2[:iiEE+1]
            omega_vec = omega_vec2[:iiEE+1]
            ct = ct_vec2[:iiEE+1]
            cp = cp_vec2[:iiEE+1]
            T = T_vec2[:iiEE+1]
            P = P_vec2[:iiEE+1]
            Eadim_vec = Eadim_vec2[:iiEE+1]
            lambda_vec = lambda_vec2[:iiEE+1]
    bbdd = pd.DataFrame()
    bbdd['vinf'] = vinf_vec
    bbdd['omega'] = omega_vec
    bbdd['ct'] = ct
    bbdd['cp'] = cp
    bbdd['T'] = T
    bbdd['P'] = P
    bbdd['E*'] = Eadim_vec
    bbdd['lambda'] = lambda_vec
    bbdd.to_csv('D:/Documentos/Doctorado/turbine_NREL_phaseiv/NREL_S809_structure'+casebeam+'/results/operation/opvalues.csv',sep=',')
    
