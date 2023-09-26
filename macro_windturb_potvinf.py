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
# from launch_abesol import launch_abesol
os.chdir(current_path)
lambda_opt = 7 # np.linspace(1,10,10)
E_young = 141.96e9
rho_inf = 1.225
lambda_max = [1,2,3,4,5,6,7,8,9,10]
f_maxval = interpolate.interp1d(lambda_max,Eadim_max)
casebeam = '2c'
if casebeam == '1c':
    Eadim_max = [500e6,250e6,125e6,250e6,125e6,31e6,31e6,31e6,31e6,31e6] 
elif casebeam == '2c':
    Eadim_max = [62e6,62e6,62e6,62e6,62e6,15e6,15e6,31e6,31e6,31e6] #
delta_v = 0.1
tol = 1e-4
vinf = 1
omega_max = 20
flagerror = 0
per_slope = 0.03
hist_v = []
hist_P = []
hist_vinf_vec=[]
hist_omega_vec=[]
hist_omega_vec=[]
hist_P=[]
hist_T=[]
hist_lambda=[]
slope0 = np.nan
slope_min = 1e6
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
while flagerror == 0: 
    flagsavecase = 0 
    flag = 1
    while flag == 1:
        print('Velocity = ' + str(vinf) + 'm/s')
        Eadim_maxval = f_maxval(lambda_val)
        Eadim_val = E_young/(1/2*rho_inf*vinf**2*(1+(lambda_val)**2))
        Eadim_vec.append(Eadim_val)
        lambda_vec.append(lambda_val)
        if Eadim_val < Eadim_maxval:
            flag = 0
        with open('NREL_S809_structure'+casebeam+'/operation/case_base.abe', 'r') as file:
            folder = 'NREL_S809_structure'+casebeam+'/operation/vinf_'+str(int(vinf*100))
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
            flagerror = 1
        print([slope_test,slope_min,slope0])
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
        if slope_min < per_slope*slope0 :
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
bbdd = pd.DataFrame()
bbdd['vinf'] = vinf_vec
bbdd['omega'] = omega_vec
bbdd['ct'] = ct
bbdd['cp'] = cp
bbdd['T'] = T
bbdd['P'] = P
bbdd['E*'] = Eadim_vec
bbdd['lambda'] = lambda_vec
bbdd.to_csv('NREL_S809_structure'+casebeam+'/results/operation/opvalues.csv',sep=',')
    
