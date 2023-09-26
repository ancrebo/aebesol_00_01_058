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
# from launch_abesol import launch_abesol
os.chdir(current_path)
lambda_vec =  np.linspace(0,0.1,8)
ct = []
cp = []
eta = []
omega_vec = []
Jvec = np.pi*lambda_vec
for lambda_val in lambda_vec:
    print(lambda_val)
    with open('D:/Documentos/Doctorado/rotating_plate/steady_batch/case_base.abe', 'r') as file:
        folder = 'D:/Documentos/Doctorado/rotating_plate/steady_batch/lambda_'+str(lambda_val)
        # data = file.read().replace('$FOLDER$', folder)
        data = file.read()
        pattern = '(?<=BC_VROT=)(.*)'
        result = re.search(pattern, data)
        omega_val = float(result.groups(0)[0])
        pattern = '(?<=BC_RADIUS=)(.*)'
        result = re.search(pattern, data)
        rad = float(result.groups(0)[0])
        pattern = '(?<=BC_RHO=)(.*)'
        result = re.search(pattern, data)
        rho = float(result.groups(0)[0])
        vinf = omega_val*lambda_val*rad
        data = data.replace('$VINF$', str(vinf))
    try:
        os.mkdir(folder)
        os.chdir(folder)
        os.mkdir('results')
        os.mkdir('vibration')
        os.chdir(current_path)
    except:
        pass
    file = "prop_"+str(lambda_val)+".abe"
    text_file = open(folder+'/'+file, "wt")
    n = text_file.write(data)
    text_file.close()
    solution = launch_abesol(folder,file,current_path)
    ct.append(solution.CT*0.5**4*(2*np.pi)**2*np.pi)
    cp.append(solution.CP*0.5**5*(2*np.pi)**3*np.pi)
    eta.append(solution.PE)
    omega_vec.append(omega_val)
bbdd = pd.DataFrame()
bbdd['lambda'] = lambda_vec
bbdd['J'] = Jvec
bbdd['ct'] = ct
bbdd['cp'] = cp
bbdd['eta'] = eta
bbdd.to_csv('D:/Documentos/Doctorado/rotating_plate/steady_batch/rigid_values.csv',sep=',')
    
