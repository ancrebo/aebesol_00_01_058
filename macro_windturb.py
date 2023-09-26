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
lambda_vec = np.linspace(2,10,9)
ct = []
cp = []
omega_vec = []
fol = '25r'
for lambda_val in lambda_vec:
    print(lambda_val)
    with open('D:/Documentos/Doctorado/turbine_NREL_phaseiv/NREL_S809_structure'+fol+'/steady_5ms/case_base.abe', 'r') as file:
        folder = 'D:/Documentos/Doctorado/turbine_NREL_phaseiv/NREL_S809_structure'+fol+'/steady_5ms/lambda_'+str(lambda_val)
        # data = file.read().replace('$FOLDER$', folder)
        data = file.read()
        pattern = '(?<=BC_VINF=)(.*)'
        result = re.search(pattern, data)
        vinf = float(result.groups(0)[0])
        pattern = '(?<=BC_RADIUS=)(.*)'
        result = re.search(pattern, data)
        rad = float(result.groups(0)[0])
        pattern = '(?<=BC_RHO=)(.*)'
        result = re.search(pattern, data)
        rho = float(result.groups(0)[0])
        omega_val = lambda_val/rad*vinf
        data = data.replace('$OMEGA$', str(omega_val))
    try:
        os.mkdir(folder)
        os.chdir(folder)
        os.mkdir('results')
        os.mkdir('vibration')
        os.chdir(current_path)
    except:
        pass
    file = "aerogenerador_"+str(lambda_val)+".abe"
    text_file = open(folder+'/'+file, "wt")
    n = text_file.write(data)
    text_file.close()
    solution = launch_abesol(folder,file,current_path)
    ct.append(solution.CT*(np.pi*rad**2*(omega_val*rad)**2*rho)/(1/2*np.pi*rad**2*(vinf)**2*rho))
    cp.append(-solution.CP*(np.pi*rad**2*(omega_val*rad)**3*rho)/(1/2*np.pi*rad**2*(vinf)**3*rho))
    omega_vec.append(omega_val)
bbdd = pd.DataFrame()
bbdd['lambda'] = lambda_vec
bbdd['ct'] = ct
bbdd['cp'] = cp
bbdd.to_csv('NREL_S809_structure1d/results/rigid/rigid_values.csv',sep=',')
    
