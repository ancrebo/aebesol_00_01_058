# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:38:26 2020

@author: ancrebo
"""
import os
import numpy as np
import re
current_path = os.getcwd()
#os.chdir('../../')
# os.path.join('../../')
from launch_abesol import launch_abesol
# from launch_abesol import launch_abesol
os.chdir(current_path)
lambda_vec =  np.array([7]) #np.linspace(1,10,10) # 
#vinf_vec  = [10] # np.linspace(5,25,5)
young_vec = [80e6] # ,80e6 #1e9/2**0 # 2e9/2**0 ###     /2**4, 1e9/10, 1e9/12] # np.divide(141.96e9,vinf_vec) 0.125/4
E_young = 141.96e9
rho_inf = 1.225
ct = []
cp = []
omega_vec = []
orient = []
perc_rad = 1#2/3
fol = '25r'
for young_val in young_vec:
    ct2 = []
    cp2 = []
    omega_vec2 = []
    for lambda_val in lambda_vec:
        with open('D:/Documentos/Doctorado/turbine_NREL_phaseiv/NREL_S809_structure'+fol+'/case_base_elast.abe', 'r') as file:
            vinf = np.sqrt(E_young/(1/2*rho_inf*young_val*(1+(perc_rad*lambda_val)**2))) # 141.96e9/young_val
            if len(orient) > 0:
                folder0 = 'D:/Documentos/Doctorado/turbine_NREL_phaseiv/NREL_S809_structure'+fol+'/E_adim_'+str(int(young_val*1e-6))+orient
                folder = folder0 + '/lambda_'+str(int(lambda_val))+orient
            else:
                folder0 = 'D:/Documentos/Doctorado/turbine_NREL_phaseiv/NREL_S809_structure'+fol+'/E_adim_'+str(int(young_val*1e-6))+'b'
                folder = folder0 + '/lambda_'+str(int(lambda_val))
            # data = file.read().replace('$FOLDER$', folder)
            data = file.read()
#            pattern = '(?<=BC_VINF=)(.*)'
#            result = re.search(pattern, data)
#            vinf = float(result.groups(0)[0]) 
            pattern = '(?<=BC_RADIUS=)(.*)'
            result = re.search(pattern, data)
            rad = float(result.groups(0)[0])
            pattern = '(?<=BC_RHO=)(.*)'
            result = re.search(pattern, data)
            rho = float(result.groups(0)[0])
            omega_val = lambda_val/rad*vinf
            timetot_val = 2 #np.min([1.5,4*np.pi/omega_val])
            print(timetot_val)
            data = data.replace('$TIMETOT$', str(timetot_val))
            data = data.replace('$OMEGA$', str(omega_val))
            data = data.replace('$VINF$', str(vinf))
        try:
            os.mkdir(folder0)
            os.mkdir(folder)
            os.chdir(folder)
            os.mkdir('results')
            os.mkdir('vibration')
            os.chdir(current_path)
        except:
            try:
                os.mkdir(folder)
                os.chdir(folder)
                os.mkdir('results')
                os.mkdir('vibration')
                os.chdir(current_path)
            except:
                pass
        file = "aerogenerador_"+str(int(young_val*1e-6))+'_'+str(int(lambda_val))+".abe"
        text_file = open(folder+'/'+file, "wt")
        n = text_file.write(data)
        text_file.close()
        solution = launch_abesol(folder,file,current_path)
        ct2.append(np.array(solution.CT)*(np.pi*rad**2*(omega_val*rad)**2*rho)/(1/2*np.pi*rad**2*(vinf)**2*rho))
        cp2.append(-np.array(solution.CP)*(np.pi*rad**2*(omega_val*rad)**3*rho)/(1/2*np.pi*rad**2*(vinf)**3*rho))
        omega_vec2.append(omega_val)
    ct.append(ct2)
    cp.append(cp2)
    omega_vec.append(omega_vec2)

        
        
