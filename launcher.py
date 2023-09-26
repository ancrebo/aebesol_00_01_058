# -*- coding: utf-8 -*-
"""
Created on Wed Nov 25 18:38:26 2020

@author: ancrebo
"""
import os
current_path = os.getcwd()
#os.chdir('../../')
# os.path.join('../../')
from launch_abesol import launch_abesol
# from launch_abesol import launch_abesol
os.chdir(current_path)
omega_vec =  [10,20,30,40,50]
for omega_val in omega_vec:
    with open('aerogenerador_pruebas/case_base.abe', 'r') as file:
        folder = 'aerogenerador_pruebas/omega_'+str(omega_val)+'m_s'
        # data = file.read().replace('$FOLDER$', folder)
        data = file.read().replace('$OMEGA$', str(omega_val))
    try:
        os.mkdir(folder)
        os.chdir(folder)
        os.mkdir('results')
        os.mkdir('vibration')
        os.chdir(current_path)
    except:
        pass
    file = "aerogenerador_"+str(omega_val)+"m_s.abe"
    text_file = open(folder+'/'+file, "wt")
    n = text_file.write(data)
    text_file.close()
    launch_abesol(folder,file,current_path)
    
