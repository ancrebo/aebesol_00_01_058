# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:41:02 2020

@author                 : Andres Cremades Botella - ancrebo@mot.upv.es
functions_beam_element  : file calling functions to launch single simulation
last_version            : 17-02-2021
modified_by             : Andres Cremades Botella
"""
import os
current_path = os.getcwd()
os.chdir(current_path)
import sys 
sys.path.insert(0,current_path)
import numpy as np
import matplotlib.pyplot as plt
from bin.read_setup import read_case, read_inicon
from bin.read_mesh import read_mesh, read_exmass
from bin.solver_elastic import solver_struct_stat, solver_struct_dyn, solver_vib_mod, solver_vibdamp_mod,solver_aero_stat,solver_ael_stat,solver_ael_mod,solver_fly_stat,solver_fly_struc_stat,solver_fly_ael_stat
from bin.postprocess import postproc_stat, postproc_dyn, postproc_vib, postproc_aelmod, postproc_aero
#from bin.postproc import postproc_stat, postproc_dyn, postproc_vib
from bin.read_section import read_section
import time
#%% Laucher
plt.close('all')
folder                      = './flatplate/vel_5m_s'
baseroot                    = folder + '/' 
filecase                    = baseroot + 'case_base.abe'
case_setup                  = read_case(filecase)
case_setup.root             = baseroot
meshfile                    = baseroot + case_setup.meshfile
mesh_data                   = read_mesh(meshfile,case_setup)
section,sol_phys, mesh_data = read_section(case_setup,mesh_data)
exmass,exstiff              = read_exmass(baseroot,mesh_data)
# If the structural static problem is selected
if case_setup.problem_type == "STRUC_STAT":
    print('Starting steady...')
    solution, section_globalCS = solver_struct_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    print('... Ending steady')
    postproc_stat(case_setup,mesh_data,solution, section_globalCS)
# If the structural dynamic problem is selected
elif case_setup.problem_type == "STRUC_DYN":
    if case_setup.ini_con == 1:
        [case_setup.ini_pos_data,case_setup.ini_vel_data,case_setup.ini_acel_data] = read_inicon(baseroot + case_setup.ini_con_data_file)
    print('Starting transient...')
    solution,section_globalCS = solver_struct_dyn(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    print('... Ending transient')
    postproc_dyn(case_setup,mesh_data,solution,section_globalCS)
# If the modal vibration problem is selected
elif case_setup.problem_type == "VIB_MOD":
    print('Starting modal...')  
    solution, section_globalCS = solver_vibdamp_mod(case_setup,sol_phys,mesh_data,section,exmass,exstiff) 
    print('... Ending modal') 
    postproc_vib(case_setup,mesh_data,solution,section_globalCS) 
# If the aeroelastic dynamic problem is selected   
elif case_setup.problem_type == "AEL_STAT":
    print('Starting aeroelastic steady...')
    solution,section_globalCS = solver_ael_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    print('... Ending aeroelastic steady')
    postproc_stat(case_setup,mesh_data,solution, section_globalCS)
elif case_setup.problem_type == "AEL_DYN":
    if case_setup.ini_con == 1:
        [case_setup.ini_pos_data,case_setup.ini_vel_data,case_setup.ini_acel_data] = read_inicon(baseroot + case_setup.ini_con_data_file)
    solution,section_globalCS = solver_struct_dyn(case_setup,sol_phys,mesh_data,section,exmass,exstiff)    
    postproc_dyn(case_setup,mesh_data,solution,section_globalCS)
elif case_setup.problem_type == "AERO":
    print('Starting aerodynamic...')
    solution, section_globalCS = solver_aero_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    print('... Ending aerodynamic')
    postproc_aero(case_setup,mesh_data,solution, section_globalCS)
elif case_setup.problem_type == "AEL_MOD": 
    print('Starting aeroelastic modal...')  
    solution, section_globalCS = solver_ael_mod(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    print('... Ending aeroelastic modal') 
    postproc_aelmod(case_setup,mesh_data,solution,section_globalCS)
elif case_setup.problem_type == "FLY_STAT":
    print('Starting steady flight...')
    solution, section_globalCS = solver_fly_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    print('... Ending steady flight')
    postproc_aero(case_setup,mesh_data,solution, section_globalCS)
elif case_setup.problem_type == "FLY_STRUC_STAT":
    print('Starting steady flight structural deformation...')
    solution, section_globalCS = solver_fly_struc_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    print('... Ending steady flight structural deformation')
    postproc_stat(case_setup,mesh_data,solution, section_globalCS)
elif case_setup.problem_type == "FLY_AEL_STAT":
    print('Starting aeroelastic steady flight...')
    solution, section_globalCS = solver_fly_ael_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    print('... Ending aeroelastic steady flight')
    postproc_stat(case_setup,mesh_data,solution, section_globalCS)
    
sys.path.remove(current_path) 