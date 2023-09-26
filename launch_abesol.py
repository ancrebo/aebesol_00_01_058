# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:41:02 2020

@author                 : Andres Cremades Botella - ancrebo@mot.upv.es
launch_abesol           : file calling functions to launch single simulation
last_version            : 05-03-2021
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
from bin.solver_elastic import solver_struct_stat, solver_struct_dyn, solver_vib_mod,solver_aero_stat
from bin.postprocess import postproc_stat, postproc_dyn, postproc_vib
from bin.read_section import read_section

print('ABESOL_00_01_032')    
#%% Laucher
def launch_abesol(folder,file,current_path):
    # Function to launch
    # folder           : path of the file
    # file             : name of the file
    # Close the open plots
    # baseroot         : baseroot to the folder
    # case_setup       : information of the setup
    # meshfile         : path of the mesh file
    # mesh_data        : information of the mesh
    # section          : information of the section mesh
    # sol_phys         : information of the solid physics
    # solution         : solution information
    # section_globalCS : coordinates of the section all the subsections
    plt.close('all')
    baseroot        = folder+'/'
    filecase        = baseroot + file
    case_setup      = read_case(filecase)
    case_setup.root = baseroot
    meshfile        = baseroot + case_setup.meshfile    
    mesh_data       = read_mesh(meshfile,case_setup)
    section,sol_phys, mesh_data = read_section(case_setup,mesh_data)
    exmass,exstiff              = read_exmass(baseroot,mesh_data)
    # If the structural static problem is selected
    if case_setup.problem_type == "STRUC_STAT":
        solution, section_globalCS = solver_struct_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
        postproc_stat(case_setup,mesh_data,solution, section_globalCS)
    # If the structural dynamic problem is selected
    elif case_setup.problem_type == "STRUC_DYN":
        if case_setup.ini_con == 1:
            [case_setup.ini_pos_data,case_setup.ini_vel_data,case_setup.ini_acel_data] = read_inicon(baseroot + case_setup.ini_con_data_file)
        solution,section_globalCS = solver_struct_dyn(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
        postproc_dyn(case_setup,mesh_data,solution,section_globalCS)
    # If the modal vibration problem is selected
    elif case_setup.problem_type == "VIB_MOD":
        solution, section_globalCS = solver_vib_mod(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
        postproc_vib(case_setup,mesh_data,solution,section_globalCS) 
    # If the aeroelastic dynamic problem is selected   
    elif case_setup.problem_type == "AEL_DYN":
        if case_setup.ini_con == 1:
            [case_setup.ini_pos_data,case_setup.ini_vel_data,case_setup.ini_acel_data] = read_inicon(baseroot + case_setup.ini_con_data_file)
        solution,section_globalCS = solver_struct_dyn(case_setup,sol_phys,mesh_data,section,exmass,exstiff)    
        postproc_dyn(case_setup,mesh_data,solution,section_globalCS)
    elif case_setup.problem_type == "AERO":
        solution, section_globalCS = solver_aero_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    # sys.path.remove(current_path)        
    return solution