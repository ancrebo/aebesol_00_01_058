# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:39:45 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
solver_elastic : file containing functions for solving the elastic problem
last_version   : 23-02-2021
modified_by    : Andres Cremades Botella
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from bin.global_matrix_definition import define_stiffness, define_mass, define_damp, def_secglob, define_exmassdist, define_exmasspoint,define_exstiffpoint,define_exstiffdist,def_secglob_aero
from bin.bc_functions import boundaryconditions, boundaryconditions_tran, init_boundaryconditions_tran,boundaryconditions_vibrest,boundaryconditions_aelmod, boundaryconditions_ael
from scipy import interpolate
from bin.linsys_solver import stat_linsys, dyn_linsys, dyn_linsys_mod, vib_linsys, vibae_linsys, vibaemod_linsys
from bin.store_data import store_solution_stat, store_solution_vib,store_solution_vibdamp, store_solution_aero,store_solution_vibae
from bin.save_data import save_stat, save_vib
import re
import copy
from bin.aux_functions import def_vec_param, filter_func

#%% Functions
def solver_aero_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff):
    # Function to solve the static structural problem
    # case_setup : setup information of the case
    # sol_phys   : solid physics information
    # mesh_data  : mesh of the problem
    # section    : data about the section
    # -------------------------------------------------------------------------
    # mesh_point      : nodes of the beam mesh
    # mesh_elem       : elements of the beam mesh
    # mesh_mark       : markers of the beam mesh (nodes in which the boundary conditions are applied)
    # num_elem        : number of elements of the beam mesh
    # num_point       : number of nodes of the beam mesh
    # FF_global_prima : forces of the system
    # qq_global_prima : variables/displacements of the system
    mesh_point      = mesh_data.point
    mesh_elem       = mesh_data.elem
    mesh_mark       = mesh_data.marker
    num_elem        = mesh_data.num_elem
    num_point       = mesh_data.num_point
    FF_global_prima = np.zeros((9*num_point,))
    qq_global_prima = np.nan*np.ones((9*num_point,))    
    # -------------------------------------------------------------------------
    # Definition of the stiffness matrix    
    KK_global_prima,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,LL_vec_p,rr_vec_p,rot_mat_p,rot_matT_p,sol_phys = define_stiffness(mesh_elem,sol_phys,mesh_point,num_point,num_elem,mesh_data.mesh_elem_lock,mesh_data.RR_ctrl,section)  
    mesh_data.LL = LL_vec
    mesh_data.rr = rr_vec
    mesh_data.rr_p = rr_vec_p
    MM_global_prima,mass_data           = define_mass(mesh_elem,sol_phys,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,section)
    MM_global_prima_exmass,mass_data2         = define_exmassdist(mesh_elem,exmass,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
    MM_global_prima_exmasspoint,mass_data3    = define_exmasspoint(exmass,mesh_point,num_point,Lrot_mat,LrotT_mat)
    MM_global_prima              += MM_global_prima_exmass+MM_global_prima_exmasspoint
    # -------------------------------------------------------------------------
    # Activate the information about the section internal nodes
    if mesh_data.section == "YES":    
        section_globalCS = def_secglob_aero(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point)                
    # The boundary conditions are applied on the matrices and vectors
    # KK_global   : stiffness matrix after applying the boundary conditions
    # qq_global   : displacement vector after applying the boundary conditions
    # FF_global   : load vector ater applying the boundary conditions                
    KK_global, qq_global, FF_global, FF_orig, RR_global, disp_values, mesh_data = boundaryconditions(KK_global_prima,MM_global_prima,FF_global_prima,qq_global_prima,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)
    # Solution of the static linear system
#    q_global, FF_global, q_vec = stat_linsys(KK_global,qq_global,FF_global,RR_global)
    # store solutions
    solution = store_solution_aero(qq_global,FF_orig,mesh_point,mesh_elem,section,sol_phys,rot_mat,rotT_mat, disp_values)
    solution.mass = mass_data
    # Check if a solution file must be created
    if case_setup.savefile == 'YES':
        save_stat(case_setup,mesh_data,section_globalCS,solution)     
    if case_setup.grav_fl == 1:
        weight = 0
        for ii_point in np.arange(mesh_data.num_point):
            weight_vec = 0
            for jj_point in np.arange(3):
                weight_vec += (case_setup.grav_g[jj_point]*MM_global_prima[ii_point*9+jj_point,ii_point*9+jj_point])**2
            weight += np.sqrt(weight_vec)
        solution.weight = weight
    return solution, section_globalCS
#%% Functions
def solver_struct_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff):
    # Function to solve the static structural problem
    # case_setup : setup information of the case
    # sol_phys   : solid physics information
    # mesh_data  : mesh of the problem
    # section    : data about the section
    # -------------------------------------------------------------------------
    # mesh_point      : nodes of the beam mesh
    # mesh_elem       : elements of the beam mesh
    # mesh_mark       : markers of the beam mesh (nodes in which the boundary conditions are applied)
    # num_elem        : number of elements of the beam mesh
    # num_point       : number of nodes of the beam mesh
    # FF_global_prima : forces of the system
    # qq_global_prima : variables/displacements of the system
    mesh_point      = mesh_data.point
    mesh_elem       = mesh_data.elem
    mesh_mark       = mesh_data.marker
    num_elem        = mesh_data.num_elem
    num_point       = mesh_data.num_point
    FF_global_prima = np.zeros((9*num_point,))
    qq_global_prima = np.nan*np.ones((9*num_point,))    
    # -------------------------------------------------------------------------
    # Definition of the stiffness matrix    
    KK_global_prima,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,LL_vec_p,rr_vec_p,rot_mat_p,rot_matT_p,sol_phys = define_stiffness(mesh_elem,sol_phys,mesh_point,num_point,num_elem,mesh_data.mesh_elem_lock,mesh_data.RR_ctrl,section)  
    mesh_data.LL = LL_vec
    mesh_data.rr = rr_vec
    mesh_data.rr_p = rr_vec_p
    KK_global_prima_exstiffpoint  = define_exstiffpoint(exstiff,mesh_point,mesh_elem,num_point,Lrot_mat,LrotT_mat)
    KK_global_prima_exstiffelem   = define_exstiffdist(mesh_elem,exstiff,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
    KK_global_prima              += KK_global_prima_exstiffpoint + KK_global_prima_exstiffelem
    MM_global_prima,mass_data1     = define_mass(mesh_elem,sol_phys,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,section)
    MM_global_prima_exmass,mass_data2         = define_exmassdist(mesh_elem,exmass,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
    MM_global_prima_exmasspoint,mass_data3    = define_exmasspoint(exmass,mesh_point,num_point,Lrot_mat,LrotT_mat)
    MM_global_prima              += MM_global_prima_exmass+MM_global_prima_exmasspoint
    # -------------------------------------------------------------------------
    # Activate the information about the section internal nodes
    if mesh_data.section == "YES":    
        section_globalCS = def_secglob(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point)                
    # The boundary conditions are applied on the matrices and vectors
    # KK_global   : stiffness matrix after applying the boundary conditions
    # qq_global   : displacement vector after applying the boundary conditions
    # FF_global   : load vector ater applying the boundary conditions                
    KK_global, qq_global, FF_global, FF_orig, RR_global, disp_values, mesh_data = boundaryconditions(KK_global_prima,MM_global_prima,FF_global_prima,qq_global_prima,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)
    # Solution of the static linear system
    qq_global, FF_global, q_vec = stat_linsys(KK_global,qq_global,FF_global,RR_global)
    # store solutions
    solution = store_solution_stat(qq_global,FF_global,mesh_point,mesh_elem,section,sol_phys,rot_mat,rotT_mat,q_vec, disp_values)
    solution.mass = mass_data1+mass_data2+mass_data3
    # Check if a solution file must be created
    if case_setup.savefile == 'YES':
        save_stat(case_setup,mesh_data,section_globalCS,solution) 
    if case_setup.grav_fl == 1:
        weight = 0
        for ii_point in np.arange(mesh_data.num_point):
            weight_vec = 0
            for jj_point in np.arange(3):
                weight_vec += (case_setup.grav_g[jj_point]*MM_global_prima[ii_point*9+jj_point,ii_point*9+jj_point])**2
            weight += np.sqrt(weight_vec)
        solution.weight = weight
    return solution, section_globalCS

#%%
def solver_struct_dyn(case_setup,sol_phys,mesh_data,section,exmass,exstiff):
    # Function to solve the dynamic structural problem
    # case_setup : setup information of the case
    # sol_phys   : solid physics information
    # mesh_data  : mesh of the problem
    # section    : data about the section
    # exmass     : extra mass
    # -----------------------------------------------------------------------------
    # mesh_point : nodes of the mesh
    # mesh_elem  : elements of the mesh
    # mesh_mark  : markers of the mesh, boundaries to apply conditions and nodes to visualize information
    # num_elem   : number of beam elements
    # num_point  : number of beam nodes
    mesh_point = mesh_data.point
    mesh_elem  = mesh_data.elem
    mesh_mark  = mesh_data.marker
    num_elem   = mesh_data.num_elem
    num_point  = mesh_data.num_point
    # ----------------------------------------------------------------------------
    # u_val           : value of the displacement in x axis
    # v_val           : value of the displacement in y axis
    # w_val           : value of the displacement in z axis
    # phi_val         : value of the rotation in x axis
    # psi_val         : value of the rotation in y axis
    # theta_val       : value of the rotation in z axis
    # u_dt_val        : value of the velocity in x axis
    # v_dt_val        : value of the velocity in y axis
    # w_dt_val        : value of the velocity in z axis
    # u_dtdt_val      : value of the acceleration in x axis
    # v_dtdt_val      : value of the acceleration in y axis
    # w_dtdt_val      : value of the acceleration in z axis
    # u_dtdtdt_val    : value of the celerity in x axis
    # v_dtdtdt_val    : value of the celerity in y axis
    # w_dtdtdt_val    : value of the celerity in z axis
    # phi_dt_val      : value of the angular velocity in x axis
    # psi_dt_val      : value of the angular velocity in y axis
    # theta_dt_val    : value of the angular velocity in z axis
    # phi_dtdt_val    : value of the angular acceleration in x axis
    # psi_dtdt_val    : value of the angular acceleration in y axis
    # theta_dtdt_val  : value of the angular acceleration in z axis
    u_val          = np.zeros((num_point,))
    v_val          = np.zeros((num_point,))
    w_val          = np.zeros((num_point,))
    phi_val        = np.zeros((num_point,))
    psi_val        = np.zeros((num_point,))
    theta_val      = np.zeros((num_point,))
    u_dt_val       = np.zeros((num_point,))
    v_dt_val       = np.zeros((num_point,))
    w_dt_val       = np.zeros((num_point,))
    u_dtdt_val     = np.zeros((num_point,))
    v_dtdt_val     = np.zeros((num_point,))
    w_dtdt_val     = np.zeros((num_point,))
    u_dtdtdt_val   = np.zeros((num_point,))
    v_dtdtdt_val   = np.zeros((num_point,))
    w_dtdtdt_val   = np.zeros((num_point,))
    phi_dt_val     = np.zeros((num_point,))
    psi_dt_val     = np.zeros((num_point,))
    theta_dt_val   = np.zeros((num_point,))
    phi_d_val      = np.zeros((num_point,))
    psi_d_val      = np.zeros((num_point,))
    theta_d_val    = np.zeros((num_point,))
    phi_dtdt_val   = np.zeros((num_point,))
    psi_dtdt_val   = np.zeros((num_point,))
    theta_dtdt_val = np.zeros((num_point,))
    # init_it       : initial iterations
    # time_tot      : total time of calculation
    # time_stp      : time step of the calculation
    # time_stp_op   : inner time step of the calculation
    # ii_time_save  : index of the saved magnitude in time
    # num_stp       : number of time steps
    # time_vec      : time vector
    # time_vec_save : vector to save the time steps with the required saving frequency
    # pos_time      : matrix to store the displacement of the beam nodes in time
    init_it       = case_setup.init_it
    time_tot      = case_setup.tot_time
    time_stp      = case_setup.time_stp
    time_stp_op   = case_setup.time_stp
    ii_time_save  = 0
    num_stp       = int(time_tot/time_stp)+1
    time_vec      = np.linspace(0,time_tot,num_stp)
    time_vec_save = []
    pos_time      = np.zeros((len(mesh_point),len(mesh_point[0]),num_stp))
    # If the number of time steps is higher than 10, it is fixed to 10 for the internal vector
    if num_stp >= 10:
        num_stp = 10
    # FF_global_prima        : load vector before applying boundary conditions
    # qq_global_prima        : displacement vector before applying boundary conditions
    # qq_der_global_prima    : velocity vector before applying boundary conditions
    # qq_derder_global_prima : acceleration vector before applying boundary conditions
    FF_global_prima        = np.zeros((9*num_point,num_stp))
    qq_global_prima        = np.nan*np.ones((9*num_point,num_stp))
    qq_der_global_prima    = np.nan*np.ones((9*num_point,num_stp))
    qq_derder_global_prima = np.zeros((9*num_point,num_stp))
    # Check if the initial conditions are provided by a file
    if case_setup.ini_con == 1:
        # Read the initial conditions from file
        qq_global_prima[:,0]        = case_setup.ini_pos_data
        qq_der_global_prima[:,0]    = case_setup.ini_vel_data
        qq_derder_global_prima[:,0] = case_setup.ini_acel_data
    else:
        # Set the initial conditions to zero
        qq_global_prima[:,0]        = np.zeros((9*num_point,))
        qq_der_global_prima[:,0]    = np.zeros((9*num_point,))
        qq_derder_global_prima[:,0] = np.zeros((9*num_point,))
    # fs                  : sampling frequency
    # dB                  : reduction of the filter response in the maximum frequency
    # width               : width of the filter window
    # nyq                 : nyquist frequency of the filter
    # NN                  : length of the window of the kaiserord
    # beta                : beta parameter of the kaiserord filter
    # fmax                : maximum frequency of the filter
    # delay               : delay of the phase in the filtered signal
    # Nvec1               : number of time steps required for initializing the filter
    # Nvec2               : number of time steps required for reset the filtered timesteps
    # Nvec3               : length of the filtered vectors
    # u_val_filter        : value of the filtered displacement in x axis
    # v_val_filter        : value of the filtered displacement in y axis
    # w_val_filter        : value of the filtered displacement in z axis
    # phi_val_filter      : value of the filtered rotation in x axis
    # psi_val_filter      : value of the filtered rotation in y axis
    # theta_val_filter    : value of the filtered rotation in z axis
    # ud_val_filter       : value of the filtered velocity in x axis
    # vd_val_filter       : value of the filtered velocity in y axis
    # wd_val_filter       : value of the filtered velocity in z axis
    # udd_val_filter      : value of the filtered acceleration in x axis
    # vdd_val_filter      : value of the filtered acceleration in y axis
    # wdd_val_filter      : value of the filtered acceleration in z axis
    # uddd_val_filter     : value of the filtered celerity in x axis
    # vddd_val_filter     : value of the filtered celerity in y axis
    # wddd_val_filter     : value of the filtered celerity in z axis
    # phid_val_filter     : value of the filtered angular velocity in x axis
    # psid_val_filter     : value of the filtered angular velocity in y axis
    # thetad_val_filter   : value of the filtered angular velocity in z axis
    # phidd_val_filter    : value of the filtered angular acceleration in x axis
    # psidd_val_filter    : value of the filtered angular acceleration in y axis
    # thetadd_val_filter  : value of the filtered angular acceleration in z axis
    # u_vec_filter        : vector of the filtered displacement in x axis
    # v_vec_filter        : vector of the filtered displacement in y axis
    # w_vec_filter        : vector of the filtered displacement in z axis
    # phi_vec_filter      : vector of the filtered rotation in x axis
    # psi_vec_filter      : vector of the filtered rotation in y axis
    # theta_vec_filter    : vector of the filtered rotation in z axis
    # ud_vec_filter       : vector of the filtered velocity in x axis
    # vd_vec_filter       : vector of the filtered velocity in y axis
    # wd_vec_filter       : vector of the filtered velocity in z axis
    # udd_vec_filter      : vector of the filtered acceleration in x axis
    # vdd_vec_filter      : vector of the filtered acceleration in y axis
    # wdd_vec_filter      : vector of the filtered acceleration in z axis
    # uddd_vec_filter     : vector of the filtered celerity in x axis
    # vddd_vec_filter     : vector of the filtered celerity in y axis
    # wddd_vec_filter     : vector of the filtered celerity in z axis
    # phid_vec_filter     : vector of the filtered angular velocity in x axis
    # psid_vec_filter     : vector of the filtered angular velocity in y axis
    # thetad_vec_filter   : vector of the filtered angular velocity in z axis
    # phidd_vec_filter    : vector of the filtered angular acceleration in x axis
    # psidd_vec_filter    : vector of the filtered angular acceleration in y axis
    # thetadd_vec_filter  : vector of the filtered angular acceleration in z axis
    # u_vec_filter2       : vector of the filtered displacement in x axis
    # v_vec_filter2       : vector of the filtered displacement in y axis
    # w_vec_filter2       : vector of the filtered displacement in z axis
    # phi_vec_filter2     : vector of the filtered rotation in x axis
    # psi_vec_filter2     : vector of the filtered rotation in y axis
    # theta_vec_filter2   : vector of the filtered rotation in z axis
    # ud_vec_filter2      : vector of the filtered velocity in x axis
    # vd_vec_filter2      : vector of the filtered velocity in y axis
    # wd_vec_filter2      : vector of the filtered velocity in z axis
    # udd_vec_filter2     : vector of the filtered acceleration in x axis
    # vdd_vec_filter2     : vector of the filtered acceleration in y axis
    # wdd_vec_filter2     : vector of the filtered acceleration in z axis
    # uddd_vec_filter2    : vector of the filtered celerity in x axis
    # vddd_vec_filter2    : vector of the filtered celerity in y axis
    # wddd_vec_filter2    : vector of the filtered celerity in z axis
    # phid_vec_filter2    : vector of the filtered angular velocity in x axis
    # psid_vec_filter2    : vector of the filtered angular velocity in y axis
    # thetad_vec_filter2  : vector of the filtered angular velocity in z axis
    # phidd_vec_filter2   : vector of the filtered angular acceleration in x axis
    # psidd_vec_filter2   : vector of the filtered angular acceleration in y axis
    # thetadd_vec_filter2 : vector of the filtered angular acceleration in z axis
    if case_setup.filt == 'YES':
        fs                  = 1/time_stp
        dB                  = case_setup.filt_db
        width               = case_setup.filt_wid
        nyq                 = fs/2        
        NN, beta            = signal.kaiserord(dB, width / nyq)
        fmax                = case_setup.filt_freq
        delay               = 0.5*(NN-1)/fs 
        Nvec1               = int(delay/time_stp)
        Nvec2               = 2*Nvec1
        Nvec3               = 3*Nvec1
        u_val_filter        = u_val
        ud_val_filter       = u_dt_val
        udd_val_filter      = u_dtdt_val
        uddd_val_filter     = u_dtdtdt_val
        v_val_filter        = v_val
        vd_val_filter       = v_dt_val
        vdd_val_filter      = v_dtdt_val
        vddd_val_filter     = v_dtdtdt_val
        w_val_filter        = w_val
        wd_val_filter       = w_dt_val
        wdd_val_filter      = w_dtdt_val
        wddd_val_filter     = w_dtdtdt_val
        phi_val_filter      = phi_val
        phid_val_filter     = phi_dt_val
        phidd_val_filter    = phi_dtdt_val
        psi_val_filter      = psi_val
        psid_val_filter     = psi_dt_val
        psidd_val_filter    = psi_dtdt_val
        theta_val_filter    = theta_val
        thetad_val_filter   = theta_dt_val
        thetadd_val_filter  = theta_dtdt_val        
        u_vec_filter        = np.zeros((Nvec3,num_point))
        ud_vec_filter       = np.zeros((Nvec3,num_point))
        udd_vec_filter      = np.zeros((Nvec3,num_point))
        uddd_vec_filter     = np.zeros((Nvec3,num_point))
        v_vec_filter        = np.zeros((Nvec3,num_point))
        vd_vec_filter       = np.zeros((Nvec3,num_point))
        vdd_vec_filter      = np.zeros((Nvec3,num_point))
        vddd_vec_filter     = np.zeros((Nvec3,num_point))
        w_vec_filter        = np.zeros((Nvec3,num_point))
        wd_vec_filter       = np.zeros((Nvec3,num_point))
        wdd_vec_filter      = np.zeros((Nvec3,num_point))
        wddd_vec_filter     = np.zeros((Nvec3,num_point))
        phi_vec_filter      = np.zeros((Nvec3,num_point))
        phid_vec_filter     = np.zeros((Nvec3,num_point))
        phidd_vec_filter    = np.zeros((Nvec3,num_point))
        psi_vec_filter      = np.zeros((Nvec3,num_point))
        psid_vec_filter     = np.zeros((Nvec3,num_point))
        psidd_vec_filter    = np.zeros((Nvec3,num_point))
        theta_vec_filter    = np.zeros((Nvec3,num_point))
        thetad_vec_filter   = np.zeros((Nvec3,num_point))
        thetadd_vec_filter  = np.zeros((Nvec3,num_point))
        u_vec_filter2       = np.zeros((Nvec3,num_point))
        ud_vec_filter2      = np.zeros((Nvec3,num_point))
        udd_vec_filter2     = np.zeros((Nvec3,num_point))
        uddd_vec_filter2    = np.zeros((Nvec3,num_point))
        v_vec_filter2       = np.zeros((Nvec3,num_point))
        vd_vec_filter2      = np.zeros((Nvec3,num_point))
        vdd_vec_filter2     = np.zeros((Nvec3,num_point))
        vddd_vec_filter2    = np.zeros((Nvec3,num_point))
        w_vec_filter2       = np.zeros((Nvec3,num_point))
        wd_vec_filter2      = np.zeros((Nvec3,num_point))
        wdd_vec_filter2     = np.zeros((Nvec3,num_point))
        wddd_vec_filter2    = np.zeros((Nvec3,num_point))
        phi_vec_filter2     = np.zeros((Nvec3,num_point))
        phid_vec_filter2    = np.zeros((Nvec3,num_point))
        phidd_vec_filter2   = np.zeros((Nvec3,num_point))
        psi_vec_filter2     = np.zeros((Nvec3,num_point))
        psid_vec_filter2    = np.zeros((Nvec3,num_point))
        psidd_vec_filter2   = np.zeros((Nvec3,num_point))
        theta_vec_filter2   = np.zeros((Nvec3,num_point))
        thetad_vec_filter2  = np.zeros((Nvec3,num_point))
        thetadd_vec_filter2 = np.zeros((Nvec3,num_point))
    # ii_filter   : index of the filtered vector
    # yy_L        : vector of the modal contribution
    # yy_der_L    : vector of the derivative of the modal contribution
    # yy_derder_L : vector of the second derivative of the modal contribution
    ii_filter   = 0
    yy_L        = np.zeros((case_setup.n_mod,))
    yy_der_L    = np.zeros((case_setup.n_mod,))
    yy_derder_L = np.zeros((case_setup.n_mod,))   
    # -------------------------------------------------------------------------
    # Definition of the stiffness matrix 
    # KK_global_prima : stiffness matrix before applying the joints
    # MM_global_prima : mass matrix before applying the joints
    # LL_vec          : distance between nodes vector
    # rr_vec          : vectorial distance between nodes
    # rot_mat         : rotation matrix of each beam element
    # rotT_mat        : transposed rotation matrix of each beam element
    # Lrot_mat        : complete rotation matrix of the system
    # LrotT_mat       : complete transposed rotation matrix of the system
    # LL_vec_p        : distance between nodes vector for each node in the element
    # rr_vec_p        : vectorial distance between nodes for each node in the element
    # rot_mat_p       : rotation matrix of each beam element for each node in the element
    # rotT_mat_p      : transposed rotation matrix of each beam element for each node in the element
    KK_global_prima,LL_vec,rr_vec,rot_mat,rotT_mat,\
        Lrot_mat,LrotT_mat,LL_vec_p,rr_vec_p,rot_mat_p,rot_matT_p,sol_phys  = define_stiffness(mesh_elem,sol_phys,mesh_point,num_point,num_elem,mesh_data.mesh_elem_lock,mesh_data.RR_ctrl,section)
    KK_global_prima_exstiffpoint                                   = define_exstiffpoint(exstiff,mesh_point,mesh_elem,num_point,Lrot_mat,LrotT_mat)
    KK_global_prima_exstiffelem                                    = define_exstiffdist(mesh_elem,exstiff,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
    KK_global_prima                                               += KK_global_prima_exstiffpoint + KK_global_prima_exstiffelem
    mesh_data.LL                                                   = LL_vec 
    mesh_data.rr                                                   = rr_vec
    mesh_data.rr_p                                                 = rr_vec_p
    MM_global_prima,mass_data1                                      = define_mass(mesh_elem,sol_phys,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,section)
    MM_global_prima_exmass,mass_data2                                         = define_exmassdist(mesh_elem,exmass,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
    MM_global_prima_exmasspoint,mass_data3                                     = define_exmasspoint(exmass,mesh_point,num_point,Lrot_mat,LrotT_mat)
#    MM_global_prima                                               += MM_global_prima_exmass+MM_global_prima_exmasspoint
#    KK_global, MM_global, RR_global, case_setup, ref_axis  = init_boundaryconditions_tran(KK_global_prima,MM_global_prima,[],mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)
    # -------------------------------------------------------------------------
    # Activate the information about the section internal nodes
    if mesh_data.section == "YES":   
        # section_globalCS  : section nodes in global coordinate system
        section_globalCS = def_secglob(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point) 
    KK_global, MM_global, RR_global, case_setup, ref_axis  = init_boundaryconditions_tran(KK_global_prima,MM_global_prima,[],mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)     
    # -------------------------------------------------------------------------
    # Definition of the damping matrix
    # CC_global_prima : global damping matrix
    # vibfree         : information of the free vibration results
    # tperiod         : first mode period
    # tmin_ann        : minimum time to apply artificial neural network
    # ttran_ann       : time to apply a transition between the cfd polar and the artificial neural network
    # int_per         : number of iteration in 1 period of the first mode
    vibfree, CC_global = define_damp(case_setup,sol_phys,mesh_data,section,MM_global,KK_global,solver_vib_mod,solver_struct_stat,num_point,mesh_mark)
    tperiod = 1/(vibfree.freq_mode[0])
    if case_setup.filt == 'YES': 
        tmin_ann  = tperiod   
        ttran_ann = np.max([1.1,case_setup.filt_ttran])*tmin_ann 
    else:
        tmin_ann  = time_stp    
        ttran_ann = np.max([1.1,case_setup.filt_ttran])*tmin_ann 
    int_per            = int(tperiod/time_stp)  
    clsec_vec  = def_vec_param(num_point)
    cdsec_vec  = def_vec_param(num_point)
    cmsec_vec  = def_vec_param(num_point)
    # u_vec            : displacement x axis
    # udt_vec          : velocity x axis
    # udtdt_vec        : acceleration x axis
    # u_filter_vec     : filtered displacement x axis
    # udt_filter_vec   : filtered velocity x axis
    # udtdt_filter_vec : filtered acceleration x axis
    u_vec                 = def_vec_param(num_point)
    udt_vec               = def_vec_param(num_point)
    udtdt_vec             = def_vec_param(num_point) 
    u_filter_vec          = def_vec_param(num_point) 
    udt_filter_vec        = def_vec_param(num_point)
    udtdt_filter_vec      = def_vec_param(num_point)
    # v_vec            : displacement y axis
    # vdt_vec          : velocity y axis
    # vdtdt_vec        : acceleration y axis
    # v_filter_vec     : filtered displacement y axis
    # vdt_filter_vec   : filtered velocity y axis
    # vdtdt_filter_vec : filtered acceleration y axis
    v_vec                 = def_vec_param(num_point)
    vdt_vec               = def_vec_param(num_point)
    vdtdt_vec             = def_vec_param(num_point)
    v_filter_vec          = def_vec_param(num_point)
    vdt_filter_vec        = def_vec_param(num_point)
    vdtdt_filter_vec      = def_vec_param(num_point)
    # w_vec            : displacement z axis
    # wdt_vec          : velocity z axis
    # wdtdt_vec        : acceleration z axis
    # w_filter_vec     : filtered displacement z axis
    # wdt_filter_vec   : filtered velocity z axis
    # wdtdt_filter_vec : filtered acceleration z axis
    w_vec                 = def_vec_param(num_point)
    wdt_vec               = def_vec_param(num_point)
    wdtdt_vec             = def_vec_param(num_point)
    w_filter_vec          = def_vec_param(num_point)
    wdt_filter_vec        = def_vec_param(num_point)
    wdtdt_filter_vec      = def_vec_param(num_point)
    # phi_vec            : angular displacement x axis
    # phidt_vec          : angular velocity x axis
    # phidtdt_vec        : angular acceleration x axis
    # phi_filter_vec     : filtered angular displacement x axis
    # phidt_filter_vec   : filtered angular velocity x axis
    # phidtdt_filter_vec : filtered angular acceleration x axis
    phi_vec               = def_vec_param(num_point)
    phidt_vec             = def_vec_param(num_point)
    phidtdt_vec           = def_vec_param(num_point)
    phi_filter_vec        = def_vec_param(num_point)
    phidt_filter_vec      = def_vec_param(num_point)
    phidtdt_filter_vec    = def_vec_param(num_point)
    # psi_vec            : angular displacement y axis
    # psidt_vec          : angular velocity y axis
    # psidtdt_vec        : angular acceleration y axis
    # psi_filter_vec     : filtered angular displacement y axis
    # psidt_filter_vec   : filtered angular velocity y axis
    # psidtdt_filter_vec : filtered angular acceleration y axis
    psi_vec               = def_vec_param(num_point)
    psidt_vec             = def_vec_param(num_point)
    psidtdt_vec           = def_vec_param(num_point)
    psi_filter_vec        = def_vec_param(num_point)
    psidt_filter_vec      = def_vec_param(num_point)
    psidtdt_filter_vec    = def_vec_param(num_point)
    # theta_vec            : angular displacement z axis
    # thetadt_vec          : angular velocity z axis
    # thetadtdt_vec        : angular acceleration z axis
    # theta_filter_vec     : filtered angular displacement z axis
    # thetadt_filter_vec   : filtered angular velocity z axis
    # thetadtdt_filter_vec : filtered angular acceleration z axis
    theta_vec             = def_vec_param(num_point)
    thetadt_vec           = def_vec_param(num_point)
    thetadtdt_vec         = def_vec_param(num_point)
    theta_filter_vec      = def_vec_param(num_point)
    thetadt_filter_vec    = def_vec_param(num_point)
    thetadtdt_filter_vec  = def_vec_param(num_point)
    # aoa_vec            : angle of attack
    # aoadt_vec          : angle of attack derivative
    # aoadtdt_vec        : angle of attack second derivative
    aoa_vec               = def_vec_param(num_point)
    aoadt_vec             = def_vec_param(num_point)
    aoadtdt_vec           = def_vec_param(num_point)
    # phi_d_vec    : space derivative of the angle in the x axis
    # psi_d_vec    : space derivative of the angle in the y axis
    # theta_d_vec  : space derivative of the angle in the z axis
    phi_d_vec             = def_vec_param(num_point)
    psi_d_vec             = def_vec_param(num_point)
    theta_d_vec           = def_vec_param(num_point)
    # fx_vec : force x axis
    # fy_vec : force y axis
    # fz_vec : force z axis
    # Mx_vec : moment x axis
    # My_vec : moment y axis
    # Mz_vec : moment z axis
    # Bx_vec : bimoment x axis
    # By_vec : bimoment y axis
    # Bz_vec : bimoment z axis
    fx_vec = def_vec_param(num_point)
    fy_vec = def_vec_param(num_point)
    fz_vec = def_vec_param(num_point)
    Mx_vec = def_vec_param(num_point)
    My_vec = def_vec_param(num_point)
    Mz_vec = def_vec_param(num_point)
    Bx_vec = def_vec_param(num_point)
    By_vec = def_vec_param(num_point)
    Bz_vec = def_vec_param(num_point)  
    # warp_x1_vec : warping in x axis of node 1  
    # warp_y1_vec : warping in y axis of node 1
    # warp_z1_vec : warping in z axis of node 1 
    # warp_x2_vec : warping in x axis of node 2  
    # warp_y2_vec : warping in y axis of node 2
    # warp_z2_vec : warping in z axis of node 2       
    warp_x1_vec = []
    warp_y1_vec = []
    warp_z1_vec = [] 
    warp_x2_vec = []
    warp_y2_vec = []
    warp_z2_vec = []
    # cl_vec        : section lift coefficient
    # cd_vec        : section drag coefficient
    # cm_vec        : section pitch moment coefficient
    # ct_vec        : thrust coefficient
    # cq_vec        : moment coefficient
    # cp_vec        : power coefficient
    # pe_vec        : propulsive efficiency coefficient
    # yy_vel        : modal contribution
    # yy_der_vel    : derivative of the modal contribution
    # yy_derder_vel : second derivative of the modal contribution
    # CL_val        : total lift coefficient value
    # CD_val        : total drag coefficient value
    # CM_val        : total pitch moment coefficient value
    cl_vec        = []
    cd_vec        = []
    cm_vec        = []
    ct_vec        = []
    cq_vec        = []
    cp_vec        = []
    pe_vec        = []
    yy_vec        = def_vec_param(case_setup.n_mod)
    yy_der_vec    = def_vec_param(case_setup.n_mod)
    yy_derder_vec = def_vec_param(case_setup.n_mod)
    CL_val        = 0
    CD_val        = 0
    CM_val        = 0
    # ii_time    : index of the time iteration refered to a subdivision of the time domain
    # flg_iitime : flag to specify if the code has iterated at least 10 times
    ii_time    = 0
    flg_iitime = 0
    # disp_values  : class to store all the displacements values
    class disp_values:
        aoa_store0  = [np.zeros((num_point,))]
        aoa_store1  = [np.zeros((num_point,))]
        ang_store0  = [np.zeros((num_point,))]
        ang_store1  = [np.zeros((num_point,))]
        angder_store0  = [np.zeros((num_point,))]
        angder_store1  = [np.zeros((num_point,))]
        angderder_store0  = [np.zeros((num_point,))]
        angderder_store1  = [np.zeros((num_point,))]
        dis_store0  = [np.zeros((num_point,))]
        dis_store1  = [np.zeros((num_point,))]
        disder_store0  = [np.zeros((num_point,))]
        disder_store1  = [np.zeros((num_point,))]
        disderder_store0  = [np.zeros((num_point,))]
        disderder_store1  = [np.zeros((num_point,))]
        aoaflap_store0  = [np.zeros((num_point,))]
        aoaflap_store1  = [np.zeros((num_point,))]
        angflap_store0  = [np.zeros((num_point,))]
        angflap_store1  = [np.zeros((num_point,))]
        disflap_store0  = [np.zeros((num_point,))]
        disflap_store1  = [np.zeros((num_point,))]
        CL_val      = []
        CD_val      = []
        CM_val      = []
        CT_val      = []
        CQ_val      = []
        CP_val      = []
        PE_val      = []
    class sys_prev:
        pass
    # for all the time steps in the time domain
    for ii_time0 in np.arange(len(time_vec)):
        warpx1dat = def_vec_param(num_elem)
        warpy1dat = def_vec_param(num_elem)
        warpz1dat = def_vec_param(num_elem) 
        warpx2dat = def_vec_param(num_elem)
        warpy2dat = def_vec_param(num_elem)
        warpz2dat = def_vec_param(num_elem) 
        # ref_axis   : initialization of the reference axis
        # start_cputime : starting the cpu time of the iteration
        ref_axis      = []
        start_cputime = time.process_time()
        # advance subdomain of time. The objective is to reduce the RAM required by the matrices and speed up the code
        if (ii_time0 % 10 == 0 or (ii_time0 % 5 == 0 and flg_iitime == 1)) and ii_time0 > 0:
            # Ensure that the code has iterated at least 10 times
            # and reduce the value of the internal index to the midpoint
            flg_iitime = 1
            ii_time    = 5
            # advance the unknowns and the forces
            qq_global[:,:5]        = qq_global[:,5:]
            qq_global[:,5:]        = np.nan*qq_global[:,5:]
            qq_der_global[:,:5]    = qq_der_global[:,5:]
            qq_der_global[:,5:]    = np.nan*qq_der_global[:,5:]
            qq_derder_global[:,:5] = qq_derder_global[:,5:]
            qq_derder_global[:,5:] = np.nan*qq_derder_global[:,5:]
            FF_global[:,:5]        = FF_global[:,5:]
            FF_global[:,5:]        = np.zeros((len(FF_global),5))
        elif ii_time0>0:
            ii_time += 1
        # If the time step is not the first one, calculate, if not initialize boundary conditions
        # Fill the structure of displacements
        # If the filter ofption is activated fill the class with the filtered values
        if ii_time0 == 0:
            if case_setup.filt == 'YES':
                disp_values.u          = u_val_filter2 
                disp_values.v          = v_val_filter2 
                disp_values.w          = w_val_filter2
                disp_values.u_dt       = ud_val_filter2
                disp_values.v_dt       = vd_val_filter2
                disp_values.w_dt       = wd_val_filter2
                disp_values.u_dtdt     = udd_val_filter2
                disp_values.u_dtdt     = vdd_val_filter2
                disp_values.u_dtdt     = wdd_val_filter2
                disp_values.u_dtdtdt   = uddd_val_filter2
                disp_values.u_dtdtdt   = vddd_val_filter2
                disp_values.u_dtdtdt   = wddd_val_filter2
                disp_values.phi        = phi_val_filter2
                disp_values.psi        = psi_val_filter2 
                disp_values.theta      = theta_val_filter2 
                disp_values.phi_dt     = phid_val_filter2
                disp_values.psi_dt     = psid_val_filter2
                disp_values.theta_dt   = thetad_val_filter2
                disp_values.phi_dtdt   = phidd_val_filter2
                disp_values.psi_dtdt   = psidd_val_filter2
                disp_values.theta_dtdt = thetadd_val_filter2
            else:
                disp_values.u          = u_val
                disp_values.v          = v_val 
                disp_values.w          = w_val
                disp_values.u_dt       = u_dt_val
                disp_values.v_dt       = v_dt_val
                disp_values.w_dt       = w_dt_val
                disp_values.u_dtdt     = u_dtdt_val
                disp_values.u_dtdt     = v_dtdt_val
                disp_values.u_dtdt     = w_dtdt_val
                disp_values.u_dtdtdt   = u_dtdtdt_val
                disp_values.u_dtdtdt   = v_dtdtdt_val
                disp_values.u_dtdtdt   = w_dtdtdt_val
                disp_values.phi        = phi_val
                disp_values.psi        = psi_val
                disp_values.theta      = theta_val
                disp_values.phi_dt     = phi_dt_val
                disp_values.psi_dt     = psi_dt_val
                disp_values.theta_dt   = theta_dt_val
                disp_values.phi_dtdt   = phi_dtdt_val
                disp_values.psi_dtdt   = psi_dtdt_val
                disp_values.theta_dtdt = theta_dtdt_val
        # If the calculation time is not the initial
        if ii_time0 > 0:
            # Obtain stiffness, mass, damping matrix, displacements,velocities, accelertions and loads matrix and vectors in accordance
            # with the boundary conditions
            # KK_global         : stiffness matrix 
            # MM_global         : mass matrix
            # CC_global         : damping matrix
            # FF_global         : load vector 
            # qq_global         : displacement matrix
            # qq_der_global     : velocity matrix
            # qq_derder_global  : acceleration matrix
            # polar_interp      : in the case of an interpolated polar, the loaded values still in memory during the calculation
            # solver_vibmod     : solver for the free vibration problem
            start_cputimeb = time_vec[ii_time0]
            if ii_time0+1 == len(time_vec):
                time_valvec = time_vec
            else:
                time_valvec = time_vec[:ii_time0+1]
            qq_global, qq_der_global, qq_derder_global, FF_global, FF_orig, disp_values, case_setup, ref_axis, mesh_data = boundaryconditions_tran(KK_global,MM_global,CC_global,FF_global,qq_global,qq_der_global,qq_derder_global,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,time_vec[ii_time0],tmin_ann,ttran_ann,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p,disp_values,ii_time,int_per,time_valvec) 
            print("Calculated time:" + "{:.5f}".format(time_vec[ii_time0])+" seconds-- CPU cost: " +"{:.5f}".format(time.process_time()-start_cputimeb))
            if case_setup.mod_sol == "YES":
                qq_global,qq_der_global,qq_derder_global,FF_global,ind_RR2,time_stp_op,yy_L,yy_der_L,yy_derder_L,sys_prev = dyn_linsys_mod(case_setup,KK_global,MM_global,CC_global,RR_global,qq_global,qq_der_global,qq_derder_global,FF_global,ii_time,init_it,time_stp,time_stp_op,ind_RR2,vibfree,sys_prev)
            else:
                qq_global,qq_der_global,qq_derder_global,FF_global,ind_RR2,time_stp_op = dyn_linsys(case_setup,KK_global,MM_global,CC_global,RR_global,qq_global,qq_der_global,qq_derder_global,FF_global,ii_time,init_it,time_stp,time_stp_op,ind_RR2,sys_prev)
            print("Calculated time:" + "{:.5f}".format(time_vec[ii_time0])+" seconds-- CPU cost: " +"{:.5f}".format(time.process_time()-start_cputimeb))
        else:
            # ind_RR2 : index to delete from KK_LL after adding the zero rows
            time_valvec = np.array([time_vec[0]])
            ind_RR2 = []    
            qq_global, qq_der_global, qq_derder_global, FF_global, FF_orig, disp_values, case_setup, ref_axis, mesh_data = boundaryconditions_tran(KK_global,MM_global,CC_global,FF_global_prima,qq_global_prima,qq_der_global_prima,qq_derder_global_prima,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,time_vec[ii_time0],tmin_ann,ttran_ann,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p,disp_values,ii_time,int_per,time_valvec) 
        if case_setup.problem_type == "AEL_DYN":
            # aoa_val   : value of the angle of attack
            # aoap_val  : value of the angle of attack derivative
            # aoapp_val : value of the angle of attack second derivative
            # CL_val    : value of the lift coefficient
            # CD_val    : value of the drag coefficient
            # CM_val    : vlaue of the moment coefficient
            clsec_val = disp_values.clsec_val
            cdsec_val = disp_values.cdsec_val
            cmsec_val = disp_values.cmsec_val
            aoa_val   = disp_values.aoa_val
            aoap_val  = disp_values.aoap_val
            aoapp_val = disp_values.aoapp_val
            CL_val    = disp_values.CL_val
            CD_val    = disp_values.CD_val
            CM_val    = disp_values.CM_val
            CT_val    = disp_values.CT_val
            CQ_val    = disp_values.CQ_val
            CP_val    = disp_values.CP_val
            PE_val    = disp_values.PE_val
        # Set the displacement values
        for ii_q in np.arange(len(qq_global_prima)):
            if ii_q % 9 == 0:
                u_val[int(np.floor(ii_q/9))]      = qq_global[ii_q,ii_time]
                u_dt_val[int(np.floor(ii_q/9))]   = qq_der_global[ii_q,ii_time]
                u_dtdt_val[int(np.floor(ii_q/9))] = qq_derder_global[ii_q,ii_time]
                if int(np.floor(ii_q/9))>0:
                    if case_setup.filt == 'YES':
                        u_dtdtdt_val[int(np.floor(ii_q/9))] = uddd_vec_filter[ii_filter-1,int(np.floor(ii_q/9))]+(u_dtdt_val[int(np.floor(ii_q/9))]-udd_vec_filter[ii_filter-1,int(np.floor(ii_q/9))])/time_stp
                    else:
                        u_dtdtdt_val[int(np.floor(ii_q/9))] = (qq_derder_global[ii_q,ii_time]-qq_derder_global[ii_q,ii_time-1])/time_stp
                else:
                    u_dtdtdt_val[int(np.floor(ii_q/9))] = 0
            elif ii_q % 9 == 1:
                v_val[int(np.floor(ii_q/9))]      = qq_global[ii_q,ii_time]
                v_dt_val[int(np.floor(ii_q/9))]   = qq_der_global[ii_q,ii_time]
                v_dtdt_val[int(np.floor(ii_q/9))] = qq_derder_global[ii_q,ii_time]
                if int(np.floor(ii_q/9))>0:
                    if case_setup.filt == 'YES':
                        v_dtdtdt_val[int(np.floor(ii_q/9))] = vddd_vec_filter[ii_filter-1,int(np.floor(ii_q/9))]+(v_dtdt_val[int(np.floor(ii_q/9))]-vdd_vec_filter[ii_filter-1,int(np.floor(ii_q/9))])/time_stp
                    else:
                        v_dtdtdt_val[int(np.floor(ii_q/9))] = (qq_derder_global[ii_q,ii_time]-qq_derder_global[ii_q,ii_time-1])/time_stp
                else:
                    v_dtdtdt_val[int(np.floor(ii_q/9))] = 0
            elif ii_q % 9 == 2:
                w_val[int(np.floor(ii_q/9))]      = qq_global[ii_q,ii_time]
                w_dt_val[int(np.floor(ii_q/9))]   = qq_der_global[ii_q,ii_time]
                w_dtdt_val[int(np.floor(ii_q/9))] = qq_derder_global[ii_q,ii_time]
                if int(np.floor(ii_q/9))>0:
                    if case_setup.filt == 'YES':
                        w_dtdtdt_val[int(np.floor(ii_q/9))] = wddd_vec_filter[ii_filter-1,int(np.floor(ii_q/9))]+(w_dtdt_val[int(np.floor(ii_q/9))]-wdd_vec_filter[ii_filter-1,int(np.floor(ii_q/9))])/time_stp
                    else:
                        w_dtdtdt_val[int(np.floor(ii_q/9))] = (qq_derder_global[ii_q,ii_time]-qq_derder_global[ii_q,ii_time-1])/time_stp
                else:
                    w_dtdtdt_val[int(np.floor(ii_q/9))] = 0
            elif ii_q % 9 == 3:
                phi_val[int(np.floor(ii_q/9))]      = qq_global[ii_q,ii_time]
                phi_dt_val[int(np.floor(ii_q/9))]   = qq_der_global[ii_q,ii_time]
                phi_dtdt_val[int(np.floor(ii_q/9))] = qq_derder_global[ii_q,ii_time]
            elif ii_q % 9 == 4:
                psi_val[int(np.floor(ii_q/9))]      = qq_global[ii_q,ii_time]
                psi_dt_val[int(np.floor(ii_q/9))]   = qq_der_global[ii_q,ii_time]
                psi_dtdt_val[int(np.floor(ii_q/9))] = qq_derder_global[ii_q,ii_time]
            elif ii_q % 9 == 5:
                theta_val[int(np.floor(ii_q/9))]      = qq_global[ii_q,ii_time]
                theta_dt_val[int(np.floor(ii_q/9))]   = qq_der_global[ii_q,ii_time]
                theta_dtdt_val[int(np.floor(ii_q/9))] = qq_derder_global[ii_q,ii_time]
            elif ii_q % 9 == 6:
                phi_d_val[int(np.floor(ii_q/9))]   = qq_global[ii_q,ii_time]
            elif ii_q % 9 == 7:
                psi_d_val[int(np.floor(ii_q/9))]   = qq_global[ii_q,ii_time]
            elif ii_q % 9 == 8:
                theta_d_val[int(np.floor(ii_q/9))] = qq_global[ii_q,ii_time]
        if case_setup.filt == 'YES':
            disp_values.u          = u_val_filter2 
            disp_values.v          = v_val_filter2 
            disp_values.w          = w_val_filter2
            disp_values.u_dt       = ud_val_filter2
            disp_values.v_dt       = vd_val_filter2
            disp_values.w_dt       = wd_val_filter2
            disp_values.u_dtdt     = udd_val_filter2
            disp_values.u_dtdt     = vdd_val_filter2
            disp_values.u_dtdt     = wdd_val_filter2
            disp_values.u_dtdtdt   = uddd_val_filter2
            disp_values.u_dtdtdt   = vddd_val_filter2
            disp_values.u_dtdtdt   = wddd_val_filter2
            disp_values.phi        = phi_val_filter2
            disp_values.psi        = psi_val_filter2 
            disp_values.theta      = theta_val_filter2 
            disp_values.phi_dt     = phid_val_filter2
            disp_values.psi_dt     = psid_val_filter2
            disp_values.theta_dt   = thetad_val_filter2
            disp_values.phi_dtdt   = phidd_val_filter2
            disp_values.psi_dtdt   = psidd_val_filter2
            disp_values.theta_dtdt = thetadd_val_filter2
        else:
            disp_values.u          = u_val
            disp_values.v          = v_val 
            disp_values.w          = w_val
            disp_values.u_dt       = u_dt_val
            disp_values.v_dt       = v_dt_val
            disp_values.w_dt       = w_dt_val
            disp_values.u_dtdt     = u_dtdt_val
            disp_values.u_dtdt     = v_dtdt_val
            disp_values.u_dtdt     = w_dtdt_val
            disp_values.u_dtdtdt   = u_dtdtdt_val
            disp_values.u_dtdtdt   = v_dtdtdt_val
            disp_values.u_dtdtdt   = w_dtdtdt_val
            disp_values.phi        = phi_val
            disp_values.psi        = psi_val
            disp_values.theta      = theta_val
            disp_values.phi_dt     = phi_dt_val
            disp_values.psi_dt     = psi_dt_val
            disp_values.theta_dt   = theta_dt_val
            disp_values.phi_dtdt   = phi_dtdt_val
            disp_values.psi_dtdt   = psi_dtdt_val
            disp_values.theta_dtdt = theta_dtdt_val
        if case_setup.filt == 'YES':
            # ref_axis_mat  : reference axis matrix
            # taps          : output of the signal filter
            # time_val      : value of the time
            # flag_filter   : flag to update the filter vector
            ref_axis_mat = np.resize(ref_axis,(len(ref_axis),3))
            taps         = signal.firwin(N, fmax, nyq=nyq,window=('kaiser', beta), pass_zero=True)  
            time_val     = time_vec[ii_time0]
            flag_filter  = 0
            if ii_filter == Nvec2:
                flag_filter = 1
                ii_filter   = Nvec1
            if any(ref_axis_mat[:,0]) != 0:
                phi_vec_filter,phi_vec_filter2,phi_val_filter       = filter_func(phi_vec_filter,phi_vec_filter2,phi_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                phid_vec_filter,phid_vec_filter2,phid_val_filter    = filter_func(phid_vec_filter,phid_vec_filter2,phi_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                phidd_vec_filter,phidd_vec_filter2,phidd_val_filter = filter_func(phidd_vec_filter,phidd_vec_filter2,phi_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
            if any(ref_axis_mat[:,1]) != 0:
                psi_vec_filter,phsi_vec_filter2,psi_val_filter       = filter_func(psi_vec_filter,psi_vec_filter2,psi_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                psid_vec_filter,phsid_vec_filter2,psid_val_filter    = filter_func(psid_vec_filter,psid_vec_filter2,psi_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                psidd_vec_filter,phsidd_vec_filter2,psidd_val_filter = filter_func(psidd_vec_filter,psidd_vec_filter2,psi_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
            if any(ref_axis_mat[:,2]) != 0:
                theta_vec_filter,theta_vec_filter2,theta_val_filter       = filter_func(theta_vec_filter,theta_vec_filter2,theta_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                thetad_vec_filter,thetad_vec_filter2,thetad_val_filter    = filter_func(thetad_vec_filter,thetad_vec_filter2,theta_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                thetadd_vec_filter,thetadd_vec_filter2,thetadd_val_filter = filter_func(thetadd_vec_filter,thetadd_vec_filter2,theta_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
            if any(ref_axis_mat[:,2]) != 0:
                u_vec_filter,u_vec_filter2,u_val_filter          = filter_func(u_vec_filter,u_vec_filter2,u_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                ud_vec_filter,ud_vec_filter2,ud_val_filter       = filter_func(ud_vec_filter,ud_vec_filter2,u_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                udd_vec_filter,udd_vec_filter2,udd_val_filter    = filter_func(udd_vec_filter,udd_vec_filter2,u_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                uddd_vec_filter,uddd_vec_filter2,uddd_val_filter = filter_func(uddd_vec_filter,uddd_vec_filter2,u_dtdtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                v_vec_filter,v_vec_filter2,v_val_filter          = filter_func(v_vec_filter,v_vec_filter2,v_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                vd_vec_filter,vd_vec_filter2,vd_val_filter       = filter_func(vd_vec_filter,vd_vec_filter2,v_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                vdd_vec_filter,vdd_vec_filter2,vdd_val_filter    = filter_func(vdd_vec_filter,vdd_vec_filter2,v_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                vddd_vec_filter,vddd_vec_filter2,vddd_val_filter = filter_func(vddd_vec_filter,vddd_vec_filter2,v_dtdtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
            if any(ref_axis_mat[:,1]) != 0:
                u_vec_filter,u_vec_filter2,u_val_filter          = filter_func(u_vec_filter,u_vec_filter2,u_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                ud_vec_filter,ud_vec_filter2,ud_val_filter       = filter_func(ud_vec_filter,ud_vec_filter2,u_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                udd_vec_filter,udd_vec_filter2,udd_val_filter    = filter_func(udd_vec_filter,udd_vec_filter2,u_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                uddd_vec_filter,uddd_vec_filter2,uddd_val_filter = filter_func(uddd_vec_filter,uddd_vec_filter2,u_dtdtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                wd_vec_filter,wd_vec_filter2,wd_val_filter       = filter_func(w_vec_filter,w_vec_filter2,w_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                wd_vec_filter,wd_vec_filter2,wd_val_filter       = filter_func(wd_vec_filter,wd_vec_filter2,w_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                wdd_vec_filter,wdd_vec_filter2,wdd_val_filter    = filter_func(wdd_vec_filter,wdd_vec_filter2,w_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                wddd_vec_filter,wddd_vec_filter2,wddd_val_filter = filter_func(wddd_vec_filter,wddd_vec_filter2,w_dtdtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
            if any(ref_axis_mat[:,0]) != 0:
                v_vec_filter,v_vec_filter2,v_val_filter          = filter_func(v_vec_filter,v_vec_filter2,v_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                vd_vec_filter,vd_vec_filter2,vd_val_filter       = filter_func(vd_vec_filter,vd_vec_filter2,v_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                vdd_vec_filter,vdd_vec_filter2,vdd_val_filter    = filter_func(vdd_vec_filter,vdd_vec_filter2,v_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                vddd_vec_filter,vddd_vec_filter2,vddd_val_filter = filter_func(vddd_vec_filter,vddd_vec_filter2,v_dtdtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                w_vec_filter,w_vec_filter2,w_val_filter          = filter_func(w_vec_filter,w_vec_filter2,w_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                wd_vec_filter,wd_vec_filter2,wd_val_filter       = filter_func(wd_vec_filter,wd_vec_filter2,w_dt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                wdd_vec_filter,wdd_vec_filter2,wdd_val_filter    = filter_func(wdd_vec_filter,wdd_vec_filter2,w_dtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
                wddd_vec_filter,wddd_vec_filter2,wddd_val_filter = filter_func(wddd_vec_filter,wddd_vec_filter2,w_dtdtdt_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann2,flag_filter)
        ii_filter += 1    
        # Print in the required frequency
        if  ii_time0 % case_setup.print_val == 0:           
            print("--------------------------------------------------")
            print("Calculated time:" + "{:.5f}".format(time_vec[ii_time0])+" seconds-- CPU cost: " +"{:.5f}".format(time.process_time()-start_cputime))
            if time_vec[ii_time0]>=tmin_ann and  case_setup.filt == 'YES':
                print('Filter on')
        if ii_time0 == 0:
            elemsec1 = []
            elemsec2 = []
            for elem in mesh_elem:
                for ii_sec in np.arange(len(section.elem[int(elem[1])])):
                    # For every section node:
                    for ii_point in np.arange(len(np.zeros((len(section.points[int(elem[1])][ii_sec][:,0]))))):
                        # For all the section elements check if they contain the evaluated node
                        elemsec1.append(np.where(section.elem[int(elem[1])][ii_sec][:,1:3]==ii_point)[0])
                for ii_sec in np.arange(len(section.elem[int(elem[2])])):
                    # For every section node:
                    for ii_point in np.arange(len(np.zeros((len(section.points[int(elem[2])][ii_sec][:,0]))))):
                        # For all the section elements check if they contain the evaluated node
                        elemsec2.append(np.where(section.elem[int(elem[2])][ii_sec][:,1:3]==ii_point)[0])
        if ii_time0 % case_setup.savefile_step == 0:
            # the solution is stored for the saving frequency even if the saving option is off
            time_vec_save.append(time_vec[ii_time0])
            # Save modal contribution
            for ii_q in np.arange(case_setup.n_mod):
                yy_vec[ii_q].append(yy_L[ii_q])
                yy_der_vec[ii_q].append(yy_der_L[ii_q])
                yy_derder_vec[ii_q].append(yy_derder_L[ii_q])
            # Save aerodynamic coefficients and displacements
            for ii_q in np.arange(len(qq_global_prima)):
                if ii_q == 0 and  case_setup.problem_type == "AEL_DYN":
                    cl_vec.append(CL_val)
                    cd_vec.append(CD_val)
                    cm_vec.append(CM_val)
                    ct_vec.append(CT_val)
                    cq_vec.append(CQ_val)
                    cp_vec.append(CP_val)
                    pe_vec.append(PE_val)
                if ii_q % 9==0:
                    u_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
                    udt_vec[int(np.floor(ii_q/9))].append(qq_der_global[ii_q,ii_time])
                    udtdt_vec[int(np.floor(ii_q/9))].append(qq_derder_global[ii_q,ii_time])
                    if case_setup.filt == 'YES':
                        u_filter_vec[int(np.floor(ii_q/9))].append(u_val_filter[int(np.floor(ii_q/9))])
                        udt_filter_vec[int(np.floor(ii_q/9))].append(ud_val_filter[int(np.floor(ii_q/9))])
                        udtdt_filter_vec[int(np.floor(ii_q/9))].append(udd_val_filter[int(np.floor(ii_q/9))])
                    try:
                        aoa_vec[int(np.floor(ii_q/9))].append(aoa_val[int(np.floor(ii_q/9))])
                        clsec_vec[int(np.floor(ii_q/9))].append(clsec_val[int(np.floor(ii_q/9))])
                        cdsec_vec[int(np.floor(ii_q/9))].append(cdsec_val[int(np.floor(ii_q/9))])
                        cmsec_vec[int(np.floor(ii_q/9))].append(cmsec_val[int(np.floor(ii_q/9))])
                        aoadt_vec[int(np.floor(ii_q/9))].append(aoap_val[int(np.floor(ii_q/9))])
                        aoadtdt_vec[int(np.floor(ii_q/9))].append(aoapp_val[int(np.floor(ii_q/9))])
                    except:
                        pass
                elif ii_q % 9 == 1:
                    v_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
                    vdt_vec[int(np.floor(ii_q/9))].append(qq_der_global[ii_q,ii_time])
                    vdtdt_vec[int(np.floor(ii_q/9))].append(qq_derder_global[ii_q,ii_time])
                    if case_setup.filt == 'YES':
                        v_filter_vec[int(np.floor(ii_q/9))].append(v_val_filter[int(np.floor(ii_q/9))])
                        vdt_filter_vec[int(np.floor(ii_q/9))].append(vd_val_filter[int(np.floor(ii_q/9))])
                        vdtdt_filter_vec[int(np.floor(ii_q/9))].append(vdd_val_filter[int(np.floor(ii_q/9))])
                elif ii_q % 9 == 2:
                    w_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
                    wdt_vec[int(np.floor(ii_q/9))].append(qq_der_global[ii_q,ii_time])
                    wdtdt_vec[int(np.floor(ii_q/9))].append(qq_derder_global[ii_q,ii_time])
                    if case_setup.filt == 'YES':
                        w_filter_vec[int(np.floor(ii_q/9))].append(w_val_filter[int(np.floor(ii_q/9))])
                        wdt_filter_vec[int(np.floor(ii_q/9))].append(wd_val_filter[int(np.floor(ii_q/9))])
                        wdtdt_filter_vec[int(np.floor(ii_q/9))].append(wdd_val_filter[int(np.floor(ii_q/9))])
                elif ii_q % 9 == 3:
                    phi_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
                    phidt_vec[int(np.floor(ii_q/9))].append(qq_der_global[ii_q,ii_time])
                    if case_setup.filt == 'YES':
                        phi_filter_vec[int(np.floor(ii_q/9))].append(phi_val_filter[int(np.floor(ii_q/9))])
                        phidt_filter_vec[int(np.floor(ii_q/9))].append(phid_val_filter[int(np.floor(ii_q/9))])
                        phidtdt_filter_vec[int(np.floor(ii_q/9))].append(phidd_val_filter[int(np.floor(ii_q/9))])
                elif ii_q % 9 == 4:
                    psi_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
                    psidt_vec[int(np.floor(ii_q/9))].append(qq_der_global[ii_q,ii_time])
                    psidtdt_vec[int(np.floor(ii_q/9))].append(qq_derder_global[ii_q,ii_time])
                    if case_setup.filt == 'YES':
                        psi_filter_vec[int(np.floor(ii_q/9))].append(psi_val_filter[int(np.floor(ii_q/9))])
                        psidt_filter_vec[int(np.floor(ii_q/9))].append(psid_val_filter[int(np.floor(ii_q/9))])
                        psidtdt_filter_vec[int(np.floor(ii_q/9))].append(psidd_val_filter[int(np.floor(ii_q/9))])
                elif ii_q % 9 == 5:
                    theta_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
                    thetadt_vec[int(np.floor(ii_q/9))].append(qq_der_global[ii_q,ii_time])
                    thetadtdt_vec[int(np.floor(ii_q/9))].append(qq_derder_global[ii_q,ii_time])
                    if case_setup.filt == 'YES':
                        theta_filter_vec[int(np.floor(ii_q/9))].append(theta_val_filter[int(np.floor(ii_q/9))])
                        thetadt_filter_vec[int(np.floor(ii_q/9))].append(thetad_val_filter[int(np.floor(ii_q/9))])
                        thetadtdt_filter_vec[int(np.floor(ii_q/9))].append(thetadd_val_filter[int(np.floor(ii_q/9))])
                elif ii_q % 9 == 6:
                    phi_d_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
                elif ii_q % 9 == 7:
                    psi_d_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
                elif ii_q % 9 == 8:
                    theta_d_vec[int(np.floor(ii_q/9))].append(qq_global[ii_q,ii_time])
            # Save force vector
            for ii_q in np.arange(len(FF_global)):
                if ii_q % 9==0:
                    fx_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
                elif ii_q % 9 == 1:
                    fy_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
                elif ii_q % 9 == 2:
                    fz_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
                elif ii_q % 9 == 3:
                    Mx_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
                elif ii_q % 9 == 4:
                    My_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
                elif ii_q % 9 == 5:
                    Mz_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
                elif ii_q % 9 == 6:
                    Bx_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
                elif ii_q % 9 == 7:
                    By_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
                elif ii_q % 9 == 8:
                    Bz_vec[int(np.floor(ii_q/9))].append(FF_global[ii_q,ii_time])
            # Save the position of the nodes        
            ii_elem = 0
            # Store the initial position in the displacement matrix
            ii_time_pos = int(ii_time0/case_setup.savefile_step)
            pos_time[:,:,ii_time_pos] = mesh_point.copy()
            for ii_q in np.arange(len(qq_global_prima)):
                if ii_q % 9==0:
                    pos_time[ii_elem,0,ii_time_pos] = pos_time[ii_elem,0,0]+qq_global[ii_q,ii_time]
                elif ii_q % 9 == 1:
                    pos_time[ii_elem,1,ii_time_pos] = pos_time[ii_elem,1,0]+qq_global[ii_q,ii_time]
                elif ii_q % 9 == 2:
                    pos_time[ii_elem,2,ii_time_pos] = pos_time[ii_elem,2,0]+qq_global[ii_q,ii_time]
                elif ii_q % 9 == 3:
                    ii_elem=ii_elem+1 
#            print("Basic storage time:" + "{:.5f}".format(time_vec[ii_time0])+" seconds-- CPU cost: " +"{:.5f}".format(time.process_time()-start_cputime))
            # save the warping
            for elem in mesh_elem:
                # calculate the warping angle for every beam element
                # warp_angle1 : warping angle on node 1
                # warp_angle2 : warping angle on node 2
                warp_angle1 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[phi_d_vec[int(elem[1])][ii_time_save]],[psi_d_vec[int(elem[1])][ii_time_save]],[theta_d_vec[int(elem[1])][ii_time_save]]]))
                warp_angle2 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[phi_d_vec[int(elem[2])][ii_time_save]],[psi_d_vec[int(elem[2])][ii_time_save]],[theta_d_vec[int(elem[2])][ii_time_save]]]))
                warp1 = []
                warp2 = []
                for ii_sec in np.arange(len(section.elem[int(elem[1])])):
                    # The warping is evaluated for all the internal sections in the first beam node in the element
                    func_warp1 = np.zeros((len(section.points[int(elem[1])][ii_sec][:,0])))
                    # For every section node:
                    for ii_point in np.arange(len(func_warp1)):
                        # nn : counter of the elements to ponderate the value in each node
                        nn = 0
                        # For all the section elements check if they contain the evaluated node
#                        elemsec = np.where(section.elem[int(elem[1])][ii_sec][:,1:3]==ii_point)[0]
                        for jj_elem in np.arange(len(elemsec1[int(elem[0])])):
                            func_warp1[ii_point] += sol_phys.warp[int(elem[1])][ii_sec][jj_elem]
                            nn                   += 1
#                        for jj_elem in np.arange(len(section.elem[int(elem[1])][ii_sec])):
#                            if section.elem[int(elem[1])][ii_sec][jj_elem,1]==ii_point or section.elem[int(elem[1])][ii_sec][jj_elem,2]==ii_point:
#                                # Add the value of the warping function
#                                func_warp1[ii_point] += sol_phys.warp[int(elem[1])][ii_sec][jj_elem]
#                                nn                   += 1
                        # Calculate the average warping function
                        func_warp1[ii_point] /= nn
                    # warp1        : warping for the section nodes of the first beam element node
                    # x_warp1      : projection on x axis of the warping of the first beam element node
                    # y_warp1      : projection on y axis of the warping of the first beam element node
                    # z_warp1      : projection on z axis of the warping of the first beam element node
                    # warp_global1 : rotation of warp1 to the global axis
                    # waarp_x1_vec : time vector of the warping in node 1 and x axis
                    # waarp_y1_vec : time vector of the warping in node 1 and y axis
                    # waarp_z1_vec : time vector of the warping in node 1 and z axis
                    for ii_point in np.arange(len(func_warp1)):
                        warp1.append(func_warp1[ii_point]*warp_angle1)
                x_warp1 = np.zeros((len(warp1),))
                y_warp1 = np.zeros((len(warp1),))
                z_warp1 = np.zeros((len(warp1),))
#                print("Warp storage1 time:" + "{:.5f}".format(time_vec[ii_time0])+" seconds-- CPU cost: " +"{:.5f}".format(time.process_time()-start_cputime))
                for ii_elem in np.arange(len(warp1)):
                    warp_global1     = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp1[ii_elem]]])
                    x_warp1[ii_elem] = warp_global1[0]
                    y_warp1[ii_elem] = warp_global1[1]
                    z_warp1[ii_elem] = warp_global1[2]
                for ii_appendwarp in np.arange(len(warp1)):
                    warpx1dat[int(elem[0])].append(x_warp1[ii_appendwarp])
                    warpy1dat[int(elem[0])].append(y_warp1[ii_appendwarp])
                    warpz1dat[int(elem[0])].append(z_warp1[ii_appendwarp])
                for ii_sec in np.arange(len(section.elem[int(elem[2])])):
                    # The warping is evaluated for all the internal sections in the second beam node in the element
                    func_warp2 = np.zeros((len(section.points[int(elem[2])][ii_sec][:,0])))
                    # For every section node:
                    for ii_point in np.arange(len(func_warp2)):
                        # nn : counter of the elements to ponderate the value in each node
                        nn = 0
                        # For all the section elements check if they contain the evaluated node
#                        elemsec = np.where(section.elem[int(elem[2])][ii_sec][:,1:3]==ii_point)[0]
                        for jj_elem in np.arange(len(elemsec2[int(elem[0])])):
                            func_warp2[ii_point] += sol_phys.warp[int(elem[2])][ii_sec][jj_elem]
                            nn                   += 1
#                        for jj_elem in np.arange(len(section.elem[int(elem[2])][ii_sec])):
#                            if section.elem[int(elem[2])][ii_sec][jj_elem,1]==ii_point or  section.elem[int(elem[2])][ii_sec][jj_elem,2]==ii_point:
#                                # Add the value of the warping function
#                                func_warp2[ii_point] += sol_phys.warp[int(elem[2])][ii_sec][jj_elem]
#                                nn                   += 1
                        # Calculate the average warping function
                        func_warp2[ii_point] /= nn
                    # warp2        : warping for the section nodes of the first beam element node
                    # x_warp2      : projection on x axis of the warping of the first beam element node
                    # y_warp2      : projection on y axis of the warping of the first beam element node
                    # z_warp2      : projection on z axis of the warping of the first beam element node
                    # warp_global2 : rotation of warp1 to the global axis
                    # waarp_x2_vec : time vector of the warping in node 1 and x axis
                    # waarp_y2_vec : time vector of the warping in node 1 and y axis
                    # waarp_z2_vec : time vector of the warping in node 1 and z axis
                    for ii_point in np.arange(len(func_warp2)):
                        warp2.append(func_warp2[ii_point]*warp_angle2)
                x_warp2 = np.zeros((len(warp2),))
                y_warp2 = np.zeros((len(warp2),))
                z_warp2 = np.zeros((len(warp2),))
                for ii_elem in np.arange(len(warp2)):
                    warp_global2     = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp2[ii_elem]]])
                    x_warp2[ii_elem] = warp_global2[0]
                    y_warp2[ii_elem] = warp_global2[1]
                    z_warp2[ii_elem] = warp_global2[2]
                for ii_appendwarp in np.arange(len(warp2)):
                    warpx2dat[int(elem[0])].append(x_warp2[ii_appendwarp])
                    warpy2dat[int(elem[0])].append(y_warp2[ii_appendwarp])
                    warpz2dat[int(elem[0])].append(z_warp2[ii_appendwarp])
            warp_x1_vec.append(warpx1dat)
            warp_y1_vec.append(warpy1dat)
            warp_z1_vec.append(warpz1dat)
            warp_x2_vec.append(warpx2dat)
            warp_y2_vec.append(warpy2dat)
            warp_z2_vec.append(warpz2dat)
            # To save the information in a file
            if case_setup.savefile == 'YES':
                # Create the folder if needed
                try:
                    os.mkdir(case_setup.root+case_setup.savefile_name)
                except:
                    print("Carpeta Existente")
                # name_dips_file  : name of the displacement file
                # file_disp       : table of the displacement information
                # u_vecarrr       : u_vec in array form
                # v_vecarrr       : v_vec in array form
                # w_vecarrr       : w_vec in array form
                # phi_vecarrr     : phi_vec in array form
                # psi_vecarrr     : psi_vec in array form
                # theta_vecarrr   : theta_vec in array form
                # phi_d_vecarrr   : phi_d_vec in array form
                # psi_d_vecarrr   : psi_d_vec in array form
                # theta_d_vecarrr : theta_d_vec in array form
                # fx_vecarr       : fx_vec in array form
                # fy_vecarr       : fy_vec in array form
                # fz_vecarr       : fz_vec in array form
                # Mx_vecarr       : Mx_vec in array form
                # My_vecarr       : My_vec in array form
                # Mz_vecarr       : Mz_vec in array form
                # Bx_vecarr       : Bx_vec in array form
                # By_vecarr       : By_vec in array form
                # Bz_vecarr       : Bz_vec in array form
                name_disp_file = case_setup.root+case_setup.savefile_name+'/disp_'+str(ii_time_save)+'.csv'
                file_disp = pd.DataFrame()
                u_vecarr = np.array(u_vec)
                v_vecarr = np.array(v_vec)
                w_vecarr = np.array(w_vec)
                phi_vecarr = np.array(phi_vec)
                psi_vecarr = np.array(psi_vec)
                theta_vecarr = np.array(psi_vec)
                phi_d_vecarr = np.array(phi_d_vec)
                psi_d_vecarr = np.array(psi_d_vec)
                theta_d_vecarr = np.array(psi_d_vec)
                fx_vecarr = np.array(fx_vec)
                fy_vecarr = np.array(fy_vec)
                fz_vecarr = np.array(fz_vec)
                Mx_vecarr = np.array(Mx_vec)
                My_vecarr = np.array(My_vec)
                Mz_vecarr = np.array(Mz_vec)
                Bx_vecarr = np.array(Bx_vec)
                By_vecarr = np.array(Bx_vec)
                Bz_vecarr = np.array(Bx_vec)
                # Store the information in the table variable
                file_disp["node"]           = np.arange(len(u_vec))
                file_disp["x(m)"]           = mesh_data.point[:,0]
                file_disp["y(m)"]           = mesh_data.point[:,1]
                file_disp["z(m)"]           = mesh_data.point[:,2]
                file_disp["u(m)"]           = u_vecarr[:,ii_time_save]
                file_disp["v(m)"]           = v_vecarr[:,ii_time_save]
                file_disp["w(m)"]           = w_vecarr[:,ii_time_save]
                file_disp["phi(rad)"]       = phi_vecarr[:,ii_time_save]
                file_disp["psi(rad)"]       = psi_vecarr[:,ii_time_save]
                file_disp["theta(rad)"]     = theta_vecarr[:,ii_time_save]
                file_disp["phi_d(rad/m)"]   = phi_d_vecarr[:,ii_time_save]
                file_disp["psi_d(rad/m)"]   = psi_d_vecarr[:,ii_time_save]
                file_disp["theta_d(rad/m)"] = theta_d_vecarr[:,ii_time_save]
                # Save the information in csv file
                file_disp.to_csv(name_disp_file,index=False)
                # name_dips_file  : name of the deformed shape file
                # file_ds         : table of the deformed shape values
                name_ds_file = case_setup.root+case_setup.savefile_name+'/defshape'+str(ii_time_save)+'.csv'
                file_ds      = pd.DataFrame()
                # save the values in the table
                file_ds["node"] = np.arange(len(u_vec))
                file_ds["x(m)"] = mesh_data.point[:,0]
                file_ds["y(m)"] = mesh_data.point[:,1]
                file_ds["z(m)"] = mesh_data.point[:,2]
                file_ds["u(m)"] = pos_time[:,0,ii_time_save]
                file_ds["v(m)"] = pos_time[:,1,ii_time_save]
                file_ds["w(m)"] = pos_time[:,2,ii_time_save]
                # Save the deformed shape in a csv file
                file_ds.to_csv(name_ds_file,index=False)
                # name_forc_file  : name of the force file
                # file_forc       : table to store forces
                name_forc_file = case_setup.root+case_setup.savefile_name+'/forces'+str(ii_time_save)+'.csv'
                file_forc      = pd.DataFrame()
                # Save the values in the table
                file_forc["node"]   = np.arange(len(u_vec))
                file_forc["x(m)"]   = mesh_data.point[:,0]
                file_forc["y(m)"]   = mesh_data.point[:,1]
                file_forc["z(m)"]   = mesh_data.point[:,2]
                file_forc["fx(N)"]  = fx_vecarr[:,ii_time_save]
                file_forc["fy(N)"]  = fy_vecarr[:,ii_time_save]
                file_forc["fz(N)"]  = fz_vecarr[:,ii_time_save]
                file_forc["mx(Nm)"] = Mx_vecarr[:,ii_time_save]
                file_forc["my(Nm)"] = My_vecarr[:,ii_time_save]
                file_forc["mz(Nm)"] = Mz_vecarr[:,ii_time_save]
                file_forc["bx(Nm)"] = Bx_vecarr[:,ii_time_save]
                file_forc["by(Nm)"] = By_vecarr[:,ii_time_save]
                file_forc["bz(Nm)"] = Bz_vecarr[:,ii_time_save]
                # Save the forces in a csv file
                file_forc.to_csv(name_forc_file,index=False)
                # If the data in section nodes is calculated its deformed position can be obtained
                if mesh_data.section == "YES":
                    # file_secdef      : table to store the deformed section information
                    # file_secdef_aux  : auxiliary table for the section information
                    # name_secdef_file : name of the deformed section file
                    file_secdef      = pd.DataFrame()
                    file_secdef_aux  = pd.DataFrame()
                    name_secdef_file = case_setup.root+case_setup.savefile_name+'/secdef'+str(ii_time_save)+'.csv'
                    # For all the first nodes of the elements:
                    for ii_plot_sec in np.arange(len(section_globalCS.n1)):
                        # plot_sec : nodes of the section
                        # plot_cdg : center of gravity of the section
                        # ii_node  : beam nodes of the section
                        plot_sec = section_globalCS.n1[ii_plot_sec]
                        plot_cdg = section_globalCS.cdgn1[ii_plot_sec]
                        ii_node  = section_globalCS.nodeelem1[ii_plot_sec]
                        # xpoint   : x position of the section node
                        # ypoint   : y position of the section node
                        # zpoint   : z position of the section node
                        xpoint = pos_time[section_globalCS.nodeelem1[ii_plot_sec],0,ii_time_save]+(plot_sec[:,0]-plot_cdg[0])*np.cos(theta_vecarr[ii_node,ii_time_save])*np.cos(psi_vecarr[ii_node,ii_time_save])- \
                            (plot_sec[:,1]-plot_cdg[1])*np.sin(theta_vecarr[ii_node,ii_time_save])+ \
                            (plot_sec[:,2]-plot_cdg[2])*np.sin(psi_vecarr[ii_node,ii_time_save])+warp_x1_vec[ii_plot_sec][ii_time_save]
                        ypoint = pos_time[section_globalCS.nodeelem1[ii_plot_sec],1,ii_time_save]+(plot_sec[:,1]-plot_cdg[1])*np.cos(theta_vecarr[ii_node,ii_time_save])*np.cos(phi_vecarr[ii_node,ii_time_save])+ \
                            (plot_sec[:,0]-plot_cdg[0])*np.sin(theta_vecarr[ii_node,ii_time_save])- \
                            (plot_sec[:,2]-plot_cdg[2])*np.sin(phi_vecarr[ii_node,ii_time_save])+warp_y1_vec[ii_plot_sec][ii_time_save]
                        zpoint = pos_time[section_globalCS.nodeelem1[ii_plot_sec],2,ii_time_save]+(plot_sec[:,2]-plot_cdg[2])*np.cos(psi_vecarr[ii_node,ii_time_save])*np.cos(phi_vecarr[ii_node,ii_time_save])- \
                            (plot_sec[:,0]-plot_cdg[0])*np.sin(psi_vecarr[ii_node,ii_time_save])+ \
                            (plot_sec[:,1]-plot_cdg[1])*np.sin(phi_vecarr[ii_node,ii_time_save])+warp_z1_vec[ii_plot_sec][ii_time_save]
                        # elemsecdef : element that the node belongs
                        # nodesecdef : node of the section
                        elemsecdef = np.ones((len(xpoint),))*int(ii_plot_sec)
                        nodesecdef =  np.zeros((len(xpoint),))
                        # Store the information in the table
                        file_secdef_aux["Element"] = elemsecdef.astype('int')
                        file_secdef_aux["node"]    = nodesecdef.astype('int') 
                        file_secdef_aux["x(m)"]    = xpoint[0] 
                        file_secdef_aux["y(m)"]    = ypoint[0]  
                        file_secdef_aux["z(m)"]    = zpoint[0]  
                        file_secdef                = pd.concat([file_secdef,file_secdef_aux])
                    for ii_plot_sec in np.arange(len(section_globalCS.n2)):
                        # plot_sec : nodes of the section
                        # plot_cdg : center of gravity of the section
                        # ii_node  : beam nodes of the section
                        plot_sec = section_globalCS.n2[ii_plot_sec]
                        plot_cdg = section_globalCS.cdgn2[ii_plot_sec]
                        ii_node  = section_globalCS.nodeelem2[ii_plot_sec]
                        # xpoint   : x position of the section node
                        # ypoint   : y position of the section node
                        # zpoint   : z position of the section node
                        xpoint = pos_time[section_globalCS.nodeelem2[ii_plot_sec],0,ii_time_save]+(plot_sec[:,0]-plot_cdg[0])*np.cos(theta_vecarr[ii_node,ii_time_save])*np.cos(psi_vecarr[ii_node,ii_time_save])- \
                            (plot_sec[:,1]-plot_cdg[1])*np.sin(theta_vecarr[ii_node,ii_time_save])+ \
                            (plot_sec[:,2]-plot_cdg[2])*np.sin(psi_vecarr[ii_node,ii_time_save])+warp_x2_vec[ii_time_save][ii_plot_sec]
                        ypoint = pos_time[section_globalCS.nodeelem2[ii_plot_sec],1,ii_time_save]+(plot_sec[:,1]-plot_cdg[1])*np.cos(theta_vecarr[ii_node,ii_time_save])*np.cos(phi_vecarr[ii_node,ii_time_save])+ \
                            (plot_sec[:,0]-plot_cdg[0])*np.sin(theta_vecarr[ii_node,ii_time_save])- \
                            (plot_sec[:,2]-plot_cdg[2])*np.sin(phi_vecarr[ii_node,ii_time_save])+warp_y2_vec[ii_time_save][ii_plot_sec]
                        zpoint = pos_time[section_globalCS.nodeelem2[ii_plot_sec],2,ii_time_save]+(plot_sec[:,2]-plot_cdg[2])*np.cos(psi_vecarr[ii_node,ii_time_save])*np.cos(phi_vecarr[ii_node,ii_time_save])- \
                            (plot_sec[:,0]-plot_cdg[0])*np.sin(psi_vecarr[ii_node,ii_time_save])+ \
                            (plot_sec[:,1]-plot_cdg[1])*np.sin(phi_vecarr[ii_node,ii_time_save])+warp_z2_vec[ii_time_save][ii_plot_sec]
                        # elemsecdef : element that the node belongs
                        # nodesecdef : node of the section
                        elemsecdef = np.ones((len(xpoint),))*int(ii_plot_sec)
                        nodesecdef =  np.ones((len(xpoint),))
                        # Store the information in the table 
                        file_secdef_aux["Element"] = elemsecdef.astype('int')
                        file_secdef_aux["node"]    = nodesecdef.astype('int')
                        file_secdef_aux["x(m)"]    = xpoint[0]  
                        file_secdef_aux["y(m)"]    = ypoint[0]  
                        file_secdef_aux["z(m)"]    = zpoint[0]
                        file_secdef                = pd.concat([file_secdef,file_secdef_aux])
                    # Save in csv file
                    file_secdef.to_csv(name_secdef_file,index=False)
            ii_time_save +=1
#        if  ii_time0 % case_setup.print_val == 0:           
#            print("--------------------------------------------------")
#            print("Storing time:" + "{:.5f}".format(time_vec[ii_time0])+" seconds-- CPU cost: " +"{:.5f}".format(time.process_time()-start_cputime))
#            if time_vec[ii_time0]>=tmin_ann and  case_setup.filt == 'YES':
#                print('Filter on')
    # Store the solution variables in the solution class
    # solution : class to store the time solution
    class solution:
        pass
    solution.xdef           = pos_time[:,0,:]
    solution.ydef           = pos_time[:,1,:]
    solution.zdef           = pos_time[:,2,:]
    solution.u              = u_vec
    solution.v              = v_vec
    solution.w              = w_vec
    solution.phi            = phi_vec
    solution.psi            = psi_vec
    solution.theta          = theta_vec
    solution.phi_filt       = phi_filter_vec
    solution.psi_filt       = psi_filter_vec
    solution.theta_filt     = theta_filter_vec
    solution.udt            = udt_vec
    solution.vdt            = vdt_vec
    solution.wdt            = wdt_vec
    solution.udt_filt       = udt_filter_vec
    solution.vdt_filt       = vdt_filter_vec
    solution.vdt_filt       = vdt_filter_vec
    solution.phidt          = phidt_vec
    solution.psidt          = psidt_vec
    solution.thetadt        = thetadt_vec
    solution.phidt_filt     = phidt_vec
    solution.psidt_filt     = psidt_vec
    solution.thetadt_filt   = thetadt_filter_vec
    solution.udtdt          = udtdt_vec
    solution.vdtdt          = vdtdt_vec
    solution.wdtdt          = wdtdt_vec
    solution.udtdt_filt     = udtdt_filter_vec
    solution.vdtdt_filt     = vdtdt_filter_vec
    solution.wdtdt_filt     = wdtdt_filter_vec
    solution.phidtdt        = phidtdt_vec
    solution.psidtdt        = psidtdt_vec
    solution.thetadtdt      = thetadtdt_vec
    solution.phidtdt_filt   = phidtdt_filter_vec
    solution.psidtdt_filt   = psidtdt_filter_vec
    solution.thetadtdt_filt = thetadtdt_filter_vec
    solution.phi_d          = phi_d_vec
    solution.psi_d          = psi_d_vec
    solution.theta_d        = theta_d_vec
    solution.aoa            = aoa_vec
    solution.cl_vec           = clsec_vec
    solution.cd_vec            = cdsec_vec
    solution.cm_vec            = cmsec_vec
    solution.aoadt          = aoadt_vec
    solution.aoadtdt        = aoadtdt_vec
    solution.CL             = cl_vec
    solution.CD             = cd_vec
    solution.CM             = cm_vec
    solution.CT             = ct_vec
    solution.CQ             = cq_vec
    solution.CP             = cp_vec
    solution.EP             = pe_vec
    solution.ext_fx         = fx_vec
    solution.ext_fy         = fy_vec
    solution.ext_fz         = fz_vec
    solution.ext_mx         = Mx_vec
    solution.ext_my         = My_vec
    solution.ext_mz         = Mz_vec
    solution.ext_bx         = Bx_vec
    solution.ext_by         = By_vec
    solution.ext_bz         = Bz_vec
    solution.time           = time_vec_save
    solution.warp_x1        = warp_x1_vec
    solution.warp_y1        = warp_y1_vec
    solution.warp_z1        = warp_z1_vec
    solution.warp_x2        = warp_x2_vec
    solution.warp_y2        = warp_y2_vec
    solution.warp_z2        = warp_z2_vec
    solution.yy             = yy_vec
    solution.yy_der         = yy_der_vec
    solution.yy_derder      = yy_derder_vec
    solution.mass = mass_data1+mass_data2+mass_data3
    if case_setup.grav_fl == 1:
        weight = 0
        for ii_point in np.arange(mesh_data.num_point):
            weight_vec = 0
            for jj_point in np.arange(3):
                weight_vec += (case_setup.grav_g[jj_point]*MM_global[ii_point*9+jj_point,ii_point*9+jj_point])**2
            weight += np.sqrt(weight_vec)
        solution.weight = weight
    return solution,section_globalCS

#%%
def solver_vibdamp_mod(case_setup,sol_phys,mesh_data,section,exmass,exstiff): 
    # Function to solve the modal vibration structural problem
    # case_setup : setup information of the case
    # sol_phys   : solid physics information
    # mesh_data  : mesh of the problem
    # section    : data about the section
    # exmass     : extra mass
    # -----------------------------------------------------------------------------
    # mesh_point : nodes of the mesh
    # mesh_elem  : elements of the mesh
    # mesh_mark  : markers of the mesh, boundaries to apply conditions and nodes to visualize information
    # num_elem   : number of beam elements
    # num_point  : number of beam nodes     
    mesh_point = mesh_data.point
    mesh_elem  = mesh_data.elem
    mesh_mark  = mesh_data.marker
    num_elem   = mesh_data.num_elem
    num_point  = mesh_data.num_point
    # qq_global_prima : variables/displacements of the system
    qq_global_prima = np.nan*np.ones((9*num_point,))   
    # -------------------------------------------------------------------------
    # Definition of the stiffness matrix 
    # KK_global_prima : stiffness matrix before applying the joints
    # MM_global_prima : mass matrix before applying the joints
    # LL_vec          : distance between nodes vector
    # rr_vec          : vectorial distance between nodes
    # rot_mat         : rotation matrix of each beam element
    # rotT_mat        : transposed rotation matrix of each beam element
    # Lrot_mat        : complete rotation matrix of the system
    # LrotT_mat       : complete transposed rotation matrix of the system
    # LL_vec_p        : distance between nodes vector for each node in the element
    # rr_vec_p        : vectorial distance between nodes for each node in the element
    # rot_mat_p       : rotation matrix of each beam element for each node in the element
    # rotT_mat_p      : transposed rotation matrix of each beam element for each node in the element
    # ref_axis        : reference axis
    KK_global_prima,LL_vec,rr_vec,rot_mat,rotT_mat,\
        Lrot_mat,LrotT_mat,LL_vec_p,rr_vec_p,rot_mat_p,rot_matT_p,sol_phys  = define_stiffness(mesh_elem,sol_phys,mesh_point,num_point,num_elem,mesh_data.mesh_elem_lock,mesh_data.RR_ctrl,section) 
    mesh_data.LL                                                   = LL_vec
    mesh_data.rr                                                   = rr_vec
    mesh_data.rr_p                                                 = rr_vec_p
    MM_global_prima,mass_data1                                      = define_mass(mesh_elem,sol_phys,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,section)
    mass_data2 = 0*mass_data1
    mass_data3 = mass_data2
    if exmass != []:
        KK_global_prima_exstiffpoint                                   = define_exstiffpoint(exstiff,mesh_point,mesh_elem,num_point,Lrot_mat,LrotT_mat)
        KK_global_prima_exstiffelem                                    = define_exstiffdist(mesh_elem,exstiff,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
        KK_global_prima                                               += KK_global_prima_exstiffpoint + KK_global_prima_exstiffelem
        MM_global_prima_exmass ,mass_data2                                         = define_exmassdist(mesh_elem,exmass,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
        MM_global_prima_exmasspoint,mass_data3                                     = define_exmasspoint(exmass,mesh_point,num_point,Lrot_mat,LrotT_mat)
        MM_global_prima                                               += MM_global_prima_exmass+MM_global_prima_exmasspoint
    # ind_RR  : index to delete from KK_LL
    # q_RR    : displacement in the restricted nodes 
    ind_RR = []
    q_RR = [] 
    # Find the restricted nodes
    ind_RR,q_RR = boundaryconditions_vibrest(mesh_mark,case_setup,num_point,ind_RR,q_RR)
    # -------------------------------------------------------------------------
    # Activate the information about the section internal nodes
    if mesh_data.section == "YES":   
        # section_globalCS  : section nodes in global coordinate system
        section_globalCS = def_secglob(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point)  
    KK_global, MM_global, RR_global, case_setup, ref_axis = init_boundaryconditions_tran(KK_global_prima,MM_global_prima,[],mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)  
    # MM_pow    : mass matrix power to 0.5
    # MM_pow2   : mass matrix power to -0.5
    # w_mode    : mode angular frequecies 
    # mode      : mode shape
    # idx       : index to sort modes
    # freq_mode : frequency 
    vibfree, CC_global = define_damp(case_setup,sol_phys,mesh_data,section,MM_global,KK_global,solver_vib_mod,solver_struct_stat,num_point,mesh_mark)                                       
    l_mode, freq_mode, w_mode, mode, MM_pow, MM_pow2,idx, MM_LL, KK_LL, ind_RR2, mode_tot = vibae_linsys(KK_global,CC_global,MM_global,RR_global,ind_RR,q_RR,num_point)
    # Store the solution
    # solution : class containing the solution data
    solution = store_solution_vibdamp(case_setup,w_mode,mode,freq_mode,idx,MM_pow,MM_pow2,MM_LL,KK_LL,KK_global,ind_RR2,sol_phys,section,rot_mat,rotT_mat,num_point,num_elem,mesh_point,mesh_elem,mode_tot)
    solution.mass = mass_data1+mass_data2+mass_data3
    if case_setup.savefile == 'YES':
        # save the information in files
        save_vib(case_setup,mesh_data,section_globalCS,solution)  
    if case_setup.grav_fl == 1:
        weight = 0
        for ii_point in np.arange(mesh_data.num_point):
            weight_vec = 0
            for jj_point in np.arange(3):
                weight_vec += (case_setup.grav_g[jj_point]*MM_global[ii_point*9+jj_point,ii_point*9+jj_point])**2
            weight += np.sqrt(weight_vec)
        solution.weight = weight
    return solution, section_globalCS

#%%
def solver_vib_mod(case_setup,sol_phys,mesh_data,section,exmass,exstiff): 
    # Function to solve the modal vibration structural problem
    # case_setup : setup information of the case
    # sol_phys   : solid physics information
    # mesh_data  : mesh of the problem
    # section    : data about the section
    # exmass     : extra mass
    # -----------------------------------------------------------------------------
    # mesh_point : nodes of the mesh
    # mesh_elem  : elements of the mesh
    # mesh_mark  : markers of the mesh, boundaries to apply conditions and nodes to visualize information
    # num_elem   : number of beam elements
    # num_point  : number of beam nodes     
    mesh_point = mesh_data.point
    mesh_elem  = mesh_data.elem
    mesh_mark  = mesh_data.marker
    num_elem   = mesh_data.num_elem
    num_point  = mesh_data.num_point
    # qq_global_prima : variables/displacements of the system
    qq_global_prima = np.nan*np.ones((9*num_point,))   
    # -------------------------------------------------------------------------
    # Definition of the stiffness matrix 
    # KK_global_prima : stiffness matrix before applying the joints
    # MM_global_prima : mass matrix before applying the joints
    # LL_vec          : distance between nodes vector
    # rr_vec          : vectorial distance between nodes
    # rot_mat         : rotation matrix of each beam element
    # rotT_mat        : transposed rotation matrix of each beam element
    # Lrot_mat        : complete rotation matrix of the system
    # LrotT_mat       : complete transposed rotation matrix of the system
    # LL_vec_p        : distance between nodes vector for each node in the element
    # rr_vec_p        : vectorial distance between nodes for each node in the element
    # rot_mat_p       : rotation matrix of each beam element for each node in the element
    # rotT_mat_p      : transposed rotation matrix of each beam element for each node in the element
    # ref_axis        : reference axis
    KK_global_prima,LL_vec,rr_vec,rot_mat,rotT_mat,\
        Lrot_mat,LrotT_mat,LL_vec_p,rr_vec_p,rot_mat_p,rot_matT_p,sol_phys  = define_stiffness(mesh_elem,sol_phys,mesh_point,num_point,num_elem,mesh_data.mesh_elem_lock,mesh_data.RR_ctrl,section) 
    mesh_data.LL                                                   = LL_vec
    mesh_data.rr                                                   = rr_vec
    mesh_data.rr_p                                                 = rr_vec_p
    MM_global_prima,mass_data1                                      = define_mass(mesh_elem,sol_phys,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,section)
    mass_data2 = 0*mass_data1
    mass_data3 = mass_data2
    if exmass != []:
        KK_global_prima_exstiffpoint                                   = define_exstiffpoint(exstiff,mesh_point,mesh_elem,num_point,Lrot_mat,LrotT_mat)
        KK_global_prima_exstiffelem                                    = define_exstiffdist(mesh_elem,exstiff,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
        KK_global_prima                                               += KK_global_prima_exstiffpoint + KK_global_prima_exstiffelem
        MM_global_prima_exmass ,mass_data2                                         = define_exmassdist(mesh_elem,exmass,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
        MM_global_prima_exmasspoint,mass_data3                                     = define_exmasspoint(exmass,mesh_point,num_point,Lrot_mat,LrotT_mat)
        MM_global_prima                                               += MM_global_prima_exmass+MM_global_prima_exmasspoint
    # ind_RR  : index to delete from KK_LL
    # q_RR    : displacement in the restricted nodes 
    ind_RR = []
    q_RR = [] 
    # Find the restricted nodes
    ind_RR,q_RR = boundaryconditions_vibrest(mesh_mark,case_setup,num_point,ind_RR,q_RR)
    # -------------------------------------------------------------------------
    # Activate the information about the section internal nodes
    if mesh_data.section == "YES":   
        # section_globalCS  : section nodes in global coordinate system
        section_globalCS = def_secglob(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point)   
    KK_global, MM_global, RR_global, case_setup, ref_axis = init_boundaryconditions_tran(KK_global_prima,MM_global_prima,[],mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p) 
    # MM_pow    : mass matrix power to 0.5
    # MM_pow2   : mass matrix power to -0.5
    # w_mode    : mode angular frequecies 
    # mode      : mode shape
    # idx       : index to sort modes
    # freq_mode : frequency                                        
    freq_mode, w_mode, mode, MM_pow, MM_pow2,idx, MM_LL, KK_LL, ind_RR2, mode_tot = vib_linsys(KK_global,MM_global,RR_global,ind_RR,q_RR)
    # Store the solution
    # solution : class containing the solution data
    solution = store_solution_vib(case_setup,w_mode,mode,freq_mode,idx,MM_pow,MM_pow2,MM_LL,KK_LL,KK_global,ind_RR2,sol_phys,section,rot_mat,rotT_mat,num_point,num_elem,mesh_point,mesh_elem,mode_tot)
    solution.mass = mass_data1+mass_data2+mass_data3
    if case_setup.savefile == 'YES':
        # save the information in files
        save_vib(case_setup,mesh_data,section_globalCS,solution)  
    if case_setup.grav_fl == 1:
        weight = 0
        for ii_point in np.arange(mesh_data.num_point):
            weight_vec = 0
            for jj_point in np.arange(3):
                weight_vec += (case_setup.grav_g[jj_point]*MM_global[ii_point*9+jj_point,ii_point*9+jj_point])**2
            weight += np.sqrt(weight_vec)
        solution.weight = weight
    return solution, section_globalCS
#%%
def solver_ael_mod(case_setup,sol_phys,mesh_data,section,exmass,exstiff):
    # Function to solve the modal vibration structural problem
    # case_setup : setup information of the case
    # sol_phys   : solid physics information
    # mesh_data  : mesh of the problem
    # section    : data about the section
    # exmass     : extra mass
    # -----------------------------------------------------------------------------
    # mesh_point : nodes of the mesh
    # mesh_elem  : elements of the mesh
    # mesh_mark  : markers of the mesh, boundaries to apply conditions and nodes to visualize information
    # num_elem   : number of beam elements
    # num_point  : number of beam nodes     
    mesh_point = mesh_data.point
    mesh_elem  = mesh_data.elem
    mesh_mark  = mesh_data.marker
    num_elem   = mesh_data.num_elem
    num_point  = mesh_data.num_point
    # qq_global_prima : variables/displacements of the system
    qq_global_prima = np.nan*np.ones((9*num_point,))   
    # -------------------------------------------------------------------------
    # Definition of the stiffness matrix 
    # KK_global_prima : stiffness matrix before applying the joints
    # MM_global_prima : mass matrix before applying the joints
    # LL_vec          : distance between nodes vector
    # rr_vec          : vectorial distance between nodes
    # rot_mat         : rotation matrix of each beam element
    # rotT_mat        : transposed rotation matrix of each beam element
    # Lrot_mat        : complete rotation matrix of the system
    # LrotT_mat       : complete transposed rotation matrix of the system
    # LL_vec_p        : distance between nodes vector for each node in the element
    # rr_vec_p        : vectorial distance between nodes for each node in the element
    # rot_mat_p       : rotation matrix of each beam element for each node in the element
    # rotT_mat_p      : transposed rotation matrix of each beam element for each node in the element
    # ref_axis        : reference axis
    FF_global_prima        = np.zeros((9*num_point,))
    KK_global_prima,LL_vec,rr_vec,rot_mat,rotT_mat,\
        Lrot_mat,LrotT_mat,LL_vec_p,rr_vec_p,rot_mat_p,rot_matT_p,sol_phys  = define_stiffness(mesh_elem,sol_phys,mesh_point,num_point,num_elem,mesh_data.mesh_elem_lock,mesh_data.RR_ctrl,section) 
    mesh_data.LL                                                   = LL_vec
    mesh_data.rr                                                   = rr_vec
    mesh_data.rr_p                                                 = rr_vec_p
    MM_global_prima,mass_data1                                      = define_mass(mesh_elem,sol_phys,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,section)
    mass_data2 = 0*mass_data1
    mass_data3 = mass_data2
    if exmass != []:
        KK_global_prima_exstiffpoint                                   = \
            define_exstiffpoint(exstiff,mesh_point,mesh_elem,num_point,Lrot_mat,LrotT_mat)
        KK_global_prima_exstiffelem                                    = define_exstiffdist(mesh_elem,exstiff,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
        KK_global_prima                                               += KK_global_prima_exstiffpoint + KK_global_prima_exstiffelem
        MM_global_prima_exmass ,mass_data2                                         = define_exmassdist(mesh_elem,exmass,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
        MM_global_prima_exmasspoint,mass_data3                                     = define_exmasspoint(exmass,mesh_point,num_point,Lrot_mat,LrotT_mat)
        MM_global_prima                                               += MM_global_prima_exmass+MM_global_prima_exmasspoint
    if mesh_data.section == "YES":   
        # section_globalCS  : section nodes in global coordinate system
        section_globalCS = def_secglob(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point) 
    KK_global, MM_global, RR_global,  case_setup, ref_axis                    = init_boundaryconditions_tran(KK_global_prima,MM_global_prima, [],mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)
    vibfree, CC_global = define_damp(case_setup,sol_phys,mesh_data,section,MM_global,KK_global,solver_vib_mod,solver_struct_stat,num_point,mesh_mark) 
    # ind_RR  : index to delete from KK_LL
    # q_RR    : displacement in the restricted nodes 
    ind_RR = []
    q_RR = [] 
    # Find the restricted nodes
    ind_RR,q_RR = boundaryconditions_vibrest(mesh_mark,case_setup,num_point,ind_RR,q_RR)
    # -------------------------------------------------------------------------
    # Activate the information about the section internal nodes   
    # MM_pow    : mass matrix power to 0.5
    # MM_pow2   : mass matrix power to -0.5
    # w_mode    : mode angular frequecies 
    # mode      : mode shape
    # idx       : index to sort modes
    # freq_mode : frequency 
    tol = 1e-2
    frelax = 0.95
#    fact_mat   = 9*num_point
#    ind_RR_ae = np.zeros((2*len(ind_RR),))  
#    ind_RR_ae[:len(ind_RR)] = ind_RR
#    ind_RR_ae[len(ind_RR):] = np.array(ind_RR)+fact_mat
#    ind_RR_ae = ind_RR_ae.tolist()
#    ind_RR_ae = [ int(xx) for xx in ind_RR_ae ]
#    q_RR_ae = np.zeros((2*len(ind_RR),)) 
#    q_RR_ae[:len(ind_RR)] = q_RR     
#    q_RR_ae = q_RR_ae.tolist() 
    wfreq = np.zeros((2*vibfree.nsol,))
    for aux_mod in np.arange(len(vibfree.freq_mode)):
        wfreq[2*aux_mod] = vibfree.w_mode[aux_mod]
        wfreq[2*aux_mod+1] = vibfree.w_mode[aux_mod]
    KK_global = KK_global.astype('complex')
    CC_global = CC_global.astype('complex')
    MM_global = MM_global.astype('complex')
    freq_mode = np.zeros((2*vibfree.nsol,case_setup.numv))
    l_mode = np.zeros((2*vibfree.nsol,case_setup.numv), dtype=complex)
    w_mode = np.zeros((2*vibfree.nsol,case_setup.numv))
    mode = np.zeros((2*vibfree.nsol,2*vibfree.nsol,case_setup.numv), dtype=complex)
    mode_tot = np.zeros((num_point*9,2*vibfree.nsol,case_setup.numv), dtype=complex)
    MM_pow = def_vec_param(case_setup.numv)
    MM_pow2 = def_vec_param(case_setup.numv)
    MM_LL = def_vec_param(case_setup.numv)
    KK_LL = def_vec_param(case_setup.numv)
    v_infvec = np.linspace(case_setup.vmin,case_setup.vmax,case_setup.numv)
    lenmodes = np.min([case_setup.n_mod,2*vibfree.nsol])
    for iiv_inf in np.arange(case_setup.numv):
        v_inf = v_infvec[iiv_inf]
        for aux_mod in np.arange(2*len(vibfree.freq_mode)):
            error_theo = 1
            if aux_mod < lenmodes:
                err_num = 0
                while error_theo > tol:
                    KK_global_ae,CC_global_ae,MM_global_ae = boundaryconditions_aelmod(KK_global,CC_global,MM_global,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p,wfreq[aux_mod],v_inf)    
                    l_mode_i, freq_mode_i, w_mode_i, mode_i, MM_pow_i, MM_pow2_i,idx_i, MM_LL_i, KK_LL_i, ind_RR2_i, mode_tot_i = vibae_linsys(KK_global_ae,CC_global_ae,MM_global_ae,RR_global,ind_RR,q_RR,num_point)  # vibaemod_linsys(KK_global_ae,CC_global_ae,MM_global_ae,RR_global,ind_RR,q_RR,num_point,vibfree,case_setup) # 
#                    if aux_mod == 0:
#                        idx_i0 = idx_i
                    w_mode_i = w_mode_i[idx_i]
                    l_mode_i = l_mode_i[idx_i]
                    freq_mode_i = freq_mode_i[idx_i]
                    mode_i = mode_i[:,idx_i]
                    mode_tot_i = mode_tot_i[:,idx_i]
#                    if abs(wfreq[aux_mod]) < 1e-3:
#                        break
#                    else:
                    error_theo = abs(abs(w_mode_i[aux_mod])-wfreq[aux_mod])/abs(wfreq[aux_mod])
                    wfreq[aux_mod] = (1-frelax)*wfreq[aux_mod]+frelax*abs(w_mode_i[aux_mod])
                    print('Mode: '+ str(aux_mod) + ' calculating... error: '+ str(error_theo))
                    err_num += 1
                    if err_num > 5:
                        break
            if aux_mod > 0 and iiv_inf > 0:
                if abs(np.imag(l_mode_i[aux_mod])-np.imag(l_mode[aux_mod-1,iiv_inf])) < abs(np.imag(l_mode_i[aux_mod]))*0.01 and abs(np.real(l_mode_i[aux_mod])-np.real(l_mode[aux_mod-1,iiv_inf])) < abs(np.real(l_mode_i[aux_mod]))*0.01:
                    aux_mod2 = aux_mod-1
                else:
                    aux_mod2 = aux_mod
            else:
                aux_mod2 = aux_mod
            freq_mode[aux_mod,iiv_inf] = freq_mode_i[aux_mod2] 
            l_mode[aux_mod,iiv_inf] = l_mode_i[aux_mod2] 
            w_mode[aux_mod,iiv_inf] = w_mode_i[aux_mod2] 
            mode[:,aux_mod,iiv_inf] = mode_i[:,aux_mod2] 
            mode_tot[:,aux_mod,iiv_inf] = mode_tot_i[:,aux_mod2] 
            MM_pow[iiv_inf].append(MM_pow_i)
            MM_pow2[iiv_inf].append(MM_pow2_i)
            MM_LL[iiv_inf].append(MM_LL_i)
            KK_LL[iiv_inf].append(KK_LL_i)
        print('calculating aeroelastic eigenvalues: ' + str((iiv_inf+1)/case_setup.numv*100) + '%')
    # Store the solution
    # solution : class containing the solution data
    solution = store_solution_vibae(case_setup,l_mode,w_mode,mode,freq_mode,MM_pow,MM_pow2,MM_LL,KK_LL,KK_global,ind_RR2_i,sol_phys,section,rot_mat,rotT_mat,num_point,num_elem,mesh_point,mesh_elem,mode_tot)
    solution.mass = mass_data1+mass_data2+mass_data3
    solution.vinf = v_infvec
    if case_setup.savefile == 'YES':
        # save the information in files
        save_vib(case_setup,mesh_data,section_globalCS,solution)   
    if case_setup.grav_fl == 1:
        weight = 0
        for ii_point in np.arange(mesh_data.num_point):
            weight_vec = 0
            for jj_point in np.arange(3):
                weight_vec += (case_setup.grav_g[jj_point]*MM_global[ii_point*9+jj_point,ii_point*9+jj_point])**2
            weight += np.sqrt(weight_vec)
        solution.weight = weight
    return solution, section_globalCS


def solver_fly_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff):
    mesh_data = copy.deepcopy(mesh_data)
    tol = 1e-2
    aoa_axis = np.cross(case_setup.vinf,case_setup.grav_g)
    aoa_axis /= np.linalg.norm(aoa_axis)
    ang = 0
    vinf_2 = case_setup.vinf.copy()
    error_lift = 1
    ctrl_values = np.zeros((len(case_setup.flyctrl),))
    iiexp_values = np.zeros((len(case_setup.flyctrl),))
    ii_exp = 0
    mesh_data.RR_ctrl_base = mesh_data.RR_ctrl.copy()
    matctrl_ant = mesh_data.RR_ctrl.copy()
    fl_exp = 0
    ii_exp = 0
    startaoa = 0
    while abs(error_lift) > tol:
        solution, section_globalCS = solver_aero_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
        force_mat = np.zeros((len(solution.ext_fz),3))
        force_mat[:,0] = solution.ext_fx
        force_mat[:,1] = solution.ext_fy
        force_mat[:,2] = solution.ext_fz
        force_liftweight = sum(force_mat)
        force_liftweightval = np.dot(force_liftweight,case_setup.grav_g)
        error_lift = force_liftweightval/solution.weight
        if startaoa == 1:
            startaoa = 0
            if np.sign(force_liftweightval)==np.sign(force_liftweightval2) and abs(force_liftweightval) > abs(force_liftweightval2):
                sign_lift = -1
            else:
                sign_lift = 1
        else:
            sign_lift = 1
        if force_liftweightval > 0:
            if fl_exp == 2:
                ii_exp += 1
                ang += sign_lift*2**(-ii_exp)*np.pi/180
            else:
                ang += sign_lift*2**(-ii_exp)*np.pi/180
            fl_exp = 1
        else:
            if fl_exp == 1:
                ii_exp += 1
                ang -= sign_lift*2**(-ii_exp)*np.pi/180
            else:
                ang -= sign_lift*2**(-ii_exp)*np.pi/180
            fl_exp = 2
        force_liftweightval2 = force_liftweightval
        case_setup.vinf = vinf_2.copy()
        RR_aoa = [[np.cos(ang)+aoa_axis[0]**2*(1-np.cos(ang)),aoa_axis[0]*aoa_axis[1]*(1-np.cos(ang))-aoa_axis[2]*np.sin(ang),aoa_axis[0]*aoa_axis[2]*(1-np.cos(ang))+aoa_axis[1]*np.sin(ang)],\
                   [aoa_axis[0]*aoa_axis[1]*(1-np.cos(ang))-aoa_axis[2]*np.sin(ang), np.cos(ang)+aoa_axis[1]**2*(1-np.cos(ang)), aoa_axis[1]*aoa_axis[2]*(1-np.cos(ang))-aoa_axis[0]*np.sin(ang)],\
                   [aoa_axis[0]*aoa_axis[2]*(1-np.cos(ang))+aoa_axis[1]*np.sin(ang),aoa_axis[1]*aoa_axis[2]*(1-np.cos(ang))-aoa_axis[0]*np.sin(ang),np.cos(ang)+aoa_axis[2]**2*(1-np.cos(ang))]]
        case_setup.vinf = np.matmul(RR_aoa,case_setup.vinf)
    print('Error aoa: ',str(error_lift)+ '; AoA:' , str(ang*180/np.pi))
    ii_ctrl = 0
    ctrl_trim  = np.zeros((len(case_setup.flyctrl),))
    finish = np.zeros((len(case_setup.flyctrl),))
    for ctrl in case_setup.flyctrl:
        if finish[ii_ctrl] == 0:
            index_ctrl = 0
            fl_exp_ctrl = 0
            ii_exp_ctrl = iiexp_values[ii_ctrl]
            ctrl_val = 0
            error_ctrl = 1
            ii_exp_ctrl = 0
            startctrl = 0
            sign_mom_proj = 1
            count_sign = 0
            while abs(error_ctrl) > tol:
                solution, section_globalCS = solver_aero_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
                ctrl_trim[ii_ctrl] = ctrl_val
                dist_max = np.mean(np.linalg.norm(mesh_data.point-mesh_data.cdg,axis=1))
                moment_mat = np.zeros((len(solution.ext_mz),3))
                moment_mat[:,0] = solution.ext_mx+(mesh_data.point[:,1]-mesh_data.cdg[1])*solution.ext_fz-(mesh_data.point[:,2]-mesh_data.cdg[2])*solution.ext_fy
                moment_mat[:,1] = solution.ext_my+(mesh_data.point[:,2]-mesh_data.cdg[2])*solution.ext_fx-(mesh_data.point[:,0]-mesh_data.cdg[0])*solution.ext_fz
                moment_mat[:,2] = solution.ext_mz+(mesh_data.point[:,0]-mesh_data.cdg[0])*solution.ext_fy-(mesh_data.point[:,1]-mesh_data.cdg[1])*solution.ext_fx
                mom_tot = sum(moment_mat)
                if index_ctrl > 0:
                    if ctrl.obj == 'MOMENT_X':
                        mom_proj = np.dot(mom_tot,[1,0,0])
                    elif  ctrl.obj == 'MOMENT_Y':
                        mom_proj = np.dot(mom_tot,[0,1,0])
                    elif  ctrl.obj == 'MOMENT_Z':
                        mom_proj = np.dot(mom_tot,[0,0,1])
                else:
                    if ctrl.obj == 'MOMENT_X':
                        mom_proj = np.dot(mom_tot,[1,0,0])
                    elif  ctrl.obj == 'MOMENT_Y':
                        mom_proj = np.dot(mom_tot,[0,1,0])
                    elif  ctrl.obj == 'MOMENT_Z':
                        mom_proj = np.dot(mom_tot,[0,0,1])
                mom_proj2 = mom_proj.copy()
                error_ctrl = mom_proj/dist_max/solution.weight
                if abs(error_ctrl) < tol:
                    finish[ii_ctrl] = 1
                    break
                else:
                    if index_ctrl > 0:
                        if startctrl == 1:
                            if (np.sign(mom_proj) == np.sign(mom_proj2) and abs(mom_proj2) < abs(mom_proj)): 
                                count_sign += 1
                            else:
                                count_sign = 0
                            if count_sign == 5:
                                sign_mom_proj *= -1
                        if mom_proj > 0:
                            if fl_exp_ctrl == 2:
                                ii_exp_ctrl += 1
                                ctrl_val -= sign_mom_proj*2**(-ii_exp_ctrl)*np.pi/180
                            else:
                                ctrl_val -= sign_mom_proj*2**(-ii_exp_ctrl)*np.pi/180
                            fl_exp_ctrl = 1
                        else:
                            if fl_exp_ctrl == 1:
                                ii_exp_ctrl += 1
                                ctrl_val += sign_mom_proj*2**(-ii_exp_ctrl)*np.pi/180
                            else:
                                ctrl_val += sign_mom_proj*2**(-ii_exp_ctrl)*np.pi/180
                            fl_exp_ctrl = 2
                    else:
                        startctrl = 1
                        if mom_proj > 0:
                            ctrl_val -= 2**(-ii_exp_ctrl)*np.pi/180
                        else:
                            ctrl_val += 2**(-ii_exp_ctrl)*np.pi/180
                    for elem in mesh_data.elem:
                        for jj_ctrl in np.arange(len(ctrl.mark)): 
                            for marker in mesh_data.marker:
                                if ctrl.mark[jj_ctrl] == marker.name:
                                    if len(np.where(marker.node==elem[1])[0])>0 or len(np.where(marker.node==elem[2])[0])>0:
                                        ctrl_axis = mesh_data.point[int(elem[2]),:]-mesh_data.point[int(elem[1]),:]
                                        ctrl_val_sg = ctrl.sign[jj_ctrl]*ctrl_val
                                        matctr    = np.array([[np.cos(ctrl_val_sg)+ctrl_axis[0]**2*(1-np.cos(ctrl_val_sg)),ctrl_axis[0]*ctrl_axis[1]*(1-np.cos(ctrl_val_sg))-ctrl_axis[2]*np.sin(ctrl_val_sg),ctrl_axis[0]*ctrl_axis[2]*(1-np.cos(ctrl_val_sg))+ctrl_axis[1]*np.sin(ctrl_val_sg)],\
                                                           [ctrl_axis[0]*ctrl_axis[1]*(1-np.cos(ctrl_val_sg))-ctrl_axis[2]*np.sin(ctrl_val_sg), np.cos(ctrl_val_sg)+ctrl_axis[1]**2*(1-np.cos(ctrl_val_sg)), ctrl_axis[1]*ctrl_axis[2]*(1-np.cos(ctrl_val_sg))-ctrl_axis[0]*np.sin(ctrl_val_sg)],\
                                                           [ctrl_axis[0]*ctrl_axis[2]*(1-np.cos(ctrl_val_sg))+ctrl_axis[1]*np.sin(ctrl_val_sg),ctrl_axis[1]*ctrl_axis[2]*(1-np.cos(ctrl_val_sg))-ctrl_axis[0]*np.sin(ctrl_val_sg),np.cos(ctrl_val_sg)+ctrl_axis[2]**2*(1-np.cos(ctrl_val_sg))]])
                                        mesh_data.RR_ctrl[int(elem[0])] = np.matmul(np.matmul(matctr,matctrl_ant[int(elem[0])]),mesh_data.RR_ctrl_base[int(elem[0])])
                index_ctrl += 1
                fl_exp = 0
                ii_exp = 0
                startaoa = 0
                if index_ctrl % 5 == 0:
                    error_lift = 1
                    while abs(error_lift) > tol:
                        solution, section_globalCS = solver_aero_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
                        force_mat = np.zeros((len(solution.ext_fz),3))
                        force_mat[:,0] = solution.ext_fx
                        force_mat[:,1] = solution.ext_fy
                        force_mat[:,2] = solution.ext_fz
                        force_liftweight = sum(force_mat)
                        force_liftweightval = np.dot(force_liftweight,case_setup.grav_g)
                        error_lift = force_liftweightval/solution.weight
                        if startaoa == 1:
                            startaoa = 0
                            if np.sign(force_liftweightval)==np.sign(force_liftweightval2) and abs(force_liftweightval) > abs(force_liftweightval2):
                                sign_lift = -1
                            else:
                                sign_lift = 1
                        else:
                            sign_lift = 1
                        if force_liftweightval > 0:
                            if fl_exp == 2:
                                ii_exp += 1
                                ang += sign_lift*2**(-ii_exp)*np.pi/180
                            else:
                                ang += sign_lift*2**(-ii_exp)*np.pi/180
                            fl_exp = 1
                        else:
                            if fl_exp == 1:
                                ii_exp += 1
                                ang -= sign_lift*2**(-ii_exp)*np.pi/180
                            else:
                                ang -= sign_lift*2**(-ii_exp)*np.pi/180
                            fl_exp = 2
                        force_liftweightval2 = force_liftweightval
                        case_setup.vinf = vinf_2.copy()
                        RR_aoa = [[np.cos(ang)+aoa_axis[0]**2*(1-np.cos(ang)),aoa_axis[0]*aoa_axis[1]*(1-np.cos(ang))-aoa_axis[2]*np.sin(ang),aoa_axis[0]*aoa_axis[2]*(1-np.cos(ang))+aoa_axis[1]*np.sin(ang)],\
                                   [aoa_axis[0]*aoa_axis[1]*(1-np.cos(ang))-aoa_axis[2]*np.sin(ang), np.cos(ang)+aoa_axis[1]**2*(1-np.cos(ang)), aoa_axis[1]*aoa_axis[2]*(1-np.cos(ang))-aoa_axis[0]*np.sin(ang)],\
                                   [aoa_axis[0]*aoa_axis[2]*(1-np.cos(ang))+aoa_axis[1]*np.sin(ang),aoa_axis[1]*aoa_axis[2]*(1-np.cos(ang))-aoa_axis[0]*np.sin(ang),np.cos(ang)+aoa_axis[2]**2*(1-np.cos(ang))]]
                        case_setup.vinf = np.matmul(RR_aoa,case_setup.vinf)
            print('Error ctrl: ',str(error_ctrl)+ '; Ctrl:' , str(ctrl_val*180/np.pi))
            print('Error aoa: ',str(error_lift)+ '; AoA:' , str(ang*180/np.pi))
        force_mat = np.zeros((len(solution.ext_fz),3))
        force_mat[:,0] = solution.ext_fx
        force_mat[:,1] = solution.ext_fy
        force_mat[:,2] = solution.ext_fz
        force_liftweight = sum(force_mat)
        force_liftweightval = np.dot(force_liftweight,case_setup.grav_g)
        error_lift = force_liftweightval/solution.weight
        ctrl_values[ii_ctrl] += ctrl_val
        iiexp_values[ii_ctrl] = ii_exp_ctrl
        matctrl_ant = mesh_data.RR_ctrl
        ii_ctrl += 1
    solution.aoa_trim = ang
    solution.ctrl_trim = ctrl_trim
    return solution, section_globalCS



def solver_fly_struc_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff):
    solution, section_globalCS = solver_fly_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    ang = solution.aoa_trim
    ctrl_trim = solution.ctrl_trim
    solution, section_globalCS = solver_struct_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
    solution.aoa_trim = ang
    solution.ctrl_trim = ctrl_trim
    return solution, section_globalCS

#%%
    

#%% Functions
def solver_ael_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff):
#    print('Starting steady...')
    # Function to solve the static structural problem
    # case_setup : setup information of the case
    # sol_phys   : solid physics information
    # mesh_data  : mesh of the problem
    # section    : data about the section
    # -------------------------------------------------------------------------
    # mesh_point      : nodes of the beam mesh
    # mesh_elem       : elements of the beam mesh
    # mesh_mark       : markers of the beam mesh (nodes in which the boundary conditions are applied)
    # num_elem        : number of elements of the beam mesh
    # num_point       : number of nodes of the beam mesh
    # FF_global_prima : forces of the system
    # qq_global_prima : variables/displacements of the system
    mesh_point      = mesh_data.point
    mesh_elem       = mesh_data.elem
    mesh_mark       = mesh_data.marker
    num_elem        = mesh_data.num_elem
    num_point       = mesh_data.num_point
    FF_global_prima = np.zeros((9*num_point,))
    qq_global_prima = np.nan*np.ones((9*num_point,))    
    # -------------------------------------------------------------------------
    # Definition of the stiffness matrix    
    KK_global_prima,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,LL_vec_p,rr_vec_p,rot_mat_p,rot_matT_p,sol_phys = define_stiffness(mesh_elem,sol_phys,mesh_point,num_point,num_elem,mesh_data.mesh_elem_lock,mesh_data.RR_ctrl,section)  
    mesh_data.LL = LL_vec
    mesh_data.rr = rr_vec
    mesh_data.rr_p = rr_vec_p
    KK_global_prima_exstiffpoint  = define_exstiffpoint(exstiff,mesh_point,mesh_elem,num_point,Lrot_mat,LrotT_mat)
    KK_global_prima_exstiffelem   = define_exstiffdist(mesh_elem,exstiff,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
    KK_global_prima              += KK_global_prima_exstiffpoint + KK_global_prima_exstiffelem
    MM_global_prima,mass_data1     = define_mass(mesh_elem,sol_phys,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,section)
    MM_global_prima_exmass,mass_data2         = define_exmassdist(mesh_elem,exmass,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat)
    MM_global_prima_exmasspoint,mass_data3    = define_exmasspoint(exmass,mesh_point,num_point,Lrot_mat,LrotT_mat)
    MM_global_prima              += MM_global_prima_exmass+MM_global_prima_exmasspoint
    # -------------------------------------------------------------------------
    # Activate the information about the section internal nodes
    if mesh_data.section == "YES":    
        section_globalCS = def_secglob(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point)                
    # The boundary conditions are applied on the matrices and vectors
    # KK_global   : stiffness matrix after applying the boundary conditions
    # qq_global   : displacement vector after applying the boundary conditions
    # FF_global   : load vector ater applying the boundary conditions   
    tol = 1e-2
    error = 1
    class disp_values:
        phi        = np.zeros((num_point,))
        psi        = np.zeros((num_point,))
        theta      = np.zeros((num_point,))
    q_global2  = np.zeros((9*num_point,))
    F_global2  = np.zeros((9*num_point,))
    q_global = qq_global_prima.copy()
    flag_ini = 0
    FF2_global = np.zeros((9*num_point,))
    while error > tol:          
        KK_global, qq_global, FF_global, FF_orig, RR_global, disp_values, mesh_data = boundaryconditions_ael(KK_global_prima,MM_global_prima,FF_global_prima,qq_global_prima,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p,disp_values)
        # Solution of the static linear system
        q_global, FF_global, q_vec = stat_linsys(KK_global,qq_global,FF_global,RR_global)
#        if flag_ini == 0:
#            flag_ini = 1
#            q_global = Deltaq_global
        # store solutions
        if np.linalg.norm(q_global2)>0:
            error = np.linalg.norm(q_global-q_global2)/np.linalg.norm(q_global2)
        else:
            error = 1
        if error < 1:
            frelax = 0.1-0.152003*np.log(error)
        else:
            frelax = 0.1
        q_global = frelax*q_global + (1-frelax)*q_global2
        q_global2 = q_global.copy() 
        solution = store_solution_stat(q_global,FF_global,mesh_point,mesh_elem,section,sol_phys,rot_mat,rotT_mat,q_vec, disp_values)   
        disp_values.phi        = solution.phi
        disp_values.psi        = solution.psi
        disp_values.theta      = solution.theta
#        if flag_ini == 0:
#            flag_ini = 1
#        else:
    solution.mass = mass_data1+mass_data2+mass_data3
    # Check if a solution file must be created
    if case_setup.savefile == 'YES':
        save_stat(case_setup,mesh_data,section_globalCS,solution) 
#    print('... Ending steady')
    if case_setup.grav_fl == 1:
        weight = 0
        for ii_point in np.arange(mesh_data.num_point):
            weight_vec = 0
            for jj_point in np.arange(3):
                weight_vec += (case_setup.grav_g[jj_point]*MM_global_prima[ii_point*9+jj_point,ii_point*9+jj_point])**2
            weight += np.sqrt(weight_vec)
        solution.weight = weight
    return solution, section_globalCS

#%%

def solver_fly_ael_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff):
    tol = 1e-2
    ii_iter = 0
    error = 1
    while error > tol:
        solution, section_globalCS = solver_fly_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
        ang = solution.aoa_trim
        ctrl_trim = solution.ctrl_trim
        solution, section_globalCS = solver_ael_stat(case_setup,sol_phys,mesh_data,section,exmass,exstiff)
        if ii_iter == 1:
            error = abs(ang-ang2)
        else:
            ii_iter = 1
        ang2 = ang
        print('Iteration Error: '+ str(error))
    solution.aoa_trim = ang
    solution.ctrl_trim = ctrl_trim
    return solution, section_globalCS