# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 11:34:43 2020

@author       : Andres Cremades Botella - ancrebo@mot.upv.es
bc_functions  : file containing the boundary condition functions for the beam element solver
last_version  : 17-02-2021
modified_by   : Andres Cremades Botella
"""

import numpy as np
import pandas as pd
from scipy import interpolate
import tensorflow as tf
from tensorflow import keras                                                   
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import time
import random
from scipy.fftpack import fft, fftfreq, ifft
from scipy.special import jv,yv
from scipy import signal
import sys

#%% Function
def time_func(bound,time_val):   
    # If a time function is loaded to the boundary condition
    # SIN   : sinusoid function
    # SIGM  : sigmoid function
    # TABLE : table of values (periodic or not)
    # --------------------------------------------------------
    # func  : value of the time fuction
    if bound.func == "SIN:":
        func   = bound.ampl*np.sin(2*np.pi*bound.freq*time_val+bound.phas)
        funcder = 2*np.pi*bound.freq*bound.ampl*np.cos(2*np.pi*bound.freq*time_val+bound.phas)
        funcderder   = -(2*np.pi*bound.freq)**2*bound.ampl*np.sin(2*np.pi*bound.freq*time_val+bound.phas)
    elif bound.func == "SIGM:":
        func   = 1/(1+np.e**(-4*bound.slp*(time_val-bound.m_poin)))
        funcder = 4*np.e**((-4*bound.slp*(time_val-bound.m_poin)))*bound.slp/(1+np.e**(-4*bound.slp*(time_val-bound.m_poin)))
        funcderder = 32*np.e**((-8*bound.slp*(time_val-bound.m_poin)))*bound.slp**2/(1+np.e**(-4*bound.slp*(time_val-bound.m_poin)))**3-16*np.e**((-4*bound.slp*(time_val-bound.m_poin)))*bound.slp**2/(1+np.e**(-4*bound.slp*(time_val-bound.m_poin)))**2
    elif bound.func == "TABLE:":
        f_time = time_val
        try:
            len(bound.t_func)
        except:
            # bound.datafunc  : data of the table
            # bound.t_func    : time vector of the data
            # bound.f_func    : time vector of the fuction
            # bound.func_func : interpolated function of the table
            bound.datafunc  = pd.read_csv(case_setup.root + bound.file_tran) 
            bound.datafunc  = datafunc.values
            bound.t_func    = datafunc[:,0]
            bound.f_func    = datafunc[:,1]
            try:
                bound.f_funcder = datafunc[:,2]
            except:
                bound.f_funcder = 0*bound.f_func
            try:
                bound.f_funcderder = datafunc[:,3]
            except:
                bound.f_funcderder = 0*bound.f_func
            bound.func_func = interpolate.interp1d(t_func,f_func)
            bound.func_funcder = interpolate.interp1d(t_func,f_funcder)
            bound.func_funcderder = interpolate.interp1d(t_func,f_funcderder)
        if bound.periodic == "YES":
            if  f_time > bound.t_func[-1]:
                # ncyclefun : cycle number of the oscillation
                ncyclefun = np.floor(f_time/bound.t_func[-1])
                f_time    = f_time - bound.t_func[-1]*ncyclefun
                f_der_time = f_der_time - bound.t_funcder[-1]*ncyclefun
                f_derder_time = f_derder_time - bound.t_funcderder[-1]*ncyclefun
        else: 
            if  f_time > bound.t_func[-1]:
                f_time = bound.t_func[-1]
                f_der_time = bound.t_funcder[-1]
                f_derder_time = bound.t_funcderder[-1]
        func = bound.func_func(f_time)
        funcder = bound.func_funcder(f_time)
        funcderder = bound.func_funcderder(f_time)
    else:
        func = 1
        funcder = 0
        funcderder = 0
    return bound,func,funcder,funcderder

#%%
def theodorsen(k_red_vec):
    F_theo = np.zeros((len(k_red_vec),))
    G_theo = np.zeros((len(k_red_vec),))
    C_theo = np.zeros((len(k_red_vec),),dtype='complex')
    for kk in np.arange(len(k_red_vec)):
        k_red = k_red_vec[kk]
        if k_red == 0:
            C_theo[kk] = 1
#            F_theo[kk] = 1
#            G_theo[kk] = 0
        else:
            C_theo[kk] = (jv(1,k_red)-1j*yv(1,k_red))/((jv(1,k_red)-1j*yv(1,k_red))+1j*(jv(0,k_red)-1j*yv(0,k_red)))
#            F_theo[kk] = (jv(1,k_red)*(jv(1,k_red)+yv(0,k_red))+yv(1,k_red)*(yv(1,k_red)-jv(0,k_red)))/((jv(1,k_red)+yv(0,k_red))**2+(yv(1,k_red)-jv(0,k_red))**2)
#            G_theo[kk] = -(yv(1,k_red)*yv(0,k_red)+jv(1,k_red)*jv(0,k_red))/((jv(1,k_red)+yv(0,k_red))**2+(yv(1,k_red)-jv(0,k_red))**2)
#    C_theo = F_theo + 1j*G_theo
    return C_theo

#%%
def ref_axis_aero(bound,mark,polar_bc,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,v_induced,vdir0,vdir1,aoa0_geom,aoa1_geom,rr_vec_p,rot_matT_p,chord_vec,eff_3d,f_flap):
    # bound      : information about the boundary condition
    # mark       : information about the marker
    # polar_bc   : information about the aerodynamic polar
    # refaxis_0  : moment reference axis of the element 0 of the node
    # refaxis_1  : moment reference axis of the element 1 of the node
    # refaxis2_0 : lift reference axis of the element 0 of the node
    # refaxis2_1 : lift reference axis of the element 1 of the node
    # v_induced  : induced velocity
    # r_vinduced : nondimensional radius
    # vdir0      : velocity vector in the element 0 of the node
    # vdir1      : velocity vector in the element 0 of the node
    # aoa0_geom  : geometric angle of attack in the element 0 of the node
    # aoa1_geom  : geometric angle of attack in the element 1 of the node
    # rr_vec_p   : element vector, distance between nodes
    # rot_matT_p : transposed rotation matrix
    # chord_vec  : vector of the chord
    # For all the nodes of the liftin surface
    refaxis_0 = refaxis_0.copy()
    refaxis_1 = refaxis_1.copy()
    refaxis2_0 = refaxis2_0.copy()
    refaxis2_1 = refaxis2_1.copy()
    v_induced = v_induced.copy()
    vdir0     = vdir0.copy()
    vdir1     = vdir1.copy()
    aoa0_geom = aoa0_geom.copy()
    aoa1_geom = aoa1_geom.copy()
    rr_vec_p  = rr_vec_p.copy()
    rot_matT_p = rot_matT_p.copy()
    chord_vec = chord_vec.copy()
    vec_vrot0 = refaxis_0.copy()
    vec_vrot1 = refaxis_1.copy()
    aaaa= []
    for iiaero_node in np.arange(len(mark.node)):
        iimark_node = int(mark.node.flatten()[iiaero_node])
        if iimark_node >= 0:
            # Define the moment reference axis as the oposite of the element vector
            refaxis_0[iiaero_node,:]  = -rr_vec_p[0,iimark_node,:]/np.linalg.norm(rr_vec_p[0,iimark_node,:])
            # Depending of the aerodynamic 3D model, the direction of the velocity is calculated
            if polar_bc.eff3d == 'BEM':
                # vec_vrot0   : vector of the linear velocity of the rotation
                # vec_veltot0 : total velocity vector
                vec_vrot0[iiaero_node,:] = np.cross(refaxis_0[iiaero_node,:],bound.refrot)/np.linalg.norm(np.cross(refaxis_0[iiaero_node,:],bound.refrot))
                vec_veltot0              = bound.vinf*bound.vdir+v_induced[iiaero_node,:]+(bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node]+eff_3d.v_induced_tan0[iiaero_node,:])*vec_vrot0[iiaero_node,:]
                aaaa.append(np.arctan2((chord_vec[iiaero_node,1]/np.linalg.norm(chord_vec[iiaero_node,:])),chord_vec[iiaero_node,0]/np.linalg.norm(chord_vec[iiaero_node,:])))
                aoa0_geom[iiaero_node]   = np.arctan2(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_vrot0[iiaero_node,:])[1],np.matmul(rot_matT_p[iimark_node,0,:,:],vec_vrot0[iiaero_node,:])[0])-np.arctan2(np.matmul(rot_matT_p[iimark_node,0,:,:],chord_vec[iiaero_node,:])[1]/np.linalg.norm(chord_vec[iiaero_node,:]),np.matmul(rot_matT_p[iimark_node,0,:,:],chord_vec[iiaero_node,:])[0]/np.linalg.norm(chord_vec[iiaero_node,:])) #np.arctan2(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_vrot0[iiaero_node,:])[1],np.matmul(rot_matT_p[iimark_node,0,:,:],vec_vrot0[iiaero_node,:])[0])-np.arctan2((chord_vec[iiaero_node,1]/np.linalg.norm(chord_vec[iiaero_node,:])),chord_vec[iiaero_node,0]/np.linalg.norm(chord_vec[iiaero_node,:]))
                if np.linalg.norm(vec_veltot0) == 0:
                    vdir0[iiaero_node,:] = bound.vdir/np.linalg.norm(bound.vdir)
                else:
                    vdir0[iiaero_node,:] = (vec_veltot0-np.dot(vec_veltot0,refaxis_0[iiaero_node,:])*refaxis_0[iiaero_node,:])/np.linalg.norm((vec_veltot0-np.dot(vec_veltot0,refaxis_0[iiaero_node,:])*refaxis_0[iiaero_node,:]))
            else:
                vdir0[iiaero_node,:]   = (bound.vinf*bound.vdir-np.dot(bound.vinf*bound.vdir,refaxis_0[iiaero_node,:])*refaxis_0[iiaero_node,:])/np.linalg.norm(bound.vinf*bound.vdir-np.dot(bound.vinf*bound.vdir,refaxis_0[iiaero_node,:])*refaxis_0[iiaero_node,:])
                aoa0_geom[iiaero_node] = np.arctan2(np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)[1],np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)[0])-np.arctan2(np.matmul(rot_matT_p[iimark_node,0,:,:],chord_vec[iiaero_node,:])[1]/np.linalg.norm(chord_vec[iiaero_node,:]),np.matmul(rot_matT_p[iimark_node,0,:,:],chord_vec[iiaero_node,:])[0]/np.linalg.norm(chord_vec[iiaero_node,:])) #np.arctan2(np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)[1],np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)[0])-np.arctan2((chord_vec[iiaero_node,1]/np.linalg.norm(chord_vec[iiaero_node,:])),chord_vec[iiaero_node,0]/np.linalg.norm(chord_vec[iiaero_node,:]))
            # Define the lift reference axis as the trihedral of the velocity vector and the moment vector
            refaxis2_0[iiaero_node,:] = np.cross(vdir0[iiaero_node,:],refaxis_0[iiaero_node,:])                            
            # If the node is in contact with more than one element
            if np.linalg.norm(rr_vec_p[1,iimark_node,:]) != 0:
                # Define the moment reference axis as the oposite of the element vector
                refaxis_1[iiaero_node,:] = -rr_vec_p[1,iimark_node,:]/np.linalg.norm(rr_vec_p[1,iimark_node,:])
                # Depending of the aerodynamic 3D model, the direction of the velocity is calculated
                if polar_bc.eff3d == 'BEM':
                    # vec_vrot1   : vector of the linear velocity of the rotation
                    # vec_veltot1 : total velocity vector
                    vec_vrot1[iiaero_node,:] = np.cross(refaxis_1[iiaero_node,:],bound.refrot)/np.linalg.norm(np.cross(refaxis_1[iiaero_node,:],bound.refrot))
                    vec_veltot1              = bound.vinf*bound.vdir+v_induced[iiaero_node,:]+(bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node]+eff_3d.v_induced_tan1[iiaero_node,:])*vec_vrot1[iiaero_node,:]
                    aoa1_geom[iiaero_node]   = np.arctan2(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_vrot0[iiaero_node,:])[1],np.matmul(rot_matT_p[iimark_node,1,:,:],vec_vrot0[iiaero_node,:])[0])-np.arctan2(np.matmul(rot_matT_p[iimark_node,1,:,:],chord_vec[iiaero_node,:])[1]/np.linalg.norm(chord_vec[iiaero_node,:]),np.matmul(rot_matT_p[iimark_node,1,:,:],chord_vec[iiaero_node,:])[0]/np.linalg.norm(chord_vec[iiaero_node,:])) #np.arctan2(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_vrot1[iiaero_node,:])[1],np.matmul(rot_matT_p[iimark_node,1,:,:],vec_vrot1[iiaero_node,:])[0])-np.arctan2((chord_vec[iiaero_node,:])[1]/np.linalg.norm(chord_vec[iiaero_node,:]),np.matmul(rot_matT_p[iimark_node,1,:,:],chord_vec[iiaero_node,:])[0]/np.linalg.norm(chord_vec[iiaero_node,:]))
                    if np.linalg.norm(vec_veltot1) == 0:
                        vdir1[iiaero_node,:] = bound.vdir/np.linalg.norm(bound.vdir)
                    else:
                        vdir1[iiaero_node,:] = (vec_veltot1-np.dot(vec_veltot1,refaxis_1[iiaero_node,:])*refaxis_1[iiaero_node,:])/np.linalg.norm(vec_veltot1-np.dot(vec_veltot1,refaxis_1[iiaero_node,:])*refaxis_1[iiaero_node,:])
                else:
                    vdir1[iiaero_node,:]   = (bound.vinf*bound.vdir-np.dot(bound.vinf*bound.vdir,refaxis_1[iiaero_node,:])*refaxis_1[iiaero_node,:])/np.linalg.norm(bound.vinf*bound.vdir-np.dot(bound.vinf*bound.vdir,refaxis_1[iiaero_node,:])*refaxis_1[iiaero_node,:])
                    aoa1_geom[iiaero_node] = np.arctan2(np.matmul(rot_matT_p[iimark_node,1,:,:],bound.vdir)[1],np.matmul(rot_matT_p[iimark_node,1,:,:],bound.vdir)[0])-np.arctan2(np.matmul(rot_matT_p[iimark_node,1,:,:],chord_vec[iiaero_node,:])[1]/np.linalg.norm(chord_vec[iiaero_node,:]),np.matmul(rot_matT_p[iimark_node,1,:,:],chord_vec[iiaero_node,:])[0]/np.linalg.norm(chord_vec[iiaero_node,:])) #np.arctan2(np.matmul(rot_matT_p[iimark_node,1,:,:],bound.vdir)[1],np.matmul(rot_matT_p[iimark_node,1,:,:],bound.vdir)[0])-np.arctan2((chord_vec[iiaero_node,1]/np.linalg.norm(chord_vec[iiaero_node,:])),chord_vec[iiaero_node,0]/np.linalg.norm(chord_vec[iiaero_node,:]))
                # Define the lift reference axis as the trihedral of the velocity vector and the moment vector
                refaxis2_1[iiaero_node,:] = np.cross(vdir1[iiaero_node,:],refaxis_1[iiaero_node,:])
            # In the case there is not a second element, define null vectors
            else:
                refaxis_1[iiaero_node,:]   = [0,0,0]
                refaxis2_1[iiaero_node,:]  = [0,0,0]
    eff_3d.vec_vrot0 = vec_vrot0
    eff_3d.vec_vrot1 = vec_vrot1
    if f_flap == 0:
        return refaxis_0, refaxis_1, refaxis2_0, refaxis2_1, vdir0, vdir1, aoa0_geom, aoa1_geom,eff_3d
    else:
        return  aoa0_geom, aoa1_geom

#%%
def converge_vtan(bound,mark,polar_bc,cdst_value0,cdst_value1,clst_value0,clst_value1,v_induced,LL_vec_p,eff_3d,b_chord,vtan_induced):
    # For every node in the lifting surface
    f_wrongerr = 0
    err = np.zeros((len(mark.node),))
    if np.dot(bound.vdir,bound.refrot) > 0:
        velref = bound.vinf
    else:
        velref = bound.vrot*bound.radius
    if eff_3d.errorout1 > eff_3d.errorout2:
        eff_3d.frelax_vt *= 0.8
        if eff_3d.frelax_vt < 0.1:
            eff_3d.frelax_vt = 0.1
    for iiaero_node in np.arange(len(mark.node)):
        vtan_before = vtan_induced[iiaero_node,:]
        if np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:]) > 0:
            flag = 0
            iimark_node = int(mark.node.flatten()[iiaero_node])
            if polar_bc.lltnodesflag == 1:
                if len(np.where(np.array(polar_bc.lltnodes)== iimark_node)[0]) > 0:
                    flag_calculate = 1
                else:
                    flag_calculate = 0                
            if flag_calculate == 1:
                dcq_dr_a0 = eff_3d.r_vinduced[iiaero_node]*bound.radius*bound.Nb*2*b_chord[iiaero_node]*bound.rho/2*(np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:])**2+np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])**2)*(clst_value0[iiaero_node]*np.sin(eff_3d.phi_vinduced0[iiaero_node])-cdst_value0[iiaero_node]*np.cos(eff_3d.phi_vinduced0[iiaero_node]))
                dcq_dr_a1 = eff_3d.r_vinduced[iiaero_node]*bound.radius*bound.Nb*2*b_chord[iiaero_node]*bound.rho/2*(np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot1[iiaero_node,:]+eff_3d.v_induced_tan1[iiaero_node,:])**2+np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])**2)*(clst_value1[iiaero_node]*np.sin(eff_3d.phi_vinduced1[iiaero_node])-cdst_value1[iiaero_node]*np.cos(eff_3d.phi_vinduced1[iiaero_node]))
                dcq_dr_a  = (dcq_dr_a0*LL_vec_p[0,iimark_node]+dcq_dr_a1*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
                if eff_3d.F_vinduced0[iiaero_node] == 0 or eff_3d.F_vinduced1[iiaero_node] == 0:
                    eff_3d.v_induced_tan0[iiaero_node,:] = np.sign(eff_3d.v_induced_tan0[iiaero_node,:])*abs(eff_3d.v_induced_tan0[iiaero_node,:])*eff_3d.vec_vrot0[iiaero_node,:]#abs(np.max(np.max(eff_3d.v_induced_tan0)))*eff_3d.vec_vrot0[iiaero_node,:]
                    eff_3d.v_induced_tan1[iiaero_node,:] = np.sign(eff_3d.v_induced_tan1[iiaero_node,:])*abs(eff_3d.v_induced_tan1[iiaero_node,:])*eff_3d.vec_vrot1[iiaero_node,:]#abs(np.max(np.max(eff_3d.v_induced_tan1)))*eff_3d.vec_vrot1[iiaero_node,:]
                else:
                    eff_3d.v_induced_tan0[iiaero_node,:] = eff_3d.frelax_vt*(dcq_dr_a0/(4*np.max([eff_3d.F_vinduced0[iiaero_node],1e-5])*(np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:]))*bound.rho*np.pi*(eff_3d.r_vinduced[iiaero_node]*bound.radius)**2)*eff_3d.vec_vrot0[iiaero_node,:])+(1-eff_3d.frelax_vt)*eff_3d.v_induced_tan0[iiaero_node,:]
                    eff_3d.v_induced_tan1[iiaero_node,:] = eff_3d.frelax_vt*(dcq_dr_a1/(4*np.max([eff_3d.F_vinduced1[iiaero_node],1e-5])*(np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:]))*bound.rho*np.pi*(eff_3d.r_vinduced[iiaero_node]*bound.radius)**2)*eff_3d.vec_vrot1[iiaero_node,:])+(1-eff_3d.frelax_vt)*eff_3d.v_induced_tan1[iiaero_node,:]
                    flag = 1
                dcq_dr_b0 = eff_3d.r_vinduced[iiaero_node]*bound.radius*bound.Nb*2*b_chord[iiaero_node]*bound.rho/2*(np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:])**2+np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])**2)*(clst_value0[iiaero_node]*np.sin(eff_3d.phi_vinduced0[iiaero_node])-cdst_value0[iiaero_node]*np.cos(eff_3d.phi_vinduced0[iiaero_node]))
                dcq_dr_b1 = eff_3d.r_vinduced[iiaero_node]*bound.radius*bound.Nb*2*b_chord[iiaero_node]*bound.rho/2*(np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot1[iiaero_node,:]+eff_3d.v_induced_tan1[iiaero_node,:])**2+np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])**2)*(clst_value1[iiaero_node]*np.sin(eff_3d.phi_vinduced1[iiaero_node])-cdst_value1[iiaero_node]*np.cos(eff_3d.phi_vinduced1[iiaero_node]))
                dcq_dr_b  = (dcq_dr_b0*LL_vec_p[0,iimark_node]+dcq_dr_b1*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
                vtan_new = (eff_3d.v_induced_tan0[iiaero_node,:]+eff_3d.v_induced_tan1[iiaero_node,:])/2 
                if flag == 1:
                    err[iiaero_node] = np.linalg.norm(vtan_new-vtan_before)/velref #(dcq_dr_a-dcq_dr_b)/dcq_dr_b
                if np.isnan(dcq_dr_a) == 1:
                    eff_3d.notconverged = 1
                    break
                vtan_induced[iiaero_node,:]  = vtan_new
            if np.linalg.norm(vtan_induced[iiaero_node,:])/np.linalg.norm((bound.vrot*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:]+bound.vinf*bound.vdir+v_induced[iiaero_node,:]))>0.4:
                f_wrongerr = 1
                break
    if f_wrongerr == 1:
        error_out = 0
        vtan_induced *= 0
    else:
        error_out = np.linalg.norm(err[polar_bc.lltnodes_ind])
    eff_3d.errorout2 = eff_3d.errorout1 
    eff_3d.errorout1 = error_out
    return eff_3d,error_out,vtan_induced
#%%
def effects_3daero_inner(polar_bc,bound,mesh_data,mark,eff_3d,lpos,LL_cum,LL_vec_p,v_induced,iiaero_node,iimark_node,tol_vind,cl_sec,b_chord,cl_alpha,cdst_value0,cdst_value1,clst_value0,clst_value1):
    # polar_bc    : polar boundary conditions
    # bound       : boundary condition information
    # mesh_data   : mesh information
    # mark        : marker information
    # eff_3d      : 3D aerodynamic effects information
    # lpos        : position from reference point
    # LL_cum      : length of the surface
    # LL_vec_p    : element distance corresponding to each node
    # iiaero_node : index of the node
    # iimark_node : index of the node of the marker
    # tol_vind    : tolerance for the induced velocity
    # cl_sec      : section lift
    # b_chord     : semichord
    # cl_alpha    : lift slope
    # clst_value0 : steady value of the lift coefficient for the element 0 of the node
    # clst_value1 : steady value of the lift coefficient for the element 1 of the node
    # cdst_value0 : steady value of the lift coefficient for the element 0 of the node
    # cdst_value1 : steady value of the lift coefficient for the element 1 of the node
    # If the 3D effects are accounted by Lifting Line Theory
    if polar_bc.eff3d == 'LLT':
        # If the node is in the surface tip, it should be deleted from the calculation
        # If not, the matrix is constructed (Lifting Line Theory)
        if polar_bc.lltnodesflag == 1:
            if len(np.where(np.array(polar_bc.lltnodes)== iimark_node)[0]) > 0:
                flag_calculate = 1
            else:
                flag_calculate = 0
                eff_3d.ind_del.append(iiaero_node*2)
                eff_3d.ind_del.append(iiaero_node*2+1)
        else:
            flag_calculate = 1
        if flag_calculate == 1:
            if lpos[iiaero_node] > 0.999*LL_cum:
                eff_3d.ind_del.append(iiaero_node*2)
                eff_3d.ind_del.append(iiaero_node*2+1)
            else:
                # For every node in the lifting surface a coefficient is added
                for jjaero_node in np.arange(len(mark.node)):
                    # nn : number of the coefficient LLT
                    nn                                    = jjaero_node+1
                    eff_3d.mat_pllt[iiaero_node*2,jjaero_node]   = np.sin(eff_3d.lpos_cir[iiaero_node]*nn)*(np.sin(eff_3d.lpos_cir[iiaero_node])+nn*cl_alpha[iiaero_node]*2*b_chord[iiaero_node]/(8*LL_cum))
                    eff_3d.mat_pllt[iiaero_node*2+1,jjaero_node] = np.sin((np.pi-eff_3d.lpos_cir[iiaero_node])*nn)*(np.sin(np.pi-eff_3d.lpos_cir[iiaero_node])+nn*cl_alpha[iiaero_node]*2*b_chord[iiaero_node]/(8*LL_cum))
                # For every node in the lifting surface an equation is added
                eff_3d.vec_pllt[iiaero_node*2]   = cl_sec*bound.l_ref*mesh_data.mesh_point_scale[iiaero_node][0]/(8*LL_cum)*np.sin(eff_3d.lpos_cir[iiaero_node])
                eff_3d.vec_pllt[iiaero_node*2+1] = cl_sec*bound.l_ref*mesh_data.mesh_point_scale[iiaero_node][0]/(8*LL_cum)*np.sin(eff_3d.lpos_cir[iiaero_node])
    elif polar_bc.eff3d == 'BEM':
        if polar_bc.lltnodesflag == 1:
            if len(np.where(np.array(polar_bc.lltnodes)== iimark_node)[0]) > 0:
                if polar_bc.flag_lltnodes_ind == 1:
                    polar_bc.lltnodes_ind.append(iiaero_node)
                flag_calculate = 1
            else:
                flag_calculate = 0
                eff_3d.ind_del.append(iiaero_node*2)
                eff_3d.ind_del.append(iiaero_node*2+1)
        else:
            flag_calculate = 1
        if flag_calculate == 1:
            if polar_bc.startbem == 1:
                finc = 0.5
            else:
                finc = 0.05
            alim = 1/3
            if abs(eff_3d.error[iiaero_node])>tol_vind/10: #/10
                # Calculate the distribution of thrust
                if np.dot(bound.vdir,bound.refrot) > 0:
                    velref = bound.vinf
                    aainduced = np.dot(v_induced[iiaero_node,:],bound.vdir)/bound.vinf
                    if aainduced == 0:
                        bbinduced = 0
                    else:
                        bbinduced = 2*aainduced+1e-5
                        error_aaind = polar_bc.vind_tol*10
                        lambdaind = bound.vrot*bound.radius/bound.vinf
                        lamitnum = 0
                        while error_aaind > polar_bc.vind_tol and lamitnum < polar_bc.vind_maxit :
                            bbinduced2 = bbinduced
                            bbinduced = 2*aainduced/(1-bbinduced**2*(1-aainduced*eff_3d.sign_vi)/(4*lambdaind**2*(bbinduced-aainduced)*eff_3d.sign_vi))
                            error_aaind = abs(bbinduced-bbinduced2)
                            lamitnum += 1
                            if np.isnan(bbinduced) == 1:
                                bbinduced = aainduced
                                lamitnum = polar_bc.vind_maxit
                    dct_dr_a0 = bound.Nb*2*b_chord[iiaero_node]*bound.rho/2*(np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:])**2+np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])**2)*(clst_value0[iiaero_node]*np.cos(eff_3d.phi_vinduced0[iiaero_node])+cdst_value0[iiaero_node]*np.sin(eff_3d.phi_vinduced0[iiaero_node]))
                    dct_dr_a1 = bound.Nb*2*b_chord[iiaero_node]*bound.rho/2*(np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot1[iiaero_node,:]+eff_3d.v_induced_tan1[iiaero_node,:])**2+np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])**2)*(clst_value1[iiaero_node]*np.cos(eff_3d.phi_vinduced1[iiaero_node])+cdst_value1[iiaero_node]*np.sin(eff_3d.phi_vinduced1[iiaero_node]))
                    if abs(aainduced)<abs(alim):
                        dct_dr_b0 = (2*np.pi*bound.rho*eff_3d.r_vinduced[iiaero_node]*bound.radius*bound.vinf**2*(1+aainduced)*eff_3d.F_vinduced0[iiaero_node]+1e-10)*bbinduced*eff_3d.sign_vi #(4*np.pi*bound.rho*eff_3d.r_vinduced[iiaero_node]*bound.radius*np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])*eff_3d.F_vinduced0[iiaero_node]+1e-10)*np.dot(v_induced[iiaero_node,:],bound.vdir)*eff_3d.sign_vi
                        dct_dr_b1 = (2*np.pi*bound.rho*eff_3d.r_vinduced[iiaero_node]*bound.radius*bound.vinf**2*(1+aainduced)*eff_3d.F_vinduced1[iiaero_node]+1e-10)*bbinduced*eff_3d.sign_vi #(4*np.pi*bound.rho*eff_3d.r_vinduced[iiaero_node]*bound.radius*np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])*eff_3d.F_vinduced1[iiaero_node]+1e-10)*np.dot(v_induced[iiaero_node,:],bound.vdir)*eff_3d.sign_vi
                    else:
                        dct_dr_b0 = np.pi*bound.rho*eff_3d.r_vinduced[iiaero_node]*bound.radius*bound.vinf**2*2*bbinduced*(1-aainduced*eff_3d.sign_vi/4*(5-3*aainduced*eff_3d.sign_vi))*eff_3d.sign_vi*(eff_3d.F_vinduced0[iiaero_node]+1e-10)   #1/2*bound.rho*bound.vinf**2*2*np.pi*bound.radius*eff_3d.r_vinduced[iiaero_node]*1.55594*(0.571509-0.286*(-np.dot(v_induced[iiaero_node,:],bound.vdir)/bound.vinf)+(-np.dot(v_induced[iiaero_node,:],bound.vdir)/bound.vinf)**2)*eff_3d.F_vinduced0[iiaero_node]
                        dct_dr_b1 = np.pi*bound.rho*eff_3d.r_vinduced[iiaero_node]*bound.radius*bound.vinf**2*2*bbinduced*(1-aainduced*eff_3d.sign_vi/4*(5-3*aainduced*eff_3d.sign_vi))*eff_3d.sign_vi*(eff_3d.F_vinduced0[iiaero_node]+1e-10)   #1/2*bound.rho*bound.vinf**2*2*np.pi*bound.radius*eff_3d.r_vinduced[iiaero_node]*1.55594*(0.571509-0.286*(-np.dot(v_induced[iiaero_node,:],bound.vdir)/bound.vinf)+(-np.dot(v_induced[iiaero_node,:],bound.vdir)/bound.vinf)**2)*eff_3d.F_vinduced0[iiaero_node]
                else:
                    velref = bound.vrot*bound.radius
                    dct_dr_a0 = bound.Nb*2*b_chord[iiaero_node]*bound.rho/2*(np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:])**2+np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])**2)*(clst_value0[iiaero_node]*np.cos(eff_3d.phi_vinduced0[iiaero_node])+cdst_value0[iiaero_node]*np.sin(eff_3d.phi_vinduced0[iiaero_node]))
                    dct_dr_a1 = bound.Nb*2*b_chord[iiaero_node]*bound.rho/2*(np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot1[iiaero_node,:]+eff_3d.v_induced_tan1[iiaero_node,:])**2+np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])**2)*(clst_value1[iiaero_node]*np.cos(eff_3d.phi_vinduced1[iiaero_node])+cdst_value1[iiaero_node]*np.sin(eff_3d.phi_vinduced1[iiaero_node]))
                    dct_dr_b0 = (4*np.pi*bound.rho*eff_3d.r_vinduced[iiaero_node]*bound.radius*np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])*eff_3d.F_vinduced0[iiaero_node]+1e-10)*np.dot(v_induced[iiaero_node,:],bound.vdir)*eff_3d.sign_vi
                    dct_dr_b1 = (4*np.pi*bound.rho*eff_3d.r_vinduced[iiaero_node]*bound.radius*np.linalg.norm(bound.vinf*bound.vdir+v_induced[iiaero_node,:])*eff_3d.F_vinduced1[iiaero_node]+1e-10)*np.dot(v_induced[iiaero_node,:],bound.vdir)*eff_3d.sign_vi
                eff_3d.err_dct_dr_induced0[iiaero_node] = (dct_dr_a0-dct_dr_b0)
                eff_3d.err_dct_dr_induced1[iiaero_node] = (dct_dr_a1-dct_dr_b1)
                eff_3d.err_dct_dr_induced[iiaero_node]  = (eff_3d.err_dct_dr_induced0[iiaero_node]*LL_vec_p[0,iimark_node]+eff_3d.err_dct_dr_induced1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
                if eff_3d.flag_start[iiaero_node] == 0 and eff_3d.flag_start2[iiaero_node] == 0: 
                    eff_3d.error1ct[iiaero_node] = eff_3d.err_dct_dr_induced[iiaero_node]
                    if eff_3d.flag_start1[iiaero_node] == 0:
                        eff_3d.v_induced2[iiaero_node,0] += finc*eff_3d.sign_vi*bound.vdir[0]/np.linalg.norm(bound.vdir)*velref
                        eff_3d.v_induced2[iiaero_node,1] += finc*eff_3d.sign_vi*bound.vdir[1]/np.linalg.norm(bound.vdir)*velref
                        eff_3d.v_induced2[iiaero_node,2] += finc*eff_3d.sign_vi*bound.vdir[2]/np.linalg.norm(bound.vdir)*velref
                        eff_3d.flag_start1[iiaero_node] = 1
                    v_induced[iiaero_node] = eff_3d.v_induced2[iiaero_node]
                    eff_3d.error[iiaero_node] = np.linalg.norm(eff_3d.v_induced1[iiaero_node,:] -eff_3d.v_induced2[iiaero_node,:])/velref
                    eff_3d.flag_start2[iiaero_node] = 1
                else:
                    if eff_3d.flag_start[iiaero_node] == 0:
                        eff_3d.error2ct[iiaero_node] = eff_3d.err_dct_dr_induced[iiaero_node] 
                        if np.sign(eff_3d.error1ct[iiaero_node]) == np.sign(eff_3d.error2ct[iiaero_node]):
                            if eff_3d.aux_vind_e%5 == 0:
                                eff_3d.v_induced1[iiaero_node,:] -=  finc*eff_3d.sign_vi*bound.vdir/np.linalg.norm(bound.vdir)*velref
                            else:
                                eff_3d.v_induced2[iiaero_node,:] +=  finc*eff_3d.sign_vi*bound.vdir/np.linalg.norm(bound.vdir)*velref
                            v_induced[iiaero_node,:]          =  eff_3d.v_induced1[iiaero_node,:]
                        else:
                            eff_3d.flag_start[iiaero_node] = 1
                            v_induced[iiaero_node,:]  = (eff_3d.v_induced1[iiaero_node,:]+eff_3d.v_induced2[iiaero_node,:])/2 
                        eff_3d.flag_start2[iiaero_node] = 0
                        eff_3d.error[iiaero_node] = np.linalg.norm(eff_3d.v_induced1[iiaero_node,:] -eff_3d.v_induced2[iiaero_node,:])/velref
                    else:
                        if np.sign(eff_3d.err_dct_dr_induced[iiaero_node]) == np.sign(eff_3d.error2ct[iiaero_node]):
                            eff_3d.error2ct[iiaero_node]       = eff_3d.err_dct_dr_induced[iiaero_node]
                            eff_3d.v_induced2[iiaero_node,:] = v_induced[iiaero_node,:] 
                            v_induced[iiaero_node,:]         = (eff_3d.v_induced1[iiaero_node,:]+eff_3d.v_induced2[iiaero_node,:])/2
                            eff_3d.error[iiaero_node] = np.linalg.norm(eff_3d.v_induced1[iiaero_node,:] -eff_3d.v_induced2[iiaero_node,:])/velref
                        elif np.sign(eff_3d.err_dct_dr_induced[iiaero_node]) == np.sign(eff_3d.error1ct[iiaero_node]):
                            eff_3d.error1ct[iiaero_node]       = eff_3d.err_dct_dr_induced[iiaero_node]
                            eff_3d.v_induced1[iiaero_node,:] = v_induced[iiaero_node,:] 
                            v_induced[iiaero_node,:]         = (eff_3d.v_induced1[iiaero_node,:]+eff_3d.v_induced2[iiaero_node,:])/2
                            eff_3d.error[iiaero_node] = np.linalg.norm(eff_3d.v_induced1[iiaero_node,:] -eff_3d.v_induced2[iiaero_node,:])/velref
        if np.isnan(np.linalg.norm(v_induced[iiaero_node,:]))==1: #> np.max([bound.vinf,bound.vrot*bound.radius]):
            eff_3d.notconverged = 1#v_induced[iiaero_node,:] *= np.max([bound.vinf,bound.vrot*bound.radius])/np.linalg.norm(v_induced[iiaero_node,:])
        if iiaero_node == len(mark.node)-1:
#            v_induced_mod = v_induced.copy()
#            v_ind_int = v_induced[polar_bc.lltnodes_ind,:]
#            l_ind_int = lpos[polar_bc.lltnodes_ind]
            polar_bc.flag_lltnodes_ind = 0
#            f_vxind = interpolate.interp1d(l_ind_int,v_ind_int[:,0],'cubic')
#            f_vyind = interpolate.interp1d(l_ind_int,v_ind_int[:,1],'cubic')
#            f_vzind = interpolate.interp1d(l_ind_int,v_ind_int[:,2],'cubic') 
#            for auxaeronode in np.arange(len(mark.node)):
#                v_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
#            v_induced = v_induced_mod.copy()
#            if iiaero_node ==14:
#                print(v_induced[iiaero_node,1])
#                print([dct_dr_a0-dct_dr_b0,dct_dr_a1-dct_dr_b1])lpos[polar_bc.lltnodes_ind]
#                print(1)
    return eff_3d,v_induced,polar_bc
#%%
def effects_3daero_outter(polar_bc,eff_3d,mark,Vinf0,Vinf1,LL_vec_p,v_induced,cltot,LL_cum,tol_vind,refaxis2_0,refaxis2_1,aoa_vec0,aoa_vec1,aoa_0lift,cd_aoa0,aoa_0liftflap,cd_aoa0flap,non_acac_dist,b_chord,b_chordflap,lpos,bound,aoa0_geom,aoa1_geom,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1):
    # polar_bc   : information about the polar boundary condition
    # eff_3d     : information about the 3D effects
    # mark       : information about the markers
    # Vinf0      : information about the free stream velocity of the element 0 of the node
    # Vinf1      : information about the free stream velocity of the element 1 of the node
    # LL_vec_p   : length of the element
    # v_induced  : induced velocity
    # cltot      : total lift coefficient
    # LL_cum     : total length of the surface
    # tol_vind   : tolerance of the induced velocity
    # refaxis2_0 : lift reference axis in element 0 of node
    # refaxis2_1 : lift reference axis in element 1 of node
    # If the 3D effects are accounted by Lifting Line Theory
    if polar_bc.eff3d == 'LLT':
        error2 = []
        # Tip points are deleted from the system of equations
        eff_3d.mat_pllt = np.delete(eff_3d.mat_pllt,eff_3d.ind_del,0)
        eff_3d.vec_pllt = np.delete(eff_3d.vec_pllt,eff_3d.ind_del,0)
        # To avoid problems in case of empty vector
        if len(eff_3d.ind_del)>0:
            eff_3d.mat_pllt =  eff_3d.mat_pllt[:,:-int(len(eff_3d.ind_del)/2)] 
        error      = abs(cltot-eff_3d.cltot2) 
        eff_3d.cltot2     = cltot
        # A_vec : vector of the coefficients of the LLT (lifting line theory)
        A_vec = np.matmul(np.linalg.inv(np.matmul(np.transpose(eff_3d.mat_pllt),eff_3d.mat_pllt)),np.matmul(np.transpose(eff_3d.mat_pllt),eff_3d.vec_pllt))
        # For every node in the lifting surface
        for iiaero_node in np.arange(len(mark.node)):
            iimark_node = int(mark.node.flatten()[iiaero_node])
            # The induced vorticity and velocity are initialized to 0
            nn            = 1
            v_induced_it0 = 0
            v_induced_it1 = 0
            gamma_it0     = 0
            gamma_it1     = 0
            # For every term in the summatory
            for aa in A_vec:
                # nn       : number of the coefficient LLT
                v_induced_it0 += Vinf0[iiaero_node]*nn*aa*np.sin(nn*(eff_3d.lpos_cir[iiaero_node]+1e-10))/np.sin(eff_3d.lpos_cir[iiaero_node]+1e-10)*refaxis2_0[iiaero_node,:]
                v_induced_it1 += Vinf1[iiaero_node]*nn*aa*np.sin(nn*(eff_3d.lpos_cir[iiaero_node]+1e-10))/np.sin(eff_3d.lpos_cir[iiaero_node]+1e-10)*refaxis2_1[iiaero_node,:]
                gamma_it0     += 4*LL_cum*Vinf0[iiaero_node]*aa*np.sin(nn*eff_3d.lpos_cir[iiaero_node])
                gamma_it1     += 4*LL_cum*Vinf1[iiaero_node]*aa*np.sin(nn*eff_3d.lpos_cir[iiaero_node])
                nn            += 1
            # A relaxation factor is included to stabilize the problem
            gamma_it                   = (gamma_it0*LL_vec_p[0,iimark_node]+gamma_it1*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
            v_induced_it               = (v_induced_it0*LL_vec_p[0,iimark_node]+v_induced_it1*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
            frelax                     = 0.2
            v_induced[iiaero_node,:]   = frelax*v_induced_it+(1-frelax)*v_induced[iiaero_node,:]
            eff_3d.Gamma_induced[iiaero_node] = frelax*gamma_it+(1-frelax)*eff_3d.Gamma_induced[iiaero_node]
    elif polar_bc.eff3d == 'BEM':
        f_wrongerr = 0
        if polar_bc.cor3dflag == 1:
#            if eff_3d.sign_vi > 0:
#                for iiaero_node in np.arange(len(polar_bc.cor3dnodes)):
#                    nnstall = 1
#                    cr = 2*b_chord/eff_3d.r_vinduced/bound.radius
#                    Kstall = (0.1517/cr)**(1/1.084)
#                    delta_aoa0 = (aoa_vec0[iiaero_node]-aoa_0lift[iiaero_node])*((Kstall/0.136)**nnstall - 1)
#                    delta_aoa1 = (aoa_vec1[iiaero_node]-aoa_0lift[iiaero_node])*((Kstall/0.136)**nnstall - 1)
#                    deltaflap_aoa0 = (aoa_vec0[iiaero_node]-aoa_0liftflap[iiaero_node])*((Kstall/0.136)**nnstall - 1)
#                    deltaflap_aoa1 = (aoa_vec1[iiaero_node]-aoa_0liftflap[iiaero_node])*((Kstall/0.136)**nnstall - 1)
#                    
#            else:
            for iiaero_node in np.arange(len(polar_bc.cor3dnodes)):
                clp0        = 2*np.pi*(aoa_vec0[iiaero_node]-aoa_0lift[iiaero_node])
                clp1        = 2*np.pi*(aoa_vec1[iiaero_node]-aoa_0lift[iiaero_node])
                cmp0        = non_acac_dist[iiaero_node]*2*np.pi*(aoa_vec0[iiaero_node]-aoa_0lift[iiaero_node])
                cmp1        = non_acac_dist[iiaero_node]*2*np.pi*(aoa_vec1[iiaero_node]-aoa_0lift[iiaero_node])
                clp0flap        = 2*np.pi*(aoa_vec0[iiaero_node]-aoa_0liftflap[iiaero_node])
                clp1flap        = 2*np.pi*(aoa_vec1[iiaero_node]-aoa_0liftflap[iiaero_node])
                cmp0flap        = non_acac_dist[iiaero_node]*2*np.pi*(aoa_vec0[iiaero_node]-aoa_0liftflap[iiaero_node])
                cmp1flap        = non_acac_dist[iiaero_node]*2*np.pi*(aoa_vec1[iiaero_node]-aoa_0liftflap[iiaero_node])
                aafac       = 2.2 # 3
                hhfac       = 1 #2
                nnfac       = 4
                factor0 = aafac*(b_chord[iiaero_node]/lpos[iiaero_node])**hhfac*np.cos(aoa0_geom[iiaero_node])**nnfac
                factor1 = aafac*(b_chord[iiaero_node]/lpos[iiaero_node])**hhfac*np.cos(aoa1_geom[iiaero_node])**nnfac
                clst_value0[iiaero_node] += factor0*(clp0-clst_value0[iiaero_node])
                clst_value1[iiaero_node] += factor1*(clp1-clst_value1[iiaero_node])
                cmst_value0[iiaero_node] += factor0*(cmp0-cmst_value0[iiaero_node])
                cmst_value1[iiaero_node] += factor1*(cmp1-cmst_value1[iiaero_node])
                cdst_value0[iiaero_node] -= factor0*(cdst_value0[iiaero_node]-cd_aoa0[iiaero_node])
                cdst_value1[iiaero_node] -= factor1*(cdst_value1[iiaero_node]-cd_aoa0[iiaero_node])
                cdst_value0[iiaero_node] -= 0.1*cd_aoa0[iiaero_node]
                cdst_value1[iiaero_node] -= 0.1*cd_aoa0[iiaero_node]  
                clstflap_value0[iiaero_node] += factor0*(clp0flap-clstflap_value0[iiaero_node])
                clstflap_value1[iiaero_node] += factor1*(clp1flap-clstflap_value1[iiaero_node])
                cmstflap_value0[iiaero_node] += factor0*(cmp0flap-cmstflap_value0[iiaero_node])
                cmstflap_value1[iiaero_node] += factor1*(cmp1flap-cmstflap_value1[iiaero_node])
                cdstflap_value0[iiaero_node] -= factor0*(cdstflap_value0[iiaero_node]-cd_aoa0flap[iiaero_node])
                cdstflap_value1[iiaero_node] -= factor1*(cdstflap_value1[iiaero_node]-cd_aoa0flap[iiaero_node])
                cdstflap_value0[iiaero_node] -= 0.1*cd_aoa0flap[iiaero_node]
                cdstflap_value1[iiaero_node] -= 0.1*cd_aoa0flap[iiaero_node]   
                if np.linalg.norm(v_induced[iiaero_node,:])>np.linalg.norm((bound.vrot*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:]+bound.vinf*bound.vdir+v_induced[iiaero_node,:])):
                    f_wrongerr = 1
            if f_wrongerr == 1:
                v_induced *=0
                eff_3d.err_dct_dr_induced *= 0       
        # measure the error
        error  = eff_3d.error[polar_bc.lltnodes_ind]
        error2 = error.copy()
        error = sum(error)/len(error)
    else:
        error = tol_vind/10
        error2 = []
    return error, eff_3d, v_induced,error2

#%%
def interp_aeroconditions(case_setup,bound,polar_bctot,mesh_data,marktot,polar_bctot_ind,indep_interp0st,indep_interp1st,indep_interp0,indep_interp1,indep_interp0b,indep_interp1b,aoa_0lift,aoa_0liftflap,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,v_induced,refaxis2_0,refaxis2_1,Vinf0,Vinf1,LL_vec_p,inc,lpos,LL_cum,cltot,eff_3d,tol_vind,b_chord,b_chordflap,cd_aoa0,cd_aoa0flap,f_flap,coreflapdist_adim):
    # case_setup      : information about the case configuration
    # bound           : boundary information
    # polar_bc        : information about the polar
    # mesh_data       : information about the mesh
    # mark            : information about the marker
    # polar_bc_ind    : number of independent polar boundary conditions
    # indep_interp0st : steady independent variable of the interpolation in element 0 of the node
    # indep_interp1st : steady independent variable of the interpolation in element 0 of the node
    # indep_interp0   : independent variable of the interpolation in element 0 of the node
    # indep_interp1   : independent variable of the interpolation in element 0 of the node
    # indep_interp0b  : independent variable of the interpolation in element 0 of the node after an increment
    # indep_interp1b  : independent variable of the interpolation in element 0 of the node after an increment
    # indel           : index to delete
    # aoa_0lift       : 0 lift angle of attack
    # cdst_value0     : drag coefficient value for element 0 of the node
    # cdst_value1     : drag coefficient value for element 1 of the node
    # clst_value0     : lift coefficient value for element 0 of the node
    # clst_value1     : lift coefficient value for element 1 of the node
    # cmst_value0     : moment coefficient value for element 0 of the node
    # cmst_value1     : moment coefficient value for element 1 of the node
    # v_induced       : induced velocity
    # refaxis2_0      : lift reference axis in element 0 of the node
    # refaxis2_1      : lift reference axis in element 1 of the node
    # Vinf0           : free stream velocity in element 0 of the node
    # Vinf1           : free stream velocity in element 1 of the node
    # LL_vec_p        : distance between nodes corresponding to each node
    # inc             : increment for derivatives
    # lpos            : position from the reference point
    # LL_cum          : total length of the surface
    # cltot           : total lift coefficient
    # eff_3d          : information of the 3d effects
    # tol_vind        : tolerance for the induced velocity
    # b_chord         : semichord
    # cd_ind0         : induced drag coefficient in the element 0 of the node
    # cd_ind1         : induced drag coefficient in the element 1 of the node
    if len(marktot) == 1:
        mark = marktot[0]
        polar_bc = polar_bctot[0]
        polar_bc_ind = polar_bctot_ind[0]
    else:
        mark = marktot[0]
        polar_bc = polar_bctot[0]
        polar_bc_ind = polar_bctot_ind[0]
        markflap = marktot[1]
        polar_bcflap = polar_bctot[1]
        polar_bcflap_ind = polar_bctot_ind[1]
    cl_alpha = np.zeros((len(np.arange(len(mark.node))),)) 
    cd_ind0      = np.zeros((len(mark.node),))
    cd_ind1      = np.zeros((len(mark.node),))
    cl_alphaflap = np.zeros((len(np.arange(len(mark.node))),)) 
    cd_ind0flap      = np.zeros((len(mark.node),))
    cd_ind1flap      = np.zeros((len(mark.node),))
    # For every node in the lifting surface
    for iiaero_node in np.arange(len(mark.node)):
        iimark_node = int(mark.node.flatten()[iiaero_node])
        # Calculate all the dependent parameters
        # limit the values of the interpolation inputs into the limits of the interpolation domain
        if case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0] > 0:
            fcorrerr00 = 1.0001
        else:
            fcorrerr00 = 1/1.0001
        if case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1] > 0:
            fcorrerr01 = 1.0001
        else:
            fcorrerr01 = 1/1.0001
        if polar_bc.nind == 2:
            if case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0] > 0:
                fcorrerr10 = 1.0001
            else:
                fcorrerr10 = 1/1.0001
            if case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1] > 0:
                fcorrerr11 = 1.0001
            else:
                fcorrerr11 = 1/1.0001
        if indep_interp0st[0][iiaero_node] < case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]:
            indep_interp0st[0][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]*fcorrerr00
        elif indep_interp0st[0][iiaero_node] > case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]:
            indep_interp0st[0][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]/fcorrerr01
        if indep_interp1st[0][iiaero_node] < case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]:
            indep_interp1st[0][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]*fcorrerr00
        elif indep_interp1st[0][iiaero_node] > case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]:
            indep_interp1st[0][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]/fcorrerr01
        if indep_interp0[0][iiaero_node] < case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]:
            indep_interp0b[0][iiaero_node] = (case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]+(indep_interp0b[0][iiaero_node]-indep_interp0[0][iiaero_node]))*fcorrerr00
            indep_interp0[0][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]*fcorrerr00
        elif indep_interp0b[0][iiaero_node] > case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]:
            indep_interp0[0][iiaero_node] = (case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]-(indep_interp0b[0][iiaero_node]-indep_interp0[0][iiaero_node]))/fcorrerr01
            indep_interp0b[0][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]/fcorrerr01
        if indep_interp1[0][iiaero_node] < case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]:
            indep_interp1b[0][iiaero_node] = (case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]+(indep_interp1b[0][iiaero_node]-indep_interp1[0][iiaero_node]))*fcorrerr00
            indep_interp1[0][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][0]*fcorrerr00
        elif indep_interp1b[0][iiaero_node] > case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]:
            indep_interp1[0][iiaero_node] = (case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]-(indep_interp1b[0][iiaero_node]-indep_interp1[0][iiaero_node]))/fcorrerr01
            indep_interp1b[0][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][0][1]/fcorrerr01
        if polar_bc.nind == 2:
            if indep_interp0st[1][iiaero_node] < case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]:
                indep_interp0st[1][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]*fcorrerr10
            elif indep_interp0st[1][iiaero_node] > case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]:
                indep_interp0st[1][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]/fcorrerr11
            if indep_interp1st[1][iiaero_node] < case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]:
                indep_interp1st[1][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]*fcorrerr10
            elif indep_interp1st[1][iiaero_node] > case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]:
                indep_interp1st[1][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]/fcorrerr11
            if indep_interp0[1][iiaero_node] < case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]:
                indep_interp0b[1][iiaero_node] = (case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]+(indep_interp0b[1][iiaero_node]-indep_interp0[1][iiaero_node]))*fcorrerr10
                indep_interp0[1][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]*fcorrerr10
            elif indep_interp0b[1][iiaero_node] > case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]:
                indep_interp0[1][iiaero_node] = (case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]-(indep_interp0b[1][iiaero_node]-indep_interp0[1][iiaero_node]))/fcorrerr11
                indep_interp0b[1][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]/fcorrerr11
            if indep_interp1[1][iiaero_node] < case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]:
                indep_interp1b[1][iiaero_node] = (case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]+(indep_interp1b[1][iiaero_node]-indep_interp1[1][iiaero_node]))*fcorrerr10
                indep_interp1[1][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][0]*fcorrerr10
            elif indep_interp1b[1][iiaero_node] > case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]:
                indep_interp1[1][iiaero_node] = (case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]-(indep_interp1b[1][iiaero_node]-indep_interp1[1][iiaero_node]))/fcorrerr11
                indep_interp1b[1][iiaero_node] = case_setup.polar[polar_bc_ind].limitint[iiaero_node][1][1]/fcorrerr11
        for n_dep in np.arange(len(polar_bc.dep)):
            # Drag coefficient
            if polar_bc.dep[n_dep] == "CD":
                if polar_bc.eff3d == 'LLT':
                    cd_ind0[iiaero_node] = eff_3d.Gamma_induced[iiaero_node]*np.dot(v_induced[iiaero_node,:],refaxis2_0[iiaero_node,:])/(0.5*Vinf0[iiaero_node]**2*bound.s_ref)
                    cd_ind1[iiaero_node] = eff_3d.Gamma_induced[iiaero_node]*np.dot(v_induced[iiaero_node,:],refaxis2_1[iiaero_node,:])/(0.5*Vinf1[iiaero_node]**2*bound.s_ref)
                if polar_bc.nind == 1:
                    cd_aoa0[iiaero_node] = case_setup.polar[polar_bc_ind].f_aoa0_drag[iiaero_node]
                    cdst_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node](indep_interp0[0][iiaero_node])+cd_ind0[iiaero_node]
                    cdst_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node](indep_interp1[0][iiaero_node])+cd_ind1[iiaero_node]
                elif polar_bc.nind == 2:
                    cd_aoa0[iiaero_node] = case_setup.polar[polar_bc_ind].f_aoa0_drag[iiaero_node](indep_interp0[1][iiaero_node])
#                    print([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])
                    try:
                        cdst_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]+cd_ind0[iiaero_node]
                        cdst_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]+cd_ind1[iiaero_node]
                    except:
                        cdst_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]+cd_ind0[iiaero_node]
                        cdst_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]+cd_ind1[iiaero_node]
                        
            # Lift coefficient
            elif polar_bc.dep[n_dep] == "CL":
                # clst_value0b : steady lift coefficient in node 0 with increment
                # clst_value1b : steady lift coefficient in node 1 with increment
                # cl_alpha     : derivative of the steady lift coefficient respect to the angle of attack
                # cl_sec       : lift coefficient of the section
                # cltot        : total lift coefficient
                if polar_bc.nind == 1:
                    aoa_0lift[iiaero_node]   = case_setup.polar[polar_bc_ind].f_aoa_0lift[iiaero_node]*np.pi/180
                    clst_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node](indep_interp0[0][iiaero_node])
                    clst_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node](indep_interp1[0][iiaero_node])
                    clst_value0st            = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node](indep_interp0st[0][iiaero_node])
                    clst_value1st            = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node](indep_interp1st[0][iiaero_node])
                    clst_value0b             = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node](indep_interp0b[0][iiaero_node])
                    clst_value1b             = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node](indep_interp1b[0][iiaero_node])
                elif polar_bc.nind == 2:
                    aoa_0lift[iiaero_node]   = case_setup.polar[polar_bc_ind].f_aoa_0lift[iiaero_node](indep_interp0[1][iiaero_node])*np.pi/180
                    clst_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]
                    clst_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]
                    clst_value0st            = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]
                    clst_value1st            = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]
                    clst_value0b             = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node]([indep_interp0b[0][iiaero_node],indep_interp0b[1][iiaero_node]])[0]
                    clst_value1b             = case_setup.polar[polar_bc_ind].cl_interp[iiaero_node]([indep_interp1b[0][iiaero_node],indep_interp1b[1][iiaero_node]])[0]
                cl_alpha[iiaero_node] = ((clst_value0b-clst_value0[iiaero_node])*LL_vec_p[0,iimark_node]+(clst_value1b-clst_value1[iiaero_node])*LL_vec_p[1,iimark_node])/inc/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
#                cl_sec   = (clst_value0st*LL_vec_p[0,iimark_node]+clst_value1st*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
#                cltot   += (clst_value0[iiaero_node]*mesh_data.mesh_point_scale[iimark_node][0]*LL_vec_p[0,iimark_node]+clst_value1[iiaero_node]*mesh_data.mesh_point_scale[iimark_node][0]*LL_vec_p[1,iimark_node])/LL_cum
            # Moment coefficient
            elif polar_bc.dep[n_dep] == "CM":
                if polar_bc.nind == 1:
                    cmst_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cm_interp[iiaero_node](indep_interp0[0][iiaero_node])
                    cmst_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cm_interp[iiaero_node](indep_interp1[0][iiaero_node])
                elif polar_bc.nind == 2:
                    cmst_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cm_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]
                    cmst_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cm_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]
            # Moment coefficient measured in the negative axis 
            elif polar_bc.dep[n_dep] == "-CM":
                if polar_bc.nind == 1:
                    cmst_value0[iiaero_node] = -case_setup.polar[polar_bc_ind].cm_interp[iiaero_node](indep_interp0[0][iiaero_node])
                    cmst_value1[iiaero_node] = -case_setup.polar[polar_bc_ind].cm_interp[iiaero_node](indep_interp1[0][iiaero_node])
                elif polar_bc.nind == 2:
                    cmst_value0[iiaero_node] = -case_setup.polar[polar_bc_ind].cm_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]
                    cmst_value1[iiaero_node] = -case_setup.polar[polar_bc_ind].cm_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]
        if f_flap == 1 and markflap.node.flatten()[iiaero_node] >= 0:
            iimarkflap_node = int(markflap.node.flatten()[iiaero_node])
            for n_dep in np.arange(len(polar_bcflap.dep)):
                # Drag coefficient
                if polar_bcflap.dep[n_dep] == "CD":
                    if polar_bcflap.eff3d == 'LLT':
                        cd_ind0flap[iiaero_node] = eff_3d.Gamma_induced[iiaero_node]*np.dot(v_induced[iiaero_node,:],refaxis2_0[iiaero_node,:])/(0.5*Vinf0[iiaero_node]**2*bound.s_ref)
                        cd_ind1flap[iiaero_node] = eff_3d.Gamma_induced[iiaero_node]*np.dot(v_induced[iiaero_node,:],refaxis2_1[iiaero_node,:])/(0.5*Vinf1[iiaero_node]**2*bound.s_ref)
                    if polar_bc.nind == 1:
                        cd_aoa0flap[iiaero_node] = case_setup.polar[polar_bc_ind].f_aoa0_drag[iiaero_node]
                        cdstflap_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node](indep_interp0[0][iiaero_node])+cd_ind0flap[iiaero_node]
                        cdstflap_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node](indep_interp1[0][iiaero_node])+cd_ind1flap[iiaero_node]
                    elif polar_bcflap.nind == 2:
                        cd_aoa0flap[iiaero_node] = case_setup.polar[polar_bc_ind].f_aoa0_drag[iiaero_node](indep_interp0[1][iiaero_node])
    #                    print([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])
                        try:
                            cdstflap_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]+cd_ind0flap[iiaero_node]
                            cdstflap_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]+cd_ind1flap[iiaero_node]
                        except:
                            cdstflap_value0[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]+cd_ind0flap[iiaero_node]
                            cdstflap_value1[iiaero_node] = case_setup.polar[polar_bc_ind].cd_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]+cd_ind1flap[iiaero_node]
                            
                # Lift coefficient
                elif polar_bcflap.dep[n_dep] == "CL":
                    # clst_value0b : steady lift coefficient in node 0 with increment
                    # clst_value1b : steady lift coefficient in node 1 with increment
                    # cl_alpha     : derivative of the steady lift coefficient respect to the angle of attack
                    # cl_sec       : lift coefficient of the section
                    # cltot        : total lift coefficient
                    if polar_bcflap.nind == 1:
                        aoa_0liftflap[iiaero_node]   = case_setup.polar[polar_bcflap_ind].f_aoa_0lift[iiaero_node]*np.pi/180
                        clstflap_value0[iiaero_node] = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node](indep_interp0[0][iiaero_node])
                        clstflap_value1[iiaero_node] = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node](indep_interp1[0][iiaero_node])
                        clstflap_value0st            = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node](indep_interp0st[0][iiaero_node])
                        clstflap_value1st            = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node](indep_interp1st[0][iiaero_node])
                        clstflap_value0b             = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node](indep_interp0b[0][iiaero_node])
                        clstflap_value1b             = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node](indep_interp1b[0][iiaero_node])
                    elif polar_bc.nind == 2:
                        aoa_0liftflap[iiaero_node]   = case_setup.polar[polar_bcflap_ind].f_aoa_0lift[iiaero_node](indep_interp0[1][iiaero_node])*np.pi/180
                        clstflap_value0[iiaero_node] = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]
                        clstflap_value1[iiaero_node] = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]
                        clstflap_value0st            = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]
                        clstflap_value1st            = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]
                        clstflap_value0b             = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node]([indep_interp0b[0][iiaero_node],indep_interp0b[1][iiaero_node]])[0]
                        clstflap_value1b             = case_setup.polar[polar_bcflap_ind].cl_interp[iiaero_node]([indep_interp1b[0][iiaero_node],indep_interp1b[1][iiaero_node]])[0]
                    cl_alphaflap[iiaero_node] = ((clstflap_value0b-clstflap_value0[iiaero_node])*LL_vec_p[0,iimark_node]+(clstflap_value1b-clstflap_value1[iiaero_node])*LL_vec_p[1,iimark_node])/inc/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
#                    cl_secflap   = (clst_value0st*LL_vec_p[0,iimark_node]+clst_value1st*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
#                    cltotflap   += (clst_value0[iiaero_node]*mesh_data.mesh_point_scale[iimark_node][0]*LL_vec_p[0,iimark_node]+clst_value1[iiaero_node]*mesh_data.mesh_point_scale[iimark_node][0]*LL_vec_p[1,iimark_node])/LL_cum
                # Moment coefficient
                elif polar_bcflap.dep[n_dep] == "CM":
                    if polar_bcflap.nind == 1:
                        cmstflap_value0[iiaero_node] = case_setup.polar[polar_bcflap_ind].cm_interp[iiaero_node](indep_interp0[0][iiaero_node])
                        cmstflap_value1[iiaero_node] = case_setup.polar[polar_bcflap_ind].cm_interp[iiaero_node](indep_interp1[0][iiaero_node])
                    elif polar_bc.nind == 2:
                        cmstflap_value0[iiaero_node] = case_setup.polar[polar_bcflap_ind].cm_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]
                        cmstflap_value1[iiaero_node] = case_setup.polar[polar_bcflap_ind].cm_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]
                # Moment coefficient measured in the negative axis 
                elif polar_bcflap.dep[n_dep] == "-CM":
                    if polar_bcflap.nind == 1:
                        cmstflap_value0[iiaero_node] = -case_setup.polar[polar_bcflap_ind].cm_interp[iiaero_node](indep_interp0[0][iiaero_node])
                        cmstflap_value1[iiaero_node] = -case_setup.polar[polar_bcflap_ind].cm_interp[iiaero_node](indep_interp1[0][iiaero_node])
                    elif polar_bc.nind == 2:
                        cmstflap_value0[iiaero_node] = -case_setup.polar[polar_bcflap_ind].cm_interp[iiaero_node]([indep_interp0[0][iiaero_node],indep_interp0[1][iiaero_node]])[0]
                        cmstflap_value1[iiaero_node] = -case_setup.polar[polar_bcflap_ind].cm_interp[iiaero_node]([indep_interp1[0][iiaero_node],indep_interp1[1][iiaero_node]])[0]  
        clst_tot_value0[iiaero_node] = clst_value0[iiaero_node]*b_chord[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])+clstflap_value0[iiaero_node]*b_chordflap[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])
        clst_tot_value1[iiaero_node] = clst_value1[iiaero_node]*b_chord[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])+clstflap_value1[iiaero_node]*b_chordflap[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])
        cdst_tot_value0[iiaero_node] = cdst_value0[iiaero_node]*b_chord[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])+cdstflap_value0[iiaero_node]*b_chordflap[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])
        cdst_tot_value1[iiaero_node] = cdst_value1[iiaero_node]*b_chord[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])+cdstflap_value1[iiaero_node]*b_chordflap[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])
        cmst_tot_value0[iiaero_node] = cmst_value0[iiaero_node]*b_chord[iiaero_node]**2/(b_chord[iiaero_node]+b_chordflap[iiaero_node])**2+cmstflap_value0[iiaero_node]*b_chordflap[iiaero_node]**2/(b_chord[iiaero_node]+b_chordflap[iiaero_node])**2+clst_tot_value0[iiaero_node]*coreflapdist_adim[iiaero_node]
        cmst_tot_value1[iiaero_node] = cmst_value1[iiaero_node]*b_chord[iiaero_node]**2/(b_chord[iiaero_node]+b_chordflap[iiaero_node])**2+cmstflap_value1[iiaero_node]*b_chordflap[iiaero_node]**2/(b_chord[iiaero_node]+b_chordflap[iiaero_node])**2+clst_tot_value1[iiaero_node]*coreflapdist_adim[iiaero_node]
        cl_sec   = (clst_tot_value0[iiaero_node]*LL_vec_p[0,iimark_node]+clst_tot_value1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
        cltot   += (clst_tot_value0[iiaero_node]*mesh_data.mesh_point_scale[iimark_node][0]*LL_vec_p[0,iimark_node]+clst_tot_value1[iiaero_node]*mesh_data.mesh_point_scale[iimark_node][0]*LL_vec_p[1,iimark_node])/LL_cum  
        eff_3d,v_induced,polar_bc = effects_3daero_inner(polar_bc,bound,mesh_data,mark,eff_3d,lpos,LL_cum,LL_vec_p,v_induced,iiaero_node,iimark_node,tol_vind,cl_sec,b_chord,cl_alpha,cdst_value0,cdst_value1,clst_value0,clst_value1)
        #print(cl_alpha)
    return indep_interp0st,indep_interp1st,indep_interp0,indep_interp1,indep_interp0b,indep_interp1b,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,cltot,eff_3d,cl_alpha,cl_alphaflap,aoa_0lift,aoa_0liftflap,cd_aoa0,cd_aoa0flap

#%%
def steady_aero(case_setup,bound,polar_bctot,mesh_data,marktot,polar_bctot_ind,aoa_vec0,aoa_vec1,aoa_vec0_st,aoa_vec1_st,pos_vec0,pos_vec1,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,aoa_0lift,aoa_0liftflap,v_induced,refaxis2_0,refaxis2_1,Vinf0,Vinf1,LL_vec_p,lpos,LL_cum,eff_3d,tol_vind,b_chord,non_acac_dist,b_chordflap,non_acac_distflap,coreflapdist_adim,aoa_mean0,aoa_mean1,cd_aoa0,cd_aoa0flap,aoa0_geom,aoa1_geom,reynolds0,reynolds1,deltaflap_vec0,deltaflap_vec1,f_flap):
    # case_setup    : information about the case configuration
    # bound         : boundary information
    # polar_bc      : information about the polar
    # mesh_data     : information about the mesh
    # mark          : information about the marker
    # polar_bc_ind  : number of independent polar boundary conditions
    # aoa_vec0      : angle of attack in the element 0 of the node
    # aoa_vec1      : angle of attack in the element 1 of the node
    # aoa_vec0_st   : angle of attack in the element 0 of the node
    # aoa_vec1_st   : angle of attack in the element 1 of the node
    # pos_vec0      : position vector in the element 0 of the node
    # pos_vec1      : position vector in the element 1 of the node
    # cdst_value0   : steady drag coefficient in the element 0 fo the node
    # cdst_value1   : steady drag coefficient in the element 1 fo the node
    # clst_value0   : steady lift coefficient in the element 0 fo the node
    # clst_value1   : steady lift coefficient in the element 1 fo the node
    # cmst_value0   : steady moment coefficient in the element 0 fo the node
    # cmst_value1   : steady moment coefficient in the element 1 fo the node
    # aoa_0lift     : 0 lift angle of attack
    # refaxis2_0    : lift reference axis of the element 0 of the node
    # refaxis2_1    : lift reference axis of the element 1 of the node
    # Vinf0         : free stream velocity in 0
    # Vinf1         : free stream velocity in 1
    # LL_vec_p      : distance between nodes corresponding to each node
    # lpos          : distance to the reference point
    # LL_cum        : total distance of the surface
    # eff_3d        : 3d effects
    # tol_vind      : tolerance for the induced velocity
    # b_chord       : semichord
    # non_acac_dist : nondimensional distance between the section aerodynamic reference center and the quarter of chord
    # If the polar or the polar+ann aerodynamic model is chosen 
    # cltot  : total lift coefficient
    if len(marktot) == 1:
        mark = marktot[0]
        polar_bc = polar_bctot[0]
        polar_bc_ind = polar_bctot_ind[0]
    else:
        mark = marktot[0]
        polar_bc = polar_bctot[0]
        polar_bc_ind = polar_bctot_ind[0]
        markflap = marktot[1]
        polar_bcflap = polar_bctot[1]
        polar_bcflap_ind = polar_bctot_ind[1]
    cltot  = 0   
    clst_tot_value0 = np.zeros((len(mark.node),)) 
    clst_tot_value1 = np.zeros((len(mark.node),)) 
    cdst_tot_value0 = np.zeros((len(mark.node),)) 
    cdst_tot_value1 = np.zeros((len(mark.node),)) 
    cmst_tot_value0 = np.zeros((len(mark.node),)) 
    cmst_tot_value1 = np.zeros((len(mark.node),))                                  
    if case_setup.aero_model == "POLAR" or case_setup.aero_model == "POLAR_ANN" or case_setup.aero_model == "THEO_POLAR":
        # indep_interp0     : independent variable in node 0
        # indep_interp1     : independent variable in node 1
        # indep_interp0b    : independent variable in node 0 with increment
        # indep_interp1b    : independent variable in node 1 with increment
        rotmatflag = 0
        indep_interp0     = []
        indep_interp1     = []
        indep_interp0st   = []
        indep_interp1st   = []
        indep_interp0b    = []
        indep_interp1b    = []
        # polar       : polar data
        # polar_indep : polar independent variables
        # polar_dep   : polar dependent variables
        # uniq_indep  : number of unique independent variables 1 of the polar (ie vertical positions)
        # tot_polar   : number of total points of the polar
        # l_polar     : number of points per polar
        npolars = len(marktot)
        for auxnpolars in np.arange(npolars):
            if auxnpolars == 0:
                pobc_ind = polar_bc_ind
            elif auxnpolars == 1:
                pobc_ind = polar_bcflap_ind
            polar = []
            if case_setup.polar[pobc_ind].created == 0:
                case_setup.polar[pobc_ind].created = 1
                case_setup.polar[pobc_ind].cd_interp = []
                case_setup.polar[pobc_ind].f_aoa0_drag = []
                case_setup.polar[pobc_ind].f_aoa_0lift = []
                case_setup.polar[pobc_ind].cl_interp = []
                case_setup.polar[pobc_ind].cm_interp = []
                case_setup.polar[pobc_ind].limitint = [] 
                kkpol = 0
                for iiaero_node in np.arange(len(mark.node)):
                    if polar_bc.pointfile[0] == -1:
                        polar = pd.read_csv(case_setup.root+polar_bctot[auxnpolars].file[0]).values
                    else:
                        if lpos[iiaero_node] <= polar_bc.sortedlref[0]:
                            polar = pd.read_csv(case_setup.root+polar_bctot[auxnpolars].file[0]).values
                        elif lpos[iiaero_node] >= polar_bc.sortedlref[-1]:
                            polar = pd.read_csv(case_setup.root+polar_bctot[auxnpolars].file[-1]).values
                        elif lpos[iiaero_node] <= polar_bc.sortedlref[kkpol]:
                            polar0 = pd.read_csv(case_setup.root+polar_bctot[auxnpolars].file[kkpol-1]).values
                            polar1 = pd.read_csv(case_setup.root+polar_bctot[auxnpolars].file[kkpol]).values
                            bp0 = lpos[iiaero_node]-polar_bctot[auxnpolars].sortedlref[kkpol-1]
                            bp1 = polar_bctot[auxnpolars].sortedlref[kkpol]-lpos[iiaero_node]
                            bpt = polar_bctot[auxnpolars].sortedlref[kkpol]-polar_bctot[auxnpolars].sortedlref[kkpol-1]
                            polar = polar0*bp1/bpt+polar1*bp0/bpt
                        else:
                            polar0 = pd.read_csv(case_setup.root+polar_bctot[auxnpolars].file[kkpol]).values
                            polar1 = pd.read_csv(case_setup.root+polar_bctot[auxnpolars].file[kkpol+1]).values
                            bp0 = lpos[iiaero_node]-polar_bctot[auxnpolars].sortedlref[kkpol]
                            bp1 = polar_bctot[auxnpolars].sortedlref[kkpol+1]-lpos[iiaero_node]
                            bpt = polar_bctot[auxnpolars].sortedlref[kkpol+1]-polar_bctot[auxnpolars].sortedlref[kkpol]
                            polar = polar0*bp1/bpt+polar1*bp0/bpt
                            kkpol += 1
                    polar_indep = polar[:,:len(polar_bc.indep)]
                    polar_dep   = polar[:,len(polar_bc.indep):]
                    uniq_indep  = len(set(polar_indep[:,0].tolist()))
                    tot_polar   = len(polar_indep)
                    l_polar     = int(tot_polar/uniq_indep)
                    # polar_indep_mesh  : polar independent data in matrix form (number of independent varibles)
                    # polar_dep_mesh    : polar dependent data in matrix form (number of independent varibles)
                    polar_indep_mesh = np.zeros((uniq_indep,l_polar,len(polar_bc.indep)))
                    polar_dep_mesh   = np.zeros((uniq_indep,l_polar,len(polar_bc.dep)))        # Set the number of independent variables           
                    if len(polar_indep[0]) == 1:
                        case_setup.polar[pobc_ind].nind = 1
                    elif len(polar_indep[0]) == 2:
                        case_setup.polar[pobc_ind].nind = 2
                    for ii_im in np.arange(len(polar_indep[0])):
                        auxii_im = 0
                        auxjj_im = 0
                        for jj_im in np.arange(len(polar_indep)):
                            polar_indep_mesh[auxjj_im,auxii_im,ii_im] = polar_indep[jj_im,ii_im]
                            auxii_im += 1
                            if auxii_im == l_polar:
                                auxii_im = 0
                                auxjj_im += 1
                    for ii_dm in np.arange(len(polar_dep[0])):
                        auxii_dm = 0
                        auxjj_dm = 0
                        for jj_dm in np.arange(len(polar_dep)):
                            polar_dep_mesh[auxjj_dm,auxii_dm,ii_dm] = polar_dep[jj_dm,ii_dm]
                            auxii_dm += 1
                            if auxii_dm == l_polar:
                                auxii_dm = 0
                                auxjj_dm += 1
                    for ii_indeppol in np.arange(len(polar_bc.indep)):
                        if polar_bc.indep[ii_indeppol] == 'ANGLE':
                            inc = np.pi/180
                            if ii_indeppol > 0:
                                rotmatflag = 1
                                polarindepmeshcopy = polar_indep_mesh.copy()
                                polar_indep_mesh = np.zeros((len(polarindepmeshcopy[0,:,0]),len(polarindepmeshcopy[:,0,0]),len(polarindepmeshcopy[0,0,:])))
                                polar_indep_mesh[:,:,0] = polarindepmeshcopy[:,:,1].transpose()
                                polar_indep_mesh[:,:,1] = polarindepmeshcopy[:,:,0].transpose()
                        elif polar_bc.indep[ii_indeppol] == 'ANGLE_DEG':
                            inc = np.pi/180
                            if ii_indeppol > 0:
                                rotmatflag = 1
                                polarindepmeshcopy = polar_indep_mesh.copy()
                                polar_indep_mesh = np.zeros((len(polarindepmeshcopy[0,:,0]),len(polarindepmeshcopy[:,0,0]),len(polarindepmeshcopy[0,0,:])))
                                polar_indep_mesh[:,:,0] = polarindepmeshcopy[:,:,1].transpose()
                                polar_indep_mesh[:,:,1] = polarindepmeshcopy[:,:,0].transpose()
                    # Store interpolations of the polar in the setup class
                    #   - .cd_interp : drag coefficient
                    #   - .cl_interp : lift coefficient
                    #   - .cm_interp : moment coefficient
                    # cd_mesh        : drag coefficient in the interpolation size
                    # cl_mesh        : lift coefficient in the interpolation size
                    # cm_mesh        : moment coefficient in the interpolation size
                    for ii_deppol in np.arange(len(polar_bc.dep)):
                        if len(polar_indep[0]) == 1:
                            case_setup.polar[pobc_ind].limitint.append([[min(polar_indep_mesh[:,0,0]),max(polar_indep_mesh[:,0,0])]])
                        elif len(polar_indep[0]) == 2:
                            case_setup.polar[pobc_ind].limitint.append([[min(polar_indep_mesh[:,0,0]),max(polar_indep_mesh[:,0,0])],[min(polar_indep_mesh[0,:,1]),max(polar_indep_mesh[0,:,1])]])
                        if polar_bc.dep[ii_deppol] == 'CD':
                            if rotmatflag == 0:
                                cd_mesh = polar_dep_mesh[:,:,ii_deppol]
                            else:
                                cd_mesh = polar_dep_mesh[:,:,ii_deppol].transpose()
                            if len(polar_indep[0]) == 1:
                                cd_mesh                                  = cd_mesh.flatten()
                                case_setup.polar[pobc_ind].cd_interp.append(interpolate.interp1d(polar_indep_mesh[:,0,0],cd_mesh,'cubic'))
                                case_setup.polar[pobc_ind].f_aoa0_drag.append(case_setup.polar[pobc_ind].cd_interp[iiaero_node](0))
                            elif len(polar_indep[0]) == 2:
                                case_setup.polar[pobc_ind].cd_interp.append(interpolate.RegularGridInterpolator((polar_indep_mesh[:,0,0],polar_indep_mesh[0,:,1]),cd_mesh)) 
                                cd_aoa0_vec = np.zeros((len(polar_indep_mesh[0,:,1]),))
                                for auxindep in np.arange(len(cd_aoa0_vec)):
                                    cd_aoa0_vec[auxindep] = case_setup.polar[pobc_ind].cd_interp[iiaero_node]((0,polar_indep_mesh[0,auxindep,1]))
                                case_setup.polar[pobc_ind].f_aoa0_drag.append(interpolate.interp1d(polar_indep_mesh[0,:,1],cd_aoa0_vec))
                        elif polar_bc.dep[ii_deppol] == 'CL':
                            if rotmatflag == 0:
                                cl_mesh = polar_dep_mesh[:,:,ii_deppol]
                            else:
                                cl_mesh = polar_dep_mesh[:,:,ii_deppol].transpose()
                            if len(polar_indep[0]) == 1:
                                cl_mesh                                    = cl_mesh.flatten()
                                min_cl                                     = np.argmin(abs(cl_mesh))
                                f_aoa_cl                                   = interpolate.interp1d(cl_mesh[min_cl-5:min_cl+5],polar_indep_mesh[min_cl-5:min_cl+5,0,0],'linear') 
                                f_aoa_0lift                                = f_aoa_cl(0)
                                case_setup.polar[pobc_ind].f_aoa_0lift.append(f_aoa_0lift)
                                case_setup.polar[pobc_ind].cl_interp.append(interpolate.interp1d(polar_indep_mesh[:,0,0],cl_mesh,'cubic')) 
                            elif len(polar_indep[0]) == 2:
                                min_cl                                     = np.argmin(abs(polar_indep_mesh[:,0,0]))#np.argmin(abs(cl_mesh[:,0]))
                                auxlim                                     = np.min([5,min_cl])
                                f_aoa_cl                                   = interpolate.interp2d(cl_mesh[min_cl-auxlim:min_cl+auxlim],polar_indep_mesh[min_cl-auxlim:min_cl+auxlim,:,1],polar_indep_mesh[min_cl-auxlim:min_cl+auxlim,:,0],kind='linear')  
                                f_aoa_0lift                                = interpolate.interp1d(np.linspace(np.min(polar_indep_mesh[0,:,1]),np.max(polar_indep_mesh[0,:,1]),100),f_aoa_cl(0,np.linspace(np.min(polar_indep_mesh[0,:,1]),np.max(polar_indep_mesh[0,:,1]),100))[:,0])
                                case_setup.polar[pobc_ind].f_aoa_0lift.append(f_aoa_0lift)
                                case_setup.polar[pobc_ind].cl_interp.append(interpolate.RegularGridInterpolator((polar_indep_mesh[:,0,0],polar_indep_mesh[0,:,1]),cl_mesh)) 
                        elif polar_bc.dep[ii_deppol] == 'CM' or polar_bc.dep[ii_deppol] == '-CM' :
                            if rotmatflag == 0: 
                                cm_mesh = polar_dep_mesh[:,:,ii_deppol]
                            else:
                                cm_mesh = polar_dep_mesh[:,:,ii_deppol].transpose()
                            if len(polar_indep[0]) == 1:
                                cm_mesh                                  = cm_mesh.flatten()
                                case_setup.polar[pobc_ind].cm_interp.append(interpolate.interp1d(polar_indep_mesh[:,0,0],cm_mesh,'cubic')) 
                            elif len(polar_indep[0]) == 2:
                                case_setup.polar[pobc_ind].cm_interp.append(interpolate.RegularGridInterpolator((polar_indep_mesh[:,0,0],polar_indep_mesh[0,:,1]),cm_mesh))
        # For every independent variable check the type
        if case_setup.aero_model == "THEO_POLAR":
            vecintaoa0 = aoa_mean0
            vecintaoa1 = aoa_mean1
            vecintaoa0st = aoa_mean0
            vecintaoa1st = aoa_mean1
        else:
            vecintaoa0 = aoa_vec0
            vecintaoa1 = aoa_vec1
            vecintaoa0st = aoa_vec0_st
            vecintaoa1st = aoa_vec1_st            
        for ii_indeppol in np.arange(len(polar_bc.indep)):
            if polar_bc.indep[ii_indeppol] == 'ANGLE':
                inc = np.pi/180
                if ii_indeppol > 0:
                    rotmatflag = 1
                    auxindep0 = []
                    auxindep1 = []
                    auxindep0st = []
                    auxindep1st = []
                    auxindep0b = []
                    auxindep1b = []
                    auxindep0.append(vecintaoa0)
                    auxindep1.append(vecintaoa1)
                    auxindep0st.append(vecintaoa0st)
                    auxindep1st.append(vecintaoa1st)
                    auxindep0b.append(vecintaoa0+inc)
                    auxindep1b.append(vecintaoa1+inc)
                    for aux in np.arange(len(indep_interp0)):
                        auxindep0.append(indep_interp0[aux])
                        auxindep1.append(indep_interp1[aux])
                        auxindep0st.append(indep_interp0st[aux])
                        auxindep1st.append(indep_interp1st[aux])
                        auxindep0b.append(indep_interp0b[aux])
                        auxindep1b.append(indep_interp1b[aux])
                    indep_interp0 = auxindep0
                    indep_interp1 = auxindep1
                    indep_interp0st = auxindep0st
                    indep_interp1st = auxindep1st
                    indep_interp0b = auxindep0b
                    indep_interp1b = auxindep1b
                else:
                    indep_interp0.append(vecintaoa0)
                    indep_interp1.append(vecintaoa1)
                    indep_interp0st.append(vecintaoa0st)
                    indep_interp1st.append(vecintaoa1st)
                    indep_interp0b.append(vecintaoa0+inc)
                    indep_interp1b.append(vecintaoa1+inc)
            elif polar_bc.indep[ii_indeppol] == 'ANGLE_DEG':
                inc = np.pi/180
                if ii_indeppol > 0:
                    rotmatflag = 1
                    auxindep0 = []
                    auxindep1 = []
                    auxindep0st = []
                    auxindep1st = []
                    auxindep0b = []
                    auxindep1b = []
                    auxindep0.append(vecintaoa0*180/np.pi)
                    auxindep1.append(vecintaoa1*180/np.pi)
                    auxindep0st.append(vecintaoa0st*180/np.pi)
                    auxindep1st.append(vecintaoa1st*180/np.pi)
                    auxindep0b.append(vecintaoa0*180/np.pi+inc*180/np.pi)
                    auxindep1b.append(vecintaoa1*180/np.pi+inc*180/np.pi)
                    for aux in np.arange(len(indep_interp0)):
                        auxindep0.append(indep_interp0[aux])
                        auxindep1.append(indep_interp1[aux])
                        auxindep0st.append(indep_interp0st[aux])
                        auxindep1st.append(indep_interp1st[aux])
                        auxindep0b.append(indep_interp0b[aux])
                        auxindep1b.append(indep_interp1b[aux])
                    indep_interp0 = auxindep0
                    indep_interp1 = auxindep1
                    indep_interp0st = auxindep0st
                    indep_interp1st = auxindep1st
                    indep_interp0b = auxindep0b
                    indep_interp1b = auxindep1b
                else:
                    indep_interp0.append(vecintaoa0*180/np.pi)
                    indep_interp1.append(vecintaoa1*180/np.pi)
                    indep_interp0st.append(vecintaoa0st*180/np.pi)
                    indep_interp1st.append(vecintaoa1st*180/np.pi)
                    indep_interp0b.append(vecintaoa0*180/np.pi+inc*180/np.pi)
                    indep_interp1b.append(vecintaoa1st*180/np.pi+inc*180/np.pi)
            elif polar_bc.indep[ii_indeppol] == 'POSITION':
                indep_interp0.append(pos_vec0)
                indep_interp1.append(pos_vec1)
                indep_interp0st.append(pos_vec0)
                indep_interp1st.append(pos_vec1)
                indep_interp0b.append(pos_vec0)
                indep_interp1b.append(pos_vec1)
            elif polar_bc.indep[ii_indeppol] == 'REYNOLDS':
                indep_interp0.append(reynolds0)
                indep_interp1.append(reynolds1)
                indep_interp0st.append(reynolds0)
                indep_interp1st.append(reynolds1)
                indep_interp0b.append(reynolds0)
                indep_interp1b.append(reynolds1)
            elif polar_bc.indep[ii_indeppol] == 'ANGLEFLAP':
                indep_interp0.append(deltaflap_vec0)
                indep_interp1.append(deltaflap_vec1)
                indep_interp0st.append(deltaflap_vec0)
                indep_interp1st.append(deltaflap_vec1)
                indep_interp0b.append(deltaflap_vec0)
                indep_interp1b.append(deltaflap_vec1)
            elif polar_bc.indep[ii_indeppol] == 'ANGLEFLAP_DEG':
                indep_interp0.append(deltaflap_vec0*180/np.pi)
                indep_interp1.append(deltaflap_vec1*180/np.pi)
                indep_interp0st.append(deltaflap_vec0*180/np.pi)
                indep_interp1st.append(deltaflap_vec1*180/np.pi)
                indep_interp0b.append(deltaflap_vec0*180/np.pi)
                indep_interp1b.append(deltaflap_vec1*180/np.pi)
        # ind_del : vector to delete rows and columns of the matrix in the induced velocity calculation (tips)
        eff_3d.ind_del = []
        indep_interp0st,indep_interp1st,indep_interp0,indep_interp1,indep_interp0b,indep_interp1b,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,cltot,eff_3d,cl_alpha,cl_alphaflap,aoa_0lift,aoa_0liftflap,cd_aoa0,cd_aoa0flap = interp_aeroconditions(case_setup,bound,polar_bctot,mesh_data,marktot,polar_bctot_ind,indep_interp0st,indep_interp1st,indep_interp0,indep_interp1,indep_interp0b,indep_interp1b,aoa_0lift,aoa_0liftflap,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,v_induced,refaxis2_0,refaxis2_1,Vinf0,Vinf1,LL_vec_p,inc,lpos,LL_cum,cltot,eff_3d,tol_vind,b_chord,b_chordflap,cd_aoa0,cd_aoa0flap,f_flap,coreflapdist_adim)
        error, eff_3d, v_induced,error2 = effects_3daero_outter(polar_bc,eff_3d,mark,Vinf0,Vinf1,LL_vec_p,v_induced,cltot,LL_cum,tol_vind,refaxis2_0,refaxis2_1,aoa_vec0,aoa_vec1,aoa_0lift,cd_aoa0,aoa_0liftflap,cd_aoa0flap,non_acac_dist,b_chord,b_chordflap,lpos,bound,aoa0_geom,aoa1_geom,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1)
    # If a linear aerodynamic model is preferred     
    elif case_setup.aero_model == "TEOR_SLOPE" or case_setup.aero_model == "THEO_POTEN":
        # ind_del : vector with the nodes to delete from the Lifting line theory problem
        cl_alpha = np.zeros((len(mark.node),)) 
        cl_alphaflap = np.zeros((len(mark.node),)) 
        eff_3d.ind_del = []
        # For every node in the lifting surface
        for iiaero_node in np.arange(len(mark.node)):
            iimark_node = int(mark.node.flatten()[iiaero_node])
            # cl_alpha : derivative of the lift coefficient respect to the angle of attack
            # cl_sec   : lift coefficient of the section
            if  case_setup.aero_model != "THEO_POTEN":
                clst_value0[iiaero_node] = 2*np.pi*aoa_mean0[iiaero_node]
                clst_value1[iiaero_node] = 2*np.pi*aoa_mean1[iiaero_node]
                cmst_value0[iiaero_node] = non_acac_dist[iiaero_node]*2*np.pi*aoa_mean0[iiaero_node]
                cmst_value1[iiaero_node] = non_acac_dist[iiaero_node]*2*np.pi*aoa_mean1[iiaero_node]
                cl_alpha[iiaero_node]    = 2*np.pi
                if f_flap == 1:
                    clstflap_value0[iiaero_node] = 2*np.pi*(aoa_mean0[iiaero_node]+deltaflap_vec0[iiaero_node])
                    clstflap_value1[iiaero_node] = 2*np.pi*(aoa_mean1[iiaero_node]+deltaflap_vec1[iiaero_node])
                    cmstflap_value0[iiaero_node] = non_acac_dist[iiaero_node]*2*np.pi*(aoa_mean0[iiaero_node]+deltaflap_vec0[iiaero_node])
                    cmstflap_value1[iiaero_node] = non_acac_dist[iiaero_node]*2*np.pi*(aoa_mean1[iiaero_node]+deltaflap_vec1[iiaero_node])
            else:
                clst_value0[iiaero_node] = 2*np.pi*(aoa_vec0[iiaero_node])
                clst_value1[iiaero_node] = 2*np.pi*(aoa_vec1[iiaero_node])
                cmst_value0[iiaero_node] = non_acac_dist[iiaero_node]*2*np.pi*(aoa_vec0[iiaero_node])
                cmst_value1[iiaero_node] = non_acac_dist[iiaero_node]*2*np.pi*(aoa_vec1[iiaero_node])
                cl_alpha[iiaero_node]    = 2*np.pi
                if f_flap == 1:
                    clstflap_value0[iiaero_node] = 2*np.pi*(aoa_vec0[iiaero_node]+deltaflap_vec0[iiaero_node])
                    clstflap_value1[iiaero_node] = 2*np.pi*(aoa_vec1[iiaero_node]+deltaflap_vec1[iiaero_node])
                    cmstflap_value0[iiaero_node] = non_acac_distflap[iiaero_node]*2*np.pi*(aoa_vec0[iiaero_node]+deltaflap_vec0[iiaero_node])
                    cmstflap_value1[iiaero_node] = non_acac_distflap[iiaero_node]*2*np.pi*(aoa_vec1[iiaero_node]+deltaflap_vec1[iiaero_node])
                    cl_alphaflap[iiaero_node]    = 2*np.pi
            clst_tot_value0[iiaero_node] = clst_value0[iiaero_node]*b_chord[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])+clstflap_value0[iiaero_node]*b_chordflap[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])
            clst_tot_value1[iiaero_node] = clst_value1[iiaero_node]*b_chord[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])+clstflap_value1[iiaero_node]*b_chordflap[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])
            cdst_tot_value0[iiaero_node] = cdst_value0[iiaero_node]*b_chord[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])+cdstflap_value0[iiaero_node]*b_chordflap[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])
            cdst_tot_value1[iiaero_node] = cdst_value1[iiaero_node]*b_chord[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])+cdstflap_value1[iiaero_node]*b_chordflap[iiaero_node]/(b_chord[iiaero_node]+b_chordflap[iiaero_node])
            cmst_tot_value0[iiaero_node] = cmst_value0[iiaero_node]*b_chord[iiaero_node]**2/(b_chord[iiaero_node]+b_chordflap[iiaero_node])**2+cmstflap_value0[iiaero_node]*b_chordflap[iiaero_node]**2/(b_chord[iiaero_node]+b_chordflap[iiaero_node])**2+clst_tot_value0[iiaero_node]*coreflapdist_adim[iiaero_node]
            cmst_tot_value1[iiaero_node] = cmst_value1[iiaero_node]*b_chord[iiaero_node]**2/(b_chord[iiaero_node]+b_chordflap[iiaero_node])**2+cmstflap_value1[iiaero_node]*b_chordflap[iiaero_node]**2/(b_chord[iiaero_node]+b_chordflap[iiaero_node])**2+clst_tot_value1[iiaero_node]*coreflapdist_adim[iiaero_node]
            cl_sec   = (clst_tot_value0[iiaero_node]*LL_vec_p[0,iimark_node]+clst_tot_value1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
            cltot   += (clst_tot_value0[iiaero_node]*mesh_data.mesh_point_scale[iimark_node][0]*LL_vec_p[0,iimark_node]+clst_tot_value1[iiaero_node]*mesh_data.mesh_point_scale[iimark_node][0]*LL_vec_p[1,iimark_node])/LL_cum
            eff_3d,v_induced,polar_bc = effects_3daero_inner(polar_bc,bound,mesh_data,mark,eff_3d,lpos,LL_cum,LL_vec_p,v_induced,iiaero_node,iimark_node,tol_vind,cl_sec,b_chord,cl_alpha,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1)
        error, eff_3d, v_induced,error2 = effects_3daero_outter(polar_bc,eff_3d,mark,Vinf0,Vinf1,LL_vec_p,v_induced,cltot,LL_cum,tol_vind,refaxis2_0,refaxis2_1,aoa_vec0,aoa_vec1,aoa_0lift,cd_aoa0,aoa_0liftflap,cd_aoa0flap,non_acac_dist,b_chord,b_chordflap,lpos,bound,aoa0_geom,aoa1_geom,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1)
    return case_setup, polar_bc_ind,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,eff_3d,error,cl_alpha,cl_alphaflap,error2,v_induced

#%%
def trans_aero(case_setup,bound,mark,polar_bc,time_val,tmin_ann,aoa_mean0,aoa_mean1,ang_mean0,ang_mean1,dis_mean0,dis_mean1,aoa_vec0,aoa_vec1,aoap_vec0,aoap_vec1,aoapp_vec0,aoapp_vec1,deltaflap_vec0,deltaflap_vec1,ang_vec0,ang_vec1,disp_vec0,disp_vec1,aoa_storearray0,aoa_storearray1,ang_storearray0,ang_storearray1,angder_storearray0,angder_storearray1,angderder_storearray0,angderder_storearray1,dis_storearray0,dis_storearray1,disder_storearray0,disder_storearray1,disderder_storearray0,disderder_storearray1,cddyn_value0,cddyn_value1,cldyn_value0,cldyn_value1,cmdyn_value0,cmdyn_value1,polar_bc_ind,int_per,b_chord,cl_alpha,non_eamc_dist,non_eaac_dist,non_acac_dist,Vinf0,Vinf1,time_valvec):
    aoa_mean0 = aoa_mean0.copy()
    aoa_mean1 = aoa_mean1.copy()
    ang_mean0 = ang_mean0.copy()
    ang_mean1 = ang_mean1.copy()
    dis_mean0 = dis_mean0.copy()
    dis_mean1 = dis_mean1.copy()
    aoa_vec0  = aoa_vec0.copy()
    aoa_vec1  = aoa_vec1.copy()
    aoap_vec0 = aoap_vec0.copy()
    aoap_vec1 = aoap_vec1.copy()
    aoapp_vec0 = aoapp_vec0.copy()
    aoapp_vec1 = aoapp_vec1.copy()
    deltaflap_vec0 = deltaflap_vec0.copy()
    deltaflap_vec1 = deltaflap_vec1.copy()
    ang_vec0 = ang_vec0.copy()
    ang_vec1 = ang_vec1.copy()
    disp_vec0 = disp_vec0.copy()
    disp_vec1 = disp_vec1.copy()
    aoa_storearray0 = aoa_storearray0.copy()
    aoa_storearray1 = aoa_storearray1.copy()
    ang_storearray0 = ang_storearray0.copy()
    ang_storearray1 = ang_storearray1.copy()
    angder_storearray0 = angder_storearray0.copy()
    angder_storearray1 = angder_storearray1.copy()
    angderder_storearray0 = angderder_storearray0.copy()
    angderder_storearray1 = angderder_storearray1.copy()
    dis_storearray0 = dis_storearray0.copy()
    dis_storearray1 = dis_storearray1.copy()
    disder_storearray0 = disder_storearray0.copy()
    disder_storearray1 = disder_storearray1.copy()
    disderder_storearray0 = disderder_storearray0.copy()
    disderder_storearray1 = disderder_storearray1.copy()
    cddyn_value0 = cddyn_value0.copy()
    cddyn_value1 = cddyn_value1.copy()
    cldyn_value0 = cldyn_value0.copy()
    cldyn_value1 = cldyn_value1.copy()
    cmdyn_value0 = cmdyn_value0.copy()
    cmdyn_value1 = cmdyn_value1.copy()
    b_chord = b_chord.copy()
    cl_alpha = cl_alpha.copy()
    non_eamc_dist = non_eamc_dist.copy()
    non_eaac_dist = non_eaac_dist.copy()
    non_acac_dist = non_acac_dist.copy()
    Vinf0 = Vinf0.copy()
    Vinf1 = Vinf1.copy()
    # case_setup      : information about the case configuration
    # bound           : information of the boundary
    # mark            : information of the marker
    # polar_bc        : polar boundary conditions
    # time_val        : value of the time
    # tmin_ann        : time of the ann
    # aoa_mean0       : mean value of the angle of attack element 0 of the node
    # aoa_mean1       : mean value of the angle of attack element 1 of the node
    # ang_mean0       : mean value of the angle of twist element 0 of the node
    # ang_mean1       : mean value of the angle of twist element 1 of the node
    # dis_mean0       : mean value of the displacement element 0 of the node
    # dis_mean1       : mean value of the displacement element 1 of the node
    # aoa_vec0        : vector of the angle of attack element 0 of the node
    # aoa_vec1        : vector of the angle of attack element 1 of the node
    # aoap_vec0       : vector of the angle of attack derivative of element 0 of the node
    # aoap_vec1       : vector of the angle of attack derivative of element 1 of the node
    # aoapp_vec0      : vector of the angle of attack second derivative of element 0 of the node
    # aoapp_vec1      : vector of the angle of attack second derivative of element 1 of the node
    # ang_vec0        : vector of the twist angle of element 0 of the node
    # ang_vec1        : vector of the twist angle of element 1 of the node
    # disp_vec0       : vector of the displacement of element 0 of the node
    # disp_vec1       : vector of the displacement of element 1 of the node
    # aoa_storearray0 : historical of the angle of attack of element 0 of the node
    # aoa_storearray1 : historical of the angle of attack of element 1 of the node
    # ang_storearray0 : historical of the angle of twist of element 0 of the node
    # ang_storearray1 : historical of the angle of twist of element 1 of the node
    # dis_storearray0 : historical of the displacement of element 0 of the node
    # dis_storearray1 : historical of the displacement of element 1 of the node
    # cddyn_value0    : drag coefficient of element 0 of the node
    # cddyn_value1    : drag coefficient of element 1 of the node
    # cldyn_value0    : lift coefficient of element 0 of the node
    # cldyn_value1    : lift coefficient of element 1 of the node
    # cmdyn_value0    : moment coefficient of element 0 of the node
    # cmdyn_value1    : moment coefficient of element 1 of the node
    # polar_bc_ind    : index of the polar
    # int_per         : period of the 1st mode
    # b_chord         : semichord
    # cl_alpha        : lift coefficient derivative
    # non_eamc_dist   : nondimensional distance between the elastic axis and the mean chord
    # non_eaac_dist   : nondimensional distance between the elastic axis and the aerodynamic center
    # non_acac_dist   : nondimensional distance between the aerodynamic center and the quarter of chord
    # Vinf0           : free stream velocity in element 0 of the node
    # Vinf1           : free stream velocity in element 1 of the node
    # If an Artificial Neural Network is required and the time value is higher than the minimum time
#    print(cl_alpha[-15]/(2*np.pi))
#    cl_alpha = np.ones((len(cl_alpha),))*2*np.pi
    if case_setup.aero_model == "POLAR_ANN"  and time_val >= tmin_ann:
        # annindep_0 : independent terms of the ANN in node 0
        # annindep_1 : independent terms of the ANN in node 1
        annindep_0 = []
        annindep_1 = []
        # For every polar, check the type is Fed Forward Network FNN
        # ann_dynpolar : ANN loaded from file
        # norm_ann     : normalization data of the network, it is read from file
        # The ANN is loaded if it has not been done before
        if polar_bc.anntype == "FNN":
            # Save the ann in case it is not loaded
            try:
                ann_dynpolar = polar_bc.annload
                norm_ann     = polar_bc.norm_ann
            except:
                ann_dynpolar                            = keras.models.load_model(case_setup.root+polar_bc.annfile)
                case_setup.polar[polar_bc_ind].annload  = ann_dynpolar
                norm_ann                                = pd.read_csv(case_setup.root+polar_bc.annnorm).values
                case_setup.polar[polar_bc_ind].norm_ann = norm_ann
        # For every independent parameter of the ANN check its meaning
        for ii_indeppol in np.arange(len(polar_bc.annindep)):
            # Mean angle of attack in degrees
            if polar_bc.annindep[ii_indeppol] == 'AOA_MEAN_DEG':
                # Find the normalization parameters and normalize
                for norm_par in norm_ann:
                    if norm_par[0] == 'AOA_MEAN_DEG':
                        # aoa_mean0_norm : normalized mean angle of attack for node 0
                        # aoa_mean1_norm : normalized mean angle of attack for node 1
                        aoa_mean0_norm = (aoa_mean0*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        aoa_mean1_norm = (aoa_mean1*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        # For every element in the vector, restric extrapolation
                        for jj_indeppol in np.arange(len(aoa_mean0_norm)):
                            if aoa_mean0_norm[jj_indeppol] < 0:
                                aoa_mean0_norm[jj_indeppol] = 0
                            elif aoa_mean0_norm[jj_indeppol] > 1:
                                aoa_mean0_norm[jj_indeppol] = 1
                            if aoa_mean1_norm[jj_indeppol] < 0:
                                aoa_mean1_norm[jj_indeppol] = 0
                            elif aoa_mean1_norm[jj_indeppol] > 1:
                                aoa_mean1_norm[jj_indeppol] = 1
                        # append the value of the mean angle of attack to the ann independent parameter matrix
                        annindep_0.append(aoa_mean0_norm)
                        annindep_1.append(aoa_mean1_norm)
            # Increment of the angle of attack in degrees
            elif polar_bc.annindep[ii_indeppol] == 'DELTA_AOA_DEG':
                for norm_par in norm_ann:
                    if norm_par[0] == 'DELTA_AOA_DEG':
                        # aoa_delta0_norm : normalized increment of the angle of attack for node 0
                        # aoa_delta1_norm : normalized increment of the angle of attack for node 1
                        aoa_delta0_norm = ((aoa_vec0-aoa_mean0)*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        aoa_delta1_norm = ((aoa_vec1-aoa_mean1)*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        # For every element in the vector, restric extrapolation
                        for jj_indeppol in np.arange(len(aoa_mean0_norm)):
                            if aoa_delta0_norm[jj_indeppol] < 0:
                                aoa_delta0_norm[jj_indeppol] = 0
                            elif aoa_delta0_norm[jj_indeppol] > 1:
                                aoa_delta0_norm[jj_indeppol] = 1
                            if aoa_delta1_norm[jj_indeppol] < 0:
                                aoa_delta1_norm[jj_indeppol] = 0
                            elif aoa_delta1_norm[jj_indeppol] > 1:
                                aoa_delta1_norm[jj_indeppol] = 1
                        # append the value of the increment of the angle of attack to the ann independent parameter matrix
                        annindep_0.append(aoa_delta0_norm)
                        annindep_1.append(aoa_delta1_norm)
            # Derivative of the angle of attack in degrees
            elif polar_bc.annindep[ii_indeppol] == 'DER_AOA_DEG':
                for norm_par in norm_ann:
                    if norm_par[0] == 'DER_AOA_DEG':
                        # aoa_der0_norm  : normalized derivative of the angle of attack for node 0
                        # aoa_der1_norm  : normalized derivative of the angle of attack for node 1
                        aoa_der0_norm = (aoap_vec0*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        aoa_der1_norm = (aoap_vec1*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        # For every element in the vector, restric extrapolation
                        for jj_indeppol in np.arange(len(aoa_mean0_norm)):
                            if aoa_der0_norm[jj_indeppol] < 0:
                                aoa_der0_norm[jj_indeppol] = 0
                            elif aoa_der0_norm[jj_indeppol] > 1:
                                aoa_der0_norm[jj_indeppol] = 1
                            if aoa_der1_norm[jj_indeppol] < 0:
                                aoa_der1_norm[jj_indeppol] = 0
                            elif aoa_der1_norm[jj_indeppol] > 1:
                                aoa_der1_norm[jj_indeppol] = 1
                        # append the value of the derivative of the angle of attack to the ann independent parameter matrix
                        annindep_0.append(aoa_der0_norm)
                        annindep_1.append(aoa_der1_norm)
            # second derivative of the angle of attack in degrees
            elif polar_bc.annindep[ii_indeppol] == 'DER_DER_AOA_DEG':
                for norm_par in norm_ann:
                    if norm_par[0] == 'DER_DER_AOA_DEG':
                        # aoa_derder0_norm  : normalized derivative of the angle of attack for node 0
                        # aoa_derder1_norm  : normalized derivative of the angle of attack for node 1
                        aoa_derder0_norm = (aoapp_vec0*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        aoa_derder1_norm = (aoapp_vec1*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        # For every element in the vector, restric extrapolation
                        for jj_indeppol in np.arange(len(aoa_mean0_norm)):
                            if aoa_derder0_norm[jj_indeppol] < 0:
                                aoa_derder0_norm[jj_indeppol] = 0
                            elif aoa_derder0_norm[jj_indeppol] > 1:
                                aoa_derder0_norm[jj_indeppol] = 1
                            if aoa_derder1_norm[jj_indeppol] < 0:
                                aoa_derder1_norm[jj_indeppol] = 0
                            elif aoa_derder1_norm[jj_indeppol] > 1:
                                aoa_derder1_norm[jj_indeppol] = 1
                        # append the value of the second derivative of the angle of attack to the ann independent paramenter matrix
                        annindep_0.append(aoa_derder0_norm)
                        annindep_1.append(aoa_derder1_norm)
            elif polar_bc.annindep[ii_indeppol] == 'DELTAFLAP_DEG':
                for norm_par in norm_ann:
                    if norm_par[0] == 'DELTAFLAP_DEG':
                        # aoa_derder0_norm  : normalized derivative of the angle of attack for node 0
                        # aoa_derder1_norm  : normalized derivative of the angle of attack for node 1
                        deltaflap0_norm = (deltaflap_vec0*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        deltaflap1_norm = (deltaflap_vec1*180/np.pi-norm_par[1])/(norm_par[2]-norm_par[1])
                        # For every element in the vector, restric extrapolation
                        for jj_indeppol in np.arange(len(aoa_mean0_norm)):
                            if deltaflap0_norm[jj_indeppol] < 0:
                                deltaflap0_norm[jj_indeppol] = 0
                            elif deltaflap0_norm[jj_indeppol] > 1:
                                deltaflap0_norm[jj_indeppol] = 1
                            if deltaflap1_norm[jj_indeppol] < 0:
                                deltaflap1_norm[jj_indeppol] = 0
                            elif deltaflap1_norm[jj_indeppol] > 1:
                                deltaflap1_norm[jj_indeppol] = 1
                        # append the value of the second derivative of the angle of attack to the ann independent paramenter matrix
                        annindep_0.append(deltaflap0_norm)
                        annindep_1.append(deltaflap1_norm)
        # For every node in the lifting surface, calculate the dynamic term
        for iiaero_node in np.arange(len(mark.node)):
            if polar_bc.anntype == "FNN":
                # indep_annpolar0 : independent terms to enter the ANN in node 0
                # indep_annpolar1 : independent terms to enter the ANN in node 1
                indep_annpolar0 = []
                indep_annpolar1 = []
                # Append the corresponding terms to each node and convert it into array
                for n_indann in np.arange(len(polar_bc.annindep)):
                    indep_annpolar0.append(annindep_0[n_indann][iiaero_node])
                    indep_annpolar1.append(annindep_1[n_indann][iiaero_node])
                indep_annpolar0 = np.array([indep_annpolar0])
                indep_annpolar1 = np.array([indep_annpolar1])
            # coeff_dyn0 : dynamic coefficients obtained from the ANN in node 0
            # coeff_dyn1 : dynamic coefficients obtained from the ANN in node 1
            coeff_dyn0 = ann_dynpolar.predict(indep_annpolar0).flatten()
            coeff_dyn1 = ann_dynpolar.predict(indep_annpolar1).flatten()
            # For each dependent parameter of the ANN
            for n_depann in np.arange(len(polar_bc.anndep)):
                # Drag coefficient
                if polar_bc.anndep[n_depann] == "CD_DYN":
                    for norm_par in norm_ann:
                        if norm_par[0] == 'CD_DYN':
                            cddyn_value0[iiaero_node]  = coeff_dyn0[n_depann]*(norm_par[2]-norm_par[1])+norm_par[1]
                            cddyn_value1[iiaero_node]  = coeff_dyn1[n_depann]*(norm_par[2]-norm_par[1])+norm_par[1]
                # Lift coefficient
                elif polar_bc.anndep[n_depann] == "CL_DYN":
                    for norm_par in norm_ann:
                        if norm_par[0] == 'CL_DYN':
                            cldyn_value0[iiaero_node]  = coeff_dyn0[n_depann]*(norm_par[2]-norm_par[1])+norm_par[1]
                            cldyn_value1[iiaero_node]  = coeff_dyn1[n_depann]*(norm_par[2]-norm_par[1])+norm_par[1]
                # Moment coefficient
                elif polar_bc.anndep[n_depann] == "CM_DYN":
                    for norm_par in norm_ann:
                        if norm_par[0] == 'CM_DYN':
                            cmdyn_value0[iiaero_node]  = (coeff_dyn0[n_depann]*(norm_par[2]-norm_par[1])+norm_par[1])
                            cmdyn_value1[iiaero_node]  = (coeff_dyn1[n_depann]*(norm_par[2]-norm_par[1])+norm_par[1])
                # Negative axis moment coefficient
                elif polar_bc.anndep[n_depann] == "-CM_DYN":
                    for norm_par in norm_ann:
                        if norm_par[0] == '-CM_DYN':
                            cmdyn_value0[iiaero_node]  = -(coeff_dyn0[n_depann]*(norm_par[2]-norm_par[1])+norm_par[1])
                            cmdyn_value1[iiaero_node]  = -(coeff_dyn1[n_depann]*(norm_par[2]-norm_par[1])+norm_par[1])                            
    elif case_setup.aero_model == "THEO_POTEN" or case_setup.aero_model == "THEO_POLAR":
        case_setup.duhamel = 1
        if case_setup.duhamel == 0:
            # N0 : number of steps in node 0
            # N1 : number of steps in node 1
            if len(aoa_storearray0) == 0:
                N0 = 0
                N1 = 0
            else:
                N0 = len(aoa_storearray0[0,:])
                N1 = len(aoa_storearray1[0,:])
            # For every node in the lifting surface
            for iiaero_node in np.arange(len(mark.node)):
                # time_sampling : time step
                # ind_del       : vector with the nodes to delete from the Lifting line theory problem
                time_sampling = case_setup.time_stp
                #  If the number of time steps duplicate the period of the first mode (enough points to make fft with resolution)
                nnper = 1
    #            print(N0)
                if N0>nnper*int_per:
                    # Lfft   : length of the fourier fast transformation
                    # window : windowing of the signal
                    # thfreq0 : twist in frequency node 0
                    # thfreq1 : twist in frequency node 1
                    # hfreq0  : plunge in frequency node 0
                    # hfreq1  : plunge in frequency node 1
                    # xfreq0  : frequency of node 0
                    # xfreq1  : frequency of node 1
                    # wfft0   : angular frequency of node 0
                    # wfft1   : angular frequency of node 1
                    # kfft0   : reduced frequency of node 0
                    # kfft1   : reduced frequency of node 1
                    # C_theo0 : Theodorsen coefficient node 0
                    # C_theo1 : Theodorsen coefficient node 1
                    # clfreq0_a : lift coefficient in frequency - inertial forces node 0
                    # clfreq0_b : lift coefficient in frequency - theodorsen 0
                    # clfreq1_a : lift coefficient in frequency - inertial forces node 1
                    # clfreq1_b : lift coefficient in frequency - theodorsen 1
                    # clfreq0   : lift coefficient in frequency node 0
                    # clfreq1   : lift coefficient in frequency node 1
                    # cmfreq0_a : moment coefficient in frequency - inertial forces node 0
                    # cmfreq0_b : moment coefficient in frequency - theodorsen 0
                    # cmfreq1_a : moment coefficient in frequency - inertial forces node 1
                    # cmfreq1_b : moment coefficient in frequency - theodorsen 1
                    # cmfreq0   : moment coefficient in frequency node 0
                    # cmfreq1   : moment coefficient in frequency node 1
                    Lfft      = np.min([N0,nnper*int_per])
                    window    = signal.windows.blackman(Lfft)
                    thfreq0   = 2.0/N0 *fft((ang_storearray0[iiaero_node,-Lfft:]-ang_mean0[iiaero_node])*window)[0:Lfft//2] # 
                    thfreq1   = 2.0/N1 *fft((ang_storearray1[iiaero_node,-Lfft:]-ang_mean1[iiaero_node])*window)[0:Lfft//2]
                    hfreq0    = 2.0/N0 *fft((dis_storearray0[iiaero_node,-Lfft:]-dis_mean0[iiaero_node])*window)[0:Lfft//2]
                    hfreq1    = 2.0/N1 *fft((dis_storearray1[iiaero_node,-Lfft:]-dis_mean1[iiaero_node])*window)[0:Lfft//2]
                    xfreq0    = fftfreq(N0, time_sampling)[:Lfft//2]
                    xfreq1    = fftfreq(N1, time_sampling)[:Lfft//2]
                    wfft0     = xfreq0*2*np.pi
                    wfft1     = xfreq1*2*np.pi
                    kfft0     = wfft0*bound.l_ref/2/Vinf0[iiaero_node]
                    kfft1     = wfft1*bound.l_ref/2/Vinf1[iiaero_node]
                    C_theo0   = theodorsen(abs(kfft0))
                    C_theo1   = theodorsen(abs(kfft1))
                    clfreq0_a = np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])*(-wfft0**2*hfreq0+1j*wfft0*Vinf0[iiaero_node]*thfreq0+b_chord[iiaero_node]*non_eamc_dist[iiaero_node]*wfft0**2*thfreq0)
                    clfreq0_b = cl_alpha[iiaero_node]*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]*C_theo0/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])*(1j*wfft0*hfreq0+Vinf0[iiaero_node]*thfreq0+b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])*1j*wfft0*thfreq0)
                    clfreq0   = clfreq0_a+clfreq0_b
                    clfreq1_a = np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])*(-wfft1**2*hfreq1+1j*wfft1*Vinf1[iiaero_node]*thfreq1+b_chord[iiaero_node]*non_eamc_dist[iiaero_node]*wfft1**2*thfreq1)
                    clfreq1_b = cl_alpha[iiaero_node]*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]*C_theo1/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])*(1j*wfft1*hfreq1+Vinf1[iiaero_node]*thfreq1+b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])*1j*wfft1*thfreq1)
                    clfreq1   = clfreq1_a+clfreq1_b
                    cmfreq0_a = non_eaac_dist[iiaero_node]*clfreq0+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(-b_chord[iiaero_node]*non_eamc_dist[iiaero_node]*wfft0**2*hfreq0-Vinf0[iiaero_node]*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])*1j*wfft0*thfreq0+b_chord[iiaero_node]**2*(1/8+non_eamc_dist[iiaero_node]**2)*wfft0**2*thfreq0)
                    cmfreq0_b = cl_alpha[iiaero_node]*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])*C_theo0/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(1j*wfft0*hfreq0+Vinf0[iiaero_node]*thfreq0+b_chord[iiaero_node]*(1/2+non_eamc_dist[iiaero_node])*1j*wfft0*thfreq0)
                    cmfreq0   = cmfreq0_a+cmfreq0_b
                    cmfreq1_a = non_eaac_dist[iiaero_node]*clfreq1+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf1[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(-b_chord[iiaero_node]*non_eamc_dist[iiaero_node]*wfft1**2*hfreq1-Vinf1[iiaero_node]*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])*1j*wfft1*thfreq1+b_chord[iiaero_node]**2*(1/8+non_eamc_dist[iiaero_node]**2)*wfft1**2*thfreq1)
                    cmfreq1_b = cl_alpha[iiaero_node]*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])*C_theo1/(1/2*bound.rho*Vinf1[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(1j*wfft1*hfreq1+Vinf1[iiaero_node]*thfreq1+b_chord[iiaero_node]*(1/2+non_eamc_dist[iiaero_node])*1j*wfft1*thfreq1)
                    cmfreq1   = cmfreq1_a+cmfreq1_b
                    # Smooth transition between models
                    if N0>2*nnper*int_per:
                        cldyn_value0[iiaero_node] = N0/2*np.real(ifft(clfreq0)[-1])
                        cldyn_value1[iiaero_node] = N1/2*np.real(ifft(clfreq1)[-1])
                        cmdyn_value0[iiaero_node] = N0/2*np.real(ifft(cmfreq0)[-1])
                        cmdyn_value1[iiaero_node] = N1/2*np.real(ifft(cmfreq1)[-1])
                    else: 
                        cldyn_value0[iiaero_node] = (1-(N0-Lfft)/Lfft)*(cl_alpha[iiaero_node]*(ang_vec0[iiaero_node]-ang_mean0[iiaero_node]+np.arctan(disp_vec0[iiaero_node]/Vinf0[iiaero_node])))+(N0-Lfft)/Lfft*N0/2*np.real(ifft(clfreq0)[-1])
                        cldyn_value1[iiaero_node] = (1-(N1-Lfft)/Lfft)*(cl_alpha[iiaero_node]*(ang_vec1[iiaero_node]-ang_mean1[iiaero_node]+np.arctan(disp_vec1[iiaero_node]/Vinf1[iiaero_node])))+(N1-Lfft)/Lfft*N1/2*np.real(ifft(clfreq1)[-1])
                        cmdyn_value0[iiaero_node] = (1-(N0-Lfft)/Lfft)*(non_eaac_dist[iiaero_node]*cl_alpha[iiaero_node]*(ang_vec0[iiaero_node]-ang_mean0[iiaero_node]+np.arctan(disp_vec0[iiaero_node]/Vinf0[iiaero_node])))+(N0-Lfft)/Lfft*N0/2*np.real(ifft(cmfreq0)[-1])
                        cmdyn_value1[iiaero_node] = (1-(N1-Lfft)/Lfft)*(non_eaac_dist[iiaero_node]*cl_alpha[iiaero_node]*(ang_vec1[iiaero_node]-ang_mean1[iiaero_node]+np.arctan(disp_vec1[iiaero_node]/Vinf1[iiaero_node])))+(N1-Lfft)/Lfft*N1/2*np.real(ifft(cmfreq1)[-1])
                else:
                    cldyn_value0[iiaero_node] = cl_alpha[iiaero_node]*(ang_vec0[iiaero_node]-ang_mean0[iiaero_node]+np.arctan(disp_vec0[iiaero_node]/Vinf0[iiaero_node]))
                    cldyn_value1[iiaero_node] = cl_alpha[iiaero_node]*(ang_vec1[iiaero_node]-ang_mean1[iiaero_node]+np.arctan(disp_vec1[iiaero_node]/Vinf1[iiaero_node]))
                    cmdyn_value0[iiaero_node] = non_eaac_dist[iiaero_node]*cl_alpha[iiaero_node]*(ang_vec0[iiaero_node]-ang_mean0[iiaero_node]+np.arctan(disp_vec0[iiaero_node]/Vinf0[iiaero_node]))
                    cmdyn_value1[iiaero_node] = non_eaac_dist[iiaero_node]*cl_alpha[iiaero_node]*(ang_vec1[iiaero_node]-ang_mean1[iiaero_node]+np.arctan(disp_vec1[iiaero_node]/Vinf1[iiaero_node]))
        elif case_setup.duhamel == 1:
            for iiaero_node in np.arange(len(mark.node)):
                stime0 =  Vinf0[iiaero_node]*time_valvec/b_chord[iiaero_node]
                stime1 =  Vinf1[iiaero_node]*time_valvec/b_chord[iiaero_node]
                thp0 = angder_storearray0[iiaero_node,-1]
                thp1 = angder_storearray1[iiaero_node,-1]
                thpp0 = angderder_storearray0[iiaero_node,-1]
                thpp1 = angderder_storearray1[iiaero_node,-1]
                hpp0 = disderder_storearray0[iiaero_node,-1]
                hpp1 = disderder_storearray1[iiaero_node,-1]
                wa0 = -(disder_storearray0[iiaero_node,:]+Vinf0[iiaero_node]*(ang_storearray0[iiaero_node,:]-ang_mean0[iiaero_node])+b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])*angder_storearray0[iiaero_node,:])
                wa1 = -(disder_storearray1[iiaero_node,:]+Vinf1[iiaero_node]*(ang_storearray1[iiaero_node,:]-ang_mean1[iiaero_node])+b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])*angder_storearray1[iiaero_node,:])
                if len(stime0) > 1:
                    dsigma0 = np.gradient(stime0)
                    dwa0ds = np.divide(np.gradient(wa0),dsigma0)
                    dsigma1 = np.gradient(stime1)
                    dwa1ds = np.divide(np.gradient(wa1),dsigma1)
                    phi0 = 1 - 0.165*np.exp(-0.0455*(stime0[-1]))-0.335*np.exp(-0.3*(stime0[-1]))
                    phi1 = 1 - 0.165*np.exp(-0.0455*(stime1[-1]))-0.335*np.exp(-0.3*(stime1[-1]))
                    integ_conv0 = 0
                    integ_conv1 = 0
                    for ii_conv in np.arange(len(stime0)):
                        phi0int = 1 - 0.165*np.exp(-0.0455*(stime0[-1]-stime0[ii_conv]))-0.335*np.exp(-0.3*(stime0[-1]-stime0[ii_conv])) #0.165*np.exp(-0.091*(stime0[-1]-stime0[ii_conv]))-0.335*np.exp(-0.6*(stime0[-1]-stime0[ii_conv]))
                        phi1int = 1 - 0.165*np.exp(-0.0455*(stime1[-1]-stime1[ii_conv]))-0.335*np.exp(-0.3*(stime1[-1]-stime1[ii_conv]))
                        integ_conv0 += dwa0ds[ii_conv]*phi0int*dsigma0[ii_conv]
                        integ_conv1 += dwa1ds[ii_conv]*phi1int*dsigma1[ii_conv]
                else:
                    dsigma0 = np.array([0])
                    dwa0ds = np.array([0])
                    dsigma1 = np.array([0])
                    dwa1ds = np.array([0])
                    phi0 = 1 - 0.165*np.exp(-0.0455*(stime0))-0.335*np.exp(-0.3*(stime0))
                    phi1 = 1 - 0.165*np.exp(-0.0455*(stime1))-0.335*np.exp(-0.3*(stime1))
                    integ_conv0 = 0
                    integ_conv1 = 0
                clfreq0_a = np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])*(hpp0+Vinf0[iiaero_node]*thp0-b_chord[iiaero_node]*non_eamc_dist[iiaero_node]*thpp0)
                clfreq1_a = np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])*(hpp1+Vinf1[iiaero_node]*thp1-b_chord[iiaero_node]*non_eamc_dist[iiaero_node]*thpp1)
                clfreq0_b = -2*np.pi*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])*(wa0[0]*phi0+integ_conv0) # cl_alpha[iiaero_node]
                clfreq1_b = -2*np.pi*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])*(wa1[0]*phi1+integ_conv1) # cl_alpha[iiaero_node]
                cldyn_value0[iiaero_node]   = clfreq0_a+clfreq0_b #non_eaac_dist[iiaero_node]*cldyn_value0[iiaero_node]+
                cldyn_value1[iiaero_node]   = clfreq1_a+clfreq1_b # non_eaac_dist[iiaero_node]*cldyn_value1[iiaero_node]+
#                print([non_eaac_dist[iiaero_node],cldyn_value0[iiaero_node]])
                cmfreq0_a = non_eaac_dist[iiaero_node]*cldyn_value0[iiaero_node]+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(b_chord[iiaero_node]*non_eamc_dist[iiaero_node]*hpp0-Vinf0[iiaero_node]*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])*thp0-b_chord[iiaero_node]**2*(1/8+non_eamc_dist[iiaero_node]**2)*thpp0)
                cmfreq1_a = non_eaac_dist[iiaero_node]*cldyn_value1[iiaero_node]+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf1[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(b_chord[iiaero_node]*non_eamc_dist[iiaero_node]*hpp1-Vinf1[iiaero_node]*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])*thp1-b_chord[iiaero_node]**2*(1/8+non_eamc_dist[iiaero_node]**2)*thpp1)
                cmfreq0_b = -2*np.pi*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(wa0[0]*phi0+integ_conv0) #cl_alpha[iiaero_node]
                cmfreq1_b = -2*np.pi*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])/(1/2*bound.rho*Vinf1[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(wa1[0]*phi1+integ_conv1) # cl_alpha[iiaero_node]
                cmdyn_value0[iiaero_node]   = cmfreq0_a+cmfreq0_b
                cmdyn_value1[iiaero_node]   = cmfreq1_a+cmfreq1_b
#                if clfreq0_a != 0:
#                    print([clfreq0_a,clfreq0_b])
    return case_setup,cldyn_value0,cldyn_value1,cddyn_value0,cddyn_value1,cmdyn_value0,cmdyn_value1
#%%
def total_forces(clst_value0,clst_value1,cmst_value0,cmst_value1,cdst_value0,cdst_value1,cldyn_value0,cldyn_value1,cmdyn_value0,cmdyn_value1,cddyn_value0,cddyn_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdstflap_value0,cdstflap_value1,cldynflap_value0,cldynflap_value1,cmdynflap_value0,cmdynflap_value1,cddynflap_value0,cddynflap_value1,LL_vec_p,rot_mat_p,b_chord,b_chordflap,lpos,Vinf0,Vinf1,vdir0,vdir1,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,FF_global,bound,section,polar_bc,marktot,disp_values,num_point,time_val,tmin_ann,ttran_ann,ii_time,eff_3d,f_flap):
#    print('steady: '+str(clst_value0[-15])+' // dynamic: '+str(cldyn_value0[-15]))
    # clst_value0  : steady value of the lift coefficient for the element 0 of the node
    # clst_value1  : steady value of the lift coefficient for the element 1 of the node   
    # cmst_value0  : steady value of the moment coefficient for the element 0 of the node
    # cmst_value1  : steady value of the moment coefficient for the element 1 of the 
    # cdst_value0  : steady value of the drag coefficient for the element 0 of the node
    # cdst_value1  : steady value of the drag coefficient for the element 1 of the node 
    # cldyn_value0 : dynamic value of the lift coefficient for the element 0 of the node
    # cldyn_value1 : dynamic value of the lift coefficient for the element 1 of the node   
    # cmdyn_value0 : dynamic value of the moment coefficient for the element 0 of the node
    # cmdyn_value1 : dynamic value of the moment coefficient for the element 1 of the 
    # cddyn_value0 : dynamic value of the drag coefficient for the element 0 of the node
    # cddyn_value1 : dynamic value of the drag coefficient for the element 1 of the node 
    # LL_vec_p     : length of the element associated with the nodes
    # rot_mat_p    : rotation matrix of the elements
    # b_chord      : semichord
    # lpos         : distance to the reference point
    # Vinf0        : free stream velocity of element 0 of the node
    # Vinf1        : free stream velocity of element 1 of the node
    # vdir0        : vector of the velocity direction in element 0 of the node
    # vdir1        : vector of the velocity direction in element 1 of the node
    # refaxis_0    : moment reference axis of element 0 of the node
    # refaxis_1    : moment reference axis of element 1 of the node
    # refaxis2_0   : lift reference axis of element 0 of the node
    # refaxis2_1   : lift reference axis of element 1 of the node
    # FF_global    : load vector
    # bound        : boundary condition information
    # section      : information about the section
    # polar_bc     : aerodynamic polar information
    # mark         : marker information
    # disp_values. : values of the displacements and forces
    # num_point    : number of nodes
    # time_val     : time
    # tmin_ann     : time for the ann
    # ttran_ann    : transition time for the ann
    # ii_time      : time index
    # eff_3d       : information of the 3D effects
    # -------------------------------------------------------------------------
    if len(marktot) == 1:
        mark = marktot[0]
    else:
        mark = marktot[0]
        markflap = marktot[1]
    # cl_value0    : lift coefficient of node 0
    # cl_value1    : lift coefficient of node 1
    # cd_value0    : drag coefficient of node 0
    # cd_value1    : drag coefficient of node 1
    # cm_value0    : pitch moment coefficient of node 0
    # cm_value1    : pitch moment coefficient of node 1
    cd_tot = np.zeros((num_point,))
    cl_tot = np.zeros((num_point,))
    cm_tot = np.zeros((num_point,))
    cd_value0    = np.zeros((len(mark.node),))
    cd_value1    = np.zeros((len(mark.node),))
    cl_value0    = np.zeros((len(mark.node),))
    cl_value1    = np.zeros((len(mark.node),))
    cm_value0    = np.zeros((len(mark.node),))
    cm_value1    = np.zeros((len(mark.node),))
    # drag_vec0      : drag vector of node 0
    # drag_vec1      : drag vector of node 1
    # lift_vec0      : lift vector of node 0
    # lift_vec1      : lift vector of node 1
    # cmmoment_vec0  : moment coefficient vector node 0
    # cmmoment_vec1  : moment coefficient vector node 1
    # aero_force     : aerodynamic force
    # aero_moment    : aerodynamic moment
    # r_aerovec      : distance vector between center of gravity and aerodynamic center
    # load_vector    : vector to add in the global loads definition
    # centrif_force  : centrifugal force
    drag_vec0     = np.zeros((len(mark.node),3))
    drag_vec1     = np.zeros((len(mark.node),3))
    lift_vec0     = np.zeros((len(mark.node),3))
    lift_vec1     = np.zeros((len(mark.node),3))
    cmmoment_vec0 = np.zeros((len(mark.node),3))
    cmmoment_vec1 = np.zeros((len(mark.node),3))
    aero_force    = np.zeros((len(mark.node),3))
    aero_moment   = np.zeros((len(mark.node),3))
    aero_bimoment = np.zeros((len(mark.node),3))
    aero_mom_pow  = np.zeros((len(mark.node),3))
    aero_pot_pow  = np.zeros((len(mark.node),))
    r_aerovec     = np.zeros((len(mark.node),3))
    load_vector   = np.zeros((len(mark.node),9))
    centrif_force = np.zeros((len(mark.node),3)) 
    cdflap_value0    = np.zeros((len(mark.node),))
    cdflap_value1    = np.zeros((len(mark.node),))
    clflap_value0    = np.zeros((len(mark.node),))
    clflap_value1    = np.zeros((len(mark.node),))
    cmflap_value0    = np.zeros((len(mark.node),))
    cmflap_value1    = np.zeros((len(mark.node),))   
    dragflap_vec0     = np.zeros((len(mark.node),3))
    dragflap_vec1     = np.zeros((len(mark.node),3))
    liftflap_vec0     = np.zeros((len(mark.node),3))
    liftflap_vec1     = np.zeros((len(mark.node),3))
    cmmomentflap_vec0 = np.zeros((len(mark.node),3))
    cmmomentflap_vec1 = np.zeros((len(mark.node),3))
    aero_forceflap    = np.zeros((len(mark.node),3))
    aero_momentflap   = np.zeros((len(mark.node),3))
    aero_bimomentflap = np.zeros((len(mark.node),3))
    aero_mom_powflap  = np.zeros((len(mark.node),3))
    aero_pot_powflap  = np.zeros((len(mark.node),))
    r_aerovecflap     = np.zeros((len(mark.node),3))
    load_vectorflap   = np.zeros((len(mark.node),9))
    centrif_forceflap = np.zeros((len(mark.node),3))
    aerocoeff         = np.zeros((len(mark.node),3))
    # For each node in the surface calculate the final load
    lsum = 0
    for iiaero_node in np.arange(len(mark.node)):
        iimark_node = int(mark.node.flatten()[iiaero_node])
        # alpha : transition factor
        alpha = (time_val-tmin_ann)/(ttran_ann-tmin_ann)
        if alpha > 1:
            alpha = 1
        elif alpha < 0:
            alpha = 0
        cl_value0[iiaero_node]        = alpha*(clst_value0[iiaero_node] + cldyn_value0[iiaero_node])+(1-alpha)*clst_value0[iiaero_node]
        cd_value0[iiaero_node]        = alpha*(cdst_value0[iiaero_node] + cddyn_value0[iiaero_node])+(1-alpha)*cdst_value0[iiaero_node]
        cm_value0[iiaero_node]        = alpha*(cmst_value0[iiaero_node] + cmdyn_value0[iiaero_node])+(1-alpha)*cmst_value0[iiaero_node]
        cl_value1[iiaero_node]        = alpha*(clst_value1[iiaero_node] + cldyn_value1[iiaero_node])+(1-alpha)*clst_value1[iiaero_node]
        cd_value1[iiaero_node]        = alpha*(cdst_value1[iiaero_node] + cddyn_value1[iiaero_node])+(1-alpha)*cdst_value1[iiaero_node]
        cm_value1[iiaero_node]        = alpha*(cmst_value1[iiaero_node] + cmdyn_value1[iiaero_node])+(1-alpha)*cmst_value1[iiaero_node]
        cl_tot[iimark_node]           = (cl_value0[iiaero_node]*LL_vec_p[0,iimark_node]+cl_value1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
        cd_tot[iimark_node]           = (cd_value0[iiaero_node]*LL_vec_p[0,iimark_node]+cd_value1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
        cm_tot[iimark_node]           = (cm_value0[iiaero_node]*LL_vec_p[0,iimark_node]+cm_value1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
        lift_vec0[iiaero_node,:]     += 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node]*cl_value0[iiaero_node]*np.array(refaxis2_0[iiaero_node,:])  
        drag_vec0[iiaero_node,:]     += 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node]*cd_value0[iiaero_node]*np.array(vdir0[iiaero_node,:])
        aero_force[iiaero_node,:]    += lift_vec0[iiaero_node,:]+drag_vec0[iiaero_node,:]
        aerocoeff[iiaero_node,:] = aero_force[iiaero_node,:]/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node])
        lsum += 2*(b_chord[iiaero_node]+b_chordflap[iiaero_node])*(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
        lift_vec1[iiaero_node,:]     += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[1,iimark_node]*cl_value1[iiaero_node]*np.array(refaxis2_1[iiaero_node,:])  
        drag_vec1[iiaero_node,:]     += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[1,iimark_node]*cd_value1[iiaero_node]*np.array(vdir1[iiaero_node,:])
        aero_force[iiaero_node,:]    += lift_vec1[iiaero_node,:]+drag_vec1[iiaero_node,:]
        cmmoment_vec0[iiaero_node,:] += 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[0,iimark_node]*cm_value0[iiaero_node]*np.array(refaxis_0[iiaero_node,:])
        aero_moment[iiaero_node,:]    = cmmoment_vec0[iiaero_node,:]
        cmmoment_vec1[iiaero_node,:] += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[1,iimark_node]*cm_value1[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
        aero_moment[iiaero_node,:]   += cmmoment_vec1[iiaero_node,:]
        r_aerovec[iiaero_node,:]      = np.matmul(rot_mat_p[iimark_node,1,:,:],np.concatenate(([section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]],[0])))
        aero_moment[iiaero_node,:]   += np.cross(r_aerovec[iiaero_node],aero_force[iiaero_node])
#        print(2*b_chord*2/(bound.radius*eff_3d.r_vinduced*np.pi))
        # if the rotative aerodynamics are activated
        if polar_bc.eff3d == 'BEM':
            fcent = 0
            for jjaero_node in np.arange(len(mark.node)):
                if lpos[jjaero_node]>=lpos[iiaero_node]:
                    jjmark_node = int(mark.node.flatten()[jjaero_node])
                    fcent      += eff_3d.masa[jjaero_node]*bound.vrot**2*lpos[jjaero_node] #eff_3d.masa[jjmark_node]*bound.vrot**2*lpos[jjaero_node]*(LL_vec_p[0,jjmark_node]+LL_vec_p[1,jjmark_node])
            pos_arm                       = np.matmul(rot_mat_p[iimark_node,0,:,:],[0,0,lpos[iiaero_node]])
            vec_centrif                   = pos_arm/np.linalg.norm(pos_arm)-np.dot(bound.refrot/np.linalg.norm(bound.refrot),pos_arm/np.linalg.norm(pos_arm))*bound.refrot/np.linalg.norm(bound.refrot)
            centrif_force[iiaero_node,:]  = fcent*vec_centrif/np.linalg.norm(vec_centrif)
            aero_mom_pow[iiaero_node,:]   = np.cross((pos_arm),aero_force[iiaero_node])
            aero_pot_pow[iiaero_node]     = np.dot(np.cross((pos_arm),aero_force[iiaero_node]),bound.vrot*bound.refrot/np.linalg.norm(bound.refrot))
        load_vector[iiaero_node,:]    = np.concatenate((aero_force[iiaero_node,:]+centrif_force[iiaero_node,:],aero_moment[iiaero_node,:],aero_bimoment[iiaero_node,:]))
        # Check if the node is the final one
        if ii_time == []:
            if iimark_node < num_point:
                FF_global[int(9*iimark_node):int(9*iimark_node+9)] += load_vector[iiaero_node,:]
            else:
                FF_global[int(iimark_node):] += load_vector[iiaero_node,:]
        else:
            if iimark_node < num_point:
                FF_global[int(9*iimark_node):int(9*iimark_node+9),ii_time] += load_vector[iiaero_node,:]
            else:
                FF_global[int(iimark_node):,ii_time] += load_vector[iiaero_node,:]
        if f_flap == 1  and markflap.node.flatten()[iiaero_node] >= 0:
            iimarkflap_node = int(markflap.node.flatten()[iiaero_node])
            clflap_value0[iiaero_node]        = alpha*(clstflap_value0[iiaero_node] + cldynflap_value0[iiaero_node])+(1-alpha)*clstflap_value0[iiaero_node]
            cdflap_value0[iiaero_node]        = alpha*(cdstflap_value0[iiaero_node] + cddynflap_value0[iiaero_node])+(1-alpha)*cdstflap_value0[iiaero_node]
            cmflap_value0[iiaero_node]        = alpha*(cmstflap_value0[iiaero_node] + cmdynflap_value0[iiaero_node])+(1-alpha)*cmstflap_value0[iiaero_node]
            clflap_value1[iiaero_node]        = alpha*(clstflap_value1[iiaero_node] + cldynflap_value1[iiaero_node])+(1-alpha)*clstflap_value1[iiaero_node]
            cdflap_value1[iiaero_node]        = alpha*(cdstflap_value1[iiaero_node] + cddynflap_value1[iiaero_node])+(1-alpha)*cdstflap_value1[iiaero_node]
            cmflap_value1[iiaero_node]        = alpha*(cmstflap_value1[iiaero_node] + cmdynflap_value1[iiaero_node])+(1-alpha)*cmstflap_value1[iiaero_node]
            cl_tot[iimarkflap_node]           = (clflap_value0[iiaero_node]*LL_vec_p[0,iimarkflap_node]+clflap_value1[iiaero_node]*LL_vec_p[1,iimarkflap_node])/(LL_vec_p[0,iimarkflap_node]+LL_vec_p[1,iimarkflap_node])
            cd_tot[iimarkflap_node]           = (cdflap_value0[iiaero_node]*LL_vec_p[0,iimarkflap_node]+cdflap_value1[iiaero_node]*LL_vec_p[1,iimarkflap_node])/(LL_vec_p[0,iimarkflap_node]+LL_vec_p[1,iimarkflap_node])
            cm_tot[iimarkflap_node]           = (cmflap_value0[iiaero_node]*LL_vec_p[0,iimarkflap_node]+cmflap_value1[iiaero_node]*LL_vec_p[1,iimarkflap_node])/(LL_vec_p[0,iimarkflap_node]+LL_vec_p[1,iimarkflap_node])
            liftflap_vec0[iiaero_node,:]     += 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[0,iimarkflap_node]*clflap_value0[iiaero_node]*np.array(refaxis2_0[iiaero_node,:])  
            dragflap_vec0[iiaero_node,:]     += 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[0,iimarkflap_node]*cdflap_value0[iiaero_node]*np.array(vdir0[iiaero_node,:])
            aero_forceflap[iiaero_node,:]    += liftflap_vec0[iiaero_node,:]+dragflap_vec0[iiaero_node,:]
            liftflap_vec1[iiaero_node,:]     += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[1,iimarkflap_node]*clflap_value1[iiaero_node]*np.array(refaxis2_1[iiaero_node,:])  
            dragflap_vec1[iiaero_node,:]     += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[1,iimarkflap_node]*cdflap_value1[iiaero_node]*np.array(vdir1[iiaero_node,:])
            aero_forceflap[iiaero_node,:]    += liftflap_vec1[iiaero_node,:]+dragflap_vec1[iiaero_node,:]
            cmmomentflap_vec0[iiaero_node,:] += 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimarkflap_node]*cmflap_value0[iiaero_node]*np.array(refaxis_0[iiaero_node,:])
            aero_momentflap[iiaero_node,:]    = cmmomentflap_vec0[iiaero_node,:]
            cmmomentflap_vec1[iiaero_node,:] += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[1,iimarkflap_node]*cmflap_value1[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
            aero_momentflap[iiaero_node,:]   += cmmomentflap_vec1[iiaero_node,:]
            r_aerovecflap[iiaero_node,:]      = np.matmul(rot_mat_p[iimarkflap_node,1,:,:],np.concatenate(([section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]],[0])))
            aero_momentflap[iiaero_node,:]   += np.cross(r_aerovecflap[iiaero_node],aero_forceflap[iiaero_node])
            # if the rotative aerodynamics are activated
            if polar_bc.eff3d == 'BEM':
                fcent = 0
                for jjaero_node in np.arange(len(mark.node)):
                    if lpos[jjaero_node]>=lpos[iiaero_node]:
                        jjmark_node = int(mark.node.flatten()[jjaero_node])
                        fcent      += eff_3d.masa[jjaero_node]*bound.vrot**2*lpos[jjaero_node] #eff_3d.masa[jjmark_node]*bound.vrot**2*lpos[jjaero_node]*(LL_vec_p[0,jjmark_node]+LL_vec_p[1,jjmark_node])
                pos_arm                       = np.matmul(rot_mat_p[iimarkflap_node,0,:,:],[0,0,lpos[iiaero_node]])
                vec_centrif                   = pos_arm/np.linalg.norm(pos_arm)-np.dot(bound.refrot/np.linalg.norm(bound.refrot),pos_arm/np.linalg.norm(pos_arm))*bound.refrot/np.linalg.norm(bound.refrot)
                centrif_forceflap[iiaero_node,:]  = fcent*vec_centrif/np.linalg.norm(vec_centrif)
                aero_mom_powflap[iiaero_node,:]   = np.cross((pos_arm),aero_forceflap[iiaero_node])
                aero_pot_powflap[iiaero_node]     = np.dot(np.cross((pos_arm),aero_forceflap[iiaero_node]),bound.vrot*bound.refrot/np.linalg.norm(bound.refrot))
            load_vectorflap[iiaero_node,:]    = np.concatenate((aero_forceflap[iiaero_node,:]+centrif_forceflap[iiaero_node,:],aero_momentflap[iiaero_node,:],aero_bimomentflap[iiaero_node,:]))
            # Check if the node is the final one
            if ii_time == []:
                if iimarkflap_node < num_point:
                    FF_global[int(9*iimarkflap_node):int(9*iimarkflap_node+9)] += load_vectorflap[iiaero_node,:]
                else:
                    FF_global[int(iimarkflap_node):] += load_vectorflap[iiaero_node,:]
            else:
                if iimarkflap_node < num_point:
                    FF_global[int(9*iimarkflap_node):int(9*iimarkflap_node+9),ii_time] += load_vectorflap[iiaero_node,:]
                else:
                    FF_global[int(iimarkflap_node):,ii_time] += load_vectorflap[iiaero_node,:]
    if polar_bc.eff3d == 'BEM':
        # CT_val : thrust coefficient
        # CQ_val : moment coefficient
        # CP_val : power coefficient
        # PE_val : propulsive efficiency
        disp_values.CT_val = bound.Nb*np.dot(sum(aero_force+aero_forceflap),bound.refrot/np.linalg.norm(bound.refrot))/(bound.rho*(bound.radius*bound.vrot)**2*np.pi*bound.radius**2)
        disp_values.CQ_val = bound.Nb*np.dot(sum(aero_mom_pow+aero_mom_powflap),bound.refrot/np.linalg.norm(bound.refrot))/(bound.rho*(bound.radius*bound.vrot)**2*np.pi*bound.radius**3)
        disp_values.CP_val = bound.Nb*sum(aero_pot_pow+aero_pot_powflap)/(bound.rho*(bound.radius*bound.vrot)**3*np.pi*bound.radius**2)
        if disp_values.CT_val > 0:
            disp_values.PE_val = disp_values.CT_val*bound.vinf/(bound.vrot*bound.radius)/disp_values.CP_val
        else:
            disp_values.PE_val = 0
        # save values of the aerodynamic parameters
#        print('CT: '+ str(disp_values.CT_val)+' CQ: '+str(disp_values.CQ_val)+' CP: '+str(disp_values.CP_val)+ ' PE: '+ str(disp_values.PE_val))
    else:
        # CL_val : Lift coefficient total
        # CD_val : Drag coefficient total
        # CM_val : Moment coefficient total
        disp_values.CL_val = np.dot(sum(aero_force+aero_forceflap),bound.refCL/np.linalg.norm(bound.refCL))/(1/2*bound.rho*bound.vinf**2*bound.s_ref)
        disp_values.CD_val = np.dot(sum(aero_force+aero_forceflap),bound.refCD/np.linalg.norm(bound.refCD))/(1/2*bound.rho*bound.vinf**2*bound.s_ref)
        disp_values.CM_val = np.dot(sum(aero_moment+aero_momentflap),bound.refCM/np.linalg.norm(bound.refCM)) /(1/2*bound.rho*bound.vinf**2*bound.s_ref*bound.l_ref)
        # save values of the aerodynamic parameters
#        print('CL: '+ str(disp_values.CL_val)+' CD: '+str(disp_values.CD_val)+' CM:'+str(disp_values.CM_val))

    disp_values.clsec_val = cl_tot
    disp_values.cdsec_val = cd_tot
    disp_values.cmsec_val = cm_tot 
    return FF_global,cl_value0,cl_value1,cm_value0,cm_value1,disp_values


#%%
def init_boundaryconditions_tran(KK_global,MM_global,CC_global,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p): 
    # Functions to determine the boundary conditions
    # -------------------------------------------------------------------------
    # KK_global         : stiffness matrix 
    # MM_global         : mass matrix
    # FF_global         : load vector 
    # mesh_mark         : markers of the mesh
    # case_setup        : information about the case setup
    # num_point         : number of points of the boundary conditions
    # sol_phys          : information of the solid physics
    # mesh_data         : data of the beam mesh
    # section           : information of the sectionk
    # rot_mat_p         : rotation matrix for each node and element
    # rot_matT_p        : transposed rotation matrix for each node and element
    # LL_vec_p          : beam longitude associated with each beam node
    # rr_vec_p          : distance vector associated with each beam element
    # -------------------------------------------------------------------------
    # ref_axis   : initialization of the reference axis
    KK_global = KK_global.copy()
    MM_global = MM_global.copy()
    CC_global = CC_global.copy()
    ref_axis   = []
    RR_global  = np.zeros((9*num_point,9*num_point))
    for mark in mesh_mark:
        for bound in case_setup.boundary:
            # For each marker in the mesh find if there is some boundary condition
            if mark.name == bound.marker:
                if bound.type == "BC_JOINT":
                    # If the bc is a joint between nodes
                    # node_1 : node 1 of the joint
                    # node_2 : node 2 of the joint
                    node_1 = int(mark.node[0])
                    node_2 = int(mark.node[1])
                    # xvec : distance between nodes in axis x
                    # yvec : distance between nodes in axis y
                    # zvec : distance between nodes in axis z
                    xvec = mesh_data.point[node_2,0]-mesh_data.point[node_1,0]
                    yvec = mesh_data.point[node_2,1]-mesh_data.point[node_1,1]
                    zvec = mesh_data.point[node_2,2]-mesh_data.point[node_1,2]
                    if bound.joint_type == "FIXED":
                        for ii_fixed in np.arange(9):
                            # for the 9 rows of the node 1
                            # add the values of node 2 in stiffness and loads
                            KK_global[node_1*9+ii_fixed,:] += KK_global[node_2*9+ii_fixed,:]
                            MM_global[node_1*9+ii_fixed,:] += MM_global[node_2*9+ii_fixed,:]
                        for ii_fixed in np.arange(9):
                            KK_global[:,node_1*9+ii_fixed] += KK_global[:,node_2*9+ii_fixed]
                            MM_global[:,node_1*9+ii_fixed] += MM_global[:,node_2*9+ii_fixed]
                        vj3 = -zvec*KK_global[:,node_2*9+1]+yvec*KK_global[:,node_2*9+2]
                        vj4 = zvec*KK_global[:,node_2*9]-xvec*KK_global[:,node_2*9+2]
                        vj5 = -yvec*KK_global[:,node_2*9]+xvec*KK_global[:,node_2*9+1]
                        vi3 = yvec*KK_global[node_2*9+2,:]-zvec*KK_global[node_2*9+1,:]
                        vi4 = zvec*KK_global[node_2*9,:]-xvec*KK_global[node_2*9+2,:]
                        vi5 = xvec*KK_global[node_2*9+1,:]-yvec*KK_global[node_2*9,:]
                        # If the boundary condition is a rigid solid connection between bodies
                        KK_global[:,node_1*9+3] += vj3
                        KK_global[:,node_1*9+4] += vj4
                        KK_global[:,node_1*9+5] += vj5
                        KK_global[node_1*9+3,:] += vi3
                        KK_global[node_1*9+4,:] += vi4
                        KK_global[node_1*9+5,:] += vi5
                        KK_global[node_1*9+3,node_1*9+3] += yvec**2*KK_global[node_2*9+2,node_2*9+2]+zvec**2*KK_global[node_2*9+1,node_2*9+1]+2*yvec*(-zvec)*KK_global[node_2*9+1,node_2*9+2]
                        KK_global[node_1*9+4,node_1*9+4] += xvec**2*KK_global[node_2*9+2,node_2*9+2]+zvec**2*KK_global[node_2*9,node_2*9]+2*zvec*(-xvec)*KK_global[node_2*9,node_2*9+2]
                        KK_global[node_1*9+5,node_1*9+5] += yvec**2*KK_global[node_2*9,node_2*9]+xvec**2*KK_global[node_2*9+1,node_2*9+1]+2*xvec*(-yvec)*KK_global[node_2*9,node_2*9+1]
                        KK_global[node_1*9+3,node_1*9+4] += -xvec*yvec*KK_global[node_2*9+2,node_2*9+2]+xvec*zvec*KK_global[node_2*9+1,node_2*9+2]+yvec*zvec*KK_global[node_2*9,node_2*9+2]-zvec**2*KK_global[node_2*9,node_2*9+1]
                        KK_global[node_1*9+4,node_1*9+3] += -xvec*yvec*KK_global[node_2*9+2,node_2*9+2]+xvec*zvec*KK_global[node_2*9+1,node_2*9+2]+yvec*zvec*KK_global[node_2*9,node_2*9+2]-zvec**2*KK_global[node_2*9,node_2*9+1]
                        KK_global[node_1*9+3,node_1*9+5] += xvec*yvec*KK_global[node_2*9+1,node_2*9+2]-xvec*zvec*KK_global[node_2*9+1,node_2*9+1]+yvec*zvec*KK_global[node_2*9,node_2*9+1]-yvec**2*KK_global[node_2*9,node_2*9+2]
                        KK_global[node_1*9+5,node_1*9+3] += xvec*yvec*KK_global[node_2*9+1,node_2*9+2]-xvec*zvec*KK_global[node_2*9+1,node_2*9+1]+yvec*zvec*KK_global[node_2*9,node_2*9+1]-yvec**2*KK_global[node_2*9,node_2*9+2]
                        KK_global[node_1*9+4,node_1*9+5] += xvec*yvec*KK_global[node_2*9,node_2*9+2]+xvec*zvec*KK_global[node_2*9,node_2*9+1]-yvec*zvec*KK_global[node_2*9,node_2*9]-xvec**2*KK_global[node_2*9+1,node_2*9+2]
                        KK_global[node_1*9+5,node_1*9+4] += xvec*yvec*KK_global[node_2*9,node_2*9+2]+xvec*zvec*KK_global[node_2*9,node_2*9+1]-yvec*zvec*KK_global[node_2*9,node_2*9]-xvec**2*KK_global[node_2*9+1,node_2*9+2]
                        vj3 = -zvec*MM_global[:,node_2*9+1]+yvec*MM_global[:,node_2*9+2]
                        vj4 = zvec*MM_global[:,node_2*9]-xvec*MM_global[:,node_2*9+2]
                        vj5 = -yvec*MM_global[:,node_2*9]+xvec*MM_global[:,node_2*9+1]
                        vi3 = yvec*MM_global[node_2*9+2,:]-zvec*MM_global[node_2*9+1,:]
                        vi4 = zvec*MM_global[node_2*9,:]-xvec*MM_global[node_2*9+2,:]
                        vi5 = xvec*MM_global[node_2*9+1,:]-yvec*MM_global[node_2*9,:]
                        MM_global[:,node_1*9+3] += vj3
                        MM_global[:,node_1*9+4] += vj4
                        MM_global[:,node_1*9+5] += vj5
                        MM_global[node_1*9+3,:] += vi3
                        MM_global[node_1*9+4,:] += vi4
                        MM_global[node_1*9+5,:] += vi5
                        MM_global[node_1*9+3,node_1*9+3] += zvec**2*MM_global[node_2*9+1,node_2*9+1]+yvec**2*MM_global[node_2*9+2,node_2*9+2]+2*yvec*(-zvec)*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_1*9+4,node_1*9+4] += xvec**2*MM_global[node_2*9+2,node_2*9+2]+zvec**2*MM_global[node_2*9,node_2*9]+2*zvec*(-xvec)*MM_global[node_2*9,node_2*9+2]
                        MM_global[node_1*9+5,node_1*9+5] += yvec**2*MM_global[node_2*9,node_2*9]+xvec**2*MM_global[node_2*9+1,node_2*9+1]+2*xvec*(-yvec)*MM_global[node_2*9,node_2*9+1]
                        MM_global[node_1*9+3,node_1*9+4] += -xvec*yvec*MM_global[node_2*9+2,node_2*9+2]+xvec*zvec*MM_global[node_2*9+1,node_2*9+2]+yvec*zvec*MM_global[node_2*9,node_2*9+2]-zvec**2*MM_global[node_2*9,node_2*9+1]
                        MM_global[node_1*9+4,node_1*9+3] += -xvec*yvec*MM_global[node_2*9+2,node_2*9+2]+xvec*zvec*MM_global[node_2*9+1,node_2*9+2]+yvec*zvec*MM_global[node_2*9,node_2*9+2]-zvec**2*MM_global[node_2*9,node_2*9+1]
                        MM_global[node_1*9+3,node_1*9+5] += xvec*yvec*MM_global[node_2*9+1,node_2*9+2]-xvec*zvec*MM_global[node_2*9+1,node_2*9+1]+yvec*zvec*MM_global[node_2*9,node_2*9+1]-yvec**2*MM_global[node_2*9,node_2*9+2]
                        MM_global[node_1*9+5,node_1*9+3] += xvec*yvec*MM_global[node_2*9+1,node_2*9+2]-xvec*zvec*MM_global[node_2*9+1,node_2*9+1]+yvec*zvec*MM_global[node_2*9,node_2*9+1]-yvec**2*MM_global[node_2*9,node_2*9+2]
                        MM_global[node_1*9+4,node_1*9+5] += xvec*yvec*MM_global[node_2*9,node_2*9+2]+xvec*zvec*MM_global[node_2*9,node_2*9+1]-yvec*zvec*MM_global[node_2*9,node_2*9]-xvec**2*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_1*9+5,node_1*9+4] += xvec*yvec*MM_global[node_2*9,node_2*9+2]+xvec*zvec*MM_global[node_2*9,node_2*9+1]-yvec*zvec*MM_global[node_2*9,node_2*9]-xvec**2*MM_global[node_2*9+1,node_2*9+2]
                        for ii_fixed in np.arange(9):
                            KK_global[node_2*9+ii_fixed,:] *= 0
                            KK_global[:,node_2*9+ii_fixed] *= 0
                            MM_global[node_2*9+ii_fixed,:] *= 0
                            MM_global[:,node_2*9+ii_fixed] *= 0
                        RR_global[node_2*9,node_1*9] = 1
                        RR_global[node_2*9,node_2*9] = -1
                        RR_global[node_2*9,node_1*9+4] = zvec
                        RR_global[node_2*9,node_1*9+5] = -yvec
                        RR_global[node_2*9+1,node_1*9+1] = 1
                        RR_global[node_2*9+1,node_2*9+1] = -1
                        RR_global[node_2*9+1,node_1*9+3] = -zvec
                        RR_global[node_2*9+1,node_1*9+5] = xvec
                        RR_global[node_2*9+2,node_1*9+2] = 1
                        RR_global[node_2*9+2,node_2*9+2] = -1
                        RR_global[node_2*9+2,node_1*9+3] = yvec
                        RR_global[node_2*9+2,node_1*9+4] = -xvec
                        RR_global[node_2*9+3,node_1*9+3] = 1
                        RR_global[node_2*9+3,node_2*9+3] = -1
                        RR_global[node_2*9+4,node_1*9+4] = 1
                        RR_global[node_2*9+4,node_2*9+4] = -1
                        RR_global[node_2*9+5,node_1*9+5] = 1
                        RR_global[node_2*9+5,node_2*9+5] = -1
                        RR_global[node_2*9+6,node_1*9+6] = 1
                        RR_global[node_2*9+6,node_2*9+6] = -1
                        RR_global[node_2*9+7,node_1*9+7] = 1
                        RR_global[node_2*9+7,node_2*9+7] = -1
                        RR_global[node_2*9+8,node_1*9+8] = 1
                        RR_global[node_2*9+8,node_2*9+8] = -1
#                        if len(CC_global) > 0:
#                            for ii_fixed in np.arange(9):
#                                # for the 9 rows of the node 1
#                                # add the values of node 2 in stiffness and loads
#                                CC_global[node_1*9+ii_fixed,:] += CC_global[node_2*9+ii_fixed,:]
#                            for ii_fixed in np.arange(9):
#                                CC_global[:,node_1*9+ii_fixed] += CC_global[:,node_2*9+ii_fixed]
#                            vj3 = -zvec*CC_global[:,node_2*9+1]+yvec*CC_global[:,node_2*9+2]
#                            vj4 = zvec*CC_global[:,node_2*9]-xvec*CC_global[:,node_2*9+2]
#                            vj5 = -yvec*CC_global[:,node_2*9]+xvec*CC_global[:,node_2*9+1]
#                            vi3 = yvec*CC_global[node_2*9+2,:]-zvec*CC_global[node_2*9+1,:]
#                            vi4 = zvec*CC_global[node_2*9,:]-xvec*CC_global[node_2*9+2,:]
#                            vi5 = xvec*CC_global[node_2*9+1,:]-yvec*CC_global[node_2*9,:]
#                            CC_global[:,node_1*9+3] += vj3
#                            CC_global[:,node_1*9+4] += vj4
#                            CC_global[:,node_1*9+5] += vj5
#                            CC_global[node_1*9+3,:] += vi3
#                            CC_global[node_1*9+4,:] += vi4
#                            CC_global[node_1*9+5,:] += vi5
#                            CC_global[node_1*9+3,node_1*9+3] += zvec**2*CC_global[node_2*9+1,node_2*9+1]+yvec**2*CC_global[node_2*9+2,node_2*9+2]+2*yvec*(-zvec)*CC_global[node_2*9+1,node_2*9+2]
#                            CC_global[node_1*9+4,node_1*9+4] += xvec**2*CC_global[node_2*9+2,node_2*9+2]+zvec**2*CC_global[node_2*9,node_2*9]+2*zvec*(-xvec)*CC_global[node_2*9,node_2*9+2]
#                            CC_global[node_1*9+5,node_1*9+5] += yvec**2*CC_global[node_2*9,node_2*9]+xvec**2*CC_global[node_2*9+1,node_2*9+1]+2*xvec*(-yvec)*CC_global[node_2*9,node_2*9+1]
#                            CC_global[node_1*9+3,node_1*9+4] += -xvec*yvec*CC_global[node_2*9+2,node_2*9+2]+xvec*zvec*CC_global[node_2*9+1,node_2*9+2]+yvec*zvec*CC_global[node_2*9,node_2*9+2]-zvec**2*CC_global[node_2*9,node_2*9+1]
#                            CC_global[node_1*9+4,node_1*9+3] += -xvec*yvec*CC_global[node_2*9+2,node_2*9+2]+xvec*zvec*CC_global[node_2*9+1,node_2*9+2]+yvec*zvec*CC_global[node_2*9,node_2*9+2]-zvec**2*CC_global[node_2*9,node_2*9+1]
#                            CC_global[node_1*9+3,node_1*9+5] += xvec*yvec*CC_global[node_2*9+1,node_2*9+2]-xvec*zvec*CC_global[node_2*9+1,node_2*9+1]+yvec*zvec*CC_global[node_2*9,node_2*9+1]-yvec**2*CC_global[node_2*9,node_2*9+2]
#                            CC_global[node_1*9+5,node_1*9+3] += xvec*yvec*CC_global[node_2*9+1,node_2*9+2]-xvec*zvec*CC_global[node_2*9+1,node_2*9+1]+yvec*zvec*CC_global[node_2*9,node_2*9+1]-yvec**2*CC_global[node_2*9,node_2*9+2]
#                            CC_global[node_1*9+4,node_1*9+5] += xvec*yvec*CC_global[node_2*9,node_2*9+2]+xvec*zvec*CC_global[node_2*9,node_2*9+1]-yvec*zvec*CC_global[node_2*9,node_2*9]-xvec**2*CC_global[node_2*9+1,node_2*9+2]
#                            CC_global[node_1*9+5,node_1*9+4] += xvec*yvec*CC_global[node_2*9,node_2*9+2]+xvec*zvec*CC_global[node_2*9,node_2*9+1]-yvec*zvec*CC_global[node_2*9,node_2*9]-xvec**2*CC_global[node_2*9+1,node_2*9+2]
#                            for ii_fixed in np.arange(9):
#                                CC_global[node_2*9+ii_fixed,:] *= 0
#                                CC_global[:,node_2*9+ii_fixed] *= 0
                        # The rows of the node 2 must be deleted and the relation between displacements is added
                    elif bound.joint_type == "ROTATE_AXIS":
                        # if joint is a rotation joint
                        # a_axis : axis of rotation of the beam
                        # r_axis : unitary vector of the distance between nodes
                        # r_0_2  : distance from point of rotation to section 2
                        # r_1_0  : distance from section 1 to point of rotation
                        # n_axis : normal axis from rotation axis and nodes relative position
                        # x1     : distance between node 1 and joint in axis x
                        # y1     : distance between node 1 and joint in axis y
                        # x2     : distance between joint and node 2 in axis x
                        # y2     : distance between joint and node 2 in axis y
                        a_axis = np.array(bound.joint_axis)/np.linalg.norm(np.array(bound.joint_axis))
                        p_dir  = np.argmax(abs(a_axis))
                        if p_dir == 0:
                            ii_1 = 1
                            ii_2 = 2
                            ii_a = -1
                            ii_b = -2
                        elif p_dir == 1:
                            ii_1 = 0
                            ii_2 = 2
                            ii_a = 1
                            ii_b = -1
                        elif p_dir == 2:
                            ii_1 = 0
                            ii_2 = 1
                            ii_a = 2
                            ii_b = 1
                        if np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])>0:
                            r_axis = (mesh_data.point[node_2]-mesh_data.point[node_1])/np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])
                        else:
                            if a_axis[0] == 0:
                                r_axis = np.array([1,0,0])
                            elif a_axis[1] == 0:
                                r_axis = np.array([0,1,0])
                            elif a_axis[2] == 0:
                                r_axis = np.array([0,0,1])
                            else:
                                r_axis = np.array([a_axis[1],-a_axis[0],a_axis[2]])/np.linalg.norm(np.array([a_axis[1],-a_axis[0],a_axis[2]]))
                        r_0_2 = mesh_data.point[node_2]-bound.point_axis
                        r_0_1 = mesh_data.point[node_1]-bound.point_axis
                        x1 = r_0_1[0]
                        y1 = r_0_1[1]
                        x2 = r_0_2[0]
                        y2 = r_0_2[1]
                        n_axis = np.cross(a_axis,r_axis)
                        # rot_mat_axis : Rotation matrix of the degree of freedom. From rotation axis to global 
                        rot_mat_axis = np.array([[np.dot(r_axis,[1,0,0]),np.dot(n_axis,[1,0,0]),np.dot(a_axis,[1,0,0])],
                                                  [np.dot(r_axis,[0,1,0]),np.dot(n_axis,[0,1,0]),np.dot(a_axis,[0,1,0])],
                                                  [np.dot(r_axis,[0,0,1]),np.dot(n_axis,[0,0,1]),np.dot(a_axis,[0,0,1])]])
                        # stiffness matrix and load vector in the nodes is rotated to the joint axis
                        for ii_point in np.arange(num_point):
                            for jj_point in np.arange(num_point):
                                for ii_row in np.arange(3):
                                    for jj_col in np.arange(3):
                                        KK_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)] = np.matmul(rot_mat_axis,np.matmul(KK_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)],np.transpose(rot_mat_axis)))
                                        MM_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)] = np.matmul(rot_mat_axis,np.matmul(MM_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)],np.transpose(rot_mat_axis)))
                                        RR_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)] = np.matmul(rot_mat_axis,np.matmul(RR_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)],np.transpose(rot_mat_axis)))
                        # For all the rows of the points in the stiffness matrix and load vector
                        for ii_fixed in np.arange(9):
                            # Apply the restrictions in all the displacements except of the rotation around the axis
                            if ii_fixed != 5 and ii_fixed != 8:                         
                                KK_global[node_1*9+ii_fixed,:] += KK_global[node_2*9+ii_fixed,:]
                                MM_global[node_1*9+ii_fixed,:] += MM_global[node_2*9+ii_fixed,:]
                        for ii_fixed in np.arange(9):
                            # Apply the restrictions in all the displacements except of the rotation around the axis
                            if ii_fixed != 5 and ii_fixed != 8: 
                                KK_global[:,node_1*9+ii_fixed] += KK_global[:,node_2*9+ii_fixed]
                                MM_global[:,node_1*9+ii_fixed] += MM_global[:,node_2*9+ii_fixed]
                        vj3 = -zvec*KK_global[:,node_2*9+1]+yvec*KK_global[:,node_2*9+2]
                        vj4 = zvec*KK_global[:,node_2*9]-xvec*KK_global[:,node_2*9+2]
                        vj5 = -yvec*KK_global[:,node_2*9]+xvec*KK_global[:,node_2*9+1]
                        vi3 = yvec*KK_global[node_2*9+2,:]-zvec*KK_global[node_2*9+1,:]
                        vi4 = zvec*KK_global[node_2*9,:]-xvec*KK_global[node_2*9+2,:]
                        vi5 = xvec*KK_global[node_2*9+1,:]-yvec*KK_global[node_2*9,:]
                        vj5_2 = -y2*KK_global[:,node_2*9]+x2*KK_global[:,node_2*9+1]
                        vi5_2 = x2*KK_global[node_2*9+1,:]-y2*KK_global[node_2*9,:]
                        # If the boundary condition is a rigid solid connection between bodies
                        KK_global[:,node_1*9+3] += vj3
                        KK_global[:,node_1*9+4] += vj4
                        KK_global[:,node_1*9+5] += vj5
                        KK_global[node_1*9+3,:] += vi3
                        KK_global[node_1*9+4,:] += vi4
                        KK_global[node_1*9+5,:] += vi5
                        KK_global[:,node_2*9+5] += vj5_2
                        KK_global[node_2*9+5,:] += vi5_2
                        KK_global[node_1*9+3,node_1*9+3] += y1**2*KK_global[node_2*9+2,node_2*9+2]+zvec**2*KK_global[node_2*9+1,node_2*9+1]+2*y1*zvec*KK_global[node_2*9+1,node_2*9+2]
                        KK_global[node_1*9+4,node_1*9+4] += x1**2*KK_global[node_2*9+2,node_2*9+2]+zvec**2*KK_global[node_2*9,node_2*9]+2*zvec*x1*KK_global[node_2*9,node_2*9+2]
                        KK_global[node_1*9+5,node_1*9+5] += y1**2*KK_global[node_2*9,node_2*9]+x1**2*KK_global[node_2*9+1,node_2*9+1]-2*x1*y1*KK_global[node_2*9,node_2*9+1]
                        KK_global[node_1*9+3,node_1*9+4] += -x1*y1*KK_global[node_2*9+2,node_2*9+2]-x1*zvec*KK_global[node_2*9+1,node_2*9+2]-y1*zvec*KK_global[node_2*9,node_2*9+2]-zvec**2*KK_global[node_2*9,node_2*9+1]
                        KK_global[node_1*9+4,node_1*9+3] += -x1*y1*KK_global[node_2*9+2,node_2*9+2]-x1*zvec*KK_global[node_2*9+1,node_2*9+2]-y1*zvec*KK_global[node_2*9,node_2*9+2]-zvec**2*KK_global[node_2*9,node_2*9+1]
                        KK_global[node_1*9+3,node_1*9+5] += x1*y1*KK_global[node_2*9+1,node_2*9+2]+x1*zvec*KK_global[node_2*9+1,node_2*9+1]-y1*zvec*KK_global[node_2*9,node_2*9+1]-y1**2*KK_global[node_2*9,node_2*9+2]
                        KK_global[node_1*9+5,node_1*9+3] += x1*y1*KK_global[node_2*9+1,node_2*9+2]+x1*zvec*KK_global[node_2*9+1,node_2*9+1]-y1*zvec*KK_global[node_2*9,node_2*9+1]-y1**2*KK_global[node_2*9,node_2*9+2]
                        KK_global[node_1*9+4,node_1*9+5] += x1*y1*KK_global[node_2*9,node_2*9+2]-x1*zvec*KK_global[node_2*9,node_2*9+1]+y1*zvec*KK_global[node_2*9,node_2*9]-x1**2*KK_global[node_2*9+1,node_2*9+2]
                        KK_global[node_1*9+5,node_1*9+4] += x1*y1*KK_global[node_2*9,node_2*9+2]-x1*zvec*KK_global[node_2*9,node_2*9+1]+y1*zvec*KK_global[node_2*9,node_2*9]-x1**2*KK_global[node_2*9+1,node_2*9+2]                       
                        KK_global[node_1*9+3,node_2*9+5] += -x2*y1*KK_global[node_2*9+1,node_2*9+2]-x2*zvec*KK_global[node_2*9+1,node_2*9+1]+y2*zvec*KK_global[node_2*9,node_2*9+1]+y1*y2*KK_global[node_2*9,node_2*9+2]
                        KK_global[node_2*9+5,node_1*9+3] += -x2*y1*KK_global[node_2*9+1,node_2*9+2]-x2*zvec*KK_global[node_2*9+1,node_2*9+1]+y2*zvec*KK_global[node_2*9,node_2*9+1]+y1*y2*KK_global[node_2*9,node_2*9+2]
                        KK_global[node_1*9+4,node_2*9+5] += -x1*y2*KK_global[node_2*9,node_2*9+2]+x2*zvec*KK_global[node_2*9,node_2*9+1]-y2*zvec*KK_global[node_2*9,node_2*9]-x1*x2*KK_global[node_2*9+1,node_2*9+2]
                        KK_global[node_2*9+5,node_1*9+4] += -x1*y2*KK_global[node_2*9,node_2*9+2]+x2*zvec*KK_global[node_2*9,node_2*9+1]-y2*zvec*KK_global[node_2*9,node_2*9]-x1*x2*KK_global[node_2*9+1,node_2*9+2]
                        KK_global[node_1*9+5,node_2*9+5] += x1*y2*KK_global[node_2*9,node_2*9+2]+x2*y1*KK_global[node_2*9,node_2*9+1]-y2*y1*KK_global[node_2*9,node_2*9]-x1*x2*KK_global[node_2*9+1,node_2*9+2]
                        KK_global[node_2*9+5,node_1*9+5] += x1*y2*KK_global[node_2*9,node_2*9+2]+x2*y1*KK_global[node_2*9,node_2*9+1]-y2*y1*KK_global[node_2*9,node_2*9]-x1*x2*KK_global[node_2*9+1,node_2*9+2]
                        KK_global[node_2*9+5,node_2*9+5] += y2**2*KK_global[node_2*9,node_2*9]+x2**2*KK_global[node_2*9+1,node_2*9+1]-2*x2*y2*KK_global[node_2*9,node_2*9+1]  
                        vj3 = -zvec*MM_global[:,node_2*9+1]+yvec*MM_global[:,node_2*9+2]
                        vj4 = zvec*MM_global[:,node_2*9]-xvec*MM_global[:,node_2*9+2]
                        vj5 = -yvec*MM_global[:,node_2*9]+xvec*MM_global[:,node_2*9+1]
                        vi3 = yvec*MM_global[node_2*9+2,:]-zvec*MM_global[node_2*9+1,:]
                        vi4 = zvec*MM_global[node_2*9,:]-xvec*MM_global[node_2*9+2,:]
                        vi5 = xvec*MM_global[node_2*9+1,:]-yvec*MM_global[node_2*9,:]
                        vj5_2 = -y2*MM_global[:,node_2*9]+x2*MM_global[:,node_2*9+1]
                        vi5_2 = x2*MM_global[node_2*9+1,:]-y2*MM_global[node_2*9,:]
                        # If the boundary condition is a rigid solid connection between bodies
                        MM_global[:,node_1*9+3] += vj3
                        MM_global[:,node_1*9+4] += vj4
                        MM_global[:,node_1*9+5] += vj5
                        MM_global[node_1*9+3,:] += vi3
                        MM_global[node_1*9+4,:] += vi4
                        MM_global[node_1*9+5,:] += vi5
                        MM_global[:,node_2*9+5] += vj5_2
                        MM_global[node_2*9+5,:] += vi5_2                      
                        MM_global[node_1*9+3,node_1*9+3] += y1**2*MM_global[node_2*9+2,node_2*9+2]+zvec**2*MM_global[node_2*9+1,node_2*9+1]+2*y1*zvec*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_1*9+4,node_1*9+4] += x1**2*MM_global[node_2*9+2,node_2*9+2]+zvec**2*MM_global[node_2*9,node_2*9]+2*zvec*x1*MM_global[node_2*9,node_2*9+2]
                        MM_global[node_1*9+5,node_1*9+5] += y1**2*MM_global[node_2*9,node_2*9]+x1**2*MM_global[node_2*9+1,node_2*9+1]-2*x1*y1*MM_global[node_2*9,node_2*9+1]
                        MM_global[node_1*9+3,node_1*9+4] += -x1*y1*MM_global[node_2*9+2,node_2*9+2]-x1*zvec*MM_global[node_2*9+1,node_2*9+2]-y1*zvec*MM_global[node_2*9,node_2*9+2]-zvec**2*MM_global[node_2*9,node_2*9+1]
                        MM_global[node_1*9+4,node_1*9+3] += -x1*y1*MM_global[node_2*9+2,node_2*9+2]-x1*zvec*MM_global[node_2*9+1,node_2*9+2]-y1*zvec*MM_global[node_2*9,node_2*9+2]-zvec**2*MM_global[node_2*9,node_2*9+1]
                        MM_global[node_1*9+3,node_1*9+5] += x1*y1*MM_global[node_2*9+1,node_2*9+2]+x1*zvec*MM_global[node_2*9+1,node_2*9+1]-y1*zvec*MM_global[node_2*9,node_2*9+1]-y1**2*MM_global[node_2*9,node_2*9+2]
                        MM_global[node_1*9+5,node_1*9+3] += x1*y1*MM_global[node_2*9+1,node_2*9+2]+x1*zvec*MM_global[node_2*9+1,node_2*9+1]-y1*zvec*MM_global[node_2*9,node_2*9+1]-y1**2*MM_global[node_2*9,node_2*9+2]
                        MM_global[node_1*9+4,node_1*9+5] += x1*y1*MM_global[node_2*9,node_2*9+2]-x1*zvec*MM_global[node_2*9,node_2*9+1]+y1*zvec*MM_global[node_2*9,node_2*9]-x1**2*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_1*9+5,node_1*9+4] += x1*y1*MM_global[node_2*9,node_2*9+2]-x1*zvec*MM_global[node_2*9,node_2*9+1]+y1*zvec*MM_global[node_2*9,node_2*9]-x1**2*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_1*9+3,node_2*9+5] += -x2*y1*MM_global[node_2*9+1,node_2*9+2]-x2*zvec*MM_global[node_2*9+1,node_2*9+1]+y2*zvec*MM_global[node_2*9,node_2*9+1]+y1*y2*MM_global[node_2*9,node_2*9+2]
                        MM_global[node_2*9+5,node_1*9+3] += -x2*y1*MM_global[node_2*9+1,node_2*9+2]-x2*zvec*MM_global[node_2*9+1,node_2*9+1]+y2*zvec*MM_global[node_2*9,node_2*9+1]+y1*y2*MM_global[node_2*9,node_2*9+2]
                        MM_global[node_1*9+4,node_2*9+5] += -x1*y2*MM_global[node_2*9,node_2*9+2]+x2*zvec*MM_global[node_2*9,node_2*9+1]-y2*zvec*MM_global[node_2*9,node_2*9]-x1*x2*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_2*9+5,node_1*9+4] += -x1*y2*MM_global[node_2*9,node_2*9+2]+x2*zvec*MM_global[node_2*9,node_2*9+1]-y2*zvec*MM_global[node_2*9,node_2*9]-x1*x2*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_1*9+5,node_2*9+5] += x1*y2*MM_global[node_2*9,node_2*9+2]+x2*y1*MM_global[node_2*9,node_2*9+1]-y2*y1*MM_global[node_2*9,node_2*9]-x1*x2*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_2*9+5,node_1*9+5] += x1*y2*MM_global[node_2*9,node_2*9+2]+x2*y1*MM_global[node_2*9,node_2*9+1]-y2*y1*MM_global[node_2*9,node_2*9]-x1*x2*MM_global[node_2*9+1,node_2*9+2]
                        MM_global[node_2*9+5,node_2*9+5] += y2**2*MM_global[node_2*9,node_2*9]+x2**2*MM_global[node_2*9+1,node_2*9+1]-2*x2*y2*MM_global[node_2*9,node_2*9+1]
                        RR_global[node_2*9,node_1*9]   = -1
                        RR_global[node_2*9,node_2*9]   = 1
                        RR_global[node_2*9,node_1*9+4] = -zvec
                        RR_global[node_2*9,node_1*9+5] = -y1
                        RR_global[node_2*9,node_2*9+5] = y2
                        RR_global[node_2*9+1,node_1*9+1] = -1
                        RR_global[node_2*9+1,node_2*9+1] = 1
                        RR_global[node_2*9+1,node_1*9+3] = zvec
                        RR_global[node_2*9+1,node_1*9+5] = x1
                        RR_global[node_2*9+1,node_2*9+5] = -x2
                        RR_global[node_2*9+2,node_1*9+2] = -1
                        RR_global[node_2*9+2,node_2*9+2] = 1
                        RR_global[node_2*9+2,node_1*9+3] = -yvec
                        RR_global[node_2*9+2,node_1*9+4] = xvec
                        RR_global[node_2*9+3,node_1*9+3] = -1
                        RR_global[node_2*9+3,node_2*9+3] = 1
                        RR_global[node_2*9+4,node_1*9+4] = -1
                        RR_global[node_2*9+4,node_2*9+4] = 1
                        RR_global[node_2*9+6,node_1*9+6] = -1
                        RR_global[node_2*9+6,node_2*9+6] = 1
                        RR_global[node_2*9+7,node_1*9+7] = -1
                        RR_global[node_2*9+7,node_2*9+7] = 1 
                        for ii_fixed in np.arange(9):
                            # Apply the restrictions in all the displacements except of the rotation around the axis
                            if ii_fixed != 5 and ii_fixed != 8:  
                                KK_global[node_2*9+ii_fixed,:] *= 0
                                KK_global[:,node_2*9+ii_fixed] *= 0  
                                MM_global[node_2*9+ii_fixed,:] *= 0
                                MM_global[:,node_2*9+ii_fixed] *= 0            
                                # The rows of the node 2 must be deleted and the relation between displacements is added
                        # The stiffness matrix and the load vector are rotated back to the global frame
                        for ii_point in np.arange(num_point):
                            for jj_point in np.arange(num_point):
                                for ii_row in np.arange(3):
                                    for jj_col in np.arange(3):
                                        KK_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)] = np.matmul(np.transpose(rot_mat_axis),np.matmul(KK_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)],rot_mat_axis))
                                        MM_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)] = np.matmul(np.transpose(rot_mat_axis),np.matmul(MM_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)],rot_mat_axis))
                                        RR_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)] = np.matmul(np.transpose(rot_mat_axis),np.matmul(RR_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)],rot_mat_axis)) 
#                        if len(CC_global) > 0:
#                            for ii_point in np.arange(num_point):
#                                for jj_point in np.arange(num_point):
#                                    for ii_row in np.arange(3):
#                                        for jj_col in np.arange(3):
#                                            CC_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)] = np.matmul(rot_mat_axis,np.matmul(CC_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)],np.transpose(rot_mat_axis)))
#                            # For all the rows of the points in the stiffness matrix and load vector
#                            for ii_fixed in np.arange(9):
#                                # Apply the restrictions in all the displacements except of the rotation around the axis
#                                if ii_fixed != 5 and ii_fixed != 8:                         
#                                    CC_global[node_1*9+ii_fixed,:] += CC_global[node_2*9+ii_fixed,:]
#                            for ii_fixed in np.arange(9):
#                                # Apply the restrictions in all the displacements except of the rotation around the axis
#                                if ii_fixed != 5 and ii_fixed != 8: 
#                                    CC_global[:,node_1*9+ii_fixed] += CC_global[:,node_2*9+ii_fixed]
#                            vj3 = -zvec*CC_global[:,node_2*9+1]+yvec*CC_global[:,node_2*9+2]
#                            vj4 = zvec*CC_global[:,node_2*9]-xvec*CC_global[:,node_2*9+2]
#                            vj5 = -yvec*CC_global[:,node_2*9]+xvec*CC_global[:,node_2*9+1]
#                            vi3 = yvec*CC_global[node_2*9+2,:]-zvec*CC_global[node_2*9+1,:]
#                            vi4 = zvec*CC_global[node_2*9,:]-xvec*CC_global[node_2*9+2,:]
#                            vi5 = xvec*CC_global[node_2*9+1,:]-yvec*CC_global[node_2*9,:]
#                            vj5_2 = -y2*CC_global[:,node_2*9]+x2*CC_global[:,node_2*9+1]
#                            vi5_2 = x2*CC_global[node_2*9+1,:]-y2*CC_global[node_2*9,:]
#                            # If the boundary condition is a rigid solid connection between bodies
#                            CC_global[:,node_1*9+3] += vj3
#                            CC_global[:,node_1*9+4] += vj4
#                            CC_global[:,node_1*9+5] += vj5
#                            CC_global[node_1*9+3,:] += vi3
#                            CC_global[node_1*9+4,:] += vi4
#                            CC_global[node_1*9+5,:] += vi5
#                            CC_global[:,node_2*9+5] += vj5_2
#                            CC_global[node_2*9+5,:] += vi5_2
#                            CC_global[node_1*9+3,node_1*9+3] += y1**2*CC_global[node_2*9+2,node_2*9+2]+zvec**2*CC_global[node_2*9+1,node_2*9+1]+2*y1*zvec*CC_global[node_2*9+1,node_2*9+2]
#                            CC_global[node_1*9+4,node_1*9+4] += x1**2*CC_global[node_2*9+2,node_2*9+2]+zvec**2*CC_global[node_2*9,node_2*9]+2*zvec*x1*CC_global[node_2*9,node_2*9+2]
#                            CC_global[node_1*9+5,node_1*9+5] += y1**2*CC_global[node_2*9,node_2*9]+x1**2*CC_global[node_2*9+1,node_2*9+1]-2*x1*y1*CC_global[node_2*9,node_2*9+1]
#                            CC_global[node_1*9+3,node_1*9+4] += -x1*y1*CC_global[node_2*9+2,node_2*9+2]-x1*zvec*CC_global[node_2*9+1,node_2*9+2]-y1*zvec*CC_global[node_2*9,node_2*9+2]-zvec**2*CC_global[node_2*9,node_2*9+1]
#                            CC_global[node_1*9+4,node_1*9+3] += -x1*y1*CC_global[node_2*9+2,node_2*9+2]-x1*zvec*CC_global[node_2*9+1,node_2*9+2]-y1*zvec*CC_global[node_2*9,node_2*9+2]-zvec**2*CC_global[node_2*9,node_2*9+1]
#                            CC_global[node_1*9+3,node_1*9+5] += x1*y1*CC_global[node_2*9+1,node_2*9+2]+x1*zvec*CC_global[node_2*9+1,node_2*9+1]-y1*zvec*CC_global[node_2*9,node_2*9+1]-y1**2*CC_global[node_2*9,node_2*9+2]
#                            CC_global[node_1*9+5,node_1*9+3] += x1*y1*CC_global[node_2*9+1,node_2*9+2]+x1*zvec*CC_global[node_2*9+1,node_2*9+1]-y1*zvec*CC_global[node_2*9,node_2*9+1]-y1**2*CC_global[node_2*9,node_2*9+2]
#                            CC_global[node_1*9+4,node_1*9+5] += x1*y1*CC_global[node_2*9,node_2*9+2]-x1*zvec*CC_global[node_2*9,node_2*9+1]+y1*zvec*CC_global[node_2*9,node_2*9]-x1**2*CC_global[node_2*9+1,node_2*9+2]
#                            CC_global[node_1*9+5,node_1*9+4] += x1*y1*CC_global[node_2*9,node_2*9+2]-x1*zvec*CC_global[node_2*9,node_2*9+1]+y1*zvec*CC_global[node_2*9,node_2*9]-x1**2*CC_global[node_2*9+1,node_2*9+2]                       
#                            CC_global[node_1*9+3,node_2*9+5] += -x2*y1*CC_global[node_2*9+1,node_2*9+2]-x2*zvec*CC_global[node_2*9+1,node_2*9+1]+y2*zvec*CC_global[node_2*9,node_2*9+1]+y1*y2*CC_global[node_2*9,node_2*9+2]
#                            CC_global[node_2*9+5,node_1*9+3] += -x2*y1*CC_global[node_2*9+1,node_2*9+2]-x2*zvec*CC_global[node_2*9+1,node_2*9+1]+y2*zvec*CC_global[node_2*9,node_2*9+1]+y1*y2*CC_global[node_2*9,node_2*9+2]
#                            CC_global[node_1*9+4,node_2*9+5] += -x1*y2*CC_global[node_2*9,node_2*9+2]+x2*zvec*CC_global[node_2*9,node_2*9+1]-y2*zvec*CC_global[node_2*9,node_2*9]-x1*x2*CC_global[node_2*9+1,node_2*9+2]
#                            CC_global[node_2*9+5,node_1*9+4] += -x1*y2*CC_global[node_2*9,node_2*9+2]+x2*zvec*CC_global[node_2*9,node_2*9+1]-y2*zvec*CC_global[node_2*9,node_2*9]-x1*x2*CC_global[node_2*9+1,node_2*9+2]
#                            CC_global[node_1*9+5,node_2*9+5] += x1*y2*CC_global[node_2*9,node_2*9+2]+x2*y1*CC_global[node_2*9,node_2*9+1]-y2*y1*CC_global[node_2*9,node_2*9]-x1*x2*CC_global[node_2*9+1,node_2*9+2]
#                            CC_global[node_2*9+5,node_1*9+5] += x1*y2*CC_global[node_2*9,node_2*9+2]+x2*y1*CC_global[node_2*9,node_2*9+1]-y2*y1*CC_global[node_2*9,node_2*9]-x1*x2*CC_global[node_2*9+1,node_2*9+2]
#                            CC_global[node_2*9+5,node_2*9+5] += y2**2*CC_global[node_2*9,node_2*9]+x2**2*CC_global[node_2*9+1,node_2*9+1]-2*x2*y2*CC_global[node_2*9,node_2*9+1]  
#                            for ii_fixed in np.arange(9):
#                                # Apply the restrictions in all the displacements except of the rotation around the axis
#                                if ii_fixed != 5 and ii_fixed != 8:  
#                                    CC_global[node_2*9+ii_fixed,:] *= 0
#                                    CC_global[:,node_2*9+ii_fixed] *= 0              
#                                    # The rows of the node 2 must be deleted and the relation between displacements is added
#                            # The stiffness matrix and the load vector are rotated back to the global frame
#                            for ii_point in np.arange(num_point):
#                                for jj_point in np.arange(num_point):
#                                    for ii_row in np.arange(3):
#                                        for jj_col in np.arange(3):
#                                            CC_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)] = np.matmul(np.transpose(rot_mat_axis),np.matmul(CC_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),jj_point*9+3*jj_col:jj_point*9+(3*jj_col+3)],rot_mat_axis))
                        for ii_row in np.arange(len(KK_global)):
                            if  ii_row  == 9*node_2+3+ii_1 and sum(KK_global[ii_row,:])!=0:
                                RR_global[ii_row,:] *= 0
                                RR_global[ii_row,ii_row] = 1
                                RR_global[ii_row,ii_row+ii_a] = rot_mat_axis[ii_1,ii_1]/rot_mat_axis[ii_1+ii_a,ii_1]  #KK_global[ii_row+ii_a,ii_row]/KK_global[ii_row,ii_row]
                                KK_global[ii_row,:] *= 0
                                KK_global[:,ii_row] *= 0
                                MM_global[ii_row,:] *= 0
                                MM_global[:,ii_row] *= 0
#                                if len(CC_global) > 0:
#                                    CC_global[ii_row,:] *= 0
#                                    CC_global[:,ii_row] *= 0
                            elif ii_row  == 9*node_2+3+ii_2 and sum(KK_global[ii_row,:])!=0:
                                RR_global[ii_row,:] *= 0
                                RR_global[ii_row,ii_row] = 1
                                RR_global[ii_row,ii_row+ii_b] = rot_mat_axis[ii_2,ii_2]/rot_mat_axis[ii_2+ii_b,ii_2] # KK_global[ii_row+ii_b,ii_row]/KK_global[ii_row,ii_row]
                                KK_global[ii_row,:] *= 0
                                KK_global[:,ii_row] *= 0
                                MM_global[ii_row,:] *= 0
                                MM_global[:,ii_row] *= 0
#                                if len(CC_global) > 0:
#                                    CC_global[ii_row,:] *= 0
#                                    CC_global[:,ii_row] *= 0

    for ii_row in np.arange(len(KK_global)):   
        ii_point = int(ii_row/9)
        p_dir  = np.argmax(abs(mesh_data.rr_p[0,ii_point,:]))
        if p_dir == 0:
            ii_1 = 1
            ii_2 = 2
            ii_a = -1
            ii_b = -2
        elif p_dir == 1:
            ii_1 = 0
            ii_2 = 2
            ii_a = 1
            ii_b = -1
        elif p_dir == 2:
            ii_1 = 0
            ii_2 = 1
            ii_a = 2
            ii_b = 1
        if ii_row % 9 == 6+ii_1 and sum(KK_global[ii_row,:])!=0:
            RR_global[ii_row,:] *= 0
            RR_global[ii_row,ii_row] = 1
            RR_global[ii_row,ii_row+ii_a] = rot_mat_p[ii_point,0,ii_1+ii_a,ii_1]/rot_mat_p[ii_point,0,ii_1,ii_1] #KK_global[ii_row+ii_a,ii_row]/KK_global[ii_row,ii_row]
            KK_global[ii_row,:] *= 0
            KK_global[:,ii_row] *= 0
            MM_global[ii_row,:] *= 0
            MM_global[:,ii_row] *= 0
#            if len(CC_global) > 0:
#                CC_global[ii_row,:] *= 0
#                CC_global[:,ii_row] *= 0
        elif ii_row % 9 ==6+ii_2 and sum(KK_global[ii_row,:])!=0:
            RR_global[ii_row,:] *= 0
            RR_global[ii_row,ii_row] = 1
            RR_global[ii_row,ii_row+ii_b] = rot_mat_p[ii_point,0,ii_2+ii_b,ii_2]/rot_mat_p[ii_point,0,ii_2,ii_2] #KK_global[ii_row+ii_b,ii_row]/KK_global[ii_row,ii_row]
            KK_global[ii_row,:] *= 0
            KK_global[:,ii_row] *= 0
            MM_global[ii_row,:] *= 0
            MM_global[:,ii_row] *= 0
#            if len(CC_global) > 0:
#                CC_global[ii_row,:] *= 0
#                CC_global[:,ii_row] *= 0
    if len(CC_global) > 0:
        return KK_global, MM_global, RR_global, case_setup, ref_axis, CC_global
    else:
        return KK_global, MM_global, RR_global, case_setup, ref_axis
#%%
def boundaryconditions(KK_global,MM_global,FF_global,qq_global,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p): 
    # Functions to determine the boundary conditions
    # -------------------------------------------------------------------------
    # KK_global         : stiffness matrix 
    # FF_global         : load vector 
    # qq_global   : displacement vector before applying the boundary conditions
    # mesh_mark         : markers of the mesh
    # case_setup        : information about the case setup
    # num_point         : number of points of the boundary conditions
    # sol_phys          : information of the solid physics
    # mesh_data         : data of the beam mesh
    # section           : information of the section
    # solver_vib_mod    : information about vibrational modes (to apply the loads as a function of the modal deformation)
    # -------------------------------------------------------------------------
    # ref_axis   : initialization of the reference axis
    ref_axis   = []
    RR_global  = np.zeros((9*num_point,9*num_point)) 
    fgrav      = np.zeros((9*num_point))
    FF_orig    = FF_global.copy()
    if case_setup.grav_fl == 1:
        cdg_x = 0
        cdg_y = 0 
        cdg_z = 0
        fgrav_sumx = 0
        fgrav_sumy = 0
        fgrav_sumz = 0
        for ii_point in np.arange(num_point):
            for jj_point in np.arange(3):
                fgrav[ii_point*9+jj_point] =  MM_global[ii_point*9+jj_point,ii_point*9+jj_point]
                FF_global[ii_point*9+jj_point] += case_setup.grav_g[jj_point] *fgrav[ii_point*9+jj_point]
                if jj_point == 0:
                    cdg_x += mesh_data.point[ii_point,0]*fgrav[ii_point*9+jj_point]
                    fgrav_sumx += fgrav[ii_point*9+jj_point]
                elif jj_point == 1:
                    cdg_y += mesh_data.point[ii_point,1]*fgrav[ii_point*9+jj_point]
                    fgrav_sumy += fgrav[ii_point*9+jj_point]
                elif jj_point == 2:
                    cdg_z += mesh_data.point[ii_point,2]*fgrav[ii_point*9+jj_point]
                    fgrav_sumz += fgrav[ii_point*9+jj_point]
        cdg_x /= fgrav_sumx
        cdg_y /= fgrav_sumy
        cdg_z /= fgrav_sumz
        mesh_data.cdg = [cdg_x,cdg_y,cdg_z]
    class disp_values:
        pass
    for mark in mesh_mark:
        for bound in case_setup.boundary:
            # For each marker in the mesh find if there is some boundary condition
            if mark.name == bound.marker:
                if bound.type == "BC_NODELOAD":
                    # If bc is a load in a node, add the value to the global force vector
                    for node in mark.node:
                        bound2val = bound.values.copy()
                        r_ref  = np.matmul(np.transpose(rot_mat_p[int(node),0,:,:]),np.concatenate(([section.ref_x[int(node)],section.ref_y[int(node)]],[0])))
                        bound2val[3:6] += np.cross(r_ref,bound.values[:3])
                        if node<num_point:
                            FF_global[int(9*node):int(9*node+9)] = bound2val
                        else:
                            FF_global[int(node):] = bound2val
                    FF_orig = FF_global.copy()
                elif bound.type == "BC_AERO" :
                    vel2 = bound.vinf
                    vel2dir = bound.vdir
                    if bound.vinf == "VINF_DAT":
                        bound.vinf = np.linalg.norm(case_setup.vinf)
                        bound.vdir = case_setup.vinf/np.linalg.norm(case_setup.vinf)
                    marktot = [mark]
                    if bound.f_flap == 1:
                        for markflap in mesh_mark:
                            if markflap.name == bound.flap:
                                if len(markflap.node) != len(mark.node):
                                    print('Error: core and flap have different number of nodes')
                                    sys.exit()
                                marktot.append(markflap)
                                break
                    # Add a reference axis to lifting surface
                    # do the boundary condition for the polar specified in the marker
                    # polar_bc_ind : number of independent polar boundary conditions
                    polar_bc_ind = 0
                    for polar_bc in case_setup.polar:
                        if polar_bc.id == bound.polar:
                            if polar_bc.lltnodesflag == 1:
                                for mark_llt in mesh_mark:
                                    if polar_bc.lltnodesmark == mark_llt.name:
                                        polar_bc.lltnodes = mark_llt.node
                                        polar_bc.lltnodes_ind = []
                                        polar_bc.flag_lltnodes_ind = 1
                            if polar_bc.cor3dflag == 1:
                                for mark_llt in mesh_mark:
                                    if polar_bc.cor3dmark == mark_llt.name:
                                        polar_bc.cor3dnodes = mark_llt.node
                                        polar_bc.cor3dnodes_ind = []
                                        polar_bc.flag_cor3dnodes_ind = 1
                            polar_bctot = [polar_bc]
                            polar_bctot_ind = [polar_bc_ind] 
                            break
                        polar_bc_ind += 1
                    # cdst_value0  : stationary drag coefficient of node 0
                    # cdst_value1  : stationary drag coefficient of node 1
                    # clst_value0  : stationary lift coefficient of node 0
                    # clst_value1  : stationary lift coefficient of node 1
                    # cmst_value0  : stationary pitch moment of node 0
                    # cmst_value1  : stationary pitch moment of node 1
                    # cd_ind       : induced drag
                    # lpos         : distance to the root of the aerodynamic surface
                    cdst_value0   = np.zeros((len(mark.node),))
                    cdst_value1   = np.zeros((len(mark.node),))
                    clst_value0   = np.zeros((len(mark.node),))
                    clst_value1   = np.zeros((len(mark.node),))
                    cmst_value0   = np.zeros((len(mark.node),))
                    cmst_value1   = np.zeros((len(mark.node),))
                    cddyn_value0  = np.zeros((len(mark.node),))
                    cddyn_value1  = np.zeros((len(mark.node),))
                    cldyn_value0  = np.zeros((len(mark.node),))
                    cldyn_value1  = np.zeros((len(mark.node),))
                    cmdyn_value0  = np.zeros((len(mark.node),))
                    cmdyn_value1  = np.zeros((len(mark.node),))
                    lpos          = np.zeros((len(mark.node),))
                    # Geometric characteristics of the cross-sections
                    # chord_vec       : chord of the airfoil in vectorial form
                    # b_chord         : half of the chord length
                    # vdir0           : velocity vector in the direction 0
                    # vdir1           : velocity vector in the direction 1
                    # aoa0_geom       : geometric angle of attack of the problem elem 0
                    # aoa1_geom       : geometric angle of attack of the problem elem 1
                    # LL_cum          : total cumulated longitude
                    # refaxis_0       : moment coefficient direction elem 0
                    # refaxis_0       : moment coefficient direction elem 1
                    # refaxis2_0      : lift coefficient direction elem 0
                    # refaxis2_0      : lift coefficient direction elem 1
                    # aoa_0lift       : null lift angle of attack
                    # lpos            : distance to the reference node
                    # iimark_node     : index of the node in the marker
                    b_chord       = np.zeros((len(mark.node),))
                    chord_vec     = np.zeros((len(mark.node),3))
                    non_acac_dist = np.zeros((len(mark.node),))
                    non_eaac_dist = np.zeros((len(mark.node),))
                    non_eamc_dist = np.zeros((len(mark.node),))
                    vdir0         = np.zeros((len(mark.node),3))
                    vdir1         = np.zeros((len(mark.node),3))
                    aoa0_geom     = np.zeros((len(mark.node),))
                    aoa1_geom     = np.zeros((len(mark.node),))
                    refaxis_0     = np.zeros((len(mark.node),3))
                    refaxis_1     = np.zeros((len(mark.node),3))
                    refaxis2_0    = np.zeros((len(mark.node),3))
                    refaxis2_1    = np.zeros((len(mark.node),3))
                    aoa_0lift     = np.zeros((len(mark.node),))
                    aoa_0liftflap = np.zeros((len(mark.node),))
                    cd_aoa0       = np.zeros((len(mark.node),))
                    if bound.f_flap == 1:
                        polar_bcflap_ind = 0
                        for polar_bcflap in case_setup.polar:
                            if polar_bcflap.id == bound.flappolar:
                                if polar_bcflap.lltnodesflag == 1:
                                    for mark_llt in mesh_mark:
                                        if polar_bcflap.lltnodesmark == mark_llt.name:
                                            polar_bcflap.lltnodes = mark_llt.node
                                            polar_bcflap.lltnodes_ind = []
                                            polar_bcflap.flag_lltnodes_ind = 1
                                if polar_bcflap.cor3dflag == 1:
                                    for mark_llt in mesh_mark:
                                        if polar_bcflap.cor3dmark == mark_llt.name:
                                            polar_bcflap.cor3dnodes = mark_llt.node
                                            polar_bcflap.cor3dnodes_ind = []
                                            polar_bcflap.flag_cor3dnodes_ind = 1
                                polar_bctot.append(polar_bcflap)
                                polar_bctot_ind.append(polar_bcflap_ind)
                                break
                            polar_bcflap_ind += 1
                    cdstflap_value0   = np.zeros((len(mark.node),))
                    cdstflap_value1   = np.zeros((len(mark.node),))
                    clstflap_value0   = np.zeros((len(mark.node),))
                    clstflap_value1   = np.zeros((len(mark.node),))
                    cmstflap_value0   = np.zeros((len(mark.node),))
                    cmstflap_value1   = np.zeros((len(mark.node),))
                    cddynflap_value0   = np.zeros((len(mark.node),))
                    cddynflap_value1   = np.zeros((len(mark.node),))
                    cldynflap_value0   = np.zeros((len(mark.node),))
                    cldynflap_value1   = np.zeros((len(mark.node),))
                    cmdynflap_value0   = np.zeros((len(mark.node),))
                    cmdynflap_value1   = np.zeros((len(mark.node),))
                    b_chordflap       = np.zeros((len(mark.node),))
                    chord_vecflap     = np.zeros((len(mark.node),3))
                    non_acac_distflap = np.zeros((len(mark.node),))
                    non_eaac_distflap = np.zeros((len(mark.node),))
                    non_eamc_distflap = np.zeros((len(mark.node),))
                    aoa0_geomflap    = np.zeros((len(mark.node),))
                    aoa1_geomflap    = np.zeros((len(mark.node),))
                    coreflapdist_adim = np.zeros((len(mark.node),))
                    deltaf_0lift     = np.zeros((len(mark.node),))
                    cd_aoa0flap       = np.zeros((len(mark.node),))
                    # Add the value of the longitude simulated in each beam element node
                    for iiaero_node in np.arange(len(mark.node)):
                        iimark_node = int(mark.node.flatten()[iiaero_node])
                        lpos[iiaero_node] = np.linalg.norm(mesh_data.point[iimark_node,:]-mesh_data.point[bound.refpoint])
                    LL_cum = np.max(lpos)
                    # If more than 1 polar
                    if polar_bc.pointfile[0] != -1:
                        pointref = np.where(mark.node==polar_bc.pointfile)[0].tolist()
                        lrefpol  = lpos[pointref]
                        polar_bc.iirefpol = np.argsort(lrefpol)
                        polar_bc.sortedlref = lrefpol[polar_bc.iirefpol]  
                    # For all the nodes in the surface
                    # cent_secx       : center of the airfoil x position
                    # cent_secy       : center of the airfoil y position
                    # acac_dist       : distance between the aerodynamic reference and the 0.25 chord distance
                    # acac_dist_sign  : sign of the distance between the aerodynamic reference and the 0.25 chord distance
                    # non_acac_dist   : nondimensional distance between the aerodynamic reference and the 0.25 chord distance with the moment sign
                    # eaac_dist       : distance between the aerodynamic reference and the elastic reference
                    # eaac_dist_sign  : sign of the distance between the aerodynamic reference and the elastic reference
                    # non_eaac_dist   : nondimensional distance between the aerodynamic reference and the elastic reference with the moment sign
                    # eamc_dist       : distance between the aerodynamic reference and the mean chord
                    # eamc_dist_sign  : sign of the distance between the aerodynamic reference and the mean chord
                    # non_eamc_dist   : nondimensional distance between the aerodynamic reference and the mean chord with the moment sign
                    for iiaero_node in np.arange(len(mark.node)):
                        iimark_node                = int(mark.node.flatten()[iiaero_node])
                        cent_secx                  = (section_globalCS.point_3d[iimark_node][section.te[iimark_node],0]+section_globalCS.point_3d[iimark_node][section.le[iimark_node],0])/2 #(section.points[iimark_node][0][section.te[iimark_node],0]+section.points[iimark_node][0][section.le[iimark_node],0])/2
                        cent_secy                  = (section_globalCS.point_3d[iimark_node][section.te[iimark_node],1]+section_globalCS.point_3d[iimark_node][section.le[iimark_node],1])/2 #(section.points[iimark_node][0][section.te[iimark_node],1]+section.points[iimark_node][0][section.le[iimark_node],1])/2
                        chord_vec[iiaero_node,:]   = section_globalCS.point_3d[iimark_node][section.te[iimark_node],:]-section_globalCS.point_3d[iimark_node][section.le[iimark_node],:] #section.points[iimark_node][0][section.te[iimark_node],:]-section.points[iimark_node][0][section.le[iimark_node],:]
                        b_chord[iiaero_node]       = np.linalg.norm(chord_vec[iiaero_node,:])/2
                        acac_dist                  = section_globalCS.aero_cent[iimark_node]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section_globalCS.point_3d[iimark_node][section.le[iimark_node],0]) #[section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]]-b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimark_node][0][section.le[iimark_node],:]
                        acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_acac_dist[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
                        eaac_dist                  = section_globalCS.aero_cent[iimark_node] #[section.ae_orig_x[iimark_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimark_node]-sol_phys.ysc[iiaero_node]]
                        eaac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],eaac_dist),np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_eaac_dist[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
                        eamc_dist                  = [-cent_secx,-cent_secy,0] #[sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
                        eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_eamc_dist[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chord[iiaero_node]   
                        if bound.f_flap == 1 and markflap.node.flatten()[iiaero_node] >= 0:
                            iimarkflap_node                = int(markflap.node.flatten()[iiaero_node])
                            if iimarkflap_node > 0:
                                cent_secx                  = (section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],0]+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],0])/2 #(section.points[iimarkflap_node][0][section.te[iimarkflap_node],0]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],0])/2
                                cent_secy                  = (section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],1]+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],1])/2 #(section.points[iimarkflap_node][0][section.te[iimarkflap_node],1]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],1])/2
                                chord_vecflap[iiaero_node,:]   = section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],:]-section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],:] #section.points[iimarkflap_node][0][section.te[iimarkflap_node],:]-section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
                                b_chordflap[iiaero_node]       = np.linalg.norm(chord_vecflap[iiaero_node,:])/2
                                acac_dist                  = section_globalCS.aero_cent[iimarkflap_node]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],0]) #[section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]]-b_chordflap[iiaero_node]/2*chord_vecflap[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
                                acac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimarkflap_node,0,:,:],acac_dist),np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_acac_distflap[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
                                eaac_dist                  = section_globalCS.aero_cent[iimark_node] #[section.ae_orig_x[iimarkflap_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimarkflap_node]-sol_phys.ysc[iiaero_node]]
                                eaac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimarkflap_node,0,:,:],eaac_dist),np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_eaac_distflap[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
                                eamc_dist                  = [-cent_secx,-cent_secy,0] # [sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
                                eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_eamc_distflap[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chordflap[iiaero_node] 
                                coreflapdist  = mesh_data.point[iimark_node,:]-mesh_data.point[iimarkflap_node,:]
                                coreflapdist_adim[iiaero_node] = np.sign(np.dot(coreflapdist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))/(b_chord[iiaero_node]+b_chordflap[iiaero_node])  
                    # If a rotative aerodynamic model is selected the nondimensional radius is calculated
                    # flag_start    : flag to start the bisection method
                    # flag_start1   : flag to start the bisection method
                    # flag_start2   : flag to start the bisection method
                    # r_vinduced    : nondimensional radius
                    # cltot2        : total value of the lift coefficient in the previous iteration
                    # cttot2        : total value of the thrust coefficient in the previous iteration
                    # v_induced     : induced velocity 
                    # Gamma_induced : induced circulation
                    # tol_vind      : tolerance of the induced velocity error
                    # aux_vind_e    : index to count the number of iterations in the induced velocity calculation loop
                    # lpos_cir      : distance to the root of the aerodynamic surface in circular coordinates
                    # lpc_in        : position of the section relative to the total length
                    # mat_pllt      : lifting line theory matrix
                    # vec_pllt      : lifting line theory vector
                    if polar_bc.eff3d == 'LLT':
                        class eff_3d:
                            Gamma_induced = np.zeros((len(mark.node),))
                            cltot2 = 1e3
                            lpos_cir  = np.zeros((len(mark.node),))
                            sign_vi   = -1
                        v_induced = np.zeros((len(mark.node),3))
                        for iiaero_node in np.arange(len(mark.node)):
                            lpc_in = lpos[iiaero_node]/LL_cum
                            if lpc_in>1:
                                lpc_in = 1
                            eff_3d.lpos_cir[iiaero_node] = np.arccos(lpc_in) 
                    elif polar_bc.eff3d == 'BEM':
                        class eff_3d:
                            flag_start      = np.zeros((len(mark.node),))
                            flag_start1     = np.zeros((len(mark.node),))
                            flag_start2     = np.zeros((len(mark.node),))
                            error1          = np.ones((len(mark.node),))
                            error2          = np.ones((len(mark.node),))
                            sign_vi         = -np.sign(np.dot(bound.vdir,bound.refrot))
                            r_vinduced      = lpos/LL_cum
                            v_induced2      = np.zeros((len(mark.node),3))
                            masa            = np.zeros((len(mark.node),)) #sol_phys.m11
                            frelax_vt       = 1
                            errorout1       = 1
                            errorout2       = 2
                        for ii_massind in np.arange(len(mark.node)):
                            ii_mass = int(mark.node[ii_massind])
                            eff_3d.masa[ii_massind] = MM_global[ii_mass,ii_mass]
                        v_induced         = 0*bound.vrot*bound.radius*np.ones((len(mark.node),3))
                        v_induced[:,0]   *= eff_3d.sign_vi*bound.vdir[0]/np.linalg.norm(bound.vdir)
                        v_induced[:,1]   *= eff_3d.sign_vi*bound.vdir[1]/np.linalg.norm(bound.vdir)
                        v_induced[:,2]   *= eff_3d.sign_vi*bound.vdir[2]/np.linalg.norm(bound.vdir)
                        vtan_induced = np.zeros((len(mark.node),3))
                        eff_3d.v_induced_tan0     = np.zeros((len(mark.node),3))
                        eff_3d.v_induced_tan1     = np.zeros((len(mark.node),3))
                        eff_3d.v_induced1 = v_induced.copy()
                    else: 
                        v_induced = np.zeros((len(mark.node),3))                         
                        class eff_3d:
                            sign_vi         = 1
                    try:
                        tol_vind = polar_bc.vind_tol
                    except:
                        tol_vind = 1e-5
                    try:
                        maxit_vind = polar_bc.vind_maxit
                    except:
                        maxit_vind = 1e3
                    eff_3d.aux_vind_e_out = 0 
                    # initialize the error
                    error_out = tol_vind*10
                    flag_conv = 0
                    flag_convout = 0
                    # While error is  higher than the required tolerance
                    while abs(error_out)>tol_vind or flag_convout == 0:
                        if abs(error_out)<tol_vind:
                            if polar_bc.eff3d == 'BEM':
                                vtan_induced_mod = vtan_induced.copy()
                                vtan_ind_int = vtan_induced[polar_bc.lltnodes_ind,:]
                                l_ind_int = lpos[polar_bc.lltnodes_ind]
                                f_vxind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,0],'cubic', fill_value='extrapolate')
                                f_vyind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,1],'cubic', fill_value='extrapolate')
                                f_vzind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,2],'cubic', fill_value='extrapolate') 
                                for auxaeronode in np.arange(len(mark.node)):
                                    vtan_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
                                vtan_induced = vtan_induced_mod.copy()
                                break
                            flag_convout = 1
                        eff_3d.aux_vind_e_out += 1
                        # if the loop cannot find the solution in a determined number of iterations, stop the loop
                        if eff_3d.aux_vind_e_out > maxit_vind: 
                            print('Induced tangential velocity: Not converged - '+str(error_out))
                            break
                        # error  : error of the lift coefficient in each iteration
                        error  = tol_vind*10
                        eff_3d.error = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.error1ct = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.error2ct = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.flag_start = np.zeros((len(mark.node),))
                        eff_3d.flag_start1 = np.zeros((len(mark.node),))
                        eff_3d.flag_start2 = np.zeros((len(mark.node),))
                        eff_3d.aux_vind_e = 1 
                        while abs(error)>tol_vind or flag_conv == 0:
                            if abs(error)<tol_vind:
                                if polar_bc.eff3d == 'BEM':
                                    v_induced_mod = v_induced.copy()
                                    v_ind_int = v_induced[polar_bc.lltnodes_ind,:]
                                    l_ind_int = lpos[polar_bc.lltnodes_ind]
                                    try:
                                        f_vxind = interpolate.interp1d(l_ind_int,v_ind_int[:,0],'cubic', fill_value='extrapolate')
                                        f_vyind = interpolate.interp1d(l_ind_int,v_ind_int[:,1],'cubic', fill_value='extrapolate')
                                        f_vzind = interpolate.interp1d(l_ind_int,v_ind_int[:,2],'cubic', fill_value='extrapolate')
                                    except:
                                        f_vxind = interpolate.interp1d(l_ind_int,v_ind_int[:,0], fill_value='extrapolate')
                                        f_vyind = interpolate.interp1d(l_ind_int,v_ind_int[:,1], fill_value='extrapolate')
                                        f_vzind = interpolate.interp1d(l_ind_int,v_ind_int[:,2], fill_value='extrapolate') 
                                    for auxaeronode in np.arange(len(mark.node)):
                                        v_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
                                    v_induced = v_induced_mod.copy()
                                    break
                                flag_conv = 1
                            # if the loop cannot find the solution in a determined number of iterations, stop the loop
                            if eff_3d.aux_vind_e > maxit_vind: 
                                print('Induced velocity: Not converged - '+str(error))
                                break
                            eff_3d.aux_vind_e += 1
                            # Define variables for the rotative aerodynamics
                            if polar_bc.eff3d == 'LLT':            
                                eff_3d.mat_pllt  = np.zeros((2*len(mark.node),len(mark.node)))
                                eff_3d.vec_pllt  = np.zeros((2*len(mark.node),))
                            elif polar_bc.eff3d == 'BEM':
                                # r_vinduced     : nondimensional distance of the blade radius
                                # vi_vinduced    : induced velocity
                                # phi_vinduced   : induced velocity on the blade
                                # f_vinduced     : 3D effect factor of the blade
                                # F_vinduced     : 3D effect factor of the blade
                                # dct_dr_induced : variation of thrust with radial distance
                                eff_3d.phi_vinduced0       = np.zeros((len(mark.node),))
                                eff_3d.f_vinduced0         = np.zeros((len(mark.node),))
                                eff_3d.F_vinduced0         = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced0 = np.zeros((len(mark.node),))
                                eff_3d.phi_vinduced1       = np.zeros((len(mark.node),))
                                eff_3d.f_vinduced1         = np.zeros((len(mark.node),))
                                eff_3d.F_vinduced1         = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced1 = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced  = np.zeros((len(mark.node),))
                            # aoa_vec0    : angle of attack of node 0
                            # aoa_vec1    : angle of attack of node 1
                            # aoap_vec0   : angle of attack derivative of node 0
                            # aoap_vec1   : angle of attack derivative of node 1
                            # aoa_val     : value of the angle of attack
                            # aoap_val    : value of the angle of attack derivative
                            # aoapp_val   : value of the angle of attack second derivative
                            # pos_vec0    : position vector of node 0
                            # pos_vec1    : position vector of node 1
                            aoa_vec0    = np.zeros((len(mark.node),))
                            aoa_vec1    = np.zeros((len(mark.node),))
                            aoa_val     = np.zeros((len(mark.node),))
                            pos_vec0    = np.zeros((len(mark.node),))
                            pos_vec1    = np.zeros((len(mark.node),))
                            aoa_vec0_st = np.zeros((len(mark.node),))
                            aoa_vec1_st = np.zeros((len(mark.node),))
                            Vinf0       = np.zeros((len(mark.node),))
                            Vinf1       = np.zeros((len(mark.node),))
                            reynolds0   = np.zeros((len(mark.node),))
                            reynolds1   = np.zeros((len(mark.node),))
                            deltaflap_vec0    = np.zeros((len(mark.node),))
                            deltaflap_vec1    = np.zeros((len(mark.node),))
                            # Calculate the reference axis
                            refaxis_0, refaxis_1, refaxis2_0, refaxis2_1, vdir0, vdir1, aoa0_geom, aoa1_geom, eff_3d = ref_axis_aero(bound,mark,polar_bc,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,v_induced,vdir0,vdir1,aoa0_geom,aoa1_geom,rr_vec_p,rot_matT_p,chord_vec,eff_3d,0)
                            if bound.f_flap == 1:
                                # Calculate the reference axis
                                aoa0_geomflap, aoa1_geomflap = ref_axis_aero(bound,markflap,polar_bcflap,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,v_induced,vdir0,vdir1,aoa0_geomflap,aoa1_geomflap,rr_vec_p,rot_matT_p,chord_vecflap,eff_3d,1)
                            # for each node in the surface
                            for iiaero_node in np.arange(len(mark.node)):
                                iimark_node = int(mark.node.flatten()[iiaero_node])
                                # Calculate the velocity of the free stream for the airfoil for the different aerodynamic models
                                # refvind0       : reference induced velocity direction in element 0
                                # refvind1       : reference induced velocity direction in element 1
                                # vi_vinduced0   : induced velocity scalar in element 0
                                # vi_vinduced1   : induced velocity scalar in element 1
                                # Vinf0          : free stream velocity in element 0
                                # Vinf1          : free stream velocity in element 1
                                # Vinf_vindaoa   : velocity to calculate the angle of the induced angle of attack
                                if polar_bc.eff3d == 'LLT':
                                    refvind0           = -refaxis2_0[iiaero_node,:]
                                    refvind1           = -refaxis2_1[iiaero_node,:]
                                    vi_vinduced0       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(refvind0))
                                    vi_vinduced1       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(refvind1))
                                    Vinf0[iiaero_node] = np.sqrt(bound.vinf**2+vi_vinduced0**2)
                                    Vinf1[iiaero_node] = np.sqrt(bound.vinf**2+vi_vinduced1**2)
                                    Vinf_vindaoa       = bound.vinf
                                elif polar_bc.eff3d == 'BEM':
                                    refvind0           = eff_3d.sign_vi*bound.vdir
                                    refvind1           = eff_3d.sign_vi*bound.vdir
                                    vi_vinduced0       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(bound.vdir))
                                    vi_vinduced1       = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_induced[iiaero_node,:]),np.array(bound.vdir))
                                    vt_vinduced0       = np.dot(eff_3d.v_induced_tan0[iiaero_node,:],eff_3d.vec_vrot0[iiaero_node,:])
                                    vt_vinduced1       = np.dot(eff_3d.v_induced_tan1[iiaero_node,:],eff_3d.vec_vrot1[iiaero_node,:])
                                    Vinf0[iiaero_node] = np.sqrt((bound.vinf+vi_vinduced0)**2+(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius+vt_vinduced0)**2)+1e-10
                                    Vinf1[iiaero_node] = np.sqrt((bound.vinf+vi_vinduced1)**2+(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius+vt_vinduced1)**2)+1e-10
                                    v_induced_tan      = (eff_3d.v_induced_tan0[iiaero_node,:]*LL_vec_p[0,iimark_node]+eff_3d.v_induced_tan1[iiaero_node,:]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
                                    vtan_induced[iiaero_node] = v_induced_tan
                                    Vinf_vindaoa       = np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+v_induced_tan)+1e-10
                                else:
                                    refvind0           = -refaxis2_0[iiaero_node,:]
                                    refvind1           = -refaxis2_1[iiaero_node,:]
                                    Vinf0[iiaero_node] = bound.vinf 
                                    Vinf1[iiaero_node] = bound.vinf 
                                    Vinf_vindaoa       = bound.vinf
                                # For lifting line theory set the angle of attack at the tip as zero, calculate in the rest of points
                                if (polar_bc.eff3d == 'LLT' ) and lpos[iiaero_node] > 0.999*LL_cum:  # or polar_bc.eff3d == 'BEM'
                                    aoa_vec0[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoa_vec1[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoa_val[iiaero_node]     = aoa_0lift[iiaero_node]
                                    deltaflap_vec0[iiaero_node]    = deltaf_0lift[iiaero_node]
                                    deltaflap_vec1[iiaero_node]    = deltaf_0lift[iiaero_node]
                                    v_induced[iiaero_node,:] = np.matmul(rot_mat_p[iimark_node,1,:,:],Vinf0[iiaero_node]*np.tan(aoa_val[iiaero_node]-aoa0_geom[iiaero_node])*np.array(refvind1))
                                # In the rest of the points
                                else:
                                    # if needed include rotational effects
                                    if polar_bc.eff3d == 'BEM':
                                        if eff_3d.r_vinduced[iiaero_node] == 0:
                                            eff_3d.phi_vinduced0[iiaero_node] = np.arctan((bound.vinf)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                            eff_3d.phi_vinduced1[iiaero_node] = np.arctan((bound.vinf)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                        else:
                                            eff_3d.phi_vinduced0[iiaero_node] = np.arctan((bound.vinf+vi_vinduced0)/np.max([np.linalg.norm(bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node]*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:]),1e-10])) 
                                            eff_3d.phi_vinduced1[iiaero_node] = np.arctan((bound.vinf+vi_vinduced1)/np.max([np.linalg.norm(bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node]*eff_3d.vec_vrot1[iiaero_node,:]+eff_3d.v_induced_tan1[iiaero_node,:]),1e-10])) 
                                        eff_3d.f_vinduced0[iiaero_node]   = bound.Nb/2*((1-(eff_3d.r_vinduced[iiaero_node]))/np.max([eff_3d.r_vinduced[iiaero_node]*eff_3d.phi_vinduced0[iiaero_node],1e-10]))
                                        eff_3d.f_vinduced1[iiaero_node]   = bound.Nb/2*((1-(eff_3d.r_vinduced[iiaero_node]))/np.max([eff_3d.r_vinduced[iiaero_node]*eff_3d.phi_vinduced1[iiaero_node],1e-10]))
                                        eff_3d.F_vinduced0[iiaero_node]   = 2/np.pi*np.arccos(np.exp(-eff_3d.f_vinduced0[iiaero_node]))
                                        eff_3d.F_vinduced1[iiaero_node]   = 2/np.pi*np.arccos(np.exp(-eff_3d.f_vinduced1[iiaero_node]))
                                        v_vert                            = eff_3d.sign_vi*(bound.vinf*bound.vdir+v_induced[iiaero_node,:])
                                    else:
                                        v_vert = v_induced[iiaero_node,:] #eff_3d.sign_vi*
                                    aoa_vec0[iiaero_node] = (aoa0_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_vert),np.array(refvind0))/Vinf_vindaoa)
                                    aoa_vec1[iiaero_node] = (aoa1_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_vert),np.array(refvind1))/Vinf_vindaoa)
                                    aoa_val[iiaero_node]  = (aoa_vec0[iiaero_node]*LL_vec_p[0,iiaero_node]+aoa_vec1[iiaero_node]*LL_vec_p[1,iiaero_node])/(LL_vec_p[0,iiaero_node]+LL_vec_p[1,iiaero_node])
                                    if bound.f_flap == 1  and markflap.node.flatten()[iiaero_node] >= 0:
                                        deltaflap_vec0[iiaero_node]    = (aoa0_geomflap[iiaero_node]-aoa0_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_vert),np.array(refvind0))/Vinf_vindaoa)
                                        deltaflap_vec1[iiaero_node]    = (aoa1_geomflap[iiaero_node]-aoa1_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_vert),np.array(refvind1))/Vinf_vindaoa)
                                    else:
                                        deltaflap_vec0[iiaero_node] = 0
                                        deltaflap_vec1[iiaero_node] = 0
                                aoa_vec0_st[iiaero_node]  = aoa0_geom[iiaero_node]
                                aoa_vec1_st[iiaero_node]  = aoa1_geom[iiaero_node]
                                pos_vec0[iiaero_node] = 0
                                pos_vec1[iiaero_node] = 0
                                reynolds0[iiaero_node] = Vinf0[iiaero_node]*2*b_chord[iiaero_node]*bound.rho/bound.mu
                                reynolds1[iiaero_node] = Vinf1[iiaero_node]*2*b_chord[iiaero_node]*bound.rho/bound.mu
#                                    print([reynolds0,reynolds1])
                            case_setup, polar_bc_ind,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,eff_3d,error,cl_alpha,cl_alphaflap,error2,v_induced = steady_aero(case_setup,bound,polar_bctot,mesh_data,marktot,polar_bctot_ind,aoa_vec0,aoa_vec1,aoa_vec0_st,aoa_vec1_st,pos_vec0,pos_vec1,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,aoa_0lift,aoa_0liftflap,v_induced,refaxis2_0,refaxis2_1,Vinf0,Vinf1,LL_vec_p,lpos,LL_cum,eff_3d,tol_vind,b_chord,non_acac_dist,b_chordflap,non_acac_distflap,coreflapdist_adim,aoa_vec0,aoa_vec1,cd_aoa0,cd_aoa0flap,aoa0_geom,aoa1_geom,reynolds0,reynolds1,deltaflap_vec0,deltaflap_vec1,bound.f_flap)
                        if polar_bc.eff3d == 'BEM':
                            eff_3d,error_out,vtan_induced = converge_vtan(bound,mark,polar_bc,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,v_induced,LL_vec_p,eff_3d,b_chord,vtan_induced)
                        else:
                            error_out = tol_vind/10
                    FF_global,cl_value0,cl_value1,cm_value0,cm_value1,disp_values = total_forces(clst_value0,clst_value1,cmst_value0,cmst_value1,cdst_value0,cdst_value1,cldyn_value0,cldyn_value1,cmdyn_value0,cmdyn_value1,cddyn_value0,cddyn_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdstflap_value0,cdstflap_value1,cldynflap_value0,cldynflap_value1,cmdynflap_value0,cmdynflap_value1,cddynflap_value0,cddynflap_value1,LL_vec_p,rot_mat_p,b_chord,b_chordflap,lpos,Vinf0,Vinf1,vdir0,vdir1,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,FF_global,bound,section,polar_bc,marktot,disp_values,num_point,0,0,1,[],eff_3d,bound.f_flap)
                    FF_orig = FF_global.copy()
                    # save values of the aerodynamic parameters
                    disp_values.aoa_val   = aoa_vec0
                    disp_values.clst_value0 = clst_value0
                    disp_values.clst_value1 = clst_value1
                    disp_values.cmst_value0 = cmst_value0
                    disp_values.cmst_value1 = cmst_value1
                    disp_values.cdst_value0 = cdst_value0
                    disp_values.cdst_value1 = cdst_value1
                    disp_values.cldyn_value0 = cldyn_value0
                    disp_values.cldyn_value1 = cldyn_value1
                    disp_values.cmdyn_value0 = cmdyn_value0
                    disp_values.cmdyn_value1 = cmdyn_value1
                    disp_values.cddyn_value0 = cddyn_value0
                    disp_values.cddyn_value1 = cddyn_value1
                    disp_values.aoa_val   = aoa_val
                    bound.vinf = vel2
                    bound.vdir = vel2dir
                elif bound.type == "BC_DISP":
                    # If bc is a displacement in a node, add the value to the global displacement vector
                    for node in mark.node:
                        if node<num_point:
                            if all(np.isnan(qq_global[int(9*node):int(9*node+9)])):
                                qq_global[int(9*node):int(9*node+9)] = bound.values
                            else:
                                qq_global[int(9*node):int(9*node+9)] += bound.values
                        else:
                            if all(np.isnan(qq_global[int(9*node):int(9*node+9)])):
                                qq_global[int(9*node):] = bound.values
                            else:
                                qq_global[int(9*node):] += bound.values
                elif bound.type == "BC_FUNC":
                    # If the bc is a function of the vibration modes
                    try:
                        # mode_bc      : vibration mode used for the load
                        # vibfree      : information of the free vibration analysis
                        # sec_coor_vib : information of the section points used in the vibration analysis
                        # func_bc      : function used for the bc
                        mode_bc               = int(bound.funcmode)
                        vibfree, sec_coor_vib = solver_vib_mod(case_setup,sol_phys,mesh_data,section)
                        func_bc               = np.zeros((len(mark.node),9))
                        # for every node in the marker determine the function
                        for node in mark.node:
                            node = int(node)
                            # The function is calculated as the projection in the bc values_load 
                            # vector of the modal shape
                            func_bc[node,:] = np.dot([vibfree.u[node,mode_bc],vibfree.v[node,mode_bc],vibfree.w[node,mode_bc],\
                                       vibfree.phi[node,mode_bc],vibfree.psi[node,mode_bc],vibfree.theta[node,mode_bc],\
                                       vibfree.phi_d[node,mode_bc],vibfree.psi_d[node,mode_bc],vibfree.theta_d[node,mode_bc]],bound.values_load)
                        # For every mesh marker if the marker name is the normalization function 
                        # name divide the value by the value in the normalization node and multiplied 
                        # by the value specified in the setup
                        for mark2 in mesh_mark:
                            if mark2.name == bound.funcnorm:
                                for aux_mark2 in np.arange(9):
                                    if func_bc[int(mark2.node[0]),aux_mark2] != 0:
                                        func_bc /= func_bc[int(mark2.node[0]),aux_mark2]*np.array(bound.values)
                    except:
                        pass
                    # The values calculated preivously are updated to the forces vector         
                    for elem in mesh_data.elem:
                        if len(np.where(mark.node==elem[1])[0])>0:
                            FF_global[int(9*elem[1]):int(9*elem[1]+9)]+=func_bc[int(elem[1])]*np.linalg.norm(mesh_data.point[int(elem[1]),:]-mesh_data.point[int(elem[2]),:])/2
                        if len(np.where(mark.node==elem[2])[0])>0:
                            FF_global[int(9*elem[2]):int(9*elem[2]+9)]+=func_bc[int(elem[2])]*np.linalg.norm(mesh_data.point[int(elem[1]),:]-mesh_data.point[int(elem[2]),:])/2
                elif bound.type == "BC_JOINT":
                    # If the bc is a joint between nodes
                    # node_1 : node 1 of the joint
                    # node_2 : node 2 of the joint
                    node_1 = int(mark.node[0])
                    node_2 = int(mark.node[1])
                    # xvec : distance between nodes in axis x
                    # yvec : distance between nodes in axis y
                    # zvec : distance between nodes in axis z
                    xvec = mesh_data.point[node_2,0]-mesh_data.point[node_1,0]
                    yvec = mesh_data.point[node_2,1]-mesh_data.point[node_1,1]
                    zvec = mesh_data.point[node_2,2]-mesh_data.point[node_1,2]
                    if bound.joint_type == "FIXED":
                        # If the boundary condition is a rigid solid connection between bodies
                        for ii_fixed in np.arange(9):
                            # for the 9 rows of the node 1
                            # add the values of node 2 in stiffness and loads
                            FF_global[node_1*9+ii_fixed] += FF_global[node_2*9+ii_fixed]
                            # The moment added by the distance between reference points
                            if ii_fixed == 0:
                                # moment added by force in x axis
                                FF_global[node_1*9+4] += zvec*FF_global[node_2*9]
                                FF_global[node_1*9+5] += -yvec*FF_global[node_2*9]
                            elif ii_fixed == 1:
                                # moment added by force in y axis
                                FF_global[node_1*9+3] += -zvec*FF_global[node_2*9+1]
                                FF_global[node_1*9+5] += xvec*FF_global[node_2*9+1]
                            elif ii_fixed == 2:
                                # moment added by force in z axis
                                FF_global[node_1*9+3] += yvec*FF_global[node_2*9+2]
                                FF_global[node_1*9+4] += -xvec*FF_global[node_2*9+2]
                        # The rows of the node 2 must be deleted and the relation between displacements is added
                        FF_global[node_2*9:node_2*9+9] = np.zeros((9,))
                    elif bound.joint_type == "ROTATE_AXIS":
                        # if joint is a rotation joint
                        # a_axis : axis of rotation of the beam
                        # r_axis : unitary vector of the distance between nodes
                        # r_0_2  : distance from point of rotation to section 2
                        # r_1_0  : distance from section 1 to point of rotation
                        # n_axis : normal axis from rotation axis and nodes relative position
                        # x1     : distance between node 1 and joint in axis x
                        # y1     : distance between node 1 and joint in axis y
                        # x2     : distance between joint and node 2 in axis x
                        # y2     : distance between joint and node 2 in axis y
                        a_axis = np.array(bound.joint_axis)/np.linalg.norm(np.array(bound.joint_axis))
                        if np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])>0:
                            r_axis = (mesh_data.point[node_2]-mesh_data.point[node_1])/np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])
                        else:
                            if a_axis[0] == 0:
                                r_axis = np.array([1,0,0])
                            elif a_axis[1] == 0:
                                r_axis = np.array([0,1,0])
                            elif a_axis[2] == 0:
                                r_axis = np.array([0,0,1])
                            else:
                                r_axis = np.array([a_axis[1],-a_axis[0],a_axis[2]])/np.linalg.norm(np.array([a_axis[1],-a_axis[0],a_axis[2]]))
                        r_0_2 = mesh_data.point[node_2]-bound.point_axis
                        r_0_1 = mesh_data.point[node_1]-bound.point_axis
                        x1 = r_0_1[0]
                        y1 = r_0_1[1]
                        x2 = r_0_2[0]
                        y2 = r_0_2[1]
                        n_axis = np.cross(a_axis,r_axis)
                        # rot_mat_axis : Rotation matrix of the degree of freedom. From rotation axis to global 
                        rot_mat_axis = np.array([[np.dot(r_axis,[1,0,0]),np.dot(n_axis,[1,0,0]),np.dot(a_axis,[1,0,0])],
                                                  [np.dot(r_axis,[0,1,0]),np.dot(n_axis,[0,1,0]),np.dot(a_axis,[0,1,0])],
                                                  [np.dot(r_axis,[0,0,1]),np.dot(n_axis,[0,0,1]),np.dot(a_axis,[0,0,1])]])
                        # stiffness matrix and load vector in the nodes is rotated to the joint axis
                        for ii_point in np.arange(num_point):
                                for ii_row in np.arange(3):
                                    FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3)] = np.matmul(rot_mat_axis,FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3)])
                        # For all the rows of the points in the stiffness matrix and load vector
                        for ii_fixed in np.arange(9):
                            # Apply the restrictions in all the displacements except of the rotation around the axis
                            if ii_fixed != 5 and ii_fixed != 8:                                
                                FF_global[node_1*9+ii_fixed]   += FF_global[node_2*9+ii_fixed]
                                # The moment added by the distance between reference points
                                if ii_fixed == 0:
                                    # moment added by force in x axis
                                    FF_global[node_1*9+4]   += zvec*FF_global[node_2*9]
                                    FF_global[node_1*9+5]   += -y1*FF_global[node_2*9]
                                elif ii_fixed == 1:
                                    # moment added by force in y axis
                                    FF_global[node_1*9+3] += -zvec*FF_global[node_2*9+1]
                                    FF_global[node_1*9+5] += -x1*FF_global[node_2*9+1]
                                elif ii_fixed == 2:
                                    # moment added by force in z axis
                                    FF_global[node_1*9+3]   += yvec*FF_global[node_2*9+2]
                                    FF_global[node_1*9+4]   += -xvec*FF_global[node_2*9+2] 
                                FF_global[node_2*9+ii_fixed]    = 0          
                                # The rows of the node 2 must be deleted and the relation between displacements is added
                        # The stiffness matrix and the load vector are rotated back to the global frame
                        for ii_point in np.arange(num_point):
                                for ii_row in np.arange(3):
                                    FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3)] = np.matmul(np.transpose(rot_mat_axis),FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3)])
    KK_global, MM_global, RR_global, case_setup, ref_axis = init_boundaryconditions_tran(KK_global,MM_global, [],mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)
    return  KK_global, qq_global, FF_global, FF_orig, RR_global, disp_values, mesh_data
#%%  
def boundaryconditions_tran(KK_global,MM_global,CC_global,FF_global,qq_global,qq_der_global,qq_derder_global,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,time_val,tmin_ann,ttran_ann,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p,disp_values,ii_time,int_per,time_vec): 
    time_valvec = time_vec #[:ii_time+1]
    # Functions to determine the boundary conditions
    # -------------------------------------------------------------------------
    # KK_global         : stiffness matrix 
    # MM_global         : mass matrix
    # CC_global         : damping matrix
    # FF_global         : load vector 
    # qq_global         : displacement matrix
    # qq_der_global     : velocity matrix
    # qq_derder_global  : acceleration matrix
    # mesh_mark         : markers of the mesh
    # case_setup        : information about the case setup
    # num_point         : number of points of the boundary conditions
    # sol_phys          : information of the solid physics
    # mesh_data         : data of the beam mesh
    # section           : information of the section
    # solver_vib_mod    : information about vibrational modes (to apply the loads as a function of the modal deformation)
    # time_val          : value of time
    # tmin_ann          : minimum time to apply the Artificial Neural Network
    # ttran_ann         : transition time of the Artificial Neural Network
    # rot_mat_p         : rotation matrix for each node and element
    # rot_matT_p        : transposed rotation matrix for each node and element
    # LL_vec_p          : beam longitude associated with each beam node
    # rr_vec_p          : distance vector associated with each beam element
    # disp_values       : class containing the displacement values of each iteration
    # ii_time           : time index in the solution matrix
    # int_per           : iterations in one period
    # -------------------------------------------------------------------------
    # ref_axis   : initialization of the reference axis
    vical = 1
    ref_axis   = []
    flag_aerodyn = 0
    # polar_bc_ind : number of independent polar boundary conditions
    polar_bc_ind = 0
    fgrav      = np.zeros((9*num_point))
    FF_orig    = FF_global.copy()
    if case_setup.problem_type == "AEL_DYN":
        aoa_totvec0 = np.zeros((mesh_data.num_point,))
        aoa_totvec1 = np.zeros((mesh_data.num_point,))
        ang_totvec0 = np.zeros((mesh_data.num_point,))
        ang_totvec1 = np.zeros((mesh_data.num_point,))
        angp_totvec0 = np.zeros((mesh_data.num_point,))
        angp_totvec1 = np.zeros((mesh_data.num_point,))
        angpp_totvec0 = np.zeros((mesh_data.num_point,))
        angpp_totvec1 = np.zeros((mesh_data.num_point,))
        dis_totvec0 = np.zeros((mesh_data.num_point,))
        dis_totvec1 = np.zeros((mesh_data.num_point,))
        disp_totvec0 = np.zeros((mesh_data.num_point,))
        disp_totvec1 = np.zeros((mesh_data.num_point,))
        dispp_totvec0 = np.zeros((mesh_data.num_point,))
        dispp_totvec1 = np.zeros((mesh_data.num_point,))
        disppp_totvec0 = np.zeros((mesh_data.num_point,))
        disppp_totvec1 = np.zeros((mesh_data.num_point,))
    if case_setup.grav_fl == 1:
        cdg_x = 0
        cdg_y = 0 
        cdg_z = 0
        fgrav_sumx = 0
        fgrav_sumy = 0
        fgrav_sumz = 0
        for ii_point in np.arange(num_point):
            for jj_point in np.arange(3):
                fgrav[ii_point*9+jj_point] =  MM_global[ii_point*9+jj_point,ii_point*9+jj_point]
                FF_global[ii_point*9+jj_point,ii_time] += case_setup.grav_g[jj_point] *fgrav[ii_point*9+jj_point]
                if jj_point == 0:
                    cdg_x += mesh_data.point[ii_point,0]*fgrav[ii_point*9+jj_point]
                    fgrav_sumx += fgrav[ii_point*9+jj_point]
                elif jj_point == 1:
                    cdg_y += mesh_data.point[ii_point,1]*fgrav[ii_point*9+jj_point]
                    fgrav_sumy += fgrav[ii_point*9+jj_point]
                elif jj_point == 2:
                    cdg_z += mesh_data.point[ii_point,2]*fgrav[ii_point*9+jj_point]
                    fgrav_sumz += fgrav[ii_point*9+jj_point]
        cdg_x /= fgrav_sumx
        cdg_y /= fgrav_sumy
        cdg_z /= fgrav_sumz
        mesh_data.cdg = [cdg_x,cdg_y,cdg_z]
    for mark in mesh_mark:
        for bound in case_setup.boundary:
            # For each marker in the mesh find if there is some boundary condition
            if mark.name == bound.marker:
                if bound.type == "BC_NODELOAD":
                    bound,func,funcder,funcderder = time_func(bound,time_val)
                    # If bc is a load in a node, add the value to the global force vector
                    for node in mark.node:
                        r_ref  = np.matmul(rot_mat_p[int(node),1,:,:],np.concatenate(([section.ref_x[int(node)],section.ref_y[int(node)]],[0])))
                        bound.values[3:6] += np.cross(r_ref,bound.values[:3])
                        if node<num_point:
                            FF_global[int(9*node):int(9*node+9),ii_time] = np.array(bound.values)*func
                        else:
                            FF_global[int(node):,ii_time] = np.array(bound.values)*func
                    FF_orig = FF_global.copy()
                elif bound.type == "BC_AERO" and case_setup.problem_type == "AEL_DYN":
                    flag_aerodyn = 1
                    aoa_totvec0 = np.zeros((mesh_data.num_point,))
                    aoa_totvec1 = np.zeros((mesh_data.num_point,))
                    ang_totvec0 = np.zeros((mesh_data.num_point,))
                    ang_totvec1 = np.zeros((mesh_data.num_point,))
                    dis_totvec0 = np.zeros((mesh_data.num_point,))
                    dis_totvec1 = np.zeros((mesh_data.num_point,))
                    if bound.vinf == "VINF_DAT":
                        vel2 = bound.vinf
                        vel2dir = bound.vdir
                        bound.vinf = np.linalg.norm(case_setup.vinf)
                        bound.vdir = case_setup.vinf/np.linalg.norm(case_setup.vinf)
                    marktot = [mark]
                    if bound.f_flap == 1:
                        for markflap in mesh_mark:
                            if markflap.name == bound.flap:
                                if len(markflap.node) != len(mark.node):
                                    print('Error: core and flap have different number of nodes')
                                    sys.exit()
                                marktot.append(markflap)
                                break
                    bound,func,funcder,funcderder = time_func(bound,time_val)
                    try:
                        vinf_copy = bound.vinf
                        bound.vinf *= func
                    except:
                        vinf_copy = 0
                    try:
                        vrot_copy = bound.vrot
                        bound.vrot *= func
                    except:
                        vrot_copy = 0
#                    vinf_copy = bound.vinf
#                    vrot_copy = bound.vrot
#                    bound,func = time_func(bound,time_val)
#                    bound.vinf *= func
#                    bound.vrot *= func
                    # u_val              : value of the displacement in axis x
                    # v_val              : value of the displacement in axis y
                    # w_val              : value of the displacement in axis z
                    # u_dt_val           : value of the velocity in axis x
                    # v_dt_val           : value of the velocity in axis y
                    # w_dt_val           : value of the velocity in axis z
                    # udd_val_filter     : value of the filtered acceleration in axis x
                    # vdd_val_filter     : value of the filtered acceleration in axis y
                    # wdd_val_filter     : value of the filtered acceleration in axis z
                    # uddd_val_filter    : value of the filtered celerity in axis x
                    # vddd_val_filter    : value of the filtered celerity in axis y
                    # wddd_val_filter    : value of the filtered celerity in axis z
                    # phi_val            : value of the rotation in axis x
                    # psi_val            : value of the rotation in axis y
                    # theta_val          : value of the rotation in axis z
                    # phi_dt_val         : value of the angular velocity in axis x
                    # psi_dt_val         : value of the angular velocity in axis y
                    # theta_dt_val       : value of the angular velocity in axis z
                    # phidd_val_filter   : value of the filtered angular acceleration in axis x
                    # psidd_val_filter   : value of the filtered angular acceleration in axis y
                    # thetadd_val_filter : value of the filtered angular acceleration in axis z
                    u_val              = disp_values.u
                    v_val              = disp_values.v
                    w_val              = disp_values.w
                    ud_val_filter      = disp_values.u_dt
                    vd_val_filter      = disp_values.v_dt
                    wd_val_filter      = disp_values.w_dt
                    udd_val_filter     = disp_values.u_dtdt
                    vdd_val_filter     = disp_values.u_dtdt
                    wdd_val_filter     = disp_values.u_dtdt
                    uddd_val_filter    = disp_values.u_dtdtdt
                    vddd_val_filter    = disp_values.u_dtdtdt
                    wddd_val_filter    = disp_values.u_dtdtdt
                    phi_val            = disp_values.phi
                    psi_val            = disp_values.psi
                    theta_val          = disp_values.theta
                    phid_val_filter    = disp_values.phi_dt
                    psid_val_filter    = disp_values.psi_dt
                    thetad_val_filter  = disp_values.theta_dt
                    phidd_val_filter   = disp_values.phi_dtdt
                    psidd_val_filter   = disp_values.psi_dtdt
                    thetadd_val_filter = disp_values.theta_dtdt
                    # aoa_storearray0 : angle of attack storage list in array form for nodes 0
                    # aoa_storearray1 : angle of attack storage list in array form for nodes 1
                    # ang_storearray0 : angle of twist storage list in array form for nodes 0
                    # ang_storearray1 : angle of twist storage list in array form for nodes 1
                    # dis_storearray0 : diplacement storage list in array form for nodes 0
                    # dis_storearray1 : diplacement storage list in array form for nodes 1
                    aoa_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    aoa_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    ang_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    ang_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    angder_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    angder_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    angderder_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    angderder_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    dis_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    dis_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    disder_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    disder_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    disderder_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    disderder_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                    for iiaero_node in np.arange(len(mark.node)):
                        for jjaero_node in np.arange(len(disp_values.aoa_store0)):
                            iimark_node = int(mark.node.flatten()[iiaero_node])
                            aoa_storearray0[iiaero_node,jjaero_node] = disp_values.aoa_store0[jjaero_node][iimark_node]
                            aoa_storearray1[iiaero_node,jjaero_node] = disp_values.aoa_store1[jjaero_node][iimark_node]
                            ang_storearray0[iiaero_node,jjaero_node] = disp_values.ang_store0[jjaero_node][iimark_node]
                            ang_storearray1[iiaero_node,jjaero_node] = disp_values.ang_store1[jjaero_node][iimark_node]
                            angder_storearray0[iiaero_node,jjaero_node] = disp_values.angder_store0[jjaero_node][iimark_node]
                            angder_storearray1[iiaero_node,jjaero_node] = disp_values.angder_store1[jjaero_node][iimark_node]
                            angderder_storearray0[iiaero_node,jjaero_node] = disp_values.angderder_store0[jjaero_node][iimark_node]
                            angderder_storearray1[iiaero_node,jjaero_node] = disp_values.angderder_store1[jjaero_node][iimark_node]
                            dis_storearray0[iiaero_node,jjaero_node] = disp_values.dis_store0[jjaero_node][iimark_node]
                            dis_storearray1[iiaero_node,jjaero_node] = disp_values.dis_store1[jjaero_node][iimark_node]
                            disder_storearray0[iiaero_node,jjaero_node] = disp_values.disder_store0[jjaero_node][iimark_node]
                            disder_storearray1[iiaero_node,jjaero_node] = disp_values.disder_store1[jjaero_node][iimark_node]
                            disderder_storearray0[iiaero_node,jjaero_node] = disp_values.disderder_store0[jjaero_node][iimark_node]
                            disderder_storearray1[iiaero_node,jjaero_node] = disp_values.disderder_store1[jjaero_node][iimark_node]
                    if bound.f_flap == 1:
                        aoaflap_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        aoaflap_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        angflap_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        angflap_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        angderflap_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        angderflap_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        angderderflap_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        angderderflap_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        disflap_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        disflap_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        disderflap_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        disderflap_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        disderderflap_storearray0 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        disderderflap_storearray1 = np.zeros((len(mark.node),len(disp_values.aoa_store0)))
                        for iiaero_node in np.arange(len(mark.node)):
                            for jjaero_node in np.arange(len(disp_values.aoa_store0)):
                                iimark_node = int(mark.node.flatten()[iiaero_node])
                                aoaflap_storearray0[iiaero_node,jjaero_node] = disp_values.aoa_store0[jjaero_node][iimark_node]
                                aoaflap_storearray1[iiaero_node,jjaero_node] = disp_values.aoa_store1[jjaero_node][iimark_node]
                                angflap_storearray0[iiaero_node,jjaero_node] = disp_values.ang_store0[jjaero_node][iimark_node]
                                angflap_storearray1[iiaero_node,jjaero_node] = disp_values.ang_store1[jjaero_node][iimark_node]
                                angderflap_storearray0[iiaero_node,jjaero_node] = disp_values.angder_store0[jjaero_node][iimark_node]
                                angderflap_storearray1[iiaero_node,jjaero_node] = disp_values.angder_store1[jjaero_node][iimark_node]
                                angderderflap_storearray0[iiaero_node,jjaero_node] = disp_values.angderder_store0[jjaero_node][iimark_node]
                                angderderflap_storearray1[iiaero_node,jjaero_node] = disp_values.angderder_store1[jjaero_node][iimark_node]
                                disflap_storearray0[iiaero_node,jjaero_node] = disp_values.dis_store0[jjaero_node][iimark_node]
                                disflap_storearray1[iiaero_node,jjaero_node] = disp_values.dis_store1[jjaero_node][iimark_node]
                                disderflap_storearray0[iiaero_node,jjaero_node] = disp_values.disder_store0[jjaero_node][iimark_node]
                                disderflap_storearray1[iiaero_node,jjaero_node] = disp_values.disder_store1[jjaero_node][iimark_node]
                                disderderflap_storearray0[iiaero_node,jjaero_node] = disp_values.disderder_store0[jjaero_node][iimark_node]
                                disderderflap_storearray1[iiaero_node,jjaero_node] = disp_values.disderder_store1[jjaero_node][iimark_node]
                    # do the boundary condition for the polar specified in the marker
                    polar_bc_ind = 0
                    for polar_bc in case_setup.polar:
                        if polar_bc.id == bound.polar:
                            if polar_bc.lltnodesflag == 1:
                                for mark_llt in mesh_mark:
                                    if polar_bc.lltnodesmark == mark_llt.name:
                                        polar_bc.lltnodes = mark_llt.node
                                        polar_bc.lltnodes_ind = []
                                        polar_bc.flag_lltnodes_ind = 1
                            if polar_bc.cor3dflag == 1:
                                for mark_llt in mesh_mark:
                                    if polar_bc.cor3dmark == mark_llt.name:
                                        polar_bc.cor3dnodes = mark_llt.node
                                        polar_bc.cor3dnodes_ind = []
                                        polar_bc.flag_cor3dnodes_ind = 1
                            polar_bctot = [polar_bc]
                            polar_bctot_ind = [polar_bc_ind] 
                            break
                        polar_bc_ind += 1
                    # cdst_value0  : stationary drag coefficient of node 0
                    # cdst_value1  : stationary drag coefficient of node 1
                    # clst_value0  : stationary lift coefficient of node 0
                    # clst_value1  : stationary lift coefficient of node 1
                    # cmst_value0  : stationary pitch moment of node 0
                    # cmst_value1  : stationary pitch moment of node 1
                    # cddyn_value0 : dynamic drag coefficient of node 0
                    # cddyn_value1 : dynamic drag coefficient of node 1
                    # cldyn_value0 : dynamic lift coefficient of node 0
                    # cldyn_value1 : dynamic lift coefficient of node 1
                    # cmdyn_value0 : dynamic pitch moment coefficient of node 0
                    # cmdyn_value1 : dynamic pitch moment coefficient of node 1
                    # cd_ind       : induced drag
                    # cl_value0    : lift coefficient of node 0
                    # cl_value1    : lift coefficient of node 1
                    # cd_value0    : drag coefficient of node 0
                    # cd_value1    : drag coefficient of node 1
                    # cm_value0    : pitch moment coefficient of node 0
                    # cm_value1    : pitch moment coefficient of node 1
                    # lpos         : distance to the root of the aerodynamic surface
                    cdst_value0  = np.zeros((len(mark.node),))
                    cdst_value1  = np.zeros((len(mark.node),))
                    clst_value0  = np.zeros((len(mark.node),))
                    clst_value1  = np.zeros((len(mark.node),))
                    cmst_value0  = np.zeros((len(mark.node),))
                    cmst_value1  = np.zeros((len(mark.node),))
                    cddyn_value0 = np.zeros((len(mark.node),))
                    cddyn_value1 = np.zeros((len(mark.node),))
                    cldyn_value0 = np.zeros((len(mark.node),))
                    cldyn_value1 = np.zeros((len(mark.node),))
                    cmdyn_value0 = np.zeros((len(mark.node),))
                    cmdyn_value1 = np.zeros((len(mark.node),))
                    lpos         = np.zeros((len(mark.node),))
                    # Geometric characteristics of the cross-sections
                    # chord_vec       : chord of the airfoil in vectorial form
                    # b_chord         : half of the chord length
                    # vdir0           : velocity vector in the direction 0
                    # vdir1           : velocity vector in the direction 1
                    # aoa0_geom       : geometric angle of attack of the problem elem 0
                    # aoa1_geom       : geometric angle of attack of the problem elem 1
                    # LL_cum          : total cumulated longitude
                    # refaxis_0       : moment coefficient direction elem 0
                    # refaxis_0       : moment coefficient direction elem 1
                    # refaxis2_0      : lift coefficient direction elem 0
                    # refaxis2_0      : lift coefficient direction elem 1
                    # aoa_0lift       : null lift angle of attack
                    # lpos            : distance to the reference node
                    # iimark_node     : index of the node in the marker
                    b_chord       = np.zeros((len(mark.node),))
                    chord_vec     = np.zeros((len(mark.node),3))
                    non_acac_dist = np.zeros((len(mark.node),))
                    non_eaac_dist = np.zeros((len(mark.node),))
                    non_eamc_dist = np.zeros((len(mark.node),))
                    vdir0         = np.zeros((len(mark.node),3))
                    vdir1         = np.zeros((len(mark.node),3))
                    aoa0_geom     = np.zeros((len(mark.node),))
                    aoa1_geom     = np.zeros((len(mark.node),))
                    refaxis_0     = np.zeros((len(mark.node),3))
                    refaxis_1     = np.zeros((len(mark.node),3))
                    refaxis2_0    = np.zeros((len(mark.node),3))
                    refaxis2_1    = np.zeros((len(mark.node),3))
                    aoa_0lift     = np.zeros((len(mark.node),))
                    cd_aoa0       = np.zeros((len(mark.node),)) 
                    # Add the value of the longitude simulated in each beam element node
                    for iiaero_node in np.arange(len(mark.node)):
                        iimark_node = int(mark.node.flatten()[iiaero_node])
                        lpos[iiaero_node] = np.linalg.norm(mesh_data.point[iimark_node,:]-mesh_data.point[bound.refpoint])
                    LL_cum = np.max(lpos)
                    # If more than 1 polar
                    if polar_bc.pointfile[0] != -1:
                        pointref = np.where(mark.node==polar_bc.pointfile)[0].tolist()
                        lrefpol  = lpos[pointref]
                        polar_bc.iirefpol = np.argsort(lrefpol)
                        polar_bc.sortedlref = lrefpol[polar_bc.iirefpol]
                    if bound.f_flap == 1:
                        polar_bcflap_ind = 0
                        for polar_bcflap in case_setup.polar:
                            if polar_bcflap.id == bound.flappolar:
                                if polar_bcflap.lltnodesflag == 1:
                                    for mark_llt in mesh_mark:
                                        if polar_bcflap.lltnodesmark == mark_llt.name:
                                            polar_bcflap.lltnodes = mark_llt.node
                                            polar_bcflap.lltnodes_ind = []
                                            polar_bcflap.flag_lltnodes_ind = 1
                                if polar_bcflap.cor3dflag == 1:
                                    for mark_llt in mesh_mark:
                                        if polar_bcflap.cor3dmark == mark_llt.name:
                                            polar_bcflap.cor3dnodes = mark_llt.node
                                            polar_bcflap.cor3dnodes_ind = []
                                            polar_bcflap.flag_cor3dnodes_ind = 1
                                polar_bctot.append(polar_bcflap)
                                polar_bctot_ind.append(polar_bcflap_ind)
                                break
                            polar_bcflap_ind += 1
                    cdstflap_value0   = np.zeros((len(mark.node),))
                    cdstflap_value1   = np.zeros((len(mark.node),))
                    clstflap_value0   = np.zeros((len(mark.node),))
                    clstflap_value1   = np.zeros((len(mark.node),))
                    cmstflap_value0   = np.zeros((len(mark.node),))
                    cmstflap_value1   = np.zeros((len(mark.node),))
                    cddynflap_value0   = np.zeros((len(mark.node),))
                    cddynflap_value1   = np.zeros((len(mark.node),))
                    cldynflap_value0   = np.zeros((len(mark.node),))
                    cldynflap_value1   = np.zeros((len(mark.node),))
                    cmdynflap_value0   = np.zeros((len(mark.node),))
                    cmdynflap_value1   = np.zeros((len(mark.node),))
                    b_chordflap       = np.zeros((len(mark.node),))
                    chord_vecflap     = np.zeros((len(mark.node),3))
                    non_acac_distflap = np.zeros((len(mark.node),))
                    non_eaac_distflap = np.zeros((len(mark.node),))
                    non_eamc_distflap = np.zeros((len(mark.node),))
                    aoa0_geomflap    = np.zeros((len(mark.node),))
                    aoa1_geomflap    = np.zeros((len(mark.node),))
                    coreflapdist_adim = np.zeros((len(mark.node),))
                    deltaf_0lift     = np.zeros((len(mark.node),))
                    aoa_0liftflap = np.zeros((len(mark.node),))
                    cd_aoa0flap       = np.zeros((len(mark.node),))
                    # For all the nodes in the surface
                    # cent_secx       : center of the airfoil x position
                    # cent_secy       : center of the airfoil y position
                    # acac_dist       : distance between the aerodynamic reference and the 0.25 chord distance
                    # acac_dist_sign  : sign of the distance between the aerodynamic reference and the 0.25 chord distance
                    # non_acac_dist   : nondimensional distance between the aerodynamic reference and the 0.25 chord distance with the moment sign
                    # eaac_dist       : distance between the aerodynamic reference and the elastic reference
                    # eaac_dist_sign  : sign of the distance between the aerodynamic reference and the elastic reference
                    # non_eaac_dist   : nondimensional distance between the aerodynamic reference and the elastic reference with the moment sign
                    # eamc_dist       : distance between the aerodynamic reference and the mean chord
                    # eamc_dist_sign  : sign of the distance between the aerodynamic reference and the mean chord
                    # non_eamc_dist   : nondimensional distance between the aerodynamic reference and the mean chord with the moment sign
                    for iiaero_node in np.arange(len(mark.node)):
                        iimark_node                = int(mark.node.flatten()[iiaero_node])
                        refaxis_0[iiaero_node,:]  = -rr_vec_p[0,iimark_node,:]
                        if polar_bc.eff3d == 'BEM':
                            vrefc =  np.cross(refaxis_0[iiaero_node,:],bound.refrot)/np.linalg.norm(np.cross(refaxis_0[iiaero_node,:],bound.refrot))
                        else:
                            vrefc = bound.vdir
                        cent_secx                  = (section_globalCS.point_3d[iimark_node][section.te[iimark_node],0]+section_globalCS.point_3d[iimark_node][section.le[iimark_node],0])/2 #(section.points[iimark_node][0][section.te[iimark_node],0]+section.points[iimark_node][0][section.le[iimark_node],0])/2
                        cent_secy                  = (section_globalCS.point_3d[iimark_node][section.te[iimark_node],1]+section_globalCS.point_3d[iimark_node][section.le[iimark_node],1])/2 #(section.points[iimark_node][0][section.te[iimark_node],1]+section.points[iimark_node][0][section.le[iimark_node],1])/2
                        chord_vec[iiaero_node,:]   = section_globalCS.point_3d[iimark_node][section.te[iimark_node],:]-section_globalCS.point_3d[iimark_node][section.le[iimark_node],:] #section.points[iimark_node][0][section.te[iimark_node],:]-section.points[iimark_node][0][section.le[iimark_node],:]
                        b_chord[iiaero_node]       = np.linalg.norm(chord_vec[iiaero_node,:])/2
                        acac_dist                  = section_globalCS.aero_cent[iimark_node]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section_globalCS.point_3d[iimark_node][section.le[iimark_node],0]) #[section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]]-b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimark_node][0][section.le[iimark_node],:]
                        acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_acac_dist[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
                        eaac_dist                  = section_globalCS.aero_cent[iimark_node] #[section.ae_orig_x[iimark_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimark_node]-sol_phys.ysc[iiaero_node]]
                        eaac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],eaac_dist),np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_eaac_dist[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
                        eamc_dist                  = [-cent_secx,-cent_secy,0] #[sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
                        eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_eamc_dist[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chord[iiaero_node]   
                        if bound.f_flap == 1 and markflap.node.flatten()[iiaero_node] >= 0:
                            iimarkflap_node                = int(markflap.node.flatten()[iiaero_node])
                            if iimarkflap_node > 0:
                                cent_secx                  = (section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],0]+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],0])/2 #(section.points[iimarkflap_node][0][section.te[iimarkflap_node],0]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],0])/2
                                cent_secy                  = (section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],1]+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],1])/2 #(section.points[iimarkflap_node][0][section.te[iimarkflap_node],1]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],1])/2
                                chord_vecflap[iiaero_node,:]   = section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],:]-section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],:] #section.points[iimarkflap_node][0][section.te[iimarkflap_node],:]-section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
                                b_chordflap[iiaero_node]       = np.linalg.norm(chord_vecflap[iiaero_node,:])/2
                                acac_dist                  = section_globalCS.aero_cent[iimarkflap_node]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],0]) #[section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]]-b_chordflap[iiaero_node]/2*chord_vecflap[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
                                acac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimarkflap_node,0,:,:],acac_dist),np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_acac_distflap[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
                                eaac_dist                  = section_globalCS.aero_cent[iimark_node] #[section.ae_orig_x[iimarkflap_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimarkflap_node]-sol_phys.ysc[iiaero_node]]
                                eaac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimarkflap_node,0,:,:],eaac_dist),np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_eaac_distflap[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
                                eamc_dist                  = [-cent_secx,-cent_secy,0] # [sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
                                eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_eamc_distflap[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chordflap[iiaero_node] 
                                coreflapdist  = mesh_data.point[iimark_node,:]-mesh_data.point[iimarkflap_node,:]
                                coreflapdist_adim[iiaero_node] = np.sign(np.dot(coreflapdist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))/(b_chord[iiaero_node]+b_chordflap[iiaero_node])                             
#                        cent_secx                  = (section.points[iimark_node][0][section.te[iimark_node],0]+section.points[iimark_node][0][section.le[iimark_node],0])/2
#                        cent_secy                  = (section.points[iimark_node][0][section.te[iimark_node],1]+section.points[iimark_node][0][section.le[iimark_node],1])/2
#                        chord_vec[iiaero_node,:]   = section.points[iimark_node][0][section.te[iimark_node],:]-section.points[iimark_node][0][section.le[iimark_node],:]
#                        b_chord[iiaero_node]       = np.linalg.norm(chord_vec[iiaero_node,:])/2
#                        acac_dist                  = [section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]]-b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimark_node][0][section.le[iimark_node],:]
#                        acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],vrefc)[:2]))
#                        non_acac_dist[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
#                        chord34                    = section.points[iimark_node][0][section.le[iimark_node],:]+0.75*chord_vec[iiaero_node,:] 
#                        eaac_dist                  = [section.ae_orig_x[iimark_node]-sol_phys.xsc[iimark_node],section.ae_orig_y[iimark_node]-sol_phys.ysc[iimark_node]] #[section.ae_orig_x[iimark_node]-chord34[0],section.ae_orig_y[iimark_node]-sol_phys.ysc[iimark_node]-chord34[1]] #
#                        eaac_dist_sign             = np.sign(np.dot(eaac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],vrefc)[:2]))
#                        non_eaac_dist[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
#                        eamc_dist                  =  [-cent_secx,-cent_secy,0] # [sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
#                        eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],vrefc)))
#                        non_eamc_dist[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chord[iiaero_node]   
#                        if bound.f_flap == 1  and markflap.node.flatten()[iiaero_node] >= 0:
#                            iimarkflap_node                = int(markflap.node.flatten()[iiaero_node])
#                            cent_secx                  = (section.points[iimarkflap_node][0][section.te[iimarkflap_node],0]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],0])/2
#                            cent_secy                  = (section.points[iimarkflap_node][0][section.te[iimarkflap_node],1]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],1])/2
#                            chord_vecflap[iiaero_node,:]   = section.points[iimarkflap_node][0][section.te[iimarkflap_node],:]-section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
#                            b_chordflap[iiaero_node]       = np.linalg.norm(chord_vecflap[iiaero_node,:])/2
#                            acac_dist                  = [section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]]-b_chordflap[iiaero_node]/2*chord_vecflap[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
#                            acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],vrefc)[:2]))
#                            non_acac_distflap[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
#                            eaac_dist                  = [section.ae_orig_x[iimarkflap_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimarkflap_node]-sol_phys.ysc[iiaero_node]]
#                            eaac_dist_sign             = np.sign(np.dot(eaac_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],vrefc)[:2]))
#                            non_eaac_distflap[iiaero_node] = -eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
#                            eamc_dist                  =  [sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
#                            eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],vrefc)))
#                            non_eamc_distflap[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chordflap[iiaero_node] 
#                            coreflapdist  = mesh_data.point[iimark_node,:]-mesh_data.point[iimarkflap_node,:]
#                            coreflapdist_adim[iiaero_node] = np.sign(np.dot(coreflapdist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],vrefc)))/(b_chord[iiaero_node]+b_chordflap[iiaero_node])  
                    # If a rotative aerodynamic model is selected the nondimensional radius is calculated
                    # flag_start    : flag to start the bisection method
                    # flag_start1   : flag to start the bisection method
                    # flag_start2   : flag to start the bisection method
                    # r_vinduced    : nondimensional radius
                    # cltot2        : total value of the lift coefficient in the previous iteration
                    # cttot2        : total value of the thrust coefficient in the previous iteration
                    # v_induced     : induced velocity 
                    # Gamma_induced : induced circulation
                    # tol_vind      : tolerance of the induced velocity error
                    # aux_vind_e    : index to count the number of iterations in the induced velocity calculation loop
                    # lpos_cir      : distance to the root of the aerodynamic surface in circular coordinates
                    # lpc_in        : position of the section relative to the total length
                    # mat_pllt      : lifting line theory matrix
                    # vec_pllt      : lifting line theory vector
                    if polar_bc.eff3d == 'LLT':
                        class eff_3d:
                            notconverged = 0
                            Gamma_induced = np.zeros((len(mark.node),))
                            cltot2 = 1e3
                            lpos_cir  = np.zeros((len(mark.node),))
                            sign_vi = -1
                        v_induced = np.zeros((len(mark.node),3))
                        for iiaero_node in np.arange(len(mark.node)):
                            lpc_in = lpos[iiaero_node]/LL_cum
                            if lpc_in>1:
                                lpc_in = 1
                            eff_3d.lpos_cir[iiaero_node] = np.arccos(lpc_in) 
                    elif polar_bc.eff3d == 'BEM':
                        class eff_3d:
                            notconverged = 0
                            flag_start      = np.zeros((len(mark.node),))
                            flag_start1     = np.zeros((len(mark.node),))
                            flag_start2     = np.zeros((len(mark.node),))
                            error1          = np.ones((len(mark.node),))
                            error2          = np.ones((len(mark.node),))
                            sign_vi         = -np.sign(np.dot(bound.vdir,bound.refrot))
                            r_vinduced      = lpos/LL_cum
                            v_induced2      = np.zeros((len(mark.node),3))
                            masa            = np.zeros((len(mark.node),)) #sol_phys.m11
                            frelax_vt       = 0.5
                            errorout1       = 1
                            errorout2       = 2
                        for ii_massind in np.arange(len(mark.node)):
                            ii_mass = int(mark.node[ii_massind])
                            eff_3d.masa[ii_massind] = MM_global[ii_mass,ii_mass]
                            sign_vi = -1
                        if polar_bc.startbem == 1:
                            v_induced         = 0*bound.vrot*bound.radius*np.ones((len(mark.node),3))
                            v_induced[:,0]   *= eff_3d.sign_vi*bound.vdir[0]/np.linalg.norm(bound.vdir)
                            v_induced[:,1]   *= eff_3d.sign_vi*bound.vdir[1]/np.linalg.norm(bound.vdir)
                            v_induced[:,2]   *= eff_3d.sign_vi*bound.vdir[2]/np.linalg.norm(bound.vdir)
                            vtan_induced = np.zeros((len(mark.node),3))
                        else:
                            v_induced    = disp_values.v_induced.copy() 
                            vtan_induced = disp_values.vtan_induced.copy()
                        eff_3d.v_induced_tan0     = vtan_induced
                        eff_3d.v_induced_tan1     = vtan_induced
                        eff_3d.v_induced1 = v_induced.copy()
                    else: 
                        v_induced = np.zeros((len(mark.node),3))                         
                        class eff_3d:
                            notconverged = 0
                            sign_vi      = 1
                    try:
                        tol_vind = polar_bc.vind_tol
                    except:
                        tol_vind = 1e-5
                    try:
                        maxit_vind = polar_bc.vind_maxit
                    except:
                        maxit_vind = 1e3
                    eff_3d.aux_vind_e_out = 0 
                    # initialize the error
                    # error  : error of the lift coefficient in each iteration
                    error_out  = tol_vind*10 
                    flag_conv = 0
                    flag_convout = 0
                    # While error is  higher than the required tolerance
                    while abs(error_out)>tol_vind or flag_convout == 0:
                        if abs(error_out)<tol_vind:
                            flag_convout = 1
                            if polar_bc.eff3d == 'BEM':
                                vtan_induced_mod = vtan_induced.copy()
                                vtan_ind_int = vtan_induced[polar_bc.lltnodes_ind,:]
                                l_ind_int = lpos[polar_bc.lltnodes_ind]
                                f_vxind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,0],'cubic', fill_value='extrapolate')
                                f_vyind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,1],'cubic', fill_value='extrapolate')
                                f_vzind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,2],'cubic', fill_value='extrapolate') 
                                for auxaeronode in np.arange(len(mark.node)):
                                    vtan_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
                                vtan_induced = vtan_induced_mod.copy()
                            else:
                                break
                        eff_3d.aux_vind_e_out += 1
#                                print([error_out,eff_3d.frelax_vt])
                        # if the loop cannot find the solution in a determined number of iterations, stop the loop
                        if eff_3d.aux_vind_e_out > maxit_vind or eff_3d.notconverged == 1: 
                            print('Induced tangential velocity: Not converged - '+str(error_out))
                            try:
                                vtan_induced = disp_values.vtan_induced.copy()
                            except:
                                vtan_induced *= 0
                            break
                        # error  : error of the lift coefficient in each iteration
                        error  = tol_vind*10
                        if polar_bc.eff3d == 'BEM' and time_val/case_setup.time_stp % vical != 0:
                                vtan_induced = disp_values.vtan_induced.copy()
                        eff_3d.error = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.error1ct = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.error2ct = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.flag_start = np.zeros((len(mark.node),))
                        eff_3d.flag_start1 = np.zeros((len(mark.node),))
                        eff_3d.flag_start2 = np.zeros((len(mark.node),))
                        eff_3d.aux_vind_e = 1            
                        # While error is  higher than the required tolerance
                        while abs(error)>tol_vind or flag_conv == 0:
                            if abs(error)<tol_vind:
                                if polar_bc.eff3d == 'BEM':
                                    v_induced_mod = v_induced.copy()
                                    v_ind_int = v_induced[polar_bc.lltnodes_ind,:]
                                    l_ind_int = lpos[polar_bc.lltnodes_ind]
                                    f_vxind = interpolate.interp1d(l_ind_int,v_ind_int[:,0],'cubic', fill_value='extrapolate')
                                    f_vyind = interpolate.interp1d(l_ind_int,v_ind_int[:,1],'cubic', fill_value='extrapolate')
                                    f_vzind = interpolate.interp1d(l_ind_int,v_ind_int[:,2],'cubic', fill_value='extrapolate') 
                                    for auxaeronode in np.arange(len(mark.node)):
                                        v_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
                                    v_induced = v_induced_mod.copy()
                                flag_conv = 1
                            # Calculate the reference axis
                            refaxis_0, refaxis_1, refaxis2_0, refaxis2_1, vdir0, vdir1, aoa0_geom, aoa1_geom, eff_3d = ref_axis_aero(bound,mark,polar_bc,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,v_induced,vdir0,vdir1,aoa0_geom,aoa1_geom,rr_vec_p,rot_matT_p,chord_vec,eff_3d,0)
                            # if the loop cannot find the solution in a determined number of iterations, stop the loop
                            if eff_3d.aux_vind_e > maxit_vind or eff_3d.notconverged == 1: 
                                print('Induced velocity: Not converged - '+str(error))
                                try:
                                    v_induced    = disp_values.v_induced.copy()
                                except:
                                    v_induced *= 0
                                eff_3d.notconverged = 1
                                break
                            eff_3d.aux_vind_e += 1
                            if time_val/case_setup.time_stp % vical != 0:
                                    v_induced    = disp_values.v_induced.copy()
                            # Define variables for the rotative aerodynamics
                            if polar_bc.eff3d == 'LLT':            
                                eff_3d.mat_pllt  = np.zeros((2*len(mark.node),len(mark.node)))
                                eff_3d.vec_pllt  = np.zeros((2*len(mark.node),))
                            elif polar_bc.eff3d == 'BEM':
                                # r_vinduced     : nondimensional distance of the blade radius
                                # vi_vinduced    : induced velocity
                                # phi_vinduced   : induced velocity on the blade
                                # f_vinduced     : 3D effect factor of the blade
                                # F_vinduced     : 3D effect factor of the blade
                                # dct_dr_induced : variation of thrust with radial distance
                                eff_3d.phi_vinduced0       = np.zeros((len(mark.node),))
                                eff_3d.f_vinduced0         = np.zeros((len(mark.node),))
                                eff_3d.F_vinduced0         = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced0 = np.zeros((len(mark.node),))
                                eff_3d.phi_vinduced1       = np.zeros((len(mark.node),))
                                eff_3d.f_vinduced1         = np.zeros((len(mark.node),))
                                eff_3d.F_vinduced1         = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced1 = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced  = np.zeros((len(mark.node),))
                            # ang_vec0    : twist of node 0
                            # ang_vec1    : twist of node 1
                            # dis_vec0    : displacement of node 0
                            # dis_vec1    : displacement of node 1
                            # disp_vec0   : velocity of node 0
                            # disp_vec1   : velocity of node 1
                            # aoa_vec0    : angle of attack of node 0
                            # aoa_vec1    : angle of attack of node 1
                            # aoa_vec0_st : angle of attack of node 0
                            # aoa_vec1_st : angle of attack of node 1
                            # aoap_vec0   : angle of attack derivative of node 0
                            # aoap_vec1   : angle of attack derivative of node 1
                            # aoap_vec0   : angle of attack derivative of node 0
                            # aoap_vec1   : angle of attack derivative of node 1
                            # aoa_val     : value of the angle of attack
                            # aoap_val    : value of the angle of attack derivative
                            # aoapp_val   : value of the angle of attack second derivative
                            # pos_vec0    : position vector of node 0
                            # pos_vec1    : position vector of node 1
                            ang_vec0    = np.zeros((len(mark.node),))
                            ang_vec1    = np.zeros((len(mark.node),))
                            angp_vec0    = np.zeros((len(mark.node),))
                            angp_vec1    = np.zeros((len(mark.node),))
                            angpp_vec0    = np.zeros((len(mark.node),))
                            angpp_vec1    = np.zeros((len(mark.node),))
                            dis_vec0    = np.zeros((len(mark.node),))
                            dis_vec1    = np.zeros((len(mark.node),))
                            disp_vec0   = np.zeros((len(mark.node),))
                            disp_vec1   = np.zeros((len(mark.node),))
                            dispp_vec0   = np.zeros((len(mark.node),))
                            dispp_vec1   = np.zeros((len(mark.node),))
                            disppp_vec0   = np.zeros((len(mark.node),))
                            disppp_vec1   = np.zeros((len(mark.node),))
                            aoa_ini0    = np.zeros((len(mark.node),))
                            aoa_ini1    = np.zeros((len(mark.node),))
                            aoa_vec0    = np.zeros((len(mark.node),))
                            aoa_vec1    = np.zeros((len(mark.node),))
                            aoap_vec0   = np.zeros((len(mark.node),))
                            aoap_vec1   = np.zeros((len(mark.node),))
                            aoapp_vec0  = np.zeros((len(mark.node),))
                            aoapp_vec1  = np.zeros((len(mark.node),))
                            aoa_val     = np.zeros((len(mark.node),))
                            aoap_val    = np.zeros((len(mark.node),))
                            aoapp_val   = np.zeros((len(mark.node),))
                            pos_vec0    = np.zeros((len(mark.node),))
                            pos_vec1    = np.zeros((len(mark.node),))
                            aoa_vec0_st = np.zeros((len(mark.node),))
                            aoa_vec1_st = np.zeros((len(mark.node),))
                            ang_vec0_st = np.zeros((len(mark.node),))
                            ang_vec1_st = np.zeros((len(mark.node),))
                            Vinf0       = np.zeros((len(mark.node),))
                            Vinf1       = np.zeros((len(mark.node),))
                            reynolds0   = np.zeros((len(mark.node),))
                            reynolds1   = np.zeros((len(mark.node),))
                            angflap_vec0 = np.zeros((len(mark.node),))
                            angflap_vec1 = np.zeros((len(mark.node),))
                            angpflap_vec0 = np.zeros((len(mark.node),)) 
                            angpflap_vec1 = np.zeros((len(mark.node),))
                            angppflap_vec0 = np.zeros((len(mark.node),)) 
                            angppflap_vec1 = np.zeros((len(mark.node),))  
                            disflap_vec0 = np.zeros((len(mark.node),))
                            disflap_vec1 = np.zeros((len(mark.node),))
                            dispflap_vec0 = np.zeros((len(mark.node),))
                            dispflap_vec1 = np.zeros((len(mark.node),))
                            disppflap_vec0 = np.zeros((len(mark.node),)) 
                            disppflap_vec1 = np.zeros((len(mark.node),))
                            dispppflap_vec0 = np.zeros((len(mark.node),)) 
                            dispppflap_vec1 = np.zeros((len(mark.node),))
                            aoapflap_vec0 = np.zeros((len(mark.node),))
                            aoapflap_vec1 = np.zeros((len(mark.node),))
                            aoappflap_vec0 = np.zeros((len(mark.node),))
                            aoappflap_vec1 = np.zeros((len(mark.node),))
                            aoapflap_val = np.zeros((len(mark.node),))
                            aoappflap_val = np.zeros((len(mark.node),))
                            deltaflap_vec0    = np.zeros((len(mark.node),))
                            deltaflap_vec1    = np.zeros((len(mark.node),))
                            deltaflap = np.zeros((len(mark.node),)) 
                            aoa_mean0 = np.zeros((len(mark.node),)) 
                            aoa_mean1 = np.zeros((len(mark.node),))
                            ang_mean0 = np.zeros((len(mark.node),)) 
                            ang_mean1 = np.zeros((len(mark.node),)) 
                            dis_mean0 = np.zeros((len(mark.node),)) 
                            dis_mean1 = np.zeros((len(mark.node),)) 
                            aoaflap_mean0 = np.zeros((len(mark.node),)) 
                            aoaflap_mean1 = np.zeros((len(mark.node),))
                            angflap_mean0 = np.zeros((len(mark.node),)) 
                            angflap_mean1 = np.zeros((len(mark.node),)) 
                            disflap_mean0 = np.zeros((len(mark.node),)) 
                            disflap_mean1 = np.zeros((len(mark.node),))
                            # add a time function if required
                            # func   : value of the time function
                            # Add the value of the longitude simulated in each beam element node
                            # for each node in the beam
                            for iiaero_node in np.arange(len(mark.node)):
                                iimark_node = int(mark.node.flatten()[iiaero_node])
                                if bound.f_flap == 1:
                                    iimark_nodeflap = int(markflap.node.flatten()[iiaero_node])
                                # Calculate the velocity of the free stream for the airfoil for the different aerodynamic models
                                # refvind0       : reference induced velocity direction in element 0
                                # refvind1       : reference induced velocity direction in element 1
                                # vi_vinduced0   : induced velocity scalar in element 0
                                # vi_vinduced1   : induced velocity scalar in element 1
                                # Vinf0          : free stream velocity in element 0
                                # Vinf1          : free stream velocity in element 1
                                # Vinf_vindaoa   : velocity to calculate the angle of the induced angle of attack
                                if polar_bc.eff3d == 'LLT':
                                    refvind0           = -refaxis2_0[iiaero_node,:]
                                    refvind1           = -refaxis2_1[iiaero_node,:]
                                    v_induced[iiaero_node] = v_induced[iiaero_node]
                                    vi_vinduced0       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(refvind0))
                                    vi_vinduced1       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(refvind1))
                                    Vinf0[iiaero_node] = np.sqrt(bound.vinf**2+vi_vinduced0**2)
                                    Vinf1[iiaero_node] = np.sqrt(bound.vinf**2+vi_vinduced1**2)
                                    Vinf_vindaoa       = bound.vinf
                                elif polar_bc.eff3d == 'BEM':
                                    v_induced[iiaero_node] = v_induced[iiaero_node]
                                    refvind0           = eff_3d.sign_vi*bound.vdir
                                    refvind1           = eff_3d.sign_vi*bound.vdir
                                    vi_vinduced0       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(bound.vdir))
                                    vi_vinduced1       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(bound.vdir))
                                    Vinf0[iiaero_node] = np.sqrt((bound.vinf+vi_vinduced0)**2+(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius)**2)+1e-10
                                    Vinf1[iiaero_node] = np.sqrt((bound.vinf+vi_vinduced1)**2+(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius)**2)+1e-10
                                    v_induced_tan      = vtan_induced[iiaero_node] #(eff_3d.v_induced_tan0[iiaero_node,:]*LL_vec_p[0,iiaero_node]+eff_3d.v_induced_tan1[iiaero_node,:]*LL_vec_p[1,iiaero_node])/(LL_vec_p[0,iiaero_node]+LL_vec_p[1,iiaero_node])
#                                            vtan_induced[iiaero_node] = v_induced_tan
                                    Vinf_vindaoa       = np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+v_induced_tan)+1e-10
                                else:
                                    v_induced[iiaero_node] = v_induced[iiaero_node]
                                    refvind0           = -refaxis2_0[iiaero_node,:]
                                    refvind1           = -refaxis2_1[iiaero_node,:]
                                    Vinf0[iiaero_node] = bound.vinf 
                                    Vinf1[iiaero_node] = bound.vinf 
                                    Vinf_vindaoa       = bound.vinf
                                # vec_ang     : vector of the rotated angle
                                # vec_angp    : vector of the angular velocity
                                # vec_angpp   : vector of the angular acceleration
                                # vec_pos_ind : vector of the linear position
                                # vec_vel_ind : vector of the linear velocity
                                # vec_acc_ind : vector of the linear acceleration
                                # vec_cel_ind : vector of the linear celerity
                                vec_ang           = [phi_val[iimark_node],psi_val[iimark_node],theta_val[iimark_node]]
                                vec_angp          = [phid_val_filter[iimark_node],psid_val_filter[iimark_node],thetad_val_filter[iimark_node]]
                                vec_angpp         = [phidd_val_filter[iimark_node],psidd_val_filter[iimark_node],thetadd_val_filter[iimark_node]]
                                vec_pos_ind       = [u_val[iimark_node],v_val[iimark_node],w_val[iimark_node]]
                                vec_vel_ind       = [ud_val_filter[iimark_node],vd_val_filter[iimark_node],wd_val_filter[iimark_node]] 
                                vec_acc_ind       = [udd_val_filter[iimark_node],vdd_val_filter[iimark_node],wdd_val_filter[iimark_node]]
                                vec_cel_ind       = [uddd_val_filter[iimark_node],vddd_val_filter[iimark_node],wddd_val_filter[iimark_node]]
                                if bound.f_flap == 1:
                                    vec_angflap           = [phi_val[iimark_nodeflap],psi_val[iimark_nodeflap],theta_val[iimark_nodeflap]]
                                    vec_angpflap          = [phid_val_filter[iimark_nodeflap],psid_val_filter[iimark_nodeflap],thetad_val_filter[iimark_nodeflap]]
                                    vec_angppflap         = [phidd_val_filter[iimark_nodeflap],psidd_val_filter[iimark_nodeflap],thetadd_val_filter[iimark_nodeflap]]
                                    vec_pos_indflap       = [u_val[iimark_nodeflap],v_val[iimark_nodeflap],w_val[iimark_nodeflap]]
                                    vec_vel_indflap       = [ud_val_filter[iimark_nodeflap],vd_val_filter[iimark_nodeflap],wd_val_filter[iimark_nodeflap]] 
                                    vec_acc_indflap       = [udd_val_filter[iimark_nodeflap],vdd_val_filter[iimark_nodeflap],wdd_val_filter[iimark_nodeflap]]
                                    vec_cel_indflap       = [uddd_val_filter[iimark_nodeflap],vddd_val_filter[iimark_nodeflap],wddd_val_filter[iimark_nodeflap]]
                                # For lifting line theory set the angle of attack at the tip as zero, calculate in the rest of points
                                if (polar_bc.eff3d == 'LLT') and lpos[iiaero_node] > 0.999*LL_cum:
                                    aoa_ini0[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoa_ini1[iiaero_node]    = aoa_0lift[iiaero_node]
                                    ang_vec0[iiaero_node]    = 0
                                    ang_vec1[iiaero_node]    = 0
                                    dis_vec0[iiaero_node]    = 0
                                    dis_vec1[iiaero_node]    = 0
                                    disp_vec0[iiaero_node]   = 0
                                    disp_vec1[iiaero_node]   = 0
                                    aoa_vec0[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoa_vec1[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoap_vec0[iiaero_node]   = 0
                                    aoap_vec1[iiaero_node]   = 0
                                    aoapp_vec0[iiaero_node]  = 0
                                    aoapp_vec1[iiaero_node]  = 0
                                    aoa_val[iiaero_node]     = aoa_0lift[iiaero_node]
                                    aoap_val[iiaero_node]    = 0
                                    aoapp_val[iiaero_node]   = 0
                                    v_induced[iiaero_node,:] = np.matmul(rot_mat_p[iiaero_node,1,:,:],Vinf0[iiaero_node]*np.tan(aoa_val[iiaero_node]-aoa0_geom[iiaero_node])*np.array(refvind1))
                                # In the rest of the points
                                else:
                                    # if needed include rotational effects
                                    if polar_bc.eff3d == 'BEM':
                                        if eff_3d.r_vinduced[iiaero_node] == 0:
                                            eff_3d.phi_vinduced0[iiaero_node] = np.arctan((bound.vinf)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                            eff_3d.phi_vinduced1[iiaero_node] = np.arctan((bound.vinf)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                        else:
                                            eff_3d.phi_vinduced0[iiaero_node] = np.arctan((bound.vinf+vi_vinduced0)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                            eff_3d.phi_vinduced1[iiaero_node] = np.arctan((bound.vinf+vi_vinduced1)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                        eff_3d.f_vinduced0[iiaero_node]   = bound.Nb/2*((1-(eff_3d.r_vinduced[iiaero_node]))/np.max([eff_3d.r_vinduced[iiaero_node]*eff_3d.phi_vinduced0[iiaero_node],1e-10]))
                                        eff_3d.f_vinduced1[iiaero_node]   = bound.Nb/2*((1-(eff_3d.r_vinduced[iiaero_node]))/np.max([eff_3d.r_vinduced[iiaero_node]*eff_3d.phi_vinduced1[iiaero_node],1e-10]))
                                        eff_3d.F_vinduced0[iiaero_node]   = 2/np.pi*np.arccos(np.exp(-eff_3d.f_vinduced0[iiaero_node]))
                                        eff_3d.F_vinduced1[iiaero_node]   = 2/np.pi*np.arccos(np.exp(-eff_3d.f_vinduced1[iiaero_node]))
                                        v_vert                            = eff_3d.sign_vi*(bound.vinf*bound.vdir+v_induced[iiaero_node,:])
                                    else:
                                        v_vert = v_induced[iiaero_node,:] #eff_3d.sign_vi*
                                    aoa_ini0[iiaero_node]    = (aoa0_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_vert),np.array(refvind0))/Vinf_vindaoa)
                                    aoa_ini1[iiaero_node]    = (aoa1_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_vert),np.array(refvind1))/Vinf_vindaoa)
                                    ang_vec0[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_ang),np.array(refaxis_0[iiaero_node,:]))
                                    ang_vec1[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_ang),np.array(refaxis_1[iiaero_node,:]))
                                    angp_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_angp),np.array(refaxis_0[iiaero_node,:]))
                                    angp_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_angp),np.array(refaxis_1[iiaero_node,:]))
                                    angpp_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_angpp),np.array(refaxis_0[iiaero_node,:]))
                                    angpp_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_angpp),np.array(refaxis_1[iiaero_node,:]))
                                    dis_vec0[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_pos_ind),np.array(refvind0))
                                    dis_vec1[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_pos_ind),np.array(refvind1))
                                    disp_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_vel_ind),np.array(refvind0))
                                    disp_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_vel_ind),np.array(refvind1))
                                    dispp_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_acc_ind),np.array(refvind0))
                                    dispp_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_acc_ind),np.array(refvind1))
                                    disppp_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_cel_ind),np.array(refvind0))
                                    disppp_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_cel_ind),np.array(refvind1))
                                    aoa_vec0[iiaero_node]    = aoa_ini0[iiaero_node]+ang_vec0[iiaero_node]+polar_bc.quasi_steady*np.arctan(disp_vec0[iiaero_node]/Vinf_vindaoa)
                                    aoa_vec1[iiaero_node]    = aoa_ini1[iiaero_node]+ang_vec1[iiaero_node]+polar_bc.quasi_steady*np.arctan(disp_vec1[iiaero_node]/Vinf_vindaoa)
                                    aoap_vec0[iiaero_node]   = angp_vec0[iiaero_node]+np.arctan(dispp_vec0[iiaero_node]/Vinf_vindaoa)
                                    aoap_vec1[iiaero_node]   = angp_vec1[iiaero_node]+np.arctan(dispp_vec1[iiaero_node]/Vinf_vindaoa)
                                    aoapp_vec0[iiaero_node]  = angpp_vec0[iiaero_node]+np.arctan(disppp_vec0[iiaero_node]/Vinf_vindaoa)
                                    aoapp_vec1[iiaero_node]  = angpp_vec1[iiaero_node]+np.arctan(disppp_vec1[iiaero_node]/Vinf_vindaoa)
                                    aoa_val[iiaero_node]     = (aoa_vec0[iiaero_node]*LL_vec_p[0,iimark_node]+aoa_vec1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
                                    aoap_val[iiaero_node]    = (aoap_vec0[iiaero_node]*LL_vec_p[0,iimark_node]+aoap_vec1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
                                    aoapp_val[iiaero_node]   = (aoapp_vec0[iiaero_node]*LL_vec_p[0,iimark_node]+aoapp_vec1[iiaero_node]*LL_vec_p[1,iimark_node])/(LL_vec_p[0,iimark_node]+LL_vec_p[1,iimark_node])
                                    if bound.f_flap == 1:
                                        angflap_vec0[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_angflap),np.array(refaxis_0[iiaero_node,:]))
                                        angflap_vec1[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_angflap),np.array(refaxis_1[iiaero_node,:]))
                                        angpflap_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_angpflap),np.array(refaxis_0[iiaero_node,:]))
                                        angpflap_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_angpflap),np.array(refaxis_1[iiaero_node,:]))
                                        angppflap_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_angppflap),np.array(refaxis_0[iiaero_node,:]))
                                        angppflap_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_angppflap),np.array(refaxis_1[iiaero_node,:]))
                                        disflap_vec0[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_pos_indflap),np.array(refvind0))
                                        disflap_vec1[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_pos_indflap),np.array(refvind1))
                                        dispflap_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_vel_indflap),np.array(refvind0))
                                        dispflap_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_vel_indflap),np.array(refvind1))
                                        disppflap_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_acc_indflap),np.array(refvind0))
                                        disppflap_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_acc_indflap),np.array(refvind1))
                                        dispppflap_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_cel_indflap),np.array(refvind0))
                                        dispppflap_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_cel_indflap),np.array(refvind1))
                                        aoapflap_vec0[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_angpflap),np.array(refaxis_0[iiaero_node,:]))+np.arctan(np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_acc_indflap),np.array(refvind0))/Vinf_vindaoa)
                                        aoapflap_vec1[iiaero_node]   = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_angpflap),np.array(refaxis_1[iiaero_node,:]))+np.arctan(np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_acc_indflap),np.array(refvind1))/Vinf_vindaoa)
                                        aoappflap_vec0[iiaero_node]  = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_angppflap),np.array(refaxis_0[iiaero_node,:]))+np.arctan(np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],vec_cel_indflap),np.array(refvind0))/Vinf_vindaoa)
                                        aoappflap_vec1[iiaero_node]  = np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_angppflap),np.array(refaxis_1[iiaero_node,:]))+np.arctan(np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],vec_cel_indflap),np.array(refvind1))/Vinf_vindaoa)
                                        aoapflap_val[iiaero_node]    = (aoapflap_vec0[iiaero_node]*LL_vec_p[0,iimark_nodeflap]+aoapflap_vec1[iiaero_node]*LL_vec_p[1,iimark_nodeflap])/(LL_vec_p[0,iimark_nodeflap]+LL_vec_p[1,iimark_nodeflap])
                                        aoappflap_val[iiaero_node]   = (aoappflap_vec0[iiaero_node]*LL_vec_p[0,iimark_nodeflap]+aoappflap_vec1[iiaero_node]*LL_vec_p[1,iimark_nodeflap])/(LL_vec_p[0,iimark_nodeflap]+LL_vec_p[1,iimark_nodeflap])
                                        if markflap.node.flatten()[iiaero_node] >= 0:
                                            deltaflap_vec0[iiaero_node]    = (aoa0_geomflap[iiaero_node]-aoa0_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_nodeflap,0,:,:],v_vert),np.array(refvind0))/Vinf_vindaoa)+angflap_vec0[iiaero_node]+polar_bc.quasi_steady*np.arctan(dispflap_vec0[iiaero_node]/Vinf_vindaoa)
                                            deltaflap_vec1[iiaero_node]    = (aoa1_geomflap[iiaero_node]-aoa1_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_nodeflap,1,:,:],v_vert),np.array(refvind1))/Vinf_vindaoa)+angflap_vec1[iiaero_node]+polar_bc.quasi_steady*np.arctan(dispflap_vec1[iiaero_node]/Vinf_vindaoa)
                                            deltaflap[iiaero_node]     = (deltaflap_vec0[iiaero_node]*LL_vec_p[0,iimark_nodeflap]+deltaflap_vec1[iiaero_node]*LL_vec_p[1,iimark_nodeflap])/(LL_vec_p[0,iimark_nodeflap]+LL_vec_p[1,iimark_nodeflap])
                                        else:
                                            deltaflap_vec0[iiaero_node]    = 0
                                            deltaflap_vec1[iiaero_node]    = 0
                                            deltaflap[iiaero_node]     = 0
                                pos_vec0[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_pos_ind),np.array(refvind0))
                                pos_vec1[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_pos_ind),np.array(refvind1))
                                reynolds0[iiaero_node] = Vinf0[iiaero_node]*2*b_chord[iiaero_node]*bound.rho/bound.mu
                                reynolds1[iiaero_node] = Vinf1[iiaero_node]*2*b_chord[iiaero_node]*bound.rho/bound.mu
                                aoa_vec0_st[iiaero_node] = aoa0_geom[iiaero_node]+ang_vec0[iiaero_node]
                                aoa_vec1_st[iiaero_node] = aoa1_geom[iiaero_node]+ang_vec1[iiaero_node]
                                aoa_totvec0[iimark_node] = aoa_vec0[iiaero_node]
                                aoa_totvec1[iimark_node] = aoa_vec1[iiaero_node]
                                ang_totvec0[iimark_node] = ang_vec0[iiaero_node]
                                ang_totvec1[iimark_node] = ang_vec1[iiaero_node]
                                angp_totvec0[iimark_node] = angp_vec0[iiaero_node]
                                angp_totvec1[iimark_node] = angp_vec1[iiaero_node]
                                angpp_totvec0[iimark_node] = angpp_vec0[iiaero_node]
                                angpp_totvec1[iimark_node] = angpp_vec1[iiaero_node]
                                dis_totvec0[iimark_node] = dis_vec0[iiaero_node]
                                dis_totvec1[iimark_node] = dis_vec1[iiaero_node]
                                disp_totvec0[iimark_node] = disp_vec0[iiaero_node]
                                disp_totvec1[iimark_node] = disp_vec1[iiaero_node]
                                dispp_totvec0[iimark_node] = dispp_vec0[iiaero_node]
                                dispp_totvec1[iimark_node] = dispp_vec1[iiaero_node]
                                disppp_totvec0[iimark_node] = disppp_vec0[iiaero_node]
                                disppp_totvec1[iimark_node] = disppp_vec1[iiaero_node]
                                # aoa_mean0 : mean angle of attack of node 0
                                # aoa_mean1 : mean angle of attack of node 1
                                # ang_mean0 : mean twist of the node 0
                                # ang_mean1 : mean twist of the node 1
                                # dis_mean0 : mean displacement of the node 0
                                # dis_mean1 : mean displacement of the node 1
                                # lenmean   : length of the array of angles of attack required for calculating the mean value
                                lenmean = np.min([ii_time+1,int_per])
                                if lenmean == int_per:
                                    aoa_mean0[iiaero_node] = aoa_storearray0[iiaero_node,-1-lenmean:].mean(0) 
                                    aoa_mean1[iiaero_node] = aoa_storearray1[iiaero_node,-1-lenmean:].mean(0) 
                                    ang_mean0[iiaero_node] = ang_storearray0[iiaero_node,-1-lenmean:].mean(0) 
                                    ang_mean1[iiaero_node] = ang_storearray1[iiaero_node,-1-lenmean:].mean(0) 
                                    dis_mean0[iiaero_node] = dis_storearray0[iiaero_node,-1-lenmean:].mean(0) 
                                    dis_mean1[iiaero_node] = dis_storearray1[iiaero_node,-1-lenmean:].mean(0)
                                else:
                                    if ii_time == 0:
                                        aoa_mean0[iiaero_node] = aoa_vec0[iiaero_node]
                                        aoa_mean1[iiaero_node] = aoa_vec1[iiaero_node] 
                                        ang_mean0[iiaero_node] = ang_vec0[iiaero_node]
                                        ang_mean1[iiaero_node] = ang_vec1[iiaero_node]
                                        dis_mean0[iiaero_node] = dis_vec0[iiaero_node]
                                        dis_mean1[iiaero_node] = dis_vec1[iiaero_node]
                                    else:
                                        aoa_mean0[iiaero_node] = aoa_storearray0[iiaero_node,1]
                                        aoa_mean1[iiaero_node] = aoa_storearray1[iiaero_node,1] 
                                        ang_mean0[iiaero_node] = ang_storearray0[iiaero_node,1]
                                        ang_mean1[iiaero_node] = ang_storearray1[iiaero_node,1]
                                        dis_mean0[iiaero_node] = dis_storearray0[iiaero_node,1]
                                        dis_mean1[iiaero_node] = dis_storearray1[iiaero_node,1]
                                if  bound.f_flap == 1:
                                    if lenmean == int_per*20:
                                        angflap_mean0[iiaero_node] = angflap_storearray0[iiaero_node,-1-lenmean:].mean(0) 
                                        angflap_mean1[iiaero_node] = angflap_storearray1[iiaero_node,-1-lenmean:].mean(0) 
                                        disflap_mean0[iiaero_node] = disflap_storearray0[iiaero_node,-1-lenmean:].mean(0) 
                                        disflap_mean1[iiaero_node] = disflap_storearray1[iiaero_node,-1-lenmean:].mean(0)
                                    else:
                                        if ii_time == 0:
                                            angflap_mean0[iiaero_node] = angflap_vec0[iiaero_node] 
                                            angflap_mean1[iiaero_node] = angflap_vec1[iiaero_node] 
                                            disflap_mean0[iiaero_node] = disflap_vec0[iiaero_node]
                                            disflap_mean1[iiaero_node] = disflap_vec1[iiaero_node]
                                        else:
                                            angflap_mean0[iiaero_node] = angflap_storearray0[iiaero_node,1] 
                                            angflap_mean1[iiaero_node] = angflap_storearray1[iiaero_node,1] 
                                            disflap_mean0[iiaero_node] = disflap_storearray0[iiaero_node,1] 
                                            disflap_mean1[iiaero_node] = disflap_storearray1[iiaero_node,1]
                            case_setup, polar_bc_ind,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,eff_3d,error,cl_alpha,cl_alphaflap,error2,v_induced = steady_aero(case_setup,bound,polar_bctot,mesh_data,marktot,polar_bctot_ind,aoa_vec0,aoa_vec1,aoa_vec0_st,aoa_vec1_st,pos_vec0,pos_vec1,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,aoa_0lift,aoa_0liftflap,v_induced,refaxis2_0,refaxis2_1,Vinf0,Vinf1,LL_vec_p,lpos,LL_cum,eff_3d,tol_vind,b_chord,non_acac_dist,b_chordflap,non_acac_distflap,coreflapdist_adim,aoa_mean0,aoa_mean1,cd_aoa0,cd_aoa0flap,aoa0_geom,aoa1_geom,reynolds0,reynolds1,deltaflap_vec0,deltaflap_vec1,bound.f_flap)
                            if time_val/case_setup.time_stp % vical != 0:
#                                print (ii_time)
                                error = tol_vind/10
#                            else:
#                                print('ERROR')
                            if eff_3d.notconverged == 1:
                                break
                        if polar_bc.eff3d == 'BEM':
                            eff_3d,error_out,vtan_induced = converge_vtan(bound,mark,polar_bc,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,v_induced,LL_vec_p,eff_3d,b_chord,vtan_induced)
                            if time_val/case_setup.time_stp % vical != 0:
                                    error_out = tol_vind/10
                        else:
                            error_out = tol_vind/10
                    case_setup,cldyn_value0,cldyn_value1,cddyn_value0,cddyn_value1,cmdyn_value0,cmdyn_value1 = trans_aero(case_setup,bound,mark,polar_bc,time_val,tmin_ann,aoa_mean0,aoa_mean1,ang_mean0,ang_mean1,dis_mean0,dis_mean1,aoa_vec0,aoa_vec1,aoap_vec0,aoap_vec1,aoapp_vec0,aoapp_vec1,deltaflap_vec0,deltaflap_vec1,ang_vec0,ang_vec1,dis_vec0,dis_vec1,aoa_storearray0,aoa_storearray1,ang_storearray0,ang_storearray1,angder_storearray0,angder_storearray1,angderder_storearray0,angderder_storearray1,dis_storearray0,dis_storearray1,disder_storearray0,disder_storearray1,disderder_storearray0,disderder_storearray1,cddyn_value0,cddyn_value1,cldyn_value0,cldyn_value1,cmdyn_value0,cmdyn_value1,polar_bc_ind,int_per,b_chord,cl_alpha,non_eamc_dist,non_eaac_dist,non_acac_dist,Vinf0,Vinf1,time_valvec)
                    if  bound.f_flap == 1  and markflap.node.flatten()[iiaero_node] >= 0:
                        case_setup,cldynflap_value0,cldynflap_value1,cddynflap_value0,cddynflap_value1,cmdynflap_value0,cmdynflap_value1 = trans_aero(case_setup,bound,markflap,polar_bcflap,time_val,tmin_ann,aoaflap_mean0,aoaflap_mean1,angflap_mean0,angflap_mean1,disflap_mean0,disflap_mean1,aoa_vec0,aoa_vec1,aoapflap_vec0,aoapflap_vec1,aoappflap_vec0,aoappflap_vec1,deltaflap_vec0,deltaflap_vec1,angflap_vec0,angflap_vec1,disflap_vec0,disflap_vec1,aoa_storearray0,aoa_storearray1,angflap_storearray0,angflap_storearray1,disflap_storearray0,disflap_storearray1,cddynflap_value0,cddynflap_value1,cldynflap_value0,cldynflap_value1,cmdynflap_value0,cmdynflap_value1,polar_bcflap_ind,int_per,b_chordflap,cl_alphaflap,non_eamc_distflap,non_eaac_distflap,non_acac_distflap,Vinf0,Vinf1,time_valvec)
                    FF_global,cl_value0,cl_value1,cm_value0,cm_value1,disp_values = total_forces(clst_value0,clst_value1,cmst_value0,cmst_value1,cdst_value0,cdst_value1,cldyn_value0,cldyn_value1,cmdyn_value0,cmdyn_value1,cddyn_value0,cddyn_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdstflap_value0,cdstflap_value1,cldynflap_value0,cldynflap_value1,cmdynflap_value0,cmdynflap_value1,cddynflap_value0,cddynflap_value1,LL_vec_p,rot_mat_p,b_chord,b_chordflap,lpos,Vinf0,Vinf1,vdir0,vdir1,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,FF_global,bound,section,polar_bc,marktot,disp_values,num_point,time_val,tmin_ann,ttran_ann,ii_time,eff_3d,bound.f_flap)
                    # save values of the aerodynamic parameters
                    disp_values.aoa_val   = aoa_val
                    disp_values.aoap_val  = aoap_val
                    disp_values.aoapp_val = aoapp_val
                    disp_values.v_induced = v_induced.copy()
                    if polar_bc.eff3d == 'BEM':
                        disp_values.vtan_induced = vtan_induced.copy()
                    polar_bc_ind         += 1
                    if bound.f_flap == 1:
                        disp_values.angflap_store0.append(angflap_vec0)
                        disp_values.angflap_store1.append(angflap_vec1)
                        disp_values.disflap_store0.append(disflap_vec0)
                        disp_values.disflap_store1.append(disflap_vec1)
                    FF_orig = FF_global.copy()                   
                    bound.vinf = vinf_copy
                    bound.vrot = vrot_copy
                    polar_bc.startbem = 0
                elif bound.type == "BC_DISP":
                    # If bc is a displacement in a node, add the value to the global displacement vector
                    bound,func,funcder,funcderder = time_func(bound,time_val)
                    if ii_time > 0:
                        for node in mark.node:
                            if node<num_point:
                                if all(np.isnan(qq_global[int(9*node):int(9*node+9),ii_time])):
                                    qq_global[int(9*node):int(9*node+9),ii_time] = func*np.array(bound.values)
                                    qq_der_global[int(9*node):int(9*node+9),ii_time] = funcder*np.array(bound.values)
                                    qq_derder_global[int(9*node):int(9*node+9),ii_time] = funcderder*np.array(bound.values)
                                else:
                                    qq_global[int(9*node):int(9*node+9),ii_time] += func*np.array(bound.values)
                                    qq_der_global[int(9*node):int(9*node+9),ii_time] += funcder*np.array(bound.values)
                                    qq_derder_global[int(9*node):int(9*node+9),ii_time] += funcderder*np.array(bound.values)
                            else:
                                if all(np.isnan(qq_global[int(9*node):int(9*node+9),ii_time])):
                                    qq_global[int(9*node):,ii_time] = func*bound.values
                                    qq_der_global[int(9*node):,ii_time] = funcder*np.array(bound.values)
                                    qq_derder_global[int(9*node):,ii_time] = funcderder*np.array(bound.values)
                                else:
                                    qq_global[int(9*node):,ii_time] += func*bound.values
                                    qq_der_global[int(9*node):,ii_time] += funcder*np.array(bound.values)
                                    qq_derder_global[int(9*node):,ii_time] += funcderder*np.array(bound.values)
                elif bound.type == "BC_FUNC":
                    # If the bc is a function of the vibration modes
                    try:
                        # mode_bc      : vibration mode used for the load
                        # vibfree      : information of the free vibration analysis
                        # sec_coor_vib : information of the section points used in the vibration analysis
                        # func_bc      : function used for the bc
                        mode_bc               = int(bound.funcmode)
                        vibfree, sec_coor_vib = solver_vib_mod(case_setup,sol_phys,mesh_data,section)
                        func_bc               = np.zeros((len(mark.node),9))
                        # for every node in the marker determine the function
                        for node in mark.node:
                            node = int(node)
                            # The function is calculated as the projection in the bc values_load 
                            # vector of the modal shape
                            func_bc[node,:] = np.dot([vibfree.u[node,mode_bc],vibfree.v[node,mode_bc],vibfree.w[node,mode_bc],\
                                       vibfree.phi[node,mode_bc],vibfree.psi[node,mode_bc],vibfree.theta[node,mode_bc],\
                                       vibfree.phi_d[node,mode_bc],vibfree.psi_d[node,mode_bc],vibfree.theta_d[node,mode_bc]],bound.values_load)
                        # For every mesh marker if the marker name is the normalization function 
                        # name divide the value by the value in the normalization node and multiplied 
                        # by the value specified in the setup
                        for mark2 in mesh_mark:
                            if mark2.name == bound.funcnorm:
                                for aux_mark2 in np.arange(9):
                                    if func_bc[int(mark2.node[0]),aux_mark2] != 0:
                                        func_bc /= func_bc[int(mark2.node[0]),aux_mark2]*np.array(bound.values)
                    except:
                        pass
                    # The values calculated preivously are updated to the forces vector         
                    for elem in mesh_data.elem:
                        if len(np.where(mark.node==elem[1])[0])>0:
                            FF_global[int(9*elem[1]):int(9*elem[1]+9),ii_time] += func_bc[int(elem[1])]*np.linalg.norm(mesh_data.point[int(elem[1]),:]-mesh_data.point[int(elem[2]),:])/2
                        if len(np.where(mark.node==elem[2])[0])>0:
                            FF_global[int(9*elem[2]):int(9*elem[2]+9),ii_time] += func_bc[int(elem[2])]*np.linalg.norm(mesh_data.point[int(elem[1]),:]-mesh_data.point[int(elem[2]),:])/2
                elif bound.type == "BC_JOINT":
                    # If the bc is a joint between nodes
                    # node_1 : node 1 of the joint
                    # node_2 : node 2 of the joint
                    node_1 = int(mark.node[0])
                    node_2 = int(mark.node[1])
                    # xvec : distance between nodes in axis x
                    # yvec : distance between nodes in axis y
                    # zvec : distance between nodes in axis z
                    xvec = mesh_data.point[node_2,0]-mesh_data.point[node_1,0]
                    yvec = mesh_data.point[node_2,1]-mesh_data.point[node_1,1]
                    zvec = mesh_data.point[node_2,2]-mesh_data.point[node_1,2]
                    if bound.joint_type == "FIXED":
                        # If the boundary condition is a rigid solid connection between bodies
                        for ii_fixed in np.arange(9):
                            # for the 9 rows of the node 1
                            # add the values of node 2 in stiffness and loads
                            FF_global[node_1*9+ii_fixed,ii_time] += FF_global[node_2*9+ii_fixed,ii_time]
                            # The moment added by the distance between reference points
                            if ii_fixed == 0:
                                # moment added by force in x axis
                                FF_global[node_1*9+4,ii_time] += zvec*FF_global[node_2*9,ii_time]
                                FF_global[node_1*9+5,ii_time] += -yvec*FF_global[node_2*9,ii_time]
                            elif ii_fixed == 1:
                                # moment added by force in y axis
                                FF_global[node_1*9+3,ii_time] += -zvec*FF_global[node_2*9+1,ii_time]
                                FF_global[node_1*9+5,ii_time] += xvec*FF_global[node_2*9+1,ii_time]
                            elif ii_fixed == 2:
                                # moment added by force in z axis
                                FF_global[node_1*9+3,ii_time] += yvec*FF_global[node_2*9+2,ii_time]
                                FF_global[node_1*9+4,ii_time] += -xvec*FF_global[node_2*9+2,ii_time]
                        # The rows of the node 2 must be deleted and the relation between displacements is added
                        FF_global[node_2*9:node_2*9+9,ii_time] = np.zeros((9,))
                    elif bound.joint_type == "ROTATE_AXIS":
                        a_axis = np.array(bound.joint_axis)/np.linalg.norm(np.array(bound.joint_axis))
                        if np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])>0:
                            r_axis = (mesh_data.point[node_2]-mesh_data.point[node_1])/np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])
                        else:
                            if a_axis[0] == 0:
                                r_axis = np.array([1,0,0])
                            elif a_axis[1] == 0:
                                r_axis = np.array([0,1,0])
                            elif a_axis[2] == 0:
                                r_axis = np.array([0,0,1])
                            else:
                                r_axis = np.array([a_axis[1],-a_axis[0],a_axis[2]])/np.linalg.norm(np.array([a_axis[1],-a_axis[0],a_axis[2]]))
                        r_0_2 = mesh_data.point[node_2]-bound.point_axis
                        r_0_1 = mesh_data.point[node_1]-bound.point_axis
                        x1 = r_0_1[0]
                        y1 = r_0_1[1]
                        x2 = r_0_2[0]
                        y2 = r_0_2[1]
                        n_axis = np.cross(a_axis,r_axis)
                        # rot_mat_axis : Rotation matrix of the degree of freedom. From rotation axis to global 
                        rot_mat_axis = np.array([[np.dot(r_axis,[1,0,0]),np.dot(n_axis,[1,0,0]),np.dot(a_axis,[1,0,0])],
                                                  [np.dot(r_axis,[0,1,0]),np.dot(n_axis,[0,1,0]),np.dot(a_axis,[0,1,0])],
                                                  [np.dot(r_axis,[0,0,1]),np.dot(n_axis,[0,0,1]),np.dot(a_axis,[0,0,1])]])
                        # stiffness matrix and load vector in the nodes is rotated to the joint axis
                        for ii_point in np.arange(num_point):
                                for ii_row in np.arange(3):
                                    FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),ii_time] = np.matmul(rot_mat_axis,FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),ii_time])
                        # For all the rows of the points in the stiffness matrix and load vector
                        for ii_fixed in np.arange(9):
                            # Apply the restrictions in all the displacements except of the rotation around the axis
                            if ii_fixed != 5 and ii_fixed != 8:                                
                                FF_global[node_1*9+ii_fixed,ii_time]   += FF_global[node_2*9+ii_fixed,ii_time]
                                # The moment added by the distance between reference points
                                if ii_fixed == 0:
                                    # moment added by force in x axis
                                    FF_global[node_1*9+4,ii_time]   += zvec*FF_global[node_2*9,ii_time]
                                    FF_global[node_1*9+5,ii_time]   += -y1*FF_global[node_2*9,ii_time]
                                elif ii_fixed == 1:
                                    # moment added by force in y axis
                                    FF_global[node_1*9+3,ii_time] += -zvec*FF_global[node_2*9+1,ii_time]
                                    FF_global[node_1*9+5,ii_time] += -x1*FF_global[node_2*9+1,ii_time]
                                elif ii_fixed == 2:
                                    # moment added by force in z axis
                                    FF_global[node_1*9+3,ii_time]   += yvec*FF_global[node_2*9+2,ii_time]
                                    FF_global[node_1*9+4,ii_time]   += -xvec*FF_global[node_2*9+2,ii_time] 
                                FF_global[node_2*9+ii_fixed,ii_time]    = 0          
                                # The rows of the node 2 must be deleted and the relation between displacements is added
                        # The stiffness matrix and the load vector are rotated back to the global frame
                        for ii_point in np.arange(num_point):
                                for ii_row in np.arange(3):
                                    FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),ii_time] = np.matmul(np.transpose(rot_mat_axis),FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3),ii_time])
#                                    
#                                    #########
#                        # if joint is a rotation joint
#                        # a_axis : axis of rotation of the beam
#                        # r_axis : unitary vector of the distance between nodes
#                        # r_0_2  : distance from point of rotation to section 2
#                        # r_1_0  : distance from section 1 to point of rotation
#                        # n_axis : normal axis from rotation axis and nodes relative position
#                        # x1     : distance between node 1 and joint in axis x
#                        # y1     : distance between node 1 and joint in axis y
#                        # x2     : distance between joint and node 2 in axis x
#                        # y2     : distance between joint and node 2 in axis y
#                        a_axis = np.array(bound.joint_axis)/np.linalg.norm(np.array(bound.joint_axis))
#                        if np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])>0:
#                            r_axis = (mesh_data.point[node_2]-mesh_data.point[node_1])/np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])
#                        else:
#                            if a_axis[0] == 0:
#                                r_axis = np.array([1,0,0])
#                            elif a_axis[1] == 0:
#                                r_axis = np.array([0,1,0])
#                            elif a_axis[2] == 0:
#                                r_axis = np.array([0,0,1])
#                            else:
#                                r_axis = np.array([a_axis[1],-a_axis[0],a_axis[2]])/np.linalg.norm(np.array([a_axis[1],-a_axis[0],a_axis[2]]))
#                        r_0_2 = mesh_data.point[node_2]-bound.point_axis
#                        r_0_1 = mesh_data.point[node_1]-bound.point_axis
#                        x1 = r_0_1[0]
#                        y1 = r_0_1[1]
#                        x2 = r_0_2[0]
#                        y2 = r_0_2[1]
#                        n_axis = np.cross(a_axis,r_axis)
#                        # rot_mat_axis : Rotation matrix of the degree of freedom. From rotation axis to global 
#                        rot_mat_axis = np.array([[np.dot(r_axis,[1,0,0]),np.dot(n_axis,[1,0,0]),np.dot(a_axis,[1,0,0])],
#                                                  [np.dot(r_axis,[0,1,0]),np.dot(n_axis,[0,1,0]),np.dot(a_axis,[0,1,0])],
#                                                  [np.dot(r_axis,[0,0,1]),np.dot(n_axis,[0,0,1]),np.dot(a_axis,[0,0,1])]])
#                        # stiffness matrix and load vector in the nodes is rotated to the joint axis
#                        for ii_row in np.arange(3):
#                            FF_global[node_1*9+3*ii_row:node_1*9+(3*ii_row+3),ii_time] = np.matmul(rot_mat_axis,FF_global[node_1*9+3*ii_row:node_1*9+(3*ii_row+3),ii_time])
#                            FF_global[node_2*9+3*ii_row:node_2*9+(3*ii_row+3),ii_time] = np.matmul(rot_mat_axis,FF_global[node_2*9+3*ii_row:node_2*9+(3*ii_row+3),ii_time])
#                        # For all the rows of the points in the stiffness matrix and load vector
#                        for ii_fixed in np.arange(9):
#                            # Apply the restrictions in all the displacements except of the rotation around the axis
#                            if ii_fixed != 5 and ii_fixed != 8:                                
#                                FF_global[node_1*9+ii_fixed,ii_time]   += FF_global[node_2*9+ii_fixed,ii_time]
#                                # The moment added by the distance between reference points
#                                if ii_fixed == 0:
#                                    # moment added by force in x axis
#                                    FF_global[node_1*9+4,ii_time]   += zvec*FF_global[node_2*9,ii_time]
#                                    FF_global[node_1*9+5,ii_time]   += -y1*FF_global[node_2*9,ii_time]
#                                elif ii_fixed == 1:
#                                    # moment added by force in y axis
#                                    FF_global[node_1*9+3,ii_time] += -zvec*FF_global[node_2*9+1,ii_time]
#                                    FF_global[node_1*9+5,ii_time] += -x1*FF_global[node_2*9+1,ii_time]
#                                elif ii_fixed == 2:
#                                    # moment added by force in z axis
#                                    FF_global[node_1*9+3,ii_time]   += yvec*FF_global[node_2*9+2,ii_time]
#                                    FF_global[node_1*9+4,ii_time]   += -xvec*FF_global[node_2*9+2,ii_time]  
#                                FF_global[node_2*9+ii_fixed,ii_time]    = 0         
#                                # The rows of the node 2 must be deleted and the relation between displacements is added
#                        # The stiffness matrix and the load vector are rotated back to the global frame
#                        for ii_row in np.arange(3):
#                            FF_global[node_1*9+3*ii_row:node_1*9+(3*ii_row+3),ii_time] = np.matmul(np.transpose(rot_mat_axis),FF_global[node_1*9+3*ii_row:node_1*9+(3*ii_row+3),ii_time])
#                            FF_global[node_2*9+3*ii_row:node_2*9+(3*ii_row+3),ii_time] = np.matmul(np.transpose(rot_mat_axis),FF_global[node_2*9+3*ii_row:node_2*9+(3*ii_row+3),ii_time]) 
    if flag_aerodyn == 1:
        disp_values.aoa_store0.append(aoa_totvec0)
        disp_values.aoa_store1.append(aoa_totvec1)
        disp_values.ang_store0.append(ang_totvec0)
        disp_values.ang_store1.append(ang_totvec1)
        disp_values.angder_store0.append(angp_totvec0)
        disp_values.angder_store1.append(angp_totvec1)
        disp_values.angderder_store0.append(angpp_totvec0)
        disp_values.angderder_store1.append(angpp_totvec1)
        disp_values.dis_store0.append(dis_totvec0)
        disp_values.dis_store1.append(dis_totvec1)
        disp_values.disder_store0.append(disp_totvec0)
        disp_values.disder_store1.append(disp_totvec1)
        disp_values.disderder_store0.append(dispp_totvec0)
        disp_values.disderder_store1.append(dispp_totvec1)
    return qq_global, qq_der_global, qq_derder_global, FF_global, FF_orig, disp_values, case_setup, ref_axis, mesh_data

#%%
def boundaryconditions_vibrest(mesh_mark,case_setup,num_point,ind_RR,q_RR):
    # Function to add the restrictions to the vibration analysis
    # mesh_mark  : markers of the mesh
    # case_setup : information about the case setup
    # num_point  : number of points of the boundary conditions
    # ind_RR     : index of the restricted nodes
    # q_RR       : displacement of the restricted nodes
    # -------------------------------------------------------------------------
    # find the boundary conditions that mach the marker
    for mark in mesh_mark:
        for bound in case_setup.boundary:
            if mark.name == bound.marker:
                    if bound.type == "BC_DISP":
                        # if the displacement condition is applied
                        for node in mark.node:
                            # index : indices of the dof related with the node
                            index = np.arange(int(9*node),int(9*node+9))
                            for ii_listdisp in np.arange(9):
                                e_list = index[ii_listdisp]
                                e_bval = bound.values[ii_listdisp]
                                if ~np.isnan(e_bval):
                                    ind_RR.append(e_list)
                                    q_RR.append(e_bval) 
    return ind_RR, q_RR

#%%
def mod_aero(case_setup,bound,mark,polar_bc,wfreq,polar_bc_ind,b_chord,cl_alpha,non_eamc_dist,non_eaac_dist,non_acac_dist,Vinf0,Vinf1):
    b_chord = b_chord.copy()
    cl_alpha = cl_alpha.copy()
    non_eamc_dist = non_eamc_dist.copy()
    non_eaac_dist = non_eaac_dist.copy()
    non_acac_dist = non_acac_dist.copy()
    Vinf0 = Vinf0.copy()
    Vinf1 = Vinf1.copy()
    # For every node in the lifting surface
    class clfreq0_a:
        A1_hfreq = np.zeros((len(mark.node),))
        A2_hfreq = np.zeros((len(mark.node),))
        A3_hfreq = np.zeros((len(mark.node),))
        A1_thfreq = np.zeros((len(mark.node),))
        A2_thfreq = np.zeros((len(mark.node),))
        A3_thfreq = np.zeros((len(mark.node),))
    class clfreq0_b:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class clfreq0:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class clfreq1_a:
        A1_hfreq = np.zeros((len(mark.node),))
        A2_hfreq = np.zeros((len(mark.node),))
        A3_hfreq = np.zeros((len(mark.node),))
        A1_thfreq = np.zeros((len(mark.node),))
        A2_thfreq = np.zeros((len(mark.node),))
        A3_thfreq = np.zeros((len(mark.node),))
    class clfreq1_b:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class clfreq1:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class cmfreq0_a:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class cmfreq0_b:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class cmfreq0:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class cmfreq1_a:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class cmfreq1_b:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    class cmfreq1:
        A1_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_hfreq = np.zeros((len(mark.node),), dtype=complex)
        A1_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A2_thfreq = np.zeros((len(mark.node),), dtype=complex)
        A3_thfreq = np.zeros((len(mark.node),), dtype=complex)
    for iiaero_node in np.arange(len(mark.node)):
        if b_chord[iiaero_node] > 0:
            iimark_node = int(mark.node.flatten()[iiaero_node])
            k_red0 = 100*b_chord[iiaero_node]/Vinf0[iiaero_node] #abs(wfreq)*b_chord[iiaero_node]/Vinf0[iiaero_node] #
            k_red1 = 100*b_chord[iiaero_node]/Vinf1[iiaero_node] #abs(wfreq)*b_chord[iiaero_node]/Vinf1[iiaero_node] #
            C_theo0   =  theodorsen([k_red0])[0]
            C_theo1   =   theodorsen([k_red1])[0]
            clfreq0_a.A1_hfreq[iiaero_node] = 0 
            clfreq0_a.A2_hfreq[iiaero_node] = 0
            clfreq0_a.A3_hfreq[iiaero_node] =  np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])
            clfreq0_a.A1_thfreq[iiaero_node] = 0
            clfreq0_a.A2_thfreq[iiaero_node] =   np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])*Vinf0[iiaero_node]
            clfreq0_a.A3_thfreq[iiaero_node] =   -np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])*b_chord[iiaero_node]*non_eamc_dist[iiaero_node]
            clfreq1_a.A1_hfreq[iiaero_node] = 0 
            clfreq1_a.A2_hfreq[iiaero_node] = 0
            clfreq1_a.A3_hfreq[iiaero_node] =    np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])
            clfreq1_a.A1_thfreq[iiaero_node] = 0
            clfreq1_a.A2_thfreq[iiaero_node] =   np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])*Vinf1[iiaero_node]
            clfreq1_a.A3_thfreq[iiaero_node] =   -np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])*b_chord[iiaero_node]*non_eamc_dist[iiaero_node]
            clfreq0_b.A1_hfreq[iiaero_node] = 0
            clfreq0_b.A2_hfreq[iiaero_node] = 2*np.pi*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]*C_theo0/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])
            clfreq0_b.A3_hfreq[iiaero_node] = 0
            clfreq0_b.A1_thfreq[iiaero_node] = 2*np.pi*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]*C_theo0/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])*Vinf0[iiaero_node]
            clfreq0_b.A2_thfreq[iiaero_node] =  2*np.pi*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]*C_theo0/(1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node])*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])
            clfreq0_b.A3_thfreq[iiaero_node] = 0
            clfreq1_b.A1_hfreq[iiaero_node] = 0
            clfreq1_b.A2_hfreq[iiaero_node] = 2*np.pi*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]*C_theo1/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])
            clfreq1_b.A3_hfreq[iiaero_node] = 0
            clfreq1_b.A1_thfreq[iiaero_node] = 2*np.pi*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]*C_theo1/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])*Vinf1[iiaero_node]
            clfreq1_b.A2_thfreq[iiaero_node] =   2*np.pi*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]*C_theo1/(1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node])*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])
            clfreq1_b.A3_thfreq[iiaero_node] = 0
            clfreq0.A1_hfreq[iiaero_node] = (clfreq0_a.A1_hfreq[iiaero_node]+clfreq0_b.A1_hfreq[iiaero_node])
            clfreq0.A2_hfreq[iiaero_node] = (clfreq0_a.A2_hfreq[iiaero_node]+clfreq0_b.A2_hfreq[iiaero_node])
            clfreq0.A3_hfreq[iiaero_node] = (clfreq0_a.A3_hfreq[iiaero_node]+clfreq0_b.A3_hfreq[iiaero_node])
            clfreq0.A1_thfreq[iiaero_node] = (clfreq0_a.A1_thfreq[iiaero_node]+clfreq0_b.A1_thfreq[iiaero_node])
            clfreq0.A2_thfreq[iiaero_node] = (clfreq0_a.A2_thfreq[iiaero_node]+clfreq0_b.A2_thfreq[iiaero_node])
            clfreq0.A3_thfreq[iiaero_node] = (clfreq0_a.A3_thfreq[iiaero_node]+clfreq0_b.A3_thfreq[iiaero_node])
            clfreq1.A1_hfreq[iiaero_node] = (clfreq1_a.A1_hfreq[iiaero_node]+clfreq1_b.A1_hfreq[iiaero_node])
            clfreq1.A2_hfreq[iiaero_node] = (clfreq1_a.A2_hfreq[iiaero_node]+clfreq1_b.A2_hfreq[iiaero_node])
            clfreq1.A3_hfreq[iiaero_node] = (clfreq1_a.A3_hfreq[iiaero_node]+clfreq1_b.A3_hfreq[iiaero_node])
            clfreq1.A1_thfreq[iiaero_node] = (clfreq1_a.A1_thfreq[iiaero_node]+clfreq1_b.A1_thfreq[iiaero_node])
            clfreq1.A2_thfreq[iiaero_node] = (clfreq1_a.A2_thfreq[iiaero_node]+clfreq1_b.A2_thfreq[iiaero_node])
            clfreq1.A3_thfreq[iiaero_node] = (clfreq1_a.A3_thfreq[iiaero_node]+clfreq1_b.A3_thfreq[iiaero_node])
            cmfreq0_a.A1_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_a.A1_hfreq[iiaero_node]
            cmfreq0_a.A2_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_a.A2_hfreq[iiaero_node]
            cmfreq0_a.A3_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_a.A3_hfreq[iiaero_node]+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*b_chord[iiaero_node]*non_eamc_dist[iiaero_node]
            cmfreq0_a.A1_thfreq[iiaero_node] =   non_eaac_dist[iiaero_node]*clfreq0_a.A1_thfreq[iiaero_node]
            cmfreq0_a.A2_thfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_a.A2_thfreq[iiaero_node]+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(-Vinf0[iiaero_node]*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node]))
            cmfreq0_a.A3_thfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_a.A3_thfreq[iiaero_node]+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(-b_chord[iiaero_node]**2*(1/8+non_eamc_dist[iiaero_node]**2))
            cmfreq1_a.A1_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq1_a.A1_hfreq[iiaero_node]
            cmfreq1_a.A2_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq1_a.A2_hfreq[iiaero_node]
            cmfreq1_a.A3_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq1_a.A3_hfreq[iiaero_node]+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*b_chord[iiaero_node]*non_eamc_dist[iiaero_node]
            cmfreq1_a.A1_thfreq[iiaero_node] =   non_eaac_dist[iiaero_node]*clfreq1_a.A1_thfreq[iiaero_node]
            cmfreq1_a.A2_thfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq1_a.A2_thfreq[iiaero_node]+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(-Vinf0[iiaero_node]*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node]))
            cmfreq1_a.A3_thfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq1_a.A3_thfreq[iiaero_node]+np.pi*bound.rho*b_chord[iiaero_node]**2/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*(-b_chord[iiaero_node]**2*(1/8+non_eamc_dist[iiaero_node]**2))
           
            cmfreq0_b.A1_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_b.A1_hfreq[iiaero_node]
            cmfreq0_b.A2_hfreq[iiaero_node] = (non_eaac_dist[iiaero_node]*clfreq0_b.A2_hfreq[iiaero_node]+2*np.pi*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])*C_theo0/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2))
            cmfreq0_b.A3_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_b.A3_hfreq[iiaero_node]
            cmfreq0_b.A1_thfreq[iiaero_node] = (non_eaac_dist[iiaero_node]*clfreq0_b.A1_thfreq[iiaero_node]+2*np.pi*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])*C_theo0/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*Vinf0[iiaero_node])
            cmfreq0_b.A2_thfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_b.A2_thfreq[iiaero_node]+2*np.pi*bound.rho*Vinf0[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])*C_theo0/(1/2*bound.rho*Vinf0[iiaero_node]**2*4*b_chord[iiaero_node]**2)*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])
            cmfreq0_b.A3_thfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq0_b.A3_thfreq[iiaero_node]
            cmfreq1_b.A1_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq1_b.A1_hfreq[iiaero_node]
            cmfreq1_b.A2_hfreq[iiaero_node] = (non_eaac_dist[iiaero_node]*clfreq1_b.A2_hfreq[iiaero_node]+2*np.pi*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])*C_theo1/(1/2*bound.rho*Vinf1[iiaero_node]**2*4*b_chord[iiaero_node]**2))
            cmfreq1_b.A3_hfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq1_b.A3_hfreq[iiaero_node]
            cmfreq1_b.A1_thfreq[iiaero_node] = (non_eaac_dist[iiaero_node]*clfreq1_b.A1_thfreq[iiaero_node]+2*np.pi*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])*C_theo1/(1/2*bound.rho*Vinf1[iiaero_node]**2*4*b_chord[iiaero_node]**2)*Vinf1[iiaero_node])
            cmfreq1_b.A2_thfreq[iiaero_node] =  non_eaac_dist[iiaero_node]*clfreq1_b.A2_thfreq[iiaero_node]+2*np.pi*bound.rho*Vinf1[iiaero_node]*b_chord[iiaero_node]**2*(1/2+non_eamc_dist[iiaero_node])*C_theo1/(1/2*bound.rho*Vinf1[iiaero_node]**2*4*b_chord[iiaero_node]**2)*b_chord[iiaero_node]*(1/2-non_eamc_dist[iiaero_node])
            cmfreq1_b.A3_thfreq[iiaero_node] = non_eaac_dist[iiaero_node]*clfreq1_b.A3_thfreq[iiaero_node]
            cmfreq0.A1_hfreq[iiaero_node] = (cmfreq0_a.A1_hfreq[iiaero_node]+cmfreq0_b.A1_hfreq[iiaero_node])
            cmfreq0.A2_hfreq[iiaero_node] = (cmfreq0_a.A2_hfreq[iiaero_node]+cmfreq0_b.A2_hfreq[iiaero_node])
            cmfreq0.A3_hfreq[iiaero_node] = (cmfreq0_a.A3_hfreq[iiaero_node]+cmfreq0_b.A3_hfreq[iiaero_node])
            cmfreq0.A1_thfreq[iiaero_node] = (cmfreq0_a.A1_thfreq[iiaero_node]+cmfreq0_b.A1_thfreq[iiaero_node])
            cmfreq0.A2_thfreq[iiaero_node] = (cmfreq0_a.A2_thfreq[iiaero_node]+cmfreq0_b.A2_thfreq[iiaero_node])
            cmfreq0.A3_thfreq[iiaero_node] = (cmfreq0_a.A3_thfreq[iiaero_node]+cmfreq0_b.A3_thfreq[iiaero_node])
            cmfreq1.A1_hfreq[iiaero_node] = (cmfreq1_a.A1_hfreq[iiaero_node]+cmfreq1_b.A1_hfreq[iiaero_node])
            cmfreq1.A2_hfreq[iiaero_node] = (cmfreq1_a.A2_hfreq[iiaero_node]+cmfreq1_b.A2_hfreq[iiaero_node])
            cmfreq1.A3_hfreq[iiaero_node] = (cmfreq1_a.A3_hfreq[iiaero_node]+cmfreq1_b.A3_hfreq[iiaero_node])
            cmfreq1.A1_thfreq[iiaero_node] = (cmfreq1_a.A1_thfreq[iiaero_node]+cmfreq1_b.A1_thfreq[iiaero_node])
            cmfreq1.A2_thfreq[iiaero_node] = (cmfreq1_a.A2_thfreq[iiaero_node]+cmfreq1_b.A2_thfreq[iiaero_node])
            cmfreq1.A3_thfreq[iiaero_node] = (cmfreq1_a.A3_thfreq[iiaero_node]+cmfreq1_b.A3_thfreq[iiaero_node])
#            if Vinf0[iiaero_node] > 4.5 and Vinf0[iiaero_node]<5.5:
#                print(1)
    return clfreq0,clfreq1,cmfreq0,cmfreq1
#%%
def total_mat_mod(clfreq0,clfreq1,cmfreq0,cmfreq1,clfreq0flap,clfreq1flap,cmfreq0flap,cmfreq1flap,LL_vec_p,rot_mat_p,b_chord,b_chordflap,Vinf0,Vinf1,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,KK_global_prima,CC_global_prima,MM_global_prima,bound,section,marktot,num_point):
    if len(marktot) == 1:
        mark = marktot[0]
    else:
        mark = marktot[0]
        markflap = marktot[1]
    KK_global_ael = 0*KK_global_prima.copy()
    CC_global_ael = 0*CC_global_prima.copy()
    MM_global_ael = 0*MM_global_prima.copy()
    # For each node in the surface calculate the final load
    for iiaero_node in np.arange(len(mark.node)):
        iimark_node = int(mark.node.flatten()[iiaero_node])
        lift_A1_hfreq     = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0.A1_hfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:])
        lift_A1_hfreq    += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1.A1_hfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:]) 
        lift_A1_thfreq    = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0.A1_thfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
        lift_A1_thfreq   += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1.A1_thfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:]) 
        lift_A2_hfreq     = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0.A2_hfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
        lift_A2_hfreq    += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1.A2_hfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:])  
        lift_A2_thfreq    = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0.A2_thfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
        lift_A2_thfreq   += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1.A2_thfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:]) 
        lift_A3_hfreq     = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0.A3_hfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
        lift_A3_hfreq    += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1.A3_hfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:])  
        lift_A3_thfreq    = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0.A3_thfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
        lift_A3_thfreq   += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chord[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1.A3_thfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:])
        moment_A1_hfreq   = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0.A1_hfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
        moment_A1_hfreq  += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1.A1_hfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
        moment_A2_hfreq   = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0.A2_hfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
        moment_A2_hfreq  += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1.A2_hfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
        moment_A3_hfreq   = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0.A3_hfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
        moment_A3_hfreq  += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1.A3_hfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
        moment_A1_thfreq  = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0.A1_thfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
        moment_A1_thfreq += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1.A1_thfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
        moment_A2_thfreq  = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0.A2_thfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
        moment_A2_thfreq += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1.A2_thfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
        moment_A3_thfreq  = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0.A3_thfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
        moment_A3_thfreq += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chord[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1.A3_thfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:])
        r_aerovec = np.matmul(rot_mat_p[iimark_node,1,:,:],np.concatenate(([section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]],[0])))
        moment_A1_hfreq  += np.cross(r_aerovec,lift_A1_hfreq)
        moment_A2_hfreq  += np.cross(r_aerovec,lift_A2_hfreq)  
        moment_A3_hfreq  += np.cross(r_aerovec,lift_A3_hfreq) 
        moment_A1_thfreq += np.cross(r_aerovec,lift_A1_thfreq)
        moment_A2_thfreq += np.cross(r_aerovec,lift_A2_thfreq)
        moment_A3_thfreq += np.cross(r_aerovec,lift_A3_thfreq)
        cosxh = np.dot(-refaxis2_0[iiaero_node,:],[1,0,0])/(np.linalg.norm(refaxis2_0[iiaero_node,:]))
        cosyh = np.dot(-refaxis2_0[iiaero_node,:],[0,1,0])/(np.linalg.norm(refaxis2_0[iiaero_node,:]))
        coszh = np.dot(-refaxis2_0[iiaero_node,:],[0,0,1])/(np.linalg.norm(refaxis2_0[iiaero_node,:]))
        cosxth = np.dot(refaxis_0[iiaero_node,:],[1,0,0])/(np.linalg.norm(refaxis_0[iiaero_node,:]))
        cosyth = np.dot(refaxis_0[iiaero_node,:],[0,1,0])/(np.linalg.norm(refaxis_0[iiaero_node,:]))
        coszth = np.dot(refaxis_0[iiaero_node,:],[0,0,1])/(np.linalg.norm(refaxis_0[iiaero_node,:]))
        load_A1_force_x = [lift_A1_hfreq[0]*cosxh,lift_A1_hfreq[0]*cosyh,lift_A1_hfreq[0]*coszh,lift_A1_thfreq[0]*cosxth,lift_A1_thfreq[0]*cosyth,lift_A1_thfreq[0]*coszth,0,0,0] #load_A1_h_x    = [lift_A1_hfreq[0],0,0,moment_A1_hfreq[0],0,0,0,0,0]
        load_A1_force_y = [lift_A1_hfreq[1]*cosxh,lift_A1_hfreq[1]*cosyh,lift_A1_hfreq[1]*coszh,lift_A1_thfreq[1]*cosxth,lift_A1_thfreq[1]*cosyth,lift_A1_thfreq[1]*coszth,0,0,0] #load_A1_h_y    = [0,lift_A1_hfreq[1],0,0,moment_A1_hfreq[1],0,0,0,0]
        load_A1_force_z = [lift_A1_hfreq[2]*cosxh,lift_A1_hfreq[2]*cosyh,lift_A1_hfreq[2]*coszh,lift_A1_thfreq[2]*cosxth,lift_A1_thfreq[2]*cosyth,lift_A1_thfreq[2]*coszth,0,0,0] #load_A1_h_z    = [0,0,lift_A1_hfreq[2],0,0,moment_A1_hfreq[2],0,0,0] #np.concatenate((lift_A1_hfreq,moment_A1_hfreq,[0,0,0])) 
        load_A1_moment_x = [moment_A1_hfreq[0]*cosxh,moment_A1_hfreq[0]*cosyh,moment_A1_hfreq[0]*coszh,moment_A1_thfreq[0]*cosxth,moment_A1_thfreq[0]*cosyth,moment_A1_thfreq[0]*coszth,0,0,0] #load_A1_th_x   = [lift_A1_thfreq[0],0,0,moment_A1_thfreq[0],0,0,0,0,0]
        load_A1_moment_y = [moment_A1_hfreq[1]*cosxh,moment_A1_hfreq[1]*cosyh,moment_A1_hfreq[1]*coszh,moment_A1_thfreq[1]*cosxth,moment_A1_thfreq[1]*cosyth,moment_A1_thfreq[1]*coszth,0,0,0] #load_A1_th_y   = [0,lift_A1_thfreq[1],0,0,moment_A1_thfreq[1],0,0,0,0]
        load_A1_moment_z = [moment_A1_hfreq[2]*cosxh,moment_A1_hfreq[2]*cosyh,moment_A1_hfreq[2]*coszh,moment_A1_thfreq[2]*cosxth,moment_A1_thfreq[2]*cosyth,moment_A1_thfreq[2]*coszth,0,0,0] #load_A1_th_z   = [0,0,lift_A1_thfreq[2],0,0,moment_A1_thfreq[2],0,0,0]   #np.concatenate((lift_A1_thfreq,moment_A1_thfreq,[0,0,0]))
        load_A2_force_x = [lift_A2_hfreq[0]*cosxh,lift_A2_hfreq[0]*cosyh,lift_A2_hfreq[0]*coszh,lift_A2_thfreq[0]*cosxth,lift_A2_thfreq[0]*cosyth,lift_A2_thfreq[0]*coszth,0,0,0] #load_A2_h_x    = [lift_A2_hfreq[0],0,0,moment_A2_hfreq[0],0,0,0,0,0]
        load_A2_force_y = [lift_A2_hfreq[1]*cosxh,lift_A2_hfreq[1]*cosyh,lift_A2_hfreq[1]*coszh,lift_A2_thfreq[1]*cosxth,lift_A2_thfreq[1]*cosyth,lift_A2_thfreq[1]*coszth,0,0,0] #load_A2_h_y    = [0,lift_A2_hfreq[1],0,0,moment_A2_hfreq[1],0,0,0,0]
        load_A2_force_z = [lift_A2_hfreq[2]*cosxh,lift_A2_hfreq[2]*cosyh,lift_A2_hfreq[2]*coszh,lift_A2_thfreq[2]*cosxth,lift_A2_thfreq[2]*cosyth,lift_A2_thfreq[2]*coszth,0,0,0] #load_A2_h_z    = [0,0,lift_A2_hfreq[2],0,0,moment_A2_hfreq[2],0,0,0] 
        load_A2_moment_x = [moment_A2_hfreq[0]*cosxh,moment_A2_hfreq[0]*cosyh,moment_A2_hfreq[0]*coszh,moment_A2_thfreq[0]*cosxth,moment_A2_thfreq[0]*cosyth,moment_A2_thfreq[0]*coszth,0,0,0] #load_A2_th_x   = [lift_A2_thfreq[0],0,0,moment_A2_thfreq[0],0,0,0,0,0]
        load_A2_moment_y = [moment_A2_hfreq[1]*cosxh,moment_A2_hfreq[1]*cosyh,moment_A2_hfreq[1]*coszh,moment_A2_thfreq[1]*cosxth,moment_A2_thfreq[1]*cosyth,moment_A2_thfreq[1]*coszth,0,0,0] #load_A2_th_y   = [0,lift_A2_thfreq[1],0,0,moment_A2_thfreq[1],0,0,0,0]
        load_A2_moment_z = [moment_A2_hfreq[2]*cosxh,moment_A2_hfreq[2]*cosyh,moment_A2_hfreq[2]*coszh,moment_A2_thfreq[2]*cosxth,moment_A2_thfreq[2]*cosyth,moment_A2_thfreq[2]*coszth,0,0,0] #load_A2_th_z   = [0,0,lift_A2_thfreq[2],0,0,moment_A2_thfreq[2],0,0,0]
        load_A3_force_x = [lift_A3_hfreq[0]*cosxh,lift_A3_hfreq[0]*cosyh,lift_A3_hfreq[0]*coszh,lift_A3_thfreq[0]*cosxth,lift_A3_thfreq[0]*cosyth,lift_A3_thfreq[0]*coszth,0,0,0] #load_A3_h_x    = [lift_A3_hfreq[0],0,0,moment_A3_hfreq[0],0,0,0,0,0]
        load_A3_force_y = [lift_A3_hfreq[1]*cosxh,lift_A3_hfreq[1]*cosyh,lift_A3_hfreq[1]*coszh,lift_A3_thfreq[1]*cosxth,lift_A3_thfreq[1]*cosyth,lift_A3_thfreq[1]*coszth,0,0,0] #load_A3_h_y    = [0,lift_A3_hfreq[1],0,0,moment_A3_hfreq[1],0,0,0,0]
        load_A3_force_z = [lift_A3_hfreq[2]*cosxh,lift_A3_hfreq[2]*cosyh,lift_A3_hfreq[2]*coszh,lift_A3_thfreq[2]*cosxth,lift_A3_thfreq[2]*cosyth,lift_A3_thfreq[2]*coszth,0,0,0] #load_A3_h_z    = [0,0,lift_A3_hfreq[2],0,0,moment_A3_hfreq[2],0,0,0] 
        load_A3_moment_x = [moment_A3_hfreq[0]*cosxh,moment_A3_hfreq[0]*cosyh,moment_A3_hfreq[0]*coszh,moment_A3_thfreq[0]*cosxth,moment_A3_thfreq[0]*cosyth,moment_A3_thfreq[0]*coszth,0,0,0] #load_A3_th_x   = [lift_A3_thfreq[0],0,0,moment_A3_thfreq[0],0,0,0,0,0]
        load_A3_moment_y = [moment_A3_hfreq[1]*cosxh,moment_A3_hfreq[1]*cosyh,moment_A3_hfreq[1]*coszh,moment_A3_thfreq[1]*cosxth,moment_A3_thfreq[1]*cosyth,moment_A3_thfreq[1]*coszth,0,0,0] #load_A3_th_y   = [0,lift_A3_thfreq[1],0,0,moment_A3_thfreq[1],0,0,0,0]
        load_A3_moment_z = [moment_A3_hfreq[2]*cosxh,moment_A3_hfreq[2]*cosyh,moment_A3_hfreq[2]*coszh,moment_A3_thfreq[2]*cosxth,moment_A3_thfreq[2]*cosyth,moment_A3_thfreq[2]*coszth,0,0,0] #load_A3_th_z   = [0,0,lift_A3_thfreq[2],0,0,moment_A3_thfreq[2],0,0,0]
        # Check if the node is the final one
        for aux in np.arange(9):
            for aux2 in np.arange(9):
                if aux == 0:
                    KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A1_force_x[aux2]
                    CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A2_force_x[aux2]
                    MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A3_force_x[aux2]
                elif aux == 1:
                    KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A1_force_y[aux2]
                    CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A2_force_y[aux2]
                    MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A3_force_y[aux2]
                elif aux == 2:
                    KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A1_force_z[aux2]
                    CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A2_force_z[aux2]
                    MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A3_force_z[aux2]
                elif aux == 3:
                    KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A1_moment_x[aux2]
                    CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A2_moment_x[aux2]
                    MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A3_moment_x[aux2]
                elif aux == 4:
                    KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A1_moment_y[aux2]
                    CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A2_moment_y[aux2]
                    MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A3_moment_y[aux2]
                elif aux == 5:
                    KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A1_moment_z[aux2]
                    CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A2_moment_z[aux2]
                    MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= load_A3_moment_z[aux2]
        if bound.f_flap == 1:
            iimark_node = int(markflap.node.flatten()[iiaero_node])
            liftflap_A1_hfreq     = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0flap.A1_hfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:])
            liftflap_A1_hfreq    += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1.A1_hfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:]) 
            liftflap_A1_thfreq    = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0flap.A1_thfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
            liftflap_A1_thfreq   += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1flap.A1_thfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:]) 
            liftflap_A2_hfreq     = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0flap.A2_hfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
            liftflap_A2_hfreq    += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1flap.A2_hfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:])  
            liftflap_A2_thfreq    = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0flap.A2_thfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
            liftflap_A2_thfreq   += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1flap.A2_thfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:]) 
            liftflap_A3_hfreq     = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0flap.A3_hfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
            liftflap_A3_hfreq    += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1flap.A3_hfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:])  
            liftflap_A3_thfreq    = 1/2*bound.rho*Vinf0[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[0,iimark_node]*clfreq0flap.A3_thfreq[iiaero_node]*np.array(refaxis2_0[iiaero_node,:]) 
            liftflap_A3_thfreq   += 1/2*bound.rho*Vinf1[iiaero_node]**2*2*b_chordflap[iiaero_node]*LL_vec_p[1,iimark_node]*clfreq1flap.A3_thfreq[iiaero_node]*np.array(refaxis2_1[iiaero_node,:])
            momentflap_A1_hfreq   = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0flap.A1_hfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
            momentflap_A1_hfreq  += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq1flap.A1_hfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
            momentflap_A2_hfreq   = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0flap.A2_hfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
            momentflap_A2_hfreq  += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq1flap.A2_hfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
            momentflap_A3_hfreq   = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0flap.A3_hfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
            momentflap_A3_hfreq  += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq1flap.A3_hfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
            momentflap_A1_thfreq  = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0flap.A1_thfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
            momentflap_A1_thfreq += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1flap.A1_thfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
            momentflap_A2_thfreq  = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0flap.A2_thfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
            momentflap_A2_thfreq += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1flap.A2_thfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:]) 
            momentflap_A3_thfreq  = 1/2*bound.rho*Vinf0[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[0,iimark_node]*cmfreq0flap.A3_thfreq[iiaero_node]*np.array(refaxis_0[iiaero_node,:]) 
            momentflap_A3_thfreq += 1/2*bound.rho*Vinf1[iiaero_node]**2*(2*b_chordflap[iiaero_node])**2*LL_vec_p[1,iimark_node]*cmfreq1flap.A3_thfreq[iiaero_node]*np.array(refaxis_1[iiaero_node,:])
            r_aerovec = np.matmul(rot_mat_p[iimark_node,1,:,:],np.concatenate(([section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]],[0])))
            momentflap_A1_hfreq  += np.cross(r_aerovec,liftflap_A1_hfreq)
            momentflap_A2_hfreq  += np.cross(r_aerovec,liftflap_A2_hfreq)  
            momentflap_A3_hfreq  += np.cross(r_aerovec,liftflap_A3_hfreq) 
            momentflap_A1_thfreq += np.cross(r_aerovec,liftflap_A1_thfreq)
            momentflap_A2_thfreq += np.cross(r_aerovec,liftflap_A2_thfreq)
            momentflap_A3_thfreq += np.cross(r_aerovec,liftflap_A3_thfreq)
            cosxh = np.dot(-refaxis2_0[iiaero_node,:],[1,0,0])/(np.linalg.norm(refaxis2_0[iiaero_node,:]))
            cosyh = np.dot(-refaxis2_0[iiaero_node,:],[0,1,0])/(np.linalg.norm(refaxis2_0[iiaero_node,:]))
            coszh = np.dot(-refaxis2_0[iiaero_node,:],[0,0,1])/(np.linalg.norm(refaxis2_0[iiaero_node,:]))
            cosxth = np.dot(refaxis_0[iiaero_node,:],[1,0,0])/(np.linalg.norm(refaxis_0[iiaero_node,:]))
            cosyth = np.dot(refaxis_0[iiaero_node,:],[0,1,0])/(np.linalg.norm(refaxis_0[iiaero_node,:]))
            coszth = np.dot(refaxis_0[iiaero_node,:],[0,0,1])/(np.linalg.norm(refaxis_0[iiaero_node,:]))
            loadflap_A1_force_x = [liftflap_A1_hfreq[0]*cosxh,liftflap_A1_hfreq[0]*cosyh,liftflap_A1_hfreq[0]*coszh,liftflap_A1_thfreq[0]*cosxth,liftflap_A1_thfreq[0]*cosyth,liftflap_A1_thfreq[0]*coszth,0,0,0] 
            loadflap_A1_force_y = [liftflap_A1_hfreq[1]*cosxh,liftflap_A1_hfreq[1]*cosyh,liftflap_A1_hfreq[1]*coszh,liftflap_A1_thfreq[1]*cosxth,liftflap_A1_thfreq[1]*cosyth,liftflap_A1_thfreq[1]*coszth,0,0,0]
            loadflap_A1_force_z = [liftflap_A1_hfreq[2]*cosxh,liftflap_A1_hfreq[2]*cosyh,liftflap_A1_hfreq[2]*coszh,liftflap_A1_thfreq[2]*cosxth,liftflap_A1_thfreq[2]*cosyth,liftflap_A1_thfreq[2]*coszth,0,0,0]
            loadflap_A1_moment_x = [momentflap_A1_hfreq[0]*cosxh,momentflap_A1_hfreq[0]*cosyh,momentflap_A1_hfreq[0]*coszh,momentflap_A1_thfreq[0]*cosxth,momentflap_A1_thfreq[0]*cosyth,momentflap_A1_thfreq[0]*coszth,0,0,0] 
            loadflap_A1_moment_y = [momentflap_A1_hfreq[1]*cosxh,momentflap_A1_hfreq[1]*cosyh,momentflap_A1_hfreq[1]*coszh,momentflap_A1_thfreq[1]*cosxth,momentflap_A1_thfreq[1]*cosyth,momentflap_A1_thfreq[1]*coszth,0,0,0] 
            loadflap_A1_moment_z = [momentflap_A1_hfreq[2]*cosxh,momentflap_A1_hfreq[2]*cosyh,momentflap_A1_hfreq[2]*coszh,momentflap_A1_thfreq[2]*cosxth,momentflap_A1_thfreq[2]*cosyth,momentflap_A1_thfreq[2]*coszth,0,0,0] 
            loadflap_A2_force_x = [liftflap_A2_hfreq[0]*cosxh,liftflap_A2_hfreq[0]*cosyh,liftflap_A2_hfreq[0]*coszh,liftflap_A2_thfreq[0]*cosxth,liftflap_A2_thfreq[0]*cosyth,liftflap_A2_thfreq[0]*coszth,0,0,0] 
            loadflap_A2_force_y = [liftflap_A2_hfreq[1]*cosxh,liftflap_A2_hfreq[1]*cosyh,liftflap_A2_hfreq[1]*coszh,liftflap_A2_thfreq[1]*cosxth,liftflap_A2_thfreq[1]*cosyth,liftflap_A2_thfreq[1]*coszth,0,0,0]
            loadflap_A2_force_z = [liftflap_A2_hfreq[2]*cosxh,liftflap_A2_hfreq[2]*cosyh,liftflap_A2_hfreq[2]*coszh,liftflap_A2_thfreq[2]*cosxth,liftflap_A2_thfreq[2]*cosyth,liftflap_A2_thfreq[2]*coszth,0,0,0] 
            loadflap_A2_moment_x = [momentflap_A2_hfreq[0]*cosxh,momentflap_A2_hfreq[0]*cosyh,momentflap_A2_hfreq[0]*coszh,momentflap_A2_thfreq[0]*cosxth,momentflap_A2_thfreq[0]*cosyth,momentflap_A2_thfreq[0]*coszth,0,0,0] 
            loadflap_A2_moment_y = [momentflap_A2_hfreq[1]*cosxh,momentflap_A2_hfreq[1]*cosyh,momentflap_A2_hfreq[1]*coszh,momentflap_A2_thfreq[1]*cosxth,momentflap_A2_thfreq[1]*cosyth,momentflap_A2_thfreq[1]*coszth,0,0,0]
            loadflap_A2_moment_z = [momentflap_A2_hfreq[2]*cosxh,momentflap_A2_hfreq[2]*cosyh,momentflap_A2_hfreq[2]*coszh,momentflap_A2_thfreq[2]*cosxth,momentflap_A2_thfreq[2]*cosyth,momentflap_A2_thfreq[2]*coszth,0,0,0] 
            loadflap_A3_force_x = [liftflap_A3_hfreq[0]*cosxh,liftflap_A3_hfreq[0]*cosyh,liftflap_A3_hfreq[0]*coszh,liftflap_A3_thfreq[0]*cosxth,liftflap_A3_thfreq[0]*cosyth,liftflap_A3_thfreq[0]*coszth,0,0,0]
            loadflap_A3_force_y = [liftflap_A3_hfreq[1]*cosxh,liftflap_A3_hfreq[1]*cosyh,liftflap_A3_hfreq[1]*coszh,liftflap_A3_thfreq[1]*cosxth,liftflap_A3_thfreq[1]*cosyth,liftflap_A3_thfreq[1]*coszth,0,0,0]
            loadflap_A3_force_z = [liftflap_A3_hfreq[2]*cosxh,liftflap_A3_hfreq[2]*cosyh,liftflap_A3_hfreq[2]*coszh,liftflap_A3_thfreq[2]*cosxth,liftflap_A3_thfreq[2]*cosyth,liftflap_A3_thfreq[2]*coszth,0,0,0]
            loadflap_A3_moment_x = [momentflap_A3_hfreq[0]*cosxh,momentflap_A3_hfreq[0]*cosyh,momentflap_A3_hfreq[0]*coszh,momentflap_A3_thfreq[0]*cosxth,momentflap_A3_thfreq[0]*cosyth,momentflap_A3_thfreq[0]*coszth,0,0,0]
            loadflap_A3_moment_y = [momentflap_A3_hfreq[1]*cosxh,momentflap_A3_hfreq[1]*cosyh,momentflap_A3_hfreq[1]*coszh,momentflap_A3_thfreq[1]*cosxth,momentflap_A3_thfreq[1]*cosyth,momentflap_A3_thfreq[1]*coszth,0,0,0] 
            loadflap_A3_moment_z = [momentflap_A3_hfreq[2]*cosxh,momentflap_A3_hfreq[2]*cosyh,momentflap_A3_hfreq[2]*coszh,momentflap_A3_thfreq[2]*cosxth,momentflap_A3_thfreq[2]*cosyth,momentflap_A3_thfreq[2]*coszth,0,0,0]
            # Check if the node is the final one
            for aux in np.arange(9):
                for aux2 in np.arange(9):
                    if aux == 0:
                        KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A1_force_x[aux2]
                        CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A2_force_x[aux2]
                        MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A3_force_x[aux2]
                    elif aux == 1:
                        KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A1_force_y[aux2]
                        CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A2_force_y[aux2]
                        MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A3_force_y[aux2]
                    elif aux == 2:
                        KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A1_force_z[aux2]
                        CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A2_force_z[aux2]
                        MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A3_force_z[aux2]
                    elif aux == 3:
                        KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A1_moment_x[aux2]
                        CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A2_moment_x[aux2]
                        MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A3_moment_x[aux2]
                    elif aux == 4:
                        KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A1_moment_y[aux2]
                        CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A2_moment_y[aux2]
                        MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A3_moment_y[aux2]
                    elif aux == 5:
                        KK_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A1_moment_z[aux2]
                        CC_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A2_moment_z[aux2]
                        MM_global_ael[9*iimark_node+aux,int(9*iimark_node+aux2)] -= loadflap_A3_moment_z[aux2]
#    KK_global_ael *= 0
#    CC_global_ael *= 0
#    MM_global_ael *= 0
    KK_global_ael += KK_global_prima
    CC_global_ael += CC_global_prima
    MM_global_ael += MM_global_prima
    return KK_global_ael,CC_global_ael,MM_global_ael
#%%
def boundaryconditions_aelmod(KK_global_prima,CC_global_prima,MM_global_prima,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p,wfreq,v_inf):
    # Functions to determine the boundary conditions
    # -------------------------------------------------------------------------
    # KK_global         : stiffness matrix 
    # FF_global         : load vector 
    # qq_global   : displacement vector before applying the boundary conditions
    # mesh_mark         : markers of the mesh
    # case_setup        : information about the case setup
    # num_point         : number of points of the boundary conditions
    # sol_phys          : information of the solid physics
    # mesh_data         : data of the beam mesh
    # section           : information of the section
    # solver_vib_mod    : information about vibrational modes (to apply the loads as a function of the modal deformation)
    # -------------------------------------------------------------------------
    # ref_axis   : initialization of the reference axis
    ref_axis   = []
    polar_bc_ind = 0
    KK_global_prima = KK_global_prima.copy()
    CC_global_prima = CC_global_prima.copy()
    MM_global_prima = MM_global_prima.copy()
    for mark in mesh_mark:
        for bound in case_setup.boundary:
            # For each marker in the mesh find if there is some boundary condition
            if mark.name == bound.marker:
                if bound.type == "BC_AERO" :
                    if case_setup.aelmodomega == "YES":
                        if v_inf == 0:
                            bound.vrot = 1e-30
                        else:
                            bound.vrot = v_inf
                    else:
                        if v_inf == 0:
                            bound.vinf = 1e-30 
                        else:
                            bound.vinf = v_inf 
                    marktot = [mark]
                    if bound.f_flap == 1:
                        for markflap in mesh_mark:
                            if markflap.name == bound.flap:
                                if len(markflap.node) != len(mark.node):
                                    print('Error: core and flap have different number of nodes')
                                    sys.exit()
                                marktot.append(markflap)
                                break
                    # Add a reference axis to lifting surface
                    # do the boundary condition for the polar specified in the marker
                    # polar_bc_ind : number of independent polar boundary conditions
                    polar_bc_ind = 0
                    for polar_bc in case_setup.polar:
                        if polar_bc.id == bound.polar:
                            if polar_bc.lltnodesflag == 1:
                                for mark_llt in mesh_mark:
                                    if polar_bc.lltnodesmark == mark_llt.name:
                                        polar_bc.lltnodes = mark_llt.node
                                        polar_bc.lltnodes_ind = []
                                        polar_bc.flag_lltnodes_ind = 1
                            if polar_bc.cor3dflag == 1:
                                for mark_llt in mesh_mark:
                                    if polar_bc.cor3dmark == mark_llt.name:
                                        polar_bc.cor3dnodes = mark_llt.node
                                        polar_bc.cor3dnodes_ind = []
                                        polar_bc.flag_cor3dnodes_ind = 1
                            polar_bctot = [polar_bc]
                            polar_bctot_ind = [polar_bc_ind] 
                            break
                        polar_bc_ind += 1
                    # cdst_value0  : stationary drag coefficient of node 0
                    # cdst_value1  : stationary drag coefficient of node 1
                    # clst_value0  : stationary lift coefficient of node 0
                    # clst_value1  : stationary lift coefficient of node 1
                    # cmst_value0  : stationary pitch moment of node 0
                    # cmst_value1  : stationary pitch moment of node 1
                    # cd_ind       : induced drag
                    # lpos         : distance to the root of the aerodynamic surface
                    cdst_value0   = np.zeros((len(mark.node),))
                    cdst_value1   = np.zeros((len(mark.node),))
                    clst_value0   = np.zeros((len(mark.node),))
                    clst_value1   = np.zeros((len(mark.node),))
                    cmst_value0   = np.zeros((len(mark.node),))
                    cmst_value1   = np.zeros((len(mark.node),))
                    cddyn_value0  = np.zeros((len(mark.node),))
                    cddyn_value1  = np.zeros((len(mark.node),))
                    cldyn_value0  = np.zeros((len(mark.node),))
                    cldyn_value1  = np.zeros((len(mark.node),))
                    cmdyn_value0  = np.zeros((len(mark.node),))
                    cmdyn_value1  = np.zeros((len(mark.node),))
                    lpos          = np.zeros((len(mark.node),))
                    # Geometric characteristics of the cross-sections
                    # chord_vec       : chord of the airfoil in vectorial form
                    # b_chord         : half of the chord length
                    # vdir0           : velocity vector in the direction 0
                    # vdir1           : velocity vector in the direction 1
                    # aoa0_geom       : geometric angle of attack of the problem elem 0
                    # aoa1_geom       : geometric angle of attack of the problem elem 1
                    # LL_cum          : total cumulated longitude
                    # refaxis_0       : moment coefficient direction elem 0
                    # refaxis_0       : moment coefficient direction elem 1
                    # refaxis2_0      : lift coefficient direction elem 0
                    # refaxis2_0      : lift coefficient direction elem 1
                    # aoa_0lift       : null lift angle of attack
                    # lpos            : distance to the reference node
                    # iimark_node     : index of the node in the marker
                    b_chord       = np.zeros((len(mark.node),))
                    chord_vec     = np.zeros((len(mark.node),3))
                    non_acac_dist = np.zeros((len(mark.node),))
                    non_eaac_dist = np.zeros((len(mark.node),))
                    non_eamc_dist = np.zeros((len(mark.node),))
                    vdir0         = np.zeros((len(mark.node),3))
                    vdir1         = np.zeros((len(mark.node),3))
                    aoa0_geom     = np.zeros((len(mark.node),))
                    aoa1_geom     = np.zeros((len(mark.node),))
                    refaxis_0     = np.zeros((len(mark.node),3))
                    refaxis_1     = np.zeros((len(mark.node),3))
                    refaxis2_0    = np.zeros((len(mark.node),3))
                    refaxis2_1    = np.zeros((len(mark.node),3))
                    aoa_0lift     = np.zeros((len(mark.node),))
                    aoa_0liftflap = np.zeros((len(mark.node),))
                    cd_aoa0       = np.zeros((len(mark.node),))
                    if bound.f_flap == 1:
                        polar_bcflap_ind = 0
                        for polar_bcflap in case_setup.polar:
                            if polar_bcflap.id == bound.flappolar:
                                if polar_bcflap.lltnodesflag == 1:
                                    for mark_llt in mesh_mark:
                                        if polar_bcflap.lltnodesmark == mark_llt.name:
                                            polar_bcflap.lltnodes = mark_llt.node
                                            polar_bcflap.lltnodes_ind = []
                                            polar_bcflap.flag_lltnodes_ind = 1
                                if polar_bcflap.cor3dflag == 1:
                                    for mark_llt in mesh_mark:
                                        if polar_bcflap.cor3dmark == mark_llt.name:
                                            polar_bcflap.cor3dnodes = mark_llt.node
                                            polar_bcflap.cor3dnodes_ind = []
                                            polar_bcflap.flag_cor3dnodes_ind = 1
                                polar_bctot.append(polar_bcflap)
                                polar_bctot_ind.append(polar_bcflap_ind)
                                break
                            polar_bcflap_ind += 1
                    cdstflap_value0   = np.zeros((len(mark.node),))
                    cdstflap_value1   = np.zeros((len(mark.node),))
                    clstflap_value0   = np.zeros((len(mark.node),))
                    clstflap_value1   = np.zeros((len(mark.node),))
                    cmstflap_value0   = np.zeros((len(mark.node),))
                    cmstflap_value1   = np.zeros((len(mark.node),))
                    cddynflap_value0   = np.zeros((len(mark.node),))
                    cddynflap_value1   = np.zeros((len(mark.node),))
                    cldynflap_value0   = np.zeros((len(mark.node),))
                    cldynflap_value1   = np.zeros((len(mark.node),))
                    cmdynflap_value0   = np.zeros((len(mark.node),))
                    cmdynflap_value1   = np.zeros((len(mark.node),))
                    b_chordflap       = np.zeros((len(mark.node),))
                    chord_vecflap     = np.zeros((len(mark.node),3))
                    non_acac_distflap = np.zeros((len(mark.node),))
                    non_eaac_distflap = np.zeros((len(mark.node),))
                    non_eamc_distflap = np.zeros((len(mark.node),))
                    aoa0_geomflap    = np.zeros((len(mark.node),))
                    aoa1_geomflap    = np.zeros((len(mark.node),))
                    coreflapdist_adim = np.zeros((len(mark.node),))
                    deltaf_0lift     = np.zeros((len(mark.node),))
                    cd_aoa0flap       = np.zeros((len(mark.node),))
                    # Add the value of the longitude simulated in each beam element node
                    for iiaero_node in np.arange(len(mark.node)):
                        iimark_node = int(mark.node.flatten()[iiaero_node])
                        lpos[iiaero_node] = np.linalg.norm(mesh_data.point[iimark_node,:]-mesh_data.point[bound.refpoint])
                    LL_cum = np.max(lpos)
                    # If more than 1 polar
                    if polar_bc.pointfile[0] != -1:
                        pointref = np.where(mark.node==polar_bc.pointfile)[0].tolist()
                        lrefpol  = lpos[pointref]
                        polar_bc.iirefpol = np.argsort(lrefpol)
                        polar_bc.sortedlref = lrefpol[polar_bc.iirefpol]  
                    # For all the nodes in the surface
                    # cent_secx       : center of the airfoil x position
                    # cent_secy       : center of the airfoil y position
                    # acac_dist       : distance between the aerodynamic reference and the 0.25 chord distance
                    # acac_dist_sign  : sign of the distance between the aerodynamic reference and the 0.25 chord distance
                    # non_acac_dist   : nondimensional distance between the aerodynamic reference and the 0.25 chord distance with the moment sign
                    # eaac_dist       : distance between the aerodynamic reference and the elastic reference
                    # eaac_dist_sign  : sign of the distance between the aerodynamic reference and the elastic reference
                    # non_eaac_dist   : nondimensional distance between the aerodynamic reference and the elastic reference with the moment sign
                    # eamc_dist       : distance between the aerodynamic reference and the mean chord
                    # eamc_dist_sign  : sign of the distance between the aerodynamic reference and the mean chord
                    # non_eamc_dist   : nondimensional distance between the aerodynamic reference and the mean chord with the moment sign
                    for iiaero_node in np.arange(len(mark.node)):
                        iimark_node                = int(mark.node.flatten()[iiaero_node])
                        cent_secx                  = (section_globalCS.point_3d[iimark_node][section.te[iimark_node],0]+section_globalCS.point_3d[iimark_node][section.le[iimark_node],0])/2 #(section.points[iimark_node][0][section.te[iimark_node],0]+section.points[iimark_node][0][section.le[iimark_node],0])/2
                        cent_secy                  = (section_globalCS.point_3d[iimark_node][section.te[iimark_node],1]+section_globalCS.point_3d[iimark_node][section.le[iimark_node],1])/2 #(section.points[iimark_node][0][section.te[iimark_node],1]+section.points[iimark_node][0][section.le[iimark_node],1])/2
                        chord_vec[iiaero_node,:]   = section_globalCS.point_3d[iimark_node][section.te[iimark_node],:]-section_globalCS.point_3d[iimark_node][section.le[iimark_node],:] #section.points[iimark_node][0][section.te[iimark_node],:]-section.points[iimark_node][0][section.le[iimark_node],:]
                        b_chord[iiaero_node]       = np.linalg.norm(chord_vec[iiaero_node,:])/2
                        acac_dist                  = section_globalCS.aero_cent[iimark_node]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section_globalCS.point_3d[iimark_node][section.le[iimark_node],0]) #[section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]]-b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimark_node][0][section.le[iimark_node],:]
                        acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_acac_dist[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
                        eaac_dist                  = section_globalCS.aero_cent[iimark_node] #[section.ae_orig_x[iimark_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimark_node]-sol_phys.ysc[iiaero_node]]
                        eaac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],eaac_dist),np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_eaac_dist[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
                        eamc_dist                  = [-cent_secx,-cent_secy,0] #[sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
                        eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_eamc_dist[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chord[iiaero_node]   
                        if bound.f_flap == 1 and markflap.node.flatten()[iiaero_node] >= 0:
                            iimarkflap_node                = int(markflap.node.flatten()[iiaero_node])
                            if iimarkflap_node > 0:
                                cent_secx                  = (section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],0]+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],0])/2 #(section.points[iimarkflap_node][0][section.te[iimarkflap_node],0]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],0])/2
                                cent_secy                  = (section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],1]+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],1])/2 #(section.points[iimarkflap_node][0][section.te[iimarkflap_node],1]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],1])/2
                                chord_vecflap[iiaero_node,:]   = section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],:]-section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],:] #section.points[iimarkflap_node][0][section.te[iimarkflap_node],:]-section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
                                b_chordflap[iiaero_node]       = np.linalg.norm(chord_vecflap[iiaero_node,:])/2
                                acac_dist                  = section_globalCS.aero_cent[iimarkflap_node]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],0]) #[section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]]-b_chordflap[iiaero_node]/2*chord_vecflap[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
                                acac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimarkflap_node,0,:,:],acac_dist),np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_acac_distflap[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
                                eaac_dist                  = section_globalCS.aero_cent[iimark_node] #[section.ae_orig_x[iimarkflap_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimarkflap_node]-sol_phys.ysc[iiaero_node]]
                                eaac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimarkflap_node,0,:,:],eaac_dist),np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_eaac_distflap[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
                                eamc_dist                  = [-cent_secx,-cent_secy,0] # [sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
                                eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_eamc_distflap[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chordflap[iiaero_node] 
                                coreflapdist  = mesh_data.point[iimark_node,:]-mesh_data.point[iimarkflap_node,:]
                                coreflapdist_adim[iiaero_node] = np.sign(np.dot(coreflapdist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))/(b_chord[iiaero_node]+b_chordflap[iiaero_node])  
#                        cent_secx                  = (section.points[iimark_node][0][section.te[iimark_node],0]+section.points[iimark_node][0][section.le[iimark_node],0])/2
#                        cent_secy                  = (section.points[iimark_node][0][section.te[iimark_node],1]+section.points[iimark_node][0][section.le[iimark_node],1])/2
#                        chord_vec[iiaero_node,:]   = section.points[iimark_node][0][section.te[iimark_node],:]-section.points[iimark_node][0][section.le[iimark_node],:]
#                        b_chord[iiaero_node]       = np.linalg.norm(chord_vec[iiaero_node,:])/2
#                        acac_dist                  = [section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimark_node][0][section.le[iimark_node],:])
#                        acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)[:2]))
#                        non_acac_dist[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/(2*b_chord[iiaero_node])
#                        eaac_dist                  = [section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]]
#                        eaac_dist_sign             = np.sign(np.dot(eaac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)[:2]))
#                        non_eaac_dist[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
#                        eamc_dist                  =  [-cent_secx,-cent_secy,0]
#                        eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
#                        non_eamc_dist[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chord[iiaero_node]   
#                        if bound.f_flap == 1  and markflap.node.flatten()[iiaero_node] >= 0:
#                            iimarkflap_node                = int(markflap.node.flatten()[iiaero_node])
#                            cent_secx                  = (section.points[iimarkflap_node][0][section.te[iimarkflap_node],0]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],0])/2
#                            cent_secy                  = (section.points[iimarkflap_node][0][section.te[iimarkflap_node],1]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],1])/2
#                            chord_vecflap[iiaero_node,:]   = section.points[iimarkflap_node][0][section.te[iimarkflap_node],:]-section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
#                            b_chordflap[iiaero_node]       = np.linalg.norm(chord_vecflap[iiaero_node,:])/2
#                            acac_dist                  = [section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]]-b_chordflap[iiaero_node]/2*chord_vecflap[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
#                            acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)[:2]))
#                            non_acac_distflap[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/(2*b_chord[iiaero_node])
#                            eaac_dist                  = [section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]]
#                            eaac_dist_sign             = np.sign(np.dot(eaac_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)[:2]))
#                            non_eaac_distflap[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chordflap[iiaero_node])
#                            eamc_dist                  =  [sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
#                            eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
#                            non_eamc_distflap[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chordflap[iiaero_node] 
#                            coreflapdist  = mesh_data.point[iimark_node,:]-mesh_data.point[iimarkflap_node,:]
#                            coreflapdist_adim[iiaero_node] = np.sign(np.dot(coreflapdist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))/(b_chord[iiaero_node]+b_chordflap[iiaero_node])  
                    # If a rotative aerodynamic model is selected the nondimensional radius is calculated
                    # flag_start    : flag to start the bisection method
                    # flag_start1   : flag to start the bisection method
                    # flag_start2   : flag to start the bisection method
                    # r_vinduced    : nondimensional radius
                    # cltot2        : total value of the lift coefficient in the previous iteration
                    # cttot2        : total value of the thrust coefficient in the previous iteration
                    # v_induced     : induced velocity 
                    # Gamma_induced : induced circulation
                    # tol_vind      : tolerance of the induced velocity error
                    # aux_vind_e    : index to count the number of iterations in the induced velocity calculation loop
                    # lpos_cir      : distance to the root of the aerodynamic surface in circular coordinates
                    # lpc_in        : position of the section relative to the total length
                    # mat_pllt      : lifting line theory matrix
                    # vec_pllt      : lifting line theory vector
                    if polar_bc.eff3d == 'LLT':
                        class eff_3d:
                            Gamma_induced = np.zeros((len(mark.node),))
                            cltot2 = 1e3
                            lpos_cir  = np.zeros((len(mark.node),))
                            sign_vi   = -1
                        v_induced = np.zeros((len(mark.node),3))
                        for iiaero_node in np.arange(len(mark.node)):
                            lpc_in = lpos[iiaero_node]/LL_cum
                            if lpc_in>1:
                                lpc_in = 1
                            eff_3d.lpos_cir[iiaero_node] = np.arccos(lpc_in) 
                    elif polar_bc.eff3d == 'BEM':
                        class eff_3d:
                            flag_start      = np.zeros((len(mark.node),))
                            flag_start1     = np.zeros((len(mark.node),))
                            flag_start2     = np.zeros((len(mark.node),))
                            error1          = np.ones((len(mark.node),))
                            error2          = np.ones((len(mark.node),))
                            sign_vi         = -np.sign(np.dot(bound.vdir,bound.refrot))
                            r_vinduced      = lpos/LL_cum
                            v_induced2      = np.zeros((len(mark.node),3))
                            masa            = np.zeros((len(mark.node),)) #sol_phys.m11
                            frelax_vt       = 1
                            errorout1       = 1
                            errorout2       = 2
                        for ii_massind in np.arange(len(mark.node)):
                            ii_mass = int(mark.node[ii_massind])
                            eff_3d.masa[ii_massind] = MM_global_prima[ii_mass,ii_mass]
                        v_induced         = 0*bound.vrot*bound.radius*np.ones((len(mark.node),3))
                        v_induced[:,0]   *= eff_3d.sign_vi*bound.vdir[0]/np.linalg.norm(bound.vdir)
                        v_induced[:,1]   *= eff_3d.sign_vi*bound.vdir[1]/np.linalg.norm(bound.vdir)
                        v_induced[:,2]   *= eff_3d.sign_vi*bound.vdir[2]/np.linalg.norm(bound.vdir)
                        vtan_induced = np.zeros((len(mark.node),3))
                        eff_3d.v_induced_tan0     = np.zeros((len(mark.node),3))
                        eff_3d.v_induced_tan1     = np.zeros((len(mark.node),3))
                        eff_3d.v_induced1 = v_induced.copy()
                    else: 
                        v_induced = np.zeros((len(mark.node),3))                         
                        class eff_3d:
                            sign_vi         = 1
                    try:
                        tol_vind = polar_bc.vind_tol
                    except:
                        tol_vind = 1e-5
                    try:
                        maxit_vind = polar_bc.vind_maxit
                    except:
                        maxit_vind = 1e3
                    eff_3d.aux_vind_e_out = 0 
                    # initialize the error
                    error_out = tol_vind*10
                    flag_conv = 0
                    flag_convout = 0
                    # While error is  higher than the required tolerance
                    while abs(error_out)>tol_vind or flag_convout == 0:
                        if abs(error_out)<tol_vind:
                            if polar_bc.eff3d == 'BEM':
                                vtan_induced_mod = vtan_induced.copy()
                                vtan_ind_int = vtan_induced[polar_bc.lltnodes_ind,:]
                                l_ind_int = lpos[polar_bc.lltnodes_ind]
                                f_vxind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,0],'cubic', fill_value='extrapolate')
                                f_vyind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,1],'cubic', fill_value='extrapolate')
                                f_vzind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,2],'cubic', fill_value='extrapolate') 
                                for auxaeronode in np.arange(len(mark.node)):
                                    vtan_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
                                vtan_induced = vtan_induced_mod.copy()
                            flag_convout = 1
                        eff_3d.aux_vind_e_out += 1
                        # if the loop cannot find the solution in a determined number of iterations, stop the loop
                        if eff_3d.aux_vind_e_out > maxit_vind: 
                            print('Induced tangential velocity: Not converged - '+str(error_out))
                            break
                        # error  : error of the lift coefficient in each iteration
                        error  = tol_vind*10
                        eff_3d.error = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.error1ct = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.error2ct = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.flag_start = np.zeros((len(mark.node),))
                        eff_3d.flag_start1 = np.zeros((len(mark.node),))
                        eff_3d.flag_start2 = np.zeros((len(mark.node),))
                        eff_3d.aux_vind_e = 1 
                        while abs(error)>tol_vind or flag_conv == 0:
                            if abs(error)<tol_vind:
                                if polar_bc.eff3d == 'BEM':
                                    v_induced_mod = v_induced.copy()
                                    v_ind_int = v_induced[polar_bc.lltnodes_ind,:]
                                    l_ind_int = lpos[polar_bc.lltnodes_ind]
                                    f_vxind = interpolate.interp1d(l_ind_int,v_ind_int[:,0],'cubic', fill_value='extrapolate')
                                    f_vyind = interpolate.interp1d(l_ind_int,v_ind_int[:,1],'cubic', fill_value='extrapolate')
                                    f_vzind = interpolate.interp1d(l_ind_int,v_ind_int[:,2],'cubic', fill_value='extrapolate') 
                                    for auxaeronode in np.arange(len(mark.node)):
                                        v_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
                                    v_induced = v_induced_mod.copy()
                                flag_conv = 1
                            # if the loop cannot find the solution in a determined number of iterations, stop the loop
                            if eff_3d.aux_vind_e > maxit_vind: 
                                print('Induced velocity: Not converged - '+str(error))
                                break
                            eff_3d.aux_vind_e += 1
                            # Define variables for the rotative aerodynamics
                            if polar_bc.eff3d == 'LLT':            
                                eff_3d.mat_pllt  = np.zeros((2*len(mark.node),len(mark.node)))
                                eff_3d.vec_pllt  = np.zeros((2*len(mark.node),))
                            elif polar_bc.eff3d == 'BEM':
                                # r_vinduced     : nondimensional distance of the blade radius
                                # vi_vinduced    : induced velocity
                                # phi_vinduced   : induced velocity on the blade
                                # f_vinduced     : 3D effect factor of the blade
                                # F_vinduced     : 3D effect factor of the blade
                                # dct_dr_induced : variation of thrust with radial distance
                                eff_3d.phi_vinduced0       = np.zeros((len(mark.node),))
                                eff_3d.f_vinduced0         = np.zeros((len(mark.node),))
                                eff_3d.F_vinduced0         = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced0 = np.zeros((len(mark.node),))
                                eff_3d.phi_vinduced1       = np.zeros((len(mark.node),))
                                eff_3d.f_vinduced1         = np.zeros((len(mark.node),))
                                eff_3d.F_vinduced1         = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced1 = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced  = np.zeros((len(mark.node),))
                            # aoa_vec0    : angle of attack of node 0
                            # aoa_vec1    : angle of attack of node 1
                            # aoap_vec0   : angle of attack derivative of node 0
                            # aoap_vec1   : angle of attack derivative of node 1
                            # aoa_val     : value of the angle of attack
                            # aoap_val    : value of the angle of attack derivative
                            # aoapp_val   : value of the angle of attack second derivative
                            # pos_vec0    : position vector of node 0
                            # pos_vec1    : position vector of node 1
                            aoa_vec0    = np.zeros((len(mark.node),))
                            aoa_vec1    = np.zeros((len(mark.node),))
                            aoa_val     = np.zeros((len(mark.node),))
                            pos_vec0    = np.zeros((len(mark.node),))
                            pos_vec1    = np.zeros((len(mark.node),))
                            aoa_vec0_st = np.zeros((len(mark.node),))
                            aoa_vec1_st = np.zeros((len(mark.node),))
                            Vinf0       = np.zeros((len(mark.node),))
                            Vinf1       = np.zeros((len(mark.node),))
                            reynolds0   = np.zeros((len(mark.node),))
                            reynolds1   = np.zeros((len(mark.node),))
                            deltaflap_vec0    = np.zeros((len(mark.node),))
                            deltaflap_vec1    = np.zeros((len(mark.node),))
                            # Calculate the reference axis
                            refaxis_0, refaxis_1, refaxis2_0, refaxis2_1, vdir0, vdir1, aoa0_geom, aoa1_geom, eff_3d = ref_axis_aero(bound,mark,polar_bc,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,v_induced,vdir0,vdir1,aoa0_geom,aoa1_geom,rr_vec_p,rot_matT_p,chord_vec,eff_3d,0)
                            if bound.f_flap == 1:
                                # Calculate the reference axis
                                aoa0_geomflap, aoa1_geomflap = ref_axis_aero(bound,markflap,polar_bcflap,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,v_induced,vdir0,vdir1,aoa0_geomflap,aoa1_geomflap,rr_vec_p,rot_matT_p,chord_vecflap,eff_3d,1)
                            # for each node in the surface
                            for iiaero_node in np.arange(len(mark.node)):
                                iimark_node = int(mark.node.flatten()[iiaero_node])
                                # Calculate the velocity of the free stream for the airfoil for the different aerodynamic models
                                # refvind0       : reference induced velocity direction in element 0
                                # refvind1       : reference induced velocity direction in element 1
                                # vi_vinduced0   : induced velocity scalar in element 0
                                # vi_vinduced1   : induced velocity scalar in element 1
                                # Vinf0          : free stream velocity in element 0
                                # Vinf1          : free stream velocity in element 1
                                # Vinf_vindaoa   : velocity to calculate the angle of the induced angle of attack
                                if polar_bc.eff3d == 'LLT':
                                    refvind0           = -refaxis2_0[iiaero_node,:]
                                    refvind1           = -refaxis2_1[iiaero_node,:]
                                    vi_vinduced0       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(refvind0))
                                    vi_vinduced1       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(refvind1))
                                    Vinf0[iiaero_node] = np.sqrt(bound.vinf**2+vi_vinduced0**2)
                                    Vinf1[iiaero_node] = np.sqrt(bound.vinf**2+vi_vinduced1**2)
                                    Vinf_vindaoa       = bound.vinf
                                elif polar_bc.eff3d == 'BEM':
                                    refvind0           = eff_3d.sign_vi*bound.vdir
                                    refvind1           = eff_3d.sign_vi*bound.vdir
                                    vi_vinduced0       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(bound.vdir))
                                    vi_vinduced1       = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_induced[iiaero_node,:]),np.array(bound.vdir))
                                    vt_vinduced0       = np.dot(eff_3d.v_induced_tan0[iiaero_node,:],eff_3d.vec_vrot0[iiaero_node,:])
                                    vt_vinduced1       = np.dot(eff_3d.v_induced_tan1[iiaero_node,:],eff_3d.vec_vrot1[iiaero_node,:])
                                    Vinf0[iiaero_node] = np.sqrt((bound.vinf+vi_vinduced0)**2+(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius+vt_vinduced0)**2)+1e-10
                                    Vinf1[iiaero_node] = np.sqrt((bound.vinf+vi_vinduced1)**2+(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius+vt_vinduced1)**2)+1e-10
                                    v_induced_tan      = (eff_3d.v_induced_tan0[iiaero_node,:]*LL_vec_p[0,iiaero_node]+eff_3d.v_induced_tan1[iiaero_node,:]*LL_vec_p[1,iiaero_node])/(LL_vec_p[0,iiaero_node]+LL_vec_p[1,iiaero_node])
                                    vtan_induced[iiaero_node] = v_induced_tan
                                    Vinf_vindaoa       = np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+v_induced_tan)+1e-10
                                else:
                                    refvind0           = -refaxis2_0[iiaero_node,:]
                                    refvind1           = -refaxis2_1[iiaero_node,:]
                                    Vinf0[iiaero_node] = bound.vinf 
                                    Vinf1[iiaero_node] = bound.vinf 
                                    Vinf_vindaoa       = bound.vinf
                                # For lifting line theory set the angle of attack at the tip as zero, calculate in the rest of points
                                if (polar_bc.eff3d == 'LLT' ) and lpos[iiaero_node] > 0.999*LL_cum:  # or polar_bc.eff3d == 'BEM'
                                    aoa_vec0[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoa_vec1[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoa_val[iiaero_node]     = aoa_0lift[iiaero_node]
                                    deltaflap_vec0[iiaero_node]    = deltaf_0lift[iiaero_node]
                                    deltaflap_vec1[iiaero_node]    = deltaf_0lift[iiaero_node]
                                    v_induced[iiaero_node,:] = np.matmul(rot_mat_p[iimark_node,1,:,:],Vinf0[iiaero_node]*np.tan(aoa_val[iiaero_node]-aoa0_geom[iiaero_node])*np.array(refvind1))
                                # In the rest of the points
                                else:
                                    # if needed include rotational effects
                                    if polar_bc.eff3d == 'BEM':
                                        if eff_3d.r_vinduced[iiaero_node] == 0:
                                            eff_3d.phi_vinduced0[iiaero_node] = np.arctan((bound.vinf)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                            eff_3d.phi_vinduced1[iiaero_node] = np.arctan((bound.vinf)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                        else:
                                            eff_3d.phi_vinduced0[iiaero_node] = np.arctan((bound.vinf+vi_vinduced0)/np.max([np.linalg.norm(bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node]*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:]),1e-10])) 
                                            eff_3d.phi_vinduced1[iiaero_node] = np.arctan((bound.vinf+vi_vinduced1)/np.max([np.linalg.norm(bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node]*eff_3d.vec_vrot1[iiaero_node,:]+eff_3d.v_induced_tan1[iiaero_node,:]),1e-10])) 
                                        eff_3d.f_vinduced0[iiaero_node]   = bound.Nb/2*((1-(eff_3d.r_vinduced[iiaero_node]))/np.max([eff_3d.r_vinduced[iiaero_node]*eff_3d.phi_vinduced0[iiaero_node],1e-10]))
                                        eff_3d.f_vinduced1[iiaero_node]   = bound.Nb/2*((1-(eff_3d.r_vinduced[iiaero_node]))/np.max([eff_3d.r_vinduced[iiaero_node]*eff_3d.phi_vinduced1[iiaero_node],1e-10]))
                                        eff_3d.F_vinduced0[iiaero_node]   = 2/np.pi*np.arccos(np.exp(-eff_3d.f_vinduced0[iiaero_node]))
                                        eff_3d.F_vinduced1[iiaero_node]   = 2/np.pi*np.arccos(np.exp(-eff_3d.f_vinduced1[iiaero_node]))
                                        v_vert                            = eff_3d.sign_vi*(bound.vinf*bound.vdir+v_induced[iiaero_node,:])
                                    else:
                                        v_vert = v_induced[iiaero_node,:] #eff_3d.sign_vi*
                                    aoa_vec0[iiaero_node] = (aoa0_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_vert),np.array(refvind0))/Vinf_vindaoa)
                                    aoa_vec1[iiaero_node] = (aoa1_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_vert),np.array(refvind1))/Vinf_vindaoa)
                                    aoa_val[iiaero_node]  = (aoa_vec0[iiaero_node]*LL_vec_p[0,iiaero_node]+aoa_vec1[iiaero_node]*LL_vec_p[1,iiaero_node])/(LL_vec_p[0,iiaero_node]+LL_vec_p[1,iiaero_node])
                                    if bound.f_flap == 1  and markflap.node.flatten()[iiaero_node] >= 0:
                                        deltaflap_vec0[iiaero_node]    = (aoa0_geomflap[iiaero_node]-aoa0_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_vert),np.array(refvind0))/Vinf_vindaoa)
                                        deltaflap_vec1[iiaero_node]    = (aoa1_geomflap[iiaero_node]-aoa1_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_vert),np.array(refvind1))/Vinf_vindaoa)
                                    else:
                                        deltaflap_vec0[iiaero_node] = 0
                                        deltaflap_vec1[iiaero_node] = 0
                                aoa_vec0_st[iiaero_node]  = aoa0_geom[iiaero_node]
                                aoa_vec1_st[iiaero_node]  = aoa1_geom[iiaero_node]
                                pos_vec0[iiaero_node] = 0
                                pos_vec1[iiaero_node] = 0
                                reynolds0[iiaero_node] = Vinf0[iiaero_node]*2*b_chord[iiaero_node]*bound.rho/bound.mu
                                reynolds1[iiaero_node] = Vinf1[iiaero_node]*2*b_chord[iiaero_node]*bound.rho/bound.mu
#                                    print([reynolds0,reynolds1])
                            case_setup, polar_bc_ind,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,eff_3d,error,cl_alpha,cl_alphaflap,error2,v_induced = steady_aero(case_setup,bound,polar_bctot,mesh_data,marktot,polar_bctot_ind,aoa_vec0,aoa_vec1,aoa_vec0_st,aoa_vec1_st,pos_vec0,pos_vec1,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,aoa_0lift,aoa_0liftflap,v_induced,refaxis2_0,refaxis2_1,Vinf0,Vinf1,LL_vec_p,lpos,LL_cum,eff_3d,tol_vind,b_chord,non_acac_dist,b_chordflap,non_acac_distflap,coreflapdist_adim,aoa_vec0,aoa_vec1,cd_aoa0,cd_aoa0flap,aoa0_geom,aoa1_geom,reynolds0,reynolds1,deltaflap_vec0,deltaflap_vec1,bound.f_flap)
                        if polar_bc.eff3d == 'BEM':
                            eff_3d,error_out,vtan_induced = converge_vtan(bound,mark,polar_bc,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,v_induced,LL_vec_p,eff_3d,b_chord,vtan_induced)
                        else:
                            error_out = tol_vind/10
                    clfreq0,clfreq1,cmfreq0,cmfreq1 = mod_aero(case_setup,bound,mark,polar_bc,wfreq,polar_bc_ind,b_chord,cl_alpha,non_eamc_dist,non_eaac_dist,non_acac_dist,Vinf0,Vinf1)                        
                    if bound.f_flap == 1:
                        clfreq0flap,clfreq1flap,cmfreq0flap,cmfreq1flap = mod_aero(case_setup,bound,markflap,polar_bcflap,wfreq,polar_bcflap_ind,b_chordflap,cl_alphaflap,non_eamc_distflap,non_eaac_distflap,non_acac_distflap,Vinf0,Vinf1) 
                    else:
                        clfreq0flap = []
                        clfreq1flap = []
                        cmfreq0flap = []
                        cmfreq1flap = []
                    KK_global_ae,CC_global_ae,MM_global_ae = total_mat_mod(clfreq0,clfreq1,cmfreq0,cmfreq1,clfreq0flap,clfreq1flap,cmfreq0flap,cmfreq1flap,LL_vec_p,rot_mat_p,b_chord,b_chordflap,Vinf0,Vinf1,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,KK_global_prima,CC_global_prima,MM_global_prima,bound,section,marktot,num_point)                  

    KK_global_ae, MM_global_ae, RR_global, case_setup, ref_axis,CC_global_ae = init_boundaryconditions_tran(KK_global_ae,MM_global_ae,CC_global_ae,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)
    return KK_global_ae,CC_global_ae,MM_global_ae



#%%
#%%
def boundaryconditions_ael(KK_global,MM_global,FF_global,qq_global,mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,solver_vib_mod,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p,disp_values): 
    # Functions to determine the boundary conditions
    # -------------------------------------------------------------------------
    # KK_global         : stiffness matrix 
    # FF_global         : load vector 
    # qq_global   : displacement vector before applying the boundary conditions
    # mesh_mark         : markers of the mesh
    # case_setup        : information about the case setup
    # num_point         : number of points of the boundary conditions
    # sol_phys          : information of the solid physics
    # mesh_data         : data of the beam mesh
    # section           : information of the section
    # solver_vib_mod    : information about vibrational modes (to apply the loads as a function of the modal deformation)
    # -------------------------------------------------------------------------
    # ref_axis   : initialization of the reference axis
    ref_axis   = []
    RR_global  = np.zeros((9*num_point,9*num_point)) 
    fgrav      = np.zeros((9*num_point))
    FF_orig    = FF_global.copy()
    if case_setup.grav_fl == 1:
        cdg_x = 0
        cdg_y = 0 
        cdg_z = 0
        fgrav_sumx = 0
        fgrav_sumy = 0
        fgrav_sumz = 0
        for ii_point in np.arange(num_point):
            for jj_point in np.arange(3):
                fgrav[ii_point*9+jj_point] =  MM_global[ii_point*9+jj_point,ii_point*9+jj_point]
                FF_global[ii_point*9+jj_point] += case_setup.grav_g[jj_point] *fgrav[ii_point*9+jj_point]
                if jj_point == 0:
                    cdg_x += mesh_data.point[ii_point,0]*fgrav[ii_point*9+jj_point]
                    fgrav_sumx += fgrav[ii_point*9+jj_point]
                elif jj_point == 1:
                    cdg_y += mesh_data.point[ii_point,1]*fgrav[ii_point*9+jj_point]
                    fgrav_sumy += fgrav[ii_point*9+jj_point]
                elif jj_point == 2:
                    cdg_z += mesh_data.point[ii_point,2]*fgrav[ii_point*9+jj_point]
                    fgrav_sumz += fgrav[ii_point*9+jj_point]
        cdg_x /= fgrav_sumx
        cdg_y /= fgrav_sumy
        cdg_z /= fgrav_sumz
        mesh_data.cdg = [cdg_x,cdg_y,cdg_z]
    for mark in mesh_mark:
        for bound in case_setup.boundary:
            # For each marker in the mesh find if there is some boundary condition
            if mark.name == bound.marker:
                if bound.type == "BC_NODELOAD":
                    # If bc is a load in a node, add the value to the global force vector
                    for node in mark.node:
                        bound2val = bound.values.copy()
                        r_ref  = np.matmul(np.transpose(rot_mat_p[int(node),0,:,:]),np.concatenate(([section.ref_x[int(node)],section.ref_y[int(node)]],[0])))
                        bound2val[3:6] += np.cross(r_ref,bound.values[:3])
                        if node<num_point:
                            FF_global[int(9*node):int(9*node+9)] = bound2val
                        else:
                            FF_global[int(node):] = bound2val
                    FF_orig = FF_global.copy()
                elif bound.type == "BC_AERO" :
                    if bound.vinf == "VINF_DAT":
                        vel2 = bound.vinf
                        vel2dir = bound.vdir
                        bound.vinf = np.linalg.norm(case_setup.vinf)
                        bound.vdir = case_setup.vinf/np.linalg.norm(case_setup.vinf)
                    marktot = [mark]
                    if bound.f_flap == 1:
                        for markflap in mesh_mark:
                            if markflap.name == bound.flap:
                                if len(markflap.node) != len(mark.node):
                                    print('Error: core and flap have different number of nodes')
                                    sys.exit()
                                marktot.append(markflap)
                                break
                    # Add a reference axis to lifting surface
                    # do the boundary condition for the polar specified in the marker
                    # polar_bc_ind : number of independent polar boundary conditions
                    polar_bc_ind = 0
                    phi_val            = disp_values.phi
                    psi_val            = disp_values.psi
                    theta_val          = disp_values.theta
                    for polar_bc in case_setup.polar:
                        if polar_bc.id == bound.polar:
                            if polar_bc.lltnodesflag == 1:
                                for mark_llt in mesh_mark:
                                    if polar_bc.lltnodesmark == mark_llt.name:
                                        polar_bc.lltnodes = mark_llt.node
                                        polar_bc.lltnodes_ind = []
                                        polar_bc.flag_lltnodes_ind = 1
                            if polar_bc.cor3dflag == 1:
                                for mark_llt in mesh_mark:
                                    if polar_bc.cor3dmark == mark_llt.name:
                                        polar_bc.cor3dnodes = mark_llt.node
                                        polar_bc.cor3dnodes_ind = []
                                        polar_bc.flag_cor3dnodes_ind = 1
                            polar_bctot = [polar_bc]
                            polar_bctot_ind = [polar_bc_ind] 
                            break
                        polar_bc_ind += 1
                    # cdst_value0  : stationary drag coefficient of node 0
                    # cdst_value1  : stationary drag coefficient of node 1
                    # clst_value0  : stationary lift coefficient of node 0
                    # clst_value1  : stationary lift coefficient of node 1
                    # cmst_value0  : stationary pitch moment of node 0
                    # cmst_value1  : stationary pitch moment of node 1
                    # cd_ind       : induced drag
                    # lpos         : distance to the root of the aerodynamic surface
                    cdst_value0   = np.zeros((len(mark.node),))
                    cdst_value1   = np.zeros((len(mark.node),))
                    clst_value0   = np.zeros((len(mark.node),))
                    clst_value1   = np.zeros((len(mark.node),))
                    cmst_value0   = np.zeros((len(mark.node),))
                    cmst_value1   = np.zeros((len(mark.node),))
                    cddyn_value0  = np.zeros((len(mark.node),))
                    cddyn_value1  = np.zeros((len(mark.node),))
                    cldyn_value0  = np.zeros((len(mark.node),))
                    cldyn_value1  = np.zeros((len(mark.node),))
                    cmdyn_value0  = np.zeros((len(mark.node),))
                    cmdyn_value1  = np.zeros((len(mark.node),))
                    lpos          = np.zeros((len(mark.node),))
                    # Geometric characteristics of the cross-sections
                    # chord_vec       : chord of the airfoil in vectorial form
                    # b_chord         : half of the chord length
                    # vdir0           : velocity vector in the direction 0
                    # vdir1           : velocity vector in the direction 1
                    # aoa0_geom       : geometric angle of attack of the problem elem 0
                    # aoa1_geom       : geometric angle of attack of the problem elem 1
                    # LL_cum          : total cumulated longitude
                    # refaxis_0       : moment coefficient direction elem 0
                    # refaxis_0       : moment coefficient direction elem 1
                    # refaxis2_0      : lift coefficient direction elem 0
                    # refaxis2_0      : lift coefficient direction elem 1
                    # aoa_0lift       : null lift angle of attack
                    # lpos            : distance to the reference node
                    # iimark_node     : index of the node in the marker
                    b_chord       = np.zeros((len(mark.node),))
                    chord_vec     = np.zeros((len(mark.node),3))
                    non_acac_dist = np.zeros((len(mark.node),))
                    non_eaac_dist = np.zeros((len(mark.node),))
                    non_eamc_dist = np.zeros((len(mark.node),))
                    vdir0         = np.zeros((len(mark.node),3))
                    vdir1         = np.zeros((len(mark.node),3))
                    aoa0_geom     = np.zeros((len(mark.node),))
                    aoa1_geom     = np.zeros((len(mark.node),))
                    refaxis_0     = np.zeros((len(mark.node),3))
                    refaxis_1     = np.zeros((len(mark.node),3))
                    refaxis2_0    = np.zeros((len(mark.node),3))
                    refaxis2_1    = np.zeros((len(mark.node),3))
                    aoa_0lift     = np.zeros((len(mark.node),))
                    aoa_0liftflap = np.zeros((len(mark.node),))
                    cd_aoa0       = np.zeros((len(mark.node),))
                    if bound.f_flap == 1:
                        polar_bcflap_ind = 0
                        for polar_bcflap in case_setup.polar:
                            if polar_bcflap.id == bound.flappolar:
                                if polar_bcflap.lltnodesflag == 1:
                                    for mark_llt in mesh_mark:
                                        if polar_bcflap.lltnodesmark == mark_llt.name:
                                            polar_bcflap.lltnodes = mark_llt.node
                                            polar_bcflap.lltnodes_ind = []
                                            polar_bcflap.flag_lltnodes_ind = 1
                                if polar_bcflap.cor3dflag == 1:
                                    for mark_llt in mesh_mark:
                                        if polar_bcflap.cor3dmark == mark_llt.name:
                                            polar_bcflap.cor3dnodes = mark_llt.node
                                            polar_bcflap.cor3dnodes_ind = []
                                            polar_bcflap.flag_cor3dnodes_ind = 1
                                polar_bctot.append(polar_bcflap)
                                polar_bctot_ind.append(polar_bcflap_ind)
                                break
                            polar_bcflap_ind += 1
                    cdstflap_value0   = np.zeros((len(mark.node),))
                    cdstflap_value1   = np.zeros((len(mark.node),))
                    clstflap_value0   = np.zeros((len(mark.node),))
                    clstflap_value1   = np.zeros((len(mark.node),))
                    cmstflap_value0   = np.zeros((len(mark.node),))
                    cmstflap_value1   = np.zeros((len(mark.node),))
                    cddynflap_value0   = np.zeros((len(mark.node),))
                    cddynflap_value1   = np.zeros((len(mark.node),))
                    cldynflap_value0   = np.zeros((len(mark.node),))
                    cldynflap_value1   = np.zeros((len(mark.node),))
                    cmdynflap_value0   = np.zeros((len(mark.node),))
                    cmdynflap_value1   = np.zeros((len(mark.node),))
                    b_chordflap       = np.zeros((len(mark.node),))
                    chord_vecflap     = np.zeros((len(mark.node),3))
                    non_acac_distflap = np.zeros((len(mark.node),))
                    non_eaac_distflap = np.zeros((len(mark.node),))
                    non_eamc_distflap = np.zeros((len(mark.node),))
                    aoa0_geomflap    = np.zeros((len(mark.node),))
                    aoa1_geomflap    = np.zeros((len(mark.node),))
                    coreflapdist_adim = np.zeros((len(mark.node),))
                    deltaf_0lift     = np.zeros((len(mark.node),))
                    cd_aoa0flap       = np.zeros((len(mark.node),))
                    # Add the value of the longitude simulated in each beam element node
                    for iiaero_node in np.arange(len(mark.node)):
                        iimark_node = int(mark.node.flatten()[iiaero_node])
                        lpos[iiaero_node] = np.linalg.norm(mesh_data.point[iimark_node,:]-mesh_data.point[bound.refpoint])
                    LL_cum = np.max(lpos)
                    # If more than 1 polar
                    if polar_bc.pointfile[0] != -1:
                        pointref = np.where(mark.node==polar_bc.pointfile)[0].tolist()
                        lrefpol  = lpos[pointref]
                        polar_bc.iirefpol = np.argsort(lrefpol)
                        polar_bc.sortedlref = lrefpol[polar_bc.iirefpol]  
                    # For all the nodes in the surface
                    # cent_secx       : center of the airfoil x position
                    # cent_secy       : center of the airfoil y position
                    # acac_dist       : distance between the aerodynamic reference and the 0.25 chord distance
                    # acac_dist_sign  : sign of the distance between the aerodynamic reference and the 0.25 chord distance
                    # non_acac_dist   : nondimensional distance between the aerodynamic reference and the 0.25 chord distance with the moment sign
                    # eaac_dist       : distance between the aerodynamic reference and the elastic reference
                    # eaac_dist_sign  : sign of the distance between the aerodynamic reference and the elastic reference
                    # non_eaac_dist   : nondimensional distance between the aerodynamic reference and the elastic reference with the moment sign
                    # eamc_dist       : distance between the aerodynamic reference and the mean chord
                    # eamc_dist_sign  : sign of the distance between the aerodynamic reference and the mean chord
                    # non_eamc_dist   : nondimensional distance between the aerodynamic reference and the mean chord with the moment sign
                    for iiaero_node in np.arange(len(mark.node)):
                        iimark_node                = int(mark.node.flatten()[iiaero_node])
                        cent_secx                  = (section_globalCS.point_3d[iimark_node][section.te[iimark_node],0]+section_globalCS.point_3d[iimark_node][section.le[iimark_node],0])/2 #(section.points[iimark_node][0][section.te[iimark_node],0]+section.points[iimark_node][0][section.le[iimark_node],0])/2
                        cent_secy                  = (section_globalCS.point_3d[iimark_node][section.te[iimark_node],1]+section_globalCS.point_3d[iimark_node][section.le[iimark_node],1])/2 #(section.points[iimark_node][0][section.te[iimark_node],1]+section.points[iimark_node][0][section.le[iimark_node],1])/2
                        chord_vec[iiaero_node,:]   = section_globalCS.point_3d[iimark_node][section.te[iimark_node],:]-section_globalCS.point_3d[iimark_node][section.le[iimark_node],:] #section.points[iimark_node][0][section.te[iimark_node],:]-section.points[iimark_node][0][section.le[iimark_node],:]
                        b_chord[iiaero_node]       = np.linalg.norm(chord_vec[iiaero_node,:])/2
                        acac_dist                  = section_globalCS.aero_cent[iimark_node]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section_globalCS.point_3d[iimark_node][section.le[iimark_node],0]) #[section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]]-b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimark_node][0][section.le[iimark_node],:]
                        acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_acac_dist[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
                        eaac_dist                  = section_globalCS.aero_cent[iimark_node] #[section.ae_orig_x[iimark_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimark_node]-sol_phys.ysc[iiaero_node]]
                        eaac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],eaac_dist),np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_eaac_dist[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
                        eamc_dist                  = [-cent_secx,-cent_secy,0] #[sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
                        eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
                        non_eamc_dist[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chord[iiaero_node]   
                        if bound.f_flap == 1 and markflap.node.flatten()[iiaero_node] >= 0:
                            iimarkflap_node                = int(markflap.node.flatten()[iiaero_node])
                            if iimarkflap_node > 0:
                                cent_secx                  = (section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],0]+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],0])/2 #(section.points[iimarkflap_node][0][section.te[iimarkflap_node],0]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],0])/2
                                cent_secy                  = (section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],1]+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],1])/2 #(section.points[iimarkflap_node][0][section.te[iimarkflap_node],1]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],1])/2
                                chord_vecflap[iiaero_node,:]   = section_globalCS.point_3d[iimarkflap_node][section.te[iimarkflap_node],:]-section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],:] #section.points[iimarkflap_node][0][section.te[iimarkflap_node],:]-section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
                                b_chordflap[iiaero_node]       = np.linalg.norm(chord_vecflap[iiaero_node,:])/2
                                acac_dist                  = section_globalCS.aero_cent[iimarkflap_node]-(b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section_globalCS.point_3d[iimarkflap_node][section.le[iimarkflap_node],0]) #[section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]]-b_chordflap[iiaero_node]/2*chord_vecflap[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
                                acac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimarkflap_node,0,:,:],acac_dist),np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_acac_distflap[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
                                eaac_dist                  = section_globalCS.aero_cent[iimark_node] #[section.ae_orig_x[iimarkflap_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimarkflap_node]-sol_phys.ysc[iiaero_node]]
                                eaac_dist_sign             = np.sign(np.dot(np.matmul(rot_matT_p[iimarkflap_node,0,:,:],eaac_dist),np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_eaac_distflap[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/(2*b_chord[iiaero_node])
                                eamc_dist                  = [-cent_secx,-cent_secy,0] # [sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
                                eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
                                non_eamc_distflap[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chordflap[iiaero_node] 
                                coreflapdist  = mesh_data.point[iimark_node,:]-mesh_data.point[iimarkflap_node,:]
                                coreflapdist_adim[iiaero_node] = np.sign(np.dot(coreflapdist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))/(b_chord[iiaero_node]+b_chordflap[iiaero_node])                         
#                        cent_secx                  = (section.points[iimark_node][0][section.te[iimark_node],0]+section.points[iimark_node][0][section.le[iimark_node],0])/2
#                        cent_secy                  = (section.points[iimark_node][0][section.te[iimark_node],1]+section.points[iimark_node][0][section.le[iimark_node],1])/2
#                        chord_vec[iiaero_node,:]   = section.points[iimark_node][0][section.te[iimark_node],:]-section.points[iimark_node][0][section.le[iimark_node],:]
#                        b_chord[iiaero_node]       = np.linalg.norm(chord_vec[iiaero_node,:])/2
#                        acac_dist                  = [section.ae_orig_x[iimark_node],section.ae_orig_y[iimark_node]]-b_chord[iiaero_node]/2*chord_vec[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimark_node][0][section.le[iimark_node],:]
#                        acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)[:2]))
#                        non_acac_dist[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
#                        eaac_dist                  = [section.ae_orig_x[iimark_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimark_node]-sol_phys.ysc[iiaero_node]]
#                        eaac_dist_sign             = np.sign(np.dot(eaac_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)[:2]))
#                        non_eaac_dist[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/bound.l_ref
#                        eamc_dist                  =   [-cent_secx,-cent_secy,0]#[sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
#                        eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimark_node,0,:,:],bound.vdir)))
#                        non_eamc_dist[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chord[iiaero_node]   
#                        if bound.f_flap == 1 and markflap.node.flatten()[iiaero_node] >= 0:
#                            iimarkflap_node                = int(markflap.node.flatten()[iiaero_node])
#                            cent_secx                  = (section.points[iimarkflap_node][0][section.te[iimarkflap_node],0]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],0])/2
#                            cent_secy                  = (section.points[iimarkflap_node][0][section.te[iimarkflap_node],1]+section.points[iimarkflap_node][0][section.le[iimarkflap_node],1])/2
#                            chord_vecflap[iiaero_node,:]   = section.points[iimarkflap_node][0][section.te[iimarkflap_node],:]-section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
#                            b_chordflap[iiaero_node]       = np.linalg.norm(chord_vecflap[iiaero_node,:])/2
#                            acac_dist                  = [section.ae_orig_x[iimarkflap_node],section.ae_orig_y[iimarkflap_node]]-b_chordflap[iiaero_node]/2*chord_vecflap[iiaero_node,:]/np.linalg.norm(chord_vec[iiaero_node,:])+section.points[iimarkflap_node][0][section.le[iimarkflap_node],:]
#                            acac_dist_sign             = np.sign(np.dot(acac_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)[:2]))
#                            non_acac_distflap[iiaero_node] = acac_dist_sign*np.linalg.norm(acac_dist)/bound.l_ref
#                            eaac_dist                  = [section.ae_orig_x[iimarkflap_node]-sol_phys.xsc[iiaero_node],section.ae_orig_y[iimarkflap_node]-sol_phys.ysc[iiaero_node]]
#                            eaac_dist_sign             = np.sign(np.dot(eaac_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)[:2]))
#                            non_eaac_distflap[iiaero_node] = eaac_dist_sign*np.linalg.norm(eaac_dist)/bound.l_ref
#                            eamc_dist                  =  [sol_phys.xsc[iiaero_node]-cent_secx,sol_phys.ysc[iiaero_node]-cent_secy,0]
#                            eamc_sign                  = np.sign(np.dot(eamc_dist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))
#                            non_eamc_distflap[iiaero_node] = eamc_sign*np.linalg.norm(eamc_dist)/b_chordflap[iiaero_node] 
#                            coreflapdist  = mesh_data.point[iimark_node,:]-mesh_data.point[iimarkflap_node,:]
#                            coreflapdist_adim[iiaero_node] = np.sign(np.dot(coreflapdist,np.matmul(rot_matT_p[iimarkflap_node,0,:,:],bound.vdir)))/(b_chord[iiaero_node]+b_chordflap[iiaero_node])  
                    # If a rotative aerodynamic model is selected the nondimensional radius is calculated
                    # flag_start    : flag to start the bisection method
                    # flag_start1   : flag to start the bisection method
                    # flag_start2   : flag to start the bisection method
                    # r_vinduced    : nondimensional radius
                    # cltot2        : total value of the lift coefficient in the previous iteration
                    # cttot2        : total value of the thrust coefficient in the previous iteration
                    # v_induced     : induced velocity 
                    # Gamma_induced : induced circulation
                    # tol_vind      : tolerance of the induced velocity error
                    # aux_vind_e    : index to count the number of iterations in the induced velocity calculation loop
                    # lpos_cir      : distance to the root of the aerodynamic surface in circular coordinates
                    # lpc_in        : position of the section relative to the total length
                    # mat_pllt      : lifting line theory matrix
                    # vec_pllt      : lifting line theory vector
                    if polar_bc.eff3d == 'LLT':
                        class eff_3d:
                            Gamma_induced = np.zeros((len(mark.node),))
                            cltot2 = 1e3
                            lpos_cir  = np.zeros((len(mark.node),))
                            sign_vi   = -1
                        v_induced = np.zeros((len(mark.node),3))
                        for iiaero_node in np.arange(len(mark.node)):
                            lpc_in = lpos[iiaero_node]/LL_cum
                            if lpc_in>1:
                                lpc_in = 1
                            eff_3d.lpos_cir[iiaero_node] = np.arccos(lpc_in) 
                    elif polar_bc.eff3d == 'BEM':
                        class eff_3d:
                            flag_start      = np.zeros((len(mark.node),))
                            flag_start1     = np.zeros((len(mark.node),))
                            flag_start2     = np.zeros((len(mark.node),))
                            error1          = np.ones((len(mark.node),))
                            error2          = np.ones((len(mark.node),))
                            sign_vi         = -np.sign(np.dot(bound.vdir,bound.refrot))
                            r_vinduced      = lpos/LL_cum
                            v_induced2      = np.zeros((len(mark.node),3))
                            masa            = np.zeros((len(mark.node),)) #sol_phys.m11
                            frelax_vt       = 1
                            errorout1       = 1
                            errorout2       = 2
                        for ii_massind in np.arange(len(mark.node)):
                            ii_mass = int(mark.node[ii_massind])
                            eff_3d.masa[ii_massind] = MM_global[ii_mass,ii_mass]
                        v_induced         = 0*bound.vrot*bound.radius*np.ones((len(mark.node),3))
                        v_induced[:,0]   *= eff_3d.sign_vi*bound.vdir[0]/np.linalg.norm(bound.vdir)
                        v_induced[:,1]   *= eff_3d.sign_vi*bound.vdir[1]/np.linalg.norm(bound.vdir)
                        v_induced[:,2]   *= eff_3d.sign_vi*bound.vdir[2]/np.linalg.norm(bound.vdir)
                        vtan_induced = np.zeros((len(mark.node),3))
                        eff_3d.v_induced_tan0     = np.zeros((len(mark.node),3))
                        eff_3d.v_induced_tan1     = np.zeros((len(mark.node),3))
                        eff_3d.v_induced1 = v_induced.copy()
                    else: 
                        v_induced = np.zeros((len(mark.node),3))                         
                        class eff_3d:
                            sign_vi         = 1
                    try:
                        tol_vind = polar_bc.vind_tol
                    except:
                        tol_vind = 1e-5
                    try:
                        maxit_vind = polar_bc.vind_maxit
                    except:
                        maxit_vind = 1e3
                    eff_3d.aux_vind_e_out = 0 
                    # initialize the error
                    error_out = tol_vind*10
                    flag_conv = 0
                    flag_convout = 0
                    # While error is  higher than the required tolerance
                    while abs(error_out)>tol_vind or flag_convout == 0:
                        if abs(error_out)<tol_vind:
                            if polar_bc.eff3d == 'BEM':
                                vtan_induced_mod = vtan_induced.copy()
                                vtan_ind_int = vtan_induced[polar_bc.lltnodes_ind,:]
                                l_ind_int = lpos[polar_bc.lltnodes_ind]
                                f_vxind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,0],'cubic', fill_value='extrapolate')
                                f_vyind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,1],'cubic', fill_value='extrapolate')
                                f_vzind = interpolate.interp1d(l_ind_int,vtan_ind_int[:,2],'cubic', fill_value='extrapolate') 
                                for auxaeronode in np.arange(len(mark.node)):
                                    vtan_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
                                vtan_induced = vtan_induced_mod.copy()
                            flag_convout = 1
                        eff_3d.aux_vind_e_out += 1
                        # if the loop cannot find the solution in a determined number of iterations, stop the loop
                        if eff_3d.aux_vind_e_out > maxit_vind: 
                            print('Induced tangential velocity: Not converged - '+str(error_out))
                            break
                        # error  : error of the lift coefficient in each iteration
                        error  = tol_vind*10
                        eff_3d.error = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.error1ct = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.error2ct = np.ones((len(mark.node),))*tol_vind*10
                        eff_3d.flag_start = np.zeros((len(mark.node),))
                        eff_3d.flag_start1 = np.zeros((len(mark.node),))
                        eff_3d.flag_start2 = np.zeros((len(mark.node),))
                        eff_3d.aux_vind_e = 1 
                        while abs(error)>tol_vind or flag_conv == 0:
                            ang_vec0    = np.zeros((len(mark.node),))
                            ang_vec1    = np.zeros((len(mark.node),))
                            if abs(error)<tol_vind:
                                if polar_bc.eff3d == 'BEM':
                                    v_induced_mod = v_induced.copy()
                                    v_ind_int = v_induced[polar_bc.lltnodes_ind,:]
                                    l_ind_int = lpos[polar_bc.lltnodes_ind]
                                    f_vxind = interpolate.interp1d(l_ind_int,v_ind_int[:,0],'cubic', fill_value='extrapolate')
                                    f_vyind = interpolate.interp1d(l_ind_int,v_ind_int[:,1],'cubic', fill_value='extrapolate')
                                    f_vzind = interpolate.interp1d(l_ind_int,v_ind_int[:,2],'cubic', fill_value='extrapolate') 
                                    for auxaeronode in np.arange(len(mark.node)):
                                        v_induced_mod[auxaeronode,:] = [f_vxind(lpos[auxaeronode]),f_vyind(lpos[auxaeronode]),f_vzind(lpos[auxaeronode])]
                                    v_induced = v_induced_mod.copy()
                                flag_conv = 1
                            # if the loop cannot find the solution in a determined number of iterations, stop the loop
                            if eff_3d.aux_vind_e > maxit_vind: 
                                print('Induced velocity: Not converged - '+str(error))
                                break
                            eff_3d.aux_vind_e += 1
                            # Define variables for the rotative aerodynamics
                            if polar_bc.eff3d == 'LLT':            
                                eff_3d.mat_pllt  = np.zeros((2*len(mark.node),len(mark.node)))
                                eff_3d.vec_pllt  = np.zeros((2*len(mark.node),))
                            elif polar_bc.eff3d == 'BEM':
                                # r_vinduced     : nondimensional distance of the blade radius
                                # vi_vinduced    : induced velocity
                                # phi_vinduced   : induced velocity on the blade
                                # f_vinduced     : 3D effect factor of the blade
                                # F_vinduced     : 3D effect factor of the blade
                                # dct_dr_induced : variation of thrust with radial distance
                                eff_3d.phi_vinduced0       = np.zeros((len(mark.node),))
                                eff_3d.f_vinduced0         = np.zeros((len(mark.node),))
                                eff_3d.F_vinduced0         = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced0 = np.zeros((len(mark.node),))
                                eff_3d.phi_vinduced1       = np.zeros((len(mark.node),))
                                eff_3d.f_vinduced1         = np.zeros((len(mark.node),))
                                eff_3d.F_vinduced1         = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced1 = np.zeros((len(mark.node),))
                                eff_3d.err_dct_dr_induced  = np.zeros((len(mark.node),))
                            # aoa_vec0    : angle of attack of node 0
                            # aoa_vec1    : angle of attack of node 1
                            # aoap_vec0   : angle of attack derivative of node 0
                            # aoap_vec1   : angle of attack derivative of node 1
                            # aoa_val     : value of the angle of attack
                            # aoap_val    : value of the angle of attack derivative
                            # aoapp_val   : value of the angle of attack second derivative
                            # pos_vec0    : position vector of node 0
                            # pos_vec1    : position vector of node 1
                            aoa_vec0    = np.zeros((len(mark.node),))
                            aoa_vec1    = np.zeros((len(mark.node),))
                            aoa_val     = np.zeros((len(mark.node),))
                            pos_vec0    = np.zeros((len(mark.node),))
                            pos_vec1    = np.zeros((len(mark.node),))
                            aoa_vec0_st = np.zeros((len(mark.node),))
                            aoa_vec1_st = np.zeros((len(mark.node),))
                            Vinf0       = np.zeros((len(mark.node),))
                            Vinf1       = np.zeros((len(mark.node),))
                            reynolds0   = np.zeros((len(mark.node),))
                            reynolds1   = np.zeros((len(mark.node),))
                            deltaflap_vec0    = np.zeros((len(mark.node),))
                            deltaflap_vec1    = np.zeros((len(mark.node),))
                            # Calculate the reference axis
                            refaxis_0, refaxis_1, refaxis2_0, refaxis2_1, vdir0, vdir1, aoa0_geom, aoa1_geom, eff_3d = ref_axis_aero(bound,mark,polar_bc,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,v_induced,vdir0,vdir1,aoa0_geom,aoa1_geom,rr_vec_p,rot_matT_p,chord_vec,eff_3d,0)
                            if bound.f_flap == 1:
                                # Calculate the reference axis
                                aoa0_geomflap, aoa1_geomflap = ref_axis_aero(bound,markflap,polar_bcflap,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,v_induced,vdir0,vdir1,aoa0_geomflap,aoa1_geomflap,rr_vec_p,rot_matT_p,chord_vecflap,eff_3d,1)
                            # for each node in the surface
                            for iiaero_node in np.arange(len(mark.node)):
                                iimark_node = int(mark.node.flatten()[iiaero_node])
                                # Calculate the velocity of the free stream for the airfoil for the different aerodynamic models
                                # refvind0       : reference induced velocity direction in element 0
                                # refvind1       : reference induced velocity direction in element 1
                                # vi_vinduced0   : induced velocity scalar in element 0
                                # vi_vinduced1   : induced velocity scalar in element 1
                                # Vinf0          : free stream velocity in element 0
                                # Vinf1          : free stream velocity in element 1
                                # Vinf_vindaoa   : velocity to calculate the angle of the induced angle of attack
                                if polar_bc.eff3d == 'LLT':
                                    refvind0           = -refaxis2_0[iiaero_node,:]
                                    refvind1           = -refaxis2_1[iiaero_node,:]
                                    vi_vinduced0       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(refvind0))
                                    vi_vinduced1       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(refvind1))
                                    Vinf0[iiaero_node] = np.sqrt(bound.vinf**2+vi_vinduced0**2)
                                    Vinf1[iiaero_node] = np.sqrt(bound.vinf**2+vi_vinduced1**2)
                                    Vinf_vindaoa       = bound.vinf
                                elif polar_bc.eff3d == 'BEM':
                                    refvind0           = eff_3d.sign_vi*bound.vdir
                                    refvind1           = eff_3d.sign_vi*bound.vdir
                                    vi_vinduced0       = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_induced[iiaero_node,:]),np.array(bound.vdir))
                                    vi_vinduced1       = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_induced[iiaero_node,:]),np.array(bound.vdir))
                                    vt_vinduced0       = np.dot(eff_3d.v_induced_tan0[iiaero_node,:],eff_3d.vec_vrot0[iiaero_node,:])
                                    vt_vinduced1       = np.dot(eff_3d.v_induced_tan1[iiaero_node,:],eff_3d.vec_vrot1[iiaero_node,:])
                                    Vinf0[iiaero_node] = np.sqrt((bound.vinf+vi_vinduced0)**2+(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius+vt_vinduced0)**2)+1e-10
                                    Vinf1[iiaero_node] = np.sqrt((bound.vinf+vi_vinduced1)**2+(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius+vt_vinduced1)**2)+1e-10
                                    v_induced_tan      = (eff_3d.v_induced_tan0[iiaero_node,:]*LL_vec_p[0,iiaero_node]+eff_3d.v_induced_tan1[iiaero_node,:]*LL_vec_p[1,iiaero_node])/(LL_vec_p[0,iiaero_node]+LL_vec_p[1,iiaero_node])
                                    vtan_induced[iiaero_node] = v_induced_tan
                                    Vinf_vindaoa       = np.linalg.norm(bound.vrot*eff_3d.r_vinduced[iiaero_node]*bound.radius*eff_3d.vec_vrot0[iiaero_node,:]+v_induced_tan)+1e-10
                                else:
                                    refvind0           = -refaxis2_0[iiaero_node,:]
                                    refvind1           = -refaxis2_1[iiaero_node,:]
                                    Vinf0[iiaero_node] = bound.vinf 
                                    Vinf1[iiaero_node] = bound.vinf 
                                    Vinf_vindaoa       = bound.vinf
                                vec_ang           = [phi_val[iimark_node],psi_val[iimark_node],theta_val[iimark_node]]
                                if bound.f_flap == 1:
                                    vec_angflap           = [phi_val[iimark_nodeflap],psi_val[iimark_nodeflap],theta_val[iimark_nodeflap]]
                                # For lifting line theory set the angle of attack at the tip as zero, calculate in the rest of points
                                if (polar_bc.eff3d == 'LLT' ) and lpos[iiaero_node] > 0.999*LL_cum:  # or polar_bc.eff3d == 'BEM'
                                    aoa_vec0[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoa_vec1[iiaero_node]    = aoa_0lift[iiaero_node]
                                    aoa_val[iiaero_node]     = aoa_0lift[iiaero_node]
                                    deltaflap_vec0[iiaero_node]    = deltaf_0lift[iiaero_node]
                                    deltaflap_vec1[iiaero_node]    = deltaf_0lift[iiaero_node]
                                    v_induced[iiaero_node,:] = np.matmul(rot_mat_p[iimark_node,1,:,:],Vinf0[iiaero_node]*np.tan(aoa_val[iiaero_node]-aoa0_geom[iiaero_node])*np.array(refvind1))
                                # In the rest of the points
                                else:
                                    # if needed include rotational effects
                                    if polar_bc.eff3d == 'BEM':
                                        if eff_3d.r_vinduced[iiaero_node] == 0:
                                            eff_3d.phi_vinduced0[iiaero_node] = np.arctan((bound.vinf)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                            eff_3d.phi_vinduced1[iiaero_node] = np.arctan((bound.vinf)/np.max([bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node],1e-10])) 
                                        else:
                                            eff_3d.phi_vinduced0[iiaero_node] = np.arctan((bound.vinf+vi_vinduced0)/np.max([np.linalg.norm(bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node]*eff_3d.vec_vrot0[iiaero_node,:]+eff_3d.v_induced_tan0[iiaero_node,:]),1e-10])) 
                                            eff_3d.phi_vinduced1[iiaero_node] = np.arctan((bound.vinf+vi_vinduced1)/np.max([np.linalg.norm(bound.vrot*bound.radius*eff_3d.r_vinduced[iiaero_node]*eff_3d.vec_vrot1[iiaero_node,:]+eff_3d.v_induced_tan1[iiaero_node,:]),1e-10])) 
                                        eff_3d.f_vinduced0[iiaero_node]   = bound.Nb/2*((1-(eff_3d.r_vinduced[iiaero_node]))/np.max([eff_3d.r_vinduced[iiaero_node]*eff_3d.phi_vinduced0[iiaero_node],1e-10]))
                                        eff_3d.f_vinduced1[iiaero_node]   = bound.Nb/2*((1-(eff_3d.r_vinduced[iiaero_node]))/np.max([eff_3d.r_vinduced[iiaero_node]*eff_3d.phi_vinduced1[iiaero_node],1e-10]))
                                        eff_3d.F_vinduced0[iiaero_node]   = 2/np.pi*np.arccos(np.exp(-eff_3d.f_vinduced0[iiaero_node]))
                                        eff_3d.F_vinduced1[iiaero_node]   = 2/np.pi*np.arccos(np.exp(-eff_3d.f_vinduced1[iiaero_node]))
                                        v_vert                            = eff_3d.sign_vi*(bound.vinf*bound.vdir+v_induced[iiaero_node,:])
                                    else:
                                        v_vert = eff_3d.sign_vi*v_induced[iiaero_node,:]
                                    ang_vec0[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],vec_ang),np.array(refaxis_0[iiaero_node,:]))
                                    ang_vec1[iiaero_node]    = np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],vec_ang),np.array(refaxis_1[iiaero_node,:]))
                                    aoa_vec0[iiaero_node] = (aoa0_geom[iiaero_node])+ang_vec0[iiaero_node]-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_vert),np.array(refvind0))/Vinf_vindaoa)
                                    aoa_vec1[iiaero_node] = (aoa1_geom[iiaero_node])+ang_vec1[iiaero_node]-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_vert),np.array(refvind1))/Vinf_vindaoa)
                                    aoa_val[iiaero_node]  = (aoa_vec0[iiaero_node]*LL_vec_p[0,iiaero_node]+aoa_vec1[iiaero_node]*LL_vec_p[1,iiaero_node])/(LL_vec_p[0,iiaero_node]+LL_vec_p[1,iiaero_node])
                                    deltaflap_vec0[iiaero_node]    = (aoa0_geomflap[iiaero_node]+ang_vec0[iiaero_node]-aoa0_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,0,:,:],v_vert),np.array(refvind0))/Vinf_vindaoa)
                                    deltaflap_vec1[iiaero_node]    = (aoa1_geomflap[iiaero_node]+ang_vec1[iiaero_node]-aoa1_geom[iiaero_node])-eff_3d.sign_vi*np.arctan(np.dot(np.matmul(rot_matT_p[iimark_node,1,:,:],v_vert),np.array(refvind1))/Vinf_vindaoa)
                                aoa_vec0_st[iiaero_node]  = aoa0_geom[iiaero_node]
                                aoa_vec1_st[iiaero_node]  = aoa1_geom[iiaero_node]
                                pos_vec0[iiaero_node] = 0
                                pos_vec1[iiaero_node] = 0
                                reynolds0[iiaero_node] = Vinf0[iiaero_node]*2*b_chord[iiaero_node]*bound.rho/bound.mu
                                reynolds1[iiaero_node] = Vinf1[iiaero_node]*2*b_chord[iiaero_node]*bound.rho/bound.mu
#                                    print([reynolds0,reynolds1])
                            case_setup, polar_bc_ind,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,cmst_tot_value0,cmst_tot_value1,eff_3d,error,cl_alpha,cl_alphaflap,error2,v_induced = steady_aero(case_setup,bound,polar_bctot,mesh_data,marktot,polar_bctot_ind,aoa_vec0,aoa_vec1,aoa_vec0_st,aoa_vec1_st,pos_vec0,pos_vec1,cdst_value0,cdst_value1,clst_value0,clst_value1,cmst_value0,cmst_value1,cdstflap_value0,cdstflap_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,aoa_0lift,aoa_0liftflap,v_induced,refaxis2_0,refaxis2_1,Vinf0,Vinf1,LL_vec_p,lpos,LL_cum,eff_3d,tol_vind,b_chord,non_acac_dist,b_chordflap,non_acac_distflap,coreflapdist_adim,aoa_vec0,aoa_vec1,cd_aoa0,cd_aoa0flap,aoa0_geom,aoa1_geom,reynolds0,reynolds1,deltaflap_vec0,deltaflap_vec1,bound.f_flap)
                        if polar_bc.eff3d == 'BEM':
                            eff_3d,error_out,vtan_induced = converge_vtan(bound,mark,polar_bc,cdst_tot_value0,cdst_tot_value1,clst_tot_value0,clst_tot_value1,v_induced,LL_vec_p,eff_3d,b_chord,vtan_induced)
                        else:
                            error_out = tol_vind/10
                    FF_global,cl_value0,cl_value1,cm_value0,cm_value1,disp_values = total_forces(clst_value0,clst_value1,cmst_value0,cmst_value1,cdst_value0,cdst_value1,cldyn_value0,cldyn_value1,cmdyn_value0,cmdyn_value1,cddyn_value0,cddyn_value1,clstflap_value0,clstflap_value1,cmstflap_value0,cmstflap_value1,cdstflap_value0,cdstflap_value1,cldynflap_value0,cldynflap_value1,cmdynflap_value0,cmdynflap_value1,cddynflap_value0,cddynflap_value1,LL_vec_p,rot_mat_p,b_chord,b_chordflap,lpos,Vinf0,Vinf1,vdir0,vdir1,refaxis_0,refaxis_1,refaxis2_0,refaxis2_1,FF_global,bound,section,polar_bc,marktot,disp_values,num_point,0,0,1,[],eff_3d,bound.f_flap)
                    FF_orig = FF_global.copy()
                    # save values of the aerodynamic parameters
                    disp_values.aoa_val   = aoa_vec0
                    disp_values.clst_value0 = clst_value0
                    disp_values.clst_value1 = clst_value1
                    disp_values.cmst_value0 = cmst_value0
                    disp_values.cmst_value1 = cmst_value1
                    disp_values.cdst_value0 = cdst_value0
                    disp_values.cdst_value1 = cdst_value1
                    disp_values.cldyn_value0 = cldyn_value0
                    disp_values.cldyn_value1 = cldyn_value1
                    disp_values.cmdyn_value0 = cmdyn_value0
                    disp_values.cmdyn_value1 = cmdyn_value1
                    disp_values.cddyn_value0 = cddyn_value0
                    disp_values.cddyn_value1 = cddyn_value1
                    if bound.vinf == "VINF_DAT":
                        bound.vinf = vel2
                        bound.vdir = vel2dir
                elif bound.type == "BC_DISP":
                    # If bc is a displacement in a node, add the value to the global displacement vector
                    for node in mark.node:
                        if node<num_point:
                            if all(np.isnan(qq_global[int(9*node):int(9*node+9)])):
                                qq_global[int(9*node):int(9*node+9)] = bound.values
                            else:
                                qq_global[int(9*node):int(9*node+9)] += bound.values
                        else:
                            if all(np.isnan(qq_global[int(9*node):int(9*node+9)])):
                                qq_global[int(9*node):] = bound.values
                            else:
                                qq_global[int(9*node):] += bound.values
                elif bound.type == "BC_FUNC":
                    # If the bc is a function of the vibration modes
                    try:
                        # mode_bc      : vibration mode used for the load
                        # vibfree      : information of the free vibration analysis
                        # sec_coor_vib : information of the section points used in the vibration analysis
                        # func_bc      : function used for the bc
                        mode_bc               = int(bound.funcmode)
                        vibfree, sec_coor_vib = solver_vib_mod(case_setup,sol_phys,mesh_data,section)
                        func_bc               = np.zeros((len(mark.node),9))
                        # for every node in the marker determine the function
                        for node in mark.node:
                            node = int(node)
                            # The function is calculated as the projection in the bc values_load 
                            # vector of the modal shape
                            func_bc[node,:] = np.dot([vibfree.u[node,mode_bc],vibfree.v[node,mode_bc],vibfree.w[node,mode_bc],\
                                       vibfree.phi[node,mode_bc],vibfree.psi[node,mode_bc],vibfree.theta[node,mode_bc],\
                                       vibfree.phi_d[node,mode_bc],vibfree.psi_d[node,mode_bc],vibfree.theta_d[node,mode_bc]],bound.values_load)
                        # For every mesh marker if the marker name is the normalization function 
                        # name divide the value by the value in the normalization node and multiplied 
                        # by the value specified in the setup
                        for mark2 in mesh_mark:
                            if mark2.name == bound.funcnorm:
                                for aux_mark2 in np.arange(9):
                                    if func_bc[int(mark2.node[0]),aux_mark2] != 0:
                                        func_bc /= func_bc[int(mark2.node[0]),aux_mark2]*np.array(bound.values)
                    except:
                        pass
                    # The values calculated preivously are updated to the forces vector         
                    for elem in mesh_data.elem:
                        if len(np.where(mark.node==elem[1])[0])>0:
                            FF_global[int(9*elem[1]):int(9*elem[1]+9)]+=func_bc[int(elem[1])]*np.linalg.norm(mesh_data.point[int(elem[1]),:]-mesh_data.point[int(elem[2]),:])/2
                        if len(np.where(mark.node==elem[2])[0])>0:
                            FF_global[int(9*elem[2]):int(9*elem[2]+9)]+=func_bc[int(elem[2])]*np.linalg.norm(mesh_data.point[int(elem[1]),:]-mesh_data.point[int(elem[2]),:])/2
                elif bound.type == "BC_JOINT":
                    # If the bc is a joint between nodes
                    # node_1 : node 1 of the joint
                    # node_2 : node 2 of the joint
                    node_1 = int(mark.node[0])
                    node_2 = int(mark.node[1])
                    # xvec : distance between nodes in axis x
                    # yvec : distance between nodes in axis y
                    # zvec : distance between nodes in axis z
                    xvec = mesh_data.point[node_2,0]-mesh_data.point[node_1,0]
                    yvec = mesh_data.point[node_2,1]-mesh_data.point[node_1,1]
                    zvec = mesh_data.point[node_2,2]-mesh_data.point[node_1,2]
                    if bound.joint_type == "FIXED":
                        # If the boundary condition is a rigid solid connection between bodies
                        for ii_fixed in np.arange(9):
                            # for the 9 rows of the node 1
                            # add the values of node 2 in stiffness and loads
                            FF_global[node_1*9+ii_fixed] += FF_global[node_2*9+ii_fixed]
                            # The moment added by the distance between reference points
                            if ii_fixed == 0:
                                # moment added by force in x axis
                                FF_global[node_1*9+4] += zvec*FF_global[node_2*9]
                                FF_global[node_1*9+5] += -yvec*FF_global[node_2*9]
                            elif ii_fixed == 1:
                                # moment added by force in y axis
                                FF_global[node_1*9+3] += -zvec*FF_global[node_2*9+1]
                                FF_global[node_1*9+5] += xvec*FF_global[node_2*9+1]
                            elif ii_fixed == 2:
                                # moment added by force in z axis
                                FF_global[node_1*9+3] += yvec*FF_global[node_2*9+2]
                                FF_global[node_1*9+4] += -xvec*FF_global[node_2*9+2]
                        # The rows of the node 2 must be deleted and the relation between displacements is added
                        FF_global[node_2*9:node_2*9+9] = np.zeros((9,))
                    elif bound.joint_type == "ROTATE_AXIS":
                        # if joint is a rotation joint
                        # a_axis : axis of rotation of the beam
                        # r_axis : unitary vector of the distance between nodes
                        # r_0_2  : distance from point of rotation to section 2
                        # r_1_0  : distance from section 1 to point of rotation
                        # n_axis : normal axis from rotation axis and nodes relative position
                        # x1     : distance between node 1 and joint in axis x
                        # y1     : distance between node 1 and joint in axis y
                        # x2     : distance between joint and node 2 in axis x
                        # y2     : distance between joint and node 2 in axis y
                        a_axis = np.array(bound.joint_axis)/np.linalg.norm(np.array(bound.joint_axis))
                        if np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])>0:
                            r_axis = (mesh_data.point[node_2]-mesh_data.point[node_1])/np.linalg.norm(mesh_data.point[node_2]-mesh_data.point[node_1])
                        else:
                            if a_axis[0] == 0:
                                r_axis = np.array([1,0,0])
                            elif a_axis[1] == 0:
                                r_axis = np.array([0,1,0])
                            elif a_axis[2] == 0:
                                r_axis = np.array([0,0,1])
                            else:
                                r_axis = np.array([a_axis[1],-a_axis[0],a_axis[2]])/np.linalg.norm(np.array([a_axis[1],-a_axis[0],a_axis[2]]))
                        r_0_2 = mesh_data.point[node_2]-bound.point_axis
                        r_0_1 = mesh_data.point[node_1]-bound.point_axis
                        x1 = r_0_1[0]
                        y1 = r_0_1[1]
                        x2 = r_0_2[0]
                        y2 = r_0_2[1]
                        n_axis = np.cross(a_axis,r_axis)
                        # rot_mat_axis : Rotation matrix of the degree of freedom. From rotation axis to global 
                        rot_mat_axis = np.array([[np.dot(r_axis,[1,0,0]),np.dot(n_axis,[1,0,0]),np.dot(a_axis,[1,0,0])],
                                                  [np.dot(r_axis,[0,1,0]),np.dot(n_axis,[0,1,0]),np.dot(a_axis,[0,1,0])],
                                                  [np.dot(r_axis,[0,0,1]),np.dot(n_axis,[0,0,1]),np.dot(a_axis,[0,0,1])]])
                        # stiffness matrix and load vector in the nodes is rotated to the joint axis
                        for ii_point in np.arange(num_point):
                                for ii_row in np.arange(3):
                                    FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3)] = np.matmul(rot_mat_axis,FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3)])
                        # For all the rows of the points in the stiffness matrix and load vector
                        for ii_fixed in np.arange(9):
                            # Apply the restrictions in all the displacements except of the rotation around the axis
                            if ii_fixed != 5 and ii_fixed != 8:                                
                                FF_global[node_1*9+ii_fixed]   += FF_global[node_2*9+ii_fixed]
                                # The moment added by the distance between reference points
                                if ii_fixed == 0:
                                    # moment added by force in x axis
                                    FF_global[node_1*9+4]   += zvec*FF_global[node_2*9]
                                    FF_global[node_1*9+5]   += -y1*FF_global[node_2*9]
                                elif ii_fixed == 1:
                                    # moment added by force in y axis
                                    FF_global[node_1*9+3] += -zvec*FF_global[node_2*9+1]
                                    FF_global[node_1*9+5] += -x1*FF_global[node_2*9+1]
                                elif ii_fixed == 2:
                                    # moment added by force in z axis
                                    FF_global[node_1*9+3]   += yvec*FF_global[node_2*9+2]
                                    FF_global[node_1*9+4]   += -xvec*FF_global[node_2*9+2] 
                                FF_global[node_2*9+ii_fixed]    = 0          
                                # The rows of the node 2 must be deleted and the relation between displacements is added
                        # The stiffness matrix and the load vector are rotated back to the global frame
                        for ii_point in np.arange(num_point):
                                for ii_row in np.arange(3):
                                    FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3)] = np.matmul(np.transpose(rot_mat_axis),FF_global[ii_point*9+3*ii_row:ii_point*9+(3*ii_row+3)])
    KK_global, MM_global, RR_global, case_setup, ref_axis = init_boundaryconditions_tran(KK_global,MM_global, [],mesh_mark,case_setup,num_point,sol_phys,mesh_data,section,section_globalCS,rot_mat_p,rot_matT_p,LL_vec_p,rr_vec_p)
    return  KK_global, qq_global, FF_global, FF_orig, RR_global, disp_values, mesh_data    
