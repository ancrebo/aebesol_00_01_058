# -*- coding: utf-8 -*-
"""
Created on Tue Dec 22 10:10:30 2020

@author                 : Andres Cremades Botella - ancrebo@mot.upv.es
global_matix_definition : file containing functions for the definition of the global matrices of the system
last_version            : 17-02-2021
modified_by             : Andres Cremades Botella
"""

import numpy as np
from bin.matrix_definition_2options import mass_matrix, stiffness_matrix, pointmass_matrix
from scipy import interpolate
from bin.aux_functions import  def_vec_param

#%% Functions
def define_stiffness(mesh_elem,sol_phys,mesh_point,num_point,num_elem,lock,RR_ctrl,section):
    # Definition of the stiffness matrix
    # ------------------------------------------------------------------------
    # initialization of variables
    # KK_global_prima : stiffness matrix before joints
    # LL_vec          : distance between nodes vector
    # rr_vec          : vectorial distance between nodes
    # xxloc_vec       : internal frame x axis in global coordinates
    # yyloc_vec       : internal frame y axis in global coordinates
    # zzloc_vec       : internal frame z axis in global coordinates
    # rot_mat         : rotation matrix of each beam element
    # rotT_mat        : transposed rotation matrix of each beam element
    # Lrot_mat        : complete rotation matrix of the system
    # LrotT_mat       : complete transposed rotation matrix of the system
    # LL_vec_p        : distance between nodes vector for each node in the element
    # rr_vec_p        : vectorial distance between nodes for each node in the element
    # rot_mat_p       : rotation matrix of each beam element for each node in the element
    # rotT_mat_p      : transposed rotation matrix of each beam element for each node in the element
    KK_global_prima = np.zeros((9*num_point,9*num_point))
    LL_vec          = np.zeros((num_elem,))
    rr_vec          = np.zeros((num_elem,3))
    xxloc_vec       = np.zeros((num_elem,3))
    yyloc_vec       = np.zeros((num_elem,3))
    zzloc_vec       = np.zeros((num_elem,3))
    rot_mat         = np.zeros((num_elem,3,3))
    rotT_mat        = np.zeros((num_elem,3,3))
    Lrot_mat        = np.zeros((num_elem,9,9))
    LrotT_mat       = np.zeros((num_elem,9,9))
    LL_vec_p        = np.zeros((2,num_point))
    rr_vec_p        = np.zeros((2,num_point,3))
    rot_mat_p       = np.zeros((num_point,2,3,3))
    rot_matT_p      = np.zeros((num_point,2,3,3))
    # for each beam element
    for elem in mesh_elem:
        # define a class to import the section characteristics in the beam element
        # definition of the stiffness matrix
        # sol_phys_sec  : class to import stiffness characteristics
        class sol_phys_sec:
            pass
        sol_phys_sec.a11 = sol_phys.a11[int(elem[0])]
        sol_phys_sec.a12 = sol_phys.a12[int(elem[0])]
        sol_phys_sec.a13 = sol_phys.a13[int(elem[0])]
        sol_phys_sec.a14 = sol_phys.a14[int(elem[0])]
        sol_phys_sec.a15 = sol_phys.a15[int(elem[0])]
        sol_phys_sec.a16 = sol_phys.a16[int(elem[0])]
        sol_phys_sec.a17 = sol_phys.a17[int(elem[0])]
        sol_phys_sec.a22 = sol_phys.a22[int(elem[0])]
        sol_phys_sec.a23 = sol_phys.a23[int(elem[0])]
        sol_phys_sec.a24 = sol_phys.a24[int(elem[0])]
        sol_phys_sec.a25 = sol_phys.a25[int(elem[0])]
        sol_phys_sec.a26 = sol_phys.a26[int(elem[0])]
        sol_phys_sec.a27 = sol_phys.a27[int(elem[0])]
        sol_phys_sec.a33 = sol_phys.a33[int(elem[0])]
        sol_phys_sec.a34 = sol_phys.a34[int(elem[0])]
        sol_phys_sec.a35 = sol_phys.a35[int(elem[0])]
        sol_phys_sec.a36 = sol_phys.a36[int(elem[0])]
        sol_phys_sec.a37 = sol_phys.a37[int(elem[0])]
        sol_phys_sec.a44 = sol_phys.a44[int(elem[0])]
        sol_phys_sec.a45 = sol_phys.a45[int(elem[0])]
        sol_phys_sec.a46 = sol_phys.a46[int(elem[0])]
        sol_phys_sec.a47 = sol_phys.a47[int(elem[0])]
        sol_phys_sec.a55 = sol_phys.a55[int(elem[0])]
        sol_phys_sec.a56 = sol_phys.a56[int(elem[0])]
        sol_phys_sec.a57 = sol_phys.a57[int(elem[0])]
        sol_phys_sec.a66 = sol_phys.a66[int(elem[0])]
        sol_phys_sec.a67 = sol_phys.a67[int(elem[0])]
        sol_phys_sec.a77 = sol_phys.a77[int(elem[0])]
        # ---------------------------------------------------------------------
        # Definition of geometrical fatures
        rr_vec[int(elem[0]),:]    = mesh_point[int(elem[2]),:]-mesh_point[int(elem[1]),:]
        zzloc_vec[int(elem[0]),:] = rr_vec[int(elem[0]),:]
        zzloc_vec[int(elem[0]),:] = zzloc_vec[int(elem[0]),:]/np.linalg.norm(zzloc_vec[int(elem[0]),:]) 
        if np.linalg.norm(np.cross(zzloc_vec[int(elem[0]),:],[1,0,0]))==0 and (lock[int(elem[0])] == 1 or lock[int(elem[0])] == -1):
            lock[int(elem[0])] = 0
        elif np.linalg.norm(np.cross(zzloc_vec[int(elem[0]),:],[0,1,0]))==0 and (lock[int(elem[0])] == 2 or lock[int(elem[0])] == -2):
            lock[int(elem[0])] = 0
        if lock[int(elem[0])] == 0:
            xxloc_vec[int(elem[0]),:] = [rr_vec[int(elem[0]),2],rr_vec[int(elem[0]),1],-rr_vec[int(elem[0]),0]]
            xxloc_vec[int(elem[0]),:] = xxloc_vec[int(elem[0]),:]/np.linalg.norm(xxloc_vec[int(elem[0]),:])
            # ---------------------------------------------------------------------
            # In case longitudinal axis is coincident with y axis, x axis remains without rotation
            if abs(np.dot(zzloc_vec[int(elem[0]),:],[0,1,0])) == 1 :
                xxloc_vec[int(elem[0]),:] = [1,0,0] 
            yyloc_vec[int(elem[0]),:] = np.cross(zzloc_vec[int(elem[0]),:],xxloc_vec[int(elem[0]),:])
            yyloc_vec[int(elem[0]),:] = yyloc_vec[int(elem[0]),:]/np.linalg.norm(yyloc_vec[int(elem[0]),:]) 
        elif lock[int(elem[0])] == 1:
            xxloc_vec[int(elem[0]),:] = [rr_vec[int(elem[0]),2],0,-rr_vec[int(elem[0]),0]]
            xxloc_vec[int(elem[0]),:] *= np.sign(np.dot(xxloc_vec[int(elem[0]),:],np.array([1,0,0])))
            if np.linalg.norm(xxloc_vec[int(elem[0]),:]) == 0:
                if zzloc_vec[int(elem[0]),1] > 0:
                    xxloc_vec[int(elem[0]),:] = [1,0,0]
                elif zzloc_vec[int(elem[0]),1] < 0: 
                    xxloc_vec[int(elem[0]),:] = [-1,0,0]
            xxloc_vec[int(elem[0]),:] = xxloc_vec[int(elem[0]),:]/np.linalg.norm(xxloc_vec[int(elem[0]),:])
            yyloc_vec[int(elem[0]),:] = np.cross(zzloc_vec[int(elem[0]),:],xxloc_vec[int(elem[0]),:])
            yyloc_vec[int(elem[0]),:] = yyloc_vec[int(elem[0]),:]/np.linalg.norm(yyloc_vec[int(elem[0]),:]) 
        elif lock[int(elem[0])] == 2:
            yyloc_vec[int(elem[0]),:] = [0,rr_vec[int(elem[0]),2],-rr_vec[int(elem[0]),1]]
            yyloc_vec[int(elem[0]),:] *= np.sign(np.dot(yyloc_vec[int(elem[0]),:],np.array([0,1,0])))
            if np.linalg.norm(yyloc_vec[int(elem[0]),:]) == 0:
                if zzloc_vec[int(elem[0]),0] > 0:
                    yyloc_vec[int(elem[0]),:] = [0,1,0]
                elif zzloc_vec[int(elem[0]),0] < 0:
                    yyloc_vec[int(elem[0]),:] = [0,-1,0]
            yyloc_vec[int(elem[0]),:] = yyloc_vec[int(elem[0]),:]/np.linalg.norm(yyloc_vec[int(elem[0]),:])
            xxloc_vec[int(elem[0]),:] = np.cross(yyloc_vec[int(elem[0]),:],zzloc_vec[int(elem[0]),:])
            xxloc_vec[int(elem[0]),:] = xxloc_vec[int(elem[0]),:]/np.linalg.norm(xxloc_vec[int(elem[0]),:]) 
        elif lock[int(elem[0])] == -1:
            xxloc_vec[int(elem[0]),:] = [rr_vec[int(elem[0]),2],0,-rr_vec[int(elem[0]),0]]
            xxloc_vec[int(elem[0]),:] *= np.sign(np.dot(xxloc_vec[int(elem[0]),:],np.array([-1,0,0])))
            if np.linalg.norm(xxloc_vec[int(elem[0]),:]) == 0:
                if zzloc_vec[int(elem[0]),1] > 0:
                    xxloc_vec[int(elem[0]),:] = [-1,0,0]
                elif zzloc_vec[int(elem[0]),1] < 0: 
                    xxloc_vec[int(elem[0]),:] = [1,0,0]
            xxloc_vec[int(elem[0]),:] = xxloc_vec[int(elem[0]),:]/np.linalg.norm(xxloc_vec[int(elem[0]),:])
            yyloc_vec[int(elem[0]),:] = np.cross(zzloc_vec[int(elem[0]),:],xxloc_vec[int(elem[0]),:])
            yyloc_vec[int(elem[0]),:] = yyloc_vec[int(elem[0]),:]/np.linalg.norm(yyloc_vec[int(elem[0]),:]) 
        elif lock[int(elem[0])] == -2:
            yyloc_vec[int(elem[0]),:] = [0,rr_vec[int(elem[0]),2],-rr_vec[int(elem[0]),1]]
            yyloc_vec[int(elem[0]),:] *= np.sign(np.dot(yyloc_vec[int(elem[0]),:],np.array([0,-1,0])))
            if np.linalg.norm(yyloc_vec[int(elem[0]),:]) == 0:
                if zzloc_vec[int(elem[0]),0] > 0:
                    yyloc_vec[int(elem[0]),:] = [0,-1,0]
                elif zzloc_vec[int(elem[0]),0] < 0:
                    yyloc_vec[int(elem[0]),:] = [0,1,0]
            yyloc_vec[int(elem[0]),:] = yyloc_vec[int(elem[0]),:]/np.linalg.norm(yyloc_vec[int(elem[0]),:])
            xxloc_vec[int(elem[0]),:] = np.cross(yyloc_vec[int(elem[0]),:],zzloc_vec[int(elem[0]),:])
            xxloc_vec[int(elem[0]),:] = xxloc_vec[int(elem[0]),:]/np.linalg.norm(xxloc_vec[int(elem[0]),:]) 
        # ---------------------------------------------------------------------
        # Define rotation matrices
        rot_mat[int(elem[0]),:,:] = np.matmul(RR_ctrl[int(elem[0])],[[np.dot( xxloc_vec[int(elem[0]),:],[1,0,0]),np.dot( yyloc_vec[int(elem[0]),:],[1,0,0]),np.dot( zzloc_vec[int(elem[0]),:],[1,0,0])],
                [np.dot( xxloc_vec[int(elem[0]),:],[0,1,0]),np.dot( yyloc_vec[int(elem[0]),:],[0,1,0]),np.dot( zzloc_vec[int(elem[0]),:],[0,1,0])],
                [np.dot( xxloc_vec[int(elem[0]),:],[0,0,1]),np.dot( yyloc_vec[int(elem[0]),:],[0,0,1]),np.dot( zzloc_vec[int(elem[0]),:],[0,0,1])]])
        rotT_mat[int(elem[0]),:,:]       = np.transpose(rot_mat[int(elem[0]),:,:])
        Lrot_mat[int(elem[0]),:3,:3]     = rot_mat[int(elem[0]),:,:] 
        Lrot_mat[int(elem[0]),3:6,3:6]   = rot_mat[int(elem[0]),:,:]  
        Lrot_mat[int(elem[0]),6:,6:]     = rot_mat[int(elem[0]),:,:] 
        LrotT_mat[int(elem[0]),:3,:3]    = rotT_mat[int(elem[0]),:,:] 
        LrotT_mat[int(elem[0]),3:6,3:6]  = rotT_mat[int(elem[0]),:,:]
        LrotT_mat[int(elem[0]),6:,6:]    = rotT_mat[int(elem[0]),:,:]
        LL_vec[int(elem[0])]             = np.linalg.norm(rr_vec[int(elem[0]),:])
        # Define the distances maintaining the information of the nodes and elements
        # If the rotation matrix still null, add the values of the element rotation matrix and transposed matrix for the first value
        # if not for the second value
        # Do it for both nodes of the element
        # It is not contemplated that a node is conected with more than two nodes. In this case is recomended to create a fixed joint between nodes
        if sum(sum(rot_mat_p[int(elem[1]),0,:,:])) == 0:
            rot_mat_p[int(elem[1]),0,:,:]  = rot_mat[int(elem[0]),:,:]
            rot_matT_p[int(elem[1]),0,:,:] = np.transpose(rot_mat[int(elem[0]),:,:])
            LL_vec_p[0,int(elem[1])]       = LL_vec[int(elem[0])]/2
            rr_vec_p[0,int(elem[1]),:]     = rr_vec[int(elem[0]),:]
        else:
            rot_mat_p[int(elem[1]),1,:,:]  = rot_mat[int(elem[0]),:,:]
            rot_matT_p[int(elem[1]),1,:,:] = np.transpose(rot_mat[int(elem[0]),:,:])
            LL_vec_p[1,int(elem[1])]       = LL_vec[int(elem[0])]/2
            rr_vec_p[1,int(elem[1]),:]     = rr_vec[int(elem[0]),:]
        if sum(sum(rot_mat_p[int(elem[2]),0,:,:])) == 0:
            rot_mat_p[int(elem[2]),0,:,:]  = rot_mat[int(elem[0]),:,:]
            rot_matT_p[int(elem[2]),0,:,:] = np.transpose(rot_mat[int(elem[0]),:,:])
            LL_vec_p[0,int(elem[2])]       = LL_vec[int(elem[0])]/2
            rr_vec_p[0,int(elem[2]),:]     = rr_vec[int(elem[0]),:]
        else:
            rot_mat_p[int(elem[2]),1,:,:]  = rot_mat[int(elem[0]),:,:]
            rot_matT_p[int(elem[2]),1,:,:] = np.transpose(rot_mat[int(elem[0]),:,:])
            LL_vec_p[1,int(elem[2])]       = LL_vec[int(elem[0])]/2
            rr_vec_p[1,int(elem[2]),:]     = rr_vec[int(elem[0]),:]
        # ---------------------------------------------------------------------
        # Calculate stiffness matrix
        # kk_mat : stiffness matrix in element coordinates
        # KK_aux : inclussion of the kk_mat in the global stiffness matrix
        chord_vec0   = section.points[int(elem[1])][0][section.te[int(elem[1])],:]-section.points[int(elem[1])][0][section.le[int(elem[1])],:]
        chord_vec1   = section.points[int(elem[2])][0][section.te[int(elem[2])],:]-section.points[int(elem[2])][0][section.le[int(elem[2])],:]
        chord_vec   = (chord_vec0+chord_vec1)/2
        chord       = np.linalg.norm(chord_vec)
        if LL_vec[int(elem[0])] < chord:
            deltash = 1
            sol_phys.xsc = sol_phys.xsc_sh
            sol_phys.ysc = sol_phys.ysc_sh
        else:
            deltash = 0
            sol_phys.xsc = sol_phys.xsc_nosh
            sol_phys.ysc = sol_phys.ysc_nosh
        deltash = 0
        kk_mat = stiffness_matrix(LL_vec[int(elem[0])], sol_phys_sec,deltash)
        KK_aux = np.zeros((9*num_point,9*num_point))
        # Stiffness matrix in global coordinates, connectivity
        KK_aux[int(9*elem[1]):int(9*elem[1]+9),int(9*elem[1]):int(9*elem[1]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat.K00,LrotT_mat[int(elem[0]),:,:]))
        KK_aux[int(9*elem[1]):int(9*elem[1]+9),int(9*elem[2]):int(9*elem[2]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat.K01,LrotT_mat[int(elem[0]),:,:]))
        KK_aux[int(9*elem[2]):int(9*elem[2]+9),int(9*elem[1]):int(9*elem[1]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat.K10,LrotT_mat[int(elem[0]),:,:]))
        KK_aux[int(9*elem[2]):int(9*elem[2]+9),int(9*elem[2]):int(9*elem[2]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat.K11,LrotT_mat[int(elem[0]),:,:]))
        # ---------------------------------------------------------------------
        # Stiffness matrix in global coordinates before join elements
        KK_global_prima = KK_global_prima+KK_aux 
    return KK_global_prima,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,LL_vec_p,rr_vec_p,rot_mat_p,rot_matT_p,sol_phys

#%%
def define_mass(mesh_elem,sol_phys,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat,section):
    # Definition of the stiffness matrix
    # ------------------------------------------------------------------------
    # initialization of variables
    # MM_global_prima : mass matrix before joints
    MM_global_prima = np.zeros((9*num_point,9*num_point))
    massdata = 0
    # for each beam element
    for elem in mesh_elem:
        # define a class to import the section characteristics in the beam element
        # definition of the mass matrix
        # sol_phys_sec  : class to import mass characteristics
        class sol_phys_sec:
            pass
        sol_phys_sec.m11 = sol_phys.m11[int(elem[0])]
        sol_phys_sec.m22 = sol_phys.m22[int(elem[0])]
        sol_phys_sec.m33 = sol_phys.m33[int(elem[0])]
        sol_phys_sec.m44 = sol_phys.m44[int(elem[0])]
        sol_phys_sec.m55 = sol_phys.m55[int(elem[0])]
        sol_phys_sec.m66 = sol_phys.m66[int(elem[0])]
        sol_phys_sec.m77 = sol_phys.m77[int(elem[0])]
        sol_phys_sec.m16 = sol_phys.m16[int(elem[0])]
        sol_phys_sec.m26 = sol_phys.m26[int(elem[0])]
        sol_phys_sec.m34 = sol_phys.m34[int(elem[0])]
        sol_phys_sec.m35 = sol_phys.m35[int(elem[0])]
        sol_phys_sec.m37 = sol_phys.m37[int(elem[0])]
        sol_phys_sec.m45 = sol_phys.m45[int(elem[0])]
        sol_phys_sec.m47 = sol_phys.m47[int(elem[0])]
        sol_phys_sec.m57 = sol_phys.m57[int(elem[0])]
        # ---------------------------------------------------------------------
        # Calculate mass matrix
        # mm_mat : mass matrix in element coordinates
        # MM_aux : inclussion of the mm_mat in the global stiffness matrix
        massdata += sol_phys_sec.m11*LL_vec[int(elem[0])]
        chord_vec0   = section.points[int(elem[1])][0][section.te[int(elem[1])],:]-section.points[int(elem[1])][0][section.le[int(elem[1])],:]
        chord_vec1   = section.points[int(elem[2])][0][section.te[int(elem[2])],:]-section.points[int(elem[2])][0][section.le[int(elem[2])],:]
        chord_vec   = (chord_vec0+chord_vec1)/2
        chord       = np.linalg.norm(chord_vec)
        if LL_vec[int(elem[0])] < chord:
            deltash = 1
        else:
            deltash = 0
        deltash = 0
        mm_mat = mass_matrix(LL_vec[int(elem[0])], sol_phys_sec,deltash)
        MM_aux = np.zeros((9*num_point,9*num_point))
        # Mass matrix in global coordinates, connectivity
        MM_aux[int(9*elem[1]):int(9*elem[1]+9),int(9*elem[1]):int(9*elem[1]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(mm_mat.M00,LrotT_mat[int(elem[0]),:,:]))
        MM_aux[int(9*elem[1]):int(9*elem[1]+9),int(9*elem[2]):int(9*elem[2]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(mm_mat.M01,LrotT_mat[int(elem[0]),:,:]))
        MM_aux[int(9*elem[2]):int(9*elem[2]+9),int(9*elem[1]):int(9*elem[1]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(mm_mat.M10,LrotT_mat[int(elem[0]),:,:]))
        MM_aux[int(9*elem[2]):int(9*elem[2]+9),int(9*elem[2]):int(9*elem[2]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(mm_mat.M11,LrotT_mat[int(elem[0]),:,:]))
        # ---------------------------------------------------------------------
        # Mass matrix in global coordinates before join elements
        MM_global_prima = MM_global_prima+MM_aux 
    return MM_global_prima,massdata
#%%
def define_exmassdist(mesh_elem,exmass,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat):
    # Definition of the mass matrix of extra distributed mass
    # ------------------------------------------------------------------------
    # initialization of variables
    # MM_global_prima : mass matrix before joints
    MM_global_prima = np.zeros((9*num_point,9*num_point))
    # for each beam element
    massdata = 0
    for elem in mesh_elem:
        # define a class to import the section characteristics in the beam element
        # definition of the mass matrix
        # sol_phys_sec  : class to import mass characteristics
        class sol_phys_sec:
            pass
        sol_phys_sec.m11 = exmass.elem.rho[int(elem[0])]*exmass.elem.area[int(elem[0])]
        sol_phys_sec.m22 = exmass.elem.rho[int(elem[0])]*exmass.elem.area[int(elem[0])]
        sol_phys_sec.m33 = exmass.elem.rho[int(elem[0])]*exmass.elem.area[int(elem[0])]
        sol_phys_sec.m44 = exmass.elem.rho[int(elem[0])]*exmass.elem.Ixx[int(elem[0])]
        sol_phys_sec.m55 = exmass.elem.rho[int(elem[0])]*exmass.elem.Iyy[int(elem[0])]
        sol_phys_sec.m66 = exmass.elem.rho[int(elem[0])]*(exmass.elem.Ixx[int(elem[0])]+exmass.elem.Iyy[int(elem[0])])
        sol_phys_sec.m77 = 0
        sol_phys_sec.m16 = -exmass.elem.rho[int(elem[0])]*exmass.elem.area[int(elem[0])]*exmass.elem.yCG[int(elem[0])]
        sol_phys_sec.m26 = exmass.elem.rho[int(elem[0])]*exmass.elem.area[int(elem[0])]*exmass.elem.xCG[int(elem[0])]
        sol_phys_sec.m34 = exmass.elem.rho[int(elem[0])]*exmass.elem.area[int(elem[0])]*exmass.elem.yCG[int(elem[0])]
        sol_phys_sec.m35 = exmass.elem.rho[int(elem[0])]*exmass.elem.area[int(elem[0])]*exmass.elem.xCG[int(elem[0])]
        sol_phys_sec.m37 = 0
        sol_phys_sec.m45 = exmass.elem.rho[int(elem[0])]*exmass.elem.Ixy[int(elem[0])]
        sol_phys_sec.m47 = 0
        sol_phys_sec.m57 = 0
        # ---------------------------------------------------------------------
        # Calculate mass matrix
        # mm_mat : mass matrix in element coordinates
        # MM_aux : inclussion of the mm_mat in the global stiffness matrix
        massdata += sol_phys_sec.m11*LL_vec[int(elem[0])]
        mm_mat = mass_matrix(LL_vec[int(elem[0])], sol_phys_sec,0)
        MM_aux = np.zeros((9*num_point,9*num_point))
        # Mass matrix in global coordinates, connectivity
        MM_aux[int(9*elem[1]):int(9*elem[1]+9),int(9*elem[1]):int(9*elem[1]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(mm_mat.M00,LrotT_mat[int(elem[0]),:,:]))
        MM_aux[int(9*elem[1]):int(9*elem[1]+9),int(9*elem[2]):int(9*elem[2]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(mm_mat.M01,LrotT_mat[int(elem[0]),:,:]))
        MM_aux[int(9*elem[2]):int(9*elem[2]+9),int(9*elem[1]):int(9*elem[1]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(mm_mat.M10,LrotT_mat[int(elem[0]),:,:]))
        MM_aux[int(9*elem[2]):int(9*elem[2]+9),int(9*elem[2]):int(9*elem[2]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(mm_mat.M11,LrotT_mat[int(elem[0]),:,:]))
        # ---------------------------------------------------------------------
        # Mass matrix in global coordinates before join elements
        MM_global_prima = MM_global_prima+MM_aux 
    return MM_global_prima,massdata
#%%
def define_exstiffdist(mesh_elem,exstiff,mesh_point,num_point,num_elem,LL_vec,rr_vec,rot_mat,rotT_mat,Lrot_mat,LrotT_mat):
    # Definition of the mass matrix of extra distributed mass
    # ------------------------------------------------------------------------
    # initialization of variables
    # MM_global_prima : mass matrix before joints
    KK_global_prima = np.zeros((9*num_point,9*num_point))
    # for each beam element
    for elem in mesh_elem:
        # define a class to import the section characteristics in the beam element
        # definition of the mass matrix
        # sol_phys_sec  : class to import mass characteristics
        class sol_phys_sec:
            pass
        sol_phys_sec.a11 = exstiff.elem.EE[int(elem[0])]*exstiff.elem.area[int(elem[0])]*0
        sol_phys_sec.a12 = 0
        sol_phys_sec.a13 = 0
        sol_phys_sec.a14 = 0
        sol_phys_sec.a15 = 0
        sol_phys_sec.a16 = 0
        sol_phys_sec.a17 = 0
        sol_phys_sec.a22 = exstiff.elem.EE[int(elem[0])]*exstiff.elem.Ixx[int(elem[0])]*0
        sol_phys_sec.a23 = exstiff.elem.EE[int(elem[0])]*exstiff.elem.Ixy[int(elem[0])]*0
        sol_phys_sec.a24 = 0
        sol_phys_sec.a25 = 0
        sol_phys_sec.a26 = 0 
        sol_phys_sec.a27 = 0 #exstiff.elem.EE[int(elem[0])]*exstiff.elem.area[int(elem[0])]*exstiff.elem.xCG[int(elem[0])]
        sol_phys_sec.a33 = exstiff.elem.EE[int(elem[0])]*exstiff.elem.Iyy[int(elem[0])]*0
        sol_phys_sec.a34 = 0
        sol_phys_sec.a35 = 0
        sol_phys_sec.a36 = 0
        sol_phys_sec.a37 = 0 #exstiff.elem.EE[int(elem[0])]*exstiff.elem.area[int(elem[0])]*exstiff.elem.yCG[int(elem[0])]
        sol_phys_sec.a44 = 0
        sol_phys_sec.a45 = 0
        sol_phys_sec.a46 = 0
        sol_phys_sec.a47 = 0
        sol_phys_sec.a55 = 0
        sol_phys_sec.a56 = 0
        sol_phys_sec.a57 = 0
        sol_phys_sec.a66 = 0 #exstiff.elem.EE[int(elem[0])]*exstiff.elem.FW[int(elem[0])]
        sol_phys_sec.a67 = 0
        sol_phys_sec.a77 = exstiff.elem.GG[int(elem[0])]*exstiff.elem.JJ[int(elem[0])]*0
        # ---------------------------------------------------------------------
        # Calculate mass matrix
        # mm_mat : mass matrix in element coordinates
        # MM_aux : inclussion of the mm_mat in the global stiffness matrix
        kk_mat = stiffness_matrix(LL_vec[int(elem[0])], sol_phys_sec,0)
        KK_aux = np.zeros((9*num_point,9*num_point))
        # Mass matrix in global coordinates, connectivity
        KK_aux[int(9*elem[1]):int(9*elem[1]+9),int(9*elem[1]):int(9*elem[1]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat.K00,LrotT_mat[int(elem[0]),:,:]))
        KK_aux[int(9*elem[1]):int(9*elem[1]+9),int(9*elem[2]):int(9*elem[2]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat.K01,LrotT_mat[int(elem[0]),:,:]))
        KK_aux[int(9*elem[2]):int(9*elem[2]+9),int(9*elem[1]):int(9*elem[1]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat.K10,LrotT_mat[int(elem[0]),:,:]))
        KK_aux[int(9*elem[2]):int(9*elem[2]+9),int(9*elem[2]):int(9*elem[2]+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat.K11,LrotT_mat[int(elem[0]),:,:]))
        # ---------------------------------------------------------------------
        # Mass matrix in global coordinates before join elements
        KK_global_prima = KK_global_prima+KK_aux 
    return KK_global_prima
#%%
def define_exmasspoint(exmass,mesh_point,num_point,Lrot_mat,LrotT_mat):
    # Definition of the mass matrix of extra node mass
    # ------------------------------------------------------------------------
    # initialization of variables
    # MM_global_prima : mass matrix before joints
    MM_global_prima = np.zeros((9*num_point,9*num_point))
    # for each beam element
    massdata = 0
    for point in np.arange(num_point):
        # Calculate mass matrix
        # mm_mat : mass matrix in element coordinates
        # MM_aux : inclussion of the mm_mat in the global stiffness matrix
        mm_mat = pointmass_matrix(exmass,point)
        MM_aux = np.zeros((9*num_point,9*num_point))
        # Mass matrix in global coordinates, connectivity
        MM_aux[int(9*point):int(9*point+9),int(9*point):int(9*point+9)] = mm_mat
        # ---------------------------------------------------------------------
        # Mass matrix in global coordinates before join elements
        MM_global_prima = MM_global_prima+MM_aux 
        massdata += mm_mat[0,0]
    return MM_global_prima,massdata
#%%
def define_exstiffpoint(exstiff,mesh_point,mesh_elem,num_point,Lrot_mat,LrotT_mat):
    # Definition of the mass matrix of extra node mass
    # ------------------------------------------------------------------------
    # initialization of variables
    # MM_global_prima : mass matrix before joints
    KK_global_prima = np.zeros((9*num_point,9*num_point))
    # for each beam element
    for elem in mesh_elem:
        class sol_phys_sec:
            pass
        for iipoint in [0,1]:
            point = int(elem[iipoint+1])
            # Calculate mass matrix
            # mm_mat : mass matrix in element coordinates
            # MM_aux : inclussion of the mm_mat in the global stiffness matrix
            LL = exstiff.point.LONG[point]/2
            kk_mat = np.zeros((9,9))
            kk_mat[0,3] = -6*exstiff.point.EE[point]*exstiff.point.IIxy[point]/LL**2
            kk_mat[3,0] = kk_mat[0,3]
            kk_mat[0,4] = 6*exstiff.point.EE[point]*exstiff.point.IIy[point]/LL**2
            kk_mat[4,0] = kk_mat[0,4]
            kk_mat[1,3] = -6*exstiff.point.EE[point]*exstiff.point.IIx[point]/LL**2
            kk_mat[3,1] = kk_mat[1,3]
            kk_mat[1,4] = 6*exstiff.point.EE[point]*exstiff.point.IIxy[point]/LL**2
            kk_mat[4,1] = kk_mat[1,4]
            kk_mat[3,3] = 6*exstiff.point.EE[point]*exstiff.point.IIx[point]/LL
            kk_mat[3,4] = -6*exstiff.point.EE[point]*exstiff.point.IIxy[point]/LL
            kk_mat[4,3] = kk_mat[3,4]
            kk_mat[4,4] = 6*exstiff.point.EE[point]*exstiff.point.IIy[point]/LL
            kk_mat[5,8] = exstiff.point.GG[point]*exstiff.point.JJ[point]/10  
            kk_mat[8,5] = kk_mat[8,5]          
            kk_mat[8,8] = exstiff.point.GG[point]*exstiff.point.JJ[point]*LL/10
            KK_aux = np.zeros((9*num_point,9*num_point))
            # Stiffness matrix in global coordinates, connectivity
            KK_aux[int(9*point):int(9*point+9),int(9*point):int(9*point+9)] = np.matmul(Lrot_mat[int(elem[0]),:,:],np.matmul(kk_mat,LrotT_mat[int(elem[0]),:,:]))
            KK_global_prima = KK_global_prima+KK_aux*0.24
    return KK_global_prima
#%%
def define_damp(case_setup,sol_phys,mesh_data,section,MM_global,KK_global,solver_vib_mod,solver_struct_stat,num_point,mesh_mark): 
    # Function to define the damping matrix
    # case_setup      : information about the case configuration
    # sol_phys        : information about the solid physics
    # mesh_data       : information about the beam mesh
    # section         : information about the section
    # MM_global       : global mass matrix 
    # KK_global       : global stiffness matrix 
    # solver_vib_mod  : function to calculate vibration modes
    # num_point       : number of nodes of the beam mesh
    # -------------------------------------------------------------------------
    # CC_global       : global damping matrix
    # vibfree         : information of the free vibration results
    # sec_coor_vib    : information about the section nodes in free vibration
    MM_global = MM_global.copy()
    KK_global = KK_global.copy()
    CC_global             = np.zeros((9*num_point,9*num_point)) 
    vibfree, sec_coor_vib = solver_vib_mod(case_setup,sol_phys,mesh_data,section,[],[])
    # Selection of the damping type.
    # If Rayleigh damping is chosen:
    if case_setup.damping_type == "RAYLEIGH":
        # If the movement is consequence of an acceleration choose ACCEL if it is aerodynamic AERO
        # modal_mass   : mass of each mode of vibration
        # effmass      : effective mass of each mode of vibration
        # effmass_cum  : cumulative effective mass of the vibration modes
        # iill_vec  : ones vector to multiply the global mass matrix
        if case_setup.damping_factor == "ACCEL":
            modal_mass  = np.matmul(np.transpose(vibfree.mode),np.matmul(np.identity(len(vibfree.mode)),vibfree.mode))
            effmass     = np.zeros((len(modal_mass)+1,))
            effmass_cum = np.zeros((len(modal_mass)+1,))
            iill_vec = np.ones((len(modal_mass),))
            for ii_effmass in np.arange(len(modal_mass)):
                effmass[ii_effmass+1] = (np.matmul(np.transpose(vibfree.mode[:,ii_effmass]),np.matmul(np.identity(len(vibfree.mode)),iill_vec)))**2/modal_mass[ii_effmass,ii_effmass]
            # calculate the cumulative effect of the mass
            for ii_effmass in np.arange(len(modal_mass)):
                if ii_effmass == len(modal_mass):
                    effmass_cum[ii_effmass+1] = 1
                else:
                    effmass_cum[ii_effmass+1] = np.sum(effmass[:ii_effmass+2])/np.sum(effmass)
            # w_func  : vibration modes as a function of the effective cummulative mass
            # w0      : vibration mode for a 5% of the effective mass
            # w1      : vibration mode for a 50% of the effective mass
            w_interp = np.zeros((len(vibfree.w_mode)+1,))
            w_interp[1:] = vibfree.w_mode
            w_func = interpolate.interp1d(effmass_cum,w_interp)
            w0     = w_func(0.05)  
            w1     = w_func(case_setup.damping_mass)
            # mat_alphabeta  : matrix to calculate Rayleigh parameters
            # cteray         : vector of the Rayleigh parameters
            # alpha_rayleigh : alpha parameter
            # beta_rayleigh  : beta parameter
            mat_alphabeta   = [[1/(2*w0), w0/2],[1/(2*w1), w1/2]]
            cteray          = np.matmul(np.linalg.inv(mat_alphabeta),case_setup.damping_constant*np.ones((2,1)))
            alpha_rayleigh  = cteray[0]
            beta_rayleigh   = cteray[1]
        elif case_setup.damping_factor == "AERO":
            modal_mass  = np.matmul(np.transpose(vibfree.mode),np.matmul(np.identity(len(vibfree.mode)),vibfree.mode))
            effmass     = np.zeros((len(modal_mass)+1,))
            effmass_cum = np.zeros((len(modal_mass)+1,))
            statsol, section_globalCS = solver_struct_stat(case_setup,sol_phys,mesh_data,section)
            iill_vec = np.matmul(vibfree.MM_pow2,statsol.q_vec)
            for ii_effmass in np.arange(len(modal_mass)):
                effmass[ii_effmass+1] = (np.matmul(np.transpose(vibfree.mode[:,ii_effmass]),np.matmul(np.identity(len(vibfree.mode)),iill_vec)))**2/modal_mass[ii_effmass,ii_effmass]
            # calculate the cumulative effect of the mass
            for ii_effmass in np.arange(len(modal_mass)):
                if ii_effmass == len(modal_mass):
                    effmass_cum[ii_effmass+1] = 1
                else:
                    effmass_cum[ii_effmass+1] = np.sum(effmass[:ii_effmass+2])/np.sum(effmass)
            # w_func  : vibration modes as a function of the effective cummulative mass
            # w0      : vibration mode for a 5% of the effective mass
            # w1      : vibration mode for a 50% of the effective mass
            w_interp = np.zeros((len(vibfree.w_mode)+1,))
            w_interp[1:] = vibfree.w_mode
            w_func = interpolate.interp1d(effmass_cum,w_interp)
            w0     = w_func(0.05)  
            w1     = w_func(case_setup.damping_mass)
            # mat_alphabeta  : matrix to calculate Rayleigh parameters
            # cteray         : vector of the Rayleigh parameters
            # alpha_rayleigh : alpha parameter
            # beta_rayleigh  : beta parameter
            mat_alphabeta   = [[1/(2*w0), w0/2],[1/(2*w1), w1/2]]
            cteray          = np.matmul(np.linalg.inv(mat_alphabeta),case_setup.damping_constant*np.ones((2,1)))
            alpha_rayleigh  = cteray[0]
            beta_rayleigh   = cteray[1]
        elif case_setup.damping_factor == "DIREC":
            f1  = case_setup.Ray_f1
            f2  = case_setup.Ray_f2
            xi1 = case_setup.damping_constant1
            xi2 = case_setup.damping_constant2
#            w1  = 2*np.pi*f1
#            w2  = 2*np.pi*f2
#            R_cte = w2/w1
            alpha_rayleigh = 4*np.pi*f1*f2*(xi1*f2-xi2*f1)/(f2**2-f1**1) #2*case_setup.damping_constant*w1*(2*R_cte)/(1+R_cte+2*np.sqrt(R_cte))
            beta_rayleigh  = (xi2*f2-xi1*f1)/(np.pi*(f2**2-f1**2)) #2*case_setup.damping_constant/w1*(2)/(1+R_cte+2*np.sqrt(R_cte))
        kkmat2 = 0*KK_global
        for ii_kk in np.arange(len(KK_global)):
            for jj_kk in np.arange(len(KK_global)):
                if MM_global[ii_kk,jj_kk] != 0:
                    kkmat2[ii_kk,jj_kk] = KK_global[ii_kk,jj_kk]
        CC_global = alpha_rayleigh*MM_global + beta_rayleigh*KK_global #4*np.pi*f1*xi1*MM_global
        # Delete the influence in the kinematic restrictions       
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
                        CC_global[node_2*9:node_2*9+9,:] = np.zeros((9,len(CC_global)))
    return vibfree, CC_global

#%%
def def_secglob(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point):  
    # Function to define the internal nodes of each section in global coordinates.
    # This function is required to create a 3D visualization of the beam.
    # -------------------------------------------------------------------------
    # section   : information about every node section
    # num_elem  : number of elements of the beam
    # mesh_elem : elements of the beam mesh
    # rot_mat   : totation matrix of each element
    # sol_phys  : information of the solid physics
    # -------------------------------------------------------------------------
    # Define a class to save the section nodes
    # section_globalCS : class to include information about section nodes
    class section_globalCS:
        pass   
    # Information about:
    # * .num       : identifier of the element
    # * .n1        : section nodes of node 1 of the beam element
    # * .n2        : section nodes of node 2 of the beam element
    # * .cdgn1     : center of gravity of node 1
    # * .cdgn2     : center of gravity of node 2
    # * .nodeelem1 : node 1 of the element
    # * .nodeelem1 : node 2 of the element
    section_globalCS.num       = np.zeros((num_elem,))
    section_globalCS.n1        = []
    section_globalCS.n2        = []
    section_globalCS.e1        = []
    section_globalCS.e2        = []
    section_globalCS.cdgn1     = []
    section_globalCS.cdgn2     = []
    section_globalCS.nodeelem1 = []
    section_globalCS.nodeelem2 = []
    section_globalCS.section1  = []
    section_globalCS.section2  = []
    section_globalCS.cdg_sub1  = []
    section_globalCS.cdg_sub2  = []
    section_globalCS.nsub      = []
    section_globalCS.nvec1     = []
    section_globalCS.nvec2     = []
    section_globalCS.dsvec1    = []
    section_globalCS.dsvec2    = []
    # -------------------------------------------------------------------------
    # For each element of the beam
    nsubsec = np.zeros((len(section.pointtot_id),))
    pointsec = def_vec_param(len(mesh_point))
    pointsecnum = np.zeros((len(mesh_point),))
    aero_center = def_vec_param(len(mesh_point))
    for elem in mesh_elem:
        # Define the element identifier
        section_globalCS.num[int(elem[0])] = elem[0]
        # Define matrices to save the positions of the section nodes in global coordinates
        pointn1_mat_GCS = np.zeros((len(section.points_tot[int(elem[1])]),3))
        pointn2_mat_GCS = np.zeros((len(section.points_tot[int(elem[2])]),3))
        elemn1_mat_GCS = np.zeros((len(section.elem_tot[int(elem[1])]),3))
        elemn2_mat_GCS = np.zeros((len(section.elem_tot[int(elem[2])]),3))
        # Save positions for node 1
        pointsec1 = np.zeros((len(section.points_tot[int(elem[1])]),3))
        for ii_sectionpoint in np.arange(len(section.points_tot[int(elem[1])])):
            pointn1_GCS = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((section.points_tot[int(elem[1])][int(ii_sectionpoint)],[0])))
            pointn1_cdg = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((sol_phys.cdg[int(elem[1])],[0])))
            pointn1_mat_GCS[ii_sectionpoint,:] = pointn1_GCS  
            pointsec1[ii_sectionpoint,:] += pointn1_GCS
        pointsecnum[int(elem[1])] += 1
        if len(pointsec[int(elem[1])]) == 0:
            pointsec[int(elem[1])] = pointsec1
            aero_center[int(elem[1])] = np.matmul(rot_mat[int(elem[0]),:,:],np.array([section.ae_orig_x[int(elem[1])],section.ae_orig_y[int(elem[1])],0]))
        else:
            pointsec[int(elem[1])] += pointsec1
        # Save positions for node 2
        pointsec2 = np.zeros((len(section.points_tot[int(elem[2])]),3))
        for ii_sectionpoint in np.arange(len(section.points_tot[int(elem[2])])):
            pointn2_GCS = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((section.points_tot[int(elem[2])][int(ii_sectionpoint)],[0])))
            pointn2_cdg = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((sol_phys.cdg[int(elem[2])],[0])))
            pointn2_mat_GCS[ii_sectionpoint,:] = pointn2_GCS
            pointsec2[ii_sectionpoint,:] += pointn2_GCS
        pointsecnum[int(elem[2])] += 1
        if len(pointsec[int(elem[2])]) == 0:
            pointsec[int(elem[2])] = pointsec2
            aero_center[int(elem[2])] = np.matmul(rot_mat[int(elem[0]),:,:],np.array([section.ae_orig_x[int(elem[2])],section.ae_orig_y[int(elem[2])],0]))
        else:
            pointsec[int(elem[2])] += pointsec2
        # Save positions for node 1
        for ii_sectionpoint in np.arange(len(section.elem_tot[int(elem[1])])):
            elemn1_mat_GCS[ii_sectionpoint,:] = section.elem_tot[int(elem[1])][int(ii_sectionpoint)]  
        # Save positions for node 2
        for ii_sectionpoint in np.arange(len(section.elem_tot[int(elem[2])])):
            elemn2_mat_GCS[ii_sectionpoint,:] = section.elem_tot[int(elem[2])][int(ii_sectionpoint)] 
        # Append the information to the nodes class
        section_globalCS.n1.append(pointn1_mat_GCS)
        section_globalCS.n2.append(pointn2_mat_GCS)
        section_globalCS.e1.append(elemn1_mat_GCS)
        section_globalCS.e2.append(elemn2_mat_GCS)
        section_globalCS.cdgn1.append(pointn1_cdg)
        section_globalCS.cdgn2.append(pointn2_cdg)
        section_globalCS.nodeelem1.append(int(elem[1]))
        section_globalCS.nodeelem2.append(int(elem[2]))
        section_globalCS.section1.append(section.pointtot_id[int(elem[1])])
        section_globalCS.section2.append(section.pointtot_id[int(elem[2])])
        pointn1_cdgsub = np.zeros((len(section.pointtot_id[int(elem[1])]),3))
        pointn2_cdgsub = np.zeros((len(section.pointtot_id[int(elem[2])]),3))
        pointn1_nvec = np.zeros((len(section.pointtot_id[int(elem[1])]),3))
        pointn2_nvec = np.zeros((len(section.pointtot_id[int(elem[2])]),3))
        pointn1_dsvec = np.zeros((len(section.pointtot_id[int(elem[1])]),3))
        pointn2_dsvec = np.zeros((len(section.pointtot_id[int(elem[2])]),3))
        for ii_point in np.arange(len(section.pointtot_id[int(elem[1])])):
            nsubsec[int(elem[1])] = len(section.cg[int(elem[1])])
            for ii_subsec in np.arange(nsubsec[int(elem[1])]):
                if section.pointtot_id[int(elem[1])][ii_point] == ii_subsec:
                    pointn1_cdgsub[ii_point,:] = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((section.cg[int(elem[1])][int(ii_subsec)],[0])))+mesh_point[int(elem[1]),:]
                    for kk_elem in np.arange(len(section.elem[int(elem[1])][int(ii_subsec)])):
                        if section.elem[int(elem[1])][int(ii_subsec)][int(kk_elem),1] == ii_point or section.elem[int(elem[1])][int(ii_subsec)][int(kk_elem),2]== ii_point:
                            pointn1_nvec[ii_point,:] += np.matmul(rot_mat[int(elem[0]),:,:],section.vec_normal[int(elem[1])][int(kk_elem),:])
                            pointn1_dsvec[ii_point,:] += np.matmul(rot_mat[int(elem[0]),:,:],section.vec_ds[int(elem[1])][int(kk_elem),:])
        for ii_point in np.arange(len(section.pointtot_id[int(elem[2])])):
            nsubsec[int(elem[2])] = len(section.cg[int(elem[2])])
            for ii_subsec in np.arange(nsubsec[int(elem[2])]):
                if section.pointtot_id[int(elem[2])][ii_point] == ii_subsec:
                    pointn2_cdgsub[ii_point,:] = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((section.cg[int(elem[2])][int(ii_subsec)],[0])))+mesh_point[int(elem[2]),:]
                    for kk_elem in np.arange(len(section.elem[int(elem[2])][int(ii_subsec)])):
                        if section.elem[int(elem[2])][int(ii_subsec)][int(kk_elem),1] == ii_point or section.elem[int(elem[2])][int(ii_subsec)][int(kk_elem),2]== ii_point:
                            pointn2_nvec[ii_point,:] += np.matmul(rot_mat[int(elem[0]),:,:],section.vec_normal[int(elem[2])][int(kk_elem),:])
                            pointn2_dsvec[ii_point,:] += np.matmul(rot_mat[int(elem[0]),:,:],section.vec_ds[int(elem[2])][int(kk_elem),:])
        section_globalCS.cdg_sub1.append(pointn1_cdgsub)
        section_globalCS.cdg_sub2.append(pointn2_cdgsub)
        section_globalCS.nvec1.append(pointn1_nvec)
        section_globalCS.nvec2.append(pointn2_nvec)
        section_globalCS.dsvec1.append(pointn1_dsvec)
        section_globalCS.dsvec2.append(pointn2_dsvec)
    point_tot = []
    for ii_pointsec_uni in np.arange(len(pointsec)):
        for jj_pointsec_uni in np.arange(len(pointsec[ii_pointsec_uni])):
            pointsec[ii_pointsec_uni][jj_pointsec_uni] /= pointsecnum[ii_pointsec_uni]
        point_tot.append(pointsec[ii_pointsec_uni])
    section_globalCS.point_3d = point_tot
    section_globalCS.nsub = nsubsec
    section_globalCS.aero_cent = aero_center
    return section_globalCS

def def_secglob_aero(section,num_elem,mesh_elem,rot_mat,sol_phys,mesh_point): 
    # Function to define the internal nodes of each section in global coordinates.
    # This function is required to create a 3D visualization of the beam.
    # -------------------------------------------------------------------------
    # section   : information about every node section
    # num_elem  : number of elements of the beam
    # mesh_elem : elements of the beam mesh
    # rot_mat   : totation matrix of each element
    # sol_phys  : information of the solid physics
    # -------------------------------------------------------------------------
    # Define a class to save the section nodes
    # section_globalCS : class to include information about section nodes
    class section_globalCS:
        pass   
    # Information about:
    # * .num       : identifier of the element
    # * .n1        : section nodes of node 1 of the beam element
    # * .n2        : section nodes of node 2 of the beam element
    # * .cdgn1     : center of gravity of node 1
    # * .cdgn2     : center of gravity of node 2
    # * .nodeelem1 : node 1 of the element
    # * .nodeelem1 : node 2 of the element
    section_globalCS.num       = np.zeros((num_elem,))
    section_globalCS.n1        = []
    section_globalCS.n2        = []
    section_globalCS.e1        = []
    section_globalCS.e2        = []
    section_globalCS.nodeelem1 = []
    section_globalCS.nodeelem2 = []
    section_globalCS.section1  = []
    section_globalCS.section2  = []
    section_globalCS.cdg_sub1  = []
    section_globalCS.cdg_sub2  = []
    section_globalCS.nsub      = []
    section_globalCS.nvec1     = []
    section_globalCS.nvec2     = []
    section_globalCS.dsvec1    = []
    section_globalCS.dsvec2    = []
    # -------------------------------------------------------------------------
    # For each element of the beam
    nsubsec = np.zeros((len(section.pointtot_id),))
    pointsec = def_vec_param(len(mesh_point))
    pointsecnum = np.zeros((len(mesh_point),))
    aero_center = def_vec_param(len(mesh_point))
    for elem in mesh_elem:
        # Define the element identifier
        section_globalCS.num[int(elem[0])] = elem[0]
        # Define matrices to save the positions of the section nodes in global coordinates
        pointn1_mat_GCS = np.zeros((len(section.points_tot[int(elem[1])]),3))
        pointn2_mat_GCS = np.zeros((len(section.points_tot[int(elem[2])]),3))
        elemn1_mat_GCS = np.zeros((len(section.elem_tot[int(elem[1])]),3))
        elemn2_mat_GCS = np.zeros((len(section.elem_tot[int(elem[2])]),3))
        # Save positions for node 1
        pointsec1 = np.zeros((len(section.points_tot[int(elem[1])]),3))
        for ii_sectionpoint in np.arange(len(section.points_tot[int(elem[1])])):
            pointn1_GCS = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((section.points_tot[int(elem[1])][int(ii_sectionpoint)],[0])))
            pointn1_mat_GCS[ii_sectionpoint,:] = pointn1_GCS  
            pointsec1[ii_sectionpoint,:] += pointn1_GCS
        pointsecnum[int(elem[1])] += 1
        if len(pointsec[int(elem[1])]) == 0:
            pointsec[int(elem[1])] = pointsec1
            aero_center[int(elem[1])] = np.matmul(rot_mat[int(elem[0]),:,:],np.array([section.ae_orig_x[int(elem[1])],section.ae_orig_y[int(elem[1])],0]))
        else:
            pointsec[int(elem[1])] += pointsec1
        # Save positions for node 2
        pointsec2 = np.zeros((len(section.points_tot[int(elem[2])]),3))
        for ii_sectionpoint in np.arange(len(section.points_tot[int(elem[2])])):
            pointn2_GCS = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((section.points_tot[int(elem[2])][int(ii_sectionpoint)],[0])))
            pointn2_mat_GCS[ii_sectionpoint,:] = pointn2_GCS
            pointsec2[ii_sectionpoint,:] += pointn2_GCS
        pointsecnum[int(elem[2])] += 1
        if len(pointsec[int(elem[2])]) == 0:
            pointsec[int(elem[2])] = pointsec2
            aero_center[int(elem[2])] = np.matmul(rot_mat[int(elem[0]),:,:],np.array([section.ae_orig_x[int(elem[2])],section.ae_orig_y[int(elem[2])],0]))
        else:
            pointsec[int(elem[2])] += pointsec2
        # Save positions for node 1
        for ii_sectionpoint in np.arange(len(section.elem_tot[int(elem[1])])):
            elemn1_mat_GCS[ii_sectionpoint,:] = section.elem_tot[int(elem[1])][int(ii_sectionpoint)]  
        # Save positions for node 2
        for ii_sectionpoint in np.arange(len(section.elem_tot[int(elem[2])])):
            elemn2_mat_GCS[ii_sectionpoint,:] = section.elem_tot[int(elem[2])][int(ii_sectionpoint)] 
        # Append the information to the nodes class
        section_globalCS.n1.append(pointn1_mat_GCS)
        section_globalCS.n2.append(pointn2_mat_GCS)
        section_globalCS.e1.append(elemn1_mat_GCS)
        section_globalCS.e2.append(elemn2_mat_GCS)
        section_globalCS.nodeelem1.append(int(elem[1]))
        section_globalCS.nodeelem2.append(int(elem[2]))
        section_globalCS.section1.append(section.pointtot_id[int(elem[1])])
        section_globalCS.section2.append(section.pointtot_id[int(elem[2])])
        pointn1_cdgsub = np.zeros((len(section.pointtot_id[int(elem[1])]),3))
        pointn2_cdgsub = np.zeros((len(section.pointtot_id[int(elem[2])]),3))
        pointn1_nvec = np.zeros((len(section.pointtot_id[int(elem[1])]),3))
        pointn2_nvec = np.zeros((len(section.pointtot_id[int(elem[2])]),3))
        pointn1_dsvec = np.zeros((len(section.pointtot_id[int(elem[1])]),3))
        pointn2_dsvec = np.zeros((len(section.pointtot_id[int(elem[2])]),3))
        for ii_point in np.arange(len(section.pointtot_id[int(elem[1])])):
            nsubsec[int(elem[1])] = len(section.cg[int(elem[1])])
            for ii_subsec in np.arange(nsubsec[int(elem[1])]):
                if section.pointtot_id[int(elem[1])][ii_point] == ii_subsec:
                    pointn1_cdgsub[ii_point,:] = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((section.cg[int(elem[1])][int(ii_subsec)],[0])))+mesh_point[int(elem[1]),:]
                    for kk_elem in np.arange(len(section.elem[int(elem[1])][int(ii_subsec)])):
                        if section.elem[int(elem[1])][int(ii_subsec)][int(kk_elem),1] == ii_point or section.elem[int(elem[1])][int(ii_subsec)][int(kk_elem),2]== ii_point:
                            pointn1_nvec[ii_point,:] += np.matmul(rot_mat[int(elem[0]),:,:],section.vec_normal[int(elem[1])][int(kk_elem),:])
                            pointn1_dsvec[ii_point,:] += np.matmul(rot_mat[int(elem[0]),:,:],section.vec_ds[int(elem[1])][int(kk_elem),:])
        for ii_point in np.arange(len(section.pointtot_id[int(elem[2])])):
            nsubsec[int(elem[2])] = len(section.cg[int(elem[2])])
            for ii_subsec in np.arange(nsubsec[int(elem[2])]):
                if section.pointtot_id[int(elem[2])][ii_point] == ii_subsec:
                    pointn2_cdgsub[ii_point,:] = np.matmul(rot_mat[int(elem[0]),:,:],np.concatenate((section.cg[int(elem[2])][int(ii_subsec)],[0])))+mesh_point[int(elem[2]),:]
                    for kk_elem in np.arange(len(section.elem[int(elem[2])][int(ii_subsec)])):
                        if section.elem[int(elem[2])][int(ii_subsec)][int(kk_elem),1] == ii_point or section.elem[int(elem[2])][int(ii_subsec)][int(kk_elem),2]== ii_point:
                            pointn2_nvec[ii_point,:] += np.matmul(rot_mat[int(elem[0]),:,:],section.vec_normal[int(elem[2])][int(kk_elem),:])
                            pointn2_dsvec[ii_point,:] += np.matmul(rot_mat[int(elem[0]),:,:],section.vec_ds[int(elem[2])][int(kk_elem),:])
        section_globalCS.cdg_sub1.append(pointn1_cdgsub)
        section_globalCS.cdg_sub2.append(pointn2_cdgsub)
        section_globalCS.nvec1.append(pointn1_nvec)
        section_globalCS.nvec2.append(pointn2_nvec)
        section_globalCS.dsvec1.append(pointn1_dsvec)
        section_globalCS.dsvec2.append(pointn2_dsvec)
    point_tot = []
    for ii_pointsec_uni in np.arange(len(pointsec)):
        for jj_pointsec_uni in np.arange(len(pointsec[ii_pointsec_uni])):
            pointsec[ii_pointsec_uni][jj_pointsec_uni] /= pointsecnum[ii_pointsec_uni]
        point_tot.append(pointsec[ii_pointsec_uni])
    section_globalCS.point_3d = point_tot
    section_globalCS.nsub = nsubsec
    section_globalCS.aero_cent = aero_center
    return section_globalCS