# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 14:17:19 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
store_data     : file containing functions to store the data
last_version   : 23-02-2021
modified_by    : Andres Cremades Botella
"""

import numpy as np
from bin.aux_functions import  def_vec_param

#%% Functions
#%% Functions
def store_solution_aero(qq_global,FF_global,mesh_point,mesh_elem,section,sol_phys,rot_mat,rotT_mat,disp_values):
    # Function to store the results of the static calculation in a single class
    # qq_global  : displacement vector
    # FF_global  : load vector
    # mesh_point : beam element nodes
    # mesh_elem  : beam elements
    # section    : information of the 2D sections
    # sol_phys   : information of the solid physics
    # rot_mat    : rotation matrix
    # rotT_mat   : transposed rotation matrix
    # -------------------------------------------------------------------------
    # solution : results of the static analysis
    class solution:
        pass
    # If possible save the solution of the aerodynamic or propulsive coefficients
    try:
        solution.CL = disp_values.CL_val
        solution.CD = disp_values.CD_val
        solution.CM = disp_values.CM_val
    except:
        try:
            solution.CT = disp_values.CT_val
            solution.CQ = disp_values.CQ_val
            solution.CP = disp_values.CP_val
            solution.PE = disp_values.PE_val
        except:
            pass 
    solution.aoa = disp_values.aoa_val
    fx_vec = []
    fy_vec = []
    fz_vec = []
    Mx_vec = []
    My_vec = []
    Mz_vec = []
    Bx_vec = []
    By_vec = []
    Bz_vec = []
    solution.cl_vec           = disp_values.clsec_val
    solution.cd_vec            = disp_values.cdsec_val
    solution.cm_vec            = disp_values.cmsec_val
    # From the load vector the different displacements are taken
    for ii_q in np.arange(len(FF_global)):
        if ii_q % 9 ==0:
            fx_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 1:
            fy_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 2:
            fz_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 3:
            Mx_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 4:
            My_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 5:
            Mz_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 6:
            Bx_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 7:
            By_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 8:
            Bz_vec.append(FF_global[ii_q])
    solution.ext_fx = fx_vec
    solution.ext_fy = fy_vec
    solution.ext_fz = fz_vec
    solution.ext_mx = Mx_vec
    solution.ext_my = My_vec
    solution.ext_mz = Mz_vec
    solution.ext_Bx = Bx_vec
    solution.ext_By = By_vec
    solution.ext_Bz = Bz_vec
    pos     = mesh_point.copy()
    ii_elem = 0
    # From the displacements vector, the displacements are added to the nodes position
    for ii_q in np.arange(len(qq_global)):
        if ii_q % 9==0:
            pos[ii_elem,0] = pos[ii_elem,0]
        elif ii_q % 9 == 1:
            pos[ii_elem,1] = pos[ii_elem,1]
        elif ii_q % 9 == 2:
            pos[ii_elem,2] = pos[ii_elem,2]
        elif ii_q % 9 == 3:
            ii_elem=ii_elem+1
    # The position of the nodes is loaded in the solution class
    # solution.xdef  : position of the node in x axis        
    # solution.ydef  : position of the node in y axis       
    # solution.zdef  : position of the node in z axis           
    solution.xdef = pos[:,0]
    solution.ydef = pos[:,1]
    solution.zdef = pos[:,2]    
    return solution

def store_solution_stat(qq_global,FF_global,mesh_point,mesh_elem,section,sol_phys,rot_mat,rotT_mat,q_vec,disp_values):
    # Function to store the results of the static calculation in a single class
    # qq_global  : displacement vector
    # FF_global  : load vector
    # mesh_point : beam element nodes
    # mesh_elem  : beam elements
    # section    : information of the 2D sections
    # sol_phys   : information of the solid physics
    # rot_mat    : rotation matrix
    # rotT_mat   : transposed rotation matrix
    # -------------------------------------------------------------------------
    # solution : results of the static analysis
    class solution:
        pass
    # If possible save the solution of the aerodynamic or propulsive coefficients
    try:
        solution.CL = disp_values.CL_val
        solution.CD = disp_values.CD_val
        solution.CM = disp_values.CM_val
    except:
        try:
            solution.CT = disp_values.CT_val
            solution.CQ = disp_values.CQ_val
            solution.CP = disp_values.CP_val
            solution.EP = disp_values.PE_val
        except:
            pass
    try:
        solution.aoa = disp_values.aoa_val
    except:
        pass
    # u_vec       : vector of the linear displacement on x axis
    # v_vec       : vector of the linear displacement on y axis
    # w_vec       : vector of the linear displacement on z axis
    # phi_vec     : vector of the angular displacement on x axis
    # psi_vec     : vector of the angular displacement on y axis
    # theta_vec   : vector of the angular displacement on z axis
    # phi_d_vec   : vector of the angular displacement derivative on x axis
    # psi_d_vec   : vector of the angular displacement derivative on y axis
    # theta_d_vec : vector of the angular displacement derivative on z axis
    try:
        solution.cl_vec           = disp_values.clsec_val
        solution.cd_vec            = disp_values.cdsec_val
        solution.cm_vec            = disp_values.cmsec_val
    except:
        pass
    u_vec       = []
    v_vec       = []
    w_vec       = []
    phi_vec     = []
    psi_vec     = []
    theta_vec   = []
    phi_d_vec   = []
    psi_d_vec   = []
    theta_d_vec = []
    # From the displacement vector the different displacements are taken
    for ii_q in np.arange(len(qq_global)):
        if ii_q % 9 == 0:
            u_vec.append(qq_global[ii_q])
        elif ii_q % 9 == 1:
            v_vec.append(qq_global[ii_q])
        elif ii_q % 9 == 2:
            w_vec.append(qq_global[ii_q])
        elif ii_q % 9 == 3:
            phi_vec.append(qq_global[ii_q])
        elif ii_q % 9 == 4:
            psi_vec.append(qq_global[ii_q])
        elif ii_q % 9 == 5:
            theta_vec.append(qq_global[ii_q])
        elif ii_q % 9 == 6:
            phi_d_vec.append(qq_global[ii_q])
        elif ii_q % 9 == 7:
            psi_d_vec.append(qq_global[ii_q])
        elif ii_q % 9 == 8:
            theta_d_vec.append(qq_global[ii_q])
    # All the displacements are stored in the solution class
    # solution.u       : vector of the linear displacement on x axis
    # solution.v       : vector of the linear displacement on y axis
    # solution.w       : vector of the linear displacement on z axis
    # solution.phi     : vector of the angular displacement on x axis
    # solution.psi     : vector of the angular displacement on y axis
    # solution.theta   : vector of the angular displacement on z axis
    # solution.phi     : vector of the angular displacement derivative on x axis
    # solution.psi     : vector of the angular displacement derivative on y axis
    # solution.theta   : vector of the angular displacement derivative on z axis
    solution.u       = u_vec
    solution.v       = v_vec
    solution.w       = w_vec
    solution.phi     = phi_vec
    solution.psi     = psi_vec
    solution.theta   = theta_vec
    solution.phi_d   = phi_d_vec
    solution.psi_d   = psi_d_vec
    solution.theta_d = theta_d_vec
    # fx_vec   : force on x axis
    # fy_vec   : force on y axis
    # fz_vec   : force on z axis
    # Mx_vec   : moment on x axis
    # My_vec   : moment on y axis
    # Mz_vec   : moment on z axis
    # Bx_vec   : bimoment on x axis
    # By_vec   : bimoment on y axis
    # Bz_vec   : bimoment on z axis    
    fx_vec = []
    fy_vec = []
    fz_vec = []
    Mx_vec = []
    My_vec = []
    Mz_vec = []
    Bx_vec = []
    By_vec = []
    Bz_vec = []
    # From the load vector the different displacements are taken
    for ii_q in np.arange(len(FF_global)):
        if ii_q % 9 ==0:
            fx_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 1:
            fy_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 2:
            fz_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 3:
            Mx_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 4:
            My_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 5:
            Mz_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 6:
            Bx_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 7:
            By_vec.append(FF_global[ii_q])
        elif ii_q % 9 == 8:
            Bz_vec.append(FF_global[ii_q])
    # All the loads are stored in the solution class
    # solution.ext_fx   : force on x axis
    # solution.ext_fy   : force on y axis
    # solution.ext_fz   : force on z axis
    # solution.ext_mx   : moment on x axis
    # solution.ext_my   : moment on y axis
    # solution.ext_mz   : moment on z axis
    # solution.ext_Bx   : bimoment on x axis
    # solution.ext_By   : bimoment on y axis
    # solution.ext_Bz   : bimoment on z axis 
    solution.ext_fx = fx_vec
    solution.ext_fy = fy_vec
    solution.ext_fz = fz_vec
    solution.ext_mx = Mx_vec
    solution.ext_my = My_vec
    solution.ext_mz = Mz_vec
    solution.ext_Bx = Bx_vec
    solution.ext_By = By_vec
    solution.ext_Bz = Bz_vec
    # The position of the nodes is updated
    # pos     : position of the nodes
    # ii_elem : counter of the nodes
    pos     = mesh_point.copy()
    ii_elem = 0
    # From the displacements vector, the displacements are added to the nodes position
    for ii_q in np.arange(len(qq_global)):
        if ii_q % 9==0:
            pos[ii_elem,0] = pos[ii_elem,0]+qq_global[ii_q]
        elif ii_q % 9 == 1:
            pos[ii_elem,1] = pos[ii_elem,1]+qq_global[ii_q]
        elif ii_q % 9 == 2:
            pos[ii_elem,2] = pos[ii_elem,2]+qq_global[ii_q]
        elif ii_q % 9 == 3:
            ii_elem=ii_elem+1
    # The position of the nodes is loaded in the solution class
    # solution.xdef  : position of the node in x axis        
    # solution.ydef  : position of the node in y axis       
    # solution.zdef  : position of the node in z axis           
    solution.xdef = pos[:,0]
    solution.ydef = pos[:,1]
    solution.zdef = pos[:,2]      
    # The warping of the section is calculated for all the internal nodes
    # solution.warp_x1 : warping of the section in the x axis and the first element
    # solution.warp_y1 : warping of the section in the y axis and the first element
    # solution.warp_z1 : warping of the section in the z axis and the first element
    # solution.warp_x2 : warping of the section in the x axis and the second element
    # solution.warp_y2 : warping of the section in the y axis and the second element
    # solution.warp_z2 : warping of the section in the z axis and the second element
    solution.warp_x1 = []
    solution.warp_y1 = []
    solution.warp_z1 = []
    solution.warp_x2 = []
    solution.warp_y2 = []
    solution.warp_z2 = []
    # For every element
    for elem in mesh_elem:
        # warp_angle1 : projection of the derivative of the angular displacement in the normal direction for the fist element
        # warp_angle2 : projection of the derivative of the angular displacement in the normal direction for the second element
        warp_angle1 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[phi_d_vec[int(elem[1])]],[psi_d_vec[int(elem[1])]],[theta_d_vec[int(elem[1])]]]))
        warp_angle2 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[phi_d_vec[int(elem[2])]],[psi_d_vec[int(elem[2])]],[theta_d_vec[int(elem[2])]]]))
        # x_warp_nodes1 : warping in the global x component for the internal nodes of the first beam element node
        # y_warp_nodes1 : warping in the global y component for the internal nodes of the first beam element node
        # z_warp_nodes1 : warping in the global z component for the internal nodes of the first beam element node
        # x_warp_nodes2 : warping in the global x component for the internal nodes of the second beam element node
        # y_warp_nodes2 : warping in the global y component for the internal nodes of the second beam element node
        # z_warp_nodes2 : warping in the global z component for the internal nodes of the second beam element node
        x_warp_nodes1 = []
        y_warp_nodes1 = []
        z_warp_nodes1 = []
        x_warp_nodes2 = []
        y_warp_nodes2 = []
        z_warp_nodes2 = []
        # For every  section in the element
        warp1 = []
        for ii_sec in np.arange(len(section.elem[int(elem[1])])):
            # func_warp_elem1 : warping function of the section in the elements of the first beam element
            # func_warp1      : warping function of the section in the nodes of the first beam element
            # count           : counter of the elements that a node is contained for the first beam element
            func_warp_elem1 = np.zeros((len(section.elem[int(elem[1])][ii_sec][:,0]),))
            func_warp1      = np.zeros((len(section.points[int(elem[1])][ii_sec][:,0]),))
            count           = np.zeros((len(section.points[int(elem[1])][ii_sec][:,0]),))
            # for every node in the section
            for ii_elem in np.arange(len(func_warp_elem1)):
                # Take the warping function for the element
                # Add the value to the functions in the respective nodes
                # Count the number of times that is added to each node
                func_warp_elem1[ii_elem]                                        =  sol_phys.warp[int(elem[1])][ii_sec][ii_elem]
                func_warp1[int(section.elem[int(elem[1])][ii_sec][ii_elem,1])] += func_warp_elem1[ii_elem]
                func_warp1[int(section.elem[int(elem[1])][ii_sec][ii_elem,2])] += func_warp_elem1[ii_elem]
                count[int(section.elem[int(elem[1])][ii_sec][ii_elem,1])]      += 1
                count[int(section.elem[int(elem[1])][ii_sec][ii_elem,2])]      += 1
            # Obtain the value on the node as the mean on the elements
            func_warp1 = np.divide(func_warp1,count)
            # warp1   : warping of the section node in first element
            # x_warp1 : warping of the section node in first element and direction x
            # y_warp1 : warping of the section node in first element and direction y
            # z_warp1 : warping of the section node in first element and direction z
            for ii_point in np.arange(len(func_warp1)):
                warp1.append(func_warp1[ii_point]*warp_angle1)
        x_warp1 = np.zeros((len(warp1),))
        y_warp1 = np.zeros((len(warp1),))
        z_warp1 = np.zeros((len(warp1),))
        for ii_point in np.arange(len(warp1)):
            # Rotate the warping to the global frame and extract components
            warp_global1      = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp1[ii_point]]])
            x_warp1[ii_point] = warp_global1[0]
            y_warp1[ii_point] = warp_global1[1]
            z_warp1[ii_point] = warp_global1[2]
        # store the warping in solution class
        solution.warp_x1.append(x_warp1)
        solution.warp_y1.append(y_warp1)
        solution.warp_z1.append(z_warp1)
        # For every  section in the element
        warp2 = []
        for ii_sec in np.arange(len(section.elem[int(elem[2])])):
            # func_warp_elem2 : warping function of the section in the elements of the second beam element
            # func_warp2      : warping function of the section in the nodes of the first beam element
            # count           : counter of the elements that a node is contained for the second beam element
            func_warp_elem2 = np.zeros((len(section.elem[int(elem[2])][ii_sec][:,0]),))
            func_warp2      = np.zeros((len(section.points[int(elem[2])][ii_sec][:,0]),))
            count           = np.zeros((len(section.points[int(elem[2])][ii_sec][:,0]),))
            # for every node in the section
            for ii_elem in np.arange(len(func_warp_elem2)):
                # Take the warping function for the element
                # Add the value to the functions in the respective nodes
                # Count the number of times that is added to each node
                func_warp_elem2[ii_elem]                                        =  sol_phys.warp[int(elem[2])][ii_sec][ii_elem]
                func_warp2[int(section.elem[int(elem[2])][ii_sec][ii_elem,1])] += func_warp_elem2[ii_elem]
                func_warp2[int(section.elem[int(elem[2])][ii_sec][ii_elem,2])] += func_warp_elem2[ii_elem]
                count[int(section.elem[int(elem[2])][ii_sec][ii_elem,1])]      += 1
                count[int(section.elem[int(elem[2])][ii_sec][ii_elem,2])]      += 1
            # Obtain the value on the node as the mean on the elements
            func_warp2 = np.divide(func_warp2,count)
            # warp2   : warping of the section node in second element
            # x_warp2 : warping of the section node in second element and direction x
            # y_warp2 : warping of the section node in second element and direction y
            # z_warp2 : warping of the section node in second element and direction z
            for ii_point in np.arange(len(func_warp2)):
                warp2.append(func_warp2[ii_point]*warp_angle2)
        x_warp2 = np.zeros((len(warp2),))
        y_warp2 = np.zeros((len(warp2),))
        z_warp2 = np.zeros((len(warp2),))
        for ii_point in np.arange(len(warp2)):
            # Rotate the warping to the global frame and extract components
            warp_global2      = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp2[ii_point]]])
            x_warp2[ii_point] = warp_global2[0]
            y_warp2[ii_point] = warp_global2[1]
            z_warp2[ii_point] = warp_global2[2]
        # save components of the warping for every section in the beam element node
        solution.warp_x2.append(x_warp2)
        solution.warp_y2.append(y_warp2)
        solution.warp_z2.append(z_warp2)
    solution.q_vec = q_vec
    return solution

#%%
def store_solution_vib(case_setup,w_mode,mode,freq_mode,idx,MM_pow,MM_pow2,MM_LL,KK_LL,KK_global,ind_RR2,sol_phys,section,rot_mat,rotT_mat,num_point,num_elem,mesh_point,mesh_elem,mode_tot):
    # Function to store the results of the vibration calculation in a single class
    # case_setup : setup options
    # w_mode     : angular frequency
    # mode       : mode shape
    # freq_mode  : frequency
    # idx        : index to sort modes
    # MM_pow     : mass matrix power to 0.5
    # MM_pow2    : mass matrix power to -0.5
    # MM_LL      : mass matrix in the linear system
    # KK_LL      : stiffness matrix in the linear system
    # KK_global  : stiffness matrix
    # ind_RR2    : index to delete from the global matrix
    # sol_phys   : solid physics
    # section    : section information
    # rot_mat    : rotation matrix
    # rotT_mat   : transposed rotation matrix
    # num_point  : number of beam nodes
    # num_elem   : number of beam elements
    # mesh_point : nodes of the mesh
    # mesh_elem  : elements of the mesh
    # -------------------------------------------------------------------------
    # solution     : class to store the solution
    # store_nmodes : modes to store
    # mode_tot     : modes including restricted nodes
    nmaxmode = np.max([case_setup.n_mod*2,case_setup.savefile_modes*2])
    store_nmodes       = np.min([nmaxmode,int(len(freq_mode))])
    class solution:
        pass   
    solution.nsol = len(idx)
    solution.idx       = idx
    solution.w_mode    = w_mode[idx][:store_nmodes]
    solution.mode      = mode[:,idx][:,:store_nmodes]
    solution.modetot  = mode_tot[:,idx][:,:store_nmodes]
#    modeglob           = np.matmul(MM_pow,mode)
#    modeglob_abs       = abs(modeglob)
#    modeglob           = modeglob/modeglob_abs.max(axis=0)
#    solution.modeglob  = modeglob[:,idx]
    solution.MM_pow    = MM_pow
    solution.MM_pow2   = MM_pow2
    solution.MM_LL     = MM_LL
    solution.KK_LL     = KK_LL
    solution.freq_mode = freq_mode[idx][:store_nmodes]
#    mode_tot           = np.zeros((len(KK_global),len(KK_global)))
#    ind_ii2            = 0
#    store_nmodes       = np.min([case_setup.savefile_modes,len(freq_mode)])
    # save the global frequencies and modes
    # mode of the restriction is 0
#    for ind_ii in np.arange(len(KK_global)):
#        ind_jj2 = 0
#        for ind_jj in np.arange(len(KK_global)):
#            if ind_ii in ind_RR2:
#                mode_tot[ind_ii,ind_jj] = 0
#            elif ind_jj in ind_RR2:
#                mode_tot[ind_ii,ind_jj] = 0
#            else: 
#                mode_tot[ind_ii,ind_jj] = modeglob[ind_ii2,ind_jj2]
#                ind_jj2 = ind_jj2 +1
#        if ind_jj2>0:
#            ind_ii2 = ind_ii2+1
#    solution.modetot = mode_tot
    # u_mat     : modal shape of displacement in x axis
    # v_mat     : modal shape of displacement in y axis
    # w_mat     : modal shape of displacement in z axis
    # phi_mat   : modal shape of rotation in x axis
    # psi_mat   : modal shape of rotation in y axis
    # theta_mat : modal shape of rotation in z axis
    u_mat       = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    v_mat       = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    w_mat       = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    phi_mat     = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    psi_mat     = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    theta_mat   = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    phi_d_mat   = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    psi_d_mat   = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    theta_d_mat = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    mode_ii     = 0
    for ii_mode in np.arange(len(solution.modetot[0])): 
#        if len(np.where(ind_RR2==ii_mode)[0])==0:
        for ii_node in np.arange(len(solution.modetot)):
            if ii_node % 9==0:
                node_ii = int(np.floor(ii_node/9))
                u_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 1:
                node_ii = int(np.floor(ii_node/9))
                v_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 2:
                node_ii = int(np.floor(ii_node/9))
                w_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 3:
                node_ii = int(np.floor(ii_node/9))
                phi_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 4:
                node_ii = int(np.floor(ii_node/9))
                psi_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 5:
                node_ii = int(np.floor(ii_node/9))
                theta_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 6:
                node_ii = int(np.floor(ii_node/9))
                phi_d_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 7:
                node_ii = int(np.floor(ii_node/9))
                psi_d_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 8:
                node_ii = int(np.floor(ii_node/9))
                theta_d_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
        maxdatadis = np.max([np.max(abs(u_mat[:,mode_ii])),np.max(abs(v_mat[:,mode_ii])),np.max(abs(w_mat[:,mode_ii]))])
        maxdataang = np.max([np.max(abs(phi_mat[:,mode_ii])),np.max(abs(psi_mat[:,mode_ii])),np.max(abs(theta_mat[:,mode_ii]))])
        if maxdataang > np.pi/10:
            maxdata = maxdataang/(np.pi/6)
        elif maxdatadis > np.max(np.max(mesh_point))/10: 
            maxdata = maxdatadis/(np.max(np.max(mesh_point))/10)
        else:
            maxdata = np.max([maxdatadis,maxdataang])
        u_mat[:,mode_ii] /= maxdata
        v_mat[:,mode_ii] /= maxdata
        w_mat[:,mode_ii] /= maxdata
        phi_mat[:,mode_ii] /= maxdata
        psi_mat[:,mode_ii] /= maxdata
        theta_mat[:,mode_ii] /= maxdata
        phi_d_mat[:,mode_ii] /= maxdata
        psi_d_mat[:,mode_ii] /= maxdata
        theta_d_mat[:,mode_ii] /= maxdata
        mode_ii += 1
    solution.u = u_mat
    solution.v = v_mat
    solution.w = w_mat
    solution.phi = phi_mat
    solution.psi = psi_mat
    solution.theta = theta_mat
    solution.phi_d = phi_d_mat
    solution.psi_d = psi_d_mat
    solution.theta_d = theta_d_mat    
    # pos : position of the beam nodes
    pos = np.zeros((len(solution.u),len(solution.u[0]),3))
    ii_elem = 0
    for ii_mode in np.arange(store_nmodes):
        pos[:,ii_mode,0] = mesh_point[:,0]+solution.u[:,ii_mode]
        pos[:,ii_mode,1] = mesh_point[:,1]+solution.v[:,ii_mode]
        pos[:,ii_mode,2] = mesh_point[:,2]+solution.w[:,ii_mode]  
    solution.pos = pos
    # The position of the nodes is loaded in the solution class
    # solution.xdef  : position of the node in x axis        
    # solution.ydef  : position of the node in y axis       
    # solution.zdef  : position of the node in z axis  
    solution.xdef = pos[:,:,0]
    solution.ydef = pos[:,:,1]
    solution.zdef = pos[:,:,2]
    # warp_x1_vec : vector of the warping function x axis node 1 
    # warp_y1_vec : vector of the warping function y axis node 1   
    # warp_z1_vec : vector of the warping function z axis node 1  
    # warp_x2_vec : vector of the warping function x axis node 2 
    # warp_y2_vec : vector of the warping function y axis node 2   
    # warp_z2_vec : vector of the warping function z axis node 2
    warp_x1_vec = []
    warp_y1_vec = []
    warp_z1_vec = []
    warp_x2_vec = []
    warp_y2_vec = []
    warp_z2_vec = []
    # for all the modal shapes and elements of the mesh    
    for ii_mode in np.arange(store_nmodes):
        warpx1dat = def_vec_param(num_elem) 
        warpy1dat = def_vec_param(num_elem) 
        warpz1dat = def_vec_param(num_elem) 
        warpx2dat = def_vec_param(num_elem) 
        warpy2dat = def_vec_param(num_elem) 
        warpz2dat = def_vec_param(num_elem) 
        for elem in mesh_elem:
            # warp_angle1 : warping angle of node 1
            # warp_angle2 : warping angle of node 2
            warp_angle1 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[solution.phi_d[int(elem[1]),ii_mode]],[solution.psi_d[int(elem[1]),ii_mode]],[solution.theta_d[int(elem[1]),ii_mode]]]))
            warp_angle2 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[solution.phi_d[int(elem[2]),ii_mode]],[solution.psi_d[int(elem[2]),ii_mode]],[solution.theta_d[int(elem[2]),ii_mode]]]))
            warp1 = []
            warp2 = []
            # For all the subsections in the beam node
            for ii_sec in np.arange(len(section.elem[int(elem[1])])):
                # func_warp1 : warping function in the nodes of the section of beam node 1
                func_warp1 = np.zeros((len(section.points[int(elem[1])][ii_sec][:,0])))
                # for all the nodes in the section
                for ii_point in np.arange(len(func_warp1)):
                    # nn : number of elements using the node to calculate the average value
                    nn=0
                    # For all the elements in the section check if using the node
                    elemsec = np.where(section.elem[int(elem[1])][ii_sec][:,1:3]==ii_point)[0]
                    for jj_elem in np.arange(len(elemsec)):
                        func_warp1[ii_point] += sol_phys.warp[int(elem[1])][ii_sec][jj_elem]
                        nn                   += 1
#                    for jj_elem in np.arange(len(section.elem[int(elem[1])][ii_sec])):
#                        if section.elem[int(elem[1])][ii_sec][jj_elem,1]==ii_point or section.elem[int(elem[1])][ii_sec][jj_elem,2]==ii_point:
#                            func_warp1[ii_point] += sol_phys.warp[int(elem[1])][ii_sec][jj_elem]
#                            nn+=1
                    func_warp1[ii_point] /= nn
                # warp1   : warping of the section point of node 1
                # x_warp1 : warping of the section point of node 1 in x direction
                # y_warp1 : warping of the section point of node 1 in y direction
                # z_warp1 : warping of the section point of node 1 in z direction
                for ii_point in np.arange(len(func_warp1)):
                    warp1.append(func_warp1[ii_point]*warp_angle1)
            x_warp1 = np.zeros((len(warp1),))
            y_warp1 = np.zeros((len(warp1),))
            z_warp1 = np.zeros((len(warp1),))
            for ii_elem in np.arange(len(warp1)):
                warp_global1     = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp1[ii_elem]]])
                x_warp1[ii_elem] = warp_global1[0]
                y_warp1[ii_elem] = warp_global1[1]
                z_warp1[ii_elem] = warp_global1[2]
            for ii_appendwarp in np.arange(len(x_warp1)):
                warpx1dat[int(elem[0])].append(x_warp1[ii_appendwarp])
                warpy1dat[int(elem[0])].append(y_warp1[ii_appendwarp])
                warpz1dat[int(elem[0])].append(z_warp1[ii_appendwarp])
            for ii_sec in np.arange(len(section.elem[int(elem[2])])):
                # func_warp2 : warping function in the nodes of the section of beam node 2
                func_warp2 = np.zeros((len(section.points[int(elem[2])][ii_sec][:,0])))
                # for all the nodes in the section
                for ii_point in np.arange(len(func_warp2)):
                    # nn : number of elements using the node to calculate the average value
                    nn=0
                    # For all the elements in the section check if using the node
                    elemsec = np.where(section.elem[int(elem[2])][ii_sec][:,1:3]==ii_point)[0]
                    for jj_elem in np.arange(len(elemsec)):
                        func_warp2[ii_point] += sol_phys.warp[int(elem[2])][ii_sec][jj_elem]
                        nn                   += 1
#                    for jj_elem in np.arange(len(section.elem[int(elem[2])][ii_sec])):
#                        if section.elem[int(elem[2])][ii_sec][jj_elem,1]==ii_point or  section.elem[int(elem[2])][ii_sec][jj_elem,2]==ii_point:
#                            func_warp2[ii_point] += sol_phys.warp[int(elem[2])][ii_sec][jj_elem]
#                            nn+=1
                    func_warp2[ii_point] /= nn
                # warp2   : warping of the section point of node 2
                # x_warp2 : warping of the section point of node 2 in x direction
                # y_warp2 : warping of the section point of node 2 in y direction
                # z_warp2 : warping of the section point of node 2 in z direction
                for ii_point in np.arange(len(func_warp2)):
                    warp2.append(func_warp2[ii_point]*warp_angle2)
            x_warp2 = np.zeros((len(warp2),))
            y_warp2 = np.zeros((len(warp2),))
            z_warp2 = np.zeros((len(warp2),))
            for ii_elem in np.arange(len(warp2)):
                warp_global2 = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp2[ii_elem]]])
                x_warp2[ii_elem] = warp_global2[0]
                y_warp2[ii_elem] = warp_global2[1]
                z_warp2[ii_elem] = warp_global2[2]
            for ii_appendwarp in np.arange(len(x_warp2)):
                warpx2dat[int(elem[0])].append(x_warp2[ii_appendwarp])
                warpy2dat[int(elem[0])].append(y_warp2[ii_appendwarp])
                warpz2dat[int(elem[0])].append(z_warp2[ii_appendwarp])
        warp_x1_vec.append(warpx1dat)
        warp_y1_vec.append(warpy1dat)
        warp_z1_vec.append(warpz1dat)
        warp_x2_vec.append(warpx2dat)
        warp_y2_vec.append(warpy2dat)
        warp_z2_vec.append(warpz2dat)
    solution.warp_x1 = warp_x1_vec
    solution.warp_y1 = warp_y1_vec
    solution.warp_z1 = warp_z1_vec
    solution.warp_x2 = warp_x2_vec
    solution.warp_y2 = warp_y2_vec
    solution.warp_z2 = warp_z2_vec
    return solution

#%%
def store_solution_vibdamp(case_setup,w_mode,mode,freq_mode,idx,MM_pow,MM_pow2,MM_LL,KK_LL,KK_global,ind_RR2,sol_phys,section,rot_mat,rotT_mat,num_point,num_elem,mesh_point,mesh_elem,mode_tot):
    # Function to store the results of the vibration calculation in a single class
    # case_setup : setup options
    # w_mode     : angular frequency
    # mode       : mode shape
    # freq_mode  : frequency
    # idx        : index to sort modes
    # MM_pow     : mass matrix power to 0.5
    # MM_pow2    : mass matrix power to -0.5
    # MM_LL      : mass matrix in the linear system
    # KK_LL      : stiffness matrix in the linear system
    # KK_global  : stiffness matrix
    # ind_RR2    : index to delete from the global matrix
    # sol_phys   : solid physics
    # section    : section information
    # rot_mat    : rotation matrix
    # rotT_mat   : transposed rotation matrix
    # num_point  : number of beam nodes
    # num_elem   : number of beam elements
    # mesh_point : nodes of the mesh
    # mesh_elem  : elements of the mesh
    # -------------------------------------------------------------------------
    # solution     : class to store the solution
    # store_nmodes : modes to store
    # mode_tot     : modes including restricted nodes
    nmaxmode = np.max([case_setup.n_mod*2,case_setup.savefile_modes*2])
    store_nmodes       = np.min([nmaxmode,int(len(freq_mode)/2)])
    class solution:
        pass   
    solution.nsol = len(idx)
    solution.idx       = idx
    solution.w_mode    = w_mode[idx][:store_nmodes]
    solution.mode      = mode[:,idx][:,:store_nmodes]
    solution.modetot  = mode_tot[:,idx][:,:store_nmodes]
#    modeglob           = np.matmul(MM_pow,mode)
#    modeglob_abs       = abs(modeglob)
#    modeglob           = modeglob/modeglob_abs.max(axis=0)
#    solution.modeglob  = modeglob[:,idx]
    solution.MM_pow    = MM_pow
    solution.MM_pow2   = MM_pow2
    solution.MM_LL     = MM_LL
    solution.KK_LL     = KK_LL
    solution.freq_mode = freq_mode[idx][:store_nmodes]
#    mode_tot           = np.zeros((len(KK_global),len(KK_global)))
#    ind_ii2            = 0
#    store_nmodes       = np.min([case_setup.savefile_modes,len(freq_mode)])
    # save the global frequencies and modes
    # mode of the restriction is 0
#    for ind_ii in np.arange(len(KK_global)):
#        ind_jj2 = 0
#        for ind_jj in np.arange(len(KK_global)):
#            if ind_ii in ind_RR2:
#                mode_tot[ind_ii,ind_jj] = 0
#            elif ind_jj in ind_RR2:
#                mode_tot[ind_ii,ind_jj] = 0
#            else: 
#                mode_tot[ind_ii,ind_jj] = modeglob[ind_ii2,ind_jj2]
#                ind_jj2 = ind_jj2 +1
#        if ind_jj2>0:
#            ind_ii2 = ind_ii2+1
#    solution.modetot = mode_tot
    # u_mat     : modal shape of displacement in x axis
    # v_mat     : modal shape of displacement in y axis
    # w_mat     : modal shape of displacement in z axis
    # phi_mat   : modal shape of rotation in x axis
    # psi_mat   : modal shape of rotation in y axis
    # theta_mat : modal shape of rotation in z axis
    u_mat       = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    v_mat       = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    w_mat       = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    phi_mat     = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    psi_mat     = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    theta_mat   = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    phi_d_mat   = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    psi_d_mat   = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    theta_d_mat = np.zeros((num_point,store_nmodes)) #np.zeros((num_point,len(freq_mode)))
    mode_ii     = 0
    for ii_mode in np.arange(len(solution.modetot[0])): 
#        if len(np.where(ind_RR2==ii_mode)[0])==0:
        for ii_node in np.arange(len(solution.modetot)):
            if ii_node % 9==0:
                node_ii = int(np.floor(ii_node/9))
                u_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 1:
                node_ii = int(np.floor(ii_node/9))
                v_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 2:
                node_ii = int(np.floor(ii_node/9))
                w_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 3:
                node_ii = int(np.floor(ii_node/9))
                phi_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 4:
                node_ii = int(np.floor(ii_node/9))
                psi_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 5:
                node_ii = int(np.floor(ii_node/9))
                theta_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 6:
                node_ii = int(np.floor(ii_node/9))
                phi_d_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 7:
                node_ii = int(np.floor(ii_node/9))
                psi_d_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
            elif ii_node % 9 == 8:
                node_ii = int(np.floor(ii_node/9))
                theta_d_mat[node_ii,mode_ii] = solution.modetot[ii_node,ii_mode]
        maxdatadis = np.max([np.max(abs(u_mat[:,mode_ii])),np.max(abs(v_mat[:,mode_ii])),np.max(abs(w_mat[:,mode_ii]))])
        maxdataang = np.max([np.max(abs(phi_mat[:,mode_ii])),np.max(abs(psi_mat[:,mode_ii])),np.max(abs(theta_mat[:,mode_ii]))])
        if maxdataang > np.pi/10:
            maxdata = maxdataang/(np.pi/6)
        elif maxdatadis > np.max(np.max(mesh_point))/10: 
            maxdata = maxdatadis/(np.max(np.max(mesh_point))/10)
        else:
            maxdata = np.max([maxdatadis,maxdataang])
#            print(maxdata)
        u_mat[:,mode_ii] /= maxdata
        v_mat[:,mode_ii] /= maxdata
        w_mat[:,mode_ii] /= maxdata
        phi_mat[:,mode_ii] /= maxdata
        psi_mat[:,mode_ii] /= maxdata
        theta_mat[:,mode_ii] /= maxdata
        phi_d_mat[:,mode_ii] /= maxdata
        psi_d_mat[:,mode_ii] /= maxdata
        theta_d_mat[:,mode_ii] /= maxdata
        mode_ii += 1
    solution.u = u_mat
    solution.v = v_mat
    solution.w = w_mat
    solution.phi = phi_mat
    solution.psi = psi_mat
    solution.theta = theta_mat
    solution.phi_d = phi_d_mat
    solution.psi_d = psi_d_mat
    solution.theta_d = theta_d_mat    
    # pos : position of the beam nodes
    pos = np.zeros((len(solution.u),len(solution.u[0]),3))
    ii_elem = 0
    for ii_mode in np.arange(store_nmodes):
        pos[:,ii_mode,0] = mesh_point[:,0]+solution.u[:,ii_mode]
        pos[:,ii_mode,1] = mesh_point[:,1]+solution.v[:,ii_mode]
        pos[:,ii_mode,2] = mesh_point[:,2]+solution.w[:,ii_mode]  
    solution.pos = pos
    # The position of the nodes is loaded in the solution class
    # solution.xdef  : position of the node in x axis        
    # solution.ydef  : position of the node in y axis       
    # solution.zdef  : position of the node in z axis  
    solution.xdef = pos[:,:,0]
    solution.ydef = pos[:,:,1]
    solution.zdef = pos[:,:,2]
    # warp_x1_vec : vector of the warping function x axis node 1 
    # warp_y1_vec : vector of the warping function y axis node 1   
    # warp_z1_vec : vector of the warping function z axis node 1  
    # warp_x2_vec : vector of the warping function x axis node 2 
    # warp_y2_vec : vector of the warping function y axis node 2   
    # warp_z2_vec : vector of the warping function z axis node 2
    warp_x1_vec = []
    warp_y1_vec = []
    warp_z1_vec = []
    warp_x2_vec = []
    warp_y2_vec = []
    warp_z2_vec = []
    # for all the modal shapes and elements of the mesh    
    for ii_mode in np.arange(store_nmodes):
        warpx1dat = def_vec_param(num_elem) 
        warpy1dat = def_vec_param(num_elem) 
        warpz1dat = def_vec_param(num_elem) 
        warpx2dat = def_vec_param(num_elem) 
        warpy2dat = def_vec_param(num_elem) 
        warpz2dat = def_vec_param(num_elem) 
        for elem in mesh_elem:
            # warp_angle1 : warping angle of node 1
            # warp_angle2 : warping angle of node 2
            warp_angle1 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[solution.phi_d[int(elem[1]),ii_mode]],[solution.psi_d[int(elem[1]),ii_mode]],[solution.theta_d[int(elem[1]),ii_mode]]]))
            warp_angle2 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[solution.phi_d[int(elem[2]),ii_mode]],[solution.psi_d[int(elem[2]),ii_mode]],[solution.theta_d[int(elem[2]),ii_mode]]]))
            warp1 = []
            warp2 = []
            # For all the subsections in the beam node
            for ii_sec in np.arange(len(section.elem[int(elem[1])])):
                # func_warp1 : warping function in the nodes of the section of beam node 1
                func_warp1 = np.zeros((len(section.points[int(elem[1])][ii_sec][:,0])))
                # for all the nodes in the section
                for ii_point in np.arange(len(func_warp1)):
                    # nn : number of elements using the node to calculate the average value
                    nn=0
                    # For all the elements in the section check if using the node
                    elemsec = np.where(section.elem[int(elem[1])][ii_sec][:,1:3]==ii_point)[0]
                    for jj_elem in np.arange(len(elemsec)):
                        func_warp1[ii_point] += sol_phys.warp[int(elem[1])][ii_sec][jj_elem]
                        nn                   += 1
#                    for jj_elem in np.arange(len(section.elem[int(elem[1])][ii_sec])):
#                        if section.elem[int(elem[1])][ii_sec][jj_elem,1]==ii_point or section.elem[int(elem[1])][ii_sec][jj_elem,2]==ii_point:
#                            func_warp1[ii_point] += sol_phys.warp[int(elem[1])][ii_sec][jj_elem]
#                            nn+=1
                    func_warp1[ii_point] /= nn
                # warp1   : warping of the section point of node 1
                # x_warp1 : warping of the section point of node 1 in x direction
                # y_warp1 : warping of the section point of node 1 in y direction
                # z_warp1 : warping of the section point of node 1 in z direction
                for ii_point in np.arange(len(func_warp1)):
                    warp1.append(func_warp1[ii_point]*warp_angle1)
            x_warp1 = np.zeros((len(warp1),))
            y_warp1 = np.zeros((len(warp1),))
            z_warp1 = np.zeros((len(warp1),))
            for ii_elem in np.arange(len(warp1)):
                warp_global1     = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp1[ii_elem]]])
                x_warp1[ii_elem] = warp_global1[0]
                y_warp1[ii_elem] = warp_global1[1]
                z_warp1[ii_elem] = warp_global1[2]
            for ii_appendwarp in np.arange(len(x_warp1)):
                warpx1dat[int(elem[0])].append(x_warp1[ii_appendwarp])
                warpy1dat[int(elem[0])].append(y_warp1[ii_appendwarp])
                warpz1dat[int(elem[0])].append(z_warp1[ii_appendwarp])
            for ii_sec in np.arange(len(section.elem[int(elem[2])])):
                # func_warp2 : warping function in the nodes of the section of beam node 2
                func_warp2 = np.zeros((len(section.points[int(elem[2])][ii_sec][:,0])))
                # for all the nodes in the section
                for ii_point in np.arange(len(func_warp2)):
                    # nn : number of elements using the node to calculate the average value
                    nn=0
                    # For all the elements in the section check if using the node
                    elemsec = np.where(section.elem[int(elem[2])][ii_sec][:,1:3]==ii_point)[0]
                    for jj_elem in np.arange(len(elemsec)):
                        func_warp2[ii_point] += sol_phys.warp[int(elem[2])][ii_sec][jj_elem]
                        nn                   += 1
#                    for jj_elem in np.arange(len(section.elem[int(elem[2])][ii_sec])):
#                        if section.elem[int(elem[2])][ii_sec][jj_elem,1]==ii_point or  section.elem[int(elem[2])][ii_sec][jj_elem,2]==ii_point:
#                            func_warp2[ii_point] += sol_phys.warp[int(elem[2])][ii_sec][jj_elem]
#                            nn+=1
                    func_warp2[ii_point] /= nn
                # warp2   : warping of the section point of node 2
                # x_warp2 : warping of the section point of node 2 in x direction
                # y_warp2 : warping of the section point of node 2 in y direction
                # z_warp2 : warping of the section point of node 2 in z direction
                for ii_point in np.arange(len(func_warp2)):
                    warp2.append(func_warp2[ii_point]*warp_angle2)
            x_warp2 = np.zeros((len(warp2),))
            y_warp2 = np.zeros((len(warp2),))
            z_warp2 = np.zeros((len(warp2),))
            for ii_elem in np.arange(len(warp2)):
                warp_global2 = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp2[ii_elem]]])
                x_warp2[ii_elem] = warp_global2[0]
                y_warp2[ii_elem] = warp_global2[1]
                z_warp2[ii_elem] = warp_global2[2]
            for ii_appendwarp in np.arange(len(x_warp2)):
                warpx2dat[int(elem[0])].append(x_warp2[ii_appendwarp])
                warpy2dat[int(elem[0])].append(y_warp2[ii_appendwarp])
                warpz2dat[int(elem[0])].append(z_warp2[ii_appendwarp])
        warp_x1_vec.append(warpx1dat)
        warp_y1_vec.append(warpy1dat)
        warp_z1_vec.append(warpz1dat)
        warp_x2_vec.append(warpx2dat)
        warp_y2_vec.append(warpy2dat)
        warp_z2_vec.append(warpz2dat)
    solution.warp_x1 = warp_x1_vec
    solution.warp_y1 = warp_y1_vec
    solution.warp_z1 = warp_z1_vec
    solution.warp_x2 = warp_x2_vec
    solution.warp_y2 = warp_y2_vec
    solution.warp_z2 = warp_z2_vec
    return solution
#%%
def store_solution_vibae(case_setup,l_mode,w_mode,mode,freq_mode,MM_pow,MM_pow2,MM_LL,KK_LL,KK_global,ind_RR2,sol_phys,section,rot_mat,rotT_mat,num_point,num_elem,mesh_point,mesh_elem,mode_tot):
    # Function to store the results of the vibration calculation in a single class
    # case_setup : setup options
    # w_mode     : angular frequency
    # mode       : mode shape
    # freq_mode  : frequency
    # idx        : index to sort modes
    # MM_pow     : mass matrix power to 0.5
    # MM_pow2    : mass matrix power to -0.5
    # MM_LL      : mass matrix in the linear system
    # KK_LL      : stiffness matrix in the linear system
    # KK_global  : stiffness matrix
    # ind_RR2    : index to delete from the global matrix
    # sol_phys   : solid physics
    # section    : section information
    # rot_mat    : rotation matrix
    # rotT_mat   : transposed rotation matrix
    # num_point  : number of beam nodes
    # num_elem   : number of beam elements
    # mesh_point : nodes of the mesh
    # mesh_elem  : elements of the mesh
    # -------------------------------------------------------------------------
    # solution     : class to store the solution
    # store_nmodes : modes to store
    # mode_tot     : modes including restricted nodes
    nmaxmode = np.max([case_setup.n_mod,case_setup.savefile_modes]*2)
    store_nmodes       = np.min([nmaxmode,int(len(freq_mode)/2)])
    class solution:
        pass

    solution.l_mode    = l_mode[:store_nmodes,:] 
    solution.w_mode    = w_mode[:store_nmodes,:] 
    solution.mode      = mode[:,:store_nmodes,:]  
    solution.modetot  = mode_tot[:,:store_nmodes,:]  
    solution.freq_mode = freq_mode[:store_nmodes,:]  
#    solution.MM_pow    = MM_pow
#    solution.MM_pow2   = MM_pow2
#    solution.MM_LL     = MM_LL
#    solution.KK_LL     = KK_LL

    u_mat       = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    v_mat       = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    w_mat       = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    phi_mat     = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    psi_mat     = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    theta_mat   = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    phi_d_mat   = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    psi_d_mat   = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    theta_d_mat = np.zeros((num_point,store_nmodes,case_setup.numv), dtype=complex)#np.zeros((num_point,len(freq_mode),case_setup.numv), dtype=complex)
    for iiv_inf in np.arange(case_setup.numv):
        mode_ii     = 0
        for ii_mode in np.arange(len(solution.modetot[0,:,iiv_inf])): 
#            if len(np.where(ind_RR2==ii_mode)[0])==0:
                for ii_node in np.arange(len(solution.modetot[:9*num_point,0,iiv_inf])):
                    if ii_node % 9==0:
                        node_ii = int(np.floor(ii_node/9))
                        u_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                    elif ii_node % 9 == 1:
                        node_ii = int(np.floor(ii_node/9))
                        v_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                    elif ii_node % 9 == 2:
                        node_ii = int(np.floor(ii_node/9))
                        w_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                    elif ii_node % 9 == 3:
                        node_ii = int(np.floor(ii_node/9))
                        phi_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                    elif ii_node % 9 == 4:
                        node_ii = int(np.floor(ii_node/9))
                        psi_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                    elif ii_node % 9 == 5:
                        node_ii = int(np.floor(ii_node/9))
                        theta_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                    elif ii_node % 9 == 6:
                        node_ii = int(np.floor(ii_node/9))
                        phi_d_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                    elif ii_node % 9 == 7:
                        node_ii = int(np.floor(ii_node/9))
                        psi_d_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                    elif ii_node % 9 == 8:
                        node_ii = int(np.floor(ii_node/9))
                        theta_d_mat[node_ii,mode_ii,iiv_inf] = solution.modetot[ii_node,ii_mode,iiv_inf]
                maxdatadis = np.max([np.max(abs(u_mat[:,mode_ii,iiv_inf])),np.max(abs(v_mat[:,mode_ii,iiv_inf])),np.max(abs(w_mat[:,mode_ii,iiv_inf]))])
                maxdataang = np.max([np.max(abs(phi_mat[:,mode_ii,iiv_inf])),np.max(abs(psi_mat[:,mode_ii,iiv_inf])),np.max(abs(theta_mat[:,mode_ii,iiv_inf]))])
                if maxdataang > np.pi/50:
                    maxdata = maxdataang/(np.pi/50)
                elif maxdatadis > np.max(np.max(mesh_point))/50: 
                    maxdata = maxdatadis/(np.max(np.max(mesh_point))/50)
                else:
                    maxdata = np.max([maxdatadis,maxdataang])
                u_mat[:,mode_ii,iiv_inf] /= maxdata
                v_mat[:,mode_ii,iiv_inf] /= maxdata
                w_mat[:,mode_ii,iiv_inf] /= maxdata
                phi_mat[:,mode_ii,iiv_inf] /= maxdata
                psi_mat[:,mode_ii,iiv_inf] /= maxdata
                theta_mat[:,mode_ii,iiv_inf] /= maxdata
                phi_d_mat[:,mode_ii,iiv_inf] /= maxdata
                psi_d_mat[:,mode_ii,iiv_inf] /= maxdata
                theta_d_mat[:,mode_ii,iiv_inf] /= maxdata
                mode_ii += 1

    solution.u = np.real(u_mat)
    solution.v = np.real(v_mat)
    solution.w = np.real(w_mat)
    solution.phi = np.real(phi_mat)
    solution.psi = np.real(psi_mat)
    solution.theta = np.real(theta_mat)
    solution.phi_d = np.real(phi_d_mat)
    solution.psi_d = np.real(psi_d_mat)
    solution.theta_d = np.real(theta_d_mat)
    # pos : position of the beam nodes
    pos = np.zeros((len(solution.u),len(solution.u[0]),3,case_setup.numv), dtype=complex)
    for iiv_inf in np.arange(case_setup.numv):    
        ii_elem = 0
        for ii_mode in np.arange(store_nmodes):
            pos[:,ii_mode,0,iiv_inf] = mesh_point[:,0]+solution.u[:,ii_mode,iiv_inf]
            pos[:,ii_mode,1,iiv_inf] = mesh_point[:,1]+solution.v[:,ii_mode,iiv_inf]
            pos[:,ii_mode,2,iiv_inf] = mesh_point[:,2]+solution.w[:,ii_mode,iiv_inf]  
        solution.pos = pos
    # The position of the nodes is loaded in the solution class
    # solution.xdef  : position of the node in x axis        
    # solution.ydef  : position of the node in y axis       
    # solution.zdef  : position of the node in z axis  
    solution.xdef = pos[:,:,0]
    solution.ydef = pos[:,:,1]
    solution.zdef = pos[:,:,2]
    # warp_x1_vec : vector of the warping function x axis node 1 
    # warp_y1_vec : vector of the warping function y axis node 1   
    # warp_z1_vec : vector of the warping function z axis node 1  
    # warp_x2_vec : vector of the warping function x axis node 2 
    # warp_y2_vec : vector of the warping function y axis node 2   
    # warp_z2_vec : vector of the warping function z axis node 2
    warp_x1_vec = def_vec_param(case_setup.numv)
    warp_y1_vec = def_vec_param(case_setup.numv)
    warp_z1_vec = def_vec_param(case_setup.numv)
    warp_x2_vec = def_vec_param(case_setup.numv)
    warp_y2_vec = def_vec_param(case_setup.numv)
    warp_z2_vec = def_vec_param(case_setup.numv)
    for iiv_inf in np.arange(case_setup.numv):   
        # for all the modal shapes and elements of the mesh    
        for ii_mode in np.arange(store_nmodes):
            warpx1dat = def_vec_param(num_elem) 
            warpy1dat = def_vec_param(num_elem) 
            warpz1dat = def_vec_param(num_elem) 
            warpx2dat = def_vec_param(num_elem) 
            warpy2dat = def_vec_param(num_elem) 
            warpz2dat = def_vec_param(num_elem) 
            for elem in mesh_elem:
                # warp_angle1 : warping angle of node 1
                # warp_angle2 : warping angle of node 2
                warp_angle1 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[solution.phi_d[int(elem[1]),ii_mode,iiv_inf]],[solution.psi_d[int(elem[1]),ii_mode,iiv_inf]],[solution.theta_d[int(elem[1]),ii_mode,iiv_inf]]]))
                warp_angle2 = np.linalg.norm(np.matmul(rotT_mat[int(elem[0]),:,:],[[solution.phi_d[int(elem[2]),ii_mode,iiv_inf]],[solution.psi_d[int(elem[2]),ii_mode,iiv_inf]],[solution.theta_d[int(elem[2]),ii_mode,iiv_inf]]]))
                warp1 = []
                warp2 = []
                # For all the subsections in the beam node
                for ii_sec in np.arange(len(section.elem[int(elem[1])])):
                    # func_warp1 : warping function in the nodes of the section of beam node 1
                    func_warp1 = np.zeros((len(section.points[int(elem[1])][ii_sec][:,0])))
                    # for all the nodes in the section
                    for ii_point in np.arange(len(func_warp1)):
                        # nn : number of elements using the node to calculate the average value
                        nn=0
                        # For all the elements in the section check if using the node
                        elemsec = np.where(section.elem[int(elem[1])][ii_sec][:,1:3]==ii_point)[0]
                        for jj_elem in np.arange(len(elemsec)):
                            func_warp1[ii_point] += sol_phys.warp[int(elem[1])][ii_sec][jj_elem]
                            nn                   += 1
    #                    for jj_elem in np.arange(len(section.elem[int(elem[1])][ii_sec])):
    #                        if section.elem[int(elem[1])][ii_sec][jj_elem,1]==ii_point or section.elem[int(elem[1])][ii_sec][jj_elem,2]==ii_point:
    #                            func_warp1[ii_point] += sol_phys.warp[int(elem[1])][ii_sec][jj_elem]
    #                            nn+=1
                        func_warp1[ii_point] /= nn
                    # warp1   : warping of the section point of node 1
                    # x_warp1 : warping of the section point of node 1 in x direction
                    # y_warp1 : warping of the section point of node 1 in y direction
                    # z_warp1 : warping of the section point of node 1 in z direction
                    for ii_point in np.arange(len(func_warp1)):
                        warp1.append(func_warp1[ii_point]*warp_angle1)
                x_warp1 = np.zeros((len(warp1),))
                y_warp1 = np.zeros((len(warp1),))
                z_warp1 = np.zeros((len(warp1),))
                for ii_elem in np.arange(len(warp1)):
                    warp_global1     = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp1[ii_elem]]])
                    x_warp1[ii_elem] = warp_global1[0]
                    y_warp1[ii_elem] = warp_global1[1]
                    z_warp1[ii_elem] = warp_global1[2]
                for ii_appendwarp in np.arange(len(x_warp1)):
                    warpx1dat[int(elem[0])].append(x_warp1[ii_appendwarp])
                    warpy1dat[int(elem[0])].append(y_warp1[ii_appendwarp])
                    warpz1dat[int(elem[0])].append(z_warp1[ii_appendwarp])
                for ii_sec in np.arange(len(section.elem[int(elem[2])])):
                    # func_warp2 : warping function in the nodes of the section of beam node 2
                    func_warp2 = np.zeros((len(section.points[int(elem[2])][ii_sec][:,0])))
                    # for all the nodes in the section
                    for ii_point in np.arange(len(func_warp2)):
                        # nn : number of elements using the node to calculate the average value
                        nn=0
                        # For all the elements in the section check if using the node
                        elemsec = np.where(section.elem[int(elem[2])][ii_sec][:,1:3]==ii_point)[0]
                        for jj_elem in np.arange(len(elemsec)):
                            func_warp2[ii_point] += sol_phys.warp[int(elem[2])][ii_sec][jj_elem]
                            nn                   += 1
    #                    for jj_elem in np.arange(len(section.elem[int(elem[2])][ii_sec])):
    #                        if section.elem[int(elem[2])][ii_sec][jj_elem,1]==ii_point or  section.elem[int(elem[2])][ii_sec][jj_elem,2]==ii_point:
    #                            func_warp2[ii_point] += sol_phys.warp[int(elem[2])][ii_sec][jj_elem]
    #                            nn+=1
                        func_warp2[ii_point] /= nn
                    # warp2   : warping of the section point of node 2
                    # x_warp2 : warping of the section point of node 2 in x direction
                    # y_warp2 : warping of the section point of node 2 in y direction
                    # z_warp2 : warping of the section point of node 2 in z direction
                    for ii_point in np.arange(len(func_warp2)):
                        warp2.append(func_warp2[ii_point]*warp_angle2)
                x_warp2 = np.zeros((len(warp2),))
                y_warp2 = np.zeros((len(warp2),))
                z_warp2 = np.zeros((len(warp2),))
                for ii_elem in np.arange(len(warp2)):
                    warp_global2 = np.matmul(rot_mat[int(elem[0]),:,:],[[0],[0],[warp2[ii_elem]]])
                    x_warp2[ii_elem] = warp_global2[0]
                    y_warp2[ii_elem] = warp_global2[1]
                    z_warp2[ii_elem] = warp_global2[2]
                for ii_appendwarp in np.arange(len(x_warp2)):
                    warpx2dat[int(elem[0])].append(x_warp2[ii_appendwarp])
                    warpy2dat[int(elem[0])].append(y_warp2[ii_appendwarp])
                    warpz2dat[int(elem[0])].append(z_warp2[ii_appendwarp])
            warp_x1_vec[iiv_inf].append(warpx1dat)
            warp_y1_vec[iiv_inf].append(warpy1dat)
            warp_z1_vec[iiv_inf].append(warpz1dat)
            warp_x2_vec[iiv_inf].append(warpx2dat)
            warp_y2_vec[iiv_inf].append(warpy2dat)
            warp_z2_vec[iiv_inf].append(warpz2dat)
    solution.warp_x1 = warp_x1_vec
    solution.warp_y1 = warp_y1_vec
    solution.warp_z1 = warp_z1_vec
    solution.warp_x2 = warp_x2_vec
    solution.warp_y2 = warp_y2_vec
    solution.warp_z2 = warp_z2_vec
    return solution