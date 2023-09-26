# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 10:16:42 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
linsys_solver  : file containing functions for the resolution of the linear system
last_version   : 17-02-2021
modified_by    : Andres Cremades Botella
"""
import numpy as np
from bin.solver_schemes import select_solver
from scipy.linalg import fractional_matrix_power
from scipy import sparse
import cmath

#%% Functions
def stat_linsys(KK_global,qq_global,FF_global,RR_global):
    # Function to solve the linear equation system of the static problem
    # KK_global : stiffness matrix of the problem
    # qq_global : displacement vector of the problem
    # FF_global : load vector of the problem
    # ---------------------------------------------------------------------------------------------------------
    # KK_LR   : submatrix of the stiffness matrix that relates the restricted displacements with the loads
    # KK_RL   : submatrix of the stiffness matrix that relates the free loads with the displacements 
    # KK_RR   : submatrix of the stiffness matrix that relates the free loads with the restricted displacements
    # KK_LL   : submatrix of the stiffness matrix that relates the restricted loads with the free displacements
    # ind_LL  : index of the rows and columns of the KK_LL
    # ind_LL2 : index ind_LL after deleting only zero rows and columns
    # ind_RR : index to delete from KK_LL
    qq_global = qq_global.copy()
    FF_global = FF_global.copy()
    KK_LR   = []
    KK_RL   = []
    KK_RR   = []
    ind_RR  = []
    ind_LL  = []
    ind_LL2 = []
    # Check if the value of the displacement is restricted, 
    #   * if yes: add row to KK_RR or KK_LR
    #   * if no: add row to KK_LL or KK_LL
    for ind_qq in np.arange(len(qq_global)):
        if np.isnan(qq_global[ind_qq])==0:
            ind_RR.append(ind_qq)                
        else:
            ind_LL.append(ind_qq)
    # LL_deleterow   : rows to delete in ind_LL
    # ind_RR2        : index to delete from KK_LL after adding the zero rows
    # tol            : tolerance to determine a zero value row
    LL_deleterow = []
    ind_RR2      = ind_RR.copy()
    ind_RR2joint = []
    tol          = 1e-20
    # The zero values are checked for every row
    for ii_row in np.arange(len(KK_global)):
        # Determine if it is a zero row
        if all(KK_global[ii_row,:] == 0) or sum(abs(KK_global[ii_row,:]))<tol:
            # If the row is not deleted yet
            if len(np.where(np.array(ind_RR)==ii_row)[0])==0:
                # Give 0 value to the displacement
                qq_global[ii_row] = 0 
                if any(RR_global[ii_row,:] != 0):
                    ind_RR2joint.append(ii_row)
                # Add the value of the row to the delete row vector
                ind_RR2.append(ii_row)
                # ind_LL_del  : index to delete from ind_LL vector
                ind_LL_del = np.where(np.array(ind_LL)==ii_row)[0]
                if len(ind_LL_del)>0:
                    # If ind_LL_del is not empty delete the values in ind_LL_del
                    for kk_indLLdel in ind_LL_del:
                        LL_deleterow.append(kk_indLLdel)
    # KK_LL_0  : stiffness matrix after deleting rows
    # KK_LL_1  : stiffness matrix after deleting columns
    # KK_LL    : stiffness matrix after deleting rows and columns
    if len(LL_deleterow)>0:
        ind_LL2 = np.delete(np.array(ind_LL),np.array(LL_deleterow),0)
    else:
        ind_LL2 = ind_LL
    KK_LL_0 = np.delete(KK_global,ind_RR2,0)
    KK_LL_1 = np.delete(KK_global,ind_RR2,1)
    KK_LL   = np.delete(KK_LL_0,ind_RR2,1)
    # The submatrices are obtained
    # KK_RR_0  : submatrix of the stiffness after taking the rows of KK_RR of
    #            the global stiffness matrix
    KK_LR   = KK_LL_0[:,ind_RR2]
    KK_RL   = KK_LL_1[ind_RR2,:]
    KK_RR_0 = KK_global[ind_RR2,:]
    KK_RR   = KK_RR_0[:,ind_RR2]
    RR_J_vec = RR_global[ind_RR2joint,:]
    RR_SJ   = np.delete(RR_J_vec,ind_RR2joint,1)
    RR_JJ   = RR_J_vec[:,ind_RR2joint]
    # FF_L      : rows of the load vector corresponding with the 
    #            dertermined values
    # FF_R      : rows of the load vector corresponding with undetermined values
    # FF_LL_gen : load vector to solve after adding the determined displacements
    #             multiplied by the stiffness matrix KK_LR
    FF_L       = np.delete(FF_global,ind_RR2,0)
    FF_R       = FF_global[ind_RR2]        
    FF_LL_gen  = FF_L-np.matmul(KK_LR,qq_global[ind_RR2])
    # The undetermined displacements are added to the displacement vectror
    # q_vec : undetermined values of the displacemnt
#    for iirow in np.arange(len(KK_LL)):
#        mdiv              = max(abs(KK_LL[iirow,:]))
#        KK_LL[iirow,:]   /= mdiv
#        FF_LL_gen[iirow] /= mdiv
    q_vec              = np.linalg.solve(KK_LL,FF_LL_gen)
    qq_global[ind_LL2] = q_vec
    qq_s = np.delete(qq_global,ind_RR2joint)
    qq_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qq_s))
    qq_global[ind_RR2joint] = qq_j
    # The undetermined forces are obtained from the solved displacement field 
    # and added to the load vector
    # f_vec  : undetermined forces
    f_vec              = np.matmul(KK_RR,qq_global[ind_RR2])+np.matmul(KK_RL,q_vec)
    FF_global[ind_RR2] = f_vec
    return qq_global, FF_global, q_vec

#%%
def dyn_linsys(case_setup,KK_global,MM_global,CC_global,RR_global,qq_global,qq_der_global,qq_derder_global,FF_global,ii_time,init_it,time_stp,time_stp_op,ind_RR2,sys_prev):
    # Function to solve the linear equation system of the dynamic problem
    # KK_global        : stiffness matrix of the problem
    # qq_global        : displacement vector of the problem
    # qq_der_global    : velocity vector of the problem
    # qq_derder_global : acceleration vector of the problem
    # FF_global        : load vector of the problem
    # ii_time          : time index
    # init_it          : initial iterations
    # time_stp         : time step 
    # time_stp_op      : internal time step
    # ind_RR2          : index to delete from the matices
    # vibfree          : free vibration analysis solutions
    # sys_prev         : forces in the previous time step
    # ---------------------------------------------------------------------------------------------------------
    # KK_LR   : submatrix of the stiffness matrix that relates the restricted displacements with the loads
    # KK_RL   : submatrix of the stiffness matrix that relates the free loads with the displacements 
    # KK_RR   : submatrix of the stiffness matrix that relates the free loads with the restricted displacements
    # KK_LL   : submatrix of the stiffness matrix that relates the restricted loads with the free displacements
    # MM_LR   : submatrix of the mass matrix that relates the restricted displacements with the loads
    # MM_RL   : submatrix of the mass matrix that relates the free loads with the displacements 
    # MM_RR   : submatrix of the mass matrix that relates the free loads with the restricted displacements
    # MM_LL   : submatrix of the mass matrix that relates the restricted loads with the free displacements
    # CC_LR   : submatrix of the damping matrix that relates the restricted displacements with the loads
    # CC_RL   : submatrix of the damping matrix that relates the free loads with the displacements 
    # CC_RR   : submatrix of the damping matrix that relates the free loads with the restricted displacements
    # CC_LL   : submatrix of the damping matrix that relates the restricted loads with the free displacements
    # ind_LL  : index of the rows and columns of the KK_LL
    # ind_LL2 : index ind_LL after deleting only zero rows and columns
    # ind_RR : index to delete from KK_LL
    try:
        ind_RR = ind_RR2.copy()
    except:
        ind_RR  = []
    ind_RR2 = []        
    KK_LR   = []
    KK_RL   = []
    KK_RR   = []
    MM_LR   = []
    MM_RL   = []
    MM_RR   = []
    CC_LR   = []
    CC_RL   = []
    CC_RR   = []
    ind_RR  = []
    ind_LL  = []
    ind_LL2 = []
    ind_RR2joint = []
    # Check if the value of the displacement is restricted, 
    #   * if yes: add row to KK_RR or KK_LR
    #   * if no: add row to KK_LL or KK_LL
    for ind_qq in np.arange(len(qq_global)):
        if np.isnan(qq_global[ind_qq,ii_time])==0:
            ind_RR.append(ind_qq)                
        else:
            ind_LL.append(ind_qq)
    # LL_deleterow   : rows to delete in ind_LL
    # ind_RR2        : index to delete from KK_LL after adding the zero rows
    # tol            : tolerance to determine a zero value row
    LL_deleterow = []
    ind_RR2      = ind_RR.copy()
    tol          = 1e-20
    # The zero values are checked for every row
    for ii_row in np.arange(len(KK_global)):
        # Determine if it is a zero row
        if all(KK_global[ii_row,:] == 0) or sum(abs(KK_global[ii_row,:]))<tol:
            # If the row is not deleted yet
            if len(np.where(np.array(ind_RR)==ii_row)[0])==0:
                # Give 0 value to the displacement
                qq_global[ii_row,ii_time] = 0
                qq_der_global[ii_row,ii_time] = 0
                qq_derder_global[ii_row,ii_time] = 0
                if any(RR_global[ii_row,:] != 0):
                    ind_RR2joint.append(ii_row)
                # Add the value of the row to the delete row vector
                ind_RR2.append(ii_row)
                # ind_LL_del  : index to delete from ind_LL vector
                ind_LL_del = np.where(np.array(ind_LL)==ii_row)[0]
                if len(ind_LL_del)>0:
                    # If ind_LL_del is not empty delete the values in ind_LL_del
                    for kk_indLLdel in ind_LL_del:
                        LL_deleterow.append(kk_indLLdel)
    # KK_LL_0  : stiffness matrix after deleting rows
    # KK_LL_1  : stiffness matrix after deleting columns
    # KK_LL    : stiffness matrix after deleting rows and columns
    # MM_LL_0  : mass matrix after deleting rows
    # MM_LL_1  : mass matrix after deleting columns
    # MM_LL    : mass matrix after deleting rows and columns
    # CC_LL_0  : damping matrix after deleting rows
    # CC_LL_1  : damping matrix after deleting columns
    # CC_LL    : damping matrix after deleting rows and columns
    ind_LL2 = np.delete(np.array(ind_LL),np.array(LL_deleterow),0)
    KK_LL_0 = np.delete(KK_global,ind_RR2,0)
    KK_LL_1 = np.delete(KK_global,ind_RR2,1)
    KK_LL   = np.delete(KK_LL_0,ind_RR2,1)
    MM_LL_0 = np.delete(MM_global,ind_RR2,0)
    MM_LL_1 = np.delete(MM_global,ind_RR2,1)
    MM_LL   = np.delete(MM_LL_0,ind_RR2,1)
    CC_LL_0 =  np.delete(CC_global,ind_RR2,0)
    CC_LL_1 =  np.delete(CC_global,ind_RR2,1)
    CC_LL   = np.delete(CC_LL_0,ind_RR2,1)
    # The submatrices are obtained
    # KK_RR_0  : submatrix of the stiffness after taking the rows of KK_RR of
    #            the global stiffness matrix
    # MM_RR_0  : submatrix of the mass after taking the rows of KK_RR of
    #            the global stiffness matrix
    # CC_RR_0  : submatrix of the damping after taking the rows of KK_RR of
    #            the global stiffness matrix
    KK_LR   = KK_LL_0[:,ind_RR2]
    KK_RL   = KK_LL_1[ind_RR2,:]
    KK_RR_0 = KK_global[ind_RR2,:]
    KK_RR   = KK_RR_0[:,ind_RR2]
    MM_LR   = MM_LL_0[:,ind_RR2]
    MM_RL   = MM_LL_1[ind_RR2,:]
    MM_RR_0 = MM_global[ind_RR2,:]
    MM_RR   = MM_RR_0[:,ind_RR2]
    CC_LR   = CC_LL_0[:,ind_RR2]
    CC_RL   = CC_LL_1[ind_RR2,:]
    CC_RR_0 = CC_global[ind_RR2,:]
    CC_RR   = CC_RR_0[:,ind_RR2]
    RR_J_vec = RR_global[ind_RR2joint,:]
    RR_SJ   = np.delete(RR_J_vec,ind_RR2joint,1)
    RR_JJ   = RR_J_vec[:,ind_RR2joint]
    # qq_L         : dof of the system
    # qq_R         : restricted displacements of the system
    # qq_der_L     : derivative of the dof of the system
    # qq_der_R     : restricted displacements derivatives of the system
    # qq_derder_R  : restricted displacements second derivatives of the system
    # FF_L         : rows of the load vector corresponding with the 
    #                dertermined values
    # x_vec        : total state unknowns vector
    # AA_mat       : total state matix
    # bb_mat       : total state vector
    qq_L                          = qq_global[ind_LL2,ii_time-1]
    qq_R                          = qq_global[ind_RR2,ii_time-1]
    qq_der_L                      = qq_der_global[ind_LL2,ii_time-1]
    qq_der_R                      = qq_der_global[ind_RR2,ii_time-1]
    qq_derder_R                   = qq_derder_global[ind_RR2,ii_time-1]
    FF_L                          = FF_global[ind_LL2,ii_time-1]-np.matmul(KK_LR,qq_R)-np.matmul(MM_LR,qq_derder_R)-np.matmul(CC_LR,qq_der_R)
    x_vec                         = np.zeros((len(qq_L)+len(qq_der_L),))
    x_vec[:len(qq_L)]             = qq_L
    x_vec[len(qq_L):]             = qq_der_L   
    AA_mat                        = np.zeros((len(x_vec),len(x_vec)))
    AA_mat[:len(qq_L),len(qq_L):] = np.identity(len(qq_L))
    AA_mat[len(qq_L):,:len(qq_L)] = -np.matmul(np.linalg.inv(MM_LL),KK_LL)
    AA_mat[len(qq_L):,len(qq_L):] = -np.matmul(np.linalg.inv(MM_LL),CC_LL)
    bb_mat                        = np.zeros((len(x_vec),))
    bb_mat[len(qq_L):]            = np.matmul(np.linalg.inv(MM_LL),FF_L)
    # adatol : tolerance of the adaptative solver
    try:
        adatol = case_setup.adatol
    except:
        adatol = 1e-3
    if ii_time < init_it:
        # For the initialization of the calculation
        # The load vector of the previous internal iterations is used for extrapolate the new internal point
        # sys_prev : class containing the value of the load vectors
        if ii_time > 4:
            sys_prev.bderderderder = ((((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp-sys_prev.bderder)/time_stp-sys_prev.bderderder)/time_stp
        if ii_time > 3:
            sys_prev.bderderder = (((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp-sys_prev.bderder)/time_stp
        if ii_time > 2:
            sys_prev.bderder = ((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp
        if ii_time > 1:
            sys_prev.bder = (bb_mat-sys_prev.b)/time_stp
        if ii_time == 1:
            sys_prev.bder          = 0*bb_mat
            sys_prev.bderder       = 0*bb_mat
            sys_prev.bderderder    = 0*bb_mat
            sys_prev.bderderderder = 0*bb_mat
        sys_prev.b = bb_mat
        if case_setup.solver_init == "BACK4_IMP" or case_setup.solver_init == "AM4_IMP" or case_setup.solver_init == "AM3_PC" or case_setup.solver_init == "HAM3_PC":
            case_setup.solver_init = "ADA_RK4_EXP"
        x_vec,time_stp_op                 = select_solver(case_setup.solver_init,AA_mat,bb_mat,x_vec,time_stp,[],time_stp_op,adatol,sys_prev.bder,sys_prev.bderder,sys_prev.bderderder,sys_prev.bderderderder)
        x_vec_der                         = np.matmul(AA_mat,x_vec)+bb_mat
        qq_global[ind_LL2,ii_time]        = x_vec[:len(qq_L)]
        qq_der_global[ind_LL2,ii_time]    = x_vec[len(qq_L):]
        qq_derder_global[ind_LL2,ii_time] = x_vec_der[len(qq_L):]
        qq_s = np.delete(qq_global[:,ii_time],ind_RR2joint)
        qq_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qq_s))
        qq_global[ind_RR2joint,ii_time] = qq_j
        qqd_s = np.delete(qq_der_global[:,ii_time],ind_RR2joint)
        qqd_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qqd_s))
        qq_der_global[ind_RR2joint,ii_time] = qqd_j
        qqdd_s = np.delete(qq_derder_global[:,ii_time],ind_RR2joint)
        qqdd_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qqdd_s))
        qq_derder_global[ind_RR2joint,ii_time] = qqdd_j
        f_vec                             = np.matmul(KK_RR,qq_global[ind_RR2,ii_time-1])+np.matmul(KK_RL,qq_global[ind_LL2,ii_time-1])+np.matmul(MM_RR,qq_derder_global[ind_RR2,ii_time-1])+np.matmul(MM_RL,qq_derder_global[ind_LL2,ii_time-1])+np.matmul(CC_RR,qq_der_global[ind_RR2,ii_time-1])+np.matmul(CC_RL,qq_der_global[ind_LL2,ii_time-1])
        FF_global[ind_RR2,ii_time]        = f_vec
    else: 
        # The load vector of the previous internal iterations is used for extrapolate the new internal point
        # sys_prev : class containing the value of the load vectors
        sys_prev.bderderderder = ((((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp-sys_prev.bderder)/time_stp-sys_prev.bderderder)/time_stp
        sys_prev.bderderder    = (((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp-sys_prev.bderder)/time_stp
        sys_prev.bderder       = ((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp
        sys_prev.bder          = (bb_mat-sys_prev.b)/time_stp
        sys_prev.b             = bb_mat
        # qq_L_ini     : initial values of the displacements
        # qq_der_L_ini : initial values of the displacement derivatives
        # x_ini        : initial state unknowns vector
        if  case_setup.solver_init == "AM4_IMP" or  case_setup.solver_init == "HAM3_PC":
            qq_L_ini     = qq_global[ind_LL2,ii_time-4:ii_time-1]
            qq_der_L_ini = qq_der_global[ind_LL2,ii_time-4:ii_time-1]
            x_ini        = np.zeros((len(qq_L)+len(qq_der_L),3))
            x_ini[:len(qq_L),:] = qq_L_ini
            x_ini[len(qq_L):,:] = qq_der_L_ini
        elif case_setup.solver_init == "BACK4_IMP" or case_setup.solver_init == "AM3_PC":
            qq_L_ini     = qq_global[ind_LL2,ii_time-3:ii_time-1]
            qq_der_L_ini = qq_der_global[ind_LL2,ii_time-3:ii_time-1]
            x_ini        = np.zeros((len(qq_L)+len(qq_der_L),2))
            x_ini[:len(qq_L),:] = qq_L_ini
            x_ini[len(qq_L):,:] = qq_der_L_ini
        else:
            x_ini = []
        x_vec,time_stp_op                 = select_solver(case_setup.solver,AA_mat,bb_mat,x_vec,time_stp,x_ini,time_stp_op,adatol,sys_prev.bder,sys_prev.bderder,sys_prev.bderderder,sys_prev.bderderderder)
        x_vec_der                         = np.matmul(AA_mat,x_vec)+bb_mat
        qq_global[ind_LL2,ii_time]        = x_vec[:len(qq_L)]
        qq_der_global[ind_LL2,ii_time]    = x_vec[len(qq_L):]
        qq_derder_global[ind_LL2,ii_time] = (qq_der_global[ind_LL2,ii_time]-qq_der_global[ind_LL2,ii_time-1])/time_stp
        qq_s = np.delete(qq_global[:,ii_time],ind_RR2joint)
        qq_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qq_s))
        qq_global[ind_RR2joint,ii_time] = qq_j
        qqd_s = np.delete(qq_der_global[:,ii_time],ind_RR2joint)
        qqd_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qqd_s))
        qq_der_global[ind_RR2joint,ii_time] = qqd_j
        qqdd_s = np.delete(qq_derder_global[:,ii_time],ind_RR2joint)
        qqdd_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qqdd_s))
        qq_derder_global[ind_RR2joint,ii_time] = qqdd_j
        f_vec                             = np.matmul(KK_RR,qq_global[ind_RR2,ii_time-1])+np.matmul(KK_RL,qq_global[ind_LL2,ii_time-1])+np.matmul(MM_RR,qq_derder_global[ind_RR2,ii_time-1])+np.matmul(MM_RL,qq_derder_global[ind_LL2,ii_time-1])+np.matmul(CC_RR,qq_der_global[ind_RR2,ii_time-1])+np.matmul(CC_RL,qq_der_global[ind_LL2,ii_time-1])
        FF_global[ind_RR2,ii_time]        = f_vec
    return qq_global,qq_der_global,qq_derder_global,FF_global,ind_RR2,time_stp_op

#%%
def dyn_linsys_mod(case_setup,KK_global,MM_global,CC_global,RR_global,qq_global,qq_der_global,qq_derder_global,FF_global,ii_time,init_it,time_stp,time_stp_op,ind_RR2,vibfree,sys_prev):
    # Function to solve the linear equation system of the dinamic modal problem
    # KK_global        : stiffness matrix of the problem
    # qq_global        : displacement vector of the problem
    # qq_der_global    : velocity vector of the problem
    # qq_derder_global : acceleration vector of the problem
    # FF_global        : load vector of the problem
    # ii_time          : time index
    # init_it          : initial iterations
    # time_stp         : time step 
    # time_stp_op      : internal time step
    # ind_RR2          : index to delete from the matices
    # vibfree          : free vibration analysis solutions
    # sys_prev         : forces in the previous time step
    # ---------------------------------------------------------------------------------------------------------
    # KK_LR   : submatrix of the stiffness matrix that relates the restricted displacements with the loads
    # KK_RL   : submatrix of the stiffness matrix that relates the free loads with the displacements 
    # KK_RR   : submatrix of the stiffness matrix that relates the free loads with the restricted displacements
    # KK_LL   : submatrix of the stiffness matrix that relates the restricted loads with the free displacements
    # MM_LR   : submatrix of the mass matrix that relates the restricted displacements with the loads
    # MM_RL   : submatrix of the mass matrix that relates the free loads with the displacements 
    # MM_RR   : submatrix of the mass matrix that relates the free loads with the restricted displacements
    # MM_LL   : submatrix of the mass matrix that relates the restricted loads with the free displacements
    # CC_LR   : submatrix of the damping matrix that relates the restricted displacements with the loads
    # CC_RL   : submatrix of the damping matrix that relates the free loads with the displacements 
    # CC_RR   : submatrix of the damping matrix that relates the free loads with the restricted displacements
    # CC_LL   : submatrix of the damping matrix that relates the restricted loads with the free displacements
    # ind_LL  : index of the rows and columns of the KK_LL
    # ind_LL2 : index ind_LL after deleting only zero rows and columns
    # ind_RR : index to delete from KK_LL
    try:
        ind_RR = ind_RR2.copy()
    except:
        ind_RR  = []
    ind_RR2 = []        
    KK_LR   = []
    KK_RL   = []
    KK_RR   = []
    MM_LR   = []
    MM_RL   = []
    MM_RR   = []
    CC_LR   = []
    CC_RL   = []
    CC_RR   = []
    ind_RR  = []
    ind_LL  = []
    ind_LL2 = []
    ind_RR2joint = []
    # Check if the value of the displacement is restricted, 
    #   * if yes: add row to KK_RR or KK_LR
    #   * if no: add row to KK_LL or KK_LL
    for ind_qq in np.arange(len(qq_global)):
        if np.isnan(qq_global[ind_qq,ii_time])==0:
            ind_RR.append(ind_qq)                
        else:
            ind_LL.append(ind_qq)
    # LL_deleterow   : rows to delete in ind_LL
    # ind_RR2        : index to delete from KK_LL after adding the zero rows
    # tol            : tolerance to determine a zero value row
    LL_deleterow = []
    ind_RR2      = ind_RR.copy()
    tol          = 1e-20
    # The zero values are checked for every row
    for ii_row in np.arange(len(KK_global)):
        # Determine if it is a zero row
        if all(MM_global[ii_row,:] == 0) or sum(abs(MM_global[ii_row,:]))<tol:
            # If the row is not deleted yet
            if len(np.where(np.array(ind_RR)==ii_row)[0])==0:
                # Give 0 value to the displacement
                qq_global[ii_row,ii_time] = 0
                qq_der_global[ii_row,ii_time] = 0
                qq_derder_global[ii_row,ii_time] = 0
                if any(RR_global[ii_row,:] != 0):
                    ind_RR2joint.append(ii_row)
                # Add the value of the row to the delete row vector
                ind_RR2.append(ii_row)
                # ind_LL_del  : index to delete from ind_LL vector
                ind_LL_del = np.where(np.array(ind_LL)==ii_row)[0]
                if len(ind_LL_del)>0:
                    # If ind_LL_del is not empty delete the values in ind_LL_del
                    for kk_indLLdel in ind_LL_del:
                        LL_deleterow.append(kk_indLLdel)
    # KK_LL_0  : stiffness matrix after deleting rows
    # KK_LL_1  : stiffness matrix after deleting columns
    # KK_LL    : stiffness matrix after deleting rows and columns
    # MM_LL_0  : mass matrix after deleting rows
    # MM_LL_1  : mass matrix after deleting columns
    # MM_LL    : mass matrix after deleting rows and columns
    # CC_LL_0  : damping matrix after deleting rows
    # CC_LL_1  : damping matrix after deleting columns
    # CC_LL    : damping matrix after deleting rows and columns
    if len(LL_deleterow):
        ind_LL2 = np.delete(np.array(ind_LL),np.array(LL_deleterow),0)
    else:
        ind_LL2 = ind_LL
    KK_LL_0 = np.delete(KK_global,ind_RR2,0)
    KK_LL_1 = np.delete(KK_global,ind_RR2,1)
    KK_LL   = np.delete(KK_LL_0,ind_RR2,1)
    MM_LL_0 = np.delete(MM_global,ind_RR2,0)
    MM_LL_1 = np.delete(MM_global,ind_RR2,1)
    MM_LL   = np.delete(MM_LL_0,ind_RR2,1)
    CC_LL_0 =  np.delete(CC_global,ind_RR2,0)
    CC_LL_1 =  np.delete(CC_global,ind_RR2,1)
    CC_LL   = np.delete(CC_LL_0,ind_RR2,1)
    # The submatrices are obtained
    # KK_RR_0  : submatrix of the stiffness after taking the rows of KK_RR of
    #            the global stiffness matrix
    # MM_RR_0  : submatrix of the mass after taking the rows of KK_RR of
    #            the global stiffness matrix
    # CC_RR_0  : submatrix of the damping after taking the rows of KK_RR of
    #            the global stiffness matrix
    KK_LR   = KK_LL_0[:,ind_RR2]
    KK_RL   = KK_LL_1[ind_RR2,:]
    KK_RR_0 = KK_global[ind_RR2,:]
    KK_RR   = KK_RR_0[:,ind_RR2]
    MM_LR   = MM_LL_0[:,ind_RR2]
    MM_RL   = MM_LL_1[ind_RR2,:]
    MM_RR_0 = MM_global[ind_RR2,:]
    MM_RR   = MM_RR_0[:,ind_RR2]
    CC_LR   = CC_LL_0[:,ind_RR2]
    CC_RL   = CC_LL_1[ind_RR2,:]
    CC_RR_0 = CC_global[ind_RR2,:]
    CC_RR   = CC_RR_0[:,ind_RR2]
    RR_J_vec = RR_global[ind_RR2joint,:]
    RR_SJ   = np.delete(RR_J_vec,ind_RR2joint,1)
    RR_JJ   = RR_J_vec[:,ind_RR2joint]
    # qq_L         : dof of the system
    # qq_R         : restricted displacements of the system
    # qq_der_L     : derivative of the dof of the system
    # qq_der_R     : restricted displacements derivatives of the system
    # qq_derder_R  : restricted displacements second derivatives of the system
    # FF_L         : rows of the load vector corresponding with the 
    #                dertermined values
    # n_modes      : number of modes to use in the calculation
    # yy_L         : vector of the modal coordinates
    # yy_der_L     : derivative of the vector of modal coordinates
    # MM_mode      : mass matrix in modal coordinates
    # KK_mode      : stiffness matrix in modal coordinates
    # CC_mode      : damping matrix in modal coordinates
    # FF_mode      : load vector in modal coordinates
    # x_vec        : total state unknowns vector
    # AA_mat       : total state matix
    # bb_mat       : total state vector
    qq_L        = qq_global[ind_LL2,ii_time-1]
    qq_R        = qq_global[ind_RR2,ii_time-1]
    qq_der_L    = qq_der_global[ind_LL2,ii_time-1]
    qq_der_R    = qq_der_global[ind_RR2,ii_time-1]
    qq_derder_R = qq_derder_global[ind_RR2,ii_time-1]
    FF_L        = FF_global[ind_LL2,ii_time-1]-np.matmul(KK_LR,qq_R)-np.matmul(MM_LR,qq_derder_R)-np.matmul(CC_LR,qq_der_R)
    try:
        n_modes = case_setup.n_mod
    except:
        n_modes = len(qq_L)
    yy_L                      = np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(vibfree.MM_pow2,qq_L))
    yy_der_L                  = np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(vibfree.MM_pow2,qq_der_L))
    x_vec                     = np.zeros((2*n_modes,))
    x_vec[:n_modes]           = yy_L
    x_vec[n_modes:]           = yy_der_L
    MM_mode                   = np.matmul(np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(np.matmul(vibfree.MM_pow,MM_LL),vibfree.MM_pow)),vibfree.mode[:,:n_modes])
    CC_mode                   = np.matmul(np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(np.matmul(vibfree.MM_pow,CC_LL),vibfree.MM_pow)),vibfree.mode[:,:n_modes])
    KK_mode                   = np.matmul(np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(np.matmul(vibfree.MM_pow,KK_LL),vibfree.MM_pow)),vibfree.mode[:,:n_modes])
    FF_mode                   = np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(vibfree.MM_pow,FF_L))    
    AA_mat                    = np.zeros((n_modes*2,n_modes*2))
    AA_mat[:n_modes,n_modes:] = np.identity(n_modes)
    AA_mat[n_modes:,:n_modes] = -np.matmul(np.linalg.inv(MM_mode),KK_mode)
    AA_mat[n_modes:,n_modes:] = -np.matmul(np.linalg.inv(MM_mode),CC_mode)
    bb_mat                    = np.zeros((2*n_modes,))
    bb_mat[n_modes:]          = np.matmul(np.linalg.inv(MM_mode),FF_mode)
    # adatol : tolerance of the adaptative solver
    try:
        adatol = case_setup.adatol
    except:
        adatol = 1e-3
    if ii_time < init_it:
        # For the initialization of the calculation
        # The load vector of the previous internal iterations is used for extrapolate the new internal point
        # sys_prev : class containing the value of the load vectors
        if ii_time > 4:
            sys_prev.bderderderder = ((((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp-sys_prev.bderder)/time_stp-sys_prev.bderderder)/time_stp
        if ii_time > 3:
            sys_prev.bderderder = (((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp-sys_prev.bderder)/time_stp
        if ii_time > 2:
            sys_prev.bderder = ((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp
        if ii_time > 1:
            sys_prev.bder = (bb_mat-sys_prev.b)/time_stp
        if ii_time == 1:
            sys_prev.bder          = 0*bb_mat
            sys_prev.bderder       = 0*bb_mat
            sys_prev.bderderder    = 0*bb_mat
            sys_prev.bderderderder = 0*bb_mat
        sys_prev.b = bb_mat
        if case_setup.solver_init == "BACK4_IMP" or case_setup.solver_init == "AM4_IMP" or case_setup.solver_init == "AM3_PC" or case_setup.solver_init == "HAM3_PC":
            case_setup.solver_init = "ADA_RK4_EXP"
        x_vec,time_stp_op = select_solver(case_setup.solver_init,AA_mat,bb_mat,x_vec,time_stp,[],time_stp_op,adatol,sys_prev.bder,sys_prev.bderder,sys_prev.bderderder,sys_prev.bderderderder)
        x_vec_der = np.matmul(AA_mat,x_vec)+bb_mat
        yy_L = x_vec[:n_modes]
        yy_der_L = x_vec[n_modes:]
        yy_derder_L = x_vec_der[n_modes:] 
        qq_global[ind_LL2,ii_time] = np.matmul(vibfree.MM_pow,np.matmul(vibfree.mode[:,:n_modes],yy_L))
        qq_der_global[ind_LL2,ii_time] = np.matmul(vibfree.MM_pow,np.matmul(vibfree.mode[:,:n_modes],yy_der_L))
        qq_derder_global[ind_LL2,ii_time] = np.matmul(vibfree.MM_pow,np.matmul(vibfree.mode[:,:n_modes],yy_derder_L))
        qq_s = np.delete(qq_global[:,ii_time],ind_RR2joint)
        qq_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qq_s))
        qq_global[ind_RR2joint,ii_time] = qq_j
        qqd_s = np.delete(qq_der_global[:,ii_time],ind_RR2joint)
        qqd_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qqd_s))
        qq_der_global[ind_RR2joint,ii_time] = qqd_j
        qqdd_s = np.delete(qq_derder_global[:,ii_time],ind_RR2joint)
        qqdd_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qqdd_s))
        qq_derder_global[ind_RR2joint,ii_time] = qqdd_j
        f_vec = np.matmul(KK_RR,qq_global[ind_RR2,ii_time-1])+np.matmul(KK_RL,qq_global[ind_LL2,ii_time-1])+np.matmul(MM_RR,qq_derder_global[ind_RR2,ii_time-1])+np.matmul(MM_RL,qq_derder_global[ind_LL2,ii_time-1])+np.matmul(CC_RR,qq_der_global[ind_RR2,ii_time-1])+np.matmul(CC_RL,qq_der_global[ind_LL2,ii_time-1])
        FF_global[ind_RR2,ii_time] = f_vec
    else: 
        # The load vector of the previous internal iterations is used for extrapolate the new internal point
        # sys_prev : class containing the value of the load vectors
        sys_prev.bderderderder = ((((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp-sys_prev.bderder)/time_stp-sys_prev.bderderder)/time_stp
        sys_prev.bderderder    = (((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp-sys_prev.bderder)/time_stp
        sys_prev.bderder       = ((bb_mat-sys_prev.b)/time_stp-sys_prev.bder)/time_stp
        sys_prev.bder          = (bb_mat-sys_prev.b)/time_stp
        sys_prev.b             = bb_mat
        # qq_L_ini     : initial values of the displacements
        # qq_der_L_ini : initial values of the displacement derivatives
        # x_ini        : initial state unknowns vector
        if  case_setup.solver_init == "AM4_IMP" or  case_setup.solver_init == "HAM3_PC":
            qq_L_ini     = qq_global[ind_LL2,ii_time-4:ii_time-1]
            qq_der_L_ini = qq_der_global[ind_LL2,ii_time-4:ii_time-1]
            x_ini        = np.zeros((len(qq_L)+len(qq_der_L),3))
            x_ini[:len(qq_L),:] = qq_L_ini
            x_ini[len(qq_L):,:] = qq_der_L_ini
        elif case_setup.solver_init == "BACK4_IMP" or case_setup.solver_init == "AM3_PC":
            qq_L_ini     = qq_global[ind_LL2,ii_time-3:ii_time-1]
            qq_der_L_ini = qq_der_global[ind_LL2,ii_time-3:ii_time-1]
            x_ini        = np.zeros((len(qq_L)+len(qq_der_L),2))
            x_ini[:len(qq_L),:] = qq_L_ini
            x_ini[len(qq_L):,:] = qq_der_L_ini
        else:
            x_ini = []
        x_vec,time_stp_op = select_solver(case_setup.solver,AA_mat,bb_mat,x_vec,time_stp,x_ini,time_stp_op,adatol,sys_prev.bder,sys_prev.bderder,sys_prev.bderderder,sys_prev.bderderderder)
        x_vec_der = np.matmul(AA_mat,x_vec)+bb_mat
        yy_L = x_vec[:n_modes]
        yy_der_L = x_vec[n_modes:]
        yy_derder_L = x_vec_der[n_modes:]  
        qq_global[ind_LL2,ii_time] = np.matmul(vibfree.MM_pow,np.matmul(vibfree.mode[:,:n_modes],yy_L))
        qq_der_global[ind_LL2,ii_time] = np.matmul(vibfree.MM_pow,np.matmul(vibfree.mode[:,:n_modes],yy_der_L))
        qq_derder_global[ind_LL2,ii_time] = np.matmul(vibfree.MM_pow,np.matmul(vibfree.mode[:,:n_modes],yy_derder_L))
        qq_s = np.delete(qq_global[:,ii_time],ind_RR2joint)
        qq_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qq_s))
        qq_global[ind_RR2joint,ii_time] = qq_j
        qqd_s = np.delete(qq_der_global[:,ii_time],ind_RR2joint)
        qqd_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qqd_s))
        qq_der_global[ind_RR2joint,ii_time] = qqd_j
        qqdd_s = np.delete(qq_derder_global[:,ii_time],ind_RR2joint)
        qqdd_j = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,qqdd_s))
        qq_derder_global[ind_RR2joint,ii_time] = qqdd_j
        f_vec = np.matmul(KK_RR,qq_global[ind_RR2,ii_time-1])+np.matmul(KK_RL,qq_global[ind_LL2,ii_time-1])+np.matmul(MM_RR,qq_derder_global[ind_RR2,ii_time-1])+np.matmul(MM_RL,qq_derder_global[ind_LL2,ii_time-1])+np.matmul(CC_RR,qq_der_global[ind_RR2,ii_time-1])+np.matmul(CC_RL,qq_der_global[ind_LL2,ii_time-1])
        FF_global[ind_RR2,ii_time] = f_vec
    return qq_global,qq_der_global,qq_derder_global,FF_global,ind_RR2,time_stp_op,yy_L,yy_der_L,yy_derder_L,sys_prev
#%%
def vib_linsys(KK_global,MM_global,RR_global,ind_RR,q_RR):
    # Function to solve the linear equation system of the free vibration problem
    # KK_global : stiffness matrix of the problem
    # qq_global : displacement vector of the problem
    # -------------------------------------------------------------------------
    # ind_LL  : index of the rows and columns of the KK_LL
    # ind_RR2 : index to delete from KK_LL after adding the zero rows
    ind_LL  = []
    ind_RR2 = [] 
    ind_RR2joint = []
    # For every row of the stiffness matrix
    for ind_qq in np.arange(len(KK_global)):
        # if the row coincides with the restricted it should be deleted
        cond_ind = ind_RR == ind_qq
        # if there is no coincidence, the row is added to the equation system
        if len(np.where(cond_ind.flatten())[0])==0 :
            ind_LL.append(ind_qq)
    ind_RR2 = ind_RR.copy()
    tol = 1e-20
    # For every row in the  stiffness matrix if all the values are null, delete the row
    for ii_row in np.arange(len(KK_global)):
        if all(MM_global[ii_row,:] == 0) or sum(abs(MM_global[ii_row,:]))<tol:
            # If the row is not deleted yet add its index to the delete list
            if len(np.where(np.array(ind_RR)==ii_row)[0])==0:
                ind_RR2.append(ii_row)
                q_RR.append(0)
            if any(RR_global[ii_row,:] != 0):
                ind_RR2joint.append(ii_row)
    # KK_LL_0  : stiffness matrix after deleting rows
    # KK_LL_1  : stiffness matrix after deleting columns
    # KK_LL    : stiffness matrix after deleting rows and columns
    KK_LL_0 = np.delete(KK_global,ind_RR2,0)
    KK_LL_1 = np.delete(KK_global,ind_RR2,1)
    KK_LL   = np.delete(KK_LL_0,ind_RR2,1)
    MM_LL_0 = np.delete(MM_global,ind_RR2,0)
    MM_LL   = np.delete(MM_LL_0,ind_RR2,1) 
    RR_J_vec = RR_global[ind_RR2joint,:]
    RR_SJ   = np.delete(RR_J_vec,ind_RR2joint,1)
    RR_JJ   = RR_J_vec[:,ind_RR2joint]
#    MM_2    = MM_LL.copy()
#    KK_2    = KK_LL.copy()
#    for ii_mm in np.arange(len(MM_LL)):
#        den_mm = np.max(MM_LL[ii_mm,:])
#        MM_2[ii_mm,:] /= den_mm
#        KK_2[ii_mm,:] /=den_mm
#    KK_LL = KK_2.copy()
#    MM_LL = MM_2.copy()
    # MM_pow    : mass matrix power to 0.5
    # MM_pow2   : mass matrix power to -0.5
    # AA_mat    : eigensystem matrix
    # w_mode    : mode angular frequecies 
    # mode      : mode shape
    # idx       : index to sort modes
    # freq_mode : frequency
    MM_pow_b       = fractional_matrix_power(MM_LL,-0.5)
    MM_pow2_b      = fractional_matrix_power(MM_LL,0.5)
#    MM_pow = np.zeros((len(MM_pow_b[0,:]),len(MM_pow_b[0,:])))
#    MM_pow2 = np.zeros((len(MM_pow_b[0,:]),len(MM_pow_b[0,:])))
#    for ii_x in np.arange(len(MM_pow_b[0,:])): 
#        for jj_x in np.arange(len(MM_pow_b[0,:])):
#            MM_pow[jj_x,ii_x] = np.real(MM_pow_b[jj_x,ii_x])
#            MM_pow2[jj_x,ii_x] = np.real(MM_pow2_b[jj_x,ii_x])   
    AA_mat       = -np.matmul(np.matmul(MM_pow_b,KK_LL),MM_pow_b) 
    w_mode, mode = np.linalg.eig(AA_mat)
    idx          = np.argsort(abs(w_mode))
    w_mode2 = np.zeros((len(w_mode)))
    for ii_x in np.arange(len(idx)): 
        w_mode2[ii_x] = abs(np.imag(cmath.sqrt(w_mode[ii_x]))) #abs(cmath.sqrt(w_mode[ii_x]))
    w_mode = w_mode2.copy()
    freq_mode = w_mode/(2*np.pi)
    mode_glob = np.matmul(MM_pow_b,mode)
    mode_tot = np.zeros((len(KK_global[:,0]),len(mode[0,:])))
    ind_LL2 = np.delete(np.arange(len(KK_global[0,:])),ind_RR2,0)
    for ii_modetot in np.arange(len(mode[:,0])):
        for jj_modetot in np.arange(len(mode[0,:])):
            mode_tot[ind_LL2[ii_modetot],jj_modetot] = np.real(mode_glob[ii_modetot,jj_modetot])
    for ii_modetot in np.arange(len(mode[0,:])):
        mode_s = np.delete(mode_tot[:,ii_modetot],ind_RR2joint)
        mode_tot[ind_RR2joint,ii_modetot] = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,mode_s))
    return freq_mode, w_mode, mode, MM_pow_b, MM_pow2_b, idx, MM_LL, KK_LL, ind_RR2, mode_tot

#%%
def vibae_linsys(KK_global,CC_global,MM_global, RR_global,ind_RR,q_RR,num_point):
    # Function to solve the linear equation system of the free vibration problem
    # KK_global : stiffness matrix of the problem
    # qq_global : displacement vector of the problem
    # -------------------------------------------------------------------------
    # ind_LL  : index of the rows and columns of the KK_LL
    # ind_RR2 : index to delete from KK_LL after adding the zero rows
    ind_LL  = []
    ind_RR2 = []
    ind_RR2joint = []
    # For every row of the stiffness matrix
    for ind_qq in np.arange(len(KK_global)):
        # if the row coincides with the restricted it should be deleted
        cond_ind = ind_RR == ind_qq
        # if there is no coincidence, the row is added to the equation system
        if len(np.where(cond_ind.flatten())[0])==0:
            ind_LL.append(ind_qq)
    ind_RR2 = ind_RR.copy()
    tol = 1e-20
    # For every row in the  stiffness matrix if all the values are null, delete the row
    for ii_row in np.arange(len(KK_global)):
        if all(MM_global[ii_row,:] == 0) or sum(abs(MM_global[ii_row,:]))<tol:
            # If the row is not deleted yet add its index to the delete list
            if len(np.where(np.array(ind_RR)==ii_row)[0])==0:
                ind_RR2.append(ii_row)
                q_RR.append(0)
            if any(RR_global[ii_row,:] != 0):
                ind_RR2joint.append(ii_row)
    # KK_LL_0  : stiffness matrix after deleting rows
    # KK_LL_1  : stiffness matrix after deleting columns
    # KK_LL    : stiffness matrix after deleting rows and columns
    KK_LL_0 = np.delete(KK_global,ind_RR2,0)
    KK_LL_1 = np.delete(KK_global,ind_RR2,1)
    KK_LL   = np.delete(KK_LL_0,ind_RR2,1)
    CC_LL_0 = np.delete(CC_global,ind_RR2,0)
    CC_LL_1 = np.delete(CC_global,ind_RR2,1)
    CC_LL   = np.delete(CC_LL_0,ind_RR2,1)
    MM_LL_0 = np.delete(MM_global,ind_RR2,0)
    MM_LL   = np.delete(MM_LL_0,ind_RR2,1) 
    RR_J_vec = RR_global[ind_RR2joint,:]
    RR_SJ   = np.delete(RR_J_vec,ind_RR2joint,1)
    RR_JJ   = RR_J_vec[:,ind_RR2joint]
    l_point = len(KK_LL) 
    # MM_pow    : mass matrix power to 0.5
    # MM_pow2   : mass matrix power to -0.5
    # AA_mat    : eigensystem matrix
    # w_mode    : mode angular frequecies 
    # mode      : mode shape
    # idx       : index to sort modes
    # freq_mode : frequency
    MM_pow       = fractional_matrix_power(MM_LL,-0.5)
    MM_pow2      = fractional_matrix_power(MM_LL,0.5)     
    MM_mat =  np.zeros((2*l_point,2*l_point), dtype=complex)
    MM_mat[:l_point,:l_point] = np.identity(l_point) # MM_LL #
    MM_mat[l_point:,l_point:] = np.matmul(np.matmul(MM_pow,MM_LL),MM_pow) #MM_LL
    KK_mat =  np.zeros((2*l_point,2*l_point), dtype=complex)
    KK_mat[:l_point,l_point:] = -np.identity(l_point) # MM_LL #
    KK_mat[l_point:,:l_point] = np.matmul(np.matmul(MM_pow,KK_LL),MM_pow) #KK_LL
    KK_mat[l_point:,l_point:] = np.matmul(np.matmul(MM_pow,CC_LL),MM_pow) #CC_LL *0 
    AA_mat       = KK_mat # np.matmul(np.matmul(MM_pow,KK_mat),MM_pow) 
    l_mode, mode = np.linalg.eig(-AA_mat)

    w_mode       = np.imag(l_mode)
    idx          = np.argsort(abs(l_mode))
    freq_mode    = w_mode/(2*np.pi)
    MM_powtot    = np.zeros((2*l_point,2*l_point), dtype=complex)
    MM_powtot[:l_point,:l_point] = MM_pow
    MM_powtot[l_point:,l_point:] = MM_pow
    mode_glob = np.matmul(MM_powtot,mode)
    mode_tot = np.zeros((9*num_point,len(mode[0,:])))
    ind_LL2 = np.delete(np.arange(num_point*9),ind_RR2,0)
    for jj_modetot in np.arange(len(mode[0,:])):
        for ii_modetot in np.arange(int(len(mode[:,0])/2)):
            mode_tot[ind_LL2[ii_modetot],jj_modetot] = np.real(mode_glob[ii_modetot+int(len(mode[:,0])/2),jj_modetot])
        max_modetot = np.max(abs(mode_tot[:,jj_modetot]))
        mode_tot[:,jj_modetot] /= max_modetot
    for ii_modetot in np.arange(len(mode[0,:])):
        mode_s = np.delete(mode_tot[:,ii_modetot],ind_RR2joint)
        mode_tot[ind_RR2joint,ii_modetot] = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,mode_s))
    return l_mode, freq_mode, w_mode, mode, MM_pow, MM_pow2, idx, MM_LL, KK_LL, ind_RR2, mode_tot




def vibaemod_linsys(KK_global,CC_global,MM_global, RR_global,ind_RR,q_RR,num_point,vibfree,case_setup):
    # Function to solve the linear equation system of the free vibration problem
    # KK_global : stiffness matrix of the problem
    # qq_global : displacement vector of the problem
    # -------------------------------------------------------------------------
    # ind_LL  : index of the rows and columns of the KK_LL
    # ind_RR2 : index to delete from KK_LL after adding the zero rows
    ind_LL  = []
    ind_RR2 = []
    ind_RR2joint = []
    # For every row of the stiffness matrix
    for ind_qq in np.arange(len(KK_global)):
        # if the row coincides with the restricted it should be deleted
        cond_ind = ind_RR == ind_qq
        # if there is no coincidence, the row is added to the equation system
        if len(np.where(cond_ind.flatten())[0])==0:
            ind_LL.append(ind_qq)
    ind_RR2 = ind_RR.copy()
    tol = 1e-20
    # For every row in the  stiffness matrix if all the values are null, delete the row
    for ii_row in np.arange(len(KK_global)):
        if all(MM_global[ii_row,:] == 0) or sum(abs(MM_global[ii_row,:]))<tol:
            # If the row is not deleted yet add its index to the delete list
            if len(np.where(np.array(ind_RR)==ii_row)[0])==0:
                ind_RR2.append(ii_row)
                q_RR.append(0)
            if any(RR_global[ii_row,:] != 0):
                ind_RR2joint.append(ii_row)
    # KK_LL_0  : stiffness matrix after deleting rows
    # KK_LL_1  : stiffness matrix after deleting columns
    # KK_LL    : stiffness matrix after deleting rows and columns
    KK_LL_0 = np.delete(KK_global,ind_RR2,0)
    KK_LL_1 = np.delete(KK_global,ind_RR2,1)
    KK_LL   = np.delete(KK_LL_0,ind_RR2,1)
    CC_LL_0 = np.delete(CC_global,ind_RR2,0)
    CC_LL_1 = np.delete(CC_global,ind_RR2,1)
    CC_LL   = np.delete(CC_LL_0,ind_RR2,1)
    MM_LL_0 = np.delete(MM_global,ind_RR2,0)
    MM_LL   = np.delete(MM_LL_0,ind_RR2,1) 
    RR_J_vec = RR_global[ind_RR2joint,:]
    RR_SJ   = np.delete(RR_J_vec,ind_RR2joint,1)
    RR_JJ   = RR_J_vec[:,ind_RR2joint]
    l_point = len(KK_LL) 
    # MM_pow    : mass matrix power to 0.5
    # MM_pow2   : mass matrix power to -0.5
    # AA_mat    : eigensystem matrix
    # w_mode    : mode angular frequecies 
    # mode      : mode shape
    # idx       : index to sort modes
    # freq_mode : frequency
    try:
        n_modes = case_setup.n_mod
    except:
        n_modes = len(KK_LL)
    MM_pow       = fractional_matrix_power(MM_LL,-0.5)
    MM_pow2      = fractional_matrix_power(MM_LL,0.5)
    
    MM_mode                   = np.matmul(np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(np.matmul(MM_pow,MM_LL),MM_pow)),vibfree.mode[:,:n_modes])
    CC_mode                   = np.matmul(np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(np.matmul(MM_pow,CC_LL),MM_pow)),vibfree.mode[:,:n_modes])
    KK_mode                   = np.matmul(np.matmul(np.transpose(vibfree.mode[:,:n_modes]),np.matmul(np.matmul(MM_pow,KK_LL),MM_pow)),vibfree.mode[:,:n_modes])
    
    AA_mat                    = np.zeros((n_modes*2,n_modes*2))
    AA_mat[:n_modes,n_modes:] = np.identity(n_modes)
    AA_mat[n_modes:,:n_modes] = -np.matmul(np.linalg.inv(MM_mode),KK_mode)
    AA_mat[n_modes:,n_modes:] = -np.matmul(np.linalg.inv(MM_mode),CC_mode)
    
    
       
    MM_mat =  np.zeros((2*l_point,2*l_point), dtype=complex)
    MM_mat[:l_point,:l_point] = -np.identity(l_point) # MM_LL #
    MM_mat[l_point:,l_point:] = -np.matmul(np.matmul(MM_pow,MM_LL),MM_pow) #MM_LL
    KK_mat =  np.zeros((2*l_point,2*l_point), dtype=complex)
    KK_mat[:l_point,l_point:] = np.identity(l_point) # MM_LL #
    KK_mat[l_point:,:l_point] = -np.matmul(np.matmul(MM_pow,KK_LL),MM_pow) #KK_LL
    KK_mat[l_point:,l_point:] = -np.matmul(np.matmul(MM_pow,CC_LL),MM_pow) #CC_LL *0 
    AA_mat       = KK_mat # np.matmul(np.matmul(MM_pow,KK_mat),MM_pow) 
    l_mode, mode = np.linalg.eig(AA_mat)

    w_mode       = np.imag(l_mode)
    idx          = np.argsort(abs(l_mode))
    freq_mode    = w_mode/(2*np.pi)
    MM_powtot    = np.zeros((2*l_point,2*l_point), dtype=complex)
    MM_powtot[:l_point,:l_point] = MM_pow
    MM_powtot[l_point:,l_point:] = MM_pow
    mode_glob = np.matmul(MM_powtot,mode)
    mode_tot = np.zeros((9*num_point,len(mode[0,:])))
    ind_LL2 = np.delete(np.arange(num_point*9),ind_RR2,0)
    for jj_modetot in np.arange(len(mode[0,:])):
        for ii_modetot in np.arange(int(len(mode[:,0])/2)):
            mode_tot[ind_LL2[ii_modetot],jj_modetot] = np.real(mode_glob[ii_modetot+int(len(mode[:,0])/2),jj_modetot])
        max_modetot = np.max(abs(mode_tot[:,jj_modetot]))
        mode_tot[:,jj_modetot] /= max_modetot
    for ii_modetot in np.arange(len(mode[0,:])):
        mode_s = np.delete(mode_tot[:,ii_modetot],ind_RR2joint)
        mode_tot[ind_RR2joint,ii_modetot] = np.linalg.solve(RR_JJ,-np.matmul(RR_SJ,mode_s))
    return l_mode, freq_mode, w_mode, mode, MM_pow, MM_pow2, idx, MM_LL, KK_LL, ind_RR2, mode_tot