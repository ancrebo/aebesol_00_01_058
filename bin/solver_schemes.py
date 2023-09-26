# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 12:10:42 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
solver_schemes : file containing functions for the time integration
last_version   : 23-02-2021
modified_by    : Andres Cremades Botella
"""
        
import numpy as np

#%% Functions
def func_dyn(Amat,bmat,xvec):
    # Solution of the linear equation
    # Amat : system coefficients matrix
    # bmat : independent terms vector
    # xvec : solution in the last time step
    return np.matmul(Amat,xvec)+bmat

#%%
def rk4_exp(Amat,bmat,xvec,delta_t):
    # Runge-Kutta explicit order 4
    # Amat    : system coefficients matrix
    # bmat    : independent terms vector
    # xvec    : solution in the last time step
    # delta_t : timestep
    # -------------------------------------------------------------------------
    # rk1    = estimation of the first RK solution
    # rk2    = estimation of the second RK solution
    # rk3    = estimation of the third RK solution
    # rk4    = estimation of the fourth RK solution
    # slope  = derivative of the solution vector in the time step
    # xvec_1 = solution in the following step
    rk1    = func_dyn(Amat,bmat,xvec)
    rk2    = func_dyn(Amat,bmat,xvec+0.5*rk1*delta_t)
    rk3    = func_dyn(Amat,bmat,xvec+0.5*rk2*delta_t)
    rk4    = func_dyn(Amat,bmat,xvec+rk3*delta_t)
    slope  = 1/6*(rk1+2*rk2+2*rk3+rk4)
    xvec_1 = xvec+delta_t*slope
    return xvec_1

#%%
def rk4_pc(Amat,bmat,xvec,delta_t): 
    # Predictor corrector Runge-Kutta order 4
    # Amat    : system coefficients matrix
    # bmat    : independent terms vector
    # xvec    : solution in the last time step
    # delta_t : timestep
    # -------------------------------------------------------------------------
    # xvec_p1 = solution first prediction of the RK 
    # xvec_c1 = first correction of the RK
    # xvec_p2 = second prediction of the RK
    # xvec_c2 = solution in the next time step (second correction of RK)
    xvec_p1 = xvec+delta_t/2*func_dyn(Amat,bmat,xvec)
    xvec_c1 = xvec+delta_t/2*func_dyn(Amat,bmat,xvec_p1)
    xvec_p2 = xvec+delta_t*func_dyn(Amat,bmat,xvec_c1)
    xvec_c2 = xvec+delta_t/6*(func_dyn(Amat,bmat,xvec)+2*func_dyn(Amat,bmat,xvec_p1)+2*func_dyn(Amat,bmat,xvec_c1)+func_dyn(Amat,bmat,xvec_p2))
    return xvec_c2

#%%
def ad_rk4_pc(Amat,bmat,xvec,delta_t,delta_top,tol):
    # Adaptative solver based on Predictor-Corrector Runge-Kutta
    # Amat      : system coefficients matrix
    # bmat      : independent terms vector
    # xvec      : solution in the last time step
    # delta_t   : timestep
    # delta_top : double value of the stability time step in the previous step
    # tol       : tolerance of the solver
    # -------------------------------------------------------------------------
    
    # The maximum time step is limited to the required value
    if delta_top > delta_t:
        delta_top = delta_t
        
    #  Define constants 
    # num_timevec : number of timesteps to calculate accurately the total step
    # vec_time    : vector of the time value inside the time step
    # error       : initialization of the error in the adaptative time step
    # deltat_min  : minimum value of the delta_t
    # kkdiv       : `number of divisions of he time step
    # kkdiv       : initial number of divisions of the time step
    num_timevec = delta_t/delta_top
    vec_time    = np.linspace(0, delta_t,num=int(round(num_timevec)+1))
    error       = 1
    deltat_min  = delta_t/1e6
    kkdiv       = round(np.log2(num_timevec))
    kkdiv_ini   = kkdiv
    
    # Error reduction loop
    while error>tol:
        # define the value of the initial solution of the complete timestep
        xvec_sol = [xvec]
        # Calculate for all the divisions of the time step
        for ii_time in np.arange(len(vec_time)-1):
            # delta_t_2 : internal time step of the divisions
            delta_t_2 = vec_time[ii_time+1]-vec_time[ii_time]
            # if the internal time step is lower than the minum allowed, exit the loop
            if delta_t_2< deltat_min:
                break
            xvec_sol.append(rk4_pc(Amat,bmat,xvec_sol[-1],delta_t_2))
        # if it is the initial division, the number of divisions is the length of the time vec
        # if not calculate  the error respect to the previous number of time internal divisions
        if kkdiv == kkdiv_ini:
            kkdiv2 = len(vec_time)
        else:
            # error     : error between the normalized value in the actual iteration and the previous
            # xvec_sol0 : value in the previous iteration
            # vec_time2 : time vec in the next iteration (initialization)
            error = np.divide(np.linalg.norm(xvec_sol[-1]-xvec_sol0),np.linalg.norm(xvec_sol0))
            if np.linalg.norm(xvec_sol[-1]-xvec_sol0) == 0:
                error = tol/10
            elif  np.isnan(error):
                error = 1
            kkdiv2 = len(vec_time2)
        xvec_sol0 = xvec_sol[-1]
        vec_time2 = np.zeros((int(kkdiv2+2**kkdiv),))
        kkdiv += 1
        # calculate time vector in the next iteration
        # divide time step by 2
        for ii_time in np.arange(len(vec_time)-1):
            vec_time2[2*ii_time] = vec_time[ii_time]
            vec_time2[2*ii_time+1] = (vec_time[ii_time]+vec_time[ii_time+1])/2
        vec_time2[-1] = vec_time[-1]
        vec_time = vec_time2
    # Allow the code to increase the time step by starting in the previous time step than the chosen one
    delta_top = delta_t_2*4
    return xvec_sol[-1], delta_top

#%%
def ad_rk4_exp(Amat,bmat,xvec,delta_t,delta_top,tol):
    # Adaptative solver based on Predictor-Corrector Runge-Kutta
    # Amat      : system coefficients matrix
    # bmat      : independent terms vector
    # xvec      : solution in the last time step
    # delta_t   : timestep
    # delta_top : double value of the stability time step in the previous step
    # tol       : tolerance of the solver
    # -------------------------------------------------------------------------
    
    # The maximum time step is limited to the required value
    if delta_top > delta_t:
        delta_top = delta_t
        
    #  Define constants 
    # num_timevec : number of timesteps to calculate accurately the total step
    # vec_time    : vector of the time value inside the time step
    # error       : initialization of the error in the adaptative time step
    # deltat_min  : minimum value of the delta_t
    # kkdiv       : `number of divisions of he time step
    # kkdiv       : initial number of divisions of the time step
    num_timevec = delta_t/delta_top
    vec_time    = np.linspace(0, delta_t,num=int(round(num_timevec)+1))
    error       = 1
    deltat_min  = delta_t/1e6
    kkdiv       = round(np.log2(num_timevec))
    kkdiv_ini   = kkdiv
    
    # Error reduction loop
    while error>tol:
        # define the value of the initial solution of the complete timestep
        xvec_sol = [xvec]
        # Calculate for all the divisions of the time step
        for ii_time in np.arange(len(vec_time)-1):
            # delta_t_2 : internal time step of the divisions
            delta_t_2 = vec_time[ii_time+1]-vec_time[ii_time]
            # if the internal time step is lower than the minum allowed, exit the loop
            if delta_t_2< deltat_min:
                break
            xvec_sol.append(rk4_exp(Amat,bmat,xvec_sol[-1],delta_t_2))
        # if it is the initial division, the number of divisions is the length of the time vec
        # if not calculate  the error respect to the previous number of time internal divisions
        if kkdiv == kkdiv_ini:
            kkdiv2 = len(vec_time)
        else:
            # error     : error between the normalized value in the actual iteration and the previous
            # xvec_sol0 : value in the previous iteration
            # vec_time2 : time vec in the next iteration (initialization)
            error = np.divide(np.linalg.norm(xvec_sol[-1]-xvec_sol0),np.linalg.norm(xvec_sol0))
            if np.linalg.norm(xvec_sol[-1]-xvec_sol0) == 0:
                error = tol/10
            elif np.isnan(error):
                error = 1
            kkdiv2 = len(vec_time2)
        xvec_sol0 = xvec_sol[-1]
        vec_time2 = np.zeros((int(kkdiv2+2**kkdiv),))
        kkdiv += 1
        # calculate time vector in the next iteration
        # divide time step by 2
        for ii_time in np.arange(len(vec_time)-1):
            vec_time2[2*ii_time] = vec_time[ii_time]
            vec_time2[2*ii_time+1] = (vec_time[ii_time]+vec_time[ii_time+1])/2
        vec_time2[-1] = vec_time[-1]
        vec_time = vec_time2
    # Allow the code to increase the time step by starting in the previous time step than the chosen one
    delta_top = delta_t_2*4
    return xvec_sol[-1], delta_top

#%%
def adex_rk4_exp(Amat,bmat,xvec,delta_t,delta_top,tol,bmatder,bmatderder,bmatderderder,bmatderderderder):
    # Adaptative solver based on Predictor-Corrector Runge-Kutta
    # Amat       : system coefficients matrix
    # bmat       : independent terms vector
    # xvec       : solution in the last time step
    # delta_t    : timestep
    # delta_top  : double value of the stability time step in the previous step
    # tol        : tolerance of the solver
    # bmatder    : independent terms vector
    # bmatderder : independent terms vector
    # -------------------------------------------------------------------------
    
    # The maximum time step is limited to the required value
    if delta_top > delta_t:
        delta_top = delta_t
        
    #  Define constants 
    # num_timevec : number of timesteps to calculate accurately the total step
    # vec_time    : vector of the time value inside the time step
    # error       : initialization of the error in the adaptative time step
    # deltat_min  : minimum value of the delta_t
    # kkdiv       : `number of divisions of he time step
    # kkdiv       : initial number of divisions of the time step
    num_timevec = delta_t/delta_top
    vec_time    = np.linspace(0, delta_t,num=int(round(num_timevec)+1))
    error       = 1
    deltat_min  = delta_t/1e6
    kkdiv       = round(np.log2(num_timevec))
    kkdiv_ini   = kkdiv
    bmat0i          = bmat
    bmatder0i       = bmatder
    bmatderder0i    = bmatderder
    bmatderderder0i = bmatderderder
    
    # Error reduction loop
    while error>tol:
        bmat          = bmat0i
        bmatder       = bmatder0i
        bmatderder    = bmatderder0i
        bmatderderder = bmatderderder0i
        # define the value of the initial solution of the complete timestep
        xvec_sol = [xvec]
        # Calculate for all the divisions of the time step
        for ii_time in np.arange(len(vec_time)-1):
            # delta_t_2 : internal time step of the divisions
            delta_t_2 = vec_time[ii_time+1]-vec_time[ii_time]
            # if the internal time step is lower than the minum allowed, exit the loop
            if delta_t_2< deltat_min:
                break
            bmat0          = bmat
            bmatder0       = bmatder
            bmatderder0    = bmatderder
            bmatderderder0 = bmatderderder
            xvec_sol.append(rk4_exp(Amat,bmat,xvec_sol[-1],delta_t_2))
            bmat = bmat+bmatder*delta_t_2+bmatderder/2*delta_t_2**2+bmatderderder/6*delta_t_2**3+bmatderderderder/24*delta_t_2**4
            bmatder = (bmat-bmat0)/delta_t_2
            bmatderder = (bmatder-bmatder0)/delta_t_2
            bmatderderder = (bmatderder-bmatderder0)/delta_t_2
            bmatderderderder = (bmatderderder-bmatderderder0)/delta_t_2
        # if it is the initial division, the number of divisions is the length of the time vec
        # if not calculate  the error respect to the previous number of time internal divisions
        if kkdiv == kkdiv_ini:
            kkdiv2 = len(vec_time)
        else:
            # error     : error between the normalized value in the actual iteration and the previous
            # xvec_sol0 : value in the previous iteration
            # vec_time2 : time vec in the next iteration (initialization)
            error = np.divide(np.linalg.norm(xvec_sol[-1]-xvec_sol0),np.linalg.norm(xvec_sol0))
            if np.linalg.norm(xvec_sol[-1]-xvec_sol0) == 0:
                error = tol/10
            elif np.isnan(error):
                error = 1
            kkdiv2 = len(vec_time2)
        xvec_sol0 = xvec_sol[-1]
        vec_time2 = np.zeros((int(kkdiv2+2**kkdiv),))
        kkdiv += 1
        # calculate time vector in the next iteration
        # divide time step by 2
        if error>tol:
            for ii_time in np.arange(len(vec_time)-1):
                vec_time2[2*ii_time] = vec_time[ii_time]
                vec_time2[2*ii_time+1] = (vec_time[ii_time]+vec_time[ii_time+1])/2
            vec_time2[-1] = vec_time[-1]
            vec_time = vec_time2
    # Allow the code to increase the time step by starting in the previous time step than the chosen one
    delta_top = delta_t_2*4
    return xvec_sol[-1], delta_top
 
#%%   
def back4_imp(Amat,bmat,xvec,xini,delta_t):
    # Backward implicit solver
    # Amat      : system coefficients matrix
    # bmat      : independent terms vector
    # xvec      : solution in the last time step
    # delta_t   : timestep
    # error     : error of the iteration
    # xveccorr  : value of the new iteration
    # xsol      : previous iteration value
    xsol = xvec
    error = 1
    while error > 1e-10:
        xveccorr = delta_t*6/11*func_dyn(Amat,bmat,xsol)+18/11*xvec-9/11*xini[:,1]+2/11*xini[:,0]
        error = np.linalg.norm(np.divide(xsol-xveccorr,xveccorr))
        xsol = xveccorr
    return xveccorr

#%%
def AM4_imp(Amat,bmat,xvec,xini,delta_t):
    # 4 order adams moulton implicit solver
    # Amat      : system coefficients matrix
    # bmat      : independent terms vector
    # xvec      : solution in the last time step
    # xini      : values of the previous time steps
    # delta_t   : timestep
    # func_vecpred : result on the previous iteration
    # xveccorr     : result in current iteration
    # error        : error of the iteration respect the previous one
    func_vecpred = func_dyn(Amat,bmat,xvec)
    error = 1
    while error > 1e-10:
        xveccorr = xvec[:]+delta_t/720*(251*func_vecpred+646*func_dyn(Amat,bmat,xvec[:])-264*func_dyn(Amat,bmat,xini[:,2])+106*func_dyn(Amat,bmat,xini[:,1])-19*func_dyn(Amat,bmat,xini[:,0]))
        error = np.linalg.norm(np.divide(func_vecpred-xveccorr,xveccorr))
        func_vecpred = xveccorr
    return xveccorr

#%%
def AM3_pc(Amat,bmat,xvec,xini,delta_t):
    # Adams-Moulton predictor corrector
    # Amat      : system coefficients matrix
    # bmat      : independent terms vector
    # xvec      : solution in the last time step
    # delta_t   : timestep
    # xini      : values of the previous time steps
    # delta_t   : timestep
    # xvecpred  : predictor value
    # xveccorr  : corrector value
    xvecpred = xvec[:]+delta_t/12*(23*func_dyn(Amat,bmat,xvec[:])-16*func_dyn(Amat,bmat,xini[:,1])+5*func_dyn(Amat,bmat,xini[:,0]))
    func_vecpred = func_dyn(Amat,bmat,xvecpred)
    xveccorr = xvec[:]+delta_t/24*(9*func_vecpred+19*func_dyn(Amat,bmat,xvec[:])-5*func_dyn(Amat,bmat,xini[:,1])+1*func_dyn(Amat,bmat,xini[:,0]))
    return xveccorr

#%%
def Ham3_pc(Amat,bmat,xvec,xini,delta_t):
    # Adams-Moulton predictor corrector
    # Amat      : system coefficients matrix
    # bmat      : independent terms vector
    # xvec      : solution in the last time step
    # delta_t   : timestep
    # xvecpred  : predictor value
    # xveccorr  : corrector value
    xvecpred = xini[:,0]+4*delta_t/3*(2*func_dyn(Amat,bmat,xvec)-func_dyn(Amat,bmat,xini[:,2])+func_dyn(Amat,bmat,xini[:,1]))
    xveccorr = 9/8*xvec-1/8*xini[:,1]+3*delta_t/8*(func_dyn(Amat,bmat,xvecpred)+2*func_dyn(Amat,bmat,xvec)+func_dyn(Amat,bmat,xini[:,2]))
    return xveccorr

#%%
def select_solver(solver,Amat,bmat,xvec,delta_t,xini,delta_top,adatol,bmatder,bmatderder,bmatderderder,bderderderder):
    # Function for selecting the solver scheme
    # solver           : solver scheme to use
    # Amat             : system matrix
    # bmat             : system vector
    # xvec             : solution vector in the last state
    # delta_t          : time step
    # xini             : solution vector for previous time steps
    # delta_top        : internal time step in adaptive solvers
    # adatol           : adaptive solver tolerance
    # bmatder          : derivative of the system vector
    # bmatderder       : second derivative of the system vector
    # bmatderderder    : third derivative of the system vector
    # bmatderderderder : fourth derivative of the system vector
    # ------------------------------------------------------------------------
    # time_step_op : initial internal time step 
    time_step_op = delta_t
    # If the 4th order explicit Runge-kutta is selected 
    if solver == "RK4_EXP":
        # update the solution vector
        x_vec = rk4_exp(Amat,bmat,xvec,delta_t)
    # If the adaptive 4th order explicit Runge-kutta is selected
    elif solver == "ADA_RK4_EXP":
        # update the solution vector and initial internal time step
        x_vec,time_step_op = ad_rk4_exp(Amat,bmat,xvec,delta_t,delta_top,adatol)
    # If the extrapolated forces adaptive 4th order explicit Runge-kutta is selected
    elif solver == "ADAEX_RK4_EXP":
        # update the solution vector and initial internal time step
        x_vec,time_step_op = adex_rk4_exp(Amat,bmat,xvec,delta_t,delta_top,adatol,bmatder,bmatderder,bmatderderder,bderderderder)
    # If the 4th order predictor-corrector Runge-kutta is selected
    elif solver == "RK4_PC":
        # update the solution vector
        x_vec = rk4_pc(Amat,bmat,xvec,delta_t)
    # If the adaptive 4th order predictor-corrector Runge-kutta is selected 
    elif solver == "ADA_RK4_PC":
        # update the solution vector and initial internal time step
        x_vec,time_step_op = ad_rk4_pc(Amat,bmat,xvec,delta_t,delta_top,adatol)
    # If the implicit 4 steps backward is selected 
    elif solver == "BACK4_IMP":
        # update the solution vector
        x_vec = back4_imp(Amat,bmat,xvec,xini,delta_t)
    # If the implicit 4 steps adams-moulton is selected 
    elif solver == "AM4_IMP":
        # update the solution vector
        x_vec = AM4_imp(Amat,bmat,xvec,xini,delta_t)
    # If the predictor-corrector 3 steps adams-moulton is selected
    elif solver == "AM3_PC":
        # update the solution vector
        x_vec = AM3_pc(Amat,bmat,xvec,xini,delta_t)
    # If the implicit 3 steps hamming is selected
    elif solver == "HAM3_PC":
        # update the solution vector
        x_vec = Ham3_pc(Amat,bmat,xvec,xini,delta_t)
    return x_vec,time_step_op