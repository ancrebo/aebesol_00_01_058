# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 11:10:26 2021

@author       : Andres Cremades Botella - ancrebo@mot.upv.es
aux_functions : file containing the auxiliary functions for the calculation of aeroelastic beams
last_version  : 17-02-2021
modified_by   : Andres Cremades Botella
"""
import numpy as np
from scipy import signal

#%% Functions
def def_vec_param(num_point): 
    # function for creating empty list containing empty list
    # num_point : number of sub lists contained in parent list
    # vec_out   : solution list
    list_vec = []
    vec_out  = []
    # Create the first level of list
    vec_out  = [vec_out.append(list_vec) for ii_secn in np.arange(num_point)]
    # Create the second level inside of the first level
    for ii_secn in np.arange(num_point): vec_out[ii_secn] = []
    return vec_out

#%%
def filter_func(func,func2,func_val,ii_filter,Nvec1,Nvec2,Nvec3,time_stp,num_point,taps,time_val,tmin_ann,flag_filter):
    # function for filtering the signal using a FIR filter
    # func        : fuction to filter
    # func2       : filtered function
    # func_val    : value of the function in a certain time step
    # ii_filter   : index in the filtered signal
    # Nvec1       : index to start the application of the filter
    # Nvec2       : index to extrapolate the solution
    # Nvec3       : length of the filtered signal
    # time_stp    : time step
    # num_point   : number of points of the beam mesh
    # taps        : filter coefficients
    # time_val    : simulation time
    # tmin_ann    : minimum time to apply the artificial neural network
    # flag_filter : if the filtered signal is required to be updated
    # -------------------------------------------------------------------------
    # If the signal must be updated
    if flag_filter == 1:
        # Restart the filtered signal index
        # update the values of func and func2
        ii_filter       = Nvec1
        func[:Nvec1,:]  = func[Nvec1:Nvec2,:]
        func[Nvec1:,:]  = np.zeros((Nvec2,num_point))
        func2[:Nvec1,:] = func2[Nvec1:Nvec2,:]
        func2[Nvec1:,:] = np.zeros((Nvec2,num_point))
    # Add the simulated value of the signal
    func[ii_filter,:] = func_val
    # If the filter is activated
    if ii_filter >= Nvec1 and time_val>=tmin_ann:
        # Create a mean value signal to obtain an approximation of the value in the current timestep without phase shift
        for ii_filter2 in np.arange(Nvec3-ii_filter-1)+ii_filter+1: 
            # func_mean : mean value of the function
            func_mean = np.mean(func[int(ii_filter2-Nvec1/10):ii_filter2])
            func[ii_filter2,:] = func_mean 
        for jj_filter in np.arange(num_point):
            # filter the signal
            func2[ii_filter,jj_filter] = signal.lfilter(taps, 1.0,func[:ii_filter+Nvec1+1,jj_filter] )[-1]
    else:
        # If the conditions of the filter are not satified do not apply it
        func2[ii_filter,:] = func_val
    # func_val2 : value of the function in the current time step
    func_val2 = func2[ii_filter,:]
    return func, func2, func_val2 