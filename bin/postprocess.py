# -*- coding: utf-8 -*-
"""
Created on Thu May 28 11:31:22 2020

@author       : Andres Cremades Botella - ancrebo@mot.upv.es
postprocess   : file containing the postprocessing functions
last_version  : 24-02-2021
modified_by   : Andres Cremades Botella
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits import mplot3d
from matplotlib import animation
from matplotlib import rcParams
from matplotlib import cm
import matplotlib.colors as colors
import matplotlib.tri as mtri
from bin.aux_functions import def_vec_param 
import matplotlib.collections
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from random import random
import time

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica"]})
rcParams.update({'font.size': 18})


#%% Functions
def func_axes_equal(ax,x_middle, y_middle, z_middle, plot_radius):
    # Function to set the dimension of the axis equal in a figure
    # x_limits : limits of the plot in x
    # y_limits : limits of the plot in y
    # z_limits : limits of the plot in z
    # x_range  : range of the plot in x
    # y_range  : range of the plot in y
    # z_range  : range of the plot in z
    if bool(x_middle)==0 and x_middle != 0:
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range  = abs(x_limits[1] - x_limits[0])
        x_middle = np.mean(x_limits)
        y_range  = abs(y_limits[1] - y_limits[0])
        y_middle = np.mean(y_limits)
        z_range  = abs(z_limits[1] - z_limits[0])
        z_middle = np.mean(z_limits)
        # The plot is contained in a sphere of diameter the required maximum distance
        plot_radius = 0.55*max([x_range, y_range, z_range]) # 
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])
#    ax.xaxis.zoom(2)
#    ax.yaxis.zoom(2)
#    ax.zaxis.zoom(2)
    return x_middle, y_middle, z_middle, plot_radius

#%%
def yvalue_select(yaxis,solution,ii_node,time_ii,ii_mode,typeflag):
    # Function to select the parameter that is specified in the y axis of the function
    # yaxis    : field selected for the y value
    # solution : information of the solution
    # time_ii  : time step of the calculation, if modal plot, it represents the mode. If it is null, stationary value
    # -------------------------------------------------------------------------
    # Find the field of yaxis
    #    - U_DEF         : deformation in x axis
    #    - V_DEF         : deformation in y axis
    #    - W_DEF         : deformation in z axis
    #    - PHI_DEF       : rotation in x axis
    #    - PSI_DEF       : rotation in y axis
    #    - THETA_DEF     : rotation in z axis
    #    - PHI_DEF_DEG   : rotation in x axis in deg
    #    - PSI_DEF_DEG   : rotation in y axis in deg
    #    - THETA_DEF_DEG : rotation in z axis in deg
    #    - FX            : force in x axis
    #    - FY            : force in y axis
    #    - FZ            : force in z axis
    #    - MX            : moment in x axis
    #    - MY            : moment in y axis
    #    - MZ            : moment in z axis
    #    - X_DEF         : deformed position in x axis
    #    - Y_DEF         : deformed position in y axis
    #    - Z_DEF         : deformed position in z axis
    y_val = []
    if yaxis == "U_DEF":
        y_vec = solution.u
    elif yaxis == "V_DEF":
        y_vec = solution.v
    elif yaxis == "W_DEF":
        y_vec = solution.w
    elif yaxis == "PHI_DEF":
        y_vec = solution.phi
    elif yaxis == "PSI_DEF":
        y_vec = solution.psi
    elif yaxis == "THETA_DEF":
        y_vec = solution.theta
    elif yaxis == "PHI_DEF_DEG":
        y_vec = np.array(solution.phi)*180/np.pi
    elif yaxis == "PSI_DEF_DEG":
        y_vec = np.array(solution.psi)*180/np.pi
    elif yaxis == "THETA_DEF_DEG":
        y_vec = np.array(solution.theta)*180/np.pi
    elif yaxis == "UDT_DEF":
        y_vec = solution.udt
    elif yaxis == "VDT_DEF":
        y_vec = solution.vdt
    elif yaxis == "WDT_DEF":
        y_vec = solution.wdt
    elif yaxis == "PHIDT_DEF":
        y_vec = solution.phidt
    elif yaxis == "PSIDT_DEF":
        y_vec = solution.psidt
    elif yaxis == "THETADT_DEF":
        y_vec = solution.thetadt
    elif yaxis == "PHIDT_DEF_DEG":
        y_vec = np.array(solution.phidt)*180/np.pi
    elif yaxis == "PSIDT_DEF_DEG":
        y_vec = np.array(solution.psidt)*180/np.pi
    elif yaxis == "THETADT_DEF_DEG":
        y_vec = np.array(solution.thetadt)*180/np.pi
    elif yaxis == "CL_SEC":
        y_vec = solution.cl_vec
    elif yaxis == "CD_SEC":
        y_vec = solution.cd_vec
    elif yaxis == "CM_SEC":
        y_vec = solution.cm_vec
    elif yaxis == "ALPHA_DEF":
        y_vec = solution.aoa
    elif yaxis == "ALPHADT_DEF":
        y_vec = solution.aoadt
    elif yaxis == "ALPHADTDT_DEF":
        y_vec = solution.aoadtdt
    elif yaxis == "ALPHA_DEF_DEG":
        y_vec = np.array(solution.aoa)*180/np.pi
    elif yaxis == "ALPHADT_DEF_DEG":
        y_vec = np.array(solution.aoadt)*180/np.pi
    elif yaxis == "ALPHADTDT_DEF_DEG":
        y_vec = np.array(solution.aoadtdt)*180/np.pi
    elif yaxis == "UDTDT_DEF":
        y_vec = solution.udtdt
    elif yaxis == "VDTDT_DEF":
        y_vec = solution.vdtdt
    elif yaxis == "WDTDT_DEF":
        y_vec = solution.wdtdt
    elif yaxis == "PHIDTDT_DEF":
        y_vec = solution.phidtdt
    elif yaxis == "PSIDTDT_DEF":
        y_vec = solution.psidtdt
    elif yaxis == "THETADTDT_DEF":
        y_vec = solution.thetadtdt
    elif yaxis == "PHIDTDT_DEF_DEG":
        y_vec = np.array(solution.phidtdt)*180/np.pi
    elif yaxis == "PSIDTDT_DEF_DEG":
        y_vec = np.array(solution.psidtdt)*180/np.pi
    elif yaxis == "THETADTDT_DEF_DEG":
        y_vec = np.array(solution.thetadtdt)*180/np.pi
    elif yaxis == "FX":
        y_vec = solution.fx
    elif yaxis == "FY":
        y_vec = solution.fy
    elif yaxis == "FZ":
        y_vec = solution.fz
    elif yaxis == "MX":
        y_vec = solution.mx
    elif yaxis == "MY":
        y_vec = solution.my
    elif yaxis == "MZ":
        y_vec = solution.mz
    elif yaxis == "X_DEF":
        y_vec = solution.xdef
    elif yaxis == "Y_DEF":
        y_vec = solution.ydef
    elif yaxis == "Z_DEF":
        y_vec = solution.zdef
    elif yaxis == "CL":
        y_val = solution.CL
    elif yaxis == "CD":
        y_val = solution.CD
    elif yaxis == "CM":
        y_val = solution.CM
    elif yaxis == "CT":
        y_val = solution.CT
    elif yaxis == "CQ":
        y_val = solution.CQ
    elif yaxis == "CP":
        y_val = solution.CP
    elif yaxis == "PE":
        y_val = solution.EP
    if typeflag == 1:
        if yaxis == "MOD_INF":
            y_vec = solution.yy
    # if time type or steady is chosen
    try:
        if typeflag == 0:
            # Select the node
            if len(y_val)==0:
                y_val = y_vec[ii_node]
        elif typeflag == 1:
            if len(time_ii)> 0:
                y_val = y_vec[ii_node][time_ii]
            else:
                y_val = y_vec[ii_node]
        elif typeflag == 2:
            y_val = y_vec[ii_node,ii_mode]
    except:
        pass
    return y_val

#%%
def mvalue_select(maxis,solution,ii_vinf,ii_mode,ii_mode_ant,ii_mode_ant2,indflag,used_mode,ii_count):
    if all(used_mode>=0):
        indflag = 1
        ii_mode_ant = int(used_mode[ii_count])
    m_val = []
    if maxis == "REAL" or maxis == "IMAG":
        if indflag == 1:
            m_val1_re = np.real(solution.l_mode[ii_mode_ant,ii_vinf])
            m_val1_im = np.imag(solution.l_mode[ii_mode_ant,ii_vinf])
            if maxis == "REAL":
                m_val = m_val1_re
            elif maxis == "IMAG":
                m_val = m_val1_im
        else:
            indflag += 1
            m_val1_re = np.real(solution.l_mode[ii_mode,ii_vinf])
            m_val1_im = np.imag(solution.l_mode[ii_mode,ii_vinf])
            if ii_vinf == 0:
                if maxis == "REAL":
                    m_val = m_val1_re
                elif maxis == "IMAG":
                    m_val = m_val1_im
                ii_mode_ant = ii_mode
            else:
                m_val_re_ant = np.real(solution.l_mode[ii_mode_ant,ii_vinf-1])
                m_val_im_ant = np.imag(solution.l_mode[ii_mode_ant,ii_vinf-1])
                tol = 0.001 #abs(m_val_im_ant*0.05)
                tol_ini = tol
#                print(tol)
                add1 = 1
                add2 = -1
                add3 = 2
                add4 = -2
                add5 = 3
                add6 = -3
                add7 = 4
                add8 = -4
                if ii_mode+add2<0:
                    add2 = 5
                if ii_mode+add4<0:
                    add4 = 6
                if ii_mode+add6<0:
                    add6 = 7
                if ii_mode+add8<0:
                    add8 = 8
#                if ii_mode+add1 > len(solution.l_mode[:,0]):
#                    add1 = -5
#                if ii_mode+add3 > len(solution.l_mode[:,0]):
#                    add3 = -6
#                if ii_mode+add5 > len(solution.l_mode[:,0]):
#                    add5 = -7
#                if ii_mode+add7 > len(solution.l_mode[:,0]):
#                    add7 = -9
                m_val_ant = np.abs(solution.l_mode[ii_mode_ant,ii_vinf-1])
                m_val_ant2 = np.abs(solution.l_mode[ii_mode_ant2,ii_vinf-1])
                m_val_re_ant2 = np.real(m_val_ant2)
                m_val_im_ant2 = np.imag(m_val_ant2)
                error_1_re = abs((m_val1_re-m_val_re_ant)/m_val_ant)
                error_1_im = abs((m_val1_im-m_val_im_ant)/m_val_ant)
                if ii_mode+add1 >= 0 and ii_mode+add1 < len(solution.l_mode[:,0]):
                    m_val2_re = np.real(solution.l_mode[ii_mode+add1,ii_vinf])
                    m_val2_im = np.imag(solution.l_mode[ii_mode+add1,ii_vinf])
                    error_2_re = abs((m_val2_re-m_val_re_ant)/m_val_ant)
                    error_2_im = abs((m_val2_im-m_val_im_ant)/m_val_ant)
                    flag2 = 1
                else:
                    flag2 = 0
                if ii_mode+add2 >= 0 and ii_mode+add2 < len(solution.l_mode[:,0]):
                    m_val3_re = np.real(solution.l_mode[ii_mode+add2,ii_vinf])
                    m_val3_im = np.imag(solution.l_mode[ii_mode+add2,ii_vinf])
                    error_3_re = abs((m_val3_re-m_val_re_ant)/m_val_ant)
                    error_3_im = abs((m_val3_im-m_val_im_ant)/m_val_ant)
                    flag3 = 1
                else:
                    flag3 = 0
                if ii_mode+add3 >= 0 and ii_mode+add3 < len(solution.l_mode[:,0]):
                    m_val4_re = np.real(solution.l_mode[ii_mode+add3,ii_vinf])
                    m_val4_im = np.imag(solution.l_mode[ii_mode+add3,ii_vinf])
                    error_4_re = abs((m_val4_re-m_val_re_ant)/m_val_ant)
                    error_4_im = abs((m_val4_im-m_val_im_ant)/m_val_ant)
                    flag4 = 1
                else:
                    flag4 = 0
                if ii_mode+add4 >= 0 and ii_mode+add4 < len(solution.l_mode[:,0]):
                    m_val5_re = np.real(solution.l_mode[ii_mode+add4,ii_vinf])
                    m_val5_im = np.imag(solution.l_mode[ii_mode+add4,ii_vinf])
                    error_5_re = abs((m_val5_re-m_val_re_ant)/m_val_ant)
                    error_5_im = abs((m_val5_im-m_val_im_ant)/m_val_ant)
                    flag5 = 1
                else:
                    flag5 = 0                    
                if ii_mode+add5 >= 0 and ii_mode+add5 < len(solution.l_mode[:,0]):
                    m_val6_re = np.real(solution.l_mode[ii_mode+add5,ii_vinf])
                    m_val6_im = np.imag(solution.l_mode[ii_mode+add5,ii_vinf])
                    error_6_re = abs((m_val6_re-m_val_re_ant)/m_val_ant)
                    error_6_im = abs((m_val6_im-m_val_im_ant)/m_val_ant)
                    flag6 = 1
                else:
                    flag6 = 0
                if ii_mode+add6 >= 0 and ii_mode+add6 < len(solution.l_mode[:,0]):
                    m_val7_re = np.real(solution.l_mode[ii_mode+add6,ii_vinf])
                    m_val7_im = np.imag(solution.l_mode[ii_mode+add6,ii_vinf])
                    error_7_re = abs((m_val7_re-m_val_re_ant)/m_val_ant)
                    error_7_im = abs((m_val7_im-m_val_im_ant)/m_val_ant)
                    flag7 = 1
                else:
                    flag7 = 0
                if ii_mode+add7 >= 0 and ii_mode+add7 < len(solution.l_mode[:,0]):
                    m_val8_re = np.real(solution.l_mode[ii_mode+add7,ii_vinf])
                    m_val8_im = np.imag(solution.l_mode[ii_mode+add7,ii_vinf])
                    error_8_re = abs((m_val8_re-m_val_re_ant)/m_val_ant)
                    error_8_im = abs((m_val8_im-m_val_im_ant)/m_val_ant)
                    flag8 = 1
                else:
                    flag8 = 0
                if ii_mode+add8 >= 0 and ii_mode+add8 < len(solution.l_mode[:,0]):
                    m_val9_re = np.real(solution.l_mode[ii_mode+add8,ii_vinf])
                    m_val9_im = np.imag(solution.l_mode[ii_mode+add8,ii_vinf])
                    error_9_re = abs((m_val9_re-m_val_re_ant)/m_val_ant)
                    error_9_im = abs((m_val9_im-m_val_im_ant)/m_val_ant)
                    flag9 = 1
                else:
                    flag9 = 0
                flag = np.array([0,0,0,0,0,0,0,0,0])
                error_re = np.array([1e100,1e100,1e100,1e100,1e100,1e100,1e100,1e100,1e100])
                error_im = np.array([1e100,1e100,1e100,1e100,1e100,1e100,1e100,1e100,1e100])
                m_val_vec = np.zeros((9,))
                ftol = 100
                while True:
                    if (error_1_im < tol) and np.sign(m_val_im_ant)==np.sign(m_val1_im) and flag[0]==0 and len(np.where(used_mode==ii_mode)[0])==0:
                        flag[0] = 1
#                        print('1')
                        error_im[0] = error_1_im
                        error_re[0] = error_1_re
                        m_val_vec[0] = m_val1_re
                    if flag2 == 1 and (error_2_im < tol) and np.sign(m_val_im_ant)==np.sign(m_val2_im)and flag[1]==0 and len(np.where(used_mode==ii_mode+add1)[0])==0:
                        flag[1] = 1
                        error_im[1] = error_2_im
                        error_re[1] = error_2_re
                        m_val_vec[1] = m_val2_re
                    if flag3 == 1 and (error_3_im < tol) and np.sign(m_val_im_ant)==np.sign(m_val3_im)  and flag[2]==0 and len(np.where(used_mode==ii_mode+add2)[0])==0:
                        flag[2] = 1
                        error_im[2] = error_3_im
                        error_re[2] = error_3_re
                        m_val_vec[2] = m_val3_re
                    if flag4 == 1 and (error_4_im < tol) and np.sign(m_val_im_ant)==np.sign(m_val4_im) and flag[3]==0 and len(np.where(used_mode==ii_mode+add3)[0])==0:
                        flag[3] = 1
                        error_im[3] = error_4_im
                        error_re[3] = error_4_re
                        m_val_vec[3] = m_val4_re
                    if flag5 == 1 and (error_5_im < tol) and np.sign(m_val_im_ant)==np.sign(m_val5_im) and flag[4]==0 and len(np.where(used_mode==ii_mode+add4)[0])==0:
                        flag[4] = 1
                        error_im[4] = error_5_im
                        error_re[4] = error_5_re
                        m_val_vec[4] = m_val5_re
                    if flag6 == 1 and (error_6_im < tol ) and np.sign(m_val_im_ant)==np.sign(m_val6_im) and flag[5]==0 and len(np.where(used_mode==ii_mode+add5)[0])==0:
                        flag[5] = 1
                        error_im[5] = error_6_im
                        error_re[5] = error_6_re
                        m_val_vec[5] = m_val6_re
                    if flag7 == 1 and (error_7_im < tol) and np.sign(m_val_im_ant)==np.sign(m_val7_im) and flag[6]==0 and len(np.where(used_mode==ii_mode+add6)[0])==0:
                        flag[6] = 1
                        error_im[6] = error_7_im
                        error_re[6] = error_7_re
                        m_val_vec[6] = m_val7_re
                    if flag8 == 1 and (error_8_im < tol) and np.sign(m_val_im_ant)==np.sign(m_val8_im) and flag[7]==0 and len(np.where(used_mode==ii_mode+add7)[0])==0:
                        flag[7] = 1
                        error_im[7] = error_8_im
                        error_re[7] = error_8_re
                        m_val_vec[7] = m_val8_re
                    if flag9 == 1 and (error_9_im < tol) and np.sign(m_val_im_ant)==np.sign(m_val9_im) and flag[8]==0 and len(np.where(used_mode==ii_mode+add8)[0])==0:
                        flag[8] = 1
                        error_im[8] = error_9_im
                        error_re[8] = error_9_re
                        m_val_vec[8] = m_val9_re
                    if sum(flag) < 3 and tol<tol_ini*ftol:
                        tol *=  1.2
#                        if sum(flag) == 1:
#                            flag += 1
                    else:
#                        if ii_mode < 2:
#                            print(error_im)
#                        if ii_mode == 0:
#                            print(1)
                        if tol>tol_ini*ftol:
                            try:
                                if len(np.where(used_mode==ii_mode)[0])==0 and (abs(m_val1_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val1_im)):
                                    error_im[0] = error_1_im
                                if len(np.where(used_mode==ii_mode+add1)[0])==0 and (abs(m_val2_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val2_im)):
                                    error_im[1] = error_2_im
                                if len(np.where(used_mode==ii_mode+add2)[0])==0 and (abs(m_val3_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val3_im)):
                                    error_im[2] = error_3_im
                                if len(np.where(used_mode==ii_mode+add3)[0])==0 and (abs(m_val4_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val4_im)):
                                    error_im[3] = error_4_im
                                if len(np.where(used_mode==ii_mode+add4)[0])==0 and (abs(m_val5_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val5_im)):
                                    error_im[4] = error_5_im
                                if len(np.where(used_mode==ii_mode+add5)[0])==0 and (abs(m_val6_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val6_im)):
                                    error_im[5] = error_6_im
                                if len(np.where(used_mode==ii_mode+add6)[0])==0 and (abs(m_val7_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val7_im)):
                                    error_im[6] = error_7_im
                                if len(np.where(used_mode==ii_mode+add7)[0])==0 and (abs(m_val8_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val8_im)):
                                    error_im[7] = error_8_im
                                if len(np.where(used_mode==ii_mode+add8)[0])==0 and (abs(m_val9_im)<1e-5 or np.sign(m_val_im_ant)==np.sign(m_val9_im)):
                                    error_im[8] = error_9_im
                            except:
                                pass
                            ii_minim = np.argmin(error_im)
                            if ii_minim == 0:
                                error_re[0] = error_1_re
                            elif ii_minim == 1:
                                error_re[1] = error_2_re
                            elif ii_minim == 2:
                                error_re[2] = error_3_re
                            elif ii_minim == 3:
                                error_re[3] = error_4_re
                            elif ii_minim == 4:
                                error_re[4] = error_5_re
                            elif ii_minim == 5:
                                error_re[5] = error_6_re
                            elif ii_minim == 6:
                                error_re[6] = error_7_re
                            elif ii_minim == 7:
                                error_re[7] = error_8_re
                            elif ii_minim == 8:
                                error_re[8] = error_9_re
                        break
                slope_re = m_val_vec-m_val_re_ant
                slope_re2 = m_val_re_ant-m_val_re_ant2
                iimin = np.argmin(np.sqrt(error_re**2+error_im**2)) #np.argmin(np.sqrt(error_re**2)) #np.argmin(np.sqrt(error_re**2+error_im**2)) #np.argmin(slope_re) #error_re**2) # np.argmin(np.sqrt(error_re**2+error_im**2) ) # 
#                slope1 = []
#                error_re2 = error_re.copy()
#                error_re2[iimin] = 1e100 
#                ii_zeros = 0
#                for ii_slope in np.arange(len(slope_re)):
#                    if error_re[ii_slope] < 1e100:
#                        if all(error_re2*0.1>error_re[iimin]):
#                            slope1.append(slope_re2)
#                            ii_zeros += 1
#                        else:
#                            slope1.append(slope_re[ii_slope])
#                    else:
#                        slope1.append(1e100)
#                slope1arr = np.array(slope1)
#                print(ii_vinf)
#                if ii_mode == 2 and ii_vinf == 12:
#                    print(1)
#                if ii_zeros <= 1:
#                    iimin = np.argmin(slope1arr-slope_re2)
#                else:
#                    print(1)
#                print(ii_mode)
#                print(error_re)
                if iimin == 0:
                    if maxis == "REAL":
                        m_val = m_val1_re
                    elif maxis == "IMAG":
                        m_val = m_val1_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode
                elif iimin == 1:
                    if maxis == "REAL":
                        m_val = m_val2_re
                    elif maxis == "IMAG":
                        m_val = m_val2_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode+add1
                elif iimin == 2:
                    if maxis == "REAL":
                        m_val = m_val3_re
                    elif maxis == "IMAG":
                        m_val = m_val3_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode+add2
                elif iimin == 3:
                    if maxis == "REAL":
                        m_val = m_val4_re
                    elif maxis == "IMAG":
                        m_val = m_val4_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode+add3
                elif iimin == 4:
                    if maxis == "REAL":
                        m_val = m_val5_re
                    elif maxis == "IMAG":
                        m_val = m_val5_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode+add4
                elif iimin == 5:
                    if maxis == "REAL":
                        m_val = m_val6_re
                    elif maxis == "IMAG":
                        m_val = m_val6_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode+add5
                elif iimin == 6:
                    if maxis == "REAL":
                        m_val = m_val7_re
                    elif maxis == "IMAG":
                        m_val = m_val7_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode+add6
                elif iimin == 7:
                    if maxis == "REAL":
                        m_val = m_val8_re
                    elif maxis == "IMAG":
                        m_val = m_val8_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode+add7
                elif iimin == 8:
                    if maxis == "REAL":
                        m_val = m_val9_re
                    elif maxis == "IMAG":
                        m_val = m_val9_im
                    ii_mode_ant2 = ii_mode_ant
                    ii_mode_ant = ii_mode+add8
    if maxis == "VEL":
        m_val = solution.vinf[ii_vinf]
    return m_val,ii_mode_ant,ii_mode_ant2,indflag
 #%%
def plot_2dxy(case_setup,plot_ii,xplot,yplot,yplot2,flag2axis,time_ii):
    # Function to plot 2D xy plots
    # -------------------------------------------------------------------------
    # legendline  : curves to include in the legend
    # legendlabel : label to include in the legend
    # colormap    : colormap to use in the figure
    legendline  = []
    legendlabel = []
    try:
        colormap = cm.get_cmap(plot_ii.color, 3)
    except:
        colormap = cm.get_cmap('tab10', 3)
    # data_plot   : table to save the numerical data of the plot
    data_plot                 = pd.DataFrame()
    data_plot[plot_ii.xlabel] = xplot
    data_plot[plot_ii.ylabel] = yplot
    if flag2axis == 1:
        data_plot[plot_ii.ylabel2] = yplot2
    # Define the figure and the plotting options
    plt.figure(plot_ii.fig)
    ax         = plt.subplot(1,1,1)
    line       = ax.plot(xplot,yplot,label=plot_ii.label,color=colormap.colors[0,:])
    lab        = line[0].get_label()
    legendline = legendline + line
    legendlabel.append(lab)
    plt.xlabel(plot_ii.xlabel)
    plt.ylabel(plot_ii.ylabel)
    if flag2axis == 1:
        ax2        = ax.twinx()
        line       = ax2.plot(xplot,yplot2,label=plot_ii.label2,color=colormap.colors[1,:])
        lab        = line[0].get_label()
        legendline = legendline + line
        legendlabel.append(lab)
        ax2.set_ylabel(plot_ii.ylabel2)
        ax.legend(legendline, legendlabel)
        ax2.grid(None)
    plt.grid()
    plt.title(plot_ii.title)
    plt.tight_layout()
    try:
        if bool(time_ii) or time_ii == 0:
            plt.savefig(case_setup.root + plot_ii.save[:-4] + '_'+str(time_ii) + plot_ii.save[-4:],bbox_inches='tight')
            data_plot.to_csv(case_setup.root + plot_ii.save_data[:-4] + '_'+str(time_ii) + plot_ii.save_data[-4:])
            plt.cla()
        else:
            plt.savefig(case_setup.root + plot_ii.save,bbox_inches='tight')
            data_plot.to_csv(case_setup.root + plot_ii.save_data)
    except:
        pass   
    return
 #%%
def plot_2dxy_mod(case_setup,plot_ii,xplot,yplot,ii_color,ii_end,data_plot,legendline,legendlabel,ii_mode,v_inf):
    # Function to plot 2D xy plots
    # -------------------------------------------------------------------------
    # legendline  : curves to include in the legend
    # legendlabel : label to include in the legend
    # colormap    : colormap to use in the figure
    try:
        colormap = cm.get_cmap(plot_ii.color, ii_end)
    except:
        colormap = cm.get_cmap('tab10', ii_end)
    # data_plot   : table to save the numerical data of the plot
    data_plot['mode: ' + str(ii_color) +plot_ii.xlabel] = xplot
    data_plot['mode: ' + str(ii_color) +plot_ii.ylabel] = yplot
    if plot_ii.zaxis == 'MOD':
        str_lab = 'Mode '+ str(ii_mode)
    else:
        str_lab = '$V_\infty=$ '+ v_inf+' m/s'
    # Define the figure and the plotting options
    plt.figure(plot_ii.fig)
    ax         = plt.subplot(1,1,1)
    line       = ax.plot(xplot,yplot,label=str_lab,color=colormap.colors[ii_color,:])
    lab        = line[0].get_label()
    legendline = legendline + line
    legendlabel.append(lab)
    if ii_color == ii_end-1:
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
        ax.legend(legendline, legendlabel,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(plot_ii.xlabel)
        plt.ylabel(plot_ii.ylabel)
        plt.title(plot_ii.title)
#        plt.tight_layout()
#        plt.xscale('symlog')
#        plt.yscale('symlog')
        plt.grid()
    try:
        if ii_color == ii_end-1:
            plt.savefig(case_setup.root + plot_ii.save,bbox_inches='tight')
            data_plot.to_csv(case_setup.root + plot_ii.save_data)
    except:
        pass   
    return data_plot,legendline,legendlabel
 #%%
def plot_2dxy_multiple(case_setup,plot_ii,xplot,yplot,yplot2,flag2axis,nmax,mesh_data,markers):
    # Function to plot 2D xy plots
    # -------------------------------------------------------------------------
    # legendline  : curves to include in the legend
    # legendlabel : label to include in the legend
    # colormap    : colormap to use in the figure
    legendline  = []
    legendlabel = []
    try:
        colormap = cm.get_cmap(plot_ii.color, 2*nmax+1)
    except:
        colormap = cm.get_cmap('tab10', 2*nmax+1)
    # data_plot   : table to save the numerical data of the plot
    data_plot                 = pd.DataFrame()
    data_plot[plot_ii.xlabel] = xplot
    if flag2axis == 1 or nmax > 1:
        plt.figure(plot_ii.fig,figsize=(15,5))
    else: 
        plt.figure(plot_ii.fig)
    ax         = plt.subplot(1,1,1)
    position   = np.zeros((nmax,))
    if flag2axis == 1:
        ax2        = ax.twinx()
    # Define the figure and the plotting options
    for nii in np.arange(nmax):
        if plot_ii.yaxis == "MOD_INF":
            position[nii] = nii
        else:
            position[nii]   = np.dot(plot_ii.xvalues,mesh_data.point[int(markers[nii]),:])
    iipos = np.argsort(position)
    for nii in np.arange(nmax):
        ind_plot = iipos[nii]
        data_plot[plot_ii.ylabel+'_'+str(ind_plot)] = yplot[ind_plot]
        if plot_ii.yaxis == "MOD_INF":
            labeltxt   = plot_ii.label+' : '+str(position[ind_plot])
        else:
            labeltxt   = plot_ii.label+' : '+str(position[ind_plot])+' m'
        if len(yplot[ind_plot])>1:
            line       = ax.plot(xplot,yplot[ind_plot],label=labeltxt,color=colormap.colors[2*nii,:])
        else:
            line       = ax.plot(xplot,yplot,label=labeltxt,color=colormap.colors[2*nii,:])
        lab        = line[0].get_label()
        legendline = legendline + line
        legendlabel.append(lab)
        if flag2axis == 1:
            data_plot[plot_ii.ylabel2+'_'+str(ind_plot)] = yplot2[ind_plot]
            if plot_ii.yaxis == "MOD_INF":
                labeltxt   = plot_ii.label2+' : '+str(position[ind_plot])
            else:
                labeltxt   = plot_ii.label2+' : '+str(position[ind_plot])+' m'
            line       = ax.plot(xplot,yplot,label=labeltxt,color=colormap.colors[2*nii,:])
            line       = ax2.plot(xplot,yplot2[ind_plot],'--',label=labeltxt,color=colormap.colors[2*nii+1,:])
            lab        = line[0].get_label()
            legendline = legendline + line
            legendlabel.append(lab)
    if flag2axis == 1:
        ax2.set_ylabel(plot_ii.ylabel2)
        ax.legend(legendline, legendlabel)
        ax2.grid(None)
    if len(legendline)>1:
        ax.legend(legendline, legendlabel, bbox_to_anchor=(1.2, 1), loc='upper center') # 
    ax.set_xlabel(plot_ii.xlabel)
    ax.set_ylabel(plot_ii.ylabel)
    ax.grid()
    plt.title(plot_ii.title)
    plt.tight_layout()
    try:
        plt.savefig(case_setup.root + plot_ii.save,bbox_inches='tight')
        data_plot.to_csv(case_setup.root + plot_ii.save_data)
    except:
        pass   
    return

def plot_defshapemesh(mesh_data,section_globalCS,solution,plot_ii):
    # ------------------------------------------------------------------------
    # Take the values of x, y and z in the nodes of the marker
    # xplotu  : undeformed x coordinate
    # yplotu  : undeformed y coordinate
    # zplotu  : undeformed z coordinate
    # normalx : normal of the surface in the x direction
    # normaly : normal of the surface in the y direction
    # normalz : normal of the surface in the z direction
    xplotu  = def_vec_param(int(np.max(section_globalCS.nsub)))
    yplotu  = def_vec_param(int(np.max(section_globalCS.nsub)))
    zplotu  = def_vec_param(int(np.max(section_globalCS.nsub)))
    normalx = def_vec_param(int(np.max(section_globalCS.nsub)))
    normaly = def_vec_param(int(np.max(section_globalCS.nsub)))
    normalz = def_vec_param(int(np.max(section_globalCS.nsub)))
    if plot_ii.face == 'P2P':
        triangles_tot = []
        elements_tot = []
        try:
            markvec = plot_ii.marker 
        except:
            markvec = 'NONE'
        for markplot in markvec:
            triangles = []
            for auxmark in mesh_data.marker:
                if markplot == 'NONE':
                    num_elem = mesh_data.num_elem
                    elementmark = np.arange(num_elem)
                if auxmark.name == markplot:
                    elementmark = []
                    nodes = auxmark.node.flatten().astype(int)
                    triang = def_vec_param(int(np.max(section_globalCS.nsub[nodes])))
                    n_add1    = np.zeros((int(np.max(section_globalCS.nsub)),))
                    n_add2    = np.zeros((int(np.max(section_globalCS.nsub)),))
                    sec_add1   = np.zeros((int(np.max(section_globalCS.nsub)),))
                    sec_add2   = np.zeros((int(np.max(section_globalCS.nsub)),))
                    used_node = []
                    point_cloud = []
                    sec_add_end = np.zeros((int(np.max(section_globalCS.nsub)),))
                    for nodemark in nodes:
                        cont_elem1 = np.where(mesh_data.elem[:,1]==nodemark)[0]
                        if len(cont_elem1)>0:
                            for jjmark in np.arange(len(cont_elem1)):
                                cont_elemii = cont_elem1[jjmark]
                                if len(np.where(nodes==mesh_data.elem[cont_elemii,2])[0])>0:
                                    if len(np.where(elementmark==cont_elemii)[0])==0:
                                        elementmark.append(cont_elemii)
                        cont_elem2 = np.where(mesh_data.elem[:,2]==nodemark)[0]
                        if len(cont_elem2)>0:
                            for jjmark in np.arange(len(cont_elem2)):
                                cont_elemii = cont_elem2[jjmark]
                                if len(np.where(nodes==mesh_data.elem[cont_elemii,1])[0])>0:
                                    if len(np.where(elementmark==cont_elemii)[0])==0:
                                        elementmark.append(cont_elemii)
                    num_elem = len(elementmark)
            for ii_elemaux in np.arange(len(elementmark)):
                ii_elem = elementmark[ii_elemaux]
                # if the first node of the element has not been used
                selem1 = section_globalCS.e1[ii_elem]
                selem2 = section_globalCS.e2[ii_elem]
                lentri1 = []
                lentri2 = []
                lentri = []
                ii_sub = 0
                jj_point0 = 0
                for jj_point in np.arange(len(selem1)): 
                    if jj_point == len(selem1)-1:
                        sec_add1[ii_sub] = len(np.unique(selem1[jj_point0+1:jj_point+1,1:3]))
                        lentri1.append(jj_point-jj_point0+1)
                        jj_point0 = jj_point+1
                        ii_sub += 1
                    else:
                        if selem1[jj_point+1,0] == 0:
                            sec_add1[ii_sub] = len(np.unique(selem1[jj_point0+1:jj_point+1,1:3]))#section_globalCS.e1[ii_elem][jj_point,0]+1
                            lentri1.append(jj_point-jj_point0+1)
                            jj_point0 = jj_point+1 
                            ii_sub += 1
                ii_sub = 0
                jj_point0 = 0
                for jj_point in np.arange(len(selem2)): 
                    if jj_point == len(selem2)-1:
                        sec_add2[ii_sub] = len(np.unique(selem2[jj_point0+1:jj_point+1,1:3]))
                        lentri2.append(jj_point-jj_point0+1)
                        jj_point0 = jj_point+1
                        ii_sub += 1
                    else:
                        if selem2[jj_point+1,0] == 0:
                            sec_add2[ii_sub] = len(np.unique(selem2[jj_point0+1:jj_point+1,1:3]))#section_globalCS.e1[ii_elem][jj_point,0]+1
                            lentri2.append(jj_point-jj_point0+1)
                            jj_point0 = jj_point+1 
                            ii_sub += 1
                for ii_lentri in np.arange(min([len(lentri1),len(lentri2)])):
                    if len(lentri1) < ii_lentri:
                        lentri.append(lentri2)
                    elif len(lentri2) < ii_lentri:
                        lentri.append(lentri1) 
                    else:
                        lentri.append(np.min([lentri1[ii_lentri],lentri2[ii_lentri]]))
                n_add1 = sec_add_end
                n_add2 = sec_add1 + sec_add_end
                ii_sub = 0
                max_point = 0
                lentri0 = 0
                for ii_lentri in np.arange(len(lentri)):
                    for jj_point in np.arange(lentri[ii_lentri])+lentri0: 
                        tri_line = [n_add1[ii_sub]+selem1[jj_point,1],n_add1[ii_sub]+selem1[jj_point,2],n_add2[ii_sub]+selem2[jj_point,1]]
                        triang[ii_sub].append(tri_line) 
                        tri_line = [n_add1[ii_sub]+selem1[jj_point,2],n_add2[ii_sub]+selem2[jj_point,1],n_add2[ii_sub]+selem2[jj_point,2]]
                        triang[ii_sub].append(tri_line)
                        max_point = np.max([max_point,np.max(tri_line)])
                        if jj_point == sum(lentri)-1:
                            ii_sub += 1 
                            lentri0 += lentri[ii_lentri]
        #                    n_add1[ii_sub] = max_point+1#section_globalCS.e1[ii_elem][jj_point,0]+1
                        else:
                            if selem1[jj_point+1,0] == 0:
                                ii_sub += 1 
                                lentri0 += lentri[ii_lentri]
        #                    n_add1[ii_sub] = max_point+1#n_add2[ii_sub-1]#section_globalCS.e1[ii_elem][jj_point,0]+1
                if  ii_elem == elementmark[-1]:
                    for ii_sub in np.arange(np.min([section_globalCS.nsub[int(section_globalCS.nodeelem1[ii_elem])],section_globalCS.nsub[int(section_globalCS.nodeelem1[ii_elem])]])):
                        triang_np = np.array(triang[int(ii_sub)])
                        triangles.append(triang_np)
                else:
                    sec_add_end += sec_add1
            triangles_tot.append(triangles)   
            elements_tot.append(elementmark)
    return triangles_tot,elements_tot


def plot_defshape(case_setup,mesh_data,section_globalCS,solution,plot_ii,ii_time,ii_mode,triangles_tot,elements_tot,typeplot,x_middle, y_middle, z_middle, plot_radius,time_val):
    # Function to plot the deformed beam
    # case_setup       : configuration of the case
    # mesh_data        : information of the mesh
    # section_globalCS : section global coordinates
    # solution         : information of the solution
    # plot_ii          : information of the plot
    # ii_time          : time step or mode
    # ------------------------------------------------------------------------
    # phi   : angle axis x
    # psi   : angle axis y
    # theta : angle axis z
    # warp_x1 : warping in the x coordinate of node 1
    # warp_y1 : warping in the y coordinate of node 1
    # warp_z1 : warping in the z coordinate of node 1
    # warp_x2 : warping in the x coordinate of node 2
    # warp_y2 : warping in the y coordinate of node 2
    # warp_z2 : warping in the z coordinate of node 2
    if typeplot == 0:
        xdef    = solution.xdef
        ydef    = solution.ydef
        zdef    = solution.zdef
        uu      = solution.u
        vv      = solution.v
        ww      = solution.w
        phi     = solution.phi
        psi     = solution.psi
        theta   = solution.theta
        warp_x1 = solution.warp_x1
        warp_y1 = solution.warp_y1
        warp_z1 = solution.warp_z1
        warp_x2 = solution.warp_x2
        warp_y2 = solution.warp_y2
        warp_z2 = solution.warp_z2
    elif typeplot == 1:
        ii_time = int(ii_time)
        xdef    = solution.xdef[:,ii_time]
        ydef    = solution.ydef[:,ii_time]
        zdef    = solution.zdef[:,ii_time]
        uu      = np.zeros((len(solution.u),))
        vv      = np.zeros((len(solution.u),))
        ww      = np.zeros((len(solution.u),))
        phi     = np.zeros((len(solution.u),))
        psi     = np.zeros((len(solution.u),))
        theta   = np.zeros((len(solution.u),))
        warp_x1 = solution.warp_x1[ii_time]
        warp_y1 = solution.warp_y1[ii_time]
        warp_z1 = solution.warp_z1[ii_time]
        warp_x2 = solution.warp_x2[ii_time]
        warp_y2 = solution.warp_y2[ii_time]
        warp_z2 = solution.warp_z2[ii_time]
        for ii_node in np.arange(len(solution.u)):
            uu[ii_node]    = solution.u[ii_node][ii_time]
            vv[ii_node]    = solution.v[ii_node][ii_time]
            ww[ii_node]    = solution.w[ii_node][ii_time]
            phi[ii_node]   = solution.phi[ii_node][ii_time]
            psi[ii_node]   = solution.psi[ii_node][ii_time]
            theta[ii_node] = solution.theta[ii_node][ii_time]
    elif typeplot == 2:
        ii_mode = int(ii_mode)
        xdef    = solution.xdef[:,ii_mode]
        ydef    = solution.ydef[:,ii_mode]
        zdef    = solution.zdef[:,ii_mode]
        uu      = solution.u[:,ii_mode]
        vv      = solution.v[:,ii_mode]
        ww      = solution.w[:,ii_mode]
        phi     = solution.phi[:,ii_mode]
        psi     = solution.psi[:,ii_mode]
        theta   = solution.theta[:,ii_mode]
        warp_x1 = solution.warp_x1[ii_mode]
        warp_y1 = solution.warp_y1[ii_mode]
        warp_z1 = solution.warp_z1[ii_mode]
        warp_x2 = solution.warp_x2[ii_mode]
        warp_y2 = solution.warp_y2[ii_mode]
        warp_z2 = solution.warp_z2[ii_mode]
    elif typeplot == 3:
        ii_mode = int(ii_mode)
        ii_time = int(ii_time)
        xdef    = solution.xdef[:,ii_mode,ii_time]
        ydef    = solution.ydef[:,ii_mode,ii_time]
        zdef    = solution.zdef[:,ii_mode,ii_time]
        uu      = solution.u[:,ii_mode,ii_time]
        vv      = solution.v[:,ii_mode,ii_time]
        ww      = solution.w[:,ii_mode,ii_time]
        phi     = solution.phi[:,ii_mode,ii_time]
        psi     = solution.psi[:,ii_mode,ii_time]
        theta   = solution.theta[:,ii_mode,ii_time]
        warp_x1 = solution.warp_x1[ii_time][ii_mode]
        warp_y1 = solution.warp_y1[ii_time][ii_mode]
        warp_z1 = solution.warp_z1[ii_time][ii_mode]
        warp_x2 = solution.warp_x2[ii_time][ii_mode]
        warp_y2 = solution.warp_y2[ii_time][ii_mode]
        warp_z2 = solution.warp_z2[ii_time][ii_mode]
    # For all the elements in the mesh take the nodes that have not been used
    # used_node : used nodes
    fig = plt.figure(plot_ii.fig)
    ax  = fig.add_subplot(111, projection='3d')
    color_dimension_tot = []
    vert_tot = []
    xplotsurfu_tot = []
    yplotsurfu_tot = []
    zplotsurfu_tot = []
    for mark_aux in np.arange(len(triangles_tot)):
        used_node = []
        # Take the values of x, y and z in the nodes of the marker
        # xplot  : matrix of values of the x coordinate
        # yplot  : matrix of values of the y coordinate
        # zplot  : matrix of values of the z coordinate
        # xplotu : matrix of values of the undeformed x coordinate
        # yplotu : matrix of values of the undeformed y coordinate
        # zplotu : matrix of values of the undeformed z coordinate
        triangles = triangles_tot[mark_aux]
        xplot   = def_vec_param(len(triangles))
        yplot   = def_vec_param(len(triangles))
        zplot   = def_vec_param(len(triangles))
        xplotu  = def_vec_param(len(triangles))
        yplotu  = def_vec_param(len(triangles))
        zplotu  = def_vec_param(len(triangles))
        displot = def_vec_param(len(triangles))
        for ii_elem in elements_tot[mark_aux]:
            # ind_point : number of the beam node
            # plot_sec  : section nodes to plot
            # plot_cdg  : center of gravity of the section to plot
            # If the first node is not used before calculate the positions of the section points
            if len(np.where(np.array(used_node) == mesh_data.elem[ii_elem,1])[0]) == 0:
                used_node.append(mesh_data.elem[ii_elem,1])
                ind_point = section_globalCS.nodeelem1[ii_elem]
                for jj_point in np.arange(len(section_globalCS.n1[ii_elem])):  
                    # plot_sec : node coordinates
                    # defx     : deformation in coordinate x
                    # defy     : deformation in coordinate y
                    # defz     : deformation in coordinate z
                    plot_sec = section_globalCS.n1[ii_elem][jj_point]
                    defx     = xdef[ind_point]+(plot_sec[0])*np.cos(theta[ind_point])*np.cos(psi[ind_point])-(plot_sec[1])*np.sin(theta[ind_point])+(plot_sec[2])*np.sin(psi[ind_point])+warp_x1[ii_elem][jj_point]
                    defy     = ydef[ind_point]+(plot_sec[1])*np.cos(theta[ind_point])*np.cos(phi[ind_point])+(plot_sec[0])*np.sin(theta[ind_point])-(plot_sec[2])*np.sin(phi[ind_point])+warp_y1[ii_elem][jj_point]
                    defz     = zdef[ind_point]+(plot_sec[2])*np.cos(psi[ind_point])*np.cos(phi[ind_point])-(plot_sec[0])*np.sin(psi[ind_point])+(plot_sec[1])*np.sin(phi[ind_point])+warp_z1[ii_elem][jj_point]
                    disx     = uu[ind_point]+(plot_sec[0])*np.cos(theta[ind_point])*np.cos(psi[ind_point])-(plot_sec[1])*np.sin(theta[ind_point])+(plot_sec[2])*np.sin(psi[ind_point])+warp_x1[ii_elem][jj_point]-plot_sec[0]
                    disy     = vv[ind_point]+(plot_sec[1])*np.cos(theta[ind_point])*np.cos(phi[ind_point])+(plot_sec[0])*np.sin(theta[ind_point])-(plot_sec[2])*np.sin(phi[ind_point])+warp_y1[ii_elem][jj_point]-plot_sec[1]
                    disz     = ww[ind_point]+(plot_sec[2])*np.cos(psi[ind_point])*np.cos(phi[ind_point])-(plot_sec[0])*np.sin(psi[ind_point])+(plot_sec[1])*np.sin(phi[ind_point])+warp_z1[ii_elem][jj_point]-plot_sec[2]
                    distot   = np.sqrt(disx**2+disy**2+disz**2)
                    # save the deformed and undeformed points for each subsection
                    for ii_subsec in np.arange(len(triangles)):
                        if section_globalCS.section1[ii_elem][jj_point] == ii_subsec:
                            xplot[ii_subsec].append(defx)
                            yplot[ii_subsec].append(defy)
                            zplot[ii_subsec].append(defz)
                            displot[ii_subsec].append(distot)
                            xplotu[ii_subsec].append(plot_sec[0]+mesh_data.point[ind_point,0])
                            yplotu[ii_subsec].append(plot_sec[1]+mesh_data.point[ind_point,1])
                            zplotu[ii_subsec].append(plot_sec[2]+mesh_data.point[ind_point,2])
            # ind_point : number of the beam node
            # plot_sec  : section nodes to plot
            # plot_cdg  : center of gravity of the section to plot
            # If the second node is not used before calculate the positions of the section points
            if len(np.where(np.array(used_node) == mesh_data.elem[ii_elem,2])[0]) == 0:
                used_node.append(mesh_data.elem[ii_elem,2])
                ind_point = section_globalCS.nodeelem2[ii_elem]
                for jj_point in np.arange(len(section_globalCS.n2[ii_elem])):  
                    # plot_sec : node coordinates
                    # defx     : deformation in coordinate x
                    # defy     : deformation in coordinate y
                    # defz     : deformation in coordinate z
                    plot_sec = section_globalCS.n2[ii_elem][jj_point]
                    defx     = xdef[ind_point]+(plot_sec[0])*np.cos(theta[ind_point])*np.cos(psi[ind_point])-(plot_sec[1])*np.sin(theta[ind_point])+(plot_sec[2])*np.sin(psi[ind_point])+warp_x2[ii_elem][jj_point]
                    defy     = ydef[ind_point]+(plot_sec[1])*np.cos(theta[ind_point])*np.cos(phi[ind_point])+(plot_sec[0])*np.sin(theta[ind_point])-(plot_sec[2])*np.sin(phi[ind_point])+warp_y2[ii_elem][jj_point]
                    defz     = zdef[ind_point]+(plot_sec[2])*np.cos(psi[ind_point])*np.cos(phi[ind_point])-(plot_sec[0])*np.sin(psi[ind_point])+(plot_sec[1])*np.sin(phi[ind_point])+warp_z2[ii_elem][jj_point]
                    disx     = uu[ind_point]+(plot_sec[0])*np.cos(theta[ind_point])*np.cos(psi[ind_point])-(plot_sec[1])*np.sin(theta[ind_point])+(plot_sec[2])*np.sin(psi[ind_point])+warp_x2[ii_elem][jj_point]-plot_sec[0]
                    disy     = vv[ind_point]+(plot_sec[1])*np.cos(theta[ind_point])*np.cos(phi[ind_point])+(plot_sec[0])*np.sin(theta[ind_point])-(plot_sec[2])*np.sin(phi[ind_point])+warp_y2[ii_elem][jj_point]-plot_sec[1]
                    disz     = ww[ind_point]+(plot_sec[2])*np.cos(psi[ind_point])*np.cos(phi[ind_point])-(plot_sec[0])*np.sin(psi[ind_point])+(plot_sec[1])*np.sin(phi[ind_point])+warp_z2[ii_elem][jj_point]-plot_sec[2]
                    distot   = np.sqrt(disx**2+disy**2+disz**2) #abs(theta[ind_point]) #np.sqrt(disx**2+disy**2+disz**2)
                    # save the deformed and undeformed points for each subsection
                    for ii_subsec in np.arange(int(np.max(section_globalCS.nsub))):
                        if section_globalCS.section2[ii_elem][jj_point] == ii_subsec:
                            xplot[ii_subsec].append(defx)
                            yplot[ii_subsec].append(defy)
                            zplot[ii_subsec].append(defz)
                            displot[ii_subsec].append(distot)
                            xplotu[ii_subsec].append(plot_sec[0]+mesh_data.point[ind_point,0])
                            yplotu[ii_subsec].append(plot_sec[1]+mesh_data.point[ind_point,1])
                            zplotu[ii_subsec].append(plot_sec[2]+mesh_data.point[ind_point,2])
        # Choose a colormap for the surface
        # colormap : selected colormap for the surface
        # fig      : figure
        # ax       : axes
        try:
            colormap = cm.get_cmap(plot_ii.color)
        except:
            colormap = cm.get_cmap('viridis')
        if plot_ii.rot == "YES":
            max_x = 0
            max_y = 0
            max_z = 0
            min_x = 0
            min_y = 0
            min_z = 0
            for bound_ii in case_setup.boundary:
                if plot_ii.rot_name == bound_ii.id:
                    nblades = bound_ii.Nb
                    for ii_nb in np.arange(nblades):
                        ang    = -(2*np.pi/nblades*ii_nb+bound_ii.vrot*time_val)
                        xplot_rot  = []
                        yplot_rot  = []
                        zplot_rot  = []
                        xplot_rotu = []
                        yplot_rotu = []
                        zplot_rotu = []
                        nodes_rot  = []
                        v1_rot = bound_ii.refrot/np.linalg.norm(bound_ii.refrot)
                        for ii_plot_surf in np.arange(int(np.max(section_globalCS.nsub))):
                            for jj_plot_surf in np.arange(len(xplot[ii_plot_surf])):
                                point2_rot  = [xplot[ii_plot_surf][jj_plot_surf],yplot[ii_plot_surf][jj_plot_surf],zplot[ii_plot_surf][jj_plot_surf]]
                                point2_rotu = [xplotu[ii_plot_surf][jj_plot_surf],yplotu[ii_plot_surf][jj_plot_surf],zplotu[ii_plot_surf][jj_plot_surf]]
                                point1_rot  = mesh_data.point[bound_ii.refpoint,:]
                                v2_rot      = point2_rot-point1_rot
                                v2_rotu     = point2_rotu-point1_rot
                                mat_rot     = [[np.cos(ang)+v1_rot[0]**2*(1-np.cos(ang)), v1_rot[0]*v1_rot[1]*(1-np.cos(ang))-v1_rot[2]*np.sin(ang), v1_rot[0]*v1_rot[2]*(1-np.cos(ang))+v1_rot[1]*np.sin(ang)],
                                              [v1_rot[0]*v1_rot[1]*(1-np.cos(ang))+v1_rot[2]*np.sin(ang), np.cos(ang)+v1_rot[1]**2*(1-np.cos(ang)), v1_rot[1]*v1_rot[2]*(1-np.cos(ang))-v1_rot[0]*np.sin(ang)],
                                              [v1_rot[0]*v1_rot[2]*(1-np.cos(ang))-v1_rot[1]*np.sin(ang), v1_rot[1]*v1_rot[2]*(1-np.cos(ang))+v1_rot[0]*np.sin(ang), np.cos(ang)+v1_rot[2]**2*(1-np.cos(ang))]]
                                pos_rot    = np.matmul(mat_rot,v2_rot)+point1_rot
                                pos_rotu   = np.matmul(mat_rot,v2_rotu)+point1_rot
                                nodes_rot.append(pos_rot)
                                xplot_rot.append(pos_rot[0])
                                yplot_rot.append(pos_rot[1])
                                zplot_rot.append(pos_rot[2])
                                xplot_rotu.append(pos_rotu[0])
                                yplot_rotu.append(pos_rotu[1])
                                zplot_rotu.append(pos_rotu[2])
                            if plot_ii.dihedral == 'Z':
                                xplotsurf  = np.array(zplot_rot)
                                yplotsurf  = np.array(xplot_rot)
                                zplotsurf  = np.array(yplot_rot)
                                xplotsurfu = np.array(zplot_rotu)
                                yplotsurfu = np.array(xplot_rotu)
                                zplotsurfu = np.array(yplot_rotu)
                            elif plot_ii.dihedral == 'Y':
                                xplotsurf  = np.array(yplot_rot)
                                yplotsurf  = np.array(zplot_rot)
                                zplotsurf  = np.array(xplot_rot)
                                xplotsurfu = np.array(yplot_rotu)
                                yplotsurfu = np.array(zplot_rotu)
                                zplotsurfu = np.array(xplot_rotu)
                            else:
                                xplotsurf  = np.array(xplot_rot)
                                yplotsurf  = np.array(yplot_rot)
                                zplotsurf  = np.array(zplot_rot)
                                xplotsurfu = np.array(xplot_rotu)
                                yplotsurfu = np.array(yplot_rotu)
                                zplotsurfu = np.array(zplot_rotu)
                            verts = np.zeros((len(np.asarray(triangles)[0]),9))
                            faccol = np.zeros((len(np.asarray(triangles)[0]),))
                            for ii_row in np.arange(len(np.asarray(triangles)[0])):
                                ii_row_dat = np.asarray(triangles)[0][ii_row]
                                for jj_col in np.arange(len(ii_row_dat)):
                                    jj_row_dat = int(ii_row_dat[jj_col])
                                    verts[ii_row,3*jj_col]   = xplotsurf[jj_row_dat]
                                    verts[ii_row,3*jj_col+1] = yplotsurf[jj_row_dat]
                                    verts[ii_row,3*jj_col+2] = zplotsurf[jj_row_dat]
                                    faccol[ii_row]          += displot[ii_plot_surf][jj_row_dat]/3
                            color_dimension = faccol
                            color_dimension_tot.append(color_dimension)
                            if mark_aux > 0:
                                color_dimension_join = np.concatenate((color_dimension_join,color_dimension))
                                xplot_join = np.concatenate((xplot_join,xplotsurfu))
                                yplot_join = np.concatenate((yplot_join,yplotsurfu))
                                zplot_join = np.concatenate((zplot_join,zplotsurfu))
                            else:
                                color_dimension_join = color_dimension
                                xplot_join = xplotsurfu
                                yplot_join = yplotsurfu
                                zplot_join = zplotsurfu
                            vert_tot.append(verts)
                            xplotsurfu_tot.append(xplotsurfu)
                            yplotsurfu_tot.append(yplotsurfu)
                            zplotsurfu_tot.append(zplotsurfu)
                        max_x = np.max([np.max([np.max(xplot_join),np.max(xplot_join)]),max_x])
                        max_y = np.max([2*np.max([np.max(yplot_join),np.max(yplot_join)]),max_y])
                        max_z = np.max([2*np.max([np.max(zplot_join),np.max(zplot_join)]),max_z])
                        min_x = np.min([np.min([np.min(xplot_join),np.min(xplot_join)]),min_x])
                        min_y = np.min([2*np.min([np.min(yplot_join),np.min(yplot_join)]),min_y])
                        min_z = np.min([2*np.min([np.min(zplot_join),np.min(zplot_join)]),min_z])
#                            minn, maxx = color_dimension.min(), color_dimension.max()
#                            if minn==maxx and maxx == 0:
#                                maxx = 1
#                            norm = matplotlib.colors.Normalize(minn, maxx)
#                            m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
#                            m.set_array([])
#                            fcolors = m.to_rgba(color_dimension)
#                            poly3d = [[ verts[i, j*3:j*3+3] for j in range(3)  ] for i in range(verts.shape[0])]
#                            ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fcolors[:,:3]))#ax.plot_trisurf(xplotsurf,yplotsurf,zplotsurf,triangles=triangles[ii_plot_surf],cmap=colormap)
#                            ax.plot_trisurf(xplotsurfu,yplotsurfu,zplotsurfu,triangles=triangles[ii_plot_surf],cmap='gray',alpha=0.1)
        else:  
            # plot deformed and undeformed beams
            for ii_plot_surf in np.arange(len(triangles)):#np.arange(int(np.max(section_globalCS.nsub))):
                if plot_ii.dihedral == 'Z':
                    xplotsurf  = np.array(zplot[ii_plot_surf])
                    yplotsurf  = np.array(xplot[ii_plot_surf])
                    zplotsurf  = np.array(yplot[ii_plot_surf])
                    xplotsurfu = np.array(zplotu[ii_plot_surf])
                    yplotsurfu = np.array(xplotu[ii_plot_surf])
                    zplotsurfu = np.array(yplotu[ii_plot_surf])
                elif plot_ii.dihedral == 'Y':
                    xplotsurf  = np.array(yplot[ii_plot_surf])
                    yplotsurf  = np.array(zplot[ii_plot_surf])
                    zplotsurf  = np.array(xplot[ii_plot_surf])
                    xplotsurfu = np.array(yplotu[ii_plot_surf])
                    yplotsurfu = np.array(zplotu[ii_plot_surf])
                    zplotsurfu = np.array(xplotu[ii_plot_surf])
                else:
                    xplotsurf  = np.array(xplot[ii_plot_surf])
                    yplotsurf  = np.array(yplot[ii_plot_surf])
                    zplotsurf  = np.array(zplot[ii_plot_surf])
                    xplotsurfu = np.array(xplotu[ii_plot_surf])
                    yplotsurfu = np.array(yplotu[ii_plot_surf])
                    zplotsurfu = np.array(zplotu[ii_plot_surf])
                verts = np.zeros((len(np.asarray(triangles)[ii_plot_surf]),9))
                faccol = np.zeros((len(np.asarray(triangles)[ii_plot_surf]),))
                for ii_row in np.arange(len(np.asarray(triangles)[ii_plot_surf])):
                    ii_row_dat = np.asarray(triangles)[ii_plot_surf][ii_row]
                    for jj_col in np.arange(len(ii_row_dat)):
                        jj_row_dat = int(ii_row_dat[jj_col])
                        verts[ii_row,3*jj_col]   = xplotsurf[jj_row_dat]
                        verts[ii_row,3*jj_col+1] = yplotsurf[jj_row_dat]
                        verts[ii_row,3*jj_col+2] = zplotsurf[jj_row_dat]
                        faccol[ii_row]          += displot[ii_plot_surf][jj_row_dat]/3
                color_dimension = faccol
                color_dimension_tot.append(color_dimension)
                if mark_aux > 0:
                    color_dimension_join = np.concatenate((color_dimension_join,color_dimension))
                    xplot_join = np.concatenate((xplot_join,xplotsurfu))
                    yplot_join = np.concatenate((yplot_join,yplotsurfu))
                    zplot_join = np.concatenate((zplot_join,zplotsurfu))
                else:
                    color_dimension_join = color_dimension
                    xplot_join = xplotsurfu
                    yplot_join = yplotsurfu
                    zplot_join = zplotsurfu
                vert_tot.append(verts)
                xplotsurfu_tot.append(xplotsurfu)
                yplotsurfu_tot.append(yplotsurfu)
                zplotsurfu_tot.append(zplotsurfu)          
            max_x = np.max([np.max(xplot_join),np.max(xplot_join)])
            max_y = 2*np.max([np.max(yplot_join),np.max(yplot_join)])
            max_z = 2*np.max([np.max(zplot_join),np.max(zplot_join)])
            min_x = np.min([np.min(xplot_join),np.min(xplot_join)])
            min_y = 2*np.min([np.min(yplot_join),np.min(yplot_join)])
            min_z = 2*np.min([np.min(zplot_join),np.min(zplot_join)])
#                if ii_plot_surf == 0:
#                    try:
#                        minn = plot_ii.colmin
#                    except:
#                        minn = color_dimension.min()
#                    try:
#                        maxx = plot_ii.colmax
#                    except:
#                        maxx = color_dimension.max()
#    #                minn, maxx = color_dimension.min(), color_dimension.max()
#                    if minn==maxx and maxx == 0:
#                        maxx = 1
#                norm = matplotlib.colors.Normalize(minn, maxx)
#                m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
#                m.set_array([])
#                fcolors = m.to_rgba(color_dimension)
#                poly3d = [[ verts[i, j*3:j*3+3] for j in range(3)  ] for i in range(verts.shape[0])]
#                datamap = Poly3DCollection(poly3d, facecolors=fcolors[:,:3])
#                ax.add_collection3d(datamap)# ax.plot_trisurf(xplotsurf,yplotsurf,zplotsurf,triangles=triangles[ii_plot_surf],cmap=colormap)
#                ax.plot_trisurf(xplotsurfu,yplotsurfu,zplotsurfu,triangles=triangles[ii_plot_surf],cmap='gray',alpha=0.1)
    try:
        colormap = cm.get_cmap(plot_ii.color)
    except:
        colormap = cm.get_cmap('viridis')
#    minn, maxx = color_dimension_join.min(), color_dimension_join_join.max()
    try:
        minn = plot_ii.colmin
    except:
        minn = color_dimension_join.min()
    try:
        maxx = plot_ii.colmax
    except:
        maxx = color_dimension_join.max()
    if minn==maxx and maxx == 0:
        maxx = 1
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    m.set_array([])
    for mark_aux in np.arange(len(vert_tot)):
        verts = vert_tot[mark_aux]
        color_dimension = color_dimension_tot[mark_aux]
        xplotsurfu = xplotsurfu_tot[mark_aux]
        yplotsurfu = yplotsurfu_tot[mark_aux]
        zplotsurfu = zplotsurfu_tot[mark_aux]
        fcolors = m.to_rgba(color_dimension)
        poly3d = [[ verts[i, j*3:j*3+3] for j in range(3)  ] for i in range(verts.shape[0])]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fcolors[:,:3]))#ax.plot_trisurf(xplotsurf,yplotsurf,zplotsurf,triangles=triangles[ii_plot_surf],cmap=colormap)
#        ax.plot_trisurf(xplotsurfu,yplotsurfu,zplotsurfu,triangles=triangles[ii_plot_surf],cmap='gray',alpha=0.1)
    # equal and invisible axis
#    fig.colorbar(m, orientation='horizontal', label='Displacement (m)')
    ax.set_xlim([min_x,max_x])
    ax.set_ylim([min_y,max_y])
    ax.set_zlim([min_z,max_z])
#    x_middle = []
#    y_middle = []
#    z_middle = []
    x_middle, y_middle, z_middle, plot_radius = func_axes_equal(ax,x_middle, y_middle, z_middle, plot_radius)
    plt.tight_layout()
    plt.axis('off')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
#    ax.view_init(0, 0)
    plt.show()
    # seve the figure image
    try:
        if typeplot==0:
            plt.savefig(case_setup.root + plot_ii.save,bbox_inches='tight')
        elif typeplot==1:
            plt.savefig(case_setup.root + plot_ii.save[:-4] + '_time_'+str(ii_time) + plot_ii.save[-4:],bbox_inches='tight')
            plt.close()
            time.sleep(1)
        elif typeplot==2:
            plt.savefig(case_setup.root + plot_ii.save[:-4] + '_mod_'+str(ii_mode) + plot_ii.save[-4:],bbox_inches='tight')
#            plt.close()
            time.sleep(1)
        elif typeplot==3:
            plt.savefig(case_setup.root + plot_ii.save[:-4] + '_mod_'+str(plot_ii.mode) + '_vel_'+str(ii_time) + plot_ii.save[-4:],bbox_inches='tight')
            plt.close()
            time.sleep(1)
    except:
        pass   
    return x_middle, y_middle, z_middle, plot_radius

def plot_defshapeaero(case_setup,mesh_data,section_globalCS,solution,plot_ii,ii_time,ii_mode,triangles_tot,elements_tot,typeplot,x_middle, y_middle, z_middle, plot_radius,time_val):
    # Function to plot the deformed beam
    # case_setup       : configuration of the case
    # mesh_data        : information of the mesh
    # section_globalCS : section global coordinates
    # solution         : information of the solution
    # plot_ii          : information of the plot
    # ii_time          : time step or mode
    # ------------------------------------------------------------------------
    # phi   : angle axis x
    # psi   : angle axis y
    # theta : angle axis z
    # warp_x1 : warping in the x coordinate of node 1
    # warp_y1 : warping in the y coordinate of node 1
    # warp_z1 : warping in the z coordinate of node 1
    # warp_x2 : warping in the x coordinate of node 2
    # warp_y2 : warping in the y coordinate of node 2
    # warp_z2 : warping in the z coordinate of node 2
    if typeplot == 0:
        xdef    = solution.xdef
        ydef    = solution.ydef
        zdef    = solution.zdef
        fx    = solution.ext_fx
        fy    = solution.ext_fy
        fz    = solution.ext_fz
    # For all the elements in the mesh take the nodes that have not been used
    # used_node : used nodes
    fig = plt.figure(plot_ii.fig)
    ax  = fig.add_subplot(111, projection='3d')
    color_dimension_tot = []
    vert_tot = []
    xplotsurfu_tot = []
    yplotsurfu_tot = []
    zplotsurfu_tot = []
    for mark_aux in np.arange(len(triangles_tot)):
        used_node = []
        # Take the values of x, y and z in the nodes of the marker
        # xplot  : matrix of values of the x coordinate
        # yplot  : matrix of values of the y coordinate
        # zplot  : matrix of values of the z coordinate
        # xplotu : matrix of values of the undeformed x coordinate
        # yplotu : matrix of values of the undeformed y coordinate
        # zplotu : matrix of values of the undeformed z coordinate
        triangles = triangles_tot[mark_aux]
        xplot   = def_vec_param(len(triangles))
        yplot   = def_vec_param(len(triangles))
        zplot   = def_vec_param(len(triangles))
        xplotu  = def_vec_param(len(triangles))
        yplotu  = def_vec_param(len(triangles))
        zplotu  = def_vec_param(len(triangles))
        displot = def_vec_param(len(triangles))
        for ii_elem in elements_tot[mark_aux]:
            # ind_point : number of the beam node
            # plot_sec  : section nodes to plot
            # plot_cdg  : center of gravity of the section to plot
            # If the first node is not used before calculate the positions of the section points
            if len(np.where(np.array(used_node) == mesh_data.elem[ii_elem,1])[0]) == 0:
                used_node.append(mesh_data.elem[ii_elem,1])
                ind_point = section_globalCS.nodeelem1[ii_elem]
                for jj_point in np.arange(len(section_globalCS.n1[ii_elem])):  
                    # plot_sec : node coordinates
                    # defx     : deformation in coordinate x
                    # defy     : deformation in coordinate y
                    # defz     : deformation in coordinate z
                    plot_sec = section_globalCS.n1[ii_elem][jj_point]
                    defx     = xdef[ind_point]+plot_sec[0]
                    defy     = ydef[ind_point]+plot_sec[1]
                    defz     = zdef[ind_point]+plot_sec[2]
                    forcex   = fx[ind_point]
                    forcey   = fy[ind_point]
                    forcez   = fz[ind_point]
                    force   = np.sqrt(forcex**2+forcey**2+forcez**2)
                    # save the deformed and undeformed points for each subsection
                    for ii_subsec in np.arange(len(triangles)):
                        if section_globalCS.section1[ii_elem][jj_point] == ii_subsec:
                            xplot[ii_subsec].append(defx)
                            yplot[ii_subsec].append(defy)
                            zplot[ii_subsec].append(defz)
                            displot[ii_subsec].append(force)
                            xplotu[ii_subsec].append(plot_sec[0]+mesh_data.point[ind_point,0])
                            yplotu[ii_subsec].append(plot_sec[1]+mesh_data.point[ind_point,1])
                            zplotu[ii_subsec].append(plot_sec[2]+mesh_data.point[ind_point,2])
            # ind_point : number of the beam node
            # plot_sec  : section nodes to plot
            # plot_cdg  : center of gravity of the section to plot
            # If the second node is not used before calculate the positions of the section points
            if len(np.where(np.array(used_node) == mesh_data.elem[ii_elem,2])[0]) == 0:
                used_node.append(mesh_data.elem[ii_elem,2])
                ind_point = section_globalCS.nodeelem2[ii_elem]
                for jj_point in np.arange(len(section_globalCS.n2[ii_elem])):  
                    # plot_sec : node coordinates
                    # defx     : deformation in coordinate x
                    # defy     : deformation in coordinate y
                    # defz     : deformation in coordinate z
                    plot_sec = section_globalCS.n2[ii_elem][jj_point]
                    defx     = xdef[ind_point]+plot_sec[0]
                    defy     = ydef[ind_point]+plot_sec[1]
                    defz     = zdef[ind_point]+plot_sec[2]
                    forcex   = fx[ind_point]
                    forcey   = fy[ind_point]
                    forcez   = fz[ind_point]
                    force   = np.sqrt(forcex**2+forcey**2+forcez**2)
                    # save the deformed and undeformed points for each subsection
                    for ii_subsec in np.arange(int(np.max(section_globalCS.nsub))):
                        if section_globalCS.section2[ii_elem][jj_point] == ii_subsec:
                            xplot[ii_subsec].append(defx)
                            yplot[ii_subsec].append(defy)
                            zplot[ii_subsec].append(defz)
                            displot[ii_subsec].append(force)
                            xplotu[ii_subsec].append(plot_sec[0]+mesh_data.point[ind_point,0])
                            yplotu[ii_subsec].append(plot_sec[1]+mesh_data.point[ind_point,1])
                            zplotu[ii_subsec].append(plot_sec[2]+mesh_data.point[ind_point,2])
        # Choose a colormap for the surface
        # colormap : selected colormap for the surface
        # fig      : figure
        # ax       : axes
        try:
            colormap = cm.get_cmap(plot_ii.color)
        except:
            colormap = cm.get_cmap('viridis')
        if plot_ii.rot == "YES":
            max_x = 0
            max_y = 0
            max_z = 0
            min_x = 0
            min_y = 0
            min_z = 0
            for bound_ii in case_setup.boundary:
                if plot_ii.rot_name == bound_ii.id:
                    nblades = bound_ii.Nb
                    for ii_nb in np.arange(nblades):
                        ang    = -(2*np.pi/nblades*ii_nb+bound_ii.vrot*time_val)
                        xplot_rot  = []
                        yplot_rot  = []
                        zplot_rot  = []
                        xplot_rotu = []
                        yplot_rotu = []
                        zplot_rotu = []
                        nodes_rot  = []
                        v1_rot = bound_ii.refrot/np.linalg.norm(bound_ii.refrot)
                        for ii_plot_surf in np.arange(int(np.max(section_globalCS.nsub))):
                            for jj_plot_surf in np.arange(len(xplot[ii_plot_surf])):
                                point2_rot  = [xplot[ii_plot_surf][jj_plot_surf],yplot[ii_plot_surf][jj_plot_surf],zplot[ii_plot_surf][jj_plot_surf]]
                                point2_rotu = [xplotu[ii_plot_surf][jj_plot_surf],yplotu[ii_plot_surf][jj_plot_surf],zplotu[ii_plot_surf][jj_plot_surf]]
                                point1_rot  = mesh_data.point[bound_ii.refpoint,:]
                                v2_rot      = point2_rot-point1_rot
                                v2_rotu     = point2_rotu-point1_rot
                                mat_rot     = [[np.cos(ang)+v1_rot[0]**2*(1-np.cos(ang)), v1_rot[0]*v1_rot[1]*(1-np.cos(ang))-v1_rot[2]*np.sin(ang), v1_rot[0]*v1_rot[2]*(1-np.cos(ang))+v1_rot[1]*np.sin(ang)],
                                              [v1_rot[0]*v1_rot[1]*(1-np.cos(ang))+v1_rot[2]*np.sin(ang), np.cos(ang)+v1_rot[1]**2*(1-np.cos(ang)), v1_rot[1]*v1_rot[2]*(1-np.cos(ang))-v1_rot[0]*np.sin(ang)],
                                              [v1_rot[0]*v1_rot[2]*(1-np.cos(ang))-v1_rot[1]*np.sin(ang), v1_rot[1]*v1_rot[2]*(1-np.cos(ang))+v1_rot[0]*np.sin(ang), np.cos(ang)+v1_rot[2]**2*(1-np.cos(ang))]]
                                pos_rot    = np.matmul(mat_rot,v2_rot)+point1_rot
                                pos_rotu   = np.matmul(mat_rot,v2_rotu)+point1_rot
                                nodes_rot.append(pos_rot)
                                xplot_rot.append(pos_rot[0])
                                yplot_rot.append(pos_rot[1])
                                zplot_rot.append(pos_rot[2])
                                xplot_rotu.append(pos_rotu[0])
                                yplot_rotu.append(pos_rotu[1])
                                zplot_rotu.append(pos_rotu[2])
                            if plot_ii.dihedral == 'Z':
                                xplotsurf  = np.array(zplot_rot)
                                yplotsurf  = np.array(xplot_rot)
                                zplotsurf  = np.array(yplot_rot)
                                xplotsurfu = np.array(zplot_rotu)
                                yplotsurfu = np.array(xplot_rotu)
                                zplotsurfu = np.array(yplot_rotu)
                            elif plot_ii.dihedral == 'Y':
                                xplotsurf  = np.array(yplot_rot)
                                yplotsurf  = np.array(zplot_rot)
                                zplotsurf  = np.array(xplot_rot)
                                xplotsurfu = np.array(yplot_rotu)
                                yplotsurfu = np.array(zplot_rotu)
                                zplotsurfu = np.array(xplot_rotu)
                            else:
                                xplotsurf  = np.array(xplot_rot)
                                yplotsurf  = np.array(yplot_rot)
                                zplotsurf  = np.array(zplot_rot)
                                xplotsurfu = np.array(xplot_rotu)
                                yplotsurfu = np.array(yplot_rotu)
                                zplotsurfu = np.array(zplot_rotu)
                            verts = np.zeros((len(np.asarray(triangles)[0]),9))
                            faccol = np.zeros((len(np.asarray(triangles)[0]),))
                            for ii_row in np.arange(len(np.asarray(triangles)[0])):
                                ii_row_dat = np.asarray(triangles)[0][ii_row]
                                for jj_col in np.arange(len(ii_row_dat)):
                                    jj_row_dat = int(ii_row_dat[jj_col])
                                    verts[ii_row,3*jj_col]   = xplotsurf[jj_row_dat]
                                    verts[ii_row,3*jj_col+1] = yplotsurf[jj_row_dat]
                                    verts[ii_row,3*jj_col+2] = zplotsurf[jj_row_dat]
                                    faccol[ii_row]          += displot[ii_plot_surf][jj_row_dat]/3
                            color_dimension = faccol
                            color_dimension_tot.append(color_dimension)
                            if mark_aux > 0:
                                color_dimension_join = np.concatenate((color_dimension_join,color_dimension))
                                xplot_join = np.concatenate((xplot_join,xplotsurfu))
                                yplot_join = np.concatenate((yplot_join,yplotsurfu))
                                zplot_join = np.concatenate((zplot_join,zplotsurfu))
                            else:
                                color_dimension_join = color_dimension
                                xplot_join = xplotsurfu
                                yplot_join = yplotsurfu
                                zplot_join = zplotsurfu
                            vert_tot.append(verts)
                            xplotsurfu_tot.append(xplotsurfu)
                            yplotsurfu_tot.append(yplotsurfu)
                            zplotsurfu_tot.append(zplotsurfu)
                        max_x = np.max([np.max([np.max(xplot_join),np.max(xplot_join)]),max_x])
                        max_y = np.max([2*np.max([np.max(yplot_join),np.max(yplot_join)]),max_y])
                        max_z = np.max([2*np.max([np.max(zplot_join),np.max(zplot_join)]),max_z])
                        min_x = np.min([np.min([np.min(xplot_join),np.min(xplot_join)]),min_x])
                        min_y = np.min([2*np.min([np.min(yplot_join),np.min(yplot_join)]),min_y])
                        min_z = np.min([2*np.min([np.min(zplot_join),np.min(zplot_join)]),min_z])
#                            minn, maxx = color_dimension.min(), color_dimension.max()
#                            if minn==maxx and maxx == 0:
#                                maxx = 1
#                            norm = matplotlib.colors.Normalize(minn, maxx)
#                            m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
#                            m.set_array([])
#                            fcolors = m.to_rgba(color_dimension)
#                            poly3d = [[ verts[i, j*3:j*3+3] for j in range(3)  ] for i in range(verts.shape[0])]
#                            ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fcolors[:,:3]))#ax.plot_trisurf(xplotsurf,yplotsurf,zplotsurf,triangles=triangles[ii_plot_surf],cmap=colormap)
#                            ax.plot_trisurf(xplotsurfu,yplotsurfu,zplotsurfu,triangles=triangles[ii_plot_surf],cmap='gray',alpha=0.1)
        else:  
            # plot deformed and undeformed beams
            for ii_plot_surf in np.arange(len(triangles)):#np.arange(int(np.max(section_globalCS.nsub))):
                if plot_ii.dihedral == 'Z':
                    xplotsurf  = np.array(zplot[ii_plot_surf])
                    yplotsurf  = np.array(xplot[ii_plot_surf])
                    zplotsurf  = np.array(yplot[ii_plot_surf])
                    xplotsurfu = np.array(zplotu[ii_plot_surf])
                    yplotsurfu = np.array(xplotu[ii_plot_surf])
                    zplotsurfu = np.array(yplotu[ii_plot_surf])
                elif plot_ii.dihedral == 'Y':
                    xplotsurf  = np.array(yplot[ii_plot_surf])
                    yplotsurf  = np.array(zplot[ii_plot_surf])
                    zplotsurf  = np.array(xplot[ii_plot_surf])
                    xplotsurfu = np.array(yplotu[ii_plot_surf])
                    yplotsurfu = np.array(zplotu[ii_plot_surf])
                    zplotsurfu = np.array(xplotu[ii_plot_surf])
                else:
                    xplotsurf  = np.array(xplot[ii_plot_surf])
                    yplotsurf  = np.array(yplot[ii_plot_surf])
                    zplotsurf  = np.array(zplot[ii_plot_surf])
                    xplotsurfu = np.array(xplotu[ii_plot_surf])
                    yplotsurfu = np.array(yplotu[ii_plot_surf])
                    zplotsurfu = np.array(zplotu[ii_plot_surf])
                verts = np.zeros((len(np.asarray(triangles)[ii_plot_surf]),9))
                faccol = np.zeros((len(np.asarray(triangles)[ii_plot_surf]),))
                for ii_row in np.arange(len(np.asarray(triangles)[ii_plot_surf])):
                    ii_row_dat = np.asarray(triangles)[ii_plot_surf][ii_row]
                    for jj_col in np.arange(len(ii_row_dat)):
                        jj_row_dat = int(ii_row_dat[jj_col])
                        verts[ii_row,3*jj_col]   = xplotsurf[jj_row_dat]
                        verts[ii_row,3*jj_col+1] = yplotsurf[jj_row_dat]
                        verts[ii_row,3*jj_col+2] = zplotsurf[jj_row_dat]
                        faccol[ii_row]          += displot[ii_plot_surf][jj_row_dat]/3
                color_dimension = faccol
                color_dimension_tot.append(color_dimension)
                if mark_aux > 0:
                    color_dimension_join = np.concatenate((color_dimension_join,color_dimension))
                    xplot_join = np.concatenate((xplot_join,xplotsurfu))
                    yplot_join = np.concatenate((yplot_join,yplotsurfu))
                    zplot_join = np.concatenate((zplot_join,zplotsurfu))
                else:
                    color_dimension_join = color_dimension
                    xplot_join = xplotsurfu
                    yplot_join = yplotsurfu
                    zplot_join = zplotsurfu
                vert_tot.append(verts)
                xplotsurfu_tot.append(xplotsurfu)
                yplotsurfu_tot.append(yplotsurfu)
                zplotsurfu_tot.append(zplotsurfu)          
            max_x = np.max([np.max(xplot_join),np.max(xplot_join)])
            max_y = 2*np.max([np.max(yplot_join),np.max(yplot_join)])
            max_z = 2*np.max([np.max(zplot_join),np.max(zplot_join)])
            min_x = np.min([np.min(xplot_join),np.min(xplot_join)])
            min_y = 2*np.min([np.min(yplot_join),np.min(yplot_join)])
            min_z = 2*np.min([np.min(zplot_join),np.min(zplot_join)])
#                if ii_plot_surf == 0:
#                    try:
#                        minn = plot_ii.colmin
#                    except:
#                        minn = color_dimension.min()
#                    try:
#                        maxx = plot_ii.colmax
#                    except:
#                        maxx = color_dimension.max()
#    #                minn, maxx = color_dimension.min(), color_dimension.max()
#                    if minn==maxx and maxx == 0:
#                        maxx = 1
#                norm = matplotlib.colors.Normalize(minn, maxx)
#                m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
#                m.set_array([])
#                fcolors = m.to_rgba(color_dimension)
#                poly3d = [[ verts[i, j*3:j*3+3] for j in range(3)  ] for i in range(verts.shape[0])]
#                datamap = Poly3DCollection(poly3d, facecolors=fcolors[:,:3])
#                ax.add_collection3d(datamap)# ax.plot_trisurf(xplotsurf,yplotsurf,zplotsurf,triangles=triangles[ii_plot_surf],cmap=colormap)
#                ax.plot_trisurf(xplotsurfu,yplotsurfu,zplotsurfu,triangles=triangles[ii_plot_surf],cmap='gray',alpha=0.1)
    try:
        colormap = cm.get_cmap(plot_ii.color)
    except:
        colormap = cm.get_cmap('viridis')
#    minn, maxx = color_dimension_join.min(), color_dimension_join_join.max()
    try:
        minn = plot_ii.colmin
    except:
        minn = color_dimension_join.min()
    try:
        maxx = plot_ii.colmax
    except:
        maxx = color_dimension_join.max()
    if minn==maxx and maxx == 0:
        maxx = 1
    norm = matplotlib.colors.Normalize(minn, maxx)
    m = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    m.set_array([])
    for mark_aux in np.arange(len(vert_tot)):
        verts = vert_tot[mark_aux]
        color_dimension = color_dimension_tot[mark_aux]
        xplotsurfu = xplotsurfu_tot[mark_aux]
        yplotsurfu = yplotsurfu_tot[mark_aux]
        zplotsurfu = zplotsurfu_tot[mark_aux]
        fcolors = m.to_rgba(color_dimension)
        poly3d = [[ verts[i, j*3:j*3+3] for j in range(3)  ] for i in range(verts.shape[0])]
        ax.add_collection3d(Poly3DCollection(poly3d, facecolors=fcolors[:,:3]))#ax.plot_trisurf(xplotsurf,yplotsurf,zplotsurf,triangles=triangles[ii_plot_surf],cmap=colormap)
#        ax.plot_trisurf(xplotsurfu,yplotsurfu,zplotsurfu,triangles=triangles[ii_plot_surf],cmap='gray',alpha=0.1)
    # equal and invisible axis
#    fig.colorbar(m, orientation='horizontal', label='Displacement (m)')
    ax.set_xlim([min_x,max_x])
    ax.set_ylim([min_y,max_y])
    ax.set_zlim([min_z,max_z])
    x_middle, y_middle, z_middle, plot_radius = func_axes_equal(ax,x_middle, y_middle, z_middle, plot_radius)
    plt.tight_layout()
    plt.axis('off')
    plt.show()
    # seve the figure image
    try:
        if typeplot==0:
            plt.savefig(case_setup.root + plot_ii.save,bbox_inches='tight')
        elif typeplot==1:
            plt.savefig(case_setup.root + plot_ii.save[:-4] + '_time_'+str(ii_time) + plot_ii.save[-4:],bbox_inches='tight')
            plt.close()
            time.sleep(1)
        elif typeplot==2:
            plt.savefig(case_setup.root + plot_ii.save[:-4] + '_mod_'+str(ii_mode) + plot_ii.save[-4:],bbox_inches='tight')
#            plt.close()
            time.sleep(1)
        elif typeplot==3:
            plt.savefig(case_setup.root + plot_ii.save[:-4] + '_mod_'+str(ii_mode) + '_vel_'+str(ii_time) + plot_ii.save[-4:],bbox_inches='tight')
            plt.close()
            time.sleep(1)
    except:
        pass   
    return x_middle, y_middle, z_middle, plot_radius

#%%
def postproc_aero(case_setup,mesh_data,solution, section_globalCS):
    # postprocess of the stationary solution
    # case_setup       : class containing the setup of the simulation
    # mesh_data        : class containing the mesh information
    # solution         : class containing the solution of the simulation
    # section_globalCS : class containing the information of all the nodes of the different subsections
    # -------------------------------------------------------------------------
    # For every postprocessing option find the nodes of the marker
    # Find the typology of plot
    #    DEF_SHAPE  : deformed shape plot
    for plot_ii in case_setup.plots:
        # If the 2D plot option is selected
        if plot_ii.typeplot == 'XYPLOT':
            for mark in mesh_data.marker:
                try:
                    # If a marker is specified
                    if plot_ii.marker == mark.name:
                        # Take the values of x and y in the nodes of the marker
                        # xplot     : vector of the x axis values
                        # yplot     : vector of the y axis values
                        # yplot2    : vector of the y axis values
                        # flag2axis : flag to activate the second axis
                        xplot     = np.zeros((len(mark.node),))
                        yplot     = np.zeros((len(mark.node),))
                        yplot2    = np.zeros((len(mark.node),))
                        flag2axis = 0
                        for ii_node in np.arange(len(mark.node)):
                            xplot[int(ii_node)] = np.dot(mesh_data.point[int(mark.node[ii_node]),:],plot_ii.xvalues)
                            yplot[int(ii_node)] = yvalue_select(plot_ii.yaxis,solution,int(mark.node[ii_node]),0,[],0)
                            try:
                                yplot2[int(ii_node)] = yvalue_select(plot_ii.yaxis2,solution,int(mark.node[ii_node]),0,[],0)
                            except:
                                pass
                        # sort the index of the list
                        # index : sorted index of the plot vectors
                        index  = [ind[0] for ind in sorted(enumerate(xplot), key=lambda xx:xx[1])]
                        xplot  = xplot[index]
                        yplot  = yplot[index]
                        yplot2 = yplot2[index]
                        # if the second axis is called activate the flag
                        if abs(sum(yplot2)) > 0:
                            flag2axis = 1
                        plot_2dxy(case_setup,plot_ii,xplot,yplot,yplot2,flag2axis,[])
                except:
                    pass
        # If the deformation shape plot is chosen
        elif plot_ii.typeplot == "DEF_SHAPE":
            triangles_tot,elements_tot = plot_defshapemesh(mesh_data,section_globalCS,solution,plot_ii)
            plot_defshapeaero(case_setup,mesh_data,section_globalCS,solution,plot_ii,[],[],triangles_tot,elements_tot,0,[],[],[],[],0)
    return
#%%
def postproc_stat(case_setup,mesh_data,solution, section_globalCS):
    # postprocess of the stationary solution
    # case_setup       : class containing the setup of the simulation
    # mesh_data        : class containing the mesh information
    # solution         : class containing the solution of the simulation
    # section_globalCS : class containing the information of all the nodes of the different subsections
    # -------------------------------------------------------------------------
    # For every postprocessing option find the nodes of the marker
    # Find the typology of plot
    #    DEF_SHAPE  : deformed shape plot
    for plot_ii in case_setup.plots:
        # If the 2D plot option is selected
        if plot_ii.typeplot == 'XYPLOT':
            for mark in mesh_data.marker:
                try:
                    # If a marker is specified
                    if plot_ii.marker == mark.name:
                        # Take the values of x and y in the nodes of the marker
                        # xplot     : vector of the x axis values
                        # yplot     : vector of the y axis values
                        # yplot2    : vector of the y axis values
                        # flag2axis : flag to activate the second axis
                        xplot     = np.zeros((len(mark.node),))
                        yplot     = np.zeros((len(mark.node),))
                        yplot2    = np.zeros((len(mark.node),))
                        flag2axis = 0
                        for ii_node in np.arange(len(mark.node)):
                            xplot[int(ii_node)] = np.dot(mesh_data.point[int(mark.node[ii_node]),:],plot_ii.xvalues)
                            yplot[int(ii_node)] = yvalue_select(plot_ii.yaxis,solution,int(mark.node[ii_node]),0,[],0)
                            try:
                                yplot2[int(ii_node)] = yvalue_select(plot_ii.yaxis2,solution,int(mark.node[ii_node]),0,[],0)
                            except:
                                pass
                        # sort the index of the list
                        # index : sorted index of the plot vectors
                        index  = [ind[0] for ind in sorted(enumerate(xplot), key=lambda xx:xx[1])]
                        xplot  = xplot[index]
                        yplot  = yplot[index]
                        yplot2 = yplot2[index]
                        # if the second axis is called activate the flag
                        if abs(sum(yplot2)) > 0:
                            flag2axis = 1
                        plot_2dxy(case_setup,plot_ii,xplot,yplot,yplot2,flag2axis,[])
                except:
                    pass
        # If the deformation shape plot is chosen
        elif plot_ii.typeplot == "DEF_SHAPE":
            triangles_tot,elements_tot = plot_defshapemesh(mesh_data,section_globalCS,solution,plot_ii)
            plot_defshape(case_setup,mesh_data,section_globalCS,solution,plot_ii,[],[],triangles_tot,elements_tot,0,[],[],[],[],0)
    return

#%% 
def postproc_dyn(case_setup,mesh_data,solution, section_globalCS):
    # postprocess of the dynamic solution
    # case_setup       : class containing the setup of the simulation
    # mesh_data        : class containing the mesh information
    # solution         : class containing the solution of the simulation
    # section_globalCS : class containing the information of all the nodes of the different subsections
    # postprocess of the stationary solution
    # case_setup       : class containing the setup of the simulation
    # mesh_data        : class containing the mesh information
    # solution         : class containing the solution of the simulation
    # section_globalCS : class containing the information of all the nodes of the different subsections
    # -------------------------------------------------------------------------
    # For every postprocessing option find the nodes of the marker
    # Find the typology of plot
    #    DEF_SHAPE  : deformed shape plot
    for plot_ii in case_setup.plots:
        # If the 2D plot option is selected
        if plot_ii.typeplot == 'XYPLOT':
            for mark in mesh_data.marker:
                try:
                    # If a marker is specified
                    if plot_ii.yaxis == "MOD_INF":
                        xplot     = np.zeros((len(mark.node),))
                        yplot     = []
                        yplot2     = []
                        flag2axis = 0
                        for ii_mode in np.arange(case_setup.n_mod):
                            xplot = solution.time
                            yplot.append(yvalue_select(plot_ii.yaxis,solution,int(ii_mode),[],[],1))
                            try:
                                yplot2.append(yvalue_select(plot_ii.yaxis2,solution,int(ii_mode),[],[],1))
                            except:
                                pass
                            # if the second axis is called activate the flag
                            if len(yplot2) > 0:
                                flag2axis = 1
                        plot_2dxy_multiple(case_setup,plot_ii,xplot,yplot,yplot2,flag2axis,case_setup.n_mod,mesh_data,mark.node)
                    elif plot_ii.marker == mark.name:
                        if plot_ii.type == 'TIME':
                            # Take the values of x and y in the nodes of the marker
                            # xplot     : vector of the x axis values
                            # yplot     : vector of the y axis values
                            # yplot2    : vector of the y axis values
                            # flag2axis : flag to activate the second axis
                            xplot     = np.zeros((len(mark.node),))
                            yplot     = []
                            yplot2    = []
                            flag2axis = 0
                            for ii_node in np.arange(len(mark.node)):
                                xplot = solution.time
                                yplot.append(yvalue_select(plot_ii.yaxis,solution,int(mark.node[ii_node][0]),[],[],1))
                                try:
                                    yplot2.append(yvalue_select(plot_ii.yaxis2,solution,int(mark.node[ii_node]),[],[],1))
                                except:
                                    pass
                                # if the second axis is called activate the flag
                                if len(yplot2) > 0:
                                    flag2axis = 1
                            plot_2dxy_multiple(case_setup,plot_ii,xplot,yplot,yplot2,flag2axis,len(mark.node),mesh_data,mark.node)
                        elif plot_ii.type == 'SPACE':
                            # Take the values of x and y in the nodes of the marker
                            # xplot     : vector of the x axis values
                            # yplot     : vector of the y axis values
                            # yplot2    : vector of the y axis values
                            # flag2axis : flag to activate the second axis
                            xplot     = np.zeros((len(mark.node),))
                            yplot     = np.zeros((len(mark.node),))
                            yplot2    = np.zeros((len(mark.node),))
                            flag2axis = 0
                            for ii_time in np.arange(len(solution.time)):
                                for ii_node in np.arange(len(mark.node)):
                                    xplot[int(ii_node)] = np.dot(mesh_data.point[int(mark.node[ii_node]),:],plot_ii.xvalues)
                                    yplot[int(ii_node)] = yvalue_select(plot_ii.yaxis,solution,int(mark.node[ii_node][0]),ii_time,[],1)
                                    try:
                                        yplot2[int(ii_node)] = yvalue_select(plot_ii.yaxis2,solution,int(mark.node[ii_node][0]),ii_time,[],1)
                                    except:
                                        pass
                                # sort the index of the list
                                # index : sorted index of the plot vectors
                                index  = [ind[0] for ind in sorted(enumerate(xplot), key=lambda xx:xx[1])]
                                xplot  = xplot[index]
                                yplot  = yplot[index]
                                yplot2 = yplot2[index]
                                # if the second axis is called activate the flag
                                if abs(sum(yplot2)) > 0:
                                    flag2axis = 1
                                plot_2dxy(case_setup,plot_ii,xplot,yplot,yplot2,flag2axis,ii_time)
                except:
                    pass
                # If the deformation shape plot is chosen
        elif plot_ii.typeplot == "DEF_SHAPE":
            triangles_tot,elements_tot = plot_defshapemesh(mesh_data,section_globalCS,solution,plot_ii)
            x_middle = []
            y_middle = []
            z_middle = []
            plot_radius = []
            for ii_time in np.arange(len(solution.time)):
                x_middle, y_middle, z_middle, plot_radius = plot_defshape(case_setup,mesh_data,section_globalCS,solution,plot_ii,ii_time,[],triangles_tot,elements_tot,1,x_middle, y_middle, z_middle, plot_radius,solution.time[ii_time])
    return

#%%
def postproc_vib(case_setup,mesh_data,solution, section_globalCS):
    # postprocess of the stationary solution
    # case_setup       : class containing the setup of the simulation
    # mesh_data        : class containing the mesh information
    # solution         : class containing the solution of the simulation
    # section_globalCS : class containing the information of all the nodes of the different subsections
    # -------------------------------------------------------------------------
    # For every postprocessing option find the nodes of the marker
    # Find the typology of plot
    #    DEF_SHAPE  : deformed shape plot
    for plot_ii in case_setup.plots:
        # If the 2D plot option is selected
        if plot_ii.typeplot == 'XYPLOT':
            for mark in mesh_data.marker:
                try:
                    # If a marker is specified
                    if plot_ii.marker == mark.name:
                        # Take the values of x and y in the nodes of the marker
                        # xplot     : vector of the x axis values
                        # yplot     : vector of the y axis values
                        # yplot2    : vector of the y axis values
                        # flag2axis : flag to activate the second axis
                        xplot     = np.zeros((len(mark.node),))
                        yplot     = np.zeros((len(mark.node),))
                        yplot2    = np.zeros((len(mark.node),))
                        flag2axis = 0
                        for ii_node in np.arange(len(mark.node)):
                            xplot[int(ii_node)] = np.dot(mesh_data.point[int(mark.node[ii_node]),:],plot_ii.xvalues)
                            yplot[int(ii_node)] = yvalue_select(plot_ii.yaxis,solution,int(mark.node[ii_node][0]),0,plot_ii.mode,2)
                            try:
                                yplot2[int(ii_node)] = yvalue_select(plot_ii.yaxis2,solution,int(mark.node[ii_node][0]),0,plot_ii.mode,2)
                            except:
                                pass
                        # sort the index of the list
                        # index : sorted index of the plot vectors
                        index  = [ind[0] for ind in sorted(enumerate(xplot), key=lambda xx:xx[1])]
                        xplot  = xplot[index]
                        yplot  = yplot[index]
                        yplot2 = yplot2[index]
                        # if the second axis is called activate the flag
                        if abs(sum(yplot2)) > 0:
                            flag2axis = 1
                        plot_2dxy(case_setup,plot_ii,xplot,yplot,yplot2,flag2axis,[])
                except:
                    pass
        # If the deformation shape plot is chosen
        elif plot_ii.typeplot == "DEF_SHAPE":
            triangles_tot,elements_tot = plot_defshapemesh(mesh_data,section_globalCS,solution,plot_ii)
            plot_defshape(case_setup,mesh_data,section_globalCS,solution,plot_ii,[],plot_ii.mode,triangles_tot,elements_tot,2,[],[],[],[],0)
    return

#%%
def postproc_aelmod(case_setup,mesh_data,solution, section_globalCS):
    iimodelist = -np.ones((case_setup.n_mod,case_setup.numv))
    for plot_ii in case_setup.plots:
        # If the 2D plot option is selected
        if plot_ii.typeplot == 'XYPLOT':
            if plot_ii.type == 'MODAL':
                num_modes = case_setup.n_mod
                if plot_ii.zaxis == 'VEL':
                    ex_loop_lim = case_setup.numv
                    in_loop_lim = num_modes
                elif plot_ii.zaxis == 'MOD':
                    ex_loop_lim = num_modes
                    in_loop_lim = case_setup.numv
                xplot     = np.zeros((num_modes,case_setup.numv))
                yplot     = np.zeros((num_modes,case_setup.numv))
                data_plot = pd.DataFrame()
                legendline = []
                legendlabel = []
                ii_color = 0
                for ii_ex in np.arange(ex_loop_lim):
                    if plot_ii.zaxis == 'MOD':
                        ii_mode_ant = 0
                        ii_mode_ant2 = 0
                    for ii_in in np.arange(in_loop_lim):
                        if plot_ii.zaxis == 'VEL':
                            indflag = 1
                            ii_mode = int(in_loop_lim-ii_in-1)
                            ii_vinf = int(ii_ex)
                            ii_count = ii_in
                        elif plot_ii.zaxis == 'MOD':
                            indflag = 0
                            ii_mode = int(ex_loop_lim-ii_ex-1)
                            ii_vinf = int(ii_in)
                            ii_count = ii_ex
#                        if ii_vinf > 0:
#                            ii_v_bef = ii_vinf-1
#                        else:
#                            ii_v_bef = ii_vinf
                        xp,ii_mode_ant,ii_mode_ant2,indflag  = mvalue_select(plot_ii.xaxis,solution,ii_vinf,ii_mode,ii_mode_ant,ii_mode_ant2,indflag,iimodelist[:,int(ii_in)],ii_count)
                        yp,ii_mode_ant,ii_mode_ant2,indflag = mvalue_select(plot_ii.yaxis,solution,ii_vinf,ii_mode,ii_mode_ant,ii_mode_ant2,indflag,iimodelist[:,int(ii_in)],ii_count)
                        iimodelist[int(ii_ex),int(ii_in)] = ii_mode_ant
                        xplot[int(ii_ex),int(ii_in)] = xp 
                        yplot[int(ii_ex),int(ii_in)] = yp
                    data_plot,legendline,legendlabel = plot_2dxy_mod(case_setup,plot_ii,xplot[int(ii_ex),:],yplot[int(ii_ex),:],ii_color,num_modes,data_plot,legendline,legendlabel,ii_mode,solution.vinf[ii_vinf])
                    ii_color += 1
                    
        # If the deformation shape plot is chosen
        elif plot_ii.typeplot == "DEF_SHAPE":
            triangles_tot,elements_tot = plot_defshapemesh(mesh_data,section_globalCS,solution,plot_ii)
            for ii_vel in np.arange(case_setup.numv):
                if np.min(iimodelist) != -1:
                    ii_mode = iimodelist[-1-plot_ii.mode,ii_vel]
                else:
                    ii_mode = -1-plot_ii.mode
                plot_defshape(case_setup,mesh_data,section_globalCS,solution,plot_ii,ii_vel,ii_mode,triangles_tot,elements_tot,3,[],[],[],[],0)
    return