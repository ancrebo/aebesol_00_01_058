# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 19:31:48 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
save_data      : file containing functions for saving the data
last_version   : 23-02-2021
modified_by    : Andres Cremades Botella
"""
import numpy as np
import re
import pandas as pd
import os

#%% Functions
def save_stat(case_setup,mesh_data,section_globalCS,solution):
    # savefile_vec  : path to the files divided by folders
    # saveroot      : current folder to create
    savefile_vec = re.split('/|\n',case_setup.savefile_name)
    saveroot     = case_setup.root
    # It tries to create every folder in the path
    for ii_save in np.arange(len(savefile_vec)):
        try:
            saveroot += '/' + savefile_vec[ii_save]
            os.mkdir(saveroot)
        except:
            print("Carpeta existente")
    # Create a file for the displacements
    # name_disp_file : name of the displacement file created
    # file_disp      : data frame of displacements
    name_disp_file              = case_setup.root+case_setup.savefile_name+'/disp.csv'
    file_disp                   = pd.DataFrame()
    file_disp["node"]           = np.arange(len(solution.u))
    file_disp["x(m)"]           = mesh_data.point[:,0]
    file_disp["y(m)"]           = mesh_data.point[:,1]
    file_disp["z(m)"]           = mesh_data.point[:,2]
    file_disp["u(m)"]           = solution.u
    file_disp["v(m)"]           = solution.v
    file_disp["w(m)"]           = solution.w
    file_disp["phi(rad)"]       = solution.phi
    file_disp["psi(rad)"]       = solution.psi
    file_disp["theta(rad)"]     = solution.theta
    file_disp["phi_d(rad/m)"]   = solution.phi_d
    file_disp["psi_d(rad/m)"]   = solution.psi_d
    file_disp["theta_d(rad/m)"] = solution.theta_d
    # save as csv file
    file_disp.to_csv(name_disp_file,index=False)
    # Create a file for the deformed shape
    # name_ds_file   : name of the deformed shape file
    # file_ds        : data frame of the deformed shape
    name_ds_file    = case_setup.root+case_setup.savefile_name+'/defshape.csv'
    file_ds         = pd.DataFrame()
    file_ds["node"] = np.arange(len(solution.u))
    file_ds["x(m)"] = mesh_data.point[:,0]
    file_ds["y(m)"] = mesh_data.point[:,1]
    file_ds["z(m)"] = mesh_data.point[:,2]
    file_ds["u(m)"] = solution.xdef
    file_ds["v(m)"] = solution.ydef
    file_ds["w(m)"] = solution.zdef
    # save as csv file
    file_ds.to_csv(name_ds_file,index=False)
    # Create a file for the external forces
    # name_forc_file  : name of the loads vector
    # file_forc       : data frame of the external forces
    name_forc_file      = case_setup.root+case_setup.savefile_name+'/forces.csv'
    file_forc           = pd.DataFrame()
    file_forc["node"]   = np.arange(len(solution.u))
    file_forc["x(m)"]   = mesh_data.point[:,0]
    file_forc["y(m)"]   = mesh_data.point[:,1]
    file_forc["z(m)"]   = mesh_data.point[:,2]
    file_forc["fx(N)"]  = solution.ext_fx
    file_forc["fy(N)"]  = solution.ext_fy
    file_forc["fz(N)"]  = solution.ext_fz
    file_forc["mx(Nm)"] = solution.ext_mx
    file_forc["my(Nm)"] = solution.ext_my
    file_forc["mz(Nm)"] = solution.ext_mz
    file_forc["bx(Nm)"] = solution.ext_Bx
    file_forc["by(Nm)"] = solution.ext_By
    file_forc["bz(Nm)"] = solution.ext_Bz
    # save as csv file
    file_forc.to_csv(name_forc_file,index=False)
    # If the section is calculated the deformed shape is saved
    if mesh_data.section == "YES":
        # file_secdef     : data frame of the deformed section file
        # file_secdef_aux : auxiliar dataframe of the deformed section
        file_secdef      = pd.DataFrame()
        file_secdef_aux  = pd.DataFrame()
        name_secdef_file = case_setup.root+case_setup.savefile_name+'/secdef.csv'
        # for every node of the beam
        for ii_plot_sec in np.arange(len(section_globalCS.n1)):
            # plot_sec : internal nodes of the section
            # plot_cdg : internal center of gravity of the section
            # ii_node  : index of the nodes of the beam elements
            # xpoint   : position in x of the section nodes
            # ypoint   : position in y of the section nodes
            # zpoint   : position in z of the section nodes
            plot_sec = section_globalCS.n1[ii_plot_sec]
            plot_cdg = section_globalCS.cdgn1[ii_plot_sec]
            ii_node  = section_globalCS.nodeelem1[ii_plot_sec]
            xpoint   = solution.xdef[section_globalCS.nodeelem1[ii_plot_sec]]+(plot_sec[:,0]-plot_cdg[0])*np.cos(solution.theta[ii_node])- \
                (plot_sec[:,1]-plot_cdg[1])*np.sin(solution.theta[ii_node])+(plot_sec[:,0]-plot_cdg[0])*np.cos(solution.psi[ii_node])+ \
                (plot_sec[:,2]-plot_cdg[2])*np.sin(solution.psi[ii_node])+solution.warp_x1[ii_plot_sec]
            ypoint   = solution.ydef[section_globalCS.nodeelem1[ii_plot_sec]]+(plot_sec[:,1]-plot_cdg[1])*np.cos(solution.theta[ii_node])+ \
                (plot_sec[:,0]-plot_cdg[0])*np.sin(solution.theta[ii_node])+(plot_sec[:,1]-plot_cdg[1])*np.cos(solution.phi[ii_node])- \
                (plot_sec[:,2]-plot_cdg[2])*np.sin(solution.phi[ii_node])+solution.warp_y1[ii_plot_sec]
            zpoint   = solution.zdef[section_globalCS.nodeelem1[ii_plot_sec]]+(plot_sec[:,2]-plot_cdg[2])*np.cos(solution.psi[ii_node])- \
                (plot_sec[:,0]-plot_cdg[0])*np.sin(solution.psi[ii_node])+(plot_sec[:,2]-plot_cdg[2])*np.cos(solution.phi[ii_node])+ \
                (plot_sec[:,1]-plot_cdg[1])*np.sin(solution.phi[ii_node])+solution.warp_z1[ii_plot_sec]
            # elemsecdef : element of the deformed section
            # nodesecdef : node of the deformed section
            elemsecdef                 = np.ones((len(xpoint),))*int(ii_plot_sec)
            nodesecdef                 =  np.zeros((len(xpoint),))
            file_secdef_aux["Element"] = elemsecdef.astype('int')
            file_secdef_aux["node"]    = nodesecdef.astype('int') 
            file_secdef_aux["x(m)"]    = xpoint[0] 
            file_secdef_aux["y(m)"]    = ypoint[0]  
            file_secdef_aux["z(m)"]    = zpoint[0]  
            file_secdef                = pd.concat([file_secdef,file_secdef_aux])
        for ii_plot_sec in np.arange(len(section_globalCS.n2)):
            # plot_sec : internal nodes of the section
            # plot_cdg : internal center of gravity of the section
            # ii_node  : index of the nodes of the beam elements
            # xpoint   : position in x of the section nodes
            # ypoint   : position in y of the section nodes
            # zpoint   : position in z of the section nodes
            plot_sec = section_globalCS.n2[ii_plot_sec]
            plot_cdg = section_globalCS.cdgn2[ii_plot_sec]
            ii_node  = section_globalCS.nodeelem2[ii_plot_sec]
            xpoint   = solution.xdef[section_globalCS.nodeelem2[ii_plot_sec]]+(plot_sec[:,0]-plot_cdg[0])*np.cos(solution.theta[ii_node])- \
                (plot_sec[:,1]-plot_cdg[1])*np.sin(solution.theta[ii_node])+(plot_sec[:,0]-plot_cdg[0])*np.cos(solution.psi[ii_node])+ \
                (plot_sec[:,2]-plot_cdg[2])*np.sin(solution.psi[ii_node])+solution.warp_x2[ii_plot_sec]
            ypoint   = solution.ydef[section_globalCS.nodeelem2[ii_plot_sec]]+(plot_sec[:,1]-plot_cdg[1])*np.cos(solution.theta[ii_node])+ \
                (plot_sec[:,0]-plot_cdg[0])*np.sin(solution.theta[ii_node])+(plot_sec[:,1]-plot_cdg[1])*np.cos(solution.phi[ii_node])- \
                (plot_sec[:,2]-plot_cdg[2])*np.sin(solution.phi[ii_node])+solution.warp_y2[ii_plot_sec]
            zpoint   = solution.zdef[section_globalCS.nodeelem2[ii_plot_sec]]+(plot_sec[:,2]-plot_cdg[2])*np.cos(solution.psi[ii_node])- \
                (plot_sec[:,0]-plot_cdg[0])*np.sin(solution.psi[ii_node])+(plot_sec[:,2]-plot_cdg[2])*np.cos(solution.phi[ii_node])+ \
                (plot_sec[:,1]-plot_cdg[1])*np.sin(solution.phi[ii_node])+solution.warp_z2[ii_plot_sec]
            # elemsecdef : element of the deformed section
            # nodesecdef : node of the deformed section
            elemsecdef                 = np.ones((len(xpoint),))*int(ii_plot_sec)
            nodesecdef                 = np.ones((len(xpoint),))
            file_secdef_aux["Element"] = elemsecdef.astype('int')
            file_secdef_aux["node"]    = nodesecdef.astype('int')
            file_secdef_aux["x(m)"]    = xpoint[0]  
            file_secdef_aux["y(m)"]    = ypoint[0]  
            file_secdef_aux["z(m)"]    = zpoint[0] 
            file_secdef                = pd.concat([file_secdef,file_secdef_aux])
        # save the file as csv
        file_secdef.to_csv(name_secdef_file,index=False)
    return

#%%
def save_vib(case_setup,mesh_data,section_globalCS,solution):
    # save the information of the first n modes
    for ii_mode in np.arange(len(solution.u[0])):           
        if ii_mode < case_setup.savefile_modes:
            # if not existing create the path
            try:
                os.mkdir(case_setup.root+case_setup.savefile_name)
            except:
                print("Carpeta Existente")
            # name_disp_file : name of the displacement file
            # file_disp      : table to store displacement information
            name_disp_file              = case_setup.root+case_setup.savefile_name+'/disp_'+str(ii_mode)+'.csv'
            file_disp                   = pd.DataFrame()
            file_disp["node"]           = np.arange(len(solution.u[:,ii_mode]))
            file_disp["x(m)"]           = mesh_data.point[:,0]
            file_disp["y(m)"]           = mesh_data.point[:,1]
            file_disp["z(m)"]           = mesh_data.point[:,2]
            file_disp["u(m)"]           = solution.u[:,ii_mode]
            file_disp["v(m)"]           = solution.v[:,ii_mode]
            file_disp["w(m)"]           = solution.w[:,ii_mode]
            file_disp["phi(rad)"]       = solution.phi[:,ii_mode]
            file_disp["psi(rad)"]       = solution.psi[:,ii_mode]
            file_disp["theta(rad)"]     = solution.theta[:,ii_mode]
            file_disp["phi_d(rad/m)"]   = solution.phi_d[:,ii_mode]
            file_disp["psi_d(rad/m)"]   = solution.psi_d[:,ii_mode]
            file_disp["theta_d(rad/m)"] = solution.theta_d[:,ii_mode]
            # Save to csv file
            file_disp.to_csv(name_disp_file,index=False)
            # name_ds_file   : deformed shape file
            # file_ds        : table to store deformed shape
            name_ds_file    = case_setup.root+case_setup.savefile_name+'/defshape'+str(ii_mode)+'.csv'
            file_ds         = pd.DataFrame()
            file_ds["node"] = np.arange(len(solution.u[:,ii_mode]))
            file_ds["x(m)"] = mesh_data.point[:,0]
            file_ds["y(m)"] = mesh_data.point[:,1]
            file_ds["z(m)"] = mesh_data.point[:,2]
            file_ds["u(m)"] = solution.pos[:,ii_mode,0]
            file_ds["v(m)"] = solution.pos[:,ii_mode,1]
            file_ds["w(m)"] = solution.pos[:,ii_mode,2]
            # Save to csv file
            file_ds.to_csv(name_ds_file,index=False)
            # If the data in section nodes is calculated its deformed position can be obtained
            if mesh_data.section == "YES":
                # file_secdef      : table to store the deformed section
                # file_secdef_aux  : auxiliary table to store the deformed section
                # name_secdef_file : name of the deformed section file
                file_secdef      = pd.DataFrame()
                file_secdef_aux  = pd.DataFrame()
                name_secdef_file = case_setup.root+case_setup.savefile_name+'/secdef'+str(ii_mode)+'.csv'
                # for all the points in the section coordinates
                for ii_plot_sec in np.arange(len(section_globalCS.n1)):
                    # plot_sec : points of the section
                    # plot_cdg : center of gravity of the section
                    # ii_node  : beam node of the section
                    plot_sec = section_globalCS.n1[ii_plot_sec]
                    plot_cdg = section_globalCS.cdgn1[ii_plot_sec]
                    ii_node  = section_globalCS.nodeelem1[ii_plot_sec]
                    # xpoint : points of the section in x axis
                    # ypoint : points of the section in y axis
                    # zpoint : points of the section in z axis
                    xpoint = solution.xdef[section_globalCS.nodeelem1[ii_plot_sec],ii_mode]+(plot_sec[:,0]-plot_cdg[0])*np.cos(solution.theta[ii_node,ii_mode])*np.cos(solution.psi[ii_node,ii_mode])- \
                        (plot_sec[:,1]-plot_cdg[1])*np.sin(solution.theta[ii_node,ii_mode])+ \
                        (plot_sec[:,2]-plot_cdg[2])*np.sin(solution.psi[ii_node,ii_mode])+solution.warp_x1[ii_mode][ii_plot_sec]
                    ypoint = solution.ydef[section_globalCS.nodeelem1[ii_plot_sec],ii_mode]+(plot_sec[:,1]-plot_cdg[1])*np.cos(solution.theta[ii_node,ii_mode])*np.cos(solution.phi[ii_node,ii_mode])+ \
                        (plot_sec[:,0]-plot_cdg[0])*np.sin(solution.theta[ii_node,ii_mode])- \
                        (plot_sec[:,2]-plot_cdg[2])*np.sin(solution.phi[ii_node,ii_mode])+solution.warp_y1[ii_mode][ii_plot_sec]
                    zpoint = solution.zdef[section_globalCS.nodeelem1[ii_plot_sec],ii_mode]+(plot_sec[:,2]-plot_cdg[2])*np.cos(solution.psi[ii_node,ii_mode])*np.cos(solution.phi[ii_node,ii_mode])- \
                        (plot_sec[:,0]-plot_cdg[0])*np.sin(solution.psi[ii_node,ii_mode])+ \
                        (plot_sec[:,1]-plot_cdg[1])*np.sin(solution.phi[ii_node,ii_mode])+solution.warp_z1[ii_mode][ii_plot_sec]
                    # elemsecdef : element that the section belongs
                    # nodesecdef : node that the section belongs
                    elemsecdef                 = np.ones((len(xpoint),))*int(ii_plot_sec)
                    nodesecdef                 = np.zeros((len(xpoint),))
                    file_secdef_aux["Element"] = elemsecdef.astype('int')
                    file_secdef_aux["node"]    = nodesecdef.astype('int') 
                    file_secdef_aux["x(m)"]    = xpoint[0] 
                    file_secdef_aux["y(m)"]    = ypoint[0]  
                    file_secdef_aux["z(m)"]    = zpoint[0]  
                    file_secdef                = pd.concat([file_secdef,file_secdef_aux])
                for ii_plot_sec in np.arange(len(section_globalCS.n2)):
                    # plot_sec : points of the section
                    # plot_cdg : center of gravity of the section
                    # ii_node  : beam node of the section
                    plot_sec = section_globalCS.n2[ii_plot_sec]
                    plot_cdg = section_globalCS.cdgn2[ii_plot_sec]
                    ii_node  = section_globalCS.nodeelem2[ii_plot_sec]
                    # xpoint : points of the section in x axis
                    # ypoint : points of the section in y axis
                    # zpoint : points of the section in z axis
                    xpoint = solution.xdef[section_globalCS.nodeelem2[ii_plot_sec],ii_mode]+(plot_sec[:,0]-plot_cdg[0])*np.cos(solution.theta[ii_node,ii_mode])*np.cos(solution.psi[ii_node,ii_mode])- \
                        (plot_sec[:,1]-plot_cdg[1])*np.sin(solution.theta[ii_node,ii_mode])+ \
                        (plot_sec[:,2]-plot_cdg[2])*np.sin(solution.psi[ii_node,ii_mode])+solution.warp_x2[ii_mode][ii_plot_sec]
                    ypoint = solution.ydef[section_globalCS.nodeelem2[ii_plot_sec],ii_mode]+(plot_sec[:,1]-plot_cdg[1])*np.cos(solution.theta[ii_node,ii_mode])*np.cos(solution.phi[ii_node,ii_mode])+ \
                        (plot_sec[:,0]-plot_cdg[0])*np.sin(solution.theta[ii_node,ii_mode])- \
                        (plot_sec[:,2]-plot_cdg[2])*np.sin(solution.phi[ii_node,ii_mode])+solution.warp_y2[ii_mode][ii_plot_sec]
                    zpoint = solution.zdef[section_globalCS.nodeelem2[ii_plot_sec],ii_mode]+(plot_sec[:,2]-plot_cdg[2])*np.cos(solution.psi[ii_node,ii_mode])*np.cos(solution.phi[ii_node,ii_mode])- \
                        (plot_sec[:,0]-plot_cdg[0])*np.sin(solution.psi[ii_node,ii_mode])+ \
                        (plot_sec[:,1]-plot_cdg[1])*np.sin(solution.phi[ii_node,ii_mode])+solution.warp_z2[ii_mode][ii_plot_sec]
                    # elemsecdef : element that the section belongs
                    # nodesecdef : node that the section belongs
                    elemsecdef                 = np.ones((len(xpoint),))*int(ii_plot_sec)
                    nodesecdef                 = np.ones((len(xpoint),))
                    file_secdef_aux["Element"] = elemsecdef.astype('int')
                    file_secdef_aux["node"]    = nodesecdef.astype('int')
                    file_secdef_aux["x(m)"]    = xpoint[0]  
                    file_secdef_aux["y(m)"]    = ypoint[0]  
                    file_secdef_aux["z(m)"]    = zpoint[0] 
                    file_secdef                = pd.concat([file_secdef,file_secdef_aux])
                # Save deformed section to file
                file_secdef.to_csv(name_secdef_file,index=False)  
    # name_freq_file : name of the vibration frequencies file
    # file_freq      : table to store the information
    name_freq_file              = case_setup.root+case_setup.savefile_name+'/freq.csv'
    file_freq                   = pd.DataFrame()
    file_freq["mode"]           = np.arange(np.min([len(solution.freq_mode),case_setup.savefile_modes]))+1
    file_freq["frequency (Hz)"] = solution.freq_mode[:np.min([case_setup.savefile_modes,len(solution.freq_mode)])]
    # save to csv file
    file_freq.to_csv(name_freq_file,index=False)
    return