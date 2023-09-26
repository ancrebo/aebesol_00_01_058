# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:46:50 2021

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
section_mass   : file containing functions for calculating the warping function
last_version   : 19-02-2021
modified_by    : Andres Cremades Botella
"""

import numpy as np

#%% Functions
def func_cg_TW(section_elem,section_points,section_cell,sol_phys,section_thickply,section_ply_ncoord):
    # Function to determine the center of gravity of a thin wall section
    # section_elem       : elments of the section
    # section_points     : nodes of the section
    # section_cell       : cells of the section
    # sol_phys           : solid physical information
    # section_thickply   : thickness of the laminate plies
    # section_ply_ncoord : position of the plies
    # -------------------------------------------------------------------------
    # cg        : center of gravity
    # areatot   : total area of the section
    # area_elem : total area of the element
    # cg_local  : local center of gravity (mid point of the curvature line)
    # long      : longitude of the element
    # mass      : mass of the section
    cg        = 0
    areatot   = 0
    area_elem = np.zeros((len(section_elem),))
    cg_local  = np.zeros((len(section_elem),2))
    long      = np.zeros((len(section_elem),))
    mass      = 0
    # For all the elements in the section calculate its center of gravity and area
    for elem in section_elem:
        long[int(elem[0])]       = np.linalg.norm(section_points[int(elem[1])]-section_points[int(elem[2])])
        area_elem[int(elem[0])] += long[int(elem[0])]*elem[3]
        areatot                 += long[int(elem[0])]*elem[3]
        cg_local[int(elem[0]),:] = (section_points[int(elem[1])]+section_points[int(elem[2])])/2
        dsvec                    = (section_points[int(elem[2])]-section_points[int(elem[1])])
        dsvec                   /= np.linalg.norm(dsvec)
        dnvec                    = np.cross([0,0,1],dsvec)
        dnvec                   /= np.linalg.norm(dnvec)
        dnvec                    = dnvec[:2]
        # mass_ii : mass of the element
        mass_ii = 0
        # for every ply
        for ply in np.arange(len(sol_phys.rho[0])):
            try:
                mass_ii += sol_phys.rho[int(elem[0])][ply]*section_thickply[int(elem[0])][ply]*long[int(elem[0])]
            except:
                try:
                    mass_ii += sol_phys.rho[int(elem[0])]*section_thickply[int(elem[0])]*long[int(elem[0])]
                except:
                    mass_ii += sol_phys.rho[int(elem[0])][0]*section_thickply[int(elem[0])][0]*long[int(elem[0])]
        mass += mass_ii
        cg   += mass_ii*cg_local[int(elem[0]),:] 
    # Calculate the center of gravity of the section
    cg /= mass
    # area_cells : area of every closed cell of the section
    # beta_cells : perimeter of every closed cell of the section
    area_cells =  np.zeros((len(section_cell),))
    beta_cells =  np.zeros((len(section_cell),))    
    L_cells    =  np.zeros((len(section_cell),))
    psi_cells  =  []
    # For every section cell
    if len(section_cell) > 0:
        # ds_vec    : longitudinal vector of the element
        # ds        : length of the element
        # A2_cg     : area contained in the cell for each element
        # delta_sec : quoeficient between the length of the element and its thickness
        ds_vec    = np.zeros((len(section_cell),len(section_elem),2))
        ds        = np.zeros((len(section_elem),))
        A2_cg     = np.zeros((len(section_cell),))
        delta_sec = np.zeros((len(section_cell),))
        divply    =  np.zeros((len(section_elem),))
        for ii_sec in np.arange(len(section_cell)):
            psi_cellsu = []
            for jj_elem_ind in np.arange(len(section_cell[ii_sec])):
                # For all the elments of each cell
                # jj_elem      : element of the cell
                # jj_elem_next : following element of the cell
                jj_elem = section_cell[ii_sec][jj_elem_ind]
                try:
                    jj_elem_next = section_cell[ii_sec][jj_elem_ind+1]
                except:
                    jj_elem_next = section_cell[ii_sec][0]
                # Find the common node of both elements
                # node_ini  : index of the first node of the element
                # node_fin  : index of the final node of the element
                # point_ini : position of the first node
                # point_fin : position of the final node
                if section_elem[jj_elem,1] == section_elem[jj_elem_next,1] or section_elem[jj_elem,1] == section_elem[jj_elem_next,2]:
                    node_ini  = section_elem[jj_elem,1]
                    node_fin  = section_elem[jj_elem,2]
                    point_ini = section_points[int(node_ini),:]
                    point_fin = section_points[int(node_fin),:]
                elif section_elem[jj_elem,2] == section_elem[jj_elem_next,1] or section_elem[jj_elem,2] == section_elem[jj_elem_next,2]:
                    node_ini  = section_elem[jj_elem,2]
                    node_fin  = section_elem[jj_elem,1]
                    point_ini = section_points[int(node_ini),:]
                    point_fin = section_points[int(node_fin),:]
                # rx         : distance from the local to the global gravity center in x axis
                # ry         : distance from the local to the global gravity center in y axis
                # ds_vec     : vector containing the length of the element
                # ds         : legth of the element
                # area_vec   : vector containing the area of the cell
                # area_cells : area of the section cell
                # beta_cells : perimeter of the section cell
                rx                       = cg_local[jj_elem,0]-cg[0]
                ry                       = cg_local[jj_elem,1]-cg[1]
                ds_vec[ii_sec,jj_elem,:] = point_fin-point_ini
                ds[jj_elem]              = np.linalg.norm(ds_vec[ii_sec,jj_elem,:])
                area_vec                 = np.cross([rx,ry,0],[ds_vec[ii_sec,jj_elem,0],ds_vec[ii_sec,jj_elem,1],0])
                A2_cg[ii_sec]           += abs(area_vec[2])
                area_cells[ii_sec]      += abs(area_vec[2])/2
                beta_cells[ii_sec]      += ds[jj_elem]
                for ply in np.arange(len(sol_phys.rho[jj_elem])):
                    divply[jj_elem] += section_thickply[jj_elem][ply]*sol_phys.Qbar66[jj_elem][ply]
                L_cells[ii_sec]         += ds[jj_elem]/divply[jj_elem]
                delta_sec[ii_sec]       += ds[jj_elem]/section_elem[jj_elem,3]
            for jj_elem_ind in np.arange(len(section_cell[ii_sec])):
                jj_elem = section_cell[ii_sec][jj_elem_ind]
                psi_cellsu.append(A2_cg[ii_sec]/divply[jj_elem])
            psi_cells.append(psi_cellsu/L_cells[ii_sec])
    return cg, area_elem, cg_local, long, areatot, area_cells, beta_cells, mass, L_cells, psi_cells


