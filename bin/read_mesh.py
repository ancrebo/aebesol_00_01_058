# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:32:25 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
read_mesh      : file containing functions for reading the mesh file
last_version   : 17-02-2021
modified_by    : Andres Cremades Botella
"""

import numpy as np
from bin.read_material import read_material
from bin.aux_functions import def_vec_param

#%% Functions
def read_mesh(filemesh,case_setup):
    # Function to read the text file containing the mesh
    # filemesh      : name of the file containing the mesh
    # case_setup    : configuration of the case
    # -----------------------------------------------------------------------------------
    # word_ant      : previous word  in the text
    # flg_elem      : flag to determine when the elements are under lecture
    # flg_point     : flag to determine when the nodes are under lecture
    # flg_mark      : flag to determine when the markers are under lecture
    # flg_mark_node : flag to determine when the nodes in the markers are under lecture
    # ii_elem       : number of the elements read
    # ii_point      : number of the nodes read
    # ii_mark       : number of markers read
    # ii_mark_node  : number of nodes in a marker read
    # num_mark_node : maximum number of nodes in a marker
    # kk            : lecture index
    # num_elem      : number of elements of the mesh
    # num_point     : number of nodes of the mesh
    word_ant      = " "
    flg_elem      = 0
    flg_point     = 0
    flg_mark      = 0
    flg_mark_node = 0
    ii_elem       = 0
    ii_point      = 0
    ii_mark       = 0
    ii_mark_node  = 0
    num_mark_node = 0
    marker_all    = []
    kk            = 1
    num_elem      = 1
    num_point     = 1
    # flg_exmass  : flag to add extra mass
    flg_exmass     = 0
    flg_exmassdist = 0
    keepwarp     = 'NO'
    # Read all the words in mesh file
    with open(filemesh,"r") as textmesh:
        for line in textmesh:
            # nn_sec      : number of sections stored in a node
            # sec_profile : name of the section mesh fileç
            # rot_sec     : rotations of the section
            # scale_sec   : modify the dimensions of the section
            nn_sec      = 1
            sec_profile = []
            rot_sec     = []
            scale_sec   = []
            xdist_sec   = []
            ydist_sec   = []
            # if the function is reading nodes, advance the node counter
            if flg_point == 1 and kk>0:
                ii_point = ii_point+1
            if flg_elem == 1 and kk>0:
                ii_elem = ii_elem + 1
            kk = 0
            if flg_exmass == 1:
                mesh_point_exmass.append([])
                flg_exmass = 0
            if flg_exmassdist == 1:
                mesh_elem_exmass.append([])
                flg_exmassdist = 0
            ind_comment = line.find('%')
            line2 = line[:ind_comment]
            ndata = len(line2.split())
            for word in line.split():
                # Avoid commentary
                if word == '%':
                    break
                # if the function is reading elements
                if flg_elem==1 and ii_elem<num_elem:
                    # if the index is:
                    #   - 0 : read the element index
                    #   - 1 : read the first node
                    #   - 2 : read the second node
                    #   - 3 : advance element index
                    # mesh_elem : matrix to store the element information 
                    if kk == 0:
                        mesh_elem[ii_elem,kk] = int(word)
                        kk                    = kk+1
                    elif kk == 1:
                        mesh_elem[ii_elem,kk] = int(word)
                        kk                    = kk+1
                    elif kk == 2:
                        mesh_elem[ii_elem,kk] = int(word)
                        kk                    = kk+1
                        fk = 1
                    if kk == 3 and fk == 0:
                        kk = kk+1
                        if word[:5] == 'LOCK:':
                            mesh_elem_lock[ii_elem] = int(word[5:])
                        else:
                            flg_exmassdist = 1 
                    if kk== 4 and word[:5] != 'LOCK:':
                        flg_exmassdist = 1 
                    else:
                        fk = 0
                if flg_exmass == 1:
                    mesh_point_exmass.append(word)
                    mesh_point_exmassind.append(ii_point)
                    flg_exmass = 0
                    kk         = 10
                elif flg_exmassdist == 1:
                    mesh_elem_exmass.append(word)
                    mesh_elem_exmassind.append(ii_elem)
                    flg_exmassdist = 0
#                    if kk==4 and ndata-1==kk:
#                        flg_exmassdist = 1 
                # if the function is reading nodes
                if flg_point == 1 and ii_point<num_point:
                    # if the index is:
                    #    - 0 : read node index
                    #    - 1 : read the x coordinate
                    #    - 2 : read the y coordinate
                    #    - 3 : read the z coordinate
                    #    - 4 : read the number of sections
                    #    - 5 : read the profile mesh file
                    #    - 6 : read the rotation of the section
                    #    - 7 : read the scale of the section
                    # mesh_point         : matrix to store node information
                    # mesh_point_nsec    : number of internal sections of the nodes
                    # mesh_point_profile : section of the mesh
                    # mesh_point_rot     : rotation of the mesh
                    # mesh_point_scale   : scale of the section
                    # mesh_point_exmass  : extra mass of the section
                    # mesh_point_pri     : priority of the node to increase mesh higher priority duplicate if equal take the one with less sections if equal take first
                    if kk == 0:
                        kk = kk+1
                    elif kk == 1:
                        mesh_point[ii_point,kk-1] = float(word)
                        kk = kk+1
                    elif kk == 2:
                        mesh_point[ii_point,kk-1] = float(word)
                        kk = kk+1
                    elif kk == 3:
                        mesh_point[ii_point,kk-1] = float(word)
                        kk = kk+1
                    elif kk == 4:
                        if word[:4] == 'PRI-':
                            mesh_point_pri[ii_point] = int(word[4:])
                        else:
                            mesh_point_nsec[ii_point] = int(word)
                            kk = kk+1
                    elif kk == 5:
                        sec_profile.append(word)
                        if nn_sec == mesh_point_nsec[ii_point]:
                            mesh_point_profile.append(sec_profile) 
                        kk = kk+1
                    elif kk == 6:
                        rot_sec.append(float(word))
                        if nn_sec == mesh_point_nsec[ii_point]:
                            mesh_point_rot.append(rot_sec) 
                        kk = kk+1
                    elif kk == 7:
                        scale_sec.append(float(word))
                        if nn_sec == mesh_point_nsec[ii_point]:
                            mesh_point_scale.append(scale_sec) 
                        kk = kk+1
                    elif kk == 8:
                        xdist_sec.append(float(word))
                        if nn_sec == mesh_point_nsec[ii_point]:
                            mesh_point_xdisp.append(xdist_sec)
                        kk = kk+1
                    elif kk == 9:
                        ydist_sec.append(float(word))
                        if nn_sec == mesh_point_nsec[ii_point]:
                            mesh_point_ydisp.append(ydist_sec)
                            flg_exmass = 1
                        kk     -= 4
                        nn_sec += 1
                # If the function is reading the marker nodes
                if flg_mark_node == 1 and ii_mark_node < num_mark_node:
                    # marker  : class to store the information of the markers
                    marker.node[ii_mark_node] = int(word)
                    ii_mark_node              = ii_mark_node + 1
                    # if the nodes of the marker are read
                    if ii_mark_node == num_mark_node:
                        # marker_all : class to store the information about all markers
                        # increase the marker index and reset the number of nodes index
                        marker_all.append(marker)
                        ii_mark       = ii_mark+1
                        ii_mark_node  = 0
                        flg_mark_node = 0
                # if the function is reading a marker
                # read the marker name
                if word_ant== "MARK_NAME=":
                    class marker:
                        pass
                    marker.name = word
                # read the mareker number of nodes
                elif word_ant== "MARK_NODE=":
                    # num_mark_node : number of nodes in the marker
                    num_mark_node   = int(word)
                    marker.num_node = num_mark_node
                    flg_mark_node   = 1 
                    marker.node     = np.zeros((num_mark_node,1),dtype='int')
                # read the number of elements of the mesh
                if word_ant== "NELEM=":
                    num_elem         = int(word)
                    flg_elem         = 1
                    mesh_elem        = np.zeros((num_elem,3))
                    mesh_elem_exmass = []
                    mesh_elem_exmassind = []
                    mesh_elem_lock      = np.zeros((num_elem,))
                # determine if the section is evaluated or not
                if word_ant == "SECT=":
                    section = word
                if word_ant == "KEEPWARP=":
                    keepwarp = word
                # Read number of points
                if word_ant == "NPOINT=":
                    num_point           = int(word)
                    flg_point           = 1
                    mesh_point          = np.zeros((num_point,3))
                    mesh_point_nsec     = np.zeros((num_point,))
                    mesh_point_xdisp    = []
                    mesh_point_ydisp    = []
                    mesh_point_scale    = []
                    mesh_point_rot      = []
                    mesh_point_profile  = []
                    mesh_point_material = []
                    mesh_point_exmass   = []
                    mesh_point_exmassind = []
                    mesh_point_pri      = np.zeros((num_point,))
                # updathe the previous word
                word_ant = word
    # mesh_data : class to store mesh information
    class mesh_data:
        pass
    mesh_data.num_elem            = num_elem
    mesh_data.num_point           = num_point
    mesh_data.point               = mesh_point
    mesh_data.elem                = mesh_elem
    mesh_data.marker              = marker_all
    mesh_data.mesh_point_xdisp    = mesh_point_xdisp
    mesh_data.mesh_point_ydisp    = mesh_point_ydisp
    mesh_data.mesh_point_scale    = mesh_point_scale
    mesh_data.mesh_point_rot      = mesh_point_rot
    mesh_data.mesh_point_profile  = mesh_point_profile
    mesh_data.mesh_point_material = mesh_point_material
    mesh_data.section             = section
    mesh_data.mesh_point_nsec     = mesh_point_nsec
    mesh_data.mesh_point_exmass   = mesh_point_exmass
    mesh_data.mesh_point_exmassind   = mesh_point_exmassind
    mesh_data.mesh_elem_exmass    = mesh_elem_exmass
    mesh_data.mesh_elem_exmassind    = mesh_elem_exmassind
    mesh_data.mesh_point_pri      = mesh_point_pri
    mesh_data.mesh_elem_lock      = mesh_elem_lock
    mesh_data.keepwarp           = keepwarp
    # Refine the mesh if required           
    try:
        # refmesh : number of refinements
        refmesh = case_setup.refmesh
        # repite if refmesh is higher than 0
        while refmesh > 0:
            # point2      : copy of the point matrix
            # marker      : copy of the marker information
            # nsec2       : copy of the number of sections
            # prof2       : copy of the section file
            # scale2      : copy of the scale
            # rot2        : copy of the rotation angle
            # exmass2     : copy of the point extra mass
            # exmassdist2 : copy of the distributed extra mass
            # elem2       : matrix of elements with the double number than the original
            # kk          : index
            point2      = mesh_data.point.copy()
            marker      = mesh_data.marker.copy()
            nsec2       = mesh_data.mesh_point_nsec.copy()
            prof2       = mesh_data.mesh_point_profile.copy()
            scale2      = mesh_data.mesh_point_scale.copy()
            rot2        = mesh_data.mesh_point_rot.copy()
            xdisp2      = mesh_data.mesh_point_xdisp.copy()
            ydisp2      = mesh_data.mesh_point_ydisp.copy()
            exmass2     = mesh_data.mesh_point_exmass.copy()
            exmassind2  = mesh_data.mesh_point_exmassind.copy()
            exmassdist2 = []
            exmassdistind2 = []
            elem2       = np.zeros((2*mesh_data.num_elem,3))
            pri2        = mesh_data.mesh_point_pri.copy()
            lock2        = [] #mesh_data.mesh_elem_lock.copy()
            kk          = 0
            for elem in mesh_data.elem:
            # for all the elements separate the element one row in the elem2 matrix
            # in the space between create a new element with an extra point
                elem2[2*kk,0]   = 2*kk
                elem2[2*kk,1]   = elem[1]
                elem2[2*kk,2]   = len(point2)
                elem2[2*kk+1,0] = 2*kk+1
                elem2[2*kk+1,1] = len(point2)
                elem2[2*kk+1,2] = elem[2]
                # add the scale of the new node as the average of the neighbour nodes
                # add the rotation of the section as the average of the nighbours
                # add the number of sections as the minimum of nighbour
                # scale2it : item to add to the scale
                # rot2it   : item to add to the rotation
                try:
                    exmassdist2.append(mesh_data.mesh_elem_exmass[kk])
                    exmassdist2.append(mesh_data.mesh_elem_exmass[kk])
                    exmassdistind2.append(2*kk)
                    exmassdistind2.append(2*kk+1)
                    lock2.append(mesh_data.mesh_elem_lock[kk])
                    lock2.append(mesh_data.mesh_elem_lock[kk])
                except:
                    exmassdist2.append([])
                    exmassdist2.append([])
                    exmassdistind2.append([])
                    exmassdistind2.append([])
                    lock2.append(0)
                    lock2.append(0)                    
                if pri2[int(elem[1])] > pri2[int(elem[2])]:
                    prof2.append(prof2[int(elem[1])])
                    exmass2.append(exmass2[int(elem[1])])
                    exmassind2.append(exmassind2[int(elem[1])])
                    pri2 = np.concatenate((pri2,[pri2[int(elem[1])]]))
                    scale2it  = np.array(scale2[int(elem[1])])
                    rot2it    = np.array(rot2[int(elem[1])])
                    xdisp2it  = np.array(xdisp2[int(elem[1])])
                    ydisp2it  = np.array(ydisp2[int(elem[1])])
                elif pri2[int(elem[2])] > pri2[int(elem[1])]:
                    prof2.append(prof2[int(elem[2])])
                    exmass2.append(exmass2[int(elem[2])])
                    exmassind2.append(exmassind2[int(elem[2])])
                    pri2 = np.concatenate((pri2,[pri2[int(elem[2])]]))
                    scale2it  = np.array(scale2[int(elem[2])])
                    rot2it    = np.array(rot2[int(elem[2])])
                    xdisp2it  = np.array(xdisp2[int(elem[2])])
                    ydisp2it  = np.array(ydisp2[int(elem[2])])
                else:
                    if nsec2[int(elem[1])]<=nsec2[int(elem[2])]:
                        prof2.append(prof2[int(elem[1])])
                        exmass2.append(exmass2[int(elem[1])])
                        exmassind2.append(exmassind2[int(elem[1])])
                        pri2 = np.concatenate((pri2,[pri2[int(elem[1])]]))
                    elif nsec2[int(elem[1])]>nsec2[int(elem[2])]:
                        prof2.append(prof2[int(elem[2])])
                        exmass2.append(exmass2[int(elem[2])])
                        exmassind2.append(exmassind2[int(elem[2])])
                        pri2 = np.concatenate((pri2,[pri2[int(elem[2])]]))
                    scale2it  = (np.array(scale2[int(elem[1])])+np.array(scale2[int(elem[2])]))/2
                    rot2it    = (np.array(rot2[int(elem[1])])+np.array(rot2[int(elem[2])]))/2
                    xdisp2it  = (np.array(xdisp2[int(elem[1])])+np.array(xdisp2[int(elem[2])]))/2
                    ydisp2it  = (np.array(ydisp2[int(elem[1])])+np.array(ydisp2[int(elem[2])]))/2
                scale2.append(scale2it.tolist())
                rot2.append(rot2it.tolist())
                xdisp2.append(xdisp2it.tolist())
                ydisp2.append(ydisp2it.tolist())
                nsec2 = np.concatenate((nsec2,[np.min([nsec2[int(elem[1])],nsec2[int(elem[2])]])]),axis=0)
                kk += 1
                # add a new point
                point2 = np.concatenate((point2,[(mesh_data.point[int(elem[1]),:]+mesh_data.point[int(elem[2]),:])/2]),0)
                # add new markers nodes in the created in the middle of existing
                for ii_mark in np.arange(len(marker)):
                    pos1 = np.where(marker[ii_mark].node == elem[1])[0]
                    pos2 = np.where(marker[ii_mark].node == elem[2])[0]
                    if len(pos1)>0 and len(pos2)>0:
                        marker[ii_mark].node      = np.concatenate((np.concatenate((marker[ii_mark].node[:pos1[0]+1],[[len(point2)-1]]),axis=0),marker[ii_mark].node[pos2[0]:]),axis=0)
                        marker[ii_mark].num_node += 1
            for ii_mark in np.arange(len(marker)):
                aux_k_m = 0
                for ii_markn in np.arange(len(marker[ii_mark].node)):
                    if ii_markn > 0:
                        if marker[ii_mark].node[ii_markn+aux_k_m] < 0 and marker[ii_mark].node[ii_markn-1+aux_k_m] < 0 and marker[ii_mark].node[ii_markn+aux_k_m] == marker[ii_mark].node[ii_markn-1+aux_k_m]:
                            marker[ii_mark].node      = np.concatenate((np.concatenate((marker[ii_mark].node[:ii_markn+aux_k_m],[marker[ii_mark].node[ii_markn+aux_k_m]]),axis=0),marker[ii_mark].node[ii_markn+aux_k_m:]),axis=0)
                            marker[ii_mark].num_node += 1
                            aux_k_m += 1
            # Update class mesh_data
            mesh_data.elem               = elem2.copy()
            mesh_data.point              = point2.copy()
            mesh_data.marker             = marker.copy()
            mesh_data.num_elem           = len(elem2)
            mesh_data.num_point          = len(point2)
            mesh_data.mesh_point_nsec    = nsec2.copy()
            mesh_data.mesh_point_profile = prof2.copy()
            mesh_data.mesh_point_scale   = scale2.copy()
            mesh_data.mesh_point_rot     = rot2.copy()
            mesh_data.mesh_point_xdisp   = xdisp2.copy()
            mesh_data.mesh_point_ydisp   = ydisp2.copy()
            mesh_data.mesh_point_exmass  = exmass2.copy()
            mesh_data.mesh_point_exmassind  = exmassind2.copy()
            mesh_data.mesh_point_pri     = pri2.copy()
            mesh_data.mesh_elem_lock     = lock2.copy()
            mesh_data.mesh_elem_exmass   = exmassdist2.copy()
            mesh_data.mesh_elem_exmassind   = exmassdistind2.copy()
            # reduce the number of required refinements            
            refmesh -= 1
    except:
        pass    
    mesh_data.RR_ctrl = def_vec_param(mesh_data.num_elem)
    for ii_rr in np.arange(len(mesh_data.RR_ctrl)):
        mesh_data.RR_ctrl[ii_rr] = np.identity(3) 
    return mesh_data
#%%
def read_exmass(baseroot,mesh_data):
    # Function to read extra mass definition
    # baseroot  : base path to the files
    # mesh_data : information of the mesh
    # -------------------------------------------------------------------------
    # exmass : extra mass class
    #   - elem : subclass to store distributed mass
    #        - rho  : density
    #        - area : area
    #        - Ixx  : area inertia respect to axis x
    #        - Iyy  : area inertia respect to axis y
    #        - Ixy  : area inertia respect to axis z
    #   - point : subclass to store concentrated mass
    #        - mass : mass
    #        - Ixx  : mass inertia respect to the axis x
    #        - Iyy  : mass inertia respect to the axis y
    #        - Izz  : mass inertia respect to the axis z
    #        - Ixy  : mass inertia cross term axis xy
    #        - Iyz  : mass inertia cross term axis yz
    #        - Izx  : mass inertia cross term axis zx
    class exmass:
        class elem:
            rho  = np.zeros((mesh_data.num_elem,))
            area = np.zeros((mesh_data.num_elem,))
            Ixx  = np.zeros((mesh_data.num_elem,))
            Iyy  = np.zeros((mesh_data.num_elem,))
            Ixy  = np.zeros((mesh_data.num_elem,))
            xCG  = np.zeros((mesh_data.num_point,))
            yCG  = np.zeros((mesh_data.num_point,))
        class point:
            mass = np.zeros((mesh_data.num_point,))
            Ixx  = np.zeros((mesh_data.num_point,))
            Iyy  = np.zeros((mesh_data.num_point,))
            Izz  = np.zeros((mesh_data.num_point,))
            Ixy  = np.zeros((mesh_data.num_point,))
            Iyz  = np.zeros((mesh_data.num_point,))
            Izx  = np.zeros((mesh_data.num_point,))
    class exstiff:
        class elem:
            area = np.zeros((mesh_data.num_elem,))
            Ixx  = np.zeros((mesh_data.num_elem,))
            Iyy  = np.zeros((mesh_data.num_elem,))
            Ixy  = np.zeros((mesh_data.num_elem,))
            JJ   = np.zeros((mesh_data.num_point,)) 
            FW   = np.zeros((mesh_data.num_point,))
            EE   = np.zeros((mesh_data.num_point,))   
            GG   = np.zeros((mesh_data.num_point,))
            xCG  = np.zeros((mesh_data.num_point,))
            yCG  = np.zeros((mesh_data.num_point,))
        class point:
            AA       = np.zeros((mesh_data.num_point,))
            IIy      = np.zeros((mesh_data.num_point,))
            IIx      = np.zeros((mesh_data.num_point,))
            IIxy     = np.zeros((mesh_data.num_point,))
            UIx      = np.zeros((mesh_data.num_point,))
            UIy      = np.zeros((mesh_data.num_point,))
            JJ       = np.zeros((mesh_data.num_point,))
            LONG     = np.ones((mesh_data.num_point,)) 
            EE       = np.zeros((mesh_data.num_point,))   
            GG       = np.zeros((mesh_data.num_point,))             
    # define the properties for the distributed mass
    for ii in np.arange(len(mesh_data.mesh_elem_exmass)):
        if len(mesh_data.mesh_elem_exmass[ii])>0:
            exmassfile = baseroot+mesh_data.mesh_elem_exmass[ii]
            indexmass = mesh_data.mesh_elem_exmassind[ii]
            word_ant   = []
            fstiff = 1
            with open(exmassfile,"r") as textmass:
                for line in textmass:
                    for word in line.split():
                        if word == '%':
                            break
                        if word_ant == "STIFF=":
                            wordstiff = word
                            if wordstiff == "YES":
                                fstiff = 1
                            else:
                                fstiff = 0
                        if word_ant == "a=":
                            exmass.elem.area[indexmass] = float(word)
                            exstiff.elem.area[indexmass] = float(word)
                        elif word_ant == "Ixx=":
                            exmass.elem.Ixx[indexmass] = float(word)
                            exstiff.elem.Ixx[indexmass] = float(word)
                        elif word_ant == "Iyy=":
                            exmass.elem.Iyy[indexmass] = float(word)
                            exstiff.elem.Iyy[indexmass] = float(word)
                        elif word_ant == "Ixy=":
                            exmass.elem.Ixy[indexmass] = float(word)
                            exstiff.elem.Ixy[indexmass] = float(word)
                        elif word_ant == "MAT=":
                            matdata = read_material(baseroot+word)
                            try:
                                exstiff.elem.EE[indexmass] = matdata.EE
                                exstiff.elem.GG[indexmass] = matdata.GG
                                exmass.elem.rho[indexmass] = matdata.rho
                            except:
                                exstiff.elem.EE[indexmass] = (matdata.EEL+matdata.EET)/2
                                exstiff.elem.GG[indexmass] = (matdata.GGLT+matdata.GGTN)/2
                                exmass.elem.rho[indexmass] = matdata.rho
                        elif word_ant == "JJ=":
                            exstiff.elem.JJ[indexmass] = float(word)
                        elif word_ant == "FW=":
                            exstiff.elem.FW[indexmass] = float(word)
                        elif word_ant == "xCG=":
                            exstiff.elem.xCG[indexmass] = float(word)
                            exmass.elem.xCG[indexmass] = float(word)
                        elif word_ant == "yCG=":
                            exstiff.elem.yCG[indexmass] = float(word)
                            exmass.elem.yCG[indexmass] = float(word)
                        word_ant = word 
            if fstiff == 0:             
                exstiff.elem.Ixx[indexmass] = 0
                exstiff.elem.Iyy[indexmass] = 0
                exstiff.elem.Ixy[indexmass] = 0
                exstiff.elem.EE[indexmass] = 0
                exstiff.elem.GG[indexmass] = 0
                exstiff.elem.JJ[indexmass] = 0
                exstiff.elem.FW[indexmass] = 0                
                exstiff.elem.xCG[indexmass] = 0             
                exstiff.elem.yCG[indexmass] = 0
    # define the properties for the concentrated mass
    for ii in np.arange(len(mesh_data.mesh_point_exmass)):
        if len(mesh_data.mesh_point_exmass[ii])>0:
            exmassfile = baseroot+mesh_data.mesh_point_exmass[ii]
            indexmass = mesh_data.mesh_point_exmassind[ii]
            word_ant   = []
            with open(exmassfile,"r") as textmass:
                for line in textmass:
                    for word in line.split():
                        if word == '%':
                            break
                        if word_ant == "m=":
                            exmass.point.mass[indexmass] = float(word)
                        elif word_ant == "Ixx=":
                            exmass.point.Ixx[indexmass] = float(word)
                        elif word_ant == "Iyy=":
                            exmass.point.Iyy[indexmass] = float(word)
                        elif word_ant == "Izz=":
                            exmass.point.Izz[indexmass] = float(word)
                        elif word_ant == "Ixy=":
                            exmass.point.Ixy[indexmass] = float(word)
                        elif word_ant == "Iyz=":
                            exmass.point.Iyz[indexmass] = float(word)
                        elif word_ant == "Izx=":
                            exmass.point.Izx[indexmass] = float(word)  
                        elif word_ant == "MAT=":
                            matdata = read_material(baseroot+word)
                            exstiff.point.EE[indexmass] = matdata.EE
                            exstiff.point.GG[indexmass] = matdata.GG
                        elif word_ant == "AA=":
                            exstiff.point.AA[indexmass] = float(word)
                        elif word_ant == "IIy=":
                            exstiff.point.IIy[indexmass] = float(word)
                        elif word_ant == "IIx=":
                            exstiff.point.IIx[indexmass] = float(word)
                        elif word_ant == "IIxy=":
                            exstiff.point.IIxy[indexmass] = float(word)
                        elif word_ant == "UIx=":
                            exstiff.point.UIx[indexmass] = float(word)
                        elif word_ant == "UIy=":
                            exstiff.point.UIy[indexmass] = float(word)
                        elif word_ant == "JJ=":
                            exstiff.point.JJ[indexmass] = float(word)
                        elif word_ant == "LONG=":
                            exstiff.point.LONG[indexmass] = float(word)                          
                        word_ant = word 
    return exmass, exstiff