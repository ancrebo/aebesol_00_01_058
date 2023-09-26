# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:32:25 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
read_section   : file containing functions for reading the section file
last_version   : 18-02-2021
modified_by    : Andres Cremades Botella
"""

import numpy as np
from bin.read_material import read_material, elastic_ortho, stiff_ortho, const_ortho
from bin.solve_warp import function_warp_TWO, function_warp_TWC
from bin.section_mass import func_cg_TW
from bin.aux_functions import def_vec_param

#%% Functions
def read_section(case_setup,mesh_data):
    # Function for reading the section mesh file and calculate its properties
    # case_setup : information of the setup
    # mesh_data  : information of the mesh
    # -------------------------------------------------------------------------
    # section       : class containing the information of the section
    #        - .ae_orig_x  : aerodynamic center x axis
    #        - .ae_orig_y  : aerodynamic center y axis
    #        - .type       : type of section open/closed
    #        - .nelem      : number of elements
    #        - .npoint     : number of points
    #        - .ncell      : number of cells
    #        - .elem       : elements of the section
    #        - .points     : nodes of the section
    #        - .cell       : cells of the section
    #        - .cg         : center of gravity of the section
    #        - .area_elem  : element area
    #        - .cg_local   : local center of gravity of the elements
    #        - .long       : length of the element
    #        - .ply_orient : orientation of the ply
    #        - .ply_thick  : thickness of the ply
    #        - .ply_ncord  : position of the ply
    #        - .ref        : number of refinements
    #        - .warp       : activate warping
    #        - .points_tot : total points of multiple section
    #        - .pointtot_id : identifier of the subsection
    #        - .le         : leading edge node
    #        - .te         : trailing edge node
    # sol_phys_node : solid physics of the nodes
    #        - .AA               : area of the section
    #        - .mass             : mass of the section
    #        - .EEL              : longitudinal young modulus of the fiber
    #        - .EET              : transverse young modulus of the fiber
    #        - .GGLT             : longitudinal-transverse shear modulus of the fiber
    #        - .nuLT             : longitudinal-transverse poisson ratio
    #        - .nuTL             : transverse-longitudinal pisson ratio
    #        - .rho              : density
    #        - .cg               : center of gravity
    #        - .aij              : element of row i and column j of the stiffness 2D matrix
    #        - .mij              : element of row i and column j of the mass 2D matrix
    #        - .aijglobal        : element of row i and column j of multi subsection section of the stiffness 2D matrix
    #        - .mijglobal        : element of row i and column j of multi subsection section of the mass 2D matrix
    #        - .warp             : warping function
    #        - .mass_gobal       : mass of the whole beam node
    #        - .cdg              : center of gravity of the whole beam node
    #        - .area_global_node : global area of the node
    prev_elements = []
    class section:
        pass
    class sol_phys_node:
        pass
    ref_sec_read                   = 0
    section.ae_orig_x              = []
    section.ae_orig_y              = []
    section.ref_x                  = []
    section.ref_y                  = []
    section.type                   = []
    section.nelem                  = []
    section.npoint                 = []
    section.ncell                  = []
    section.nbranch                = []
    section.elem                   = []
    section.points                 = []
    section.cell                   = []
    section.branch                 = []
    section.cg                     = [] 
    section.area_elem              = []
    section.cg_local               = []
    section.long                   = []
    section.ply_orient             = []
    section.ply_thick              = []
    section.nplies                 = []
    section.ply_ncord              = []
    section.le                     = []
    section.te                     = []
    section.ref                    = []
    section.warp                   = []
    section.points_tot             = []
    section.pointtot_id            = []
    section.elem_tot               = []
    section.elemtot_id             = []
    section.vec_normal             = []
    section.vec_ds                 = []
    sol_phys_node.AA               = []
    sol_phys_node.mass             = []
    sol_phys_node.EEL              = []
    sol_phys_node.EET              = []
    sol_phys_node.GGLT             = []
    sol_phys_node.nuLT             = []
    sol_phys_node.nuTL             = []
    sol_phys_node.rho              = []
    sol_phys_node.cg               = []
    sol_phys_node.a11              = []
    sol_phys_node.a12              = []
    sol_phys_node.a13              = []
    sol_phys_node.a14              = []
    sol_phys_node.a15              = []
    sol_phys_node.a16              = []
    sol_phys_node.a17              = []
    sol_phys_node.a22              = []
    sol_phys_node.a23              = []
    sol_phys_node.a24              = []
    sol_phys_node.a25              = []
    sol_phys_node.a26              = []
    sol_phys_node.a27              = []
    sol_phys_node.a33              = []
    sol_phys_node.a34              = []
    sol_phys_node.a35              = []
    sol_phys_node.a36              = []
    sol_phys_node.a37              = []
    sol_phys_node.a44              = []
    sol_phys_node.a45              = []
    sol_phys_node.a46              = []
    sol_phys_node.a47              = []
    sol_phys_node.a55              = []
    sol_phys_node.a56              = []
    sol_phys_node.a57              = []
    sol_phys_node.a66              = []
    sol_phys_node.a67              = []
    sol_phys_node.a77              = []
    sol_phys_node.m11              = []
    sol_phys_node.m22              = []
    sol_phys_node.m33              = []
    sol_phys_node.m44              = []
    sol_phys_node.m55              = []
    sol_phys_node.m66              = []
    sol_phys_node.m77              = []
    sol_phys_node.m16              = []
    sol_phys_node.m26              = []
    sol_phys_node.m34              = []
    sol_phys_node.m35              = []
    sol_phys_node.m37              = []
    sol_phys_node.m45              = []
    sol_phys_node.m47              = []
    sol_phys_node.m57              = []
    sol_phys_node.a11global        = []
    sol_phys_node.a12global        = []
    sol_phys_node.a13global        = []
    sol_phys_node.a14global        = []
    sol_phys_node.a15global        = []
    sol_phys_node.a16global        = []
    sol_phys_node.a17global        = []
    sol_phys_node.a22global        = []
    sol_phys_node.a23global        = []
    sol_phys_node.a24global        = []
    sol_phys_node.a25global        = []
    sol_phys_node.a26global        = []
    sol_phys_node.a27global        = []
    sol_phys_node.a33global        = []
    sol_phys_node.a34global        = []
    sol_phys_node.a35global        = []
    sol_phys_node.a36global        = []
    sol_phys_node.a37global        = []
    sol_phys_node.a44global        = []
    sol_phys_node.a45global        = []
    sol_phys_node.a46global        = []
    sol_phys_node.a47global        = []
    sol_phys_node.a55global        = []
    sol_phys_node.a56global        = []
    sol_phys_node.a57global        = []
    sol_phys_node.a66global        = []
    sol_phys_node.a67global        = []
    sol_phys_node.a77global        = []
    sol_phys_node.m11global        = []
    sol_phys_node.m22global        = []
    sol_phys_node.m33global        = []
    sol_phys_node.m44global        = []
    sol_phys_node.m55global        = []
    sol_phys_node.m66global        = []
    sol_phys_node.m77global        = []
    sol_phys_node.m16global        = []
    sol_phys_node.m26global        = []
    sol_phys_node.m34global        = []
    sol_phys_node.m35global        = []
    sol_phys_node.m37global        = []
    sol_phys_node.m45global        = []
    sol_phys_node.m47global        = []
    sol_phys_node.m57global        = []
    sol_phys_node.warp             = []
    sol_phys_node.mass_global      = np.zeros((mesh_data.num_point,))
    sol_phys_node.cdg              = np.zeros((mesh_data.num_point,2))
    sol_phys_node.area_global_node = np.zeros((mesh_data.num_point,))
    # fro every subsection of the mesh
    for ii_sec in np.arange(mesh_data.num_point):
        # section_composed : subsection information
        #      - .cg        : subsection center of gravity
        #      - .area_elem : element area
        #      - .cg_local  : local center of gravity of the element
        #      - .long      : lenght of the element
        class section_composed:
            pass
        section_composed.cg        = [] 
        section_composed.area_elem = []
        section_composed.cg_local  = []
        section_composed.long      = []
        # vec_normal_tot : normal vectors of the whole section
        vec_normal_tot = []
        # phys_sec_composed : physics of the solid in the subsections
        #        - .AA     : area of the section
        #        - .warp   : warping function
        #        - .r_t    : tangential distance from the reference to the element
        #        - .r_n    : tangential distance from the reference to the element
        #        - .mass   : mass of the section
        #        - .EEL    : longitudinal young modulus of the fiber
        #        - .EET    : transverse young modulus of the fiber
        #        - .GGLT   : longitudinal-transverse shear modulus of the fiber
        #        - .nuLT   : longitudinal-transverse poisson ratio
        #        - .nuTL   : transverse-longitudinal pisson ratio
        #        - .rho    : density
        #        - .aij    : element of row i and column j of the stiffness 2D matrix
        #        - .mij    : element of row i and column j of the mass 2D matrix
        class phys_sec_composed:
            pass  
        phys_sec_composed.AA   = []
        phys_sec_composed.warp = []
        phys_sec_composed.r_t  = []
        phys_sec_composed.r_n  = []
        phys_sec_composed.mass = []
        phys_sec_composed.EEL  = []
        phys_sec_composed.EET  = []
        phys_sec_composed.GGLT = []
        phys_sec_composed.nuLT = []
        phys_sec_composed.nuTL = []
        phys_sec_composed.rho  = []
        phys_sec_composed.a11  = []
        phys_sec_composed.a12  = []
        phys_sec_composed.a13  = []
        phys_sec_composed.a14  = []
        phys_sec_composed.a15  = []
        phys_sec_composed.a16  = []
        phys_sec_composed.a17  = []
        phys_sec_composed.a22  = []
        phys_sec_composed.a23  = []
        phys_sec_composed.a24  = []
        phys_sec_composed.a25  = []
        phys_sec_composed.a26  = []
        phys_sec_composed.a27  = []
        phys_sec_composed.a33  = []
        phys_sec_composed.a34  = []
        phys_sec_composed.a35  = []
        phys_sec_composed.a36  = []
        phys_sec_composed.a37  = []
        phys_sec_composed.a44  = []
        phys_sec_composed.a45  = []
        phys_sec_composed.a46  = []
        phys_sec_composed.a47  = []
        phys_sec_composed.a55  = []
        phys_sec_composed.a56  = []
        phys_sec_composed.a57  = []
        phys_sec_composed.a66  = []
        phys_sec_composed.a67  = []
        phys_sec_composed.a77  = []
        phys_sec_composed.m11  = []
        phys_sec_composed.m22  = []
        phys_sec_composed.m33  = []
        phys_sec_composed.m44  = []
        phys_sec_composed.m55  = []
        phys_sec_composed.m66  = []
        phys_sec_composed.m77  = []
        phys_sec_composed.m16  = []
        phys_sec_composed.m26  = []
        phys_sec_composed.m34  = []
        phys_sec_composed.m35  = []
        phys_sec_composed.m37  = []
        phys_sec_composed.m45  = []
        phys_sec_composed.m47  = []
        phys_sec_composed.m57  = []
        if not case_setup.problem_type == "AERO":
            # Print the percentage of sections that are read and calculated
            print('calculating section: '+str(ii_sec/mesh_data.num_point*100)+'%')
        # section_cell  : cells of the section
        # type_sec      : type of sections
        # warp_sec      : activate warping
        # nelem_sec     : number of elements of the section
        # npoint_sec    : number of points of the section
        # ncell_sec     : number of cells of the section
        # elem_sec      : elements of the section
        # point_sec     : points of the section
        # cell_sec      : cells of the section
        section_cell = []
        section_branch = []
        type_sec     = []
        warp_sec     = []
        nelem_sec    = []
        npoint_sec   = []
        ncell_sec    = []
        elem_sec     = []
        point_sec    = []
        cell_sec     = []
        nbranch_sec  = 0
        # warpflg : flag to indicate if the warping is activated
        warpflg = np.zeros((int(mesh_data.mesh_point_nsec[ii_sec]),))
        # forall the subsections of the beam node
        for jj_sec in np.arange(int(mesh_data.mesh_point_nsec[ii_sec])):
            # cellu_sec     : cells of the section
            # flg_elem      : flag to read elements
            # flg_point     : flag to read points
            # ii_elem       : index of the elements
            # ii_point      : index of the points
            # flg_cell      : flag to read cells
            # flg_savepoint : flag to store the points
            # flg_saveelem  : flag to store the elements
            # flg_savecell  : flag to store the cells
            # flg_ply       : flag to read a ply
            # flg_noply     : flag to indicate there are not plies
            # ii_cell       : index of the cell
            cellu_sec     = []
            branchu_sec   = []
            flg_elem      = 0
            flg_point     = 0
            ii_elem       = 0
            ii_point      = 0
            flg_cell      = 0
            flg_branch    = 0
            flg_savepoint = 0
            flg_saveelem  = 0
            flg_savecell  = 0
            flg_ply       = 0
            flg_noply     = 0
            ii_cell       = 0
            ii_branch     = 0
            # sectionuni  : class to store the information of the unitary section
            class sectionuni:
                pass
            # sec      : section name
            # scale    : scale fator of the section
            # rot_sec  : rotation of the section
            # word_ant : previous word
            sec      = mesh_data.mesh_point_profile[ii_sec][jj_sec]
            scale    = mesh_data.mesh_point_scale[ii_sec][jj_sec]
            rot_sec  = mesh_data.mesh_point_rot[ii_sec][jj_sec]
            xdisp    = mesh_data.mesh_point_xdisp[ii_sec][jj_sec]/100
            ydisp    = mesh_data.mesh_point_ydisp[ii_sec][jj_sec]/100
            word_ant = []
            # sol_physuni : class to store the information of the solid in the unitary section    
            class sol_physuni:
                pass
            # Read the file word by word
            with open(case_setup.root + sec,"r") as textsection:
                for line in textsection:
                    # If the flag of the cell lecture is activated read cell information
                    # section_cellu : unitary section information
                    if flg_cell==1 and kk > 0:
                        cellu_sec.append(section_cellu)
                        # if the last cell is read, store the information in section_cell
                        # activate the flag to store cell information and desactivate the flag to read cells
                        if ii_cell == ncell_sec-1:
                            section_cell.append(cellu_sec)
                            flg_savecell = 1
                            flg_cell     = 0
                        ii_cell+=1                       
                    if flg_branch==1 and kk > 0:
                        branchu_sec.append(section_branchu)
                        # if the last cell is read, store the information in section_cell
                        # activate the flag to store cell information and desactivate the flag to read cells
                        if ii_branch == nbranch_sec-1:
                            section_branch.append(branchu_sec)
                            flg_savebranch = 1
                            flg_branch     = 0
                        ii_branch+=1
                    # kk : lecture index
                    kk = 0
                    # Read each word in the line
                    for word in line.split():
                        # Avoid commentary lines
                        if word == "%":
                            break
                        # If the type of section is read
                        if word_ant == "TYPE=":
                            type_sec.append(word)
                            # When it is read for all the sections then store it 
                            if jj_sec == mesh_data.mesh_point_nsec[ii_sec]-1:
                                section.type.append(type_sec)
                        # Select if warp is restricted
                        elif word_ant == "WARP=":
                            warp_sec.append(word)
                            warpflg[jj_sec] = 1
                            # When it is read for all the sections then store it 
                            if jj_sec == mesh_data.mesh_point_nsec[ii_sec]-1:
                                section.warp.append(warp_sec)
                        # If the number of elements is read
                        elif word_ant == "NELEM=":
                            nelem_sec = int(word)
                            # When it is read for all the sections then store it 
                            if jj_sec == mesh_data.mesh_point_nsec[ii_sec]-1:
                                section.nelem.append(nelem_sec)
                            # create a matrix to store the element information and activate the lecture flag
                            section_elem = np.zeros((nelem_sec,4))
                            flg_elem     = 1
                        # If the number of points is read
                        elif word_ant == "NPOINT=":
                            npoint_sec = int(word)
                            # When it is read for all the sections then store it 
                            if jj_sec == mesh_data.mesh_point_nsec[ii_sec]-1:
                                section.npoint.append(npoint_sec)
                            # create a matrix to store the node information and activate the lecture flag
                            section_points = np.zeros((npoint_sec,2))
                            flg_point      = 1
                        # If the number of cells is read
                        elif word_ant == "NCELL=":
                            ncell_sec = int(word)
                            # When it is read for all the sections then store it 
                            if jj_sec == mesh_data.mesh_point_nsec[ii_sec]-1:
                                section.ncell.append(ncell_sec)
                            # activate the flag
                            flg_cell = 1
                        # If the number of branches is read
                        elif word_ant == "NBRANCH=":
                            nbranch_sec = int(word)
                            # When it is read for all the sections then store it 
                            if jj_sec == mesh_data.mesh_point_nsec[ii_sec]-1:
                                section.nbranch.append(nbranch_sec)
                            # activate the flag
                            flg_branch = 1
                        elif word_ant == "REF=":
                            ref_sec_read = int(word)
                            section.ref.append(ref_sec_read)
                        # If element lecture flag is activated and the number of elements does not exceed the maximum
                        elif flg_elem == 1 and ii_elem<nelem_sec:
                            # read element index
                            if kk==0:
                                section_elem[ii_elem,kk] = int(word)
                                kk                       = kk+1
                            # read node 1
                            elif kk==1:
                                section_elem[ii_elem,kk] = int(word)
                                kk                       = kk+1
                            # read node 2
                            elif kk==2:
                                section_elem[ii_elem,kk] = int(word)
                                kk                       = kk+1
                            # read thickness (thickness is not affected by the scale factor)
                            elif kk==3:
                                section_elem[ii_elem,kk] = float(word)
                                kk                       = kk+1
                                # save the element flag activated
                                flg_saveelem = 1  
                            # read the information about the plies
                            elif kk == 4:
                                # if ply option is activated
                                if word_ant == 'PLY{':
                                    # initialize the ply
                                    # section_orient     : orientation of the ply
                                    # section_thickply   : thickness of the ply
                                    # section_nplies     : number of plies
                                    # section_ply_ncoord : position of the plies
                                    # secction_material  : material of the ply
                                    # nmean_ply          : mean fiber of the ply
                                    if flg_ply == 0:
                                        section_orient     = []
                                        section_thickply   = []
                                        section_nplies     = np.zeros((nelem_sec,))
                                        section_ply_ncoord = []
                                        section_material   = []
                                        nmean_ply          = []
                                    # Activate the flag of the ply
                                    flg_ply = 1
                                elif word_ant == 'NOPLY{':
                                    if flg_noply == 0:
                                        section_orient     = []
                                        section_thickply   = []
                                        section_nplies     = np.zeros((nelem_sec,))
                                        section_ply_ncoord = []
                                        section_material   = []
                                        nmean_ply          = []
                                    # Activate the flag of no plies
                                    flg_noply = 1
                                # If there is no information about plies
                                elif (flg_ply < 1 or flg_noply < 1) and (word != 'PLY{' and word != 'NOPLY{'):
                                    ii_elem = ii_elem + 1
                            # If the flag for no plies is activated
                            # ply0                : initial position of the ply
                            # sec_ply_ncoord_elem : vector of the plies position
                            # ply1                : final position of the ply
                            # nmean_ply_elem      : mean position of the ply
                            if flg_noply == 1:
                                # Read the material, set the orientation to 0 and the ply thickness is the total thickness
                                if kk == 4:
                                    section_material.append([word])
                                    section_orient.append([0])
                                    thickfiber = [section_elem[ii_elem,3]]
                                    section_thickply.append(thickfiber)
                                    kk += 1
                                # If the no plies options are finished
                                elif word == '}':
                                    ply0                = 0
                                    sec_ply_ncoord_elem = np.zeros((1,2))
                                    ply1                = ply0 + section_thickply[ii_elem][0]
                                    sec_ply_ncoord_elem = [ply0, ply1]
                                    nmean_ply_elem      = []
                                    nmean_ply_elem.append((ply0+ply1)/2)
                                    nmean_ply.append(nmean_ply_elem)
                                    ply0                = ply1
                                    section_ply_ncoord.append(sec_ply_ncoord_elem)
                                    flg_noply           = -1
                                    kk                  = 0
                                    ii_elem             = ii_elem + 1
                                    section_nplies      = np.zeros((nelem_sec,))
                            if flg_ply == 1:
                                # Read the ply options
                                # orientation_fiber : orientation of the ply
                                # thick_fiber       : thickness of the ply
                                # material_ply      : material of the ply
                                if kk == 4:
                                    orientation_fiber       = [float(x) for x in word.split('/')]
                                    section_orient.append(orientation_fiber)
                                    section_nplies[ii_elem] = len(orientation_fiber)
                                    kk                     += 1
                                elif kk == 5:
                                    thick_fiber = [float(x) for x in word.split('/')]
                                    section_thickply.append(thick_fiber)
                                    kk         += 1
                                elif kk == 6:
                                    materialply = [x for x in word.split('_')]
                                    section_material.append(materialply)
                                    kk         += 1
                                elif word == '}':
                                    ply0                = 0
                                    sec_ply_ncoord_elem = np.zeros((len(section_thickply[ii_elem]),2))
                                    nmean_ply_elem      = []
                                    for kk_plies in np.arange(len(section_thickply[ii_elem])):
                                        ply1                            = ply0 + section_thickply[ii_elem][kk_plies]
                                        sec_ply_ncoord_elem[kk_plies,:] = [ply0, ply1]
                                        nmean_ply_elem.append((ply0+ply1)/2)
                                        ply0                            = ply1
                                    section_ply_ncoord.append(sec_ply_ncoord_elem)
                                    nmean_ply.append(nmean_ply_elem)
                                    flg_ply = -1
                                    kk      = 0
                                    ii_elem = ii_elem + 1
                        # read the points of the section
                        # sec_xoriginal  : coordinate x of the point
                        # sec_yoriginal  : coordinate y of the point
                        # section_points : information of the points of the section 
                        elif flg_point == 1 and ii_point < npoint_sec:
                            if kk == 0:
                                kk = kk+1
                            elif kk == 1:
                                sec_xoriginal = scale*float(word)
                                kk            = kk+1
                            elif kk==2:
                                sec_yoriginal                 = scale*float(word)
                                section_points[ii_point,kk-2] = (sec_xoriginal-xdisp*scale)*np.cos(rot_sec*np.pi/180)-(sec_yoriginal-ydisp*scale)*np.sin(rot_sec*np.pi/180)
                                section_points[ii_point,kk-1] = (sec_yoriginal-ydisp*scale)*np.cos(rot_sec*np.pi/180)+(sec_xoriginal-xdisp*scale)*np.sin(rot_sec*np.pi/180)
                                kk                            = kk+1
                                ii_point                      = ii_point + 1
                                flg_savepoint                 = 1
                        # Read the cell information
                        elif flg_cell == 1 and ii_cell<ncell_sec:
                            if kk==0:
                                section_cellu = []
                                kk           += 1
                            else:
                                section_cellu.append(int(word))
                                kk           += 1
                        # Read the cell information
                        elif flg_branch == 1 and ii_cell<nbranch_sec:
                            if kk==0:
                                section_branchu = []
                                kk           += 1
                            else:
                                section_branchu.append(int(word))
                                kk           += 1
                        if jj_sec == 0:
                            # If the areodynamic center x axis position is read
                            if word_ant == "AE_ORIG_X=":
                                aesec_x = scale*float(word)
                            # If the areodynamic center y axis position is read
                            elif word_ant == "AE_ORIG_Y=":
                                aesec_y = scale*float(word)
                                section.ae_orig_x.append((aesec_x-xdisp*scale)*np.cos(rot_sec*np.pi/180)-(aesec_y-ydisp*scale)*np.sin(rot_sec*np.pi/180))
                                section.ae_orig_y.append((aesec_y-ydisp*scale)*np.cos(rot_sec*np.pi/180)+(aesec_x-xdisp*scale)*np.sin(rot_sec*np.pi/180))
                            elif word_ant == "LE=":
                                section.le.append(int(word))
                            elif word_ant == "TE=":
                                section.te.append(int(word))
                        # Update the last word                                
                        word_ant = word
            if warpflg[jj_sec] == 0:
                warp_sec.append('YES')
                # When it is read for all the sections then store it 
                if jj_sec == mesh_data.mesh_point_nsec[ii_sec]-1:
                    section.warp.append(warp_sec)
            # Refine the section mesh if required           
            try:
                # refmesh : number of refinements
                refmesh = ref_sec_read
                # repite if refmesh is higher than 0
                while refmesh > 0:
                    # point2              : copy of the point matrix
                    # elem2               : matrix of elements with the double number than the original
                    # section_orient2     : orientation of the plies
                    # section_thickply2   : thickness of the plies
                    # section_nplies      : number of plies 
                    # section_ply_ncoord2 : coordinates of the ply
                    # section_material2   : material of the section
                    # nmean_ply2          : mean position of the ply
                    # section_cell2       : cells of the section
                    # kk                  : index
                    point2              = section_points.copy()
                    elem2               = np.zeros((2*len(section_elem),4))
                    section_orient2     = [] #section_orient.copy()
                    section_thickply2   = [] #section_thickply.copy()
                    section_nplies2     = np.zeros((2*len(section_elem),))
                    section_ply_ncoord2 = [] #section_ply_ncoord.copy()
                    section_material2   = [] #section_material.copy()
                    nmean_ply2          = [] #nmean_ply.copy()
                    try:
                        section_cell2       = def_vec_param(len(section_cell[0]))
                        section_branch2     = def_vec_param(len(section_branch[0]))
                    except:
                        pass
                    kk                  = 0
                    for elem in section_elem:
                    # for all the elements separate the element one row in the elem2 matrix
                    # in the space between create a new element with an extra point
                        elem2[2*kk,0]   = 2*kk
                        elem2[2*kk,1]   = elem[1]
                        elem2[2*kk,2]   = len(point2)
                        elem2[2*kk,3]   = elem[3]
                        elem2[2*kk+1,0] = 2*kk+1
                        elem2[2*kk+1,1] = len(point2)
                        elem2[2*kk+1,2] = elem[2]
                        elem2[2*kk+1,3] = elem[3]
                        # add a new point and take the properties of the new eleements from the previous one
                        point2                         = np.concatenate((point2,[(section_points[int(elem[1]),:]+section_points[int(elem[2]),:])/2]),0)
                        section_nplies2[2*kk]  = section_nplies[int(elem[0])]
                        section_nplies2[2*kk+1] = section_nplies[int(elem[0])]
                        section_orient2.append(section_orient[int(elem[0])])
                        section_orient2.append(section_orient[int(elem[0])])
                        section_thickply2.append(section_thickply[int(elem[0])])
                        section_thickply2.append(section_thickply[int(elem[0])])
                        section_ply_ncoord2.append(section_ply_ncoord[int(elem[0])].copy())
                        section_ply_ncoord2.append(section_ply_ncoord[int(elem[0])].copy())
                        section_material2.append(section_material[int(elem[0])].copy())
                        section_material2.append(section_material[int(elem[0])].copy())
                        nmean_ply2.append(nmean_ply[int(elem[0])].copy())
                        nmean_ply2.append(nmean_ply[int(elem[0])].copy())
                        kk += 1
                    if type_sec[jj_sec] == 'THIN_WALL_C':
                        # Calculate the new section cells
                        # section_cell3 : new section cells
                        kk            = 0
                        section_cell3 = []
                        # For all the elements and cells 
                        for elem in section_elem:
                            for ii_secref in np.arange(len(section_cell[0])):
                                secref = section_cell[0][ii_secref]
                                # if the element is contained in the section cell append two elements to the cell
                                if len(np.where(np.array(secref) == int(elem[0]))[0]) > 0:
                                    section_cell2[ii_secref].append(2*kk)
                                    section_cell2[ii_secref].append(2*kk+1)
                            kk += 1
                        section_cell2b = def_vec_param(len(section_cell[0]))
                        for cellaux in np.arange(len(section_cell2)):
                            usednode = []
                            revelem = False
                            while len(usednode) < len(section_cell2[cellaux]):
                                for elemaux in section_cell2[cellaux]:
                                    if len(section_cell2b[cellaux]) == 0:
                                        section_cell2b[cellaux].append(elemaux)
                                        usednode.append(elemaux)
                                        break
                                    else:
                                        node_aux = elem2[elemaux,1:3]
                                        node_ant = elem2[section_cell2b[cellaux][-1],1:3]
                                        if revelem:
                                            nodeant0 = node_ant[0]
                                            nodeant1 = node_ant[1]
                                            node_ant = np.array([nodeant1,nodeant0])
                                        if (node_aux[0] == node_ant[1] or node_aux[1] == node_ant[1]) and len(np.where(np.array(usednode) == elemaux)[0])==0:
                                            section_cell2b[cellaux].append(elemaux)
                                            usednode.append(elemaux)
                                            if node_aux[1] == node_ant[1]:
                                                revelem = True
                                            else:
                                                revelem = False
                                            break
                                
                        section_cell3.append(section_cell2b)
                        kk            = 0
                        section_branch3 = []
                        # For all the elements and cells
                        if nbranch_sec > 0:
                            for elem in section_elem:
                                for ii_secref in np.arange(len(section_branch[0])):
                                    secref = section_branch[0][ii_secref]
                                    # if the element is contained in the section cell append two elements to the cell
                                    if len(np.where(np.array(secref) == int(elem[0]))[0]) > 0:
                                        section_branch2[ii_secref].append(2*kk)
                                        section_branch2[ii_secref].append(2*kk+1)
                                kk += 1
                            section_branch3.append(section_branch2)
                        kk = 0
                        section_cell       = section_cell3.copy()
                        section_branch     = section_branch3.copy()
                    # Update class mesh_data
                    section_elem       = elem2.copy()
                    section_points     = point2.copy()
                    section_orient     = section_orient2.copy()
                    section_thickply   = section_thickply2.copy()
                    section_nplies     = section_nplies2.copy()
                    section_ply_ncoord = section_ply_ncoord2.copy()
                    section_material   = section_material2.copy()
                    nmean_ply          = nmean_ply2.copy()
                    # reduce the number of required refinements            
                    refmesh -= 1
            except:
                pass

            section.ref_x.append((-xdisp*scale)*np.cos(rot_sec*np.pi/180)-(-ydisp*scale)*np.sin(rot_sec*np.pi/180))
            section.ref_y.append((-ydisp*scale)*np.cos(rot_sec*np.pi/180)+(-xdisp*scale)*np.sin(rot_sec*np.pi/180))
            
            #%%
            # Save the element information in the element list
            if flg_saveelem == 1:
                elem_sec.append(section_elem)
            # Save the node information in the node list
            if flg_savepoint == 1:
                point_sec.append(section_points)
            if not case_setup.problem_type == "AERO":
                # mat : material information
                #     - .EEL  : longitudinal young modulus
                #     - .EET  : transverse young modulus
                #     - .GGLT : longitudinal-transverse shear modulus
                #     - .GGTN : transverse-transverse shear modulus
                #     - .nu   : poisson ratio
                #     - .nuLT : longitudinal-transverse poisson ratio
                #     - .nuTL : transvese-longitudinal poisson ratio
                #     - .nuTN : transverse-transverse poisson ratio
                mat = def_vec_param(len(section_material))
                # for all the plies and elements of the section
                for ii_secmat in np.arange(len(section_material)):
                    for jj_secmat in np.arange(len(section_material[ii_secmat])):
                        mat[ii_secmat].append(read_material(case_setup.root + section_material[ii_secmat][jj_secmat]))
                        # if the material is isotropic convert the properties to orthotropic
                        if mat[ii_secmat][jj_secmat].typemat == "ISO":
                            mat[ii_secmat][jj_secmat].EEL  = mat[ii_secmat][jj_secmat].EE
                            mat[ii_secmat][jj_secmat].EET  = mat[ii_secmat][jj_secmat].EE
                            mat[ii_secmat][jj_secmat].GGLT = mat[ii_secmat][jj_secmat].GG
                            mat[ii_secmat][jj_secmat].GGTN = mat[ii_secmat][jj_secmat].GG
                            mat[ii_secmat][jj_secmat].nu   = mat[ii_secmat][jj_secmat].EE/(2*mat[ii_secmat][jj_secmat].GG)-1
                            mat[ii_secmat][jj_secmat].nuLT = mat[ii_secmat][jj_secmat].nu
                            mat[ii_secmat][jj_secmat].nuTL = mat[ii_secmat][jj_secmat].nu
                            mat[ii_secmat][jj_secmat].nuTN = mat[ii_secmat][jj_secmat].nu
                # Calculate orthotrpic constants of the material in the section
                sol_physuni, section_ply_ncoord = const_ortho(case_setup,sol_physuni,section_elem,section_orient,section_thickply,section_nplies,nmean_ply,section_ply_ncoord,mat)
                # Calculate elastic constants of the section
                sol_physuni = elastic_ortho(case_setup,sol_physuni,section_elem,section_points,section_orient,section_thickply,section_nplies,section_ply_ncoord,mat)
                # Store the material properties in the solid physics class
                sol_physuni.EEL  = [] #np.zeros((len(mat),len(mat[0])))
                sol_physuni.EET  = [] #np.zeros((len(mat),len(mat[0])))
                sol_physuni.GGLT = [] #np.zeros((len(mat),len(mat[0])))
                sol_physuni.nuLT = [] #np.zeros((len(mat),len(mat[0])))
                sol_physuni.nuTL = [] #np.zeros((len(mat),len(mat[0])))
                sol_physuni.rho  = [] #np.zeros((len(mat),len(mat[0])))
                # for every element and ply                
                for ii_mat in np.arange(len(mat)):
                    EELu = []
                    EETu = []
                    GGLTu = []
                    nuLTu = []
                    nuTLu = []
                    rhou = []
                    for jj_mat in np.arange(len(mat[ii_mat])):
                        EELu.append(mat[ii_mat][jj_mat].EEL)
                        EETu.append(mat[ii_mat][jj_mat].EET)
                        GGLTu.append(mat[ii_mat][jj_mat].GGLT)
                        nuLTu.append(mat[ii_mat][jj_mat].nuLT)
                        nuTLu.append(mat[ii_mat][jj_mat].nuTL)
                        rhou.append(mat[ii_mat][jj_mat].rho)
                    sol_physuni.EEL.append(EELu)
                    sol_physuni.EET.append(EETu)
                    sol_physuni.GGLT.append(GGLTu)
                    sol_physuni.nuLT.append(nuLTu)
                    sol_physuni.nuTL.append(nuTLu)
                    sol_physuni.rho.append(rhou)             
    #                    sol_physuni.EEL[ii_mat,jj_mat]  = mat[ii_mat][jj_mat].EEL
    #                    sol_physuni.EET[ii_mat,jj_mat]  = mat[ii_mat][jj_mat].EET
    #                    sol_physuni.GGLT[ii_mat,jj_mat] = mat[ii_mat][jj_mat].GGLT
    #                    sol_physuni.nuLT[ii_mat,jj_mat] = mat[ii_mat][jj_mat].nuLT
    #                    sol_physuni.nuTL[ii_mat,jj_mat] = mat[ii_mat][jj_mat].nuTL
    #                    sol_physuni.rho[ii_mat,jj_mat]  = mat[ii_mat][jj_mat].rho 
                phys_sec_composed.EEL.append(sol_physuni.EEL)
                phys_sec_composed.EET.append(sol_physuni.EET)
                phys_sec_composed.GGLT.append(sol_physuni.GGLT)
                phys_sec_composed.nuLT.append(sol_physuni.nuLT)
                phys_sec_composed.nuTL.append(sol_physuni.nuTL)
                phys_sec_composed.rho.append(sol_physuni.rho)
                # For open section beams
                if type_sec[jj_sec]  == "THIN_WALL_O":
                    # sectionuni : class for the unitary section information
                    #       - cg        : center of gravity
                    #       - area_elem : element area
                    #       - cg_local  : local center of gravity
                    #       - long      : length of the element
                    #       - AA        : area of the section
                    class sectionuni:
                        pass
                    # calculate the area and mass properties
                    sectionuni.cg, sectionuni.area_elem, sectionuni.cg_local, sectionuni.long, sol_physuni.AA, acell_del, bcell_del, sol_physuni.mass, L_cell_del, psi_cell_del = func_cg_TW(section_elem,section_points,[],sol_physuni,section_thickply,section_ply_ncoord)
                    section_composed.cg.append(sectionuni.cg)
                    section_composed.area_elem.append(sectionuni.area_elem)
                    section_composed.cg_local.append(sectionuni.cg_local)
                    section_composed.long.append(sectionuni.long)
                    phys_sec_composed.AA.append(sol_physuni.AA)
                    phys_sec_composed.mass.append(sol_physuni.mass)
                    sol_physuni.warp, sol_physuni.r_t, sol_physuni.r_n, vec_n, vec_ds =  function_warp_TWO(section_elem,section_points,section_cell,sol_physuni,sectionuni)
                    # calculate the warping functions
                    if warp_sec[jj_sec] == "YES":
                        phys_sec_composed.warp.append(sol_physuni.warp)
                    else:
                        sol_physuni.warp =  np.zeros((len(section_elem),))
                        phys_sec_composed.warp.append(sol_physuni.warp)
                    phys_sec_composed.r_t.append(sol_physuni.r_t)
                    phys_sec_composed.r_n.append(sol_physuni.r_n)
                    if len(vec_normal_tot)==0:
                        vec_normal_tot = vec_n
                        vec_ds_tot = vec_ds
                    else:
                        vec_normal_tot = np.concatenate((vec_normal_tot,vec_n))
                        vec_ds_tot = np.concatenate((vec_ds_tot,vec_ds))
                    # Calculate the stiffness and mass 2D matrices
                    sol_physuni = stiff_ortho(sol_physuni,section_elem,section_points,sectionuni,mat,[], [],section_nplies)
                    phys_sec_composed.a11.append(sol_physuni.a11)
                    phys_sec_composed.a12.append(sol_physuni.a12)
                    phys_sec_composed.a13.append(sol_physuni.a13)
                    phys_sec_composed.a14.append(sol_physuni.a14)
                    phys_sec_composed.a15.append(sol_physuni.a15)
                    phys_sec_composed.a16.append(sol_physuni.a16)
                    phys_sec_composed.a17.append(sol_physuni.a17)
                    phys_sec_composed.a22.append(sol_physuni.a22)
                    phys_sec_composed.a23.append(sol_physuni.a23)
                    phys_sec_composed.a24.append(sol_physuni.a24)
                    phys_sec_composed.a25.append(sol_physuni.a25)
                    phys_sec_composed.a26.append(sol_physuni.a26)
                    phys_sec_composed.a27.append(sol_physuni.a27)
                    phys_sec_composed.a33.append(sol_physuni.a33)
                    phys_sec_composed.a34.append(sol_physuni.a34)
                    phys_sec_composed.a35.append(sol_physuni.a35)
                    phys_sec_composed.a36.append(sol_physuni.a36)
                    phys_sec_composed.a37.append(sol_physuni.a37)
                    phys_sec_composed.a44.append(sol_physuni.a44)
                    phys_sec_composed.a45.append(sol_physuni.a45)
                    phys_sec_composed.a46.append(sol_physuni.a46)
                    phys_sec_composed.a47.append(sol_physuni.a47)
                    phys_sec_composed.a55.append(sol_physuni.a55)
                    phys_sec_composed.a56.append(sol_physuni.a56)
                    phys_sec_composed.a57.append(sol_physuni.a57)
                    phys_sec_composed.a66.append(sol_physuni.a66)
                    phys_sec_composed.a67.append(sol_physuni.a67)
                    phys_sec_composed.a77.append(sol_physuni.a77)
                    phys_sec_composed.m11.append(sol_physuni.m11)
                    phys_sec_composed.m22.append(sol_physuni.m22)
                    phys_sec_composed.m33.append(sol_physuni.m33)
                    phys_sec_composed.m44.append(sol_physuni.m44)
                    phys_sec_composed.m55.append(sol_physuni.m55)
                    phys_sec_composed.m66.append(sol_physuni.m66)
                    phys_sec_composed.m77.append(sol_physuni.m77)
                    phys_sec_composed.m16.append(sol_physuni.m16)
                    phys_sec_composed.m26.append(sol_physuni.m26)
                    phys_sec_composed.m34.append(sol_physuni.m34)
                    phys_sec_composed.m35.append(sol_physuni.m35)
                    phys_sec_composed.m37.append(sol_physuni.m37)
                    phys_sec_composed.m45.append(sol_physuni.m45)
                    phys_sec_composed.m47.append(sol_physuni.m47)
                    phys_sec_composed.m57.append(sol_physuni.m57)
                # for closed section beams
                elif type_sec[jj_sec]  == "THIN_WALL_C":
                    # sectionuni : class for the unitary section information
                    #       - cg        : center of gravity
                    #       - area_elem : element area
                    #       - cg_local  : local center of gravity
                    #       - long      : length of the element
                    #       - AA        : area of the section
                    class sectionuni:
                        pass
                    sectionuni.cg, sectionuni.area_elem, sectionuni.cg_local, sectionuni.long, sol_physuni.AA, sol_physuni.area_cells, sol_physuni.beta_cells,sol_physuni.mass, sol_physuni.L_cells, sol_physuni.psi_cells = func_cg_TW(section_elem,section_points,section_cell[jj_sec],sol_physuni,section_thickply,section_ply_ncoord)
                    section_composed.cg.append(sectionuni.cg)
                    section_composed.area_elem.append(sectionuni.area_elem)
                    section_composed.cg_local.append(sectionuni.cg_local)
                    section_composed.long.append(sectionuni.long)
                    phys_sec_composed.AA.append(sol_physuni.AA)
                    phys_sec_composed.mass.append(sol_physuni.mass)
                    # calculate the warping functions
                    if len(section_branch) == 0:
                        sbran = []
                    else:
                        sbran = section_branch[jj_sec]
                    sol_physuni.warp, sol_physuni.r_t, sol_physuni.r_n, vec_n, vec_ds, sol_physuni.H_mat, sol_physuni.H_mat13, sol_physuni.H_mat43, sol_physuni.H_mat53, sol_physuni.lambda_sec,sol_physuni.mat_cell_elem,prev_elements =  function_warp_TWC(section_elem,section_points,section_cell[jj_sec],sbran,sol_physuni,sectionuni,prev_elements,mesh_data.keepwarp)
                    if warp_sec[jj_sec] == "YES":
                        phys_sec_composed.warp.append(sol_physuni.warp)
                    else:
                        sol_physuni.warp =  np.zeros((len(section_elem),))
                        phys_sec_composed.warp.append(sol_physuni.warp)
                    phys_sec_composed.r_t.append(sol_physuni.r_t)
                    phys_sec_composed.r_n.append(sol_physuni.r_n)
                    if len(vec_normal_tot)==0:
                        vec_normal_tot = vec_n
                        vec_ds_tot = vec_ds
                    else:
                        vec_normal_tot = np.concatenate((vec_normal_tot,vec_n))
                        vec_ds_tot = np.concatenate((vec_ds_tot,vec_ds))
                    # Calculate the stiffness and mass 2D matrices
                    sol_physuni = stiff_ortho(sol_physuni,section_elem,section_points,sectionuni,mat,section_cell[jj_sec],sbran,section_nplies)
                    phys_sec_composed.a11.append(sol_physuni.a11)
                    phys_sec_composed.a12.append(sol_physuni.a12)
                    phys_sec_composed.a13.append(sol_physuni.a13)
                    phys_sec_composed.a14.append(sol_physuni.a14)
                    phys_sec_composed.a15.append(sol_physuni.a15)
                    phys_sec_composed.a16.append(sol_physuni.a16)
                    phys_sec_composed.a17.append(sol_physuni.a17)
                    phys_sec_composed.a22.append(sol_physuni.a22)
                    phys_sec_composed.a23.append(sol_physuni.a23)
                    phys_sec_composed.a24.append(sol_physuni.a24)
                    phys_sec_composed.a25.append(sol_physuni.a25)
                    phys_sec_composed.a26.append(sol_physuni.a26)
                    phys_sec_composed.a27.append(sol_physuni.a27)
                    phys_sec_composed.a33.append(sol_physuni.a33)
                    phys_sec_composed.a34.append(sol_physuni.a34)
                    phys_sec_composed.a35.append(sol_physuni.a35)
                    phys_sec_composed.a36.append(sol_physuni.a36)
                    phys_sec_composed.a37.append(sol_physuni.a37)
                    phys_sec_composed.a44.append(sol_physuni.a44)
                    phys_sec_composed.a45.append(sol_physuni.a45)
                    phys_sec_composed.a46.append(sol_physuni.a46)
                    phys_sec_composed.a47.append(sol_physuni.a47)
                    phys_sec_composed.a55.append(sol_physuni.a55)
                    phys_sec_composed.a56.append(sol_physuni.a56)
                    phys_sec_composed.a57.append(sol_physuni.a57)
                    phys_sec_composed.a66.append(sol_physuni.a66)
                    phys_sec_composed.a67.append(sol_physuni.a67)
                    phys_sec_composed.a77.append(sol_physuni.a77)
                    phys_sec_composed.m11.append(sol_physuni.m11)
                    phys_sec_composed.m22.append(sol_physuni.m22)
                    phys_sec_composed.m33.append(sol_physuni.m33)
                    phys_sec_composed.m44.append(sol_physuni.m44)
                    phys_sec_composed.m55.append(sol_physuni.m55)
                    phys_sec_composed.m66.append(sol_physuni.m66)
                    phys_sec_composed.m77.append(sol_physuni.m77)
                    phys_sec_composed.m16.append(sol_physuni.m16)
                    phys_sec_composed.m26.append(sol_physuni.m26)
                    phys_sec_composed.m34.append(sol_physuni.m34)
                    phys_sec_composed.m35.append(sol_physuni.m35)
                    phys_sec_composed.m37.append(sol_physuni.m37)
                    phys_sec_composed.m45.append(sol_physuni.m45)
                    phys_sec_composed.m47.append(sol_physuni.m47)
                    phys_sec_composed.m57.append(sol_physuni.m57)
        # Add the information to the section and sol_phys_node class
        section.cg.append(section_composed.cg)
        section.area_elem.append(section_composed.area_elem)
        section.cg_local.append(section_composed.cg_local)
        section.long.append(section_composed.long)
        section.elem.append(elem_sec)
        section.points.append(point_sec)
        section.cell.append(section_cell)
        section.ply_orient.append(section_orient)
        section.ply_thick.append(section_thickply)
        section.nplies.append(section_nplies)
        section.ply_ncord.append(section_ply_ncoord)
        section.vec_normal.append(vec_normal_tot)
        sol_phys_node.AA.append(phys_sec_composed.AA)
        sol_phys_node.mass.append(phys_sec_composed.mass)
        sol_phys_node.cg.append(section_composed.cg)
        sol_phys_node.EEL.append(phys_sec_composed.EEL)
        sol_phys_node.EET.append(phys_sec_composed.EET)
        sol_phys_node.GGLT.append(phys_sec_composed.GGLT)
        sol_phys_node.nuLT.append(phys_sec_composed.nuLT)
        sol_phys_node.nuTL.append(phys_sec_composed.nuTL)
        sol_phys_node.rho.append(phys_sec_composed.rho)
        sol_phys_node.a11.append(phys_sec_composed.a11)
        sol_phys_node.a12.append(phys_sec_composed.a12)
        sol_phys_node.a13.append(phys_sec_composed.a13)
        sol_phys_node.a14.append(phys_sec_composed.a14)
        sol_phys_node.a15.append(phys_sec_composed.a15)
        sol_phys_node.a16.append(phys_sec_composed.a16)
        sol_phys_node.a17.append(phys_sec_composed.a17)
        sol_phys_node.a22.append(phys_sec_composed.a22)
        sol_phys_node.a23.append(phys_sec_composed.a23)
        sol_phys_node.a24.append(phys_sec_composed.a24)
        sol_phys_node.a25.append(phys_sec_composed.a25)
        sol_phys_node.a26.append(phys_sec_composed.a26)
        sol_phys_node.a27.append(phys_sec_composed.a27)
        sol_phys_node.a33.append(phys_sec_composed.a33)
        sol_phys_node.a34.append(phys_sec_composed.a34)
        sol_phys_node.a35.append(phys_sec_composed.a35)
        sol_phys_node.a36.append(phys_sec_composed.a36)
        sol_phys_node.a37.append(phys_sec_composed.a37)
        sol_phys_node.a44.append(phys_sec_composed.a44)
        sol_phys_node.a45.append(phys_sec_composed.a45)
        sol_phys_node.a46.append(phys_sec_composed.a46)
        sol_phys_node.a47.append(phys_sec_composed.a47)
        sol_phys_node.a55.append(phys_sec_composed.a55)
        sol_phys_node.a56.append(phys_sec_composed.a56)
        sol_phys_node.a57.append(phys_sec_composed.a57)
        sol_phys_node.a66.append(phys_sec_composed.a66)
        sol_phys_node.a67.append(phys_sec_composed.a67)
        sol_phys_node.a77.append(phys_sec_composed.a77)
        sol_phys_node.m11.append(phys_sec_composed.m11)
        sol_phys_node.m22.append(phys_sec_composed.m22)
        sol_phys_node.m33.append(phys_sec_composed.m33)
        sol_phys_node.m44.append(phys_sec_composed.m44)
        sol_phys_node.m55.append(phys_sec_composed.m55)
        sol_phys_node.m66.append(phys_sec_composed.m66)
        sol_phys_node.m77.append(phys_sec_composed.m77)
        sol_phys_node.m16.append(phys_sec_composed.m16)
        sol_phys_node.m26.append(phys_sec_composed.m26)
        sol_phys_node.m34.append(phys_sec_composed.m34)
        sol_phys_node.m35.append(phys_sec_composed.m35)
        sol_phys_node.m37.append(phys_sec_composed.m37)
        sol_phys_node.m45.append(phys_sec_composed.m45)
        sol_phys_node.m47.append(phys_sec_composed.m47)
        sol_phys_node.m57.append(phys_sec_composed.m57)
        sol_phys_node.warp.append(phys_sec_composed.warp)
        if not case_setup.problem_type == "AERO":
            section.vec_ds.append(vec_ds_tot)
#    section.cec_global = np.zeros((mesh_data.num_point,2))
    
    # sol_phys : class to store the solid physics in the beam node
    #        - .aij  : stiffness element of row i and column j
    #        - .mij  : mass element of row i and column j
    #        - .warp : warping function
    #        - .cdg  : center of gravity
    class sol_phys:
        pass
    sol_phys.warp = sol_phys_node.warp 
    sol_phys.xsc_sh  = np.zeros((mesh_data.num_point,))  
    sol_phys.ysc_sh  = np.zeros((mesh_data.num_point,))  
    sol_phys.xsc_nosh  = np.zeros((mesh_data.num_point,))  
    sol_phys.ysc_nosh  = np.zeros((mesh_data.num_point,)) 
    sol_phys.a11  = np.zeros((mesh_data.num_elem,)) 
    sol_phys.a12  = np.zeros((mesh_data.num_elem,))
    sol_phys.a13  = np.zeros((mesh_data.num_elem,))
    sol_phys.a14  = np.zeros((mesh_data.num_elem,))
    sol_phys.a15  = np.zeros((mesh_data.num_elem,))
    sol_phys.a16  = np.zeros((mesh_data.num_elem,))
    sol_phys.a17  = np.zeros((mesh_data.num_elem,))  
    sol_phys.a22  = np.zeros((mesh_data.num_elem,))
    sol_phys.a23  = np.zeros((mesh_data.num_elem,))
    sol_phys.a24  = np.zeros((mesh_data.num_elem,))
    sol_phys.a25  = np.zeros((mesh_data.num_elem,))
    sol_phys.a26  = np.zeros((mesh_data.num_elem,))
    sol_phys.a27  = np.zeros((mesh_data.num_elem,)) 
    sol_phys.a33  = np.zeros((mesh_data.num_elem,))
    sol_phys.a34  = np.zeros((mesh_data.num_elem,))
    sol_phys.a35  = np.zeros((mesh_data.num_elem,))
    sol_phys.a36  = np.zeros((mesh_data.num_elem,))
    sol_phys.a37  = np.zeros((mesh_data.num_elem,))
    sol_phys.a44  = np.zeros((mesh_data.num_elem,))
    sol_phys.a45  = np.zeros((mesh_data.num_elem,))
    sol_phys.a46  = np.zeros((mesh_data.num_elem,))
    sol_phys.a47  = np.zeros((mesh_data.num_elem,))
    sol_phys.a55  = np.zeros((mesh_data.num_elem,))
    sol_phys.a56  = np.zeros((mesh_data.num_elem,))
    sol_phys.a57  = np.zeros((mesh_data.num_elem,))
    sol_phys.a66  = np.zeros((mesh_data.num_elem,))
    sol_phys.a67  = np.zeros((mesh_data.num_elem,))
    sol_phys.a77  = np.zeros((mesh_data.num_elem,))
    sol_phys.m11  = np.zeros((mesh_data.num_elem,)) 
    sol_phys.m16  = np.zeros((mesh_data.num_elem,))
    sol_phys.m22  = np.zeros((mesh_data.num_elem,))
    sol_phys.m26  = np.zeros((mesh_data.num_elem,))
    sol_phys.m33  = np.zeros((mesh_data.num_elem,))
    sol_phys.m34  = np.zeros((mesh_data.num_elem,))
    sol_phys.m35  = np.zeros((mesh_data.num_elem,))
    sol_phys.m37  = np.zeros((mesh_data.num_elem,))
    sol_phys.m44  = np.zeros((mesh_data.num_elem,))
    sol_phys.m45  = np.zeros((mesh_data.num_elem,))
    sol_phys.m47  = np.zeros((mesh_data.num_elem,))
    sol_phys.m55  = np.zeros((mesh_data.num_elem,))
    sol_phys.m57  = np.zeros((mesh_data.num_elem,))
    sol_phys.m66  = np.zeros((mesh_data.num_elem,))
    sol_phys.m77  = np.zeros((mesh_data.num_elem,))
    sol_phys.cdg  = []
    for ii in np.arange(mesh_data.num_point):
        lenelem = 0
        for jj in np.arange(mesh_data.mesh_point_nsec[ii]):
            lenelem += len(section.elem[int(ii)][int(jj)])
        sec_elem_tot  = np.zeros((lenelem,3))
        sec_el_id     = np.zeros((lenelem,))
        ll_jj = 0
        for jj in np.arange(mesh_data.mesh_point_nsec[ii]):
            jj     = int(jj)
            ll_jj2 = ll_jj 
            ll_jj += len(section.elem[int(ii)][int(jj)])
            if ll_jj2 == 0:
                sec_elem_tot[:ll_jj,:] = section.elem[ii][int(jj)][:,:3]
                sec_el_id[:ll_jj]      = int(jj)
            elif ll_jj == lenelem:
                sec_elem_tot[ll_jj2:,:] = section.elem[ii][int(jj)][:,:3]
                sec_el_id[ll_jj2:]      = int(jj)
            else:
                sec_elem_tot[ll_jj2:ll_jj,:] = section.elem[ii][int(jj)][:,:3]
                sec_el_id[ll_jj2:ll_jj]      = int(jj)
        section.elem_tot.append(sec_elem_tot)
        section.elemtot_id.append(sec_el_id)
    # for every node in the beam
    for ii in np.arange(mesh_data.num_point):
        lenpoint = 0
        for jj in np.arange(mesh_data.mesh_point_nsec[ii]):
            lenpoint += len(section.points[int(ii)][int(jj)])
        # sec_point_tot : nodes of the complete beam node
        # sec_pt_id     : identifier of the subsection index
        sec_point_tot = np.zeros((lenpoint,2))
        sec_pt_id     = np.zeros((lenpoint,))
        if not case_setup.problem_type == "AERO":
            # calculate the total mass and area
            sol_phys_node.mass_global[ii]      = sum(sol_phys_node.mass[ii])
            sol_phys_node.area_global_node[ii] = sum(sol_phys_node.AA[ii])
            # for every section in the node calculate the contribution to the center of gravity
            for jj in np.arange(mesh_data.mesh_point_nsec[ii]):
                jj = int(jj)
                sol_phys_node.cdg[ii,:] += sol_phys_node.cg[ii][jj]*sol_phys_node.mass[ii][jj]
            # calculate the center of gravity
            sol_phys_node.cdg[ii,:] /= sol_phys_node.mass_global[ii]
            sol_phys.cdg.append(sol_phys_node.cdg[ii,:])
            # obtain the total number of sections nodes in the beam node
            # lenpoint : total number of section nodes
            sol_phys.xsc_nosh[ii]  = -(-sum(sol_phys_node.a23[ii])*sum(sol_phys_node.a26[ii])+sum(sol_phys_node.a22[ii])*sum(sol_phys_node.a36[ii]))/(sum(sol_phys_node.a23[ii])**2-sum(sol_phys_node.a22[ii])*sum(sol_phys_node.a33[ii]))
            sol_phys.ysc_nosh[ii]  = -(-sum(sol_phys_node.a23[ii])*sum(sol_phys_node.a36[ii])+sum(sol_phys_node.a33[ii])*sum(sol_phys_node.a26[ii]))/(-sum(sol_phys_node.a23[ii])**2+sum(sol_phys_node.a22[ii])*sum(sol_phys_node.a33[ii]))
            sol_phys.xsc_sh[ii]  = -(-sum(sol_phys_node.a45[ii])*sum(sol_phys_node.a47[ii])+sum(sol_phys_node.a44[ii])*sum(sol_phys_node.a57[ii]))/(sum(sol_phys_node.a45[ii])**2-sum(sol_phys_node.a44[ii])*sum(sol_phys_node.a55[ii]))
            sol_phys.ysc_sh[ii]  = -(-sum(sol_phys_node.a45[ii])*sum(sol_phys_node.a57[ii])+sum(sol_phys_node.a55[ii])*sum(sol_phys_node.a47[ii]))/(-sum(sol_phys_node.a45[ii])**2+sum(sol_phys_node.a44[ii])*sum(sol_phys_node.a55[ii]))
        # Store the nodes in the same matrix
        # ll_jj  : index for the total node matrix
        # ll_jj2 : index for the total node matrix (previous iteration)
        ll_jj = 0
        for jj in np.arange(mesh_data.mesh_point_nsec[ii]):
            jj     = int(jj)
            ll_jj2 = ll_jj 
            ll_jj += len(section.points[int(ii)][int(jj)])
            if ll_jj2 == 0:
                sec_point_tot[:ll_jj,:] = section.points[ii][int(jj)]
                sec_pt_id[:ll_jj]       = int(jj)
            elif ll_jj == lenpoint:
                sec_point_tot[ll_jj2:,:] = section.points[ii][int(jj)]
                sec_pt_id[ll_jj2:]       = int(jj)
            else:
                sec_point_tot[ll_jj2:ll_jj,:] = section.points[ii][int(jj)]
                sec_pt_id[ll_jj2:ll_jj]       = int(jj)
        section.points_tot.append(sec_point_tot)
        section.pointtot_id.append(sec_pt_id)
    if not case_setup.problem_type == "AERO":
        # for every node calculate the stiffness and mass matrix elements from the nodes values    
        for ii in np.arange(mesh_data.num_elem):
            # ii1 : index of the node 1
            # ii2 : index of the node 2
            ii1 = int(mesh_data.elem[ii,1])
            ii2 = int(mesh_data.elem[ii,2])
#            try:
#                chor1 = np.linalg.norm(section.points[ii1][0][section.te[ii1],:]-section.points[ii1][0][section.le[ii1],:])
#                chor2 = np.linalg.norm(section.points[ii2][0][section.te[ii2],:]-section.points[ii2][0][section.le[ii2],:])
#            except:
#                chor1 = mesh_data.mesh_point_scale[ii1][0]
#                chor2 = mesh_data.mesh_point_scale[ii2][0]
#            chortot = chor1+chor2
#            perpon1 = chor1/chortot
#            perpon2 = chor2/chortot
            # aij_ii1 : elements of the 2D stiffness matrix of the node 1
            a11_ii1 = 0
            a12_ii1 = 0
            a13_ii1 = 0
            a14_ii1 = 0
            a15_ii1 = 0
            a16_ii1 = 0
            a17_ii1 = 0
            a22_ii1 = 0
            a23_ii1 = 0
            a24_ii1 = 0
            a25_ii1 = 0
            a26_ii1 = 0
            a27_ii1 = 0
            a33_ii1 = 0
            a34_ii1 = 0
            a35_ii1 = 0
            a36_ii1 = 0
            a37_ii1 = 0
            a44_ii1 = 0
            a45_ii1 = 0
            a46_ii1 = 0
            a47_ii1 = 0
            a55_ii1 = 0
            a56_ii1 = 0
            a57_ii1 = 0
            a66_ii1 = 0
            a67_ii1 = 0
            a77_ii1 = 0
            # mij_ii1 : element of the mass matrix in node 1
            m11_ii1 = 0
            m16_ii1 = 0
            m22_ii1 = 0
            m26_ii1 = 0
            m33_ii1 = 0
            m34_ii1 = 0
            m35_ii1 = 0
            m37_ii1 = 0
            m44_ii1 = 0
            m45_ii1 = 0
            m47_ii1 = 0
            m55_ii1 = 0
            m57_ii1 = 0
            m66_ii1 = 0
            m77_ii1 = 0
            # aij_ii1 : elements of the 2D stiffness matrix of the node 1
            a11_ii2 = 0
            a12_ii2 = 0
            a13_ii2 = 0
            a14_ii2 = 0
            a15_ii2 = 0
            a16_ii2 = 0
            a17_ii2 = 0
            a22_ii2 = 0
            a23_ii2 = 0
            a24_ii2 = 0
            a25_ii2 = 0
            a26_ii2 = 0
            a27_ii2 = 0
            a33_ii2 = 0
            a34_ii2 = 0
            a35_ii2 = 0
            a36_ii2 = 0
            a37_ii2 = 0
            a44_ii2 = 0
            a45_ii2 = 0
            a46_ii2 = 0
            a47_ii2 = 0
            a55_ii2 = 0
            a56_ii2 = 0
            a57_ii2 = 0
            a66_ii2 = 0
            a67_ii2 = 0
            a77_ii2 = 0
            # mij_ii2 : element of the mass matrix in node 1
            m11_ii2 = 0
            m16_ii2 = 0
            m22_ii2 = 0
            m26_ii2 = 0
            m33_ii2 = 0
            m34_ii2 = 0
            m35_ii2 = 0
            m37_ii2 = 0
            m44_ii2 = 0
            m45_ii2 = 0
            m47_ii2 = 0
            m55_ii2 = 0
            m57_ii2 = 0
            m66_ii2 = 0
            m77_ii2 = 0
            for  jj in np.arange(mesh_data.mesh_point_nsec[ii1]):
                jj = int(jj)
                # aij_ii1 : elements of the 2D stiffness matrix of the node 1
                a11_ii1 += sol_phys_node.a11[ii1][jj]
                a12_ii1 += sol_phys_node.a12[ii1][jj]
                a13_ii1 += sol_phys_node.a13[ii1][jj]
                a14_ii1 += sol_phys_node.a14[ii1][jj]
                a15_ii1 += sol_phys_node.a15[ii1][jj]
                a16_ii1 += sol_phys_node.a16[ii1][jj]
                a17_ii1 += sol_phys_node.a17[ii1][jj]
                a22_ii1 += sol_phys_node.a22[ii1][jj]
                a23_ii1 += sol_phys_node.a23[ii1][jj]
                a24_ii1 += sol_phys_node.a24[ii1][jj]
                a25_ii1 += sol_phys_node.a25[ii1][jj]
                a26_ii1 += sol_phys_node.a26[ii1][jj]
                a27_ii1 += sol_phys_node.a27[ii1][jj]
                a33_ii1 += sol_phys_node.a33[ii1][jj]
                a34_ii1 += sol_phys_node.a34[ii1][jj]
                a35_ii1 += sol_phys_node.a35[ii1][jj]
                a36_ii1 += sol_phys_node.a36[ii1][jj]
                a37_ii1 += sol_phys_node.a37[ii1][jj]
                a44_ii1 += sol_phys_node.a44[ii1][jj]
                a45_ii1 += sol_phys_node.a45[ii1][jj]
                a46_ii1 += sol_phys_node.a46[ii1][jj]
                a47_ii1 += sol_phys_node.a47[ii1][jj]
                a55_ii1 += sol_phys_node.a55[ii1][jj]
                a56_ii1 += sol_phys_node.a56[ii1][jj]
                a57_ii1 += sol_phys_node.a57[ii1][jj]
                a66_ii1 += sol_phys_node.a66[ii1][jj]
                a67_ii1 += sol_phys_node.a67[ii1][jj]
                a77_ii1 += sol_phys_node.a77[ii1][jj]
                # mij_ii1 : element of the mass matrix in node 1
                m11_ii1 += sol_phys_node.m11[ii1][jj] 
                m16_ii1 += sol_phys_node.m16[ii1][jj]
                m22_ii1 += sol_phys_node.m22[ii1][jj] 
                m26_ii1 += sol_phys_node.m26[ii1][jj]
                m33_ii1 += sol_phys_node.m33[ii1][jj]
                m34_ii1 += sol_phys_node.m34[ii1][jj]
                m35_ii1 += sol_phys_node.m35[ii1][jj]
                m37_ii1 += sol_phys_node.m37[ii1][jj]
                m44_ii1 += sol_phys_node.m44[ii1][jj]
                m45_ii1 += sol_phys_node.m45[ii1][jj]
                m47_ii1 += sol_phys_node.m47[ii1][jj]
                m55_ii1 += sol_phys_node.m55[ii1][jj]
                m57_ii1 += sol_phys_node.m57[ii1][jj]
                m66_ii1 += sol_phys_node.m66[ii1][jj]
                m77_ii1 += sol_phys_node.m77[ii1][jj]
            for  jj in np.arange(mesh_data.mesh_point_nsec[ii2]):
                jj = int(jj)
                # aij_ii2 : elements of the 2D stiffness matrix of the node 2 
                a11_ii2 += sol_phys_node.a11[ii2][jj]
                a12_ii2 += sol_phys_node.a12[ii2][jj]
                a13_ii2 += sol_phys_node.a13[ii2][jj]
                a14_ii2 += sol_phys_node.a14[ii2][jj]
                a15_ii2 += sol_phys_node.a15[ii2][jj]
                a16_ii2 += sol_phys_node.a16[ii2][jj]
                a17_ii2 += sol_phys_node.a17[ii2][jj]
                a22_ii2 += sol_phys_node.a22[ii2][jj]
                a23_ii2 += sol_phys_node.a23[ii2][jj]
                a24_ii2 += sol_phys_node.a24[ii2][jj]
                a25_ii2 += sol_phys_node.a25[ii2][jj]
                a26_ii2 += sol_phys_node.a26[ii2][jj]
                a27_ii2 += sol_phys_node.a27[ii2][jj]
                a33_ii2 += sol_phys_node.a33[ii2][jj]
                a34_ii2 += sol_phys_node.a34[ii2][jj]
                a35_ii2 += sol_phys_node.a35[ii2][jj]
                a36_ii2 += sol_phys_node.a36[ii2][jj]
                a37_ii2 += sol_phys_node.a37[ii2][jj]
                a44_ii2 += sol_phys_node.a44[ii2][jj]
                a45_ii2 += sol_phys_node.a45[ii2][jj]
                a46_ii2 += sol_phys_node.a46[ii2][jj]
                a47_ii2 += sol_phys_node.a47[ii2][jj]
                a55_ii2 += sol_phys_node.a55[ii2][jj]
                a56_ii2 += sol_phys_node.a56[ii2][jj]
                a57_ii2 += sol_phys_node.a57[ii2][jj]
                a66_ii2 += sol_phys_node.a66[ii2][jj]
                a67_ii2 += sol_phys_node.a67[ii2][jj]
                a77_ii2 += sol_phys_node.a77[ii2][jj]
                # mij_ii2 : element of the mass matrix in node 2
                m11_ii2 += sol_phys_node.m11[ii2][jj] 
                m16_ii2 += sol_phys_node.m16[ii2][jj]
                m22_ii2 += sol_phys_node.m22[ii2][jj] 
                m26_ii2 += sol_phys_node.m26[ii2][jj]
                m33_ii2 += sol_phys_node.m33[ii2][jj]
                m34_ii2 += sol_phys_node.m34[ii2][jj]
                m35_ii2 += sol_phys_node.m35[ii2][jj]
                m37_ii2 += sol_phys_node.m37[ii2][jj]
                m44_ii2 += sol_phys_node.m44[ii2][jj]
                m45_ii2 += sol_phys_node.m45[ii2][jj]
                m47_ii2 += sol_phys_node.m47[ii2][jj]
                m55_ii2 += sol_phys_node.m55[ii2][jj]
                m57_ii2 += sol_phys_node.m57[ii2][jj]
                m66_ii2 += sol_phys_node.m66[ii2][jj]
                m77_ii2 += sol_phys_node.m77[ii2][jj]
            # calculate the element value as the mean of nodal values
    #        if mesh_data.mesh_point_pri[int(mesh_data.elem[ii,1])]>mesh_data.mesh_point_pri[int(mesh_data.elem[ii,2])]:
    #            sol_phys.a11[ii] += a11_ii1
    #            sol_phys.a12[ii] += a12_ii1
    #            sol_phys.a13[ii] += a13_ii1
    #            sol_phys.a14[ii] += a14_ii1
    #            sol_phys.a15[ii] += a15_ii1
    #            sol_phys.a16[ii] += a16_ii1
    #            sol_phys.a17[ii] += a17_ii1
    #            sol_phys.a22[ii] += a22_ii1
    #            sol_phys.a23[ii] += a23_ii1
    #            sol_phys.a24[ii] += a24_ii1
    #            sol_phys.a25[ii] += a25_ii1
    #            sol_phys.a26[ii] += a26_ii1
    #            sol_phys.a27[ii] += a27_ii1
    #            sol_phys.a33[ii] += a33_ii1
    #            sol_phys.a34[ii] += a34_ii1
    #            sol_phys.a35[ii] += a35_ii1
    #            sol_phys.a36[ii] += a36_ii1
    #            sol_phys.a37[ii] += a37_ii1
    #            sol_phys.a44[ii] += a44_ii1
    #            sol_phys.a45[ii] += a45_ii1
    #            sol_phys.a46[ii] += a46_ii1
    #            sol_phys.a47[ii] += a47_ii1
    #            sol_phys.a55[ii] += a55_ii1
    #            sol_phys.a56[ii] += a56_ii1
    #            sol_phys.a57[ii] += a57_ii1
    #            sol_phys.a66[ii] += a66_ii1
    #            sol_phys.a67[ii] += a67_ii1
    #            sol_phys.a77[ii] += a77_ii1
    #            sol_phys.m11[ii] += m11_ii1
    #            sol_phys.m16[ii] += m16_ii1
    #            sol_phys.m22[ii] += m22_ii1
    #            sol_phys.m26[ii] += m26_ii1
    #            sol_phys.m33[ii] += m33_ii1
    #            sol_phys.m34[ii] += m34_ii1
    #            sol_phys.m35[ii] += m35_ii1
    #            sol_phys.m37[ii] += m37_ii1
    #            sol_phys.m44[ii] += m44_ii1
    #            sol_phys.m45[ii] += m45_ii1
    #            sol_phys.m47[ii] += m47_ii1
    #            sol_phys.m55[ii] += m55_ii1
    #            sol_phys.m57[ii] += m57_ii1
    #            sol_phys.m66[ii] += m66_ii1
    #            sol_phys.m77[ii] += m77_ii1 
    #        elif mesh_data.mesh_point_pri[int(mesh_data.elem[ii,1])]<mesh_data.mesh_point_pri[int(mesh_data.elem[ii,2])]:
    #            sol_phys.a11[ii] += a11_ii2
    #            sol_phys.a12[ii] += a12_ii2
    #            sol_phys.a13[ii] += a13_ii2
    #            sol_phys.a14[ii] += a14_ii2
    #            sol_phys.a15[ii] += a15_ii2
    #            sol_phys.a16[ii] += a16_ii2
    #            sol_phys.a17[ii] += a17_ii2
    #            sol_phys.a22[ii] += a22_ii2
    #            sol_phys.a23[ii] += a23_ii2
    #            sol_phys.a24[ii] += a24_ii2
    #            sol_phys.a25[ii] += a25_ii2
    #            sol_phys.a26[ii] += a26_ii2
    #            sol_phys.a27[ii] += a27_ii2
    #            sol_phys.a33[ii] += a33_ii2
    #            sol_phys.a34[ii] += a34_ii2
    #            sol_phys.a35[ii] += a35_ii2
    #            sol_phys.a36[ii] += a36_ii2
    #            sol_phys.a37[ii] += a37_ii2
    #            sol_phys.a44[ii] += a44_ii2
    #            sol_phys.a45[ii] += a45_ii2
    #            sol_phys.a46[ii] += a46_ii2
    #            sol_phys.a47[ii] += a47_ii2
    #            sol_phys.a55[ii] += a55_ii2
    #            sol_phys.a56[ii] += a56_ii2
    #            sol_phys.a57[ii] += a57_ii2
    #            sol_phys.a66[ii] += a66_ii2
    #            sol_phys.a67[ii] += a67_ii2
    #            sol_phys.a77[ii] += a77_ii2
    #            sol_phys.m11[ii] += m11_ii2
    #            sol_phys.m16[ii] += m16_ii2
    #            sol_phys.m22[ii] += m22_ii2
    #            sol_phys.m26[ii] += m26_ii2
    #            sol_phys.m33[ii] += m33_ii2
    #            sol_phys.m34[ii] += m34_ii2
    #            sol_phys.m35[ii] += m35_ii2
    #            sol_phys.m37[ii] += m37_ii2
    #            sol_phys.m44[ii] += m44_ii2
    #            sol_phys.m45[ii] += m45_ii2
    #            sol_phys.m47[ii] += m47_ii2
    #            sol_phys.m55[ii] += m55_ii2
    #            sol_phys.m57[ii] += m57_ii2
    #            sol_phys.m66[ii] += m66_ii2
    #            sol_phys.m77[ii] += m77_ii2 
    #        else:
            sol_phys.a11[ii] += (a11_ii1+a11_ii2)/2
            sol_phys.a12[ii] += (a12_ii1+a12_ii2)/2
            sol_phys.a13[ii] += (a13_ii1+a13_ii2)/2
            sol_phys.a14[ii] += (a14_ii1+a14_ii2)/2
            sol_phys.a15[ii] += (a15_ii1+a15_ii2)/2
            sol_phys.a16[ii] += (a16_ii1+a16_ii2)/2
            sol_phys.a17[ii] += (a17_ii1+a17_ii2)/2
            sol_phys.a22[ii] += (a22_ii1+a22_ii2)/2
            sol_phys.a23[ii] += (a23_ii1+a23_ii2)/2
            sol_phys.a24[ii] += (a24_ii1+a24_ii2)/2
            sol_phys.a25[ii] += (a25_ii1+a25_ii2)/2
            sol_phys.a26[ii] += (a26_ii1+a26_ii2)/2
            sol_phys.a27[ii] += (a27_ii1+a27_ii2)/2
            sol_phys.a33[ii] += (a33_ii1+a33_ii2)/2
            sol_phys.a34[ii] += (a34_ii1+a34_ii2)/2
            sol_phys.a35[ii] += (a35_ii1+a35_ii2)/2
            sol_phys.a36[ii] += (a36_ii1+a36_ii2)/2
            sol_phys.a37[ii] += (a37_ii1+a37_ii2)/2
            sol_phys.a44[ii] += (a44_ii1+a44_ii2)/2
            sol_phys.a45[ii] += (a45_ii1+a45_ii2)/2
            sol_phys.a46[ii] += (a46_ii1+a46_ii2)/2
            sol_phys.a47[ii] += (a47_ii1+a47_ii2)/2
            sol_phys.a55[ii] += (a55_ii1+a55_ii2)/2
            sol_phys.a56[ii] += (a56_ii1+a56_ii2)/2
            sol_phys.a57[ii] += (a57_ii1+a57_ii2)/2
            sol_phys.a66[ii] += (a66_ii1+a66_ii2)/2
            sol_phys.a67[ii] += (a67_ii1+a67_ii2)/2
            sol_phys.a77[ii] += (a77_ii1+a77_ii2)/2
            sol_phys.m11[ii] += (m11_ii1+m11_ii2)/2
            sol_phys.m16[ii] += (m16_ii1+m16_ii2)/2
            sol_phys.m22[ii] += (m22_ii1+m22_ii2)/2
            sol_phys.m26[ii] += (m26_ii1+m26_ii2)/2
            sol_phys.m33[ii] += (m33_ii1+m33_ii2)/2
            sol_phys.m34[ii] += (m34_ii1+m34_ii2)/2
            sol_phys.m35[ii] += (m35_ii1+m35_ii2)/2
            sol_phys.m37[ii] += (m37_ii1+m37_ii2)/2
            sol_phys.m44[ii] += (m44_ii1+m44_ii2)/2
            sol_phys.m45[ii] += (m45_ii1+m45_ii2)/2
            sol_phys.m47[ii] += (m47_ii1+m47_ii2)/2
            sol_phys.m55[ii] += (m55_ii1+m55_ii2)/2
            sol_phys.m57[ii] += (m57_ii1+m57_ii2)/2
            sol_phys.m66[ii] += (m66_ii1+m66_ii2)/2
            sol_phys.m77[ii] += (m77_ii1+m77_ii2)/2 

    print('Meshing Finished')
    return section, sol_phys, mesh_data
        