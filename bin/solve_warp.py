# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 13:42:31 2021

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
solve_warp     : file containing functions for calculating the warping function
last_version   : 19-02-2021
modified_by    : Andres Cremades Botella
"""

import numpy as np
from bin.aux_functions import def_vec_param 

#%% Functions
def function_warp_TWO(section_elem,section_points,section_cell,sol_phys,section):
    # Function to obtain the warping function in open thin wall sections
    # section_elem   : elements of the section
    # section_points : nodes of the section
    # section_cell   : cells of the section
    # sol_phys       : solid physics
    # section        : information of the section 
    # --------------------------------------------------------------------------
    # mat_conexion : matrix to define the conexion between points and elements
    #                the rows of the matrix represent the elements and the columns the nodes
    # Add a 1 in the elements/nodes that are in the mesh
    mat_conexion = np.zeros((len(section_elem),len(section_points)))
    for elem in section_elem:
        mat_conexion[int(elem[0]),int(elem[1])] += 1
        mat_conexion[int(elem[0]),int(elem[2])] += 1
    # For each point in the mesh change the 1 in the elements by  the number of elements that contains the node    
    for ii_point in np.arange(len(section_points)):
        mat_conexion[:,ii_point] = sum( mat_conexion[:,ii_point])* mat_conexion[:,ii_point] 
    # warp            : warping function
    # c_warp          : cumulative value of the warp 
    #                   the value depends on the reference points
    # ind_next        : next index to calculate
    # num_nodes       : number of elements in contact with a node
    # warp_next       : warping initialization in the next step
    # r_t             : tangential distance from the center of gravity to the section walls
    # r_n             : normal distance from the center of gravity to the section walls
    # calculated_elem : elements that have been calculated
    # ord_next        : following elements to calculate sorted by priority
    warp = np.zeros((len(section_elem),))
    warpds = np.zeros((len(section_elem),))
    warp_node = np.zeros((len(section_points),))
    c_warp = 0
    c_sum = 0
    ds_wall = np.zeros((len(section_elem),))
    ind_next=np.arange(len(section_elem))
    nodfin_next = np.zeros((len(section_elem),),dtype='int')
    ordind_line_next = np.zeros((len(section_elem),),dtype='int')
    num_nodes = 1
    warp_next = np.zeros((len(section_elem),))
    r_t =np.zeros((len(section_elem),))
    r_n =np.zeros((len(section_elem),))
    calculated_elem = []
    ord_next = []
    n_vec           = np.zeros((len(section_elem),3))
    delta_s_vec     = np.zeros((len(section_elem),3))
    lines_wall = []
    lines_wall_node = []
    ind_line = 0
    ind_maxline = 0
    flag_inielem = True
    # the loop starts with the nodes with less elements in contact and calculates increasing
    # the number of contact elements until is higher than the maximum value 
    while num_nodes <= np.max(mat_conexion):
        # ind_now     : index of the current node
        # warp_now    : initial value of the warping function in the elements
        # ord_now     : list of following elements in every section calculation
        # ii_elem_num : index
        # ii_elem_ind : index number of elements that have been calculated
        ind_now = ind_next
        ind_next=[]
        warp_now = warp_next
        warp_next = []
        ord_now = ord_next
        ord_next = []
        nodfin_now = nodfin_next
        nodfin_next = []
        ordind_line_now = ordind_line_next
        ordind_line_next = []
        ii_elem_num = 0
        ii_elem_ind = 0
        while ii_elem_ind < len(ind_now):
            # ii_elem  : index of the current element of the calculation
            # unique   : vector containing the values apearing in the conexion matrix
            # counts   : number of times that each value appears in the matrix
            ii_elem        = ind_now[ii_elem_ind]
            unique, counts = np.unique(mat_conexion[ii_elem,:],return_counts=True)
            # if the unique value coincides with the current number of contact nodes
            if any(unique == num_nodes):
                ii_node_element = np.zeros((2,2))
                if num_nodes == 2:
                    ind_line = ordind_line_now[ii_elem_ind]
                    lines_wall[ind_line].append(ii_elem)
                else:
                    lines_wall.append([ii_elem])
                    lines_wall_node.append([])
                    ind_maxline += 1
                    ind_line = len(lines_wall)-1
                if num_nodes ==1:
                    f1 = 0
                    f2 = 0
                    for ii_point_ind in np.arange(len(mat_conexion[ii_elem,:])):
                        ii_point = mat_conexion[ii_elem,ii_point_ind]
                        if ii_point == 1:
                            ii_node_element[0,:] = section_points[int(ii_point_ind),:]
                            ii_1 = int(ii_point_ind)
                            f1 = 1
                        elif ii_point > 0:
                            ii_node_element[1,:] = section_points[int(ii_point_ind),:]
                            nodfin_now[ii_elem] = int(ii_point_ind)
                            ii_2 = int(ii_point_ind)
                            f2 = 1
                        if f1 == 1 and f2 == 1:
                            lines_wall_node[ind_line].append(ii_1)
                            lines_wall_node[ind_line].append(ii_2)
                            f1 = 0
                            f2 = 0
                else:
                    ii_node_element[0,:] = section_points[ord_now[ii_elem_num][0],:]
                    ii_node_element[1,:] = section_points[ord_now[ii_elem_num][1],:]
                    lines_wall_node[ind_line].append(ord_now[ii_elem_num][1])
                rx = section.cg_local[ii_elem,0]#-section.cg[0]
                ry = section.cg_local[ii_elem,1]#-section.cg[1]
                ds_vec2d = ii_node_element[1,:]-ii_node_element[0,:]
                ds_vec = [ds_vec2d[0],ds_vec2d[1],0]
                ds = np.linalg.norm(ds_vec)
                ds_def_adim = (section_points[int(section_elem[int(ii_elem),2]),:]-section_points[int(section_elem[int(ii_elem),1]),:])/ds
                r_t[ii_elem] = np.dot(ds_def_adim,[rx,ry])
                dn_vec = np.cross([0,0,1],ds_vec)
                n_vec[ii_elem,:] = dn_vec
                dn_vec2d = dn_vec[:2]
                dn_def_adim = dn_vec2d/np.linalg.norm(dn_vec2d)
                r_n[ii_elem] = np.dot(dn_def_adim,[rx,ry])
                A_cec = np.cross([rx,ry,0],ds_vec)
                c_sum += A_cec
                ds_wall[ii_elem] = ds
#                warp[ii_elem] += warp_now[ii_elem_ind]
                if all(calculated_elem != ii_elem):
                    warp[ii_elem] += A_cec[2]
                if warp_node[nodfin_now[ii_elem_ind]] != 0 and warp_node[nodfin_now[ii_elem_ind]] != warp[ii_elem]:
#                    flag_lines = 0
                    for auxline in np.arange(len(lines_wall)):
                        for auxline2 in np.arange(len(lines_wall[auxline])-1):
                            delta_line_warp = 0
                            if lines_wall_node[auxline][0] == nodfin_now[ii_elem_ind] or lines_wall_node[auxline][-1] == nodfin_now[ii_elem_ind]:
                                if lines_wall_node[auxline][0] == nodfin_now[ii_elem_ind]:
                                    ind_warp = lines_wall[auxline][0]
                                else:
                                    ind_warp = lines_wall[auxline][-1]
    #                            if flag_lines == 0:
    #                                delta_line_warp += warp[ind_warp] 
    #                            else:
                                delta_line_warp += warp[ind_warp] 
                                for elemline in lines_wall[auxline][:-1-auxline2]:
                                    warp[elemline] += delta_line_warp
    #                            flag_lines+=1
                else:
                    warp_node[nodfin_now[ii_elem_ind]] = warp[ii_elem]
                c_warp += A_cec[2]*ds
                delta_s_vec[ii_elem] = ds_vec
                calculated_elem.append(ii_elem)
                ind_point_equal=np.where(mat_conexion[ii_elem,:]==num_nodes)[0]
                ind_point_high=np.where(mat_conexion[ii_elem,:]==num_nodes+1)[0]
                ind_befo = ind_now.copy()
                if len(ind_point_equal)>1:
                    for jj_point in ind_point_equal:
                        for jj_elem in np.arange(len(section_elem)):
                            if mat_conexion[jj_elem,jj_point] == num_nodes and ii_elem != jj_elem and not any(ind_now==jj_elem) and not any(calculated_elem==jj_elem)  and (flag_inielem or all(ind_befo != jj_elem)):
                                ind_now.append(jj_elem)
                                if jj_point == section_elem[jj_elem,1]:
                                    jj_point2 =  int(section_elem[jj_elem,2])
                                else:
                                    jj_point2 = int(section_elem[jj_elem,1])
                                ord_now.append([int(jj_point),int(jj_point2)])
                                warp_now.append(warp[ii_elem])
                                nodfin_now.append(int(jj_point2))
                                ordind_line_now.append(int(ind_line))
                else:             
                    for jj_point in ind_point_high:
                        for jj_elem in np.arange(len(section_elem)):
                            if mat_conexion[jj_elem,jj_point] == num_nodes+1 and ii_elem != jj_elem and not any(calculated_elem==jj_elem) and (flag_inielem or all(ind_befo != jj_elem)):
                                ind_next.append(jj_elem) 
                                if jj_point == section_elem[jj_elem,1]:
                                    jj_point2 =  int(section_elem[jj_elem,2])
                                else:
                                    jj_point2 = int(section_elem[jj_elem,1])
                                ord_next.append([int(jj_point),int(jj_point2)])
                                warp_next.append(warp[ii_elem])
                                nodfin_next.append(int(jj_point2))
                                ordind_line_next.append(int(ind_line))
            ii_elem_num += 1
            ii_elem_ind += 1
        num_nodes += 1
        flag_inielem = False
    c_warp = sum(np.multiply(warp,ds_wall))/sum(ds_wall)
    # Substract the cummulative value to correct the position of the reference point
    warp = warp - c_warp*np.ones((len(warp),))
    return warp, r_t, r_n, n_vec, delta_s_vec




#%%
def function_warp_TWC(section_elem,section_points,section_cell,section_branch,sol_physuni,section,prev_elements,keepwarp):
    # Function to calculate warping in closed thin walled section beams
    # section_elem   : elements of the section
    # section_points : nodes of the section
    # section_cell   : cells of the section
    # sol_physuni    : solid physics
    # section        : information of the section 
    # ------------------------------------------------------------------------------
    # dwarp               : warping function derivative evaluated in the center of the element 
    # warp                : warping function
    # dwarpend            : warping function derivative evaluated in the end of the element
    # warpend             : warping function evaluated in the end fo the element
    # warp_cum            : cumulative value of the warping function
    # ds_modvec           : norm of the length of the element
    # ds_modvec_mean      : half of the distance of ds_modvec
    # dwarp_lambda        : contribution of lambda to the warping function
    # dwarp_lambda_vec    : contribution of lambda to the warping function in vector form
    # dwarp_r             : contribution of the distance to the warping function
    # dwarp_lambdaend     : contribution of lambda to the warping function in the end of the element
    # dwarp_lambda_vecend : contribution of lambda to the warping function in vector form in the end of the element
    # dwarp_rend          : contribution of the distance to the warping function in the end of the element
    dwarp               = np.zeros((len(section_elem),))
    warp                = np.zeros((len(section_elem),))
    dwarpend            = np.zeros((len(section_elem),))
    warpendnode         = np.zeros((len(section_points),))
    warpend             = np.zeros((len(section_elem),))
    warp_ant            = np.zeros((len(section_elem),))
    warp_cum            = 0
    ds_modvec           = np.zeros((len(section_elem),))
    ds_modvec_mean      = np.zeros((len(section_elem),))
    dwarp_lambda        = np.zeros((len(section_elem),))
    dwarp_lambda_vec    = np.zeros((len(section_elem),2))
    dwarp_r             = np.zeros((len(section_elem),))
    dwarp_lambdaend     = np.zeros((len(section_elem),))
    dwarp_lambda_vecend = np.zeros((len(section_elem),2))
    dwarp_rend          = np.zeros((len(section_elem),))
    # node1         : first node of the element
    # node2         : second node of the element
    # mat_cell_elem : matrix to relate the elements with the cells
    node1         = []
    node2         = []
    mat_cell_elem = np.zeros((len(section_cell),len(section_elem)))
    # for all the elements in every cell
    for ii_cell in np.arange(len(section_cell)):
        # cell_ii  : current cell
        cell_ii = section_cell[ii_cell]
        mat_cell_elem[ii_cell,cell_ii[0]] = 1
        for jj_elem in cell_ii:
            # Duplicate previous nodes
            # node1_b : copy of node 1
            # node2_b : copy of node 2
            node1_b = node1
            node2_b = node2
            node1   = section_elem[jj_elem,1]
            node2   = section_elem[jj_elem,2]
            # if the element follows the rotation direction set the matrix in 1 else in -1
            if node1 == node2_b or node1 == node1_b:
                mat_cell_elem[ii_cell,jj_elem] = 1
            elif node2==node1_b or node2 == node2_b:
                mat_cell_elem[ii_cell,jj_elem] = -1
        # For the first point
        if section_elem[cell_ii[0],1] == node2 or section_elem[cell_ii[0],1] == node1:
            mat_cell_elem[ii_cell,cell_ii[0]] = 1
        elif section_elem[cell_ii[0],2] == node1 or section_elem[cell_ii[0],2] == node2:
            mat_cell_elem[ii_cell,cell_ii[0]] = -1
    # cell_connet             : connection between cells if cells are in contact value is 1 if the
    #                           rotation is the same and -1 if opposite
    # cell_connect_calculated : indicates that the cell connection is done with a 1 to avoid repetitions
    cell_connect            = np.zeros((len(section_cell),len(section_cell)))
    cell_connect_calculated = np.zeros((len(section_cell),len(section_cell)))
    # For all the cells in the section get all the elements
    for jj_elem in np.arange(len(section_elem)):
        # column : column of the cell-element matrix - cells of one element
        column = mat_cell_elem[:,jj_elem]
        # Determine if the element belongs to a cell and get the cell index
        if sum(abs(column))>1:
            index_column = np.where(abs(column) == 1)[0]
            # For all the cells containing the element check if added before
            for kk in index_column:
                for kk2 in index_column:
                    if cell_connect_calculated[kk,kk2] == 0:
                        # If it is not calculated add a 1 if the cell is in the same direction and a -1 if not
                        cell_connect[kk,kk2]            = column[kk2]/column[kk]
                        cell_connect_calculated[kk,kk2] = 1
    if len(cell_connect) == 1:
        cell_connect[0,0] = 1
    # elements_used  : list containing the elements calculated
    # point1         : first point of the element
    # point2         : second point of the element
    # sgnds          : gets the value -1 if the rotation of the cell is oposite to the
    #                  rotation of the element
    # ds_vec2d       : vector of the element length in 2D
    # ds_uni         : unitary vector of the lement length in 2D
    # ds_vec         : vector of the element length in 3D
    # r_vec          : distance from the origin to the element in 2D
    # r_t            : tangential distance from the origin to the element in 2D
    # r_n            : normal distance from the origin to the element in 2D
    # A_cec          : double area contained in the section elements
    # A_cec_mid      : area contained in the section elements
    # area_double    : double area contained in the setion cells
    elements_used = []
    point1        = np.zeros((len(section_elem),))
    point2        = np.zeros((len(section_elem),))
    sgnds         = np.zeros((len(section_elem),))
    ds_vec2d      = np.zeros((len(section_elem),2))
    ds_uni        = np.zeros((len(section_elem),2))
    ds_vec        = np.zeros((len(section_elem),3))
    r_vec         = np.zeros((len(section_elem),3))
    r_t           = np.zeros((len(section_elem),))
    r_n           = np.zeros((len(section_elem),))
    n_vec         = np.zeros((len(section_elem),3))
    A_cec         = np.zeros((len(section_elem),))
    A_cec_mid     = np.zeros((len(section_elem),))
    area_double   = np.zeros((len(section_cell),))
    # ds_sum : sum of the element distance
    ds_sum = 0
    # For all the elements in every cell
    for ii_cell in np.arange(len(section_cell)):
        for jj_elem in section_cell[ii_cell]:
            # Determine if the element follows the rotation direction or not
            if mat_cell_elem[ii_cell,jj_elem] == 1:
                # if it follows set the normal order of nodes
                pp1             = section_elem[jj_elem,1]
                pp2             = section_elem[jj_elem,2]
                node1           = section_points[int(pp1)]
                node2           = section_points[int(pp2)]
                if len(np.where(np.array(elements_used) == jj_elem)[0])==0:
                    sgnds[jj_elem]  = 1
                    point1[jj_elem] = pp1
                    point2[jj_elem] = pp2
            elif mat_cell_elem[ii_cell,jj_elem] == -1:
                # if not inverse the order of nodes
                pp1             = section_elem[jj_elem,2]
                pp2             = section_elem[jj_elem,1]
                node1           = section_points[int(pp1)]
                node2           = section_points[int(pp2)]
                if len(np.where(np.array(elements_used) == jj_elem)[0])==0:
                    point1[jj_elem] = pp1
                    point2[jj_elem] = pp2
                    sgnds[jj_elem]  = -1
            # rx             : distance from the reference to the element in x axis
            # ry             : distance from the reference to the element in y axis
            # ds_modvec      : length of the element
            # ds_modvec_mean : length to to midpoint of the element
            # ds_def_adim    : normalized length element vector in mesh definition order
            # dn_vec         : normal vector of the element
            # dn_def_adim    : normalized normal vector of the element
            # if the element have not been used before calculate the generated areas
            rx                      = section.cg_local[jj_elem,0]#-section.cg[0]
            ry                      = section.cg_local[jj_elem,1]#-section.cg[1]
            ss                      = node2-node1
            rr                      = [rx,ry,0]
            if len(np.where(np.array(elements_used) == jj_elem)[0])==0:
                ds_vec2d[jj_elem,:]     = ss
                ds_uni[jj_elem,:]       = ds_vec2d[jj_elem]/np.linalg.norm(ds_vec2d[jj_elem])
                dselem                  = section_points[int(section_elem[jj_elem,2])]-section_points[int(section_elem[jj_elem,1])]
                ds_vec[jj_elem,:]       = [dselem[0],dselem[1],0]#[ds_vec2d[jj_elem,0],ds_vec2d[jj_elem,1],0]
                ds_modvec[jj_elem]      = np.linalg.norm(ds_vec[jj_elem,:])
                ds_modvec_mean[jj_elem] = np.linalg.norm(ds_vec[jj_elem,:])/2
                r_vec[jj_elem,:]        = rr
                ds_def_adim             = (section_points[int(section_elem[int(jj_elem),2]),:]-section_points[int(section_elem[int(jj_elem),1]),:])/np.linalg.norm((section_points[int(section_elem[int(jj_elem),2]),:]-section_points[int(section_elem[int(jj_elem),1]),:]))
                r_t[jj_elem]            = np.dot(ds_def_adim,[rx,ry])
                dn_vec                  = np.cross([0,0,1],ds_vec[jj_elem,:])
                dn_vec2d                = dn_vec[:2]
                n_vec[jj_elem,:]        = dn_vec          
                dn_def_adim             = dn_vec2d/np.linalg.norm(dn_vec2d)
                r_n[jj_elem]            = np.dot(dn_def_adim,[rx,ry])
                # A_cec_vec    : double area in vectorial form
                # A_cec_vecmid : area in vectorial form
                A_cec_vec          = np.cross(r_vec[jj_elem,:],ds_vec[jj_elem,:])
                A_cec_vecmid       = np.cross(r_vec[jj_elem,:],ds_vec[jj_elem,:]/2)
                A_cec[jj_elem]     = A_cec_vec[2]
                A_cec_mid[jj_elem] = A_cec_vecmid[2]
                ds_sum += ds_modvec[jj_elem]
            # Add the element to the used element list
            elements_used.append(jj_elem)
            # Add the value of the double area
            area_double[ii_cell] += np.cross(rr,[ss[0],ss[1],0])[2]#A_cec[jj_elem]
    # delta  : length of the element divided by its thickness
    # S_mat  : matrix S - division between the ratio of twist and the shear flow
    delta  = np.zeros((len(section_cell),len(section_cell)))
    delta13  = np.zeros((len(section_cell),len(section_cell)))
    delta43  = np.zeros((len(section_cell),len(section_cell)))
    delta53  = np.zeros((len(section_cell),len(section_cell)))
    S_mat  = np.zeros((len(section_cell),len(section_cell)))
    S_mat13  = np.zeros((len(section_cell),len(section_cell)))
    S_mat43  = np.zeros((len(section_cell),len(section_cell)))
    S_mat53  = np.zeros((len(section_cell),len(section_cell)))
    # for all the elements in every cell
    for ii_cell in np.arange(len(section_cell)):
        for jj_elem in section_cell[ii_cell]:
            # for all the cells calculate the value of delta
            Gxz = 0
            Gxz13 = 0
            Gxz43 = 0
            Gxz53 = 0
            for jj_ply in np.arange(len(sol_physuni.A66[jj_elem])):
                Gxz += sol_physuni.A66[jj_elem][jj_ply]-sol_physuni.A16[jj_elem][jj_ply]**2/sol_physuni.A11[jj_elem][jj_ply]
                Gxz13 += (sol_physuni.A26[jj_elem][jj_ply]-sol_physuni.A12[jj_elem][jj_ply]*sol_physuni.A16[jj_elem][jj_ply]/sol_physuni.A11[jj_elem][jj_ply])
                Gxz43 += (sol_physuni.B26[jj_elem][jj_ply]-sol_physuni.B12[jj_elem][jj_ply]*sol_physuni.A16[jj_elem][jj_ply]/sol_physuni.A11[jj_elem][jj_ply])
                Gxz53 += (sol_physuni.B66[jj_elem][jj_ply]-sol_physuni.B16[jj_elem][jj_ply]*sol_physuni.A16[jj_elem][jj_ply]/sol_physuni.A11[jj_elem][jj_ply])
            delta[ii_cell,ii_cell] += ds_modvec[jj_elem]/Gxz #(sum(sol_physuni.A66[jj_elem]))#section_elem[jj_elem,3]#
            if Gxz13 != 0:
                delta13[ii_cell,ii_cell] += ds_modvec[jj_elem]/Gxz13
            if Gxz43 != 0:
                delta43[ii_cell,ii_cell] += ds_modvec[jj_elem]/Gxz43
            if Gxz53 != 0:
                delta53[ii_cell,ii_cell] += ds_modvec[jj_elem]/Gxz53
            for jj_cell in np.arange(len(section_cell)):
                # If the element is contained in other cell 
                if abs(mat_cell_elem[jj_cell,jj_elem])>0 and jj_cell != ii_cell and len(cell_connect)>1:
                    delta[ii_cell,jj_cell] += ds_modvec[jj_elem]/Gxz # (sum(sol_physuni.A66[jj_elem]))#(section_elem[jj_elem,3])#
                    delta13[ii_cell,jj_cell] += ds_modvec[jj_elem]/Gxz13
                    delta43[ii_cell,jj_cell] += ds_modvec[jj_elem]/Gxz43
                    delta53[ii_cell,jj_cell] += ds_modvec[jj_elem]/Gxz53
        S_mat[ii_cell,:] = np.multiply(delta[ii_cell,:],cell_connect[ii_cell])/area_double[ii_cell] 
        S_mat13[ii_cell,:] = np.multiply(delta13[ii_cell,:],cell_connect[ii_cell])/area_double[ii_cell] 
        S_mat43[ii_cell,:] = np.multiply(delta43[ii_cell,:],cell_connect[ii_cell])/area_double[ii_cell] 
        S_mat53[ii_cell,:] = np.multiply(delta53[ii_cell,:],cell_connect[ii_cell])/area_double[ii_cell] 
    # H_mat          : inverse matrix of s in vectorial form
    # lambda_sec     : lambda matrix for the sections 
    # mat_elem_point : matrix to connect elements and points
    H_mat   = np.matmul(np.linalg.inv(S_mat),np.ones((len(section_cell),1)))
    if np.isnan(np.linalg.det(S_mat13)) or np.linalg.det(S_mat13)==0:
        H_mat13 = np.zeros((len(section_cell),1))
    else:
        H_mat13 = np.matmul(np.linalg.inv(S_mat13),np.ones((len(section_cell),1)))
    if np.isnan(np.linalg.det(S_mat43)) or np.linalg.det(S_mat43)==0:
        H_mat43 = np.zeros((len(section_cell),1))
    else:
        H_mat43 = np.matmul(np.linalg.inv(S_mat43),np.ones((len(section_cell),1)))
    if np.isnan(np.linalg.det(S_mat53)) or np.linalg.det(S_mat53)==0:
        H_mat53 = np.zeros((len(section_cell),1))
    else:
        H_mat53 = np.matmul(np.linalg.inv(S_mat53),np.ones((len(section_cell),1)))
    lambda_sec     = np.zeros((len(section_cell),len(section_elem)))
    mat_elem_point = np.zeros((len(section_points),len(section_elem)))
    # For all the points in the section find their elements
    for ii_point in np.arange(len(section_points)):
        # elem_point : elements containing the node
        elem_point = np.where(section_elem[:,1:3] == ii_point)
        #  for each element containing the node set the connection matrix element to 1 
        for jj_elem in elem_point[0]:
            mat_elem_point[ii_point,jj_elem] = 1
    # for each cell calculate the lambda matrix for all the elements
    for ii_cell in np.arange(len(section_cell)):
        for jj_elem in section_cell[ii_cell]:
            Gxz = 0
            for jj_ply in np.arange(len(sol_physuni.A66[jj_elem])):
                Gxz += sol_physuni.A66[jj_elem][jj_ply]-sol_physuni.A16[jj_elem][jj_ply]**2/sol_physuni.A11[jj_elem][jj_ply]
            lambda_sec[ii_cell,jj_elem] += H_mat[ii_cell]/Gxz # (sum(sol_physuni.A66[jj_elem]))#(section_elem[jj_elem,3])# 
    # reset the used elements     
    elements_used = []       
    # for all the elements in every cell
    for ii_cell in np.arange(len(section_cell)):
        for jj_elem in section_cell[ii_cell]:
            # if any element remains unused calculate its warping derivatives contributions
            if len(np.where(np.array(elements_used) == jj_elem)[0])==0:
                dwarp_r[jj_elem]               = A_cec_mid[jj_elem]
                dwarp_lambda_vec[jj_elem,:]    = sum(np.multiply(lambda_sec[:,jj_elem],mat_cell_elem[:,jj_elem]))*ds_vec2d[jj_elem,:]/2
                dwarp_lambda[jj_elem]          = np.dot(dwarp_lambda_vec[jj_elem,:],ds_uni[jj_elem])
                dwarp_rend[jj_elem]            = A_cec[jj_elem]
                dwarp_lambda_vecend[jj_elem,:] = sum(np.multiply(lambda_sec[:,jj_elem],mat_cell_elem[:,jj_elem]))*ds_vec2d[jj_elem,:]
                dwarp_lambdaend[jj_elem]       = np.dot(dwarp_lambda_vecend[jj_elem,:],ds_uni[jj_elem])
                elements_used.append(jj_elem)
    # for every element in the section calculate the total warping derivatives       
    for ii_elem in np.arange(len(section_elem)):
        dwarp[ii_elem] = dwarp_lambda[ii_elem]-dwarp_r[ii_elem]
        dwarpend[ii_elem] = dwarp_lambdaend[ii_elem]-dwarp_rend[ii_elem]
    # Reset the elements used
    # elements_used_ant  : elements used in the previous steps
    elements_used     = []
    elements_used_ant = []
    # jj         : index of the following element
    # rem_elem   : remaining elements to calculate
    # point_ini  : initial point of the calculation
    # point_cum1 : point 1 of the element in the cumulative warping
    # point_cum2 : point 2 of the element in the cumulative warping
    # iteration  : number of iterations
    # ini_point  : initial point of the following element
    # sign       : 1 if the element follows the rotation direction -1 if not
    jj         = section_cell[0][0]
    rem_elem   = []
    point_ini  = point1[jj]
    point_cum1 = np.zeros((len(section_elem),))
    point_cum2 = np.zeros((len(section_elem),))
    iteration = 1
    ini_point = []
    sign      = np.zeros((len(section_elem),))
    # Do until breaking conditions
    if len(prev_elements)==0: 
        while True: 
            # jj_elem    : element index
            # point_next : following point (final of current element, initial of next)
            jj_elem = int(section_elem[jj,0])
            # if iteration is 1 start with point_ini if not with the next element
            if iteration == 1:
                point_cum1[jj_elem] = point_ini
                point_cum2[jj_elem] = point2[jj_elem]
                point_next          = point_cum2[jj_elem]
                sign[jj_elem]       = 1
            else: 
                point_cum1[jj_elem] = ini_point_elem
                # Determine the orientation of the element and define the value of the sign
                if point1[jj_elem] == ini_point_elem:
                    point_cum2[jj_elem] = point2[jj_elem]
                    sign[jj_elem]       = 1
                else:
                    point_cum2[jj_elem] = point1[jj_elem]
                    sign[jj_elem]       = -1
                point_next = point_cum2[jj_elem]
            # If the element has not been used
            if len(np.where(np.array(elements_used)==jj_elem)[0])==0:
                # calculate warping in the middle and the end of the element
                # n_contact : elements containing the last node 
                warp[jj_elem]    = warp_ant[jj_elem] + sgnds[jj_elem]*dwarp[jj_elem]
                warpend[jj_elem] = warp_ant[jj_elem] + sgnds[jj_elem]*dwarpend[jj_elem]
                warpendnode[int(point2[jj_elem])] = warpend[jj_elem]
                warp_cum        += warp[jj_elem]*ds_modvec[jj_elem]
                n_contact        = np.where(mat_elem_point[int(point_cum2[jj_elem]),:]==1)[0]
                # For all the elements containing the node 
                for kk in n_contact:
                    # If they are different than current and has not been used
                    if kk != jj_elem and len(np.where(np.array(elements_used)==kk)[0])==0:
                        # if the element has not been added to the remaining element list add it
                        # warp_ant : warping of the remaining node as the calculated in the previous elements
                        if len(np.where(np.array(rem_elem)==kk)[0])==0:
                            for aux_sec in np.arange(len(section_cell)):
                                cellelem = section_cell[aux_sec]
                                if len(np.where(np.array(cellelem)==kk)[0])>0: 
                                    rem_elem.append(kk)
                                    ini_point.append(point_next)
                                    warp_ant[kk] = warpend[jj_elem]
                        # If the element has been added to the remaining element list
                        else:
                            # find the index of the element in the list
                            position_rem = np.where(np.array(rem_elem)==kk)[0] 
                            # if its initial point is the current final point and has not been used before add the initial value of warping
                            if ini_point[int(position_rem[0])] == point_cum2[jj_elem]:
                                if len(np.where(np.array(elements_used_ant)==kk)[0])==0:
                                    warp_ant[kk] = warpend[jj_elem]
                                    elements_used_ant.append(jj_elem)
                # add current element to the used list
                elements_used.append(jj_elem)
            # jj_ind      : index of the remaining elemnt
            # finish_flag : flag indicating the end of the loop
            # node_conlen : vector with the number of remaining contacts of the node
            jj_ind      = 0
            finish_flag = 0
            node_conlen = np.zeros((len(rem_elem),))
            # for all the remaining elements set the initial and final nodes
            for ii_remain in np.arange(len(rem_elem)):
                p1 = ini_point[ii_remain]
                if point1[rem_elem[ii_remain]] == p1:
                    p2 = point2[rem_elem[ii_remain]]
                else:
                    p2 = point1[rem_elem[ii_remain]]
                # for all the elements in contact with one point if the elements are not used
                for jj_lenrem in np.arange(len(mat_elem_point[int(p2),:])):
                    if len(np.where(elements_used == jj_lenrem)[0]) == 0:
                        # add 1 each contact without calculate
                        node_conlen[ii_remain] += mat_elem_point[int(p2),jj_lenrem]
            # node_conlennp : node_conlen as vector array
            # rem_elemnp    : remaining elements as vector array
            # indsort       : index to sort the remaining elements array
            # ini_pointnp   : initial point to array
            node_conlennp = np.array(node_conlen)
            rem_elemnp    = np.array(rem_elem)
            indsort       = node_conlennp.argsort()
            rem_elemnp    = rem_elemnp[indsort]
            node_conlennp = node_conlennp[indsort]
            rem_elem      = rem_elemnp.tolist()
            node_conlen   = node_conlennp.tolist()
            ini_pointnp   = np.array(ini_point)
            ini_pointnp   = ini_pointnp[indsort]
            ini_point     = ini_pointnp.tolist()
            # take the value of the following element and initial point if there is not a remaining point finish the loop
            try:
                jj = rem_elem[0]
            except:
                break
            ini_point_elem = ini_point[0]
            # Do until break options
            while True:
                # if the calculation is not finish and the point 1 or 2 of the remaining element is the initial point of the following element
                if (point1[int(rem_elem[jj_ind])] == point_ini or point2[int(rem_elem[jj_ind])] == point_ini) and finish_flag == 0:
                    try:
                        jj             = rem_elem[jj_ind+1]
                        ini_point_elem = ini_point[jj_ind+1]
                    except:
                        pass
                # if not, reset the remaining element vector and the initial point if only one element is inside and remove the element if not
                else:
                    if len(rem_elem) == 1:
                        rem_elem  = []
                        ini_point = []
                    else: 
                        rem_elem.remove(rem_elem[jj_ind])
                        ini_point.remove(ini_point[jj_ind])
                    break
                # advance the item
                jj_ind +=1
                # if the following element is the last one, activate the finish flag
                # take the first remaining element as the following
                if jj_ind+1 > len(rem_elem):
                    finish_flag    = 1
                    jj_ind         = 0
                    jj             = rem_elem[jj_ind]
                    ini_point_elem = ini_point[jj_ind]
            # advance the iteration
            iteration += 1
            if keepwarp == 'YES':
                prev_elements = elements_used
        else:
            for jj_elem in prev_elements:
                warp[jj_elem]    = warp_ant[jj_elem] + sgnds[jj_elem]*dwarp[jj_elem]
                warpend[jj_elem] = warp_ant[jj_elem] + sgnds[jj_elem]*dwarpend[jj_elem]
                warpendnode[int(point2[jj_elem])] = warpend[jj_elem]
                warp_cum        += warp[jj_elem]*ds_modvec[jj_elem]
                n_contact        = np.where(mat_elem_point[int(point_cum2[jj_elem]),:]==1)[0]
    # For the branch
    if len(section_branch) > 0:
        used_element = []
        aa = []
        for ii_branch in np.arange(len(section_branch)):
            aux_it = 0
            elem_branch = section_branch[ii_branch] 
            for ii_elembranch in np.arange(len(elem_branch)):
                if aux_it == 0:
                    for ii_cell in np.arange(len(section_cell)):
                        elem_i1 = elem_branch[0]
                        elem_i2 = elem_branch[-1]
                        w1e11 = np.where(section_elem[elem_i1,1]==section_elem[section_cell[ii_cell],1])[0]
                        w1e12 = np.where(section_elem[elem_i1,1]==section_elem[section_cell[ii_cell],2])[0]
                        w1e21 = np.where(section_elem[elem_i1,2]==section_elem[section_cell[ii_cell],1])[0]
                        w1e22 = np.where(section_elem[elem_i1,2]==section_elem[section_cell[ii_cell],2])[0]
                        w2e11 = np.where(section_elem[elem_i2,1]==section_elem[section_cell[ii_cell],1])[0]
                        w2e12 = np.where(section_elem[elem_i2,1]==section_elem[section_cell[ii_cell],2])[0]
                        w2e21 = np.where(section_elem[elem_i2,2]==section_elem[section_cell[ii_cell],1])[0]
                        w2e22 = np.where(section_elem[elem_i2,2]==section_elem[section_cell[ii_cell],2])[0]
                        aux_it += 1
                        if len(w1e11)>0 or len(w1e12)>0:
                            elem_i = elem_i1
                            point_i = int(section_elem[elem_i1,1])
                            point_f = int(section_elem[elem_i1,2])
                            break
                        elif len(w1e21)>0 or len(w1e22)>0:
                            elem_i = elem_i1
                            point_i = int(section_elem[elem_i1,2])
                            point_f = int(section_elem[elem_i1,1])
                            break
                        elif len(w2e11)>0 or len(w2e12)>0:
                            elem_i = elem_i2
                            point_i = int(section_elem[elem_i2,1])
                            point_f = int(section_elem[elem_i2,2])
                            break
                        elif len(w2e21)>0 or len(w2e22)>0:
                            elem_i = elem_i2
                            point_i = int(section_elem[elem_i2,2])
                            point_f = int(section_elem[elem_i2,1])
                            break  
                    warp[elem_i] = warpendnode[point_i]
                else:
                    elem_0 = elem_i
                    for ii_elbranch in np.arange(len(elem_branch)):
                        w1e11 = np.where(point_f==section_elem[elem_branch[ii_elbranch],1])[0]
                        w1e12 = np.where(point_f==section_elem[elem_branch[ii_elbranch],2])[0]
                        w1e21 = np.where(point_f==section_elem[elem_branch[ii_elbranch],1])[0]
                        w1e22 = np.where(point_f==section_elem[elem_branch[ii_elbranch],2])[0]
                        wused = np.where(elem_branch[ii_elbranch] == np.array(used_element))[0]
                        if len(w1e11)>0 and len(wused)==0:
                            elem_i = elem_branch[ii_elbranch]
                            point_i = point_f
                            point_f = int(section_elem[elem_branch[ii_elbranch],2])
                            break
                        if len(w1e12)>0 and len(wused)==0:
                            elem_i = elem_branch[ii_elbranch]
                            point_i = point_f
                            point_f = int(section_elem[elem_branch[ii_elbranch],1])
                            break
                        elif len(w1e21)>0 and len(wused)==0:
                            elem_i = elem_branch[ii_elbranch]
                            point_i = point_f
                            point_f = int(section_elem[elem_branch[ii_elbranch],2])
                            break   
                        elif len(w1e22)>0 and len(wused)==0:
                            elem_i = elem_branch[ii_elbranch]
                            point_i = point_f
                            point_f = int(section_elem[elem_branch[ii_elbranch],1])
                            break 
                    warp[elem_i] = warp[elem_0]
                used_element.append(elem_i)
                rx            = section.cg_local[elem_i,0]
                ry            = section.cg_local[elem_i,1]
                ds_vec2d      = section_points[point_f,:]-section_points[point_i,:]
                ds_vec2d_g    = section_points[int(section_elem[elem_i,2]),:]-section_points[int(section_elem[elem_i,1]),:]
                ds_vec[elem_i,:]  = [ds_vec2d[0],ds_vec2d[1],0]
                ds            = np.linalg.norm(ds_vec)
                ds_sum       += ds
                ds_def_adim   = ds_vec2d_g/ds
                r_t[ii_elem]  = np.dot(ds_def_adim,[rx,ry])
                dn_vec        = np.cross([0,0,1],ds_vec[elem_i,:])
                n_vec[ii_elem,:] = dn_vec
                dn_vec2d      = dn_vec[:2]
                dn_def_adim   = dn_vec2d/np.linalg.norm(dn_vec2d)
                r_n[ii_elem]  = np.dot(dn_def_adim,[rx,ry])
                A_cec_vec = -np.cross([rx,ry,0],ds_vec[elem_i,:])
                A_cec_vec_mid = -np.cross([rx/2,ry/2,0],ds_vec[elem_i,:])
                A_cec[elem_i] = A_cec_vec[2]
                A_cec_mid[elem_i] = A_cec_vec_mid[2]
                warpend       = warp[elem_i]+A_cec[elem_i]
                aa.append(warp[elem_i])
                warp[elem_i] += A_cec_mid[elem_i]
                warp_cum     += warpend*ds 
    # calculate the warping function
    warp_cum /= ds_sum
    warp -= warp_cum
    warp *= -1
    return warp, r_t, r_n, n_vec,ds_vec, H_mat, H_mat13, H_mat43, H_mat53, lambda_sec,mat_cell_elem,prev_elements
