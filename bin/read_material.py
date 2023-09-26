# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 13:18:42 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
read_material  : file containing functions for reading the material file and calculate material properties
last_version   : 17-02-2021
modified_by    : Andres Cremades Botella
"""

import numpy as np

#%% Functions
def read_material(material_file):
    # Function for reading the material properties from file
    # material_file  : name of the material properties file
    # -----------------------------------------------------------------------
    # word_ant        : word read before actual word
    # mat             : class to store material properties
    #   - mat.typemat : tipe of material isotropic (ISO) and orthotropic (ORT)
    #   - mat.EE      : young modulus for isotropic material
    #   - mat.GG      : shear modulus for isotropic material
    #   - mat.EEL     : longitudinal young modulus for orthotropic materials
    #   - mat.EET     : Transverse young modulus for orthotropic materials
    #   - mat.GGLT    : Longitudinal-transverse shear modulus for orthotropic materials
    #   - mat.GGTN    : Transverse-transverse shear modulus for orthotropic materials
    #   - mat.nuLT    : Longitudinal-transverse poisson ratio for orthotropic materials
    #   - mat.nuTN    : Transverse-transverse poisson ratio for orthotropic materials
    #   - mat.rho     : density
    word_ant = []
    class mat:
        pass
    mat.typemat = []
    # Read the configuration file word by word and find the values of the variables
    with open(material_file,"r") as textmaterial:
        for line in textmaterial:
            for word in line.split():
                if word_ant == "TYPE=":
                    mat.typemat = word
                if mat.typemat == "ISO":
                    if word_ant == "E=":
                        mat.EE = float(word)
                    elif word_ant == "G=":
                        mat.GG = float(word)
                    elif word_ant == "rho=":
                        mat.rho = float(word) 
                if mat.typemat == "ORT":
                    if word_ant == "EL=":
                        mat.EEL = float(word)
                    elif word_ant == "ET=":
                        mat.EET = float(word)
                    elif word_ant == "GLT=":
                        mat.GGLT = float(word)
                    elif word_ant == "GTN=":
                        mat.GGTN = float(word)
                    elif word_ant == "nuLT=":
                        mat.nuLT = float(word)
                    elif word_ant == "nuTN=":
                        mat.nuTN = float(word)
                    elif word_ant == "rho=":
                        mat.rho = float(word) 
                word_ant = word
    if mat.typemat == "ORT":
        mat.nuTL = mat.nuLT*mat.EET/mat.EEL
    return mat

#%%
def const_ortho(case_setup,sol_physuni,section_elem,section_orient,section_thick_ply,section_nplies,nmean_ply,section_ply_ncoord,mat):
    # Function to obtain the constitutive relations of an orthotropic material. Isotropic materials are transformed in orthotropic to use the same functions
    # case_setup         : configuration of the case
    # sol_physuni        : solid physics of the section
    # section_elem       : elments of the 2D section
    # section_orient     : orientation of the fiber in the section
    # section_thick_ply  : thickness of each section ply
    # section_nplies     : number of plies of each section
    # nmean_ply          : mean line of the ply
    # section_ply_ncoord : coordinate of each ply transversal to the mean chord of the wall
    # mat                : material information
    # -------------------------------------------------------------------------
    # Q11 : row 1 and column 1 of the constitutive matrix - sigma_tt - epsilon_tt
    # Q12 : row 1 and column 2 of the constitutive matrix - sigma_tt - epsilon_ll
    # Q22 : row 2 and column 2 of the constitutive matrix - sigma_ll - epsilon_ll
    # Q44 : row 4 and column 4 of the constitutive matrix - tau_ln - gamma_ln
    # Q55 : row 5 and column 5 of the constitutive matrix - tau_tn - gamma_tn
    # Q66 : row 6 and column 6 of the constitutive matrix - tua_lt - gamma_lt
    nnmax = 0
    for ii in mat:
        nnii = len(ii)
        if nnii > nnmax:
            nnmax = nnii
    Q11 = np.zeros((len(mat),nnmax))
    Q12 = np.zeros((len(mat),nnmax))
    Q22 = np.zeros((len(mat),nnmax))
    Q66 = np.zeros((len(mat),nnmax))
    Q44 = np.zeros((len(mat),nnmax))
    Q55 = np.zeros((len(mat),nnmax))
    # Calculate the constitutive matrix for every ply and element
    for ii_mat in np.arange(len(mat)):
        for jj_mat in np.arange(len(mat[ii_mat])):
            # Two models can be selected: Flat stress and beam stress. Flat stress is recommended
            if case_setup.stress_model == "FLAT_STRESS":
                Q11[ii_mat,jj_mat] = mat[ii_mat][jj_mat].EEL/(1-mat[ii_mat][jj_mat].nuLT*mat[ii_mat][jj_mat].nuTL)
                Q12[ii_mat,jj_mat] = mat[ii_mat][jj_mat].nuLT*mat[ii_mat][jj_mat].EET/(1-mat[ii_mat][jj_mat].nuLT*mat[ii_mat][jj_mat].nuTL)
                Q22[ii_mat,jj_mat] = mat[ii_mat][jj_mat].EET/(1-mat[ii_mat][jj_mat].nuLT*mat[ii_mat][jj_mat].nuTL)
                Q66[ii_mat,jj_mat] = mat[ii_mat][jj_mat].GGLT
                Q44[ii_mat,jj_mat] = mat[ii_mat][jj_mat].GGLT
                Q55[ii_mat,jj_mat] = mat[ii_mat][jj_mat].GGTN
            elif case_setup.stress_model == "BEAM_STRESS":
                Q11[ii_mat,jj_mat] = mat[ii_mat][jj_mat].EEL
                Q12[ii_mat,jj_mat] = 0
                Q22[ii_mat,jj_mat] = mat[ii_mat][jj_mat].EET
                Q66[ii_mat,jj_mat] = mat[ii_mat][jj_mat].GGLT
                Q44[ii_mat,jj_mat] = mat[ii_mat][jj_mat].GGLT
                Q55[ii_mat,jj_mat] = mat[ii_mat][jj_mat].GGTN
    # Qbar11 : row 1 and column 1 of the constitutive oriented matrix - sigma_ss - epsilon_ss
    # Qbar12 : row 1 and column 2 of the constitutive oriented matrix - sigma_ss - epsilon_zz
    # Qbar16 : row 1 and column 6 of the constitutive oriented matrix - sigma_ss - gamma_sz
    # Qbar22 : row 2 and column 2 of the constitutive oriented matrix - sigma_zz - epsilon_zz
    # Qbar26 : row 2 and column 6 of the constitutive oriented matrix - sigma_zz - gamma_sz
    # Qbar44 : row 4 and column 4 of the constitutive oriented matrix - tau_zn - gamma_zn
    # Qbar66 : row 6 and column 6 of the constitutive oriented matrix - tua_sz - gamma_sz
    sol_physuni.Qbar11 = [] 
    sol_physuni.Qbar12 = []
    sol_physuni.Qbar22 = []
    sol_physuni.Qbar16 = []
    sol_physuni.Qbar26 = []
    sol_physuni.Qbar66 = []
    sol_physuni.Qbar44 = []
    # For each ply and element of the section
    for ii_elem in np.arange(len(section_elem)):
        # if there is not a ply set to 1 the number of plies
        # jj_ply_len : number of plies
        if section_nplies[ii_elem] == 0:
            jj_ply_len = 1
        else:
            jj_ply_len = section_nplies[ii_elem]
        jj_ply_len = int(jj_ply_len)
        Qbar11     = np.zeros((int(jj_ply_len),))
        Qbar12     = np.zeros((int(jj_ply_len),))
        Qbar22     = np.zeros((int(jj_ply_len),))
        Qbar16     = np.zeros((int(jj_ply_len),))
        Qbar26     = np.zeros((int(jj_ply_len),))
        Qbar66     = np.zeros((int(jj_ply_len),))
        Qbar44     = np.zeros((int(jj_ply_len),))
        # n_orig     : position of the mean fiber of the wall
        # den_n_orig : denominator to ponderate the position of the mean fiber
        # theta      : orientation of the fiber
        n_orig     = 0
        den_n_orig = 0
        for jj_ply in np.arange(jj_ply_len):
            jj_ply         = int(jj_ply)
            theta          = section_orient[ii_elem][jj_ply]*np.pi/180
            Qbar11[jj_ply] = (Q11[ii_elem][jj_ply]*np.cos(theta)**4+Q22[ii_elem][jj_ply]*np.sin(theta)**4+2*(Q12[ii_elem][jj_ply]+2*Q66[ii_elem][jj_ply])*np.sin(theta)**2*np.cos(theta)**2)
            Qbar12[jj_ply] = ((Q11[ii_elem][jj_ply]+Q22[ii_elem][jj_ply]-4*Q66[ii_elem][jj_ply])*np.sin(theta)**2*np.cos(theta)**2+Q12[ii_elem][jj_ply]*(np.cos(theta)**4+np.sin(theta)**4))
            Qbar22[jj_ply] = (Q11[ii_elem][jj_ply]*np.sin(theta)**4+Q22[ii_elem][jj_ply]*np.cos(theta)**4+2*(Q12[ii_elem][jj_ply]+2*Q66[ii_elem][jj_ply])*np.sin(theta)**2*np.cos(theta)**2)
            Qbar16[jj_ply] = ((Q11[ii_elem][jj_ply]-Q12[ii_elem][jj_ply]-2*Q66[ii_elem][jj_ply])*np.cos(theta)**3*np.sin(theta) - (Q22[ii_elem][jj_ply]-Q12[ii_elem][jj_ply]-2*Q66[ii_elem][jj_ply])*np.cos(theta)*np.sin(theta)**3)
            Qbar26[jj_ply] = ((Q11[ii_elem][jj_ply]-Q12[ii_elem][jj_ply]-2*Q66[ii_elem][jj_ply])*np.cos(theta)*np.sin(theta)**3 - (Q22[ii_elem][jj_ply]-Q12[ii_elem][jj_ply]-2*Q66[ii_elem][jj_ply])*np.cos(theta)**3*np.sin(theta))
            Qbar66[jj_ply] = ((Q11[ii_elem][jj_ply]+Q22[ii_elem][jj_ply]-2*Q12[ii_elem][jj_ply]-2*Q66[ii_elem][jj_ply])*np.sin(theta)**2*np.cos(theta)**2+Q66[ii_elem][jj_ply]*(np.sin(theta)**4+np.cos(theta)**4))
            Qbar44[jj_ply] = (Q44[ii_elem][jj_ply]+Q55[ii_elem][jj_ply]+(Q44[ii_elem][jj_ply]-Q55[ii_elem][jj_ply])*np.cos(2*theta))/2   
            n_orig        += Qbar22[jj_ply]/Qbar22[0]*nmean_ply[ii_elem][jj_ply]*section_thick_ply[ii_elem][jj_ply] 
            den_n_orig    +=  Qbar22[jj_ply]/Qbar22[0]*section_thick_ply[ii_elem][jj_ply] 
        # Calculate the position of the mean fiber
        n_orig                      /= den_n_orig
        section_ply_ncoord[ii_elem] -= n_orig
        sol_physuni.Qbar11.append(Qbar11)
        sol_physuni.Qbar12.append(Qbar12)
        sol_physuni.Qbar22.append(Qbar22)
        sol_physuni.Qbar16.append(Qbar16)
        sol_physuni.Qbar26.append(Qbar26)
        sol_physuni.Qbar66.append(Qbar66)
        sol_physuni.Qbar44.append(Qbar44)
    return sol_physuni, section_ply_ncoord

#%%
def elastic_ortho(case_setup,sol_physuni,section_elem,section_points,section_orient,section_thick_ply,section_nplies,section_ply_ncoord,mat):
    # Function to calculate the stiffness and mass coefficients of the sections
    # case_setup         : configuration of the case
    # sol_physuni        : solid physics of the section
    # section_elem       : elments of the 2D section
    # section_points     : nodes of the 2D section
    # section_orient     : orientation of the fiber in the section
    # section_thick_ply  : thickness of each section ply
    # section_nplies     : number of plies of each section
    # section_ply_ncoord : coordinate of each ply transversal to the mean chord of the wall
    # mat                : material information
    # --------------------------------------------------------------------------------------
    # A11 : extensional stiffness - N_ss - epsilon_ss^0
    # A12 : extensional stiffness - N_zz - epsilon_ss^0
    # A22 : extensional stiffness - N_zz - epsilon_zz^0
    # A44 : shear stiffness - N_zn - gamma_zn^0
    # A16 : shear-extension stiffness - N_sz - epsilon_ss^0
    # A26 : shear-extension stiffness - N_sz - epsilon_zz^0
    # A66 : extensional stiffness - N_sz - gamma_sz^0
    # B11 : bending-extension stiffness - N_ss - epsilon_ss^1
    # B12 : bending-extension stiffness - N_zz - epsilon_ss^1
    # B22 : bending-extension stiffness - N_zz - epsilon_zz^1
    # B16 : bending-extension stiffness - N-sz - epsilon_ss^1
    # B26 : bending-extension stiffness - N_sz - epsilon_zz^1
    # B66 : bending-extension stiffness - N_sz - gamma_sz^1
    # D11 : bending stiffness - L_ss - epsilon_ss^1
    # D12 : bending stiffness - L_zz - epsilon_ss^1
    # D22 : bending stiffness - L_zz - epsilon_zz^1
    # D16 : bending stiffness - L_sz - epsilon_ss^1
    # D26 : bending stiffness - L_zz - gamma_sz^1
    # D66 : bending stiffness - L_sz - gamma_sz^1
    # m0  : reduced mass
    # m1  : reduced first mass moment
    # m2  : reduced second mass moment
    sol_physuni.A11 = []
    sol_physuni.A12 = []
    sol_physuni.A22 = []
    sol_physuni.A44 = []
    sol_physuni.A16 = []
    sol_physuni.A26 = []
    sol_physuni.A66 = []
    sol_physuni.B11 = []
    sol_physuni.B12 = []
    sol_physuni.B22 = []
    sol_physuni.B16 = [] 
    sol_physuni.B26 = []
    sol_physuni.B66 = []
    sol_physuni.D11 = []
    sol_physuni.D12 = []
    sol_physuni.D22 = []
    sol_physuni.D16 = []
    sol_physuni.D26 = []
    sol_physuni.D66 = []
    sol_physuni.m0  = []
    sol_physuni.m1  = []
    sol_physuni.m2  = []
    for ii_elem in np.arange(len(section_elem)):
        # jj_ply_len : number of plies
        if section_nplies[ii_elem] == 0:
            jj_ply_len = 1
        else:
            jj_ply_len = section_nplies[ii_elem]
        jj_ply_len = int(jj_ply_len)
        m0 = 0
        m1 = 0
        m2 = 0
        A11 = np.zeros((jj_ply_len,))
        A12 = np.zeros((jj_ply_len,))
        A22 = np.zeros((jj_ply_len,))
        A44 = np.zeros((jj_ply_len,))
        A16 = np.zeros((jj_ply_len,))
        A26 = np.zeros((jj_ply_len,))
        A66 = np.zeros((jj_ply_len,))
        B11 = np.zeros((jj_ply_len,))
        B12 = np.zeros((jj_ply_len,))
        B22 = np.zeros((jj_ply_len,))
        B16 = np.zeros((jj_ply_len,))
        B26 = np.zeros((jj_ply_len,))
        B66 = np.zeros((jj_ply_len,))
        D11 = np.zeros((jj_ply_len,))
        D12 = np.zeros((jj_ply_len,))
        D22 = np.zeros((jj_ply_len,))
        D16 = np.zeros((jj_ply_len,))
        D26 = np.zeros((jj_ply_len,))
        D66 = np.zeros((jj_ply_len,))
        # Calculate the stiffness and mass terms for each ply and element
        for jj_ply in np.arange(jj_ply_len):
            jj_ply = int(jj_ply)
            if section_nplies[ii_elem] == 0:
                A11[jj_ply] = sol_physuni.Qbar11[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]-section_ply_ncoord[ii_elem][0])
                A12[jj_ply] = sol_physuni.Qbar12[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]-section_ply_ncoord[ii_elem][0])
                A22[jj_ply] = sol_physuni.Qbar22[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]-section_ply_ncoord[ii_elem][0])
                A16[jj_ply] = sol_physuni.Qbar16[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]-section_ply_ncoord[ii_elem][0])
                A26[jj_ply] = sol_physuni.Qbar26[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]-section_ply_ncoord[ii_elem][0])
                A66[jj_ply] = sol_physuni.Qbar66[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]-section_ply_ncoord[ii_elem][0])
                A44[jj_ply] = sol_physuni.Qbar44[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]-section_ply_ncoord[ii_elem][0])
                B11[jj_ply] = 1/2*sol_physuni.Qbar11[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**2-section_ply_ncoord[ii_elem][0]**2)
                B12[jj_ply] = 1/2*sol_physuni.Qbar12[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**2-section_ply_ncoord[ii_elem][0]**2)
                B22[jj_ply] = 1/2*sol_physuni.Qbar22[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**2-section_ply_ncoord[ii_elem][0]**2)
                B16[jj_ply] = 1/2*sol_physuni.Qbar16[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**2-section_ply_ncoord[ii_elem][0]**2)
                B26[jj_ply] = 1/2*sol_physuni.Qbar26[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**2-section_ply_ncoord[ii_elem][0]**2)
                B66[jj_ply] = 1/2*sol_physuni.Qbar66[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**2-section_ply_ncoord[ii_elem][0]**2)
                D11[jj_ply] = 1/3*sol_physuni.Qbar11[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**3-section_ply_ncoord[ii_elem][0]**3)
                D12[jj_ply] = 1/3*sol_physuni.Qbar12[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**3-section_ply_ncoord[ii_elem][0]**3)
                D22[jj_ply] = 1/3*sol_physuni.Qbar22[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**3-section_ply_ncoord[ii_elem][0]**3)
                D16[jj_ply] = 1/3*sol_physuni.Qbar16[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**3-section_ply_ncoord[ii_elem][0]**3)
                D26[jj_ply] = 1/3*sol_physuni.Qbar26[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**3-section_ply_ncoord[ii_elem][0]**3)
                D66[jj_ply] = 1/3*sol_physuni.Qbar66[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][1]**3-section_ply_ncoord[ii_elem][0]**3)
                m0         += mat[jj_ply][0].rho*(section_ply_ncoord[ii_elem][1]-section_ply_ncoord[ii_elem][0])
                m1         += 1/2*mat[jj_ply][0].rho*(section_ply_ncoord[ii_elem][1]**2-section_ply_ncoord[ii_elem][0]**2)
                m2         += 1/3*mat[jj_ply][0].rho*(section_ply_ncoord[ii_elem][1]**3-section_ply_ncoord[ii_elem][0]**3)
            else:
                A11[jj_ply] = sol_physuni.Qbar11[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                A12[jj_ply] = sol_physuni.Qbar12[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                A22[jj_ply] = sol_physuni.Qbar22[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                A16[jj_ply] = sol_physuni.Qbar16[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                A26[jj_ply] = sol_physuni.Qbar26[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                A66[jj_ply] = sol_physuni.Qbar66[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                A44[jj_ply] = sol_physuni.Qbar44[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                B11[jj_ply] = 1/2*sol_physuni.Qbar11[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**2-section_ply_ncoord[ii_elem][jj_ply,0]**2)
                B12[jj_ply] = 1/2*sol_physuni.Qbar12[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**2-section_ply_ncoord[ii_elem][jj_ply,0]**2)
                B22[jj_ply] = 1/2*sol_physuni.Qbar22[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**2-section_ply_ncoord[ii_elem][jj_ply,0]**2)
                B16[jj_ply] = 1/2*sol_physuni.Qbar16[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**2-section_ply_ncoord[ii_elem][jj_ply,0]**2)
                B26[jj_ply] = 1/2*sol_physuni.Qbar26[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**2-section_ply_ncoord[ii_elem][jj_ply,0]**2)
                B66[jj_ply] = 1/2*sol_physuni.Qbar66[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**2-section_ply_ncoord[ii_elem][jj_ply,0]**2)
                D11[jj_ply] = 1/3*sol_physuni.Qbar11[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**3-section_ply_ncoord[ii_elem][jj_ply,0]**3)
                D12[jj_ply] = 1/3*sol_physuni.Qbar12[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**3-section_ply_ncoord[ii_elem][jj_ply,0]**3)
                D22[jj_ply] = 1/3*sol_physuni.Qbar22[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**3-section_ply_ncoord[ii_elem][jj_ply,0]**3)
                D16[jj_ply] = 1/3*sol_physuni.Qbar16[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**3-section_ply_ncoord[ii_elem][jj_ply,0]**3)
                D26[jj_ply] = 1/3*sol_physuni.Qbar26[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**3-section_ply_ncoord[ii_elem][jj_ply,0]**3)
                D66[jj_ply] = 1/3*sol_physuni.Qbar66[ii_elem][jj_ply]*(section_ply_ncoord[ii_elem][jj_ply,1]**3-section_ply_ncoord[ii_elem][jj_ply,0]**3)
                if section_nplies[ii_elem] == 1: 
                    m0 += mat[ii_elem][jj_ply].rho*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                    m1 += 1/2*mat[ii_elem][jj_ply].rho*(section_ply_ncoord[ii_elem][jj_ply,1]**2-section_ply_ncoord[ii_elem][jj_ply,0]**2)
                    m2 += 1/3*mat[ii_elem][jj_ply].rho*(section_ply_ncoord[ii_elem][jj_ply,1]**3-section_ply_ncoord[ii_elem][jj_ply,0]**3)
                else: 
                    m0 += mat[ii_elem][jj_ply].rho*(section_ply_ncoord[ii_elem][jj_ply,1]-section_ply_ncoord[ii_elem][jj_ply,0])
                    m1 += 1/2*mat[ii_elem][jj_ply].rho*(section_ply_ncoord[ii_elem][jj_ply,1]**2-section_ply_ncoord[ii_elem][jj_ply,0]**2)
                    m2 += 1/3*mat[ii_elem][jj_ply].rho*(section_ply_ncoord[ii_elem][jj_ply,1]**3-section_ply_ncoord[ii_elem][jj_ply,0]**3)
        sol_physuni.A11.append(A11)
        sol_physuni.A12.append(A12)
        sol_physuni.A22.append(A22)
        sol_physuni.A16.append(A16)
        sol_physuni.A26.append(A26)
        sol_physuni.A66.append(A66)
        sol_physuni.A44.append(A44)
        sol_physuni.B11.append(B11)
        sol_physuni.B12.append(B12)
        sol_physuni.B22.append(B22)
        sol_physuni.B16.append(B16) 
        sol_physuni.B26.append(B26)
        sol_physuni.B66.append(B66)
        sol_physuni.D11.append(D11)
        sol_physuni.D12.append(D12)
        sol_physuni.D22.append(D22)
        sol_physuni.D16.append(D16)
        sol_physuni.D26.append(D26)
        sol_physuni.D66.append(D66)
        sol_physuni.m0.append(m0)
        sol_physuni.m1.append(m1)
        sol_physuni.m2.append(m2)
    return sol_physuni

#%%
def stiff_ortho(sol_physuni,section_elem,section_points,section,mat,section_cell,section_branch,section_nplies):
    # Function to calculate the 2D stiffness and mass matrix
    # sol_physuni        : solid physics of the section
    # section_elem       : elments of the 2D section
    # section_points     : nodes of the 2D section
    # section            : information of the section
    # mat                : material information
    # section_cell       : for closed section the cells
    # section_nplies     : number of plies of each section
    # -------------------------------------------------------------------------
    # k11 : reduced stiffness N_zz - epsilon_zz^0
    # k12 : reduced stiffness N_zz - gamma_sz^0
    # k13 : reduced stiffness N_zz - W_M
    # k14 : reduced stiffness N_zz - epsilon_zz^1
    # k22 : reduced stiffness N_sz - epsilon_zz^0
    # k22 : reduced stiffness N_sz - gamma_sz^0
    # k23 : reduced stiffness N_sz - W_M
    # k24 : reduced stiffness N_sz - epsilon_zz^1
    # k43 : reduced stiffness L_zz - W_M
    # k44 : reduced stiffness L_zz - epsilon_zz^1
    # k51 : reduced stiffness L_sz - epsilon_zz^0
    # k52 : reduced stiffness L_sz - gamma_sz^0
    # k53 : reduced stiffness L_sz - W_M
    # k54 : reduced stiffness L_sz - epsilon_zz^1
    # a11 : stiffness matrix row 1 column 1 Tz - wo'
    # a12 : stiffness matrix row 1 column 2 Tz - thetay'
    # a13 : stiffness matrix row 1 column 3 Tz - thetax'
    # a14 : stiffness matrix row 1 column 2 Tz - up' + thetay
    # a15 : stiffness matrix row 1 column 3 Tz - vp' + thetax
    # a16 : stiffness matrix row 1 column 6 Tz - phi''
    # a17 : stiffness matrix row 1 column 7 Tz - phi'
    # a22 : stiffness matrix row 2 column 2 My - thetay'
    # a23 : stiffness matrix row 2 column 3 My - thetax'
    # a24 : stiffness matrix row 2 column 4 My - up' + thetay
    # a25 : stiffness matrix row 2 column 5 My - vp' + thetax
    # a26 : stiffness matrix row 2 column 6 My - phi''
    # a27 : stiffness matrix row 2 column 7 My - phi'
    # a33 : stiffness matrix row 3 column 3 Mx - thetax'
    # a34 : stiffness matrix row 3 column 4 Mx - up' + thetay
    # a35 : stiffness matrix row 3 column 5 Mx - vp' + thetax
    # a36 : stiffness matrix row 3 column 6 Mx - phi''
    # a37 : stiffness matrix row 3 column 7 Mx - phi'
    # a44 : stiffness matrix row 4 column 4 Qx - up' + thetay
    # a45 : stiffness matrix row 4 column 5 Qx - vp' + thetax
    # a46 : stiffness matrix row 4 column 6 Qx - phi''
    # a47 : stiffness matrix row 4 column 7 Qx - phi'
    # a55 : stiffness matrix row 5 column 5 Qy - vp' + thetax
    # a56 : stiffness matrix row 5 column 6 Qy - phi''
    # a57 : stiffness matrix row 5 column 7 Qy - phi'
    # a66 : stiffness matrix row 6 column 6 Bw - phi''
    # a67 : stiffness matrix row 6 column 7 Bw - phi'
    # a77 : stiffness matrix row 7 column 7 Mz - phi'
    # m11 : mass matrix row 1 column 1
    # m16 : mass matrix row 1 column 6
    # m22 : mass matrix row 2 column 2
    # m26 : mass matrix row 2 column 6
    # m33 : mass matrix row 3 column 3
    # m34 : mass matrix row 3 column 4
    # m35 : mass matrix row 3 column 5
    # m37 : mass matrix row 3 column 7
    # m44 : mass matrix row 4 column 4
    # m45 : mass matrix row 4 column 5
    # m47 : mass matrix row 4 column 7
    # m55 : mass matrix row 5 column 5
    # m57 : mass matrix row 5 column 7
    # m66 : mass matrix row 6 column 6
    # m77 : mass matrix row 7 column 7
    sol_physuni.k11 = []
    sol_physuni.k12 = []
    sol_physuni.k13 = []
    sol_physuni.k14 = []
    sol_physuni.k22 = []
    sol_physuni.k23 = []
    sol_physuni.k24 = []
    sol_physuni.k43 = []
    sol_physuni.k44 = []
    sol_physuni.k51 = []
    sol_physuni.k52 = []
    sol_physuni.k53 = []
    sol_physuni.k54 = []
    sol_physuni.a11 = 0
    sol_physuni.a12 = 0
    sol_physuni.a13 = 0
    sol_physuni.a14 = 0
    sol_physuni.a15 = 0
    sol_physuni.a16 = 0
    sol_physuni.a17 = 0
    sol_physuni.a22 = 0
    sol_physuni.a23 = 0
    sol_physuni.a24 = 0
    sol_physuni.a25 = 0
    sol_physuni.a26 = 0
    sol_physuni.a27 = 0
    sol_physuni.a33 = 0
    sol_physuni.a34 = 0
    sol_physuni.a35 = 0
    sol_physuni.a36 = 0
    sol_physuni.a37 = 0
    sol_physuni.a44 = 0
    sol_physuni.a45 = 0
    sol_physuni.a46 = 0
    sol_physuni.a47 = 0
    sol_physuni.a55 = 0
    sol_physuni.a56 = 0
    sol_physuni.a57 = 0
    sol_physuni.a66 = 0
    sol_physuni.a67 = 0
    sol_physuni.a77 = 0
    sol_physuni.m11 = 0
    sol_physuni.m22 = 0
    sol_physuni.m33 = 0
    sol_physuni.m16 = 0
    sol_physuni.m34 = 0
    sol_physuni.m26 = 0
    sol_physuni.m35 = 0
    sol_physuni.m44 = 0
    sol_physuni.m55 = 0
    sol_physuni.m66 = 0
    sol_physuni.m77 = 0
    sol_physuni.m37 = 0
    sol_physuni.m47 = 0
    sol_physuni.m57 = 0
    sol_physuni.m45 = 0
    yea = []
    # if it is open section    
    if len(section_cell)==0:
        # for all the elements
        for ii_elem in np.arange(len(section_elem)):
            # jj_ply_len : number of plies
            # ds         : element length
            # xe         : center of the element, coordinate x
            # ye         : center of the element, coordinate y
            # dx_ds      : variation of x in unitarian length element
            # dy_ds      : variation of y in unitarion length element
            if section_nplies[ii_elem] == 0:
                jj_ply_len = 1
            else:
                jj_ply_len = section_nplies[ii_elem]
            jj_ply_len = int(jj_ply_len)
            ds         = np.linalg.norm(section_points[int(section_elem[ii_elem, 2]),:]-section_points[int(section_elem[ii_elem, 1]),:])
            xe         = (section_points[int(section_elem[ii_elem, 2]),0]+section_points[int(section_elem[ii_elem, 1]),0])/2#-section.cg[0]
            ye         = (section_points[int(section_elem[ii_elem, 2]),1]+section_points[int(section_elem[ii_elem, 1]),1])/2#-section.cg[1]
            dx_ds      = (section_points[int(section_elem[ii_elem, 2]),0]-section_points[int(section_elem[ii_elem, 1]),0])/ds
            dy_ds      = (section_points[int(section_elem[ii_elem, 2]),1]-section_points[int(section_elem[ii_elem, 1]),1])/ds
            # Calculate the reduced stiffness matrix
            k11_elem = 0
            k12_elem = 0
            k14_elem = 0
            k22_elem = 0
            k24_elem = 0
            k44_elem = 0
            k51_elem = 0
            k52_elem = 0
            k54_elem = 0
            k13_elem = 0
            k23_elem = 0
            k43_elem = 0
            k53_elem = 0            
            for jj_ply in np.arange(jj_ply_len):
                k11_elem += sol_physuni.A22[ii_elem][jj_ply]-sol_physuni.A12[ii_elem][jj_ply]**2/sol_physuni.A11[ii_elem][jj_ply]
                k12_elem += sol_physuni.A26[ii_elem][jj_ply]-sol_physuni.A12[ii_elem][jj_ply]*sol_physuni.A16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k14_elem += sol_physuni.B22[ii_elem][jj_ply]-sol_physuni.A12[ii_elem][jj_ply]*sol_physuni.B12[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k22_elem += sol_physuni.A66[ii_elem][jj_ply]-sol_physuni.A16[ii_elem][jj_ply]**2/sol_physuni.A11[ii_elem][jj_ply]
                k24_elem += sol_physuni.B26[ii_elem][jj_ply]-sol_physuni.A16[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k44_elem += sol_physuni.D22[ii_elem][jj_ply]-sol_physuni.B12[ii_elem][jj_ply]**2/sol_physuni.A11[ii_elem][jj_ply]
                k51_elem += sol_physuni.B26[ii_elem][jj_ply]-sol_physuni.B16[ii_elem][jj_ply]*sol_physuni.A12[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k52_elem += sol_physuni.B66[ii_elem][jj_ply]-sol_physuni.B16[ii_elem][jj_ply]*sol_physuni.A16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k54_elem += sol_physuni.D26[ii_elem][jj_ply]-sol_physuni.B12[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k13_elem += 2*(sol_physuni.B26[ii_elem][jj_ply]-sol_physuni.A12[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply])
                k23_elem += 2*(sol_physuni.B66[ii_elem][jj_ply]-sol_physuni.A16[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply])
                k43_elem += 2*(sol_physuni.D26[ii_elem][jj_ply]-sol_physuni.B12[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply])
                k53_elem += 2*(sol_physuni.D66[ii_elem][jj_ply]-sol_physuni.B16[ii_elem][jj_ply]**2/sol_physuni.A11[ii_elem][jj_ply])            
            sol_physuni.k11.append(k11_elem)
            sol_physuni.k12.append(k12_elem)
            sol_physuni.k14.append(k14_elem)
            sol_physuni.k22.append(k22_elem)
            sol_physuni.k24.append(k24_elem)
            sol_physuni.k44.append(k44_elem)
            sol_physuni.k51.append(k51_elem)
            sol_physuni.k52.append(k52_elem)
            sol_physuni.k54.append(k54_elem)
            sol_physuni.k13.append(k13_elem)
            sol_physuni.k23.append(k23_elem)
            sol_physuni.k43.append(k43_elem)
            sol_physuni.k53.append(k53_elem)
            # Calculate the stiffness matrix elements
            sol_physuni.a11 += sol_physuni.k11[ii_elem]*ds
            sol_physuni.a12 += (sol_physuni.k11[ii_elem]*xe+sol_physuni.k14[ii_elem]*dy_ds)*ds
            sol_physuni.a13 += (sol_physuni.k11[ii_elem]*ye+sol_physuni.k14[ii_elem]*dx_ds)*ds
            sol_physuni.a14 += sol_physuni.k12[ii_elem]*dx_ds*ds
            sol_physuni.a15 += sol_physuni.k12[ii_elem]*dy_ds*ds
            sol_physuni.a16 += (sol_physuni.k11[ii_elem]*sol_physuni.warp[ii_elem]-sol_physuni.k14[ii_elem]*sol_physuni.r_t[ii_elem])*ds
            sol_physuni.a17 += sol_physuni.k13[ii_elem]*ds
            sol_physuni.a22 += (sol_physuni.k11[ii_elem]*xe**2+2*xe*sol_physuni.k14[ii_elem]*dy_ds+sol_physuni.k44[ii_elem]*dy_ds**2)*ds
            sol_physuni.a23 += (sol_physuni.k11[ii_elem]*xe*ye-xe*sol_physuni.k14[ii_elem]*dy_ds+ye*sol_physuni.k14[ii_elem]*dy_ds-sol_physuni.k44[ii_elem]*dy_ds*dx_ds)*ds
            sol_physuni.a24 += (sol_physuni.k12[ii_elem]*xe*dx_ds + sol_physuni.k24[ii_elem]*dx_ds*dy_ds)*ds
            sol_physuni.a25 += (sol_physuni.k12[ii_elem]*xe*dy_ds + sol_physuni.k24[ii_elem]*(dy_ds)**2)*ds
            sol_physuni.a26 += (sol_physuni.k11[ii_elem]*xe*sol_physuni.warp[ii_elem]-sol_physuni.k14[ii_elem]*xe*sol_physuni.r_t[ii_elem]+sol_physuni.warp[ii_elem]*sol_physuni.k14[ii_elem]*dy_ds-sol_physuni.k44[ii_elem]*sol_physuni.r_t[ii_elem]*dy_ds)*ds
            sol_physuni.a27 += (sol_physuni.k13[ii_elem]*xe+sol_physuni.k43[ii_elem]*dy_ds)*ds
            sol_physuni.a33 += (sol_physuni.k11[ii_elem]*ye**2-2*ye*sol_physuni.k14[ii_elem]*dx_ds+sol_physuni.k44[ii_elem]*dx_ds**2)*ds
            sol_physuni.a34 += (sol_physuni.k12[ii_elem]*ye*dx_ds-sol_physuni.k24[ii_elem]*(dx_ds)**2)*ds
            sol_physuni.a35 += (sol_physuni.k12[ii_elem]*ye*dy_ds-sol_physuni.k24[ii_elem]*dx_ds*dy_ds)*ds
            sol_physuni.a36 += (sol_physuni.k11[ii_elem]*ye*sol_physuni.warp[ii_elem]-sol_physuni.k14[ii_elem]*ye*sol_physuni.r_t[ii_elem]-sol_physuni.warp[ii_elem]*sol_physuni.k14[ii_elem]*dx_ds+sol_physuni.k44[ii_elem]*sol_physuni.r_t[ii_elem]*dx_ds)*ds #sol_physuni.k11[ii_elem]*ye*sol_physuni.warp[ii_elem]
            sol_physuni.a37 += (ye*sol_physuni.k13[ii_elem]-sol_physuni.k43[ii_elem]*dx_ds)*ds
            sol_physuni.a44 += (sol_physuni.k22[ii_elem]*dx_ds**2+sum(sol_physuni.A44[ii_elem])*dy_ds**2)*ds
            sol_physuni.a45 += (sol_physuni.k22[ii_elem]*dx_ds*dy_ds-sum(sol_physuni.A44[ii_elem])*dx_ds*dy_ds)*ds
            sol_physuni.a46 += (sol_physuni.warp[ii_elem]*sol_physuni.k12[ii_elem]*dx_ds-sol_physuni.k24[ii_elem]*sol_physuni.r_t[ii_elem]*dx_ds)*ds
            sol_physuni.a47 += sol_physuni.k23[ii_elem]*dx_ds*ds
            sol_physuni.a55 += (sol_physuni.k22[ii_elem]*dy_ds**2+sum(sol_physuni.A44[ii_elem])*dx_ds**2)*ds
            sol_physuni.a56 += (sol_physuni.warp[ii_elem]*sol_physuni.k12[ii_elem]*dy_ds-sol_physuni.k24[ii_elem]*sol_physuni.r_t[ii_elem]*dy_ds)*ds
            sol_physuni.a57 += sol_physuni.k23[ii_elem]*dy_ds*ds
            sol_physuni.a66 += (sol_physuni.k11[ii_elem]*sol_physuni.warp[ii_elem]**2-2*sol_physuni.k14[ii_elem]*sol_physuni.warp[ii_elem]*sol_physuni.r_t[ii_elem]+sol_physuni.k44[ii_elem]*sol_physuni.r_t[ii_elem]**2)*ds
            sol_physuni.a67 += (sol_physuni.k13[ii_elem]*sol_physuni.warp[ii_elem]-sol_physuni.k43[ii_elem]*sol_physuni.r_t[ii_elem])*ds
            sol_physuni.a77 += 2*sol_physuni.k53[ii_elem]*ds
            # Calculate mass parameters
            b1       = sol_physuni.m0[ii_elem]*ds
            b2       = sol_physuni.m0[ii_elem]*ye*ds
            b3       = sol_physuni.m0[ii_elem]*xe*ds
            b4       = sol_physuni.m0[ii_elem]*ye**2*ds
            b5       = sol_physuni.m0[ii_elem]*xe**2*ds
            b6       = sol_physuni.m0[ii_elem]*xe*ye*ds 
            b7       = sol_physuni.m0[ii_elem]*sol_physuni.warp[ii_elem]*ds
            b8       = sol_physuni.m0[ii_elem]*sol_physuni.warp[ii_elem]*ye*ds
            b9       = sol_physuni.m0[ii_elem]*sol_physuni.warp[ii_elem]*xe*ds
            b10      = sol_physuni.m0[ii_elem]*sol_physuni.warp[ii_elem]**2*ds
            b13      = sol_physuni.m2[ii_elem]*dx_ds*dy_ds*ds
            b14      = sol_physuni.m2[ii_elem]*dx_ds**2*ds
            b15      = sol_physuni.m2[ii_elem]*dy_ds**2*ds
            b16      = sol_physuni.m2[ii_elem]*sol_physuni.r_t[ii_elem]*dx_ds*ds
            b17      = sol_physuni.m2[ii_elem]*sol_physuni.r_t[ii_elem]*dy_ds*ds
            b18      = sol_physuni.m2[ii_elem]*sol_physuni.r_t[ii_elem]**2*ds
            b1_prima = sol_physuni.m1[ii_elem]*sol_physuni.r_t[ii_elem]*ds
            b2_prima = sol_physuni.m1[ii_elem]*dx_ds*ds
            b3_prima = sol_physuni.m1[ii_elem]*dy_ds*ds
            b4_prima = sol_physuni.m1[ii_elem]*(ye*dy_ds-xe*dx_ds)*ds
            b5_prima = sol_physuni.m1[ii_elem]*(sol_physuni.warp[ii_elem]*dx_ds-ye*sol_physuni.r_t[ii_elem])*ds
            b6_prima = sol_physuni.m1[ii_elem]*(sol_physuni.warp[ii_elem]*dy_ds+xe*sol_physuni.r_t[ii_elem])*ds
            b7_prima = sol_physuni.m1[ii_elem]*sol_physuni.warp[ii_elem]*sol_physuni.r_t[ii_elem]*ds
            b8_prima = sol_physuni.m1[ii_elem]*ye*dx_ds*ds
            b9_prima = sol_physuni.m1[ii_elem]*xe*dy_ds*ds
            # Calculate the mass matrix elements
            sol_physuni.m11 += b1
            sol_physuni.m22 += b1
            sol_physuni.m33 += b1
            delta_r          = 1
            sol_physuni.m16 += -b2+delta_r*b2_prima
            sol_physuni.m34 += b2-delta_r*b2_prima
            sol_physuni.m26 += b3+delta_r*b3_prima
            sol_physuni.m35 += b3+delta_r*b3_prima
            sol_physuni.m44 += b4+delta_r*b14-delta_r*2*b8_prima
            sol_physuni.m55 += b5+delta_r*b15+delta_r*2*b9_prima
            sol_physuni.m66 += b4+b5+delta_r*(b14+b15)+delta_r*2*(b9_prima-b8_prima)
            sol_physuni.m77 += b10+delta_r*b18-delta_r*2*b7_prima
            sol_physuni.m37 += -b7+delta_r*b1_prima
            sol_physuni.m47 += -b8-delta_r*b16+delta_r*b5_prima
            sol_physuni.m57 += -b9+delta_r*b17-delta_r*b6_prima
            sol_physuni.m45 += b6-delta_r*b13+delta_r*b4_prima
    # If the section is a closed section        
    else:
        # For each element in the section
        for ii_elem in np.arange(len(section_elem)):
            # jj_ply_len : number of plies
            # ds         : element length
            # xe         : center of the element, coordinate x
            # ye         : center of the element, coordinate y
            # dx_ds      : variation of x in unitarian length element
            # dy_ds      : variation of y in unitarion length element
            if section_nplies[ii_elem] == 0:
                jj_ply_len = 1
            else:
                jj_ply_len = section_nplies[ii_elem]
            jj_ply_len = int(jj_ply_len)
            ds         = np.linalg.norm(section_points[int(section_elem[ii_elem, 2]),:]-section_points[int(section_elem[ii_elem, 1]),:])
            xe         = (section_points[int(section_elem[ii_elem, 2]),0]+section_points[int(section_elem[ii_elem, 1]),0])/2#-section.cg[0]
            ye         = (section_points[int(section_elem[ii_elem, 2]),1]+section_points[int(section_elem[ii_elem, 1]),1])/2#-section.cg[1]
            yea.append(ye)
            dx_ds      = (section_points[int(section_elem[ii_elem, 2]),0]-section_points[int(section_elem[ii_elem, 1]),0])/ds
            dy_ds      = (section_points[int(section_elem[ii_elem, 2]),1]-section_points[int(section_elem[ii_elem, 1]),1])/ds
            # Calculate the reduced stiffness matrix
            k11_elem = 0
            k12_elem = 0
            k14_elem = 0
            k22_elem = 0
            k24_elem = 0
            k44_elem = 0
            k51_elem = 0
            k52_elem = 0
            k54_elem = 0
            for jj_ply in np.arange(jj_ply_len):
                k11_elem += sol_physuni.A22[ii_elem][jj_ply]-sol_physuni.A12[ii_elem][jj_ply]**2/sol_physuni.A11[ii_elem][jj_ply]
                k12_elem += sol_physuni.A26[ii_elem][jj_ply]-sol_physuni.A12[ii_elem][jj_ply]*sol_physuni.A16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k14_elem += sol_physuni.B22[ii_elem][jj_ply]-sol_physuni.A12[ii_elem][jj_ply]*sol_physuni.B12[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k22_elem += sol_physuni.A66[ii_elem][jj_ply]-sol_physuni.A16[ii_elem][jj_ply]**2/sol_physuni.A11[ii_elem][jj_ply]
                k24_elem += sol_physuni.B26[ii_elem][jj_ply]-sol_physuni.A16[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k44_elem += sol_physuni.D22[ii_elem][jj_ply]-sol_physuni.B12[ii_elem][jj_ply]**2/sol_physuni.A11[ii_elem][jj_ply]
                k51_elem += sol_physuni.B26[ii_elem][jj_ply]-sol_physuni.B16[ii_elem][jj_ply]*sol_physuni.A12[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k52_elem += sol_physuni.B66[ii_elem][jj_ply]-sol_physuni.B16[ii_elem][jj_ply]*sol_physuni.A16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]
                k54_elem += sol_physuni.D26[ii_elem][jj_ply]-sol_physuni.B12[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply]                
                k13_elem = sum(np.multiply(sol_physuni.H_mat13[:,0],sol_physuni.mat_cell_elem[:,ii_elem]))          
                k23_elem = sum(np.multiply(sol_physuni.H_mat[:,0],sol_physuni.mat_cell_elem[:,ii_elem]))          
                k43_elem = sum(np.multiply(sol_physuni.H_mat43[:,0],sol_physuni.mat_cell_elem[:,ii_elem]))          
                k53_elem = sum(np.multiply(sol_physuni.H_mat53[:,0],sol_physuni.mat_cell_elem[:,ii_elem]))
                k13_elem += 2*(sol_physuni.B26[ii_elem][jj_ply]-sol_physuni.A12[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply])
                k23_elem += 2*(sol_physuni.B66[ii_elem][jj_ply]-sol_physuni.A16[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply])
                k43_elem += 2*(sol_physuni.D26[ii_elem][jj_ply]-sol_physuni.B12[ii_elem][jj_ply]*sol_physuni.B16[ii_elem][jj_ply]/sol_physuni.A11[ii_elem][jj_ply])
                k53_elem += 2*(sol_physuni.D66[ii_elem][jj_ply]-sol_physuni.B16[ii_elem][jj_ply]**2/sol_physuni.A11[ii_elem][jj_ply]) 
            sol_physuni.k11.append(k11_elem)
            sol_physuni.k12.append(k12_elem)
            sol_physuni.k14.append(k14_elem)
            sol_physuni.k22.append(k22_elem)
            sol_physuni.k24.append(k24_elem)
            sol_physuni.k44.append(k44_elem)
            sol_physuni.k51.append(k51_elem)
            sol_physuni.k52.append(k52_elem)
            sol_physuni.k54.append(k54_elem)
            sol_physuni.k13.append(k13_elem)
            sol_physuni.k23.append(k23_elem)
            sol_physuni.k43.append(k43_elem)
            sol_physuni.k53.append(k53_elem)
            # Calculate the stiffness matrix elements
            sol_physuni.a11 += sol_physuni.k11[ii_elem]*ds
            sol_physuni.a12 += (sol_physuni.k11[ii_elem]*xe+sol_physuni.k14[ii_elem]*dy_ds)*ds
            sol_physuni.a13 += (sol_physuni.k11[ii_elem]*ye+sol_physuni.k14[ii_elem]*dx_ds)*ds
            sol_physuni.a14 += sol_physuni.k12[ii_elem]*dx_ds*ds
            sol_physuni.a15 += sol_physuni.k12[ii_elem]*dy_ds*ds
            sol_physuni.a16 += (sol_physuni.k11[ii_elem]*sol_physuni.warp[ii_elem]-sol_physuni.k14[ii_elem]*sol_physuni.r_t[ii_elem])*ds
            sol_physuni.a22 += (sol_physuni.k11[ii_elem]*xe**2+2*xe*sol_physuni.k14[ii_elem]*dy_ds+sol_physuni.k44[ii_elem]*dy_ds**2)*ds
            sol_physuni.a23 += (sol_physuni.k11[ii_elem]*xe*ye-xe*sol_physuni.k14[ii_elem]*dy_ds+ye*sol_physuni.k14[ii_elem]*dy_ds-sol_physuni.k44[ii_elem]*dy_ds*dx_ds)*ds
            sol_physuni.a24 += (sol_physuni.k12[ii_elem]*xe*dx_ds + sol_physuni.k24[ii_elem]*dx_ds*dy_ds)*ds
            sol_physuni.a25 += (sol_physuni.k12[ii_elem]*xe*dy_ds + sol_physuni.k24[ii_elem]*(dy_ds)**2)*ds
            sol_physuni.a26 += (sol_physuni.k11[ii_elem]*xe*sol_physuni.warp[ii_elem]-sol_physuni.k14[ii_elem]*xe*sol_physuni.r_t[ii_elem]+sol_physuni.warp[ii_elem]*sol_physuni.k14[ii_elem]*dy_ds-sol_physuni.k44[ii_elem]*sol_physuni.r_t[ii_elem]*dy_ds)*ds
            sol_physuni.a33 += (sol_physuni.k11[ii_elem]*ye**2-2*ye*sol_physuni.k14[ii_elem]*dx_ds+sol_physuni.k44[ii_elem]*dx_ds**2)*ds
            sol_physuni.a34 += (sol_physuni.k12[ii_elem]*ye*dx_ds-sol_physuni.k24[ii_elem]*(dx_ds)**2)*ds
            sol_physuni.a35 += (sol_physuni.k12[ii_elem]*ye*dy_ds-sol_physuni.k24[ii_elem]*dx_ds*dy_ds)*ds
            sol_physuni.a36 += (sol_physuni.k11[ii_elem]*ye*sol_physuni.warp[ii_elem]-sol_physuni.k14[ii_elem]*ye*sol_physuni.r_t[ii_elem]-sol_physuni.warp[ii_elem]*sol_physuni.k14[ii_elem]*dx_ds+sol_physuni.k44[ii_elem]*sol_physuni.r_t[ii_elem]*dx_ds)*ds
            sol_physuni.a44 += (sol_physuni.k22[ii_elem]*dx_ds**2+sum(sol_physuni.A44[ii_elem])*dy_ds**2)*ds
            sol_physuni.a45 += (sol_physuni.k22[ii_elem]*dx_ds*dy_ds-sum(sol_physuni.A44[ii_elem])*dx_ds*dy_ds)*ds
            sol_physuni.a46 += (sol_physuni.warp[ii_elem]*sol_physuni.k12[ii_elem]*dx_ds-sol_physuni.k24[ii_elem]*sol_physuni.r_t[ii_elem]*dx_ds)*ds
            sol_physuni.a55 += (sol_physuni.k22[ii_elem]*dy_ds**2+sum(sol_physuni.A44[ii_elem])*dx_ds**2)*ds
            sol_physuni.a56 += (sol_physuni.warp[ii_elem]*sol_physuni.k12[ii_elem]*dy_ds-sol_physuni.k24[ii_elem]*sol_physuni.r_t[ii_elem]*dy_ds)*ds
            sol_physuni.a66 += (sol_physuni.k11[ii_elem]*sol_physuni.warp[ii_elem]**2-2*sol_physuni.k14[ii_elem]*sol_physuni.warp[ii_elem]*sol_physuni.r_t[ii_elem]+sol_physuni.k44[ii_elem]*sol_physuni.r_t[ii_elem]**2)*ds
            sol_physuni.a17 +=  sol_physuni.k13[ii_elem]*ds
            sol_physuni.a27 +=  (sol_physuni.k13[ii_elem]*xe+sol_physuni.k43[ii_elem]*dy_ds)*ds
            sol_physuni.a37 +=  (ye*sol_physuni.k13[ii_elem]-sol_physuni.k43[ii_elem]*dx_ds)*ds
            sol_physuni.a47 +=  sol_physuni.k23[ii_elem]*dx_ds*ds
            sol_physuni.a57 +=  sol_physuni.k23[ii_elem]*dy_ds*ds
            sol_physuni.a67 +=  (sol_physuni.k13[ii_elem]*sol_physuni.warp[ii_elem]-sol_physuni.k43[ii_elem]*sol_physuni.r_t[ii_elem])*ds
            sol_physuni.a77 +=  sum(np.multiply(sol_physuni.lambda_sec[:,ii_elem],sol_physuni.mat_cell_elem[:,ii_elem]))*sol_physuni.k23[ii_elem]*ds+2*sol_physuni.k53[ii_elem]*ds #sol_physuni.psi_cells[ii_sec][jj_elem]*vec_k23[jj_elem]*ds+2*vec_k53[jj_elem]*ds
            # Calculate mass parameters
            b1       = sol_physuni.m0[ii_elem]*ds
            b2       = sol_physuni.m0[ii_elem]*ye*ds
            b3       = sol_physuni.m0[ii_elem]*xe*ds
            b4       = sol_physuni.m0[ii_elem]*ye**2*ds
            b5       = sol_physuni.m0[ii_elem]*xe**2*ds
            b6       = sol_physuni.m0[ii_elem]*xe*ye*ds 
            b7       = sol_physuni.m0[ii_elem]*sol_physuni.warp[ii_elem]*ds
            b8       = sol_physuni.m0[ii_elem]*sol_physuni.warp[ii_elem]*ye*ds
            b9       = sol_physuni.m0[ii_elem]*sol_physuni.warp[ii_elem]*xe*ds
            b10      = sol_physuni.m0[ii_elem]*sol_physuni.warp[ii_elem]**2*ds
            b13      = sol_physuni.m2[ii_elem]*dx_ds*dy_ds*ds
            b14      = sol_physuni.m2[ii_elem]*dx_ds**2*ds
            b15      = sol_physuni.m2[ii_elem]*dy_ds**2*ds
            b16      = sol_physuni.m2[ii_elem]*sol_physuni.r_t[ii_elem]*dx_ds*ds
            b17      = sol_physuni.m2[ii_elem]*sol_physuni.r_t[ii_elem]*dy_ds*ds
            b18      = sol_physuni.m2[ii_elem]*sol_physuni.r_t[ii_elem]**2*ds
            b1_prima = sol_physuni.m1[ii_elem]*sol_physuni.r_t[ii_elem]*ds
            b2_prima = sol_physuni.m1[ii_elem]*dx_ds*ds
            b3_prima = sol_physuni.m1[ii_elem]*dy_ds*ds
            b4_prima = sol_physuni.m1[ii_elem]*(ye*dy_ds-xe*dx_ds)*ds
            b5_prima = sol_physuni.m1[ii_elem]*(sol_physuni.warp[ii_elem]*dx_ds+ye*sol_physuni.r_t[ii_elem])*ds
            b6_prima = sol_physuni.m1[ii_elem]*(sol_physuni.warp[ii_elem]*dy_ds+xe*sol_physuni.r_t[ii_elem])*ds
            b7_prima = sol_physuni.m1[ii_elem]*sol_physuni.warp[ii_elem]*sol_physuni.r_t[ii_elem]*ds
            b8_prima = sol_physuni.m1[ii_elem]*ye*dx_ds*ds
            b9_prima = sol_physuni.m1[ii_elem]*xe*dy_ds*ds
            # Calculate the mass matrix elements
            sol_physuni.m11 += b1
            sol_physuni.m22 += b1
            sol_physuni.m33 += b1
            delta_r          = 1
            sol_physuni.m16 += -b2+delta_r*b2_prima
            sol_physuni.m34 += b2-delta_r*b2_prima
            sol_physuni.m26 += b3+delta_r*b3_prima
            sol_physuni.m35 += b3+delta_r*b3_prima
            sol_physuni.m44 += b4+delta_r*b14-delta_r*2*b8_prima
            sol_physuni.m55 += b5+delta_r*b15+delta_r*2*b9_prima
            sol_physuni.m66 += b4+b5+delta_r*(b14+b15)+delta_r*2*(b9_prima-b8_prima)
            sol_physuni.m77 += b10+delta_r*b18-delta_r*2*b7_prima
            sol_physuni.m37 += -b7*delta_r*b1_prima
            sol_physuni.m47 += -b8-delta_r*b16+delta_r*b5_prima
            sol_physuni.m57 += -b9+delta_r*b17-delta_r*b6_prima
            sol_physuni.m45 += b6-delta_r*b13+delta_r*b4_prima
        # Calculate the stiffness elements that depends on the closed cell
        vec_k13 = []
        vec_k23 = []
        vec_k43 = []
        vec_k53 = []
    return sol_physuni
