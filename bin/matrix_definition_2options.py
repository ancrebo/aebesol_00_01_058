# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:37:10 2020

@author            : Andres Cremades Botella - ancrebo@mot.upv.es
matrix_definition  : file containing functions for the definition of the element coordinate matrices
last_version       : 17-02-2021
modified_by        : Andres Cremades Botella
"""
import numpy as np

#%% Functions
def mass_matrix(L1D, mat,delta_sh):
    # Function to define the mass matrix
    # L1D : length of the beam element
    # mat : 2D mass matrix
    # -------------------------------------------------------------------------
    # M_Matrix : class to define the submatrices of the mass matrix
    #         .M00 : submatrix 0, 0
    #         .M01 : submatrix 0, 1
    #         .M10 : submatrix 1, 0
    #         .M11 : submatrix 1, 1
    class M_Matrix:
        pass    
    M_Matrix.M00 = np.zeros((9,9))
    M_Matrix.M01 = np.zeros((9,9))
    M_Matrix.M10 = np.zeros((9,9))
    M_Matrix.M11 = np.zeros((9,9))
    # Definition of the elements of the matrix
    # Notation: M_xx_yy
    #    - xx : index of the row
    #    - yy : index of the column
    M_01_01 = (1-delta_sh)*(13*L1D*mat.m11/35+6*mat.m55/(5*L1D))+delta_sh*(L1D*mat.m11/3)
    M_01_02 = (1-delta_sh)*(6*mat.m45/(5*L1D))
    M_01_03 = (1-delta_sh)*(mat.m35/2) 
    M_01_04 = (1-delta_sh)*(-mat.m45/10)
    M_01_05 = (1-delta_sh)*(1/210*(11*L1D**2*mat.m11+21*mat.m55))
    M_01_06 = (1-delta_sh)*(13*L1D*mat.m16/35-6*mat.m57/(5*L1D))+delta_sh*(7*L1D*mat.m16/20)
    M_01_07 = 0
    M_01_08 = 0
    M_01_09 = (1-delta_sh)*(1/210*(11*L1D**2*mat.m16-21*mat.m57))+delta_sh*(L1D**2*mat.m16/20)
    M_01_10 = (1-delta_sh)*(9*L1D*mat.m11/70-6*mat.m55/(5*L1D))+delta_sh*(L1D*mat.m11/6)
    M_01_11 = (1-delta_sh)*(-6*mat.m45/(5*L1D))
    M_01_12 = (1-delta_sh)*(mat.m35/2)
    M_01_13 = (1-delta_sh)*(-mat.m45/10)
    M_01_14 = (1-delta_sh)*(1/420*(-13*L1D**2*mat.m11+42*mat.m55))
    M_01_15 = (1-delta_sh)*(9*L1D*mat.m16/70+6*mat.m57/(5*L1D))+delta_sh*(3*L1D*mat.m16/20)
    M_01_16 = 0
    M_01_17 = 0
    M_01_18 = (1-delta_sh)*(1/420*(-13*L1D**2*mat.m16-42*mat.m57))+delta_sh*(-L1D**2*mat.m16/30)
    M_02_01 = M_01_02
    M_02_02 = (1-delta_sh)*(13*L1D*mat.m22/35+6*mat.m44/(5*L1D))+delta_sh*(L1D*mat.m22/3)
    M_02_03 = (1-delta_sh)*(mat.m34/2)
    M_02_04 = (1-delta_sh)*(1/210*(-11*L1D**2*mat.m22-21*mat.m44))
    M_02_05 = (1-delta_sh)*(mat.m45/10)
    M_02_06 = (1-delta_sh)*(13*L1D*mat.m26/35-6*mat.m47/(5*L1D))+delta_sh*(7*L1D*mat.m26/20)
    M_02_07 = 0
    M_02_08 = 0
    M_02_09 = (1-delta_sh)*(1/210*(11*L1D**2*mat.m26-21*mat.m47))+delta_sh*(L1D**2*mat.m26/20)
    M_02_10 = (1-delta_sh)*(-6*mat.m45/(5*L1D))
    M_02_11 = (1-delta_sh)*(9*L1D*mat.m22/70-6*mat.m44/(5*L1D))+delta_sh*(L1D*mat.m22/6)
    M_02_12 = (1-delta_sh)*(mat.m34/2)
    M_02_13 = (1-delta_sh)*(1/420*(13*L1D**2*mat.m22-42*mat.m44))
    M_02_14 = (1-delta_sh)*(mat.m45/10)
    M_02_15 = (1-delta_sh)*(9*L1D*mat.m26/70+6*mat.m47/(5*L1D))+delta_sh*(3*L1D*mat.m26/20)
    M_02_16 = 0
    M_02_17 = 0
    M_02_18 = (1-delta_sh)*(1/420*(-13*L1D**2*mat.m26-42*mat.m47))+delta_sh*(-L1D**2*mat.m26/30)
    M_03_01 = M_01_03
    M_03_02 = M_02_03
    M_03_03 = L1D*mat.m33/3
    M_03_04 = (1-delta_sh)*(L1D*mat.m34/12)+delta_sh*(L1D*mat.m34/3)
    M_03_05 = (1-delta_sh)*(-L1D*mat.m35/12)+delta_sh*(-L1D*mat.m35/3)
    M_03_06 = -mat.m37/2
    M_03_07 = 0
    M_03_08 = 0
    M_03_09 = L1D*mat.m37/12
    M_03_10 = (1-delta_sh)*(-mat.m35/2)
    M_03_11 = (1-delta_sh)*(-mat.m34/2)
    M_03_12 = L1D*mat.m33/6
    M_03_13 = (1-delta_sh)*(-L1D*mat.m34/12)+delta_sh*(L1D*mat.m34/6)
    M_03_14 = (1-delta_sh)*(L1D*mat.m35/12)+delta_sh*(-L1D*mat.m35/6)
    M_03_15 = mat.m37/2
    M_03_16 = 0
    M_03_17 = 0
    M_03_18 = -L1D*mat.m37/12
    M_04_01 = M_01_04
    M_04_02 = M_02_04
    M_04_03 = M_03_04
    M_04_04 = (1-delta_sh)*(1/105*(L1D**3*mat.m22+14*L1D*mat.m44))+delta_sh*(L1D*mat.m44/3)
    M_04_05 = (1-delta_sh)*(-2*L1D*mat.m45/15)+delta_sh*(-L1D*mat.m45/3)
    M_04_06 = (1-delta_sh)*(1/210*(-11*L1D**2*mat.m26+21*mat.m47))+delta_sh*(-mat.m47/2)
    M_04_07 = 0
    M_04_08 = 0
    M_04_09 = (1-delta_sh)*(1/105*(-L1D**3*mat.m26+14*L1D*mat.m47))+delta_sh*(L1D*mat.m47/12)
    M_04_10 = (1-delta_sh)*(mat.m45/10)
    M_04_11 = (1-delta_sh)*(1/420*(-13*L1D**2*mat.m22+42*mat.m44))
    M_04_12 = (1-delta_sh)*(-L1D*mat.m34/12)+delta_sh*(L1D*mat.m34/6)
    M_04_13 = (1-delta_sh)*(-1/420*L1D*(3*L1D**2*mat.m22+14*mat.m44))+delta_sh*(L1D*mat.m44/6)
    M_04_14 = (1-delta_sh)*(L1D*mat.m45/30)+delta_sh*(-L1D*mat.m45/6)
    M_04_15 = (1-delta_sh)*(1/420*(-13*L1D**2*mat.m26-42*mat.m47))+delta_sh*(mat.m47/2)
    M_04_16 = 0
    M_04_17 = 0
    M_04_18 = (1-delta_sh)*(1/420*(3*L1D**3*mat.m26-14*L1D*mat.m47))+delta_sh*(L1D*mat.m47/12)
    M_05_01 = M_01_05
    M_05_02 = M_02_05
    M_05_03 = M_03_05
    M_05_04 = M_04_05
    M_05_05 = (1-delta_sh)*(1/105*(L1D**3*mat.m11+14*L1D*mat.m55))+delta_sh*(L1D*mat.m55/3)
    M_05_06 = (1-delta_sh)*(1/210*(11*L1D**2*mat.m16-21*mat.m57))+delta_sh*(mat.m57/2)
    M_05_07 = 0
    M_05_08 = 0
    M_05_09 = (1-delta_sh)*(1/105*(L1D**3*mat.m16-14*L1D*mat.m57))+delta_sh*(-L1D*mat.m57/12)
    M_05_10 = (1-delta_sh)*(1/420*(13*L1D**2*mat.m11-42*mat.m55))
    M_05_11 = (1-delta_sh)*(-mat.m45/10)
    M_05_12 = (1-delta_sh)*(L1D*mat.m35/12)-delta_sh*(L1D*mat.m35/6)
    M_05_13 = (1-delta_sh)*(L1D*mat.m45/30)-delta_sh*(-L1D*mat.m45/6)
    M_05_14 = (1-delta_sh)*(-1/420*L1D*(3*L1D**2*mat.m11+14*mat.m55))+delta_sh*(L1D*mat.m55/6)
    M_05_15 = (1-delta_sh)*(1/420*(13*L1D**2*mat.m16+42*mat.m57))+delta_sh*(-mat.m57/2)
    M_05_16 = 0
    M_05_17 = 0
    M_05_18 = (1-delta_sh)*(1/420*(-3*L1D**3*mat.m16+14*L1D*mat.m57))+delta_sh*(L1D*mat.m57/12)
    M_06_01 = M_01_06
    M_06_02 = M_02_06
    M_06_03 = M_03_06
    M_06_04 = M_04_06
    M_06_05 = M_05_06
    M_06_06 = 13*L1D*mat.m66/35+6*mat.m77/(5*L1D)
    M_06_07 = 0
    M_06_08 = 0
    M_06_09 = 1/210*(11*L1D**2*mat.m66+21*mat.m77)
    M_06_10 = (1-delta_sh)*(9*L1D*mat.m16/70+6*mat.m57/(5*L1D))+delta_sh*(3*L1D*mat.m16/20)
    M_06_11 = (1-delta_sh)*(9*L1D*mat.m26/70+6*mat.m47/(5*L1D))+delta_sh*(3*L1D*mat.m26/20)
    M_06_12 = -mat.m37/2
    M_06_13 = (1-delta_sh)*(1/420*(13*L1D**2*mat.m26+42*mat.m47))+delta_sh*(-mat.m47/2)
    M_06_14 = (1-delta_sh)*(1/420*(-13*L1D**2*mat.m16-42*mat.m57))+delta_sh*(mat.m57/2)
    M_06_15 = 9*L1D*mat.m66/70-6*mat.m77/(5*L1D)
    M_06_16 = 0
    M_06_17 = 0
    M_06_18 = 1/420*(-13*L1D**2*mat.m66+42*mat.m77)
    M_07_01 = M_01_07
    M_07_02 = M_02_07
    M_07_03 = M_03_07
    M_07_04 = M_04_07
    M_07_05 = M_05_07
    M_07_06 = M_06_07
    M_07_07 = 0
    M_07_08 = 0
    M_07_09 = 0
    M_07_10 = 0
    M_07_11 = 0
    M_07_12 = 0
    M_07_13 = 0
    M_07_14 = 0
    M_07_15 = 0
    M_07_16 = 0
    M_07_17 = 0
    M_07_18 = 0
    M_08_01 = M_01_08
    M_08_02 = M_02_08
    M_08_03 = M_03_08
    M_08_04 = M_04_08
    M_08_05 = M_05_08
    M_08_06 = M_06_08
    M_08_07 = M_07_08
    M_08_08 = 0
    M_08_09 = 0
    M_08_10 = 0
    M_08_11 = 0
    M_08_12 = 0
    M_08_13 = 0
    M_08_14 = 0
    M_08_15 = 0
    M_08_16 = 0
    M_08_17 = 0
    M_08_18 = 0
    M_09_01 = M_01_09
    M_09_02 = M_02_09
    M_09_03 = M_03_09
    M_09_04 = M_04_09
    M_09_05 = M_05_09
    M_09_06 = M_06_09
    M_09_07 = M_07_09
    M_09_08 = M_08_09
    M_09_09 = 1/105*(L1D**3*mat.m66+14*L1D*mat.m77)
    M_09_10 = (1-delta_sh)*(1/420*(13*L1D**2*mat.m16+42*mat.m57))+delta_sh*(L1D**2*mat.m16/30)
    M_09_11 = (1-delta_sh)*(1/420*(13*L1D**2*mat.m26+42*mat.m47))+delta_sh*(L1D**2*mat.m26/30)
    M_09_12 = -L1D*mat.m37/12
    M_09_13 = (1-delta_sh)*(1/420*(3*L1D**3*mat.m26-14*L1D*mat.m47))+delta_sh*(-L1D*mat.m47/12)
    M_09_14 = (1-delta_sh)*(1/420*(-3*L1D**3*mat.m16+14*L1D*mat.m57))+delta_sh*(L1D*mat.m57/12)
    M_09_15 = 1/420*(13*L1D**2*mat.m66-42*mat.m77)
    M_09_16 = 0
    M_09_17 = 0
    M_09_18 = -1/420*L1D*(3*L1D**2*mat.m66+14*mat.m77)
    M_10_01 = M_01_10
    M_10_02 = M_02_10
    M_10_03 = M_03_10
    M_10_04 = M_04_10
    M_10_05 = M_05_10
    M_10_06 = M_06_10
    M_10_07 = M_07_10
    M_10_08 = M_08_10
    M_10_09 = M_09_10
    M_10_10 = (1-delta_sh)*(13*L1D*mat.m11/35+6*mat.m55/(5*L1D))+delta_sh*(L1D*mat.m11/3)
    M_10_11 = (1-delta_sh)*(6*mat.m45/(5*L1D))
    M_10_12 = (1-delta_sh)*(-mat.m35/2)
    M_10_13 = (1-delta_sh)*(mat.m45/10)
    M_10_14 = (1-delta_sh)*(1/210*(-11*L1D**2*mat.m11-21*mat.m55))
    M_10_15 = (1-delta_sh)*(13*L1D*mat.m16/35-6*mat.m57/(5*L1D))+delta_sh*(7*L1D*mat.m16/20)
    M_10_16 = 0
    M_10_17 = 0
    M_10_18 = (1-delta_sh)*(1/210*(-11*L1D**2*mat.m16+21*mat.m57))+delta_sh*(-L1D**2*mat.m16/20)
    M_11_01 = M_01_11
    M_11_02 = M_02_11
    M_11_03 = M_03_11
    M_11_04 = M_04_11
    M_11_05 = M_05_11
    M_11_06 = M_06_11
    M_11_07 = M_07_11
    M_11_08 = M_08_11
    M_11_09 = M_09_11
    M_11_10 = M_10_11
    M_11_11 = (1-delta_sh)*(13*L1D*mat.m22/35+6*mat.m44/(5*L1D))+delta_sh*(L1D*mat.m22/3)
    M_11_12 = (1-delta_sh)*(-mat.m34/2)
    M_11_13 = (1-delta_sh)*(1/210*(11*L1D**2*mat.m22+21*mat.m44))
    M_11_14 = (1-delta_sh)*(-mat.m45/10)
    M_11_15 = (1-delta_sh)*(13*L1D*mat.m26/35-6*mat.m47/(5*L1D))+delta_sh*(7*L1D*mat.m26/20)
    M_11_16 = 0
    M_11_17 = 0
    M_11_18 = (1-delta_sh)*(1/210*(-11*L1D**2*mat.m26+21*mat.m47))+delta_sh*(-L1D**2*mat.m26/20)
    M_12_01 = M_01_12
    M_12_02 = M_02_12
    M_12_03 = M_03_12
    M_12_04 = M_04_12
    M_12_05 = M_05_12
    M_12_06 = M_06_12
    M_12_07 = M_07_12
    M_12_08 = M_08_12
    M_12_09 = M_09_12
    M_12_10 = M_10_12
    M_12_11 = M_11_12
    M_12_12 = L1D*mat.m33/3
    M_12_13 = (1-delta_sh)*(L1D*mat.m34/12)+delta_sh*(L1D*mat.m34/3)
    M_12_14 = (1-delta_sh)*(-L1D*mat.m35/12)+delta_sh*(-L1D*mat.m35/3)
    M_12_15 = mat.m37/2
    M_12_16 = 0
    M_12_17 = 0
    M_12_18 = L1D*mat.m37/12
    M_13_01 = M_01_13
    M_13_02 = M_02_13
    M_13_03 = M_03_13
    M_13_04 = M_04_13
    M_13_05 = M_05_13
    M_13_06 = M_06_13
    M_13_07 = M_07_13
    M_13_08 = M_08_13
    M_13_09 = M_09_13
    M_13_10 = M_10_13
    M_13_11 = M_11_13
    M_13_12 = M_12_13
    M_13_13 = (1-delta_sh)*(1/105*(L1D**3*mat.m22+14*L1D*mat.m44))+delta_sh*(L1D*mat.m44/3)
    M_13_14 = (1-delta_sh)*(-2*L1D*mat.m45/15)+delta_sh*(-L1D*mat.m45/3)
    M_13_15 = (1-delta_sh)*(1/210*(11*L1D**2*mat.m26-21*mat.m47))+mat.m47/2
    M_13_16 = 0
    M_13_17 = 0
    M_13_18 = (1-delta_sh)*(1/105*(-L1D**3*mat.m26+14*L1D*mat.m47))+delta_sh*(L1D*mat.m47/12)
    M_14_01 = M_01_14
    M_14_02 = M_02_14
    M_14_03 = M_03_14
    M_14_04 = M_04_14
    M_14_05 = M_05_14
    M_14_06 = M_06_14
    M_14_07 = M_07_14
    M_14_08 = M_08_14
    M_14_09 = M_09_14
    M_14_10 = M_10_14
    M_14_11 = M_11_14
    M_14_12 = M_12_14
    M_14_13 = M_13_14
    M_14_14 = (1-delta_sh)*(1/105*(L1D**3*mat.m11+14*L1D*mat.m55))+delta_sh*(L1D*mat.m55/3)
    M_14_15 = (1-delta_sh)*(1/210*(-11*L1D**2*mat.m16+21*mat.m57))+delta_sh*(-mat.m57/2)
    M_14_16 = 0
    M_14_17 = 0
    M_14_18 = (1-delta_sh)*(1/105*(L1D**3*mat.m16-14*L1D*mat.m57))+delta_sh*(-L1D*mat.m57/12)
    M_15_01 = M_01_15
    M_15_02 = M_02_15
    M_15_03 = M_03_15
    M_15_04 = M_04_15
    M_15_05 = M_05_15
    M_15_06 = M_06_15
    M_15_07 = M_07_15
    M_15_08 = M_08_15
    M_15_09 = M_09_15
    M_15_10 = M_10_15
    M_15_11 = M_11_15
    M_15_12 = M_12_15
    M_15_13 = M_13_15
    M_15_14 = M_14_15
    M_15_15 = 13*L1D*mat.m66/35+6*mat.m77/(5*L1D)
    M_15_16 = 0
    M_15_17 = 0
    M_15_18 = 1/210*(-11*L1D**2*mat.m66-21*mat.m77)
    M_16_01 = M_01_16
    M_16_02 = M_02_16
    M_16_03 = M_03_16
    M_16_04 = M_04_16
    M_16_05 = M_05_16
    M_16_06 = M_06_16
    M_16_07 = M_07_16
    M_16_08 = M_08_16
    M_16_09 = M_09_16
    M_16_10 = M_10_16
    M_16_11 = M_11_16
    M_16_12 = M_12_16
    M_16_13 = M_13_16
    M_16_14 = M_14_16
    M_16_15 = M_15_16
    M_16_16 = 0
    M_16_17 = 0
    M_16_18 = 0
    M_17_01 = M_01_17
    M_17_02 = M_02_17
    M_17_03 = M_03_17
    M_17_04 = M_04_17
    M_17_05 = M_05_17
    M_17_06 = M_06_17
    M_17_07 = M_07_17
    M_17_08 = M_08_17
    M_17_09 = M_09_17
    M_17_10 = M_10_17
    M_17_11 = M_11_17
    M_17_12 = M_12_17
    M_17_13 = M_13_17
    M_17_14 = M_14_17
    M_17_15 = M_15_17
    M_17_16 = M_16_17
    M_17_17 = 0
    M_17_18 = 0
    M_18_01 = M_01_18
    M_18_02 = M_02_18
    M_18_03 = M_03_18
    M_18_04 = M_04_18
    M_18_05 = M_05_18
    M_18_06 = M_06_18
    M_18_07 = M_07_18
    M_18_08 = M_08_18
    M_18_09 = M_09_18
    M_18_10 = M_10_18
    M_18_11 = M_11_18
    M_18_12 = M_12_18
    M_18_13 = M_13_18
    M_18_14 = M_14_18
    M_18_15 = M_15_18
    M_18_16 = M_16_18
    M_18_17 = M_17_18
    M_18_18 = 1/105*(L1D**3*mat.m66+14*L1D*mat.m77)
    # Definition of the submatrices
    M_Matrix_00 = np.array([[M_01_01,M_01_02,M_01_03,M_01_04,M_01_05,M_01_06,M_01_07,M_01_08,M_01_09],
                            [M_02_01,M_02_02,M_02_03,M_02_04,M_02_05,M_02_06,M_02_07,M_02_08,M_02_09],
                            [M_03_01,M_03_02,M_03_03,M_03_04,M_03_05,M_03_06,M_03_07,M_03_08,M_03_09],
                            [M_04_01,M_04_02,M_04_03,M_04_04,M_04_05,M_04_06,M_04_07,M_04_08,M_04_09],
                            [M_05_01,M_05_02,M_05_03,M_05_04,M_05_05,M_05_06,M_05_07,M_05_08,M_05_09],
                            [M_06_01,M_06_02,M_06_03,M_06_04,M_06_05,M_06_06,M_06_07,M_06_08,M_06_09],
                            [M_07_01,M_07_02,M_07_03,M_07_04,M_07_05,M_07_06,M_07_07,M_07_08,M_07_09],
                            [M_08_01,M_08_02,M_08_03,M_08_04,M_08_05,M_08_06,M_08_07,M_08_08,M_08_09],
                            [M_09_01,M_09_02,M_09_03,M_09_04,M_09_05,M_09_06,M_09_07,M_09_08,M_09_09]])
    M_Matrix_01 = np.array([[M_01_10,M_01_11,M_01_12,M_01_13,M_01_14,M_01_15,M_01_16,M_01_17,M_01_18],
                            [M_02_10,M_02_11,M_02_12,M_02_13,M_02_14,M_02_15,M_02_16,M_02_17,M_02_18],
                            [M_03_10,M_03_11,M_03_12,M_03_13,M_03_14,M_03_15,M_03_16,M_03_17,M_03_18],
                            [M_04_10,M_04_11,M_04_12,M_04_13,M_04_14,M_04_15,M_04_16,M_04_17,M_04_18],
                            [M_05_10,M_05_11,M_05_12,M_05_13,M_05_14,M_05_15,M_05_16,M_05_17,M_05_18],
                            [M_06_10,M_06_11,M_06_12,M_06_13,M_06_14,M_06_15,M_06_16,M_06_17,M_06_18],
                            [M_07_10,M_07_11,M_07_12,M_07_13,M_07_14,M_07_15,M_07_16,M_07_17,M_07_18],
                            [M_08_10,M_08_11,M_08_12,M_08_13,M_08_14,M_08_15,M_08_16,M_08_17,M_08_18],
                            [M_09_10,M_09_11,M_09_12,M_09_13,M_09_14,M_09_15,M_09_16,M_09_17,M_09_18]])
    M_Matrix_10 = np.array([[M_10_01,M_10_02,M_10_03,M_10_04,M_10_05,M_10_06,M_10_07,M_10_08,M_10_09],
                            [M_11_01,M_11_02,M_11_03,M_11_04,M_11_05,M_11_06,M_11_07,M_11_08,M_11_09],
                            [M_12_01,M_12_02,M_12_03,M_12_04,M_12_05,M_12_06,M_12_07,M_12_08,M_12_09],
                            [M_13_01,M_13_02,M_13_03,M_13_04,M_13_05,M_13_06,M_13_07,M_13_08,M_13_09],
                            [M_14_01,M_14_02,M_14_03,M_14_04,M_14_05,M_14_06,M_14_07,M_14_08,M_14_09],
                            [M_15_01,M_15_02,M_15_03,M_15_04,M_15_05,M_15_06,M_15_07,M_15_08,M_15_09],
                            [M_16_01,M_16_02,M_16_03,M_16_04,M_16_05,M_16_06,M_16_07,M_16_08,M_16_09],
                            [M_17_01,M_17_02,M_17_03,M_17_04,M_17_05,M_17_06,M_17_07,M_17_08,M_17_09],
                            [M_18_01,M_18_02,M_18_03,M_18_04,M_18_05,M_18_06,M_18_07,M_18_08,M_18_09]])
    M_Matrix_11 = np.array([[M_10_10,M_10_11,M_10_12,M_10_13,M_10_14,M_10_15,M_10_16,M_10_17,M_10_18],
                            [M_11_10,M_11_11,M_11_12,M_11_13,M_11_14,M_11_15,M_11_16,M_11_17,M_11_18],
                            [M_12_10,M_12_11,M_12_12,M_12_13,M_12_14,M_12_15,M_12_16,M_12_17,M_12_18],
                            [M_13_10,M_13_11,M_13_12,M_13_13,M_13_14,M_13_15,M_13_16,M_13_17,M_13_18],
                            [M_14_10,M_14_11,M_14_12,M_14_13,M_14_14,M_14_15,M_14_16,M_14_17,M_14_18],
                            [M_15_10,M_15_11,M_15_12,M_15_13,M_15_14,M_15_15,M_15_16,M_15_17,M_15_18],
                            [M_16_10,M_16_11,M_16_12,M_16_13,M_16_14,M_16_15,M_16_16,M_16_17,M_16_18],
                            [M_17_10,M_17_11,M_17_12,M_17_13,M_17_14,M_17_15,M_17_16,M_17_17,M_17_18],
                            [M_18_10,M_18_11,M_18_12,M_18_13,M_18_14,M_18_15,M_18_16,M_18_17,M_18_18]])
    M_Matrix.M00 += M_Matrix_00
    M_Matrix.M01 += M_Matrix_01
    M_Matrix.M10 += M_Matrix_10
    M_Matrix.M11 += M_Matrix_11
    return M_Matrix
#%%    

def stiffness_matrix(L1D,mat,delta_sh):
    # Function to define the mass matrix
    # L1D : length of the beam element
    # mat : 2D mass matrix
    # -------------------------------------------------------------------------
    # K_Matrix : class to define the submatrices of the mass matrix
    #         .K00 : submatrix 0, 0
    #         .K01 : submatrix 0, 1
    #         .K10 : submatrix 1, 0
    #         .K11 : submatrix 1, 1   
    class K_Matrix:
        pass
    K_Matrix.K00 = np.zeros((9,9))
    K_Matrix.K01 = np.zeros((9,9))
    K_Matrix.K10 = np.zeros((9,9))
    K_Matrix.K11 = np.zeros((9,9))
    # Definition of the elements of the matrix
    # Notation: K_xx_yy
    #    - xx : index of the row
    #    - yy : index of the column
    K_01_01 = (1-delta_sh)*(12*mat.a22/(L1D**3))+delta_sh*(mat.a44/L1D)
    K_01_02 = (1-delta_sh)*(12*mat.a23/(L1D**3))+delta_sh*(mat.a45/L1D)
    K_01_03 = delta_sh*(mat.a14/L1D)
    K_01_04 = (1-delta_sh)*(-6*mat.a23/(L1D**2))+delta_sh*(-mat.a45/2+mat.a34/L1D)
    K_01_05 = (1-delta_sh)*(6*mat.a22/(L1D**2))+delta_sh*(mat.a44/2-mat.a24/L1D)
    K_01_06 = (1-delta_sh)*(12*mat.a26/L1D**3)+delta_sh*(mat.a47/L1D) 
    K_01_07 = 0
    K_01_08 = 0
    K_01_09 = (1-delta_sh)*((6*mat.a26+mat.a27*L1D)/L1D**2)+delta_sh*(-mat.a46/L1D)
    K_01_10 = (1-delta_sh)*(-12*mat.a22/(L1D**3))+delta_sh*(-mat.a44/L1D)
    K_01_11 = (1-delta_sh)*(-12*mat.a23/(L1D**3))+delta_sh*(-mat.a45/L1D)
    K_01_12 = delta_sh*(-mat.a14/L1D)
    K_01_13 = (1-delta_sh)*(-6*mat.a23/(L1D**2))+delta_sh*(-mat.a45/2-mat.a34/L1D)
    K_01_14 = (1-delta_sh)*(6*mat.a22/(L1D**2))+delta_sh*(1/2*(mat.a44+2*mat.a24/L1D))
    K_01_15 =(1-delta_sh)*(-12*mat.a26/L1D**3)+delta_sh*(-mat.a47/L1D)
    K_01_16 = 0
    K_01_17 = 0
    K_01_18 = (1-delta_sh)*(-(-6*mat.a26+mat.a27*L1D)/L1D**2)+delta_sh*(mat.a46/L1D)
    K_02_01 = K_01_02
    K_02_02 = (1-delta_sh)*(12*mat.a33/(L1D**3))+delta_sh*(mat.a55/L1D)
    K_02_03 = delta_sh*(mat.a15/L1D)
    K_02_04 = (1-delta_sh)*(-6*mat.a33/(L1D**2))+delta_sh*(-mat.a55/2+mat.a35/L1D)
    K_02_05 = (1-delta_sh)*(6*mat.a23/(L1D**2))+delta_sh*(mat.a45/2-mat.a25/L1D)
    K_02_06 = (1-delta_sh)*(12*mat.a36/L1D**3)+delta_sh*(mat.a57/L1D)
    K_02_07 = 0
    K_02_08 = 0
    K_02_09 = (1-delta_sh)*((6*mat.a36+mat.a37*L1D)/L1D**2)+delta_sh*(-mat.a56/L1D)
    K_02_10 = (1-delta_sh)*(-12*mat.a23/(L1D**3))+delta_sh*(-mat.a45/L1D)
    K_02_11 = (1-delta_sh)*(-12*mat.a33/(L1D**3))+delta_sh*(-mat.a55/L1D)
    K_02_12 = delta_sh*(-mat.a15/L1D)
    K_02_13 = (1-delta_sh)*(-6*mat.a33/(L1D**2))+delta_sh*(-mat.a55/2-mat.a35/L1D)
    K_02_14 = (1-delta_sh)*(6*mat.a23/(L1D**2))+delta_sh*(1/2*(mat.a45+2*mat.a25/L1D))
    K_02_15 = (1-delta_sh)*(-12*mat.a36/L1D**3)+delta_sh*(-mat.a57/L1D)
    K_02_16 = 0
    K_02_17 = 0
    K_02_18= (1-delta_sh)*((6*mat.a36-mat.a37*L1D)/L1D**2)+delta_sh*(mat.a56/L1D)
    K_03_01 = K_01_03
    K_03_02 = K_02_03
    K_03_03 = mat.a11/L1D
    K_03_04 = (1-delta_sh)*(mat.a13/L1D)+delta_sh*(-mat.a15/2+mat.a13/L1D)
    K_03_05 = (1-delta_sh)*(-mat.a12/L1D)+delta_sh*(mat.a14/2-mat.a12/L1D)
    K_03_06 = mat.a17/L1D
    K_03_07 = 0
    K_03_08 = 0
    K_03_09 = -mat.a16/L1D 
    K_03_10 = delta_sh*(-mat.a14/L1D)
    K_03_11 = delta_sh*(-mat.a15/L1D)
    K_03_12 = -mat.a11/L1D
    K_03_13 = (1-delta_sh)*(-mat.a13/L1D)+delta_sh*(-mat.a15/2-mat.a13/L1D)
    K_03_14 = (1-delta_sh)*(mat.a12/L1D)+delta_sh*(1/2*(mat.a14+2*mat.a12/L1D))
    K_03_15 = -mat.a17/L1D
    K_03_16 = 0
    K_03_17 = 0
    K_03_18 = mat.a16/L1D
    K_04_01 = K_01_04
    K_04_02 = K_02_04
    K_04_03 = K_03_04
    K_04_04 = (1-delta_sh)*(4*mat.a33/(L1D))+delta_sh*(-mat.a35+mat.a33/L1D+mat.a55*L1D/3)
    K_04_05 = (1-delta_sh)*(-4*mat.a23/(L1D))+delta_sh*(1/2*(mat.a25+mat.a34-2*mat.a23/L1D-2*mat.a45*L1D/3)) 
    K_04_06 = (1-delta_sh)*(-6*mat.a36+mat.a37*L1D)/(L1D**2)+delta_sh*(-mat.a57/2+(mat.a37+mat.a56)/L1D)
    K_04_07 = 0
    K_04_08 = 0
    K_04_09 = (1-delta_sh)*(-mat.a37/2-4*mat.a36/L1D)+delta_sh*(mat.a56-mat.a36/L1D+mat.a57*L1D/12)
    K_04_10 = (1-delta_sh)*(6*mat.a23/(L1D**2))+delta_sh*(1/2*(mat.a45-2*mat.a34/L1D))
    K_04_11 = (1-delta_sh)*(6*mat.a33/(L1D**2))+delta_sh*(1/2*(mat.a55-2*mat.a35/L1D))
    K_04_12 = (1-delta_sh)*(-mat.a13/L1D)+delta_sh*(1/2*(mat.a15-2*mat.a13/L1D))
    K_04_13 = (1-delta_sh)*(2*mat.a33/(L1D))+delta_sh*(-mat.a33/L1D+mat.a55*L1D/6)
    K_04_14 = (1-delta_sh)*(-2*mat.a23/(L1D))+delta_sh*(1/2*(-mat.a25+mat.a34+2*mat.a23/L1D-mat.a45*L1D/3))
    K_04_15 = (1-delta_sh)*((6*mat.a36-mat.a37*L1D)/(L1D**2))+delta_sh*(mat.a57/2-(mat.a37+mat.a56)/L1D)
    K_04_16 = 0
    K_04_17 = 0
    K_04_18 = (1-delta_sh)*(1/2*(mat.a37-4*mat.a36/L1D))+delta_sh*(mat.a36/L1D-mat.a57*L1D/12) 
    K_05_01 = K_01_05
    K_05_02 = K_02_05
    K_05_03 = K_03_05
    K_05_04 = K_04_05
    K_05_05 = (1-delta_sh)*(4*mat.a22/(L1D))+delta_sh*(-mat.a24+mat.a22/L1D+mat.a44*L1D/3)
    K_05_06 = (1-delta_sh)*(-(-6*mat.a26+mat.a27*L1D)/(L1D**2))+delta_sh*(mat.a47/2-(mat.a27+mat.a46)/L1D) 
    K_05_07 = 0
    K_05_08 = 0
    K_05_09 = (1-delta_sh)*(-(-mat.a27/2-4*mat.a26/L1D))+delta_sh*(-mat.a46+mat.a26/L1D-mat.a47*L1D/12)
    K_05_10 = (1-delta_sh)*(-6*mat.a22/(L1D**2))+delta_sh*(-mat.a44/2+mat.a24/L1D)
    K_05_11 = (1-delta_sh)*(-6*mat.a23/(L1D**2))+delta_sh*(-mat.a45/2+mat.a25/L1D)
    K_05_12 = (1-delta_sh)*(mat.a12/L1D)+delta_sh*(-mat.a14/2+mat.a12/L1D)
    K_05_13 = (1-delta_sh)*(-2*mat.a23/(L1D))+delta_sh*(1/2*(mat.a25-mat.a34+2*mat.a23/L1D-mat.a45*L1D/3))
    K_05_14 = (1-delta_sh)*(2*mat.a22/(L1D))+delta_sh*(-mat.a22/L1D+mat.a44*L1D/6)
    K_05_15 = (1-delta_sh)*(-(6*mat.a26-mat.a27*L1D)/(L1D**2))+delta_sh*(-mat.a47/2+(mat.a27+mat.a46)/L1D) 
    K_05_16 = 0
    K_05_17 = 0
    K_05_18 = (1-delta_sh)*(-1/2*(mat.a27-4*mat.a26/L1D))+delta_sh*(-mat.a26/L1D+mat.a47*L1D/12)
    K_06_01 = K_01_06
    K_06_02 = K_02_06
    K_06_03 = K_03_06
    K_06_04 = K_04_06
    K_06_05 = K_05_06
    K_06_06 = 6*(10*mat.a66+mat.a77*L1D**2)/(5*L1D**3)
    K_06_07 = 0
    K_06_08 = 0
    K_06_09 = mat.a77/10+6*mat.a66/L1D**2 
    K_06_10 = (1-delta_sh)*(-12*mat.a26/L1D**3)+delta_sh*(-mat.a47/L1D) 
    K_06_11 = (1-delta_sh)*(-12*mat.a36/L1D**3)+delta_sh*(-mat.a57/L1D)
    K_06_12 = -mat.a17/L1D
    K_06_13 = (1-delta_sh)*(-(6*mat.a36+mat.a37*L1D)/(L1D**2))+delta_sh*(-mat.a57/2-(mat.a37+mat.a56)/L1D)
    K_06_14 = (1-delta_sh)*((6*mat.a26+mat.a27*L1D)/(L1D**2))+delta_sh*(mat.a47/2+(mat.a27+mat.a46)/L1D)
    K_06_15 = -6*(10*mat.a66+mat.a77*L1D**2)/(5*L1D**3)
    K_06_16 = 0
    K_06_17 = 0
    K_06_18 = mat.a77/10+6*mat.a66/L1D**2
    K_07_01 = K_01_07
    K_07_02 = K_02_07
    K_07_03 = K_03_07
    K_07_04 = K_04_07
    K_07_05 = K_05_07
    K_07_06 = K_06_07
    K_07_07 = 0
    K_07_08 = 0
    K_07_09 = 0
    K_07_10 = 0
    K_07_11 = 0
    K_07_12 = 0
    K_07_13 = 0
    K_07_14 = 0
    K_07_15 = 0
    K_07_16 = 0
    K_07_17 = 0
    K_07_18 = 0
    K_08_01 = K_01_08
    K_08_02 = K_02_08
    K_08_03 = K_03_08
    K_08_04 = K_04_08
    K_08_05 = K_05_08
    K_08_06 = K_06_08
    K_08_07 = K_07_08
    K_08_08 = 0
    K_08_09 = 0
    K_08_10 = 0
    K_08_11 = 0
    K_08_12 = 0
    K_08_13 = 0
    K_08_14 = 0
    K_08_15 = 0
    K_08_16 = 0
    K_08_17 = 0
    K_08_18 = 0
    K_09_01 = K_01_09
    K_09_02 = K_02_09
    K_09_03 = K_03_09
    K_09_04 = K_04_09
    K_09_05 = K_05_09
    K_09_06 = K_06_09
    K_09_07 = K_07_09
    K_09_08 = K_08_09
    K_09_09 = mat.a67+4*mat.a66/L1D+2*mat.a77*L1D/15
    K_09_10 = (1-delta_sh)*(-(6*mat.a26+mat.a27*L1D)/L1D**2)+delta_sh*(mat.a46/L1D)
    K_09_11 = (1-delta_sh)*(-(6*mat.a36+mat.a37*L1D)/L1D**2)+delta_sh*(mat.a56/L1D)
    K_09_12 = mat.a16/L1D
    K_09_13 = (1-delta_sh)*(-mat.a37/2-2*mat.a36/L1D)+delta_sh*(mat.a36/L1D-mat.a57*L1D/12) 
    K_09_14 = (1-delta_sh)*(-(-mat.a27/2-2*mat.a26/L1D))+delta_sh*(-mat.a26/L1D+mat.a47*L1D/12) 
    K_09_15 = -mat.a77/10-6*mat.a66/L1D**2
    K_09_16 = 0
    K_09_17 = 0
    K_09_18 = 2*mat.a66/L1D-mat.a77*L1D/30
    K_10_01 = K_01_10
    K_10_02 = K_02_10
    K_10_03 = K_03_10
    K_10_04 = K_04_10
    K_10_05 = K_05_10
    K_10_06 = K_06_10
    K_10_07 = K_07_10
    K_10_08 = K_08_10
    K_10_09 = K_09_10
    K_10_10 = (1-delta_sh)*(12*mat.a22/(L1D**3))+delta_sh*(mat.a44/L1D)
    K_10_11 = (1-delta_sh)*(12*mat.a23/(L1D**3))+delta_sh*(mat.a45/L1D)
    K_10_12 = delta_sh*(mat.a14/L1D)
    K_10_13 = (1-delta_sh)*(6*mat.a23/(L1D**2))+delta_sh*(mat.a45/2+mat.a34/L1D)
    K_10_14 = (1-delta_sh)*(-6*mat.a22/(L1D**2))+delta_sh*(-mat.a44/2-mat.a24/L1D)
    K_10_15 = (1-delta_sh)*(12*mat.a26/L1D**3)+delta_sh*(mat.a47/L1D)
    K_10_16 = 0
    K_10_17 = 0
    K_10_18 = (1-delta_sh)*(-(6*mat.a26-mat.a27*L1D)/L1D**2)+delta_sh*(-mat.a46/L1D)
    K_11_01 = K_01_11
    K_11_02 = K_02_11
    K_11_03 = K_03_11
    K_11_04 = K_04_11
    K_11_05 = K_05_11
    K_11_06 = K_06_11
    K_11_07 = K_07_11
    K_11_08 = K_08_11
    K_11_09 = K_09_11
    K_11_10 = K_10_11
    K_11_11 = (1-delta_sh)*(12*mat.a33/(L1D**3))+delta_sh*(mat.a55/L1D)
    K_11_12 = delta_sh*(mat.a15/L1D)
    K_11_13 = (1-delta_sh)*(6*mat.a33/(L1D**2))+delta_sh*(mat.a55/2+mat.a35/L1D)
    K_11_14 = (1-delta_sh)*(-6*mat.a23/(L1D**2))+delta_sh*(-mat.a45/2-mat.a25/L1D)
    K_11_15 = (1-delta_sh)*(12*mat.a36/L1D**3)+delta_sh*(mat.a57/L1D)
    K_11_16 = 0
    K_11_17 = 0
    K_11_18 = (1-delta_sh)*((-6*mat.a36+mat.a37*L1D)/L1D**2)+delta_sh*(-mat.a56/L1D)
    K_12_01 = K_01_12
    K_12_02 = K_02_12
    K_12_03 = K_03_12
    K_12_04 = K_04_12
    K_12_05 = K_05_12
    K_12_06 = K_06_12
    K_12_07 = K_07_12
    K_12_08 = K_08_12
    K_12_09 = K_09_12
    K_12_10 = K_10_12
    K_12_11 = K_11_12
    K_12_12 = mat.a11/L1D
    K_12_13 = (1-delta_sh)*(mat.a13/L1D)+delta_sh*(mat.a15/2+mat.a13/L1D)
    K_12_14 = (1-delta_sh)*(-mat.a12/L1D)+delta_sh*(-mat.a14/2-mat.a12/L1D)
    K_12_15 = mat.a17/L1D
    K_12_16 = 0
    K_12_17 = 0
    K_12_18 = -mat.a16/L1D
    K_13_01 = K_01_13
    K_13_02 = K_02_13
    K_13_03 = K_03_13
    K_13_04 = K_04_13
    K_13_05 = K_05_13
    K_13_06 = K_06_13
    K_13_07 = K_07_13
    K_13_08 = K_08_13
    K_13_09 = K_09_13
    K_13_10 = K_10_13
    K_13_11 = K_11_13
    K_13_12 = K_12_13
    K_13_13 = (1-delta_sh)*(4*mat.a33/(L1D))+delta_sh*(mat.a35+mat.a33/L1D+mat.a55*L1D/3)
    K_13_14 = (1-delta_sh)*(-4*mat.a23/(L1D))+delta_sh*(-(6*mat.a23+L1D*(3*(mat.a25+mat.a34)+2*mat.a45*L1D))/(6*L1D))
    K_13_15 = (1-delta_sh)*((6*mat.a36+mat.a37*L1D)/(L1D**2))+delta_sh*(mat.a57/2+(mat.a37+mat.a56)/L1D)
    K_13_16 = 0
    K_13_17 = 0
    K_13_18 = (1-delta_sh)*(1/2*(mat.a37-8*mat.a36/L1D))+delta_sh*(-mat.a56-mat.a36/L1D+mat.a57*L1D/12)
    K_14_01 = K_01_14
    K_14_02 = K_02_14
    K_14_03 = K_03_14
    K_14_04 = K_04_14
    K_14_05 = K_05_14
    K_14_06 = K_06_14
    K_14_07 = K_07_14
    K_14_08 = K_08_14
    K_14_09 = K_09_14
    K_14_10 = K_10_14
    K_14_11 = K_11_14
    K_14_12 = K_12_14
    K_14_13 = K_13_14
    K_14_14 = (1-delta_sh)*(4*mat.a22/(L1D))+delta_sh*(mat.a24+mat.a22/L1D+mat.a44*L1D/3)
    K_14_15 = (1-delta_sh)*(-(6*mat.a26+mat.a27*L1D)/(L1D**2))+delta_sh*(-mat.a47/2-(mat.a27+mat.a46)/L1D)
    K_14_16 = 0
    K_14_17 = 0
    K_14_18 = (1-delta_sh)*(-1/2*(mat.a27-8*mat.a26/L1D))+delta_sh*(mat.a46+mat.a26/L1D-mat.a47*L1D/12)
    K_15_01 = K_01_15
    K_15_02 = K_02_15
    K_15_03 = K_03_15
    K_15_04 = K_04_15
    K_15_05 = K_05_15
    K_15_06 = K_06_15
    K_15_07 = K_07_15
    K_15_08 = K_08_15
    K_15_09 = K_09_15
    K_15_10 = K_10_15
    K_15_11 = K_11_15
    K_15_12 = K_12_15
    K_15_13 = K_13_15
    K_15_14 = K_14_15
    K_15_15 = 6*(10*mat.a66+mat.a77*L1D**2)/(5*L1D**3)
    K_15_16 = 0
    K_15_17 = 0
    K_15_18 = -mat.a77/10-6*mat.a66/L1D**2
    K_16_01 = K_01_16
    K_16_02 = K_02_16
    K_16_03 = K_03_16
    K_16_04 = K_04_16
    K_16_05 = K_05_16
    K_16_06 = K_06_16
    K_16_07 = K_07_16
    K_16_08 = K_08_16
    K_16_09 = K_09_16
    K_16_10 = K_10_16
    K_16_11 = K_11_16
    K_16_12 = K_12_16
    K_16_13 = K_13_16
    K_16_14 = K_14_16
    K_16_15 = K_15_16
    K_16_16 = 0
    K_16_17 = 0
    K_16_18 = 0
    K_17_01 = K_01_17
    K_17_02 = K_02_17
    K_17_03 = K_03_17
    K_17_04 = K_04_17
    K_17_05 = K_05_17
    K_17_06 = K_06_17
    K_17_07 = K_07_17
    K_17_08 = K_08_17
    K_17_09 = K_09_17
    K_17_10 = K_10_17
    K_17_11 = K_11_17
    K_17_12 = K_12_17
    K_17_13 = K_13_17
    K_17_14 = K_14_17
    K_17_15 = K_15_17
    K_17_16 = K_16_17
    K_17_17 = 0
    K_17_18 = 0
    K_18_01 = K_01_18
    K_18_02 = K_02_18
    K_18_03 = K_03_18
    K_18_04 = K_04_18
    K_18_05 = K_05_18
    K_18_06 = K_06_18
    K_18_07 = K_07_18
    K_18_08 = K_08_18
    K_18_09 = K_09_18
    K_18_10 = K_10_18
    K_18_11 = K_11_18
    K_18_12 = K_12_18
    K_18_13 = K_13_18
    K_18_14 = K_14_18
    K_18_15 = K_15_18
    K_18_16 = K_16_18
    K_18_17 = K_17_18
    K_18_18 = -mat.a67+4*mat.a66/L1D+2*mat.a77*L1D/15
    # Definition of the submatrices
    K_Matrix_00 = np.array([[K_01_01,K_01_02,K_01_03,K_01_04,K_01_05,K_01_06,K_01_07,K_01_08,K_01_09],
                            [K_02_01,K_02_02,K_02_03,K_02_04,K_02_05,K_02_06,K_02_07,K_02_08,K_02_09],
                            [K_03_01,K_03_02,K_03_03,K_03_04,K_03_05,K_03_06,K_03_07,K_03_08,K_03_09],
                            [K_04_01,K_04_02,K_04_03,K_04_04,K_04_05,K_04_06,K_04_07,K_04_08,K_04_09],
                            [K_05_01,K_05_02,K_05_03,K_05_04,K_05_05,K_05_06,K_05_07,K_05_08,K_05_09],
                            [K_06_01,K_06_02,K_06_03,K_06_04,K_06_05,K_06_06,K_06_07,K_06_08,K_06_09],
                            [K_07_01,K_07_02,K_07_03,K_07_04,K_07_05,K_07_06,K_07_07,K_07_08,K_07_09],
                            [K_08_01,K_08_02,K_08_03,K_08_04,K_08_05,K_08_06,K_08_07,K_08_08,K_08_09],
                            [K_09_01,K_09_02,K_09_03,K_09_04,K_09_05,K_09_06,K_09_07,K_09_08,K_09_09]])
    K_Matrix_01 = np.array([[K_01_10,K_01_11,K_01_12,K_01_13,K_01_14,K_01_15,K_01_16,K_01_17,K_01_18],
                            [K_02_10,K_02_11,K_02_12,K_02_13,K_02_14,K_02_15,K_02_16,K_02_17,K_02_18],
                            [K_03_10,K_03_11,K_03_12,K_03_13,K_03_14,K_03_15,K_03_16,K_03_17,K_03_18],
                            [K_04_10,K_04_11,K_04_12,K_04_13,K_04_14,K_04_15,K_04_16,K_04_17,K_04_18],
                            [K_05_10,K_05_11,K_05_12,K_05_13,K_05_14,K_05_15,K_05_16,K_05_17,K_05_18],
                            [K_06_10,K_06_11,K_06_12,K_06_13,K_06_14,K_06_15,K_06_16,K_06_17,K_06_18],
                            [K_07_10,K_07_11,K_07_12,K_07_13,K_07_14,K_07_15,K_07_16,K_07_17,K_07_18],
                            [K_08_10,K_08_11,K_08_12,K_08_13,K_08_14,K_08_15,K_08_16,K_08_17,K_08_18],
                            [K_09_10,K_09_11,K_09_12,K_09_13,K_09_14,K_09_15,K_09_16,K_09_17,K_09_18]])
    K_Matrix_10 = np.array([[K_10_01,K_10_02,K_10_03,K_10_04,K_10_05,K_10_06,K_10_07,K_10_08,K_10_09],
                            [K_11_01,K_11_02,K_11_03,K_11_04,K_11_05,K_11_06,K_11_07,K_11_08,K_11_09],
                            [K_12_01,K_12_02,K_12_03,K_12_04,K_12_05,K_12_06,K_12_07,K_12_08,K_12_09],
                            [K_13_01,K_13_02,K_13_03,K_13_04,K_13_05,K_13_06,K_13_07,K_13_08,K_13_09],
                            [K_14_01,K_14_02,K_14_03,K_14_04,K_14_05,K_14_06,K_14_07,K_14_08,K_14_09],
                            [K_15_01,K_15_02,K_15_03,K_15_04,K_15_05,K_15_06,K_15_07,K_15_08,K_15_09],
                            [K_16_01,K_16_02,K_16_03,K_16_04,K_16_05,K_16_06,K_16_07,K_16_08,K_16_09],
                            [K_17_01,K_17_02,K_17_03,K_17_04,K_17_05,K_17_06,K_17_07,K_17_08,K_17_09],
                            [K_18_01,K_18_02,K_18_03,K_18_04,K_18_05,K_18_06,K_18_07,K_18_08,K_18_09]])
    K_Matrix_11 = np.array([[K_10_10,K_10_11,K_10_12,K_10_13,K_10_14,K_10_15,K_10_16,K_10_17,K_10_18],
                            [K_11_10,K_11_11,K_11_12,K_11_13,K_11_14,K_11_15,K_11_16,K_11_17,K_11_18],
                            [K_12_10,K_12_11,K_12_12,K_12_13,K_12_14,K_12_15,K_12_16,K_12_17,K_12_18],
                            [K_13_10,K_13_11,K_13_12,K_13_13,K_13_14,K_13_15,K_13_16,K_13_17,K_13_18],
                            [K_14_10,K_14_11,K_14_12,K_14_13,K_14_14,K_14_15,K_14_16,K_14_17,K_14_18],
                            [K_15_10,K_15_11,K_15_12,K_15_13,K_15_14,K_15_15,K_15_16,K_15_17,K_15_18],
                            [K_16_10,K_16_11,K_16_12,K_16_13,K_16_14,K_16_15,K_16_16,K_16_17,K_16_18],
                            [K_17_10,K_17_11,K_17_12,K_17_13,K_17_14,K_17_15,K_17_16,K_17_17,K_17_18],
                            [K_18_10,K_18_11,K_18_12,K_18_13,K_18_14,K_18_15,K_18_16,K_18_17,K_18_18]])
    K_Matrix.K00 += K_Matrix_00
    K_Matrix.K01 += K_Matrix_01
    K_Matrix.K10 += K_Matrix_10
    K_Matrix.K11 += K_Matrix_11
    return K_Matrix
#%%
def pointmass_matrix(exmass,point):
    # Function to define the mass matrix
    # exmass : extra mass information
    # point  : index of the node
    # -------------------------------------------------------------------------
    # M_Matrix : mass matrix
    M_Matrix = np.zeros((9,9))
    # Definition of the submatrices
    M_Matrix[0,0] = exmass.point.mass[point]
    M_Matrix[1,1] = exmass.point.mass[point]
    M_Matrix[2,2] = exmass.point.mass[point]
    M_Matrix[3,3] = exmass.point.Ixx[point]
    M_Matrix[4,4] = exmass.point.Iyy[point]
    M_Matrix[5,5] = exmass.point.Izz[point]
    M_Matrix[3,4] = exmass.point.Ixy[point]
    M_Matrix[4,3] = exmass.point.Ixy[point]
    M_Matrix[3,5] = exmass.point.Izx[point]
    M_Matrix[5,3] = exmass.point.Izx[point]
    M_Matrix[4,5] = exmass.point.Iyz[point]
    M_Matrix[5,4] = exmass.point.Iyz[point]
    return M_Matrix