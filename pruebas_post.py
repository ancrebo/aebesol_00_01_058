# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 09:16:18 2022

@author: andre
"""

import matplotlib.pyplot as plt
from bin.postprocess import postproc_stat, postproc_dyn, postproc_vib, postproc_aelmod
plt.close('all')
# postproc_vib(case_setup,mesh_data,solution,section_globalCS)
postproc_aelmod(case_setup,mesh_data,solution,section_globalCS)
#postproc_dyn(case_setup,mesh_data,solution, section_globalCS) 
#postproc_aero(case_setup,mesh_data,solution, section_globalCS)