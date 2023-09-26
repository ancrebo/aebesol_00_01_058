# -*- coding: utf-8 -*-
"""
Created on Fri May 22 11:59:35 2020

@author        : Andres Cremades Botella - ancrebo@mot.upv.es
read_setup     : file containing functions for reading the configuration file
last_version   : 22-02-2021
modified_by    : Andres Cremades Botella
"""
import pandas as pd
import numpy as np

#%% Functions
def read_case(filecase):
    # Function for reading the configuration file
    # filecase : name of the configuration file
    # -------------------------------------------------------------------------
    # word_ant   : previous word
    # case_setup : class to store information of the setup
    #     - .problem_type      : type of problem 
    #     - .init_it           : initial iterations
    #     - .tot_time          : total simulation time
    #     - .time_stp          : time step
    #     - .solver_init       : initial iterations solver
    #     - .solver            : solver
    #     - .modsol            : modal solver
    #     - .n_mod             : number of modes
    #     - .adatol            : tolerance of the adaptive solver
    #     - .filt              : filter of the signal
    #     - .filt_freq         : filter frequency
    #     - .filt_ttran        : transition time of the filter
    #     - .filt_db           : damping of the filter in dB
    #     - .filt_wid          : filter width
    #     - .ini_con           : initial conditions
    #     - .savefile          : saving options
    #     - .savefile_name     : name to save the file
    #     - .savefile_step     : steps to save the solution
    #     - .savefile_modes    : modes to save
    #     - .print_val         : frequency to print the values
    #     - .meshfile          : mesh file
    #     - .refmesh           : number of refinements
    #     - .stress_model      : stress model
    #     - .aero_model        : aerodynamic model
    #     - .damping_type      : damping type
    #     - .damping_factor    : damping factor type
    #     - .damping_constant  : damping ratio
    #     - .damping_mass      : damping mass to contemplate in the damping
    #     - .ini_con_data_file : file of initial conditions
    word_ant = []
    class case_setup:
        problem_type = "STRUC_STAT"
        init_it          = 3
        tot_time         = 1
        time_ste         = 1e-3
        solver_init      = "ADAEX_RK4_EXP"
        solver           = "ADAEX_RK4_EXP"
        modsol           = "NO"
        n_mod            = 10
        adatol           = 1e-3
        filt             = "NO"
        filt_freq        = 50
        filt_ttran       = 1
        filt_db          = 100
        filt_wid         = 10
        ini_con          = 0
        savefile         = "NO"
        savefile_name    = "results"
        savefile_step    = 1
        savefile_modes   = 10
        print_val        = 1
        meshfile         = "mesh.mesh"
        refmesh          = 0
        stress_model     = "FLAT_STRESS"
        damping_type     = "NONE"
        damping_factor   = "AERO"
        damping_constant = 0.01
        damping_mass     = 0.9999
        grav_fl          = 0
        grav_g           = np.array([0,0,-9.81])
        aelmodomega      = "NO"
        pass
    # read text
    with open(filecase,"r") as textmesh:
        # flgmarker   : flag to indicate lecture of markers
        # flgplar     : flag to indicate lecture of the polar
        # flgxyplot   : flag to indicate lecture of xy plots
        # flginicon   : flag to indicate lecture of initial conditions
        # flgxlabel   : flag to indicate x label of a plot
        # flgylabel   : flag to indicate y label of a plot
        # flgylabel2  : flag to indicate second y label of a plot
        # flgzlabel   : flag to indicate z label of a plot
        # flgtitle    : flag to indicate title of a plot
        # flglabel    : flag to indicate labels
        # flglabel2   : flag to indicate second label
        # flgdefshape : flag to indicate deformed shape
        # flgtranbc   : flag that indicates the transient function
        # flg_joint   : flag that indicates a joint
        flgmarker   = 0
        flgpolar    = 0
        flgxyplot   = 0
        flginicon   = 0
        flgxlabel   = 0
        flgylabel   = 0
        flgylabel2  = 0
        flgzlabel   = 0
        flgtitle    = 0
        flglabel    = 0
        flglabel2   = 0
        flgdefshape = 0
        flgtranbc   = 0
        flg_joint   = 0
        flgflyctrl  = 0
        # boundary_all    : contains all the boundary information
        # plot_all        : contains all the plots information
        # bcvalues        : values of the boundary conditions
        # polarvalues_all : all the values of the polars
        # bcvalues_load   : values of the load at the boundary conditions
        # xvalues         : values of x axis
        boundary_all    = []
        plot_all        = []
        bcvalues        = []
        polarvalues_all = []
        bcvalues_load   = []
        xvalues         = []
        flyctrl_all     = []
        # boundary    : class to store the information relative to the boundary conditions
        # polarvalues : class to store the information of the polar
        # uniplot     : class to store the values of the plots
        class boundary:
            f_flap = 0
        class uniplot:
            pass
        class flyctrl:
            pass
        # Take every line of the text
        for line in textmesh:
            # flg_bcvalues      : boundary condition values flag
            # flg_bcvalues_load : boundary condition load values flag
            # flg_xvalues       : flag of the values of x axis
            # flg_joint_axis    : flag of the joint axis lecture
            flg_bcvalues      = 0
            flg_bcvalues_load = 0
            flgxvalues        = 0
            flg_jointaxis     = 0
            # for every word in the line
            for word in line.split():  
                # If commentary break line
                if word == '%':
                    break 
                # Problem type:
                #  - STRUC_STAT : structural stationary
                #  - STRUC_DYN  : structural dynamic
                #  - VIB_MOD    : modal vibration
                #  - AEL_DYN    : aeroelastic dynamic
                if word_ant== "PROB_TYPE=":
                    case_setup.problem_type = word
                if word_ant== "GRAV=":
                    case_setup.grav_fl = 1
                    case_setup.grav_g  = np.array(word.split(','),dtype='float')
                if word_ant == "VINF=":
                    case_setup.vinf = np.array(word.split(','),dtype='float')
                # Free stream velocity
                if word_ant == "AELMOD_OMEGA=":
                    case_setup.aelmodomega = word
                if word_ant == "AELMOD_VMAX=":
                    case_setup.vmax = float(word)
                # Free stream velocity
                if word_ant == "AELMOD_NUMV=":
                    case_setup.numv = int(word)
                # Free stream velocity
                if word_ant == "AELMOD_VMIN=":
                    case_setup.vmin = float(word)
                # Number of initial iterations
                if word_ant== "INIT_IT=":
                    case_setup.init_it = int(word)
                # Total time of simulation
                if word_ant== "TOT_TIME=":
                    case_setup.tot_time = float(word)
                # Time step
                if word_ant== "TIME_STP=":
                    case_setup.time_stp = float(word)
                # Initial solver definition (single step solvers)
                #    - RK4_EXP       : explicit runge-kutta order 4
                #    - ADA_RK4_EXP   : adaptive explicit runge-kutta order 4
                #    - ADAEX_RK4_EXP : adaptive explicit runge-kutta order 4 with force extrapolation
                #    - RK4_PC        : predictor-corrector runge-kutta order 4
                #    - ADA_RK4_PC    : adaptive predictor-corrector runge-kutta order 4
                # if a multistep solver is selected is changed by an adaptive explicit 4th order runge-kutta
                if word_ant== "INIT_SOLVER=":
                    case_setup.solver_init = word
                # Solver definition after initialization
                #    - RK4_EXP       : explicit runge-kutta order 4
                #    - ADA_RK4_EXP   : adaptive explicit runge-kutta order 4
                #    - ADAEX_RK4_EXP : adaptive explicit runge-kutta order 4 with force extrapolation
                #    - RK4_PC        : predictor-corrector runge-kutta order 4
                #    - ADA_RK4_PC    : adaptive predictor-corrector runge-kutta order 4
                #    - BACK4_IMP     : backward implicit order 4
                #    - AM4_IMP       : implicit adams-moulton order 4
                #    - AM3_PC        : predictor-corrector adams-moulton order 3
                #    - HAM3_PC       : predictor-corrector hamming order 3
                if word_ant== "SOLVER=":
                    case_setup.solver = word
                # Select if the modal truncation is activated YES/NO
                if word_ant== "MOD_SOL=":
                    case_setup.mod_sol = word
                # Number of modes in the modal truncation
                if word_ant== "N_MOD=":
                    case_setup.n_mod = int(word)
                # Adaptive solver tolerance
                if word_ant== "ADA_TOL=":
                    case_setup.adatol = float(word)
                # Activate filter YES/NO
                if word_ant== "FILT=":
                    case_setup.filt = word
                # Filter frequency
                if word_ant== "FILT_FREQ=":
                    case_setup.filt_freq = float(word)
                # Transition time after the activation of the filter
                if word_ant== "FILT_TTRAN=":
                    case_setup.filt_ttran = float(word)
                # Damping of the filter dBs
                if word_ant== "FILT_DB=":
                    case_setup.filt_db = float(word)
                # Width of the filter
                if word_ant== "FILT_WID=":
                    case_setup.filt_wid = float(word)
                # Activate initial conditions YES/NO
                if word_ant== "INI_CON=":
                    if word == "YES":
                        case_setup.ini_con = 1
                        flginicon          = 1
                    else:
                        case_setup.ini_con = 0
                # Activate saving options YES/NO
                if word_ant == "SAVE_FILE=":
                    case_setup.savefile = word
                # name of the results folder
                if word_ant == "SAVE_FILE_NAME=":
                    case_setup.savefile_name = word
                # time steps to save the file
                if word_ant == "SAVE_FILE_STEP=":
                    case_setup.savefile_step = int(word)
                # modes to save
                if word_ant == "SAVE_FILE_MODES=":
                    case_setup.savefile_modes = int(word)
                # frequency to print values
                if word_ant == "PRINT_VALUES=":
                    case_setup.print_val = int(word)
                # mesh file to read
                if word_ant== "MESH_FILE=":
                    case_setup.meshfile = word
                # number of refinements of the mesh
                if word_ant== "REF_MESH=":
                    case_setup.refmesh = int(word) 
                # read marker activate lecture flag
                if word== "MARK_ID{":
                    flgmarker = 1 
                if word== "FLY_CTRL{":
                    flgflyctrl = 1 
                # read xy plot activate lecture flag
                if word== "XYPLOT{": 
                    flgxyplot        = 1 
                    uniplot.typeplot = "XYPLOT"
                    uniplot.xlabel   = []
                    uniplot.ylabel   = []
                    uniplot.title    = []
                    uniplot.label    = []
                    uniplot.anim     = []
                # read deformed shape activate lecture flag
                if word== "DEF_SHAPE{":
                    flgdefshape      = 1
                    uniplot.typeplot = "DEF_SHAPE"
                # Read transient function for a boundary condition
                if word== "TRAN_BC{":
                    flgtranbc = 1
                # read aerodynamic polar and activate flag
                if word== "AERO_POLAR{":
                    flgpolar = 1
                # select the stress model FLAT_STRESS / BEAM_STRESS (recomended flat stress)
                if word_ant== "STRESS_MODEL=":
                    case_setup.stress_model = word
                # select the aerodynamic model TEOR_SLOPE/POLAR/POLAR_ANN
                if word_ant== "AERO_MODEL=":
                    case_setup.aero_model = word
                    class polarvalues:
                        pointfile = [-1]
                        file_node = []
                        created = 0
                # Activate damping RAYLEIGH/NONE
                if word_ant== "DAMP=":
                    case_setup.damping_type = word
                # Damping factor type AERO/ACCEL
                if word_ant== "DAMP_FACT=":
                    case_setup.damping_factor = word
                # Damping ratio
                if word_ant== "DAMP_RAY_VAL1=":
                    case_setup.damping_constant1 = float(word)
                if word_ant== "DAMP_RAY_VAL2=":
                    case_setup.damping_constant2 = float(word)
                if word_ant== "DAMP_RAY_F1=":
                    case_setup.Ray_f1 = float(word)
                if word_ant== "DAMP_RAY_F2=":
                    case_setup.Ray_f2 = float(word)
                # Damping mass to contemplate
                if word_ant== "DAMP_MASS=":
                    case_setup.damping_mass = float(word)
                # End of option
                if word == "}":
                    # End of marker
                    if flgmarker == 1:
                        flgmarker            = 0
                        flg_bcvalues         = 0 
                        boundary.values      = bcvalues 
                        boundary.values_load = bcvalues_load 
                        boundary.func        = "NONE"
                        boundary_all.append(boundary)
                        bcvalues             = []
                        class boundary:
                            f_flap = 0
                    # End of polar
                    if flgpolar == 1:
                        flgpolar = 0
                        polarvalues_all.append(polarvalues)
                        class polarvalues:
                            pass
                    # End transient function
                    if flgtranbc == 1:
                        flgtranbc = 0
                    # End xy plot
                    if flgxyplot == 1:
                        flgxyplot       = 0
                        flgxvalues      = 0
                        flgxlabel       = 0
                        flgylabel       = 0
                        flgylabel2      = 0
                        flglabel        = 0
                        uniplot.xvalues = xvalues
                        xvalues         = []
                        plot_all.append(uniplot)
                        class uniplot:
                            pass
                    # End deformed shape
                    if flgdefshape == 1:
                        flgdefshape = 0
                        plot_all.append(uniplot)
                        class uniplot:
                            pass
                    if flgflyctrl == 1:
                        flgflyctrl = 0
                        flyctrl_all.append(flyctrl)
                        class flyctrl:
                            pass
                # If iniconditions are activated
                if flginicon == 1:
                    if word_ant == "INI_CON_DATA=":
                        case_setup.ini_con_data_file = word
                if flgflyctrl == 1:
                    if word_ant == "CTRL_ID=":
                        flyctrl.id = word
                    elif word_ant == "CTRL_MARK=":
                        flyctrl.mark = word.split(',')
                    elif word_ant == "CTRL_SGN=":
                        ctrlsign = word.split(',')
                        flyctrl.sign = np.zeros((len(ctrlsign),),dtype='int')
                        for iictrl in np.arange(len(ctrlsign)):
                            flyctrl.sign[iictrl] = int(ctrlsign[iictrl])
                    elif word_ant == "CTRL_OBJ=":
                        flyctrl.obj = word
                    elif word_ant == "CTRL_OBJ_VALUE=":
                        flyctrl.obj_value = float(word)
                # If reading a marker
                if flgmarker == 1:
                    # boundary : class of the boundary conditions
                    #    - .marker : name of the marker
                    #    - .type   : type of condition
                    #    - .polar  : polar of aerodynamic load
                    #    - .vinf   : free stream velocity
                    #    - .refpoint : reference point
                    #    - .refaxis  : reference aoa axis
                    #    - .refaxis2 : lift axis
                    #    - .vdir     : velocity axis
                    #    - .aoa      : angle of attack
                    #    - .rho      : free stream density
                    #    - .l_ref    : reference length
                    #    - .s_ref    : reference surface
                    #    - .func_mode : modal function
                    #    - .funcnorm  : point to normalize the function 
                    #    - .joint_type : joint type of the boundary
                    #    - .joint_axis : axis of the joint
                    #    - .point_axis : reference point of the joint 
                    # Identifier of the boundary condition
                    if word_ant == "BC_ID=":
                        boundary.id = word
                    # If displacement condition
                    if word_ant == "BC_DISP=":
                        boundary.marker = word
                        boundary.type = "BC_DISP"
                    # If boundary condition is a function of the modes
                    if word_ant == "BC_FUNC=":
                        boundary.marker = word
                        boundary.type = "BC_FUNC"
                    # If boundary condition is a load
                    if word_ant == "BC_NODELOAD=":
                        boundary.marker = word
                        boundary.type = "BC_NODELOAD"
                    # If boundary condition is aerodynamic load
                    if word_ant == "BC_AERO=":
                        boundary.marker = word
                        boundary.type = "BC_AERO"
                    if word_ant == "BC_FLAP=":
                        boundary.f_flap = 1
                        boundary.flap = word
                    # Polar of the aerodynamic load
                    if word_ant == "BC_POLAR=":
                        boundary.polar = word
                    if word_ant == "BC_FLAP_POLAR=":
                        boundary.flappolar = word
                    # Free stream velocity
                    if word_ant == "BC_VINF=":
                        if word == "VINF_DAT":
                            boundary.vinf = word #np.linalg.norm(case_setup.vinf)
                        else:
                            boundary.vinf = float(word)
                    # Reference point
                    if word_ant == "BC_REFPOINT=":
                        boundary.refpoint = int(word)
                    if word_ant == "BC_FLAP_REFPOINT=":
                        boundary.flaprefpoint = int(word)
                    # Reference axis of the moment coefficient
                    if word_ant == "BC_REFAXIS=":
                        # refax : reference axis list
                        refax = word.split(',')
                        boundary.refaxis = np.zeros((3,))
                        for iiax in [0,1,2]:
                            boundary.refaxis[iiax]= float(refax[iiax])
                    # Reference CL axis
                    if word_ant == "BC_REFCL=":
                        # refax : reference axis list
                        refax = word.split(',')
                        boundary.refCL = np.zeros((3,))
                        for iiax in [0,1,2]:
                            boundary.refCL[iiax]= float(refax[iiax])
                    # Reference CD axis
                    if word_ant == "BC_REFCD=":
                        # refax : reference axis list
                        refax = word.split(',')
                        boundary.refCD = np.zeros((3,))
                        for iiax in [0,1,2]:
                            boundary.refCD[iiax]= float(refax[iiax])
                    # Reference CM axis
                    if word_ant == "BC_REFCM=":
                        # refax : reference axis list
                        refax = word.split(',')
                        boundary.refCM = np.zeros((3,))
                        for iiax in [0,1,2]:
                            boundary.refCM[iiax]= float(refax[iiax])
                    # Reference rotation axis
                    if word_ant == "BC_REFROT=":
                        # refax : reference axis list
                        refax = word.split(',')
                        boundary.refrot = np.zeros((3,))
                        for iiax in [0,1,2]:
                            boundary.refrot[iiax]= float(refax[iiax])
                    # Reference axis of the lift coefficient
                    if word_ant == "BC_REFAXIS2=":
                        # refax : reference axis list
                        refax = word.split(',')
                        boundary.refaxis2 = np.zeros((3,))
                        for iiax in [0,1,2]:
                            boundary.refaxis2[iiax]= float(refax[iiax])
                    # Free stream velocity direction
                    if word_ant == "BC_VDIR=":
                        if word == "VINF_DAT":
                            boundary.vdir = word #case_setup.vinf/np.linalg.norm(case_setup.vinf)
                        else:
                            # refax : reference axis list
                            refax = word.split(',')
                            boundary.vdir = np.zeros((3,))
                            for iiax in [0,1,2]:
                                boundary.vdir[iiax]= float(refax[iiax])
                    # Angle of attack of the free stream
                    if word_ant == "BC_AOA=":
                        boundary.aoa = float(word)
                    # Density of the free stream
                    if word_ant == "BC_RHO=":
                        boundary.rho = float(word) 
                    # Density of the free stream
                    if word_ant == "BC_MU=":
                        boundary.mu = float(word) 
                    # Reference length
                    if word_ant == "L_REF=":
                        boundary.l_ref = float(word)
                    # Reference surface
                    if word_ant == "S_REF=":
                        boundary.s_ref = float(word)  
                    # Blade radius
                    if word_ant == "BC_RADIUS=":
                        boundary.radius = float(word)
                    # Rotation velocity
                    if word_ant == "BC_VROT=":
                        boundary.vrot = float(word)
                    # number of blades
                    if word_ant == "BC_NB=":
                        boundary.Nb = float(word)
                    # Boundary condition joint
                    if word_ant == "BC_JOINT=":
                        boundary.marker = word
                        boundary.type   = "BC_JOINT"
                        flg_joint       = 1
                    # Modal function - number of the mode
                    if word_ant == "FUNC_MODE=":
                        boundary.funcmode = word
                    # Marker to normalize the modal function
                    if word_ant == "FUNC_NORM=":
                        boundary.funcnorm = word
                    # If it is a joint
                    if flg_joint == 1:
                        # Select the joint type
                        if word_ant == "JOINT_TYPE=":
                            boundary.joint_type= word
                            # Rotation joint
                            if boundary.joint_type == "ROTATE_AXIS":
                                boundary.joint_axis = []
                                boundary.point_axis = []
                            # Fixed joint
                            if boundary.joint_type == "FIXED":
                                bcvalues = np.zeros((9,))
                        try:
                            # Rotation axis
                            if boundary.joint_type == "ROTATE_AXIS":
                                # Axis
                                if word_ant == "AXIS=" or flg_jointaxis == 1:
                                    flg_jointaxis = 1
                                    boundary.joint_axis.append(float(word))
                                # Point of the rotation
                                if word_ant == "POINT=" or flg_pointaxis == 1:
                                    if  word_ant == "POINT=":
                                        boundary.point_axis = []
                                    flg_pointaxis = 1
                                    boundary.point_axis.append(float(word))
                        except:
                            pass
                    # Set the flag to store the values
                    if word_ant == "BC_VALUES=":
                        flg_bcvalues = 1
                    # set the flag to store load values
                    if word_ant == "BC_VALUES_LOAD=":
                        flg_bcvalues_load = 1
                    # if the flag of values is activated
                    if flg_bcvalues == 1:
                        bcvalues.append(float(word))
                    # if the flag of load values is activated
                    if flg_bcvalues_load == 1:
                        bcvalues_load.append(float(word))
                # If the flag to read the polar is activated
                if flgpolar == 1:
                    # polarvalues : data of the polar
                    #    - id         : identifier of the polar
                    #    - file       : file of the polar
                    #    - indep      : independent variables
                    #    - dep        : dependent variables
                    #    - annfile    : artificial neural network file
                    #    - anntype    : type of artificial neural network
                    #    - annindep   : independent variables of the ann
                    #    - anndep     : dependent variables of the ann
                    #    - annnorm    : ann normalization file
                    #    - eff3d      : 3D effects
                    #    - vind_tol   : tolerance of the induced velocity calculation
                    #    - vind_maxit : maximum iterations of the vind
                    # read the identifier of the polar
                    if word_ant == "POLAR_ID=":
                        polarvalues.id = word
                    # read the file of the polar
                    if word_ant == "POLAR_REFPOINT=":
                        pointstr  = word.split(';')
                        polarvalues.pointfile = np.zeros((len(pointstr),))
                        for pstrii in np.arange(len(pointstr)):
                            polarvalues.pointfile[pstrii] = pointstr[pstrii]
                    if word_ant == "POLAR_FILE=":
                        polarvalues.file = word.split(';')
                    if word_ant == "Q_STD=":
                        polarvalues.quasi_steady = int(word)
                    if word_ant == "LLT_NODES=":
                        polarvalues.lltnodesflag = 1
                        polarvalues.lltnodesmark =  word
                    if word_ant == "BEM_COR3D=": # mod_sumar
                        polarvalues.cor3dflag = 1 # mod_sumar
                        polarvalues.cor3dmark =  word # mod_sumar
                    # read the independent variables of the polar
                    if word_ant == "POLAR_INDEP=": 
                        polarvalues.indep = word.split(',')
                    # read the dependent variables of the polar
                    if word_ant == "POLAR_DEP=":
                        polarvalues.dep = word.split(',')
                    # read the artificial neural network file of the polar
                    if word_ant == "ANN_FILE=":
                        polarvalues.annfile = word
                    # read the artificial neural network type of the polar
                    if word_ant == "ANN_TYPE=":
                        polarvalues.anntype = word
                    # read the independent variables of the artificial neural network
                    if word_ant == "ANN_INDEP=": 
                        polarvalues.annindep = word.split(',')
                    # read the dependent variables of the artificial neural network
                    if word_ant == "ANN_DEP=": 
                        polarvalues.anndep = word.split(',')
                    # read the artificial neural network norm
                    if word_ant == "ANN_NORM=":
                        polarvalues.annnorm = word
                    # read the 3D effects
                    if word_ant == "3D_EFF=":
                        polarvalues.eff3d = word
                        if word == "BEM":
                            polarvalues.startbem = 1
                        polarvalues.lltnodesflag = 0
                        polarvalues.cor3dflag = 0 # mod_sumar
                    # read the induced velocity tolerance
                    if word_ant == "VIND_TOL=":
                        polarvalues.vind_tol = float(word)
                    # read the induced velocity maximum iterations
                    if word_ant == "VIND_MAXIT=":
                        polarvalues.vind_maxit = float(word)
                # uniplot : class of the plot information
                #    - .type       : type of plot
                #    - .save       : saving name
                #    - .save_data  : data saving file name
                #    - .marker     : marker of the plot
                #    - .marker2    : second marker of the plot
                #    - .mode       : mode plotted if modal results
                #    - .anim       : animation activated
                #    - .yaxis      : y axis parameter
                #    - .yaxis2     : second y axis parameter
                #    - .deltaframe : delta time between frames
                #    - .fps        : frames per second in animation
                #    - .fig        : figure number
                #    - .xlabel     : x axis label
                #    - .ylabel     : y axis label
                #    - .zlabel     : z axis label
                #    - .ylabel2    : second y axis label
                #    - .title      : title of the plot
                #    - .label      : curve label
                #    - .label2     : second curve label
                #    - .dihedral   : dihedral representation
                # if the xy plot flag is activated
                if flgxyplot == 1:
                    # read plot type
                    if word_ant == "TYPE=":
                        uniplot.type = word
                    # read image save name
                    if word_ant == "SAVE=":
                        uniplot.save = word
                    # read data save name
                    if word_ant == "SAVE_DATA=":
                        uniplot.save_data = word
                    # read marker name
                    if word_ant == "MARK=":
                        uniplot.marker = word
                    # read second marker name
                    if word_ant == "MARK2=":
                        uniplot.marker2 = word
                    # read modes to plot if modal analysis
                    if word_ant == "MODE=":
                        uniplot.mode = int(word)
                    # set if animation
                    if word_ant == "ANIM=":
                        uniplot.anim = word
                    # if readin xaxis in space monitor
                    if word_ant == "XAXIS=": # and uniplot.type == "SPACE":
                        flgxvalues = 1
                    if flgxvalues == 1:
                        try:
                            xvalues.append(float(word))
                        except:
                            uniplot.xaxis = word
                    # if reading y axis variable
                    if word_ant == "YAXIS=":
                        uniplot.yaxis = word
                    # if reading y axis variable
                    if word_ant == "ZAXIS=":
                        uniplot.zaxis = word
                    # if reading second y axis variable
                    if word_ant == "YAXIS2=":
                        uniplot.yaxis2 = word
                    # if reading time between photograms
                    if word_ant == "DELTA_FRAME=":
                        uniplot.deltaframe = int(word)
                    # if reading fps of video
                    if word_ant == "FPS=":
                        uniplot.fps = int(word)
                    # if reading the figure
                    if word_ant == "FIGURE=":
                        uniplot.fig = int(word)
                    # if reading the colormpat
                    if word_ant == "COLOR=":
                        uniplot.color = word
                    # if reading x axis label
                    if word_ant == "XLABEL=":
                        flgxlabel = 1
                    if flgxlabel == 1:
                        if word[0] == '"' and word[-1] == '"':
                            uniplot.xlabel = word[1:-1]
                            flgxlabel      = 0
                        elif word[0] == '"':
                            uniplot.xlabel = word[1:]
                        elif word[-1] == '"':
                            uniplot.xlabel = uniplot.xlabel +' '+ word[:-1]
                            flgxlabel      = 0
                        else: 
                            uniplot.xlabel = uniplot.xlabel +' '+ word
                    # if reading y axis label        
                    if word_ant == "YLABEL=":
                        flgylabel = 1
                    if flgylabel == 1:
                        if word[0] == '"' and word[-1] == '"':
                            uniplot.ylabel = word[1:-1]
                            flgylabel      = 0
                        elif word[0] == '"':
                            uniplot.ylabel =  word[1:]
                        elif word[-1] == '"':
                            uniplot.ylabel = uniplot.ylabel +' '+ word[:-1]
                            flgylabel      = 0
                        else: 
                            uniplot.ylabel = uniplot.ylabel +' '+ word
                    # if reading the second y axis label       
                    if word_ant == "YLABEL2=":
                        flgylabel2 = 1
                    if flgylabel2 == 1:
                        if word[0] == '"' and word[-1] == '"':
                            uniplot.ylabel2 = word[1:-1]
                            flgylabel2      = 0
                        elif word[0] == '"':
                            uniplot.ylabel2 = word[1:]
                        elif word[-1] == '"':
                            uniplot.ylabel2 = uniplot.ylabel2 +' '+ word[:-1]
                            flgylabel2      = 0
                        else: 
                            uniplot.ylabel2 = uniplot.ylabel2 +' '+ word
                    # title of the plot        
                    if word_ant == "TITLE=":
                        flgtitle = 1
                    if flgtitle == 1:
                        if word[0] == '"' and word[-1] == '"':
                            uniplot.title = word[1:-1]
                            flgtitle      = 0
                        elif word[0] == '"':
                            uniplot.title =  word[1:]
                        elif word[-1] == '"':
                            uniplot.title = uniplot.title +' '+ word[:-1]
                            flgtitle      = 0
                        else: 
                            uniplot.title = uniplot.title +' '+ word
                    # if reading curve label
                    if word_ant == "LABEL=":
                        flglabel = 1
                    if flglabel == 1:
                        if word == "X_POS" or word == "Y_POS" or word == "Z_POS" or word == "TIME":
                            uniplot.label = word
                            flglabel      = 0
                        elif word[0] == '"' and word[-1] == '"' and len(word)>1:
                            uniplot.label = word[1:-1]
                            flglabel      = 0
                        elif word[0] == '"' and len(word)>1:
                            uniplot.label =  word[1:]
                        elif word[-1] == '"':
                            uniplot.label = uniplot.label +' '+ word[:-1]
                            flglabel      = 0
                        else: 
                            uniplot.label = uniplot.label +' '+ word
                    # if reading curve second label
                    if word_ant == "LABEL2=":
                        flglabel2 = 1
                    if flglabel2 == 1:
                        if word == "X_POS" or word == "Y_POS" or word == "Z_POS" or word == "TIME":
                            uniplot.label = word
                            flglabel2     = 0
                        elif word[0] == '"' and word[-1] == '"':
                            uniplot.label2 = word[1:-1]
                            flglabel2      = 0
                        elif word[0] == '"':
                            uniplot.label2 = word[1:]
                        elif word[-1] == '"':
                            uniplot.label2 = uniplot.label2 +' '+ word[:-1]
                            flglabel2      = 0
                        else: 
                            uniplot.label2 = uniplot.label2 +' '+ word
                # if deformation shape is activated
                if flgdefshape == 1:
                    # read marker name
                    if word_ant == "MARK=":
                        uniplot.marker = word.split(',')
                    # read face construction
                    if word_ant == "FACE=":
                        uniplot.face = word
                    # read mode vibration analysis
                    if word_ant == "MODE=":
                        uniplot.mode = int(word)
                    # read if rotative
                    if word_ant == "ROT=":
                        uniplot.rot = word
                    # rotative conditions
                    if word_ant == "ROT_NAME=":
                        uniplot.rot_name = word
                    # read saving name
                    if word_ant == "SAVE=":
                        uniplot.save = word
                    # read saving data name
                    if word_ant == "SAVE_DATA=":
                        uniplot.save_data = word
                    # time between frames
                    if word_ant == "DELTA_FRAME=":
                        uniplot.deltaframe = float(word)
                    # frames per second
                    if word_ant == "FPS=":
                        uniplot.fps = int(word)
                    # x axis label
                    if word_ant == "XLABEL=":
                        flgxlabel = 1
                    if flgxlabel == 1:
                        if word[0] == '"' and word[-1] == '"':
                            uniplot.xlabel = word[1:-1]
                            flgxlabel      = 0
                        elif word[0] == '"':
                            uniplot.xlabel = word[1:]
                        elif word[-1] == '"':
                            uniplot.xlabel = uniplot.xlabel +' '+ word[:-1]
                            flgxlabel      = 0
                        else: 
                            uniplot.xlabel = uniplot.xlabel +' '+ word
                     # y axis label       
                    if word_ant == "YLABEL=":
                        flgylabel = 1
                    if flgylabel == 1:
                        if word[0] == '"' and word[-1] == '"':
                            uniplot.ylabel = word[1:-1]
                            flgylabel      = 0
                        elif word[0] == '"':
                            uniplot.ylabel = word[1:]
                        elif word[-1] == '"':
                            uniplot.ylabel = uniplot.ylabel +' '+ word[:-1]
                            flgylabel      = 0
                        else: 
                            uniplot.ylabel = uniplot.ylabel +' '+ word  
                    # z axis label
                    if word_ant == "ZLABEL=":
                        flgzlabel = 1
                    if flgzlabel == 1:
                        if word[0] == '"' and word[-1] == '"':
                            uniplot.zlabel = word[1:-1]
                            flgzlabel      = 0
                        elif word[0] == '"':
                            uniplot.zlabel = word[1:]
                        elif word[-1] == '"':
                            uniplot.zlabel = uniplot.zlabel +' '+ word[:-1]
                            flgzlabel      = 0
                        else: 
                            uniplot.zlabel = uniplot.zlabel +' '+ word
                    # title of the deformed shape plot                
                    if word_ant == "TITLE=":
                        flgtitle = 1
                    if flgtitle == 1:
                        if word[0] == '"' and word[-1] == '"':
                            uniplot.title = word[1:-1]
                            flgtitle      = 0
                        elif word[0] == '"':
                            uniplot.title = word[1:]
                        elif word[-1] == '"':
                            uniplot.title = uniplot.title +' '+ word[:-1]
                            flgtitle      = 0
                        else: 
                            uniplot.title = uniplot.title +' '+ word
                    # figure of the plot        
                    if word_ant == "FIGURE=":
                        uniplot.fig = int(word)
                    # dihedral of the plot
                    if word_ant == "DIHEDRAL=":
                        uniplot.dihedral = word
                    # color bar maximum    
                    if word_ant == "COLMAX=":
                        uniplot.colmax = float(word)
                    # color bar minimum
                    if word_ant == "COLMIN=":
                        uniplot.colmin = float(word)
                    # if reading the colormpat
                    if word_ant == "COLOR=":
                        uniplot.color = word
                # if the transient function is activated
                if flgtranbc == 1:
                    # read the marker of the fuction and get its index
                    if word_ant == "TRAN_MARK=":
                        for ii_tranbc in np.arange(len(boundary_all)):
                            if word == boundary_all[ii_tranbc].id: # marker:
                                # index_tranbc : index of the markers in which the function is applied
                                index_tranbc = ii_tranbc
                                break
                    # read the function
                    if word_ant == "FUNCTION=":
                        boundary_all[index_tranbc].func = word
                    # read the amplitude of the sinus
                    if word_ant == "AMPL=":
                        boundary_all[index_tranbc].ampl = float(word)
                    # read the frequency of the sinus
                    if word_ant == "FREQ=":
                        boundary_all[index_tranbc].freq = float(word)
                    # read the phase of the sinus
                    if word_ant == "PHAS=":
                        boundary_all[index_tranbc].phas = float(word)
                    # read the slope of the sigmoid
                    if word_ant == "SLP=":
                        boundary_all[index_tranbc].slp = float(word)
                    # read the midpoint of the sigmoid
                    if word_ant == "MID_POIN=":
                        boundary_all[index_tranbc].m_poin = float(word)
                    # file of the function
                    if word_ant == "FILE=":
                        boundary_all[index_tranbc].file_tran = word
                    # define the file as periodic YES/NO
                    if word_ant == "PER=":
                        boundary_all[index_tranbc].periodic = word
                # update last word                                
                word_ant = word 
        # Save boundaries, plots and polars in the setup
        case_setup.boundary = boundary_all 
        case_setup.plots = plot_all
        case_setup.polar = polarvalues_all
        case_setup.flyctrl = flyctrl_all
    return case_setup
    
#%% 
def read_inicon(ini_con_data):
    # Function to read initial conditions
    # ini_con_data : initial condition data file
    # ------------------------------------------------------------------------
    # data   : information of the initial condition
    # data_matrix : value of the information of the initial conditions
    # dataposmat  : value of the initial position
    # datavelmat  : value of the initial velocity
    # dataacelmat : value of the initial acceleration
    # dataposvec  : vector of initial position
    # datavelvec  : vector of initial velocity
    # dataacelvec : vector of initial acceleration
    data        = pd.read_csv(ini_con_data)
    datamatrix  = data.values
    dataposmat  = np.concatenate((data.values[:,:6],data.values[:,9:12]),axis=1)
    datavelmat  = np.concatenate((data.values[:,6:12],data.values[:,15:18]),axis=1)
    dataacelmat = np.concatenate((data.values[:,12:18],data.values[:,21:]),axis=1)
    dataposvec  = np.zeros((len(datamatrix)*9,))
    datavelvec  = np.zeros((len(datamatrix)*9,))
    dataacelvec = np.zeros((len(datamatrix)*9,))
    # for all the nodes add the value of position velocity and acceleration
    for ii in np.arange(len(datamatrix)):
        dataposvec[ii:ii+9]  = dataposmat[ii,:]
        datavelvec[ii:ii+9]  = datavelmat[ii,:]
        dataacelvec[ii:ii+9] = dataacelmat[ii,:]
    return dataposvec, datavelvec, dataacelvec