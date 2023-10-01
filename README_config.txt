    The configuration of the case problem is stored in the variable setup. 
    Information contained in the variable:
        * problem_type : type of problem solved in the solver
        (KEY : PROB_TYPE=)
            - AERO       : aerodynamic problem. This simulation does not take
                           into acount the structure for the analysis, the 
                           aerodynamic forces are provided as solution.
            - STRUC_STAT : structural stationary. The steady deformation of the
                           structure is calculated for a constant load. 
                           Aerodynamic loads can be added, but they are not 
                           updated with the deformation.
            - STRUC_DYN  : structural transient. The structure is evaluated for
                           a transient state of loads and the evolution of the
                           deformation is provided as output.
            - VIB_MOD    : structural modal analysis. The eigensystem of the
                           structure is calculated. The natural frequecy and 
                           the modal shapes are obtained.
            - AEL_STAT   : aeroelastic steady. The aerodynamic load is updated
                           with the displacements for a steady evolution. It is
                           similar to the STRUC_STAT with aerodynamic load, but
                           in this case the solution needs to converge the 
                           aerodynamic load which changes with the 
                           displacements
            - AEL_DYN    : aeroelastic transient. The transient aeroelastic 
                           simulations.
            - AEL_MOD    : aeroelastic modal. Eigensystem of the structure
                           including the aerodynamic loads. Used for 
                           calculating the aeroelastic eigenvalues.
        * init_it  : number of initial iterations
        (KEY : INIT_IT=)
        * tot_time : total time of the simulations
        (KEY : TOT_TIME=)
        * time_step : value of the time step of the simulation
        (KEY : TIME_STP=)
        * solver_init : initial solver of the problem
        (KEY : INIT_SOLVER=)
            - RK4_EXP    : 4th order explicit runge-kutta
            - RK4_PC     : 4th order predictor-corrector runge-kutta
            - ADA_RK4_PC : 4th order adatative time ste predictor-corrector 
                           runge-kutta
            - BACK4_IMP  : 4th order backward implicit
            - AM4_IMP    : 4th order implicit adams-moulton
            - AM3_PC     : 3rd order predictor-corrector adams-moulton
            - HAM3_PC    : 3rd order predictor-corrector hamming 
        * solver: solver of the problem
        (KEY : SOLVER=)
            - RK4_EXP    : 4th order explicit runge-kutta
            - RK4_PC     : 4th order predictor-corrector runge-kutta
            - ADA_RK4_PC : 4th order adatative time ste predictor-corrector 
                           runge-kutta
            - BACK4_IMP  : 4th order backward implicit
            - AM4_IMP    : 4th order implicit adams-moulton
            - AM3_PC     : 3rd order predictor-corrector adams-moulton
            - HAM3_PC    : 3rd order predictor-corrector hamming 
        * modsol : modal truncation of the transient simulation
        (KEY : MOD_SOL=)
            - YES/NO
        * n_mod : number of modes used for the transient simulation
        (KEY : N_MOD=)
        * adatol : adaptative solver tolerance
        (KEY : ADA_TOL=)
        * ini_con : initial conditions
        (KEY : INI_CON=)
            - 0 : no initial conditions are used
            - 1 : initial conditions are used
        * ini_con_data_file : initial conditions file
        (KEY : INI_CON_DATA=)
        * savefile: save the file
        (KEY : SAVE_FILE=)
            - NO/YES
        * savefile_name : name of the file to save the solution
        (KEY : SAVE_FILE_NAME=)
        * savefile_step : number of steps to save the information
        (KEY : SAVE_FILE_STEP=)
        * savefile_modes : number of modes to save in the file
        (KEY : SAVE_FILE_MODES=)
        * print_val : number of steps to print the values on the screen
        (KEY : PRINT_VALUES=)
        * meshfile: name of the mesh file
        (KEY : MESH_FILE=)
        * refmesh: number of refinements of the mesh
        (KEY : REF_MESH=)
        * stress_model: model to calculate the stress
        (KEY : STRESS_MODEL=)
            - FLAT_STRESS: flat stress model (recommended)
            - BEAM_STRESS: useful for some beam calculations
        * dampyng_type: model of damping
        (KEY : DAMP=)
            - NONE: no damp
            - RAYLEIGH: Rayleigh damping
        * damping_factor: 
        (KEY : DAMP_FACT=)
            - ACCEL: based on the effective mass
            - AERO: based on the aerodynamic forces
            - DIREC: on the specified frequencies (recommended)
        * damping_constant1 : damping ratio in frequency 1
        (KEY : DAMP_RAY_VAL1=)
        * damping_constant2 : damping ratio in frequency 2
        (KEY : DAMP_RAY_VAL2=)
        * Ray_f1 : damping frequency 1
        (KEY : DAMP_RAY_F1=)
        * Ray_f2 : damping frequency 2
        (KEY : DAMP_RAY_F2=)
        * damping_mass: mass to contemplate in the effective mass
        (KEY : DAMP_MASS=)
        * grav_fl : gravity acceleration flag (activates gravity)
          grav_g  : gravity acceleration vector
        (KEY : GRAV=)
        * aelmodomega: (check ini code) 
        (KEY : AELMOD_OMEGA=)
            - YES/NO 
        * vmax : maximum velocity for the aeroelastic modal analysis
        (KEY : AELMOD_VMAX=)
        * vnum : number of velocities for the aeroelastic modal analysis
        (KEY : AELMOD_NUMV=)
        * vmin : minimum velocity for the aeroelastic modal analysis
        (KEY : AELMOD_VMIN=)
        * vinf : global velocity of the free stream
        (KEY : VINF=)
        * aeromodel: aerodynamic model
        (KEY : AERO_MODEL=)
            - POLAR     : takes the aerodynamic coefficients of a 2D polar
            - POLAR_ANN : takes the steady aerodynamic coefficients of a 2D 
                          polar and the artificial neural network for the 
                          dynamic coefficients
            - THEO_POLAR : takes the steady aerodynamic coefficients of a 2D 
                           polar and the theodorsen coefficients for the 
                           dynamic terms
            - TEOR_SLOPE : steady aerodynamic coefficient from the 2pi slope
            - THEO_POTEN : steady aerodynamic coefficient form the 2pi slope 
                           and dynamic term from theodorsen coefficients
        * boundary: boundary conditions
        (KEY : MARK_ID{ \n...\n })
            * id : identifier of the boundary
            (KEY : BC_ID=)
            * type : type of boundary condition
                - BC_DISP : defines the boundary conditions as a displacement 
                            condition
                - BC_FUNC : if the boundary conditions are a function of the
                            vibration modes
                - BC_NODELOAD : defines the boundary conditions as a load
                - BC_AERO : defines the boundary conditions as an aerodynamic
                            load
                - BC_JOINT : defines a joint between 2 bodies
            * marker : name of the marker of nodes in which the condition is 
                       applied
                - (KEY : BC_DISP=) defines the boundary condition as a 
                                   displacement condtion
                - (KEY : BC_FUNC=) if the boundary condtions are a function of 
                                   the vibration modes
                - (KEY : BC_NODELOAD=) defines the boundary conditions as a 
                                       load
                - (KEY : BC_AERO=) defines the boundary condition as an 
                                   aerodynamic load
                - (KEY : BC_JOINT=) defines the joint between 2 bodies
            * values : value of the boundary conditions
            (KEY : BC_VALUES=) value of the displacement conditions
                       tx ty tz rx ry rz drx dry drz
                       tx : translation in x
                       ty : translation in y
                       tz : translation in z
                       rx : rotation in x
                       ry : rotation in y
                       rz : rotation in z
                       drx : derivative of the rotation in x
                       dry : derivative of the rotation in y
                       drz : derivative of the rotation in z
            * values_load : value of the load boundary conditions
                            tx ty tz rx ry rz drx dry drz
            (KEY : BC_VALUES_LOAD=) value of the load conditions
            * func : transient function of the boundary condition
            (KEY : FUNCTION=)
                - SIN : sinusoidal function
                - SIGM : sigmoid function
                - TABLE : table containing the function
            * f_flap : flag containing the presence of a flap
                - 1 : presence of flap
                - 0 : absence of flap
            * flap : marker of the flap
            (KEY : BC_FLAP=)
            * polar: name of the polar
            (KEY : BC_POLAR=)
            * polarflap : name of the flap polar
            (KEY : BC_FLAP_POLAR=)
            * vinf : free stream velocity (m/s)
            (KEY : BC_VINF=)
                - VINF_DAT : take global velocity of the problem
            * refpoint : reference point of the condition, used for calculating
                        the LLT and BEM from this point
            (KEY : BC_REFPOINT=)
            * flaprefpoint : reference point of the flap, used for calculating
                            the LLT and BEM from this point
            (KEY : BC_FLAP_REFPOINT=)
            * refCL : axis of reference of the lift coefficient x,y,z
            (KEY : BC_REFCL=)
            * refCD : axis of reference of the drag coefficient x,y,z
            (KEY : BC_REFCD=)
            * refCM : axis of reference of the moment coefficient x,y,z
            (KEY : BC_REFCM=)
            * refrot : rotation axis x,y,z
            (KEY : BC_REFROT=)
            * vdir : velocity direction x,y,z
            (KEY : BC_VDIR=)
            * aoa : angle of attack (deg)
            (KEY : BC_AOA=)
            * rho : density (kg/m3)
            (KEY : BC_RHO=)
            * mu : viscosity (Pa s)
            (KEY : BC_MU=)
            * l_ref : reference length (m)
            (KEY : L_REF=)
            * s_ref : reference surface (m2)
            (KEY : S_REF=)
            * radius : radius of the rotation surface (m)
            (KEY : BC_RADIUS=)
            * vrot : rotation velocity (rad/s)
            (KEY : BC_VROT=)
            * Nb : number of blades
            (KEY : BC_NB)
            * funcmode : mode function in the BC_FUNC conditions
            (KEY : FUNC_MODE=)
            * func: transient function of the boundary condition
            (KEY : FUNCTION=)
            * ampl: amplitude of the boundary condition function
            (KEY : AMPL=)
            * freq: frequency of the boundary condition function
            (KEY : FREQ=)
            * phas: phase of the boundary condition function
            (KEY : PHAS=)
            * slp: slope of the boundary condition function
            (KEY : SLP=)
            * m_poin: mid point of the boundary condition function
            (KEY : MID_POIN=)
            * file_tran: transient function file
            (KEY : FILE=)
            * periodic: periodicity of the boundary condition
            (KEY : PER=)
                - YES/NO
        * plots: postprocess of the data
            * typeplot : type of plot
                - XYPLOT : xy plot of magnitudes
                - DEF_SHAPE : 3D representation of the displacement
            * type : choose the type of xyplot
                - TIME : time in x axis
                - SPACE : coordinate in x axis
            * xlabel : label of the x axis
            * ylabel : label of the y axis
            * zlabel : label of the z axis
            * title : tile of the plot
            * label : label of the curve
            * anim : if the plot is an animation
                - YES/NO
            * xvalues : vector with the direction to plot
            * save : name of the file to save
            * save_data : name of the file to save the data
            * deltaframe : increment on the frame index in animation
            * marker : marker to measure the displacement
            * marker2 : marker to measure the displacement
            * mode : mode to plot
            * yaxis : variable to plot in the y axis
                (select the options, check in code)
            * yaxis2 : second varible to plot in the y axis
            * zaxis : variable to plot in the z axis
            * fig : figure index
            * color : colormap
            * fps : frames per second in a video
            * face : face creation (check options)
                - P2P : point to point, requirese sections with same number of
                        nodes
            * rot : rotatative aerodynamics (generates periodicity)
                - YES/NO
            * rot_name : name of the rotating boundary conditions
            * dihedral : rotate the view
            * colmax :
            * colmin : 
        * polar: contains the information about the aerodynamic polar to use
            * id : polar identifier
            * pointfile : (check)
            * file : polar file
            * quasi_steady: select quasi steady angle of attack or steady
                - 0 : steady
                - 1 : quasisteady
            * lltnodesflag: activate if the 3D effects are calculated in 
                            certain nodes
            * lltnodesmark: name of the marker where the 3D effects are 
                            calculated
            * cor3dflag : activate if the 3D effects of the rotations are 
                          required
            * cor3dmark : name of the marker where the 3D effects of the 
                          rotation are activated
            * indep : independent variables of the polar (check names)
            * dep : dependent variable of the polar (check names)
            * annfile : name of the ANN based surrogated model