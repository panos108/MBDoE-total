import time
import random
import numpy as np
import numpy.random as rnd
from scipy.spatial.distance import cdist

import sobol_seq
from scipy.optimize import minimize
from scipy.optimize import broyden1
from scipy import linalg
import scipy
import matplotlib.pyplot as plt
import functools
from matplotlib.patches import Ellipse
# from RTO_MBDoE_case.run_Bio.samples_eval_gen import samples_generation
from casadi import *


from sub_uts_BO.utilities_4_no_model import *
from sub_uts_BO import systems2
import Models_simpler_short_horizon #specifications, DAE_system, integrator_model
import Models #specifications, DAE_system, integrator_model

from utils import utilities
from models_BO import models_2_parameters

import Criteria

Model_bank = [Models_simpler_short_horizon.Bio_reactor_1, Models_simpler_short_horizon.Bio_reactor_2]

F       = []
thetas  = []
S_theta = []
for i in range(len(Model_bank)):
    F       += [Model_bank[i]().integrator_model()]
    thetas  += [Model_bank[i]().real_parameters]
    S_theta += [(0.05*np.diag(thetas[i]))**2*np.eye(len(Model_bank[i]().real_parameters))]

x0 = np.array([1, 150, 0.000])

u_apply = np.array([[400.        , 399.99999982, 399.9999974 , 400.        ],
       [ 17.84891533,  21.55702332,  39.99999838,  39.99999722]])#np.array(u_opt)#[:, 0]

uncertainty_calcs = [utilities.Uncertainty_module(Model_bank[0]),
                     utilities.Uncertainty_module(Model_bank[1])]
u0 = u_apply.reshape(2*4)
ub = np.array([[400] * 4 + [40] * 4])
lb = np.array([[120] * 4 + [0] * 4])
u = (u0 - lb)/(ub-lb)

# obj  = models_2_parameters.Objective(u_apply.reshape(2*4), x0, thetas, S_theta, uncertainty_calcs)
# con1 = models_2_parameters.Constraint1(u_apply.reshape(2*4), x0, thetas, S_theta, uncertainty_calcs)

obj  = models_2_parameters.Objective(x0, thetas, S_theta, uncertainty_calcs, u)
con1 = models_2_parameters.Constraint1(x0, thetas, S_theta, uncertainty_calcs,3, u)


# from plots_RTO import compute_obj, plot_obj, plot_obj_noise

import pickle
# from plots_RTO import Plot
#----------1) EI-NO PRIOR-UNKNOWN NOISE----------#
#---------------------------------------------#
if not(os.path.exists('figs_WO')):
    os.mkdir('figs_WO')
if not(os.path.exists('figs_noise_WO')):
    os.mkdir('figs_noise_WO')

#------------------------------------------------------------------
np.random.seed(0)
X_opt_mc = []
y_opt_mc = []
TR_l_mc = []
xnew_mc = []
backtrack_1_mc = []

for i in range(1):
    plant = 0#models.Static_PDE_reaction_system()
    X        = np.random.rand(20,8)#np.array([[0., 1.], [0.1, 1.], [0., 0.9]])#np.array([[0., 0.], [0.1, 0.], [0., 0.1]])
    xo       = np.random.rand(8)#np.array([0.0,0.85])#np.array([0.0,0.05])
    obj_model  = systems2.obj_empty#model.objective
    obj_system = functools.partial(models_2_parameters.Objective, x0, thetas, S_theta, uncertainty_calcs)
    obj_system_combined = functools.partial(models_2_parameters.combined_obj, x0, thetas, S_theta, uncertainty_calcs)
    thetas_samples = np.random.multivariate_normal(np.array(thetas[0]),
                                                   np.array(S_theta[0]), 750)
    obj_system_scenario = functools.partial(models_2_parameters.combined_obj_scenario, x0, thetas, thetas_samples, uncertainty_calcs)

    u_opt = np.array( [1.        , 1.        , 1.        , 1.        , 0.52838263,
        0.78663703, 0.63832465, 0.69328299])

       #  [0.99999999, 1.        , 0.99999999, 1.        , 1.        ,
       # 1.        , 0.51554809, 0.99999995])

       #  [1.        , 1.        , 1.        , 1.        , 0.52838263,
       # 0.78663703, 0.63832465, 0.69328299])
       #  [0.59895805, 0.45336043, 0.72163113, 0.86737233, 0.98504793,
       # 0.85326827, 0.0107703 , 0.3626316 ])
    obj_system_combined(u_opt)
    obj_system_scenario(u_opt)
    cons_model = []# l.WO_obj_ca
    #cons_model.append(model.constraint1)
    #cons_model.append(model.constraint2)

    cons_model.append(systems2.obj_empty)
    cons_model.append(systems2.obj_empty)
    cons_model.append(systems2.obj_empty)
    cons_model.append(systems2.obj_empty)

    # cons_model.append(model.constraint_agg_2)

    cons_system = []
    #cons_system.append(plant.constraint1)
    # cons_system.append(plant.constraint2)
    cons_system.append(functools.partial( models_2_parameters.Constraint1, x0, thetas, S_theta, uncertainty_calcs,0))
    cons_system.append(functools.partial( models_2_parameters.Constraint1, x0, thetas, S_theta, uncertainty_calcs,1))
    cons_system.append(functools.partial( models_2_parameters.Constraint1, x0, thetas, S_theta, uncertainty_calcs,2))
    cons_system.append(functools.partial( models_2_parameters.Constraint1, x0, thetas, S_theta, uncertainty_calcs,3))

    # cons_system.append(plant.constraint_agg_2)
    import pybobyqa

    np.random.seed(0)

    print("Demonstrate noise in function evaluation:")
    for i in range(5):
        print("objfun(x0) = %s" % str(obj_system_combined(xo)))
    print("")

    # Call Py-BOBYQA
    lower = np.array([0.0]*8)
    upper = np.array([1.]*8)
    # soln = pybobyqa.solve(obj_system_combined, xo, bounds=(lower,upper), maxfun=4000, objfun_has_noise=False)#, seek_global_minimum=True)
    # print(soln)
    soln_scenario = pybobyqa.solve(obj_system_scenario, xo, bounds=(lower,upper), maxfun=4000, objfun_has_noise=False)#, seek_global_minimum=True)
    print(soln_scenario)
    n_iter         = 20
    bounds         = ([[0., 1.]] * 8)#[[0.,1.],[0.,1.]]
    #X              = pickle.load(open('initial_data_bio_12_ca_new.p','rb'))
    Xtrain         = X#1.*(np.random.rand(model.nk*model.nu+500,model.nk*model.nu))+0.#np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
#
#1.*(np.random.rand(model.nk*model.nu+500,model.nk*model.nu))+0.#np.array([[5.7, 74.],[6.35, 74.9],[6.6,75.],[6.75,79.]]) #U0
    #Xtrain         = np.array([[7.2, 74.],[7.2, 80],[6.7,75.]])#,[6.75,83.]]) #U0
    samples_number = Xtrain.shape[0]
    data           = ['data0', Xtrain]
    u0             = xo#model.nk*model.nu+1]#np.array([*[0.6]*model.nk,*[0.8]*model.nk])#

    Delta0         = 1000
    Delta_max      =100000; eta0=0.2; eta1=0.8; gamma_red=0.8; gamma_incr=1.2
    TR_scaling_    = False
    TR_curvature_  = False
    inner_TR_      = False
    noise = None#[0.5**2, 5e-8, 5e-8]



    ITR_GP_opt         = ITR_GP_RTO(obj_model, obj_system, cons_model, cons_system, u0, Delta0,
                                    Delta_max, eta0, eta1, gamma_red, gamma_incr,
                                    n_iter, data, np.array(bounds),obj_setting=2, noise=noise, multi_opt=50,
                                    multi_hyper=5, TR_scaling=TR_scaling_, TR_curvature=TR_curvature_,
                                    store_data=True, inner_TR=inner_TR_, scale_inputs=False)#, model=model)

    print('Episode: ',i)
    if not TR_curvature_:
        X_opt, y_opt, TR_l, xnew, backtrack_l             = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    else:
        X_opt, y_opt, TR_l, TR_l_angle, xnew, backtrack_l = ITR_GP_opt.RTO_routine()
        backtrack_l                                       = [False, *backtrack_l]
    X_opt_mc       += [X_opt]
    y_opt_mc       += [y_opt]
    TR_l_mc        += [TR_l]
    xnew_mc        += [xnew]
    backtrack_1_mc += [backtrack_l]
print(2)
pickle.dump([X_opt_mc, y_opt_mc,TR_l_mc, xnew_mc, backtrack_1_mc], open('Prob_no_prior_with_exploration_ei.p','wb'))




print(2)