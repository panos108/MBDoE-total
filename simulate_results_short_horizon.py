
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

u0 = np.array([1.        , 0.59841716, 1.        , 1.        , 0.52834996,
       0.82219889, 1.        , 1.        ])

u0 = np.array([1.        , 1.        , 1.        , 1.        , 0.52838263,
       0.78663703, 0.63832465, 0.69328299])  #pyBOBYQA


u_opt1 = u_apply
u_opt = (ub-lb)*u0 + lb
u_opt = u_opt.reshape((2,4))

x_his_1 = np.zeros([1000,4,3,2])
x_his_2 = np.zeros([1000,4,3,2])
x_his_3 = np.zeros([1000,4,3,2])




for k in range(1000):
    X_models   = []
    X_models_n = []

    for j in range(len(Model_bank)):
        x0 = np.array([1, 150,0.000])
        X_his   = np.empty((0,3), int)
        X_his_n = np.empty((0,3), int)
        thetass = np.random.multivariate_normal(np.array(thetas[0]), np.array(S_theta[0]))
        for i in range(4):
            # MPC_ = utilities.MBDoE(Model_bank, 6, penalize_u=False, ukf=False, thetas=thetas, S_thetas=S_theta)

            u_apply = np.array(u_opt)[:, i]

            x1 = F[j](x0=x0, p=(np.concatenate((u_apply, np.array(thetass)))))
            x0 = np.array(x1['xf']).reshape((-1,))
            x0_noisy = x0.copy()*(1+0.0*np.random.randn())
            X_his = np.vstack((X_his,x0.reshape((1,-1))))
            X_his_n = np.vstack((X_his_n,x0_noisy.reshape((1,-1))))
            # plt.plot(np.linspace(i, 12, 12 - i), x_opt[1, :12 - i].T)
            x_his_1[k,i,:, j] = x0.copy()#X_his_n
        X_models += [X_his]
        X_models_n += [X_his_n]

    import Criteria
    c = -Criteria.HR(X_models)
    print(2)

    plt.plot(np.linspace(1/4,240,4),X_models[0][:,1], 'b', label='Correct model')
    # plt.plot(np.linspace(1/12,240,12),X_models_n[0][:,1],'*', label='Measurements')
    plt.plot(np.linspace(1/4,240,4),X_models[1][:,1], 'y', label='Wrong model')

for k in range(1000):
    X_models   = []
    X_models_n = []

    for j in range(len(Model_bank)):
        x0 = np.array([1, 150,0.000])
        X_his   = np.empty((0,3), int)
        X_his_n = np.empty((0,3), int)
        thetass = np.random.multivariate_normal(np.array(thetas[0]), np.array(S_theta[0]))
        for i in range(4):
            # MPC_ = utilities.MBDoE(Model_bank, 6, penalize_u=False, ukf=False, thetas=thetas, S_thetas=S_theta)

            u_apply = np.array(u_opt1)[:, i]

            x1 = F[j](x0=x0, p=(np.concatenate((u_apply, np.array(thetass)))))
            x0 = np.array(x1['xf']).reshape((-1,))
            x0_noisy = x0.copy()*(1+0.0*np.random.randn())
            X_his = np.vstack((X_his,x0.reshape((1,-1))))
            X_his_n = np.vstack((X_his_n,x0_noisy.reshape((1,-1))))
            # plt.plot(np.linspace(i, 12, 12 - i), x_opt[1, :12 - i].T)
        # if j ==0:
        #     x_his_mc[k,:,:] = X_his_n
            x_his_2[k, i, :, j] = x0.copy()#X_his_n

        X_models += [X_his]
        X_models_n += [X_his_n]

    import Criteria
    c = -Criteria.HR(X_models)
    print(2)

    plt.plot(np.linspace(1/4,240,4),X_models[0][:,1], 'k--', label='Correct model')
    # plt.plot(np.linspace(1/12,240,12),X_models_n[0][:,1],'*', label='Measurements')
    plt.plot(np.linspace(1/4,240,4),X_models[1][:,1], 'r--', label='Wrong model')
x0 = np.array([1, 150, 0.000])

MPC_ = utilities.MBDoE(Model_bank, 4, penalize_u=False, ukf=True, thetas=thetas, S_thetas=S_theta)
u_opt2, x_opt, w_opt, S_opt = MPC_.solve_MPC_unc(x0, t=0.)  # , thetas=thetas, S_theta=S_theta)


for k in range(1000):
    X_models   = []
    X_models_n = []

    for j in range(len(Model_bank)):
        x0 = np.array([1, 150,0.000])
        X_his   = np.empty((0,3), int)
        X_his_n = np.empty((0,3), int)
        thetass = np.random.multivariate_normal(np.array(thetas[0]), np.array(S_theta[0]))
        for i in range(4):
            # MPC_ = utilities.MBDoE(Model_bank, 6, penalize_u=False, ukf=False, thetas=thetas, S_thetas=S_theta)

            u_apply = np.array(u_opt2)[:, i]

            x1 = F[j](x0=x0, p=(np.concatenate((u_apply, np.array(thetass)))))
            x0 = np.array(x1['xf']).reshape((-1,))
            x0_noisy = x0.copy()*(1+0.0*np.random.randn())
            X_his = np.vstack((X_his,x0.reshape((1,-1))))
            X_his_n = np.vstack((X_his_n,x0_noisy.reshape((1,-1))))
            # plt.plot(np.linspace(i, 12, 12 - i), x_opt[1, :12 - i].T)
        # if j ==0:
        #     x_his_mc[k,:,:] = X_his_n
            x_his_3[k, i, :, j] = x0.copy()#X_his_n

        X_models += [X_his]
        X_models_n += [X_his_n]

    import Criteria
    c = -Criteria.HR(X_models)
    print(2)

    plt.plot(np.linspace(1/4,240,4),X_models[0][:,1], 'g*-', label='Correct model')
    # plt.plot(np.linspace(1/12,240,12),X_models_n[0][:,1],'*', label='Measurements')
    plt.plot(np.linspace(1/4,240,4),X_models[1][:,1], 'y*-', label='Wrong model')


print(2)
n=4

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_3[:,:,1,0],0.01,axis=0),
                 np.quantile(x_his_3[:,:,1,0],0.99,axis=0),
                 alpha=0.5, color='#226666')
plt.plot(np.linspace(1/n,240,n),x_his_3[:,:,1,0].mean(axis=0), color='#226666',label='Mean Satisfaction (standard)')


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,1,0],0.01,axis=0),
                 np.quantile(x_his_2[:,:,1,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,1,0].mean(axis=0), color='#7B9F35',label='Probability 0.95')


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_1[:,:,1,0],0.01,axis=0),
                 np.quantile(x_his_1[:,:,1,0],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_1[:,:,1,0].mean(axis=0), color='#AA3939',label='Mean Satisfaction 0.5 (Bayes)')

plt.xlabel('time(hrs)')
plt.ylabel('Nitrate conc (mg/L)')
plt.tight_layout()
plt.legend()
plt.savefig('CN_different_scheme.png',dpi=400)
plt.close()


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,1,0],0.01,axis=0),
                 np.quantile(x_his_2[:,:,1,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,1,0].mean(axis=0), color='#7B9F35',label='Probability 0.95 model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,1,1],0.01,axis=0),
                 np.quantile(x_his_2[:,:,1,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,1,1].mean(axis=0), color='#AA3939',label='Probability 0.95 model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Nitrate conc (mg/L)')
plt.title('Nitrate discrimination with BH chance constraint')
plt.legend()
plt.tight_layout()
plt.savefig('CN_chance constraint.png',dpi=400)
plt.close()


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_3[:,:,1,0],0.01,axis=0),
                 np.quantile(x_his_3[:,:,1,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_3[:,:,1,0].mean(axis=0), color='#7B9F35',label='model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_3[:,:,1,1],0.01,axis=0),
                 np.quantile(x_his_3[:,:,1,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_3[:,:,1,1].mean(axis=0), color='#AA3939',label='model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Nitrate conc (mg/L)')
plt.title('Nitrate discrimination with BH mean standard')
plt.legend()
plt.tight_layout()
plt.savefig('CN_mean standard.png',dpi=400)
plt.close()


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_1[:,:,1,0],0.01,axis=0),
                 np.quantile(x_his_1[:,:,1,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_1[:,:,1,0].mean(axis=0), color='#7B9F35',label='model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_1[:,:,1,1],0.01,axis=0),
                 np.quantile(x_his_1[:,:,1,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_1[:,:,1,1].mean(axis=0), color='#AA3939',label='model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Nitrate conc (mg/L)')
plt.title('Nitrate discrimination with BH mean BO')
plt.legend()
plt.tight_layout()
plt.savefig('CN_mean BO.png',dpi=400)
plt.close()

#-----------------------------------------------------#
#--------------------------------------------------------#


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,0,0],0.01,axis=0),
                 np.quantile(x_his_2[:,:,0,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,0,0].mean(axis=0), color='#7B9F35',label='Probability 0.95 model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,0,1],0.01,axis=0),
                 np.quantile(x_his_2[:,:,0,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,0,1].mean(axis=0), color='#AA3939',label='Probability 0.95 model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Biomass conc (mg/L)')
plt.title('Biomass discrimination with BH chance constraint')
plt.legend()
plt.tight_layout()
plt.savefig('CX_chance constraint.png',dpi=400)
plt.close()


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_3[:,:,0,0],0.01,axis=0),
                 np.quantile(x_his_3[:,:,0,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_3[:,:,0,0].mean(axis=0), color='#7B9F35',label='model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_3[:,:,0,1],0.01,axis=0),
                 np.quantile(x_his_3[:,:,0,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_3[:,:,0,1].mean(axis=0), color='#AA3939',label='model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Biomass conc (mg/L)')
plt.title('Biomass discrimination with BH mean standard')
plt.legend()
plt.tight_layout()
plt.savefig('CX_mean standard.png',dpi=400)
plt.close()


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_1[:,:,0,0],0.01,axis=0),
                 np.quantile(x_his_1[:,:,0,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_1[:,:,0,0].mean(axis=0), color='#7B9F35',label='model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_1[:,:,0,1],0.01,axis=0),
                 np.quantile(x_his_1[:,:,0,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_1[:,:,0,1].mean(axis=0), color='#AA3939',label='model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Biomass conc (mg/L)')
plt.title('Biomass discrimination with BH mean BO')
plt.legend()
plt.tight_layout()
plt.savefig('CX_mean BO.png',dpi=400)
plt.close()


#------------------------------------------------------------#





plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,2,0],0.01,axis=0),
                 np.quantile(x_his_2[:,:,2,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,2,0].mean(axis=0), color='#7B9F35',label='Probability 0.95 model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,2,1],0.01,axis=0),
                 np.quantile(x_his_2[:,:,2,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,2,1].mean(axis=0), color='#AA3939',label='Probability 0.95 model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Product conc (mg/L)')
plt.title('Product discrimination with BH chance constraint')
plt.legend()
plt.tight_layout()
plt.savefig('Product_chance constraint.png',dpi=400)
plt.close()


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_3[:,:,2,0],0.01,axis=0),
                 np.quantile(x_his_3[:,:,2,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_3[:,:,2,0].mean(axis=0), color='#7B9F35',label='model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_3[:,:,2,1],0.01,axis=0),
                 np.quantile(x_his_3[:,:,2,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_3[:,:,2,1].mean(axis=0), color='#AA3939',label='model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Product conc (mg/L)')
plt.title('Product discrimination with BH mean standard')
plt.legend()
plt.tight_layout()
plt.savefig('Product_mean standard.png',dpi=400)
plt.close()


plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_1[:,:,2,0],0.01,axis=0),
                 np.quantile(x_his_1[:,:,2,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_1[:,:,2,0].mean(axis=0), color='#7B9F35',label='model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_1[:,:,2,1],0.01,axis=0),
                 np.quantile(x_his_1[:,:,2,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_1[:,:,2,1].mean(axis=0), color='#AA3939',label='model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Product conc (mg/L)')
plt.title('Product discrimination with BH mean BO')
plt.legend()
plt.tight_layout()
plt.savefig('Product_mean BO.png',dpi=400)
plt.close()


#-----------------------------------------------------#







plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,0,0],0.01,axis=0),
                 np.quantile(x_his_2[:,:,0,0],0.99,axis=0),
                 alpha=0.5, color='#7B9F35')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,0,0].mean(axis=0), color='#7B9F35',label='Probability 0.95 model 1')

plt.fill_between(np.linspace(1/n,240,n),np.quantile(x_his_2[:,:,0,1],0.01,axis=0),
                 np.quantile(x_his_2[:,:,0,1],0.99,axis=0),
                 alpha=0.5, color='#AA3939')
plt.plot(np.linspace(1/n,240,n),x_his_2[:,:,0,1].mean(axis=0), color='#AA3939',label='Probability 0.95 model 2')

plt.xlabel('time(hrs)')
plt.ylabel('Biomass conc (mg/L)')
plt.title('Biomass discrimination with BH standard with mean constraints')
plt.legend()
plt.tight_layout()
plt.savefig('CX_mean_Standard.png',dpi=400)
plt.close()



plt.plot(np.linspace(1/12,240,12),X_models[0][:,0], label='Correct model')
plt.plot(np.linspace(1/12,240,12),X_models_n[0][:,0],'*', label='Measurements')
plt.plot(np.linspace(1/12,240,12),X_models[1][:,0], label='Wrong model')
plt.xlabel('time(hrs)')
plt.ylabel('Biomass conc (mg/L)')
plt.tight_layout()
plt.legend()
plt.savefig('Cb.png')
plt.close()


plt.plot(np.linspace(1/12,240,12),X_models[0][:,2], label='Correct model')
plt.plot(np.linspace(1/12,240,12),X_models_n[0][:,2],'*', label='Measurements')
plt.plot(np.linspace(1/12,240,12),X_models[1][:,2], label='Wrong model')
plt.xlabel('time(hrs)')
plt.ylabel('Product conc (mg/L)')
plt.tight_layout()
plt.legend()
plt.savefig('Cp.png')
plt.close()