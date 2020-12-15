import Models_simpler #specifications, DAE_system, integrator_model
import Models #specifications, DAE_system, integrator_model

from utils import utilities

import numpy as np
import matplotlib.pyplot as plt
import Criteria
Model_bank = [Models_simpler.Bio_reactor_1, Models_simpler.Bio_reactor_2]

F       = []
thetas  = []
S_theta = []
for i in range(len(Model_bank)):
    F       += [Model_bank[i]().integrator_model()]
    thetas  += [Model_bank[i]().real_parameters]
    S_theta += [(0.05*np.diag(thetas[i]))**2*np.eye(len(Model_bank[i]().real_parameters))]
x0 = np.array([1, 150, 0.000])

x_his_mc = np.zeros([1000, 12, 3])
MPC_ = utilities.MBDoE(Model_bank, 12, penalize_u=False, ukf=True, thetas=thetas, S_thetas=S_theta)
u_opt, x_opt, w_opt, S_opt = MPC_.solve_MPC_unc(x0, t=0.)  # , thetas=thetas, S_theta=S_theta)
# u_apply = np.array(u_opt)[:, 0]
u_apply = np.array(u_opt)#[:, 0]
u_apply = np.array([[400.        , 400.        , 400.        , 400.        ,
        399.9999969 , 399.99999248, 399.99998624, 399.99997065,
        399.99998212, 399.99998614, 399.99998556, 399.99998612],
       [ 29.68008886,   7.57284794,  27.17800014,  21.18213727,
         28.89072698,  39.43832545,  39.99999865,  39.99999573,
         39.99999338,  39.99998782,  39.99996573,  31.94088991]])





    # array([[4.00000000e+02, 4.00000000e+02, 4.00000000e+02, 4.00000000e+02,
    #                 3.99999999e+02, 3.99999997e+02, 3.99999995e+02, 3.99999987e+02,
    #                 3.99999996e+02, 3.99999998e+02, 3.99999997e+02, 3.99999997e+02],
    #                [2.86976510e+01, 5.33312861e-06, 2.35468335e+01, 2.82271785e+01,
    #                 2.69495636e+01, 3.67814904e+01, 3.99999995e+01, 3.99999985e+01,
    #                 3.99999978e+01, 3.99999967e+01, 3.99999940e+01, 3.99999861e+01]])


uncertainty_calcs = [utilities.Uncertainty_module(Model_bank[0]),
                     utilities.Uncertainty_module(Model_bank[1])]


x_his1, S_his1 = uncertainty_calcs[0].compute_full_path( u_apply.reshape(2*12), 12, x0, np.zeros([3,3]),
                                             np.array(thetas[0]), np.array(S_theta[0]), x0,
                                        1e-7 * np.eye(3), 1e-7 * np.eye(3))
x_his2, S_his2 = uncertainty_calcs[1].compute_full_path( u_apply.reshape(2*12), 12, x0, np.zeros([3,3]),
                                             np.array(thetas[0]), np.array(S_theta[0]), x0,
                                        1e-7 * np.eye(3), 1e-7 * np.eye(3))

x_his = [x_his1,x_his2]
S_his = [S_his1,S_his2]

c_1 = Criteria.BF_numpy(x_his, S_his, noise_var=1e-4*np.eye(3))


print(2)
# c = uncertainty_calcs.compute_FIM(x0, u_apply*0.5, thetas[0], S_exp=1e-6*np.eye(3),criterion='A')





#--------------------------------------------------------------#


for k in range(1000):
    X_models   = []
    X_models_n = []

    for j in range(len(Model_bank)):
        x0 = np.array([1, 150,0.000])
        X_his   = np.empty((0,3), int)
        X_his_n = np.empty((0,3), int)
        thetass = np.random.multivariate_normal(np.array(thetas[0]), np.array(S_theta[0]))
        for i in range(12):
            # MPC_ = utilities.MBDoE(Model_bank, 6, penalize_u=False, ukf=False, thetas=thetas, S_thetas=S_theta)

            u_apply = np.array(u_opt)[:, i]

            x1 = F[j](x0=x0, p=(np.concatenate((u_apply, np.array(thetass)))))
            x0 = np.array(x1['xf']).reshape((-1,))
            x0_noisy = x0.copy()*(1+0.0*np.random.randn())
            X_his = np.vstack((X_his,x0.reshape((1,-1))))
            X_his_n = np.vstack((X_his_n,x0_noisy.reshape((1,-1))))
            # plt.plot(np.linspace(i, 12, 12 - i), x_opt[1, :12 - i].T)
        if j ==0:
            x_his_mc[k,:,:] = X_his_n
        X_models += [X_his]
        X_models_n += [X_his_n]

    import Criteria
    c = -Criteria.HR(X_models)
    print(2)

    plt.plot(np.linspace(1/12,240,12),X_models[0][:,1], label='Correct model')
    # plt.plot(np.linspace(1/12,240,12),X_models_n[0][:,1],'*', label='Measurements')
    plt.plot(np.linspace(1/12,240,12),X_models[1][:,1], 'k', label='Wrong model')
plt.xlabel('time(hrs)')
plt.ylabel('Nitrate conc (mg/L)')
plt.tight_layout()
plt.legend()
plt.savefig('CN.png')
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