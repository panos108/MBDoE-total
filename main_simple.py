import Models_simpler #specifications, DAE_system, integrator_model
import Models #specifications, DAE_system, integrator_model
import Models_mesbah #specifications, DAE_system, integrator_model

from utils import utilities
import numpy as np
import matplotlib.pyplot as plt

Model_bank = [Models_mesbah.model1, Models_mesbah.model2]
F       = []
thetas  = []
S_theta = []
true_model = Models_mesbah.model1()
for i in range(len(Model_bank)):
    F       += [Model_bank[i]().integrator_model()]
    thetas  += [Model_bank[i]().real_parameters]
    S_theta += [(0.025*np.diag(thetas[i]))**2*np.eye(len(Model_bank[i]().real_parameters))]

x_his_mc = np.zeros([1000, 10, 2])
dt, x0, _, _, _ = true_model.specifications()

MPC_ = utilities.MBDoE(Model_bank, 10, penalize_u=False, ukf=True, thetas=thetas, S_thetas=S_theta)
u_opt, x_opt, w_opt, S_opt = MPC_.solve_MPC_unc(x0, t=0.)  # , thetas=thetas, S_theta=S_theta)
u_apply = np.array(u_opt)

for k in range(1000):
    X_models   = []
    X_models_n = []

    for j in range(len(Model_bank)):
        dt, x0, _, _, _ = true_model.specifications()
        X_his   = np.empty((0,2), int)
        X_his_n = np.empty((0,2), int)
        thetass = np.array(3*[np.random.uniform(0.9,1.1)])#np.random.multivariate_normal(np.array(thetas[0]), np.array(S_theta[0]))
        for i in range(10):
            # MPC_ = utilities.MBDoE(Model_bank, 6, penalize_u=False, ukf=False, thetas=thetas, S_thetas=S_theta)


            x1 = F[j](x0=x0, p=(np.concatenate((u_apply[:,i], np.array(thetass)))))
            x0 = np.array(x1['xf']).reshape((-1,))
            x0_noisy = x0.copy()*(1+0.0*np.random.randn())
            X_his = np.vstack((X_his,x0.reshape((1,-1))))
            X_his_n = np.vstack((X_his_n,x0_noisy.reshape((1,-1))))
            #plt.plot(np.linspace(i, 6, 6 - i), x_opt[1, :6 - i].T)
        if j ==0:
            x_his_mc[k,:,:] = X_his_n
        X_models += [X_his]
        X_models_n += [X_his_n]

    import Criteria
    c = -Criteria.HR(X_models)
    print(2)

    plt.plot(np.linspace(1/10,2.5,10),X_models[0][:,1], label='Correct model')
    plt.plot(np.linspace(1/10,2.5,10),X_models_n[0][:,1],'*', label='Measurements')
    plt.plot(np.linspace(1/10,2.5,10),X_models[1][:,1], label='Wrong model')
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