import Models #specifications, DAE_system, integrator_model
from utils import utilities
import numpy as np
import matplotlib.pyplot as plt

Model_bank = [Models.Bio_reactor_1, Models.Bio_reactor_2]
F = []
for i in range(len(Model_bank)):
    F += [Model_bank[i]().integrator_model()]

MPC_ = utilities.MBDoE(Model_bank, 12, penalize_u=False)

u_opt, x_opt, w_opt = MPC_.solve_MPC(np.array([1, 150,0]), t=0.)
u_apply = np.array(u_opt)

X_models   = []
X_models_n = []

for j in range(len(Model_bank)):
    x0 = np.array([1, 150, 0])
    X_his   = np.empty((0,MPC_.nd), int)
    X_his_n = np.empty((0,MPC_.nd), int)

    for i in range(MPC_.N):
        x1 = F[j](x0=x0, p=u_apply[:,i])
        x0 = np.array(x1['xf']).reshape((-1,))
        x0_noisy = x0.copy()*(1+0.05*np.random.randn())
        X_his = np.vstack((X_his,x0.reshape((1,-1))))
        X_his_n = np.vstack((X_his_n,x0_noisy.reshape((1,-1))))

    X_models += [X_his]
    X_models_n += [X_his_n]

import Criteria
c = -Criteria.HR(X_models)
print(2)

plt.plot(np.linspace(1/12,240,12),X_models[0][:,1], label='Correct model')
plt.plot(np.linspace(1/12,240,12),X_models_n[0][:,1],'*', label='Measurements')
plt.plot(np.linspace(1/12,240,12),X_models[1][:,1], label='Wrong model')
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