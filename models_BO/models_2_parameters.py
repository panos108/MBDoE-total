import Models_simpler_short_horizon #specifications, DAE_system, integrator_model
import Models #specifications, DAE_system, integrator_model

from utils import utilities

import numpy as np
import matplotlib.pyplot as plt
import Criteria





def Objective( x0, thetas, S_theta, uncertainty_calcs, u0):
    ub = np.array([[400]*4+[40]*4])
    lb = np.array([[120]*4+[0]*4])

    u = (ub-lb)*u0+lb
    x_his1, S_his1 = uncertainty_calcs[0].compute_full_path( u, 4, x0, np.zeros([3,3]),
                                                 np.array(thetas[0]), np.array(S_theta[0]), x0,
                                            1e-7 * np.eye(3), 1e-7 * np.eye(3))
    x_his2, S_his2 = uncertainty_calcs[1].compute_full_path( u, 4, x0, np.zeros([3,3]),
                                                 np.array(thetas[0]), np.array(S_theta[0]), x0,
                                            1e-7 * np.eye(3), 1e-7 * np.eye(3))

    x_his = [x_his1,x_his2]
    S_his = [S_his1,S_his2]

    c_1 = Criteria.BF_numpy(x_his, S_his, noise_var=1e-4*np.eye(3))

    return -c_1[0,0]


def Constraint1(x0, thetas, S_theta, uncertainty_calcs,i, u0):
    ub = np.array([[400]*4+[40]*4])
    lb = np.array([[120]*4+[0]*4])
    u = (ub-lb)*u0+lb


    x_his1, S_his1 = uncertainty_calcs[0].compute_full_path( u, 4, x0, np.zeros([3,3]),
                                                 np.array(thetas[0]), np.array(S_theta[0]), x0,
                                            1e-7 * np.eye(3), 1e-7 * np.eye(3))
    c = (x_his1[i, 1]) - 800
    return -np.array([c])

def combined_obj( x0, thetas, S_theta, uncertainty_calcs, u0):
    ub = np.array([[400]*4+[40]*4])
    lb = np.array([[120]*4+[0]*4])

    u = (ub-lb)*u0+lb
    x_his1, S_his1 = uncertainty_calcs[0].compute_full_path( u, 4, x0, np.zeros([3,3]),
                                                 np.array(thetas[0]), np.array(S_theta[0]), x0,
                                            1e-7 * np.eye(3), 1e-7 * np.eye(3))
    x_his2, S_his2 = uncertainty_calcs[1].compute_full_path( u, 4, x0, np.zeros([3,3]),
                                                 np.array(thetas[0]), np.array(S_theta[0]), x0,
                                            1e-7 * np.eye(3), 1e-7 * np.eye(3))

    x_his = [x_his1,x_his2]
    S_his = [S_his1,S_his2]
    c = 0
    for i in range(4):
        c += max(0,((x_his1[i, 1])/800 - 1))**2

    c_1 = Criteria.BF_numpy(x_his, S_his, noise_var=1e-4*np.eye(3))- 1e6* c

    return -c_1[0,0]
# def Constraint2(u, x0, thetas, S_theta, uncertainty_calcs):
#     x_his1, S_his1 = uncertainty_calcs[0].compute_full_path( u, 4, x0, np.zeros([3,3]),
#                                                  np.array(thetas[0]), np.array(S_theta[0]), x0,
#                                             1e-7 * np.eye(3), 1e-7 * np.eye(3))
#     x_his2, S_his2 = uncertainty_calcs[1].compute_full_path( u, 4, x0, np.zeros([3,3]),
#                                                  np.array(thetas[0]), np.array(S_theta[0]), x0,
#                                             1e-7 * np.eye(3), 1e-7 * np.eye(3))
#
#     x_his = [x_his1,x_his2]
#     S_his = [S_his1,S_his2]
#
#     c_1 = Criteria.BF_numpy(x_his, S_his, noise_var=1e-4*np.eye(3))
#
#     return c_1



def combined_obj_scenario( x0, thetas, thetas_samples, uncertainty_calcs, u0):
    ub = np.array([[400]*4+[40]*4])
    lb = np.array([[120]*4+[0]*4])

    u = (ub-lb)*u0+lb
    Samples        = 750

    x1_his = np.zeros([4,3,Samples])
    x2_his = np.zeros([4,3,Samples])
    # import time
    # start = time.time()
    S1 = []
    S2 = []

    for i in range(Samples):
        x1 = uncertainty_calcs[0].compute_full_path( u, 4, x0, np.zeros([3,3]),
                                                     thetas_samples[i,:],  np.array(thetas_samples[0])*0, x0,
                                                1e-7 * np.eye(3), 1e-7 * np.eye(3))
        x2 = uncertainty_calcs[1].compute_full_path( u, 4, x0, np.zeros([3,3]),
                                                     thetas_samples[i,:],  np.array(thetas_samples[0])*0, x0,
                                                1e-7 * np.eye(3), 1e-7 * np.eye(3))
        x1_his[:,:,i] = x1
        x2_his[:,:,i] = x2
    # print(time.time()-start)
    x_his = [x1_his.mean(2),x2_his.mean(2)]
    for i in range(4):
        S_comp_1 = np.cov(x1_his[i])+1e-7*(np.eye(3))
        S_comp_2 = np.cov(x2_his[i])+1e-7*(np.eye(3))

        S1 += [S_comp_1]
        S2 += [S_comp_2]
    S_his = [S1,S2]
    #CHECK COVARIANCE via UT


    # S = (x1_his[0,:, [0]] - x1_his.mean(2)[[0]]).T @\
    #     (x1_his[0,:, [0]] - x1_his.mean(2)[[0]])  # Y1[:,[0]] @ Y1[:,[0]].T
    # WRITE A FUNC FOR COVARIANCE
    S = 0
    for i in range(Samples):
        S += (x1_his[0,:, [i]] - x1_his.mean(2)[[0]]).T\
             @ (x1_his[0,:, [i]] - x1_his.mean(2)[[0]])  # Y1[:,[0]] @ Y1[:,[0]].T
  # Wc[i]*Y1[:,[i]] @ Y1[:,[i]].T
    S = S/Samples
    S += 1e-7 * (np.eye(3))
    c_1   = Criteria.BF_numpy(x_his, S_his, noise_var=1e-4 * np.eye(3))
    P1    = -np.sum(x1_his[:,1,:]/800-1<0,axis=1)/Samples + 0.5

    c = 0
    for i in range(4):
        c += max(0,P1[i])**2

    c_1 += - 1e1*c
    return -c_1[0,0]
