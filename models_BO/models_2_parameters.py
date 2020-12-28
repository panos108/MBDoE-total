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