from casadi import *
import numpy as np
import scipy.linalg as scipylinalg
csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
from utils.OrthogonalCollocation import construct_polynomials_basis

import Criteria
class MBDoE:

    def __init__(self, Model_Def, horizon, thetas, S_thetas, collocation_degree = 4, penalize_u=False,
                 ukf=False, theta_unc=None, S_exp = None):

        self.NoModels  = len(Model_Def)
        self.Model_def = []
        self.ukf       = ukf
        for i in range(self.NoModels):
            self.Model_def += [Model_Def[i]()]          # Take class of the dynamic system

        self.dc        = collocation_degree    # Define the degree of collocation
        self.N         = horizon               # Define the Horizon of the problem
        # FIXME  Add different horizon for control and prediction
        dt, x0, Lsolver, c_code, self.shrinking_horizon = self.Model_def[0].specifications()
        self.dt        = dt
        self.f         = []
        self.hmeas     = []
        self.S_theta   = []
        # FIXME Change the DAE to be for all the models
        if ukf:
            for i in range(self.NoModels):
                xd, _, u, uncertainty, ODEeq, _, self.u_min, self.u_max, self.x_min, self.x_max, _, \
                _, _, self.nd, _, self.nu, self.n_ref, self.ntheta, _, self.ng, self.gfcn, \
                self.Obj_M, self.Obj_L, self.Obj_D, self.R = self.Model_def[i].DAE_system(uncertain_parameters=True) # Define the System
                self.f       += [Function('f1', [xd, u, uncertainty], [vertcat(*ODEeq)])]
                self.hmeas   += [Function('h1', [xd, u], [xd])]
                # self.S_theta += [theta_unc[i]]


        else:
            for i in range(self.NoModels):
                xd, _, u, uncertainty, ODEeq, _, self.u_min, self.u_max, self.x_min, self.x_max, _, \
                _, _, self.nd, _, self.nu, self.n_ref, self.ntheta, _, self.ng, self.gfcn, \
                self.Obj_M, self.Obj_L, self.Obj_D, self.R = self.Model_def[i].DAE_system() # Define the System
                self.f += [Function('f1', [xd, u,uncertainty], [vertcat(*ODEeq)])]

        """
        Define noise and disturbances for the system
        """
        self.Q = 1e-7 * np.eye(self.nd)
        if S_exp == None:
            self.S_exp = 1e-4 * np.eye(self.nd)

        self.penalize_u = penalize_u

        # Define options for solver
        opts = {}
        opts["expand"] = True
        opts["ipopt.print_level"] = 5
        opts["ipopt.max_iter"] = 1000
        opts["ipopt.tol"] = 1e-8
        opts["calc_lam_p"] = False
        opts["calc_multipliers"] = False
        opts["ipopt.print_timing_statistics"] = "no"
        opts["print_time"] = False
        self.opts = opts
        if not(ukf):
            self.MPC_construct()
        else:

            self.MPC_construct_ukf_thetas(thetas, S_thetas)



    def MPC_construct(self):
        """
        ODEeq: Is the system of ODE
        gcn  : Is the inequality constraints
        ObjM : Is the mayer term of DO
        ObjL : Is the Lagrange term of DO
        Obj_D: The a discretized objective for each time
        :return:
        """
        N        = self.N
        dc       = self.dc
        dt       = self.dt
        NoModels = self.NoModels

        C, D, B = construct_polynomials_basis(dc, 'radau')
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        Ts = []
        t = 0
        # "Lift" initial conditions
        x_plot  = []

        u_plot  = []
        X_models= []
        X_0    = SX.sym('p_x'     , self.nd)  # This is a parameter that defines the Initial Conditions
        shrink = SX.sym('p_shrink', self.N)
        x_ref  = SX.sym('p_ref'   , self.n_ref)
        thetas = SX.sym('p_thetas', self.ntheta * NoModels)

        if self.penalize_u:
            U_past  = SX.sym('p_u', self.nu)  #This is a parameter that defines the Initial Conditions
            prev    = U_past
        u_apply = []
        for m in range(NoModels):
            Xk = SX.sym('X0', self.nd)
            w += [Xk]
            lbw += [*self.x_min]
            ubw += [*self.x_max]
            w0 += [*self.x_min]
            g += [Xk - X_0]
            lbg += [*np.zeros([self.nd])]
            ubg += [*np.zeros([self.nd])]
            x_plot += [Xk]
            X_his = []

            theta = SX.sym('theta', self.ntheta)
            w += [theta]
            lbw += [*(0*np.ones([self.ntheta]))]
#[*(0.8*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
#                                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(0*np.ones([self.ntheta]))]
            ubw += [*(1000*np.ones([self.ntheta]))]
                #[*(1.1*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
                #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(500*np.ones([self.ntheta]))]
            w0 += [*(100*np.ones([self.ntheta]))]
                #[*(np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
                #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]
            g += [theta - thetas[m*self.ntheta:(m+1)*(self.ntheta)]]
            lbg += [*np.zeros([self.ntheta])]
            ubg += [*np.zeros([self.ntheta])]

            for i in range(N):
            # Formulate the NLP
            # New NLP variable for the control
                if m ==0:
                    Uk   = SX.sym('U_' + str(i), self.nu)
                    if self.penalize_u:
                        J += (Uk-prev).T @ self.R @ (Uk - prev) * shrink[i]
                        prev = Uk
                    w += [Uk]
                    lbw += [*self.u_min]
                    ubw += [*self.u_max]
                    w0 += [*(self.u_min)]
                    u_plot  += [Uk]
                    u_apply += [Uk]

    # Integrate till the end of the interval
                w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot, _ = self.perform_orthogonal_collocation(dc, self.nd, w, lbw, ubw, w0,
                                                     self.x_min, self.x_max,
                                                     D, Xk, i, C, self.f[m], u_apply[i], dt,
                                                     g, lbg, ubg, shrink[i], x_plot, B, J, x_ref,theta)#F1(x0=Xk, p=Uk, y=yk)#, DT=DTk)

                for ig in range(self.ng):
                    g   += [self.gfcn(Xk, x_ref,    u_apply[i])[ig]*shrink[i]]
                    lbg += [-inf]
                    ubg += [0.]
                X_his = vertcat(X_his,Xk.T)
            X_models += [X_his]
        J += -Criteria.HR(X_models)

        # J+= self.Obj_D(Xk, x_ref,  Uk) * shrink[i]
        # J +=  self.Obj_M(Xk, x_ref,  Uk)
        if self.penalize_u:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [U_past]
            p += [shrink]
            p += [thetas]
            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}
        else:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [shrink]
            p += [thetas]

            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}

        trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot)], ['w'], ['x','u'])



        solver = nlpsol('solver', 'ipopt', prob, self.opts)  # 'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
        self.solver, self.trajectories, self.w0, self.lbw, self.ubw, self.lbg, self.ubg = \
            solver, trajectories, w0, lbw, ubw, lbg, ubg

        return solver, trajectories, w0, lbw, ubw, lbg, ubg

    def MPC_construct_ukf(self):
        """
        ODEeq: Is the system of ODE
        gcn  : Is the inequality constraints
        ObjM : Is the mayer term of DO
        ObjL : Is the Lagrange term of DO
        Obj_D: The a discretized objective for each time
        :return:
        """
        N = self.N
        dc = self.dc
        dt = self.dt
        NoModels = self.NoModels

        C, D, B = construct_polynomials_basis(dc, 'radau')
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        Ts = []
        t = 0
        # "Lift" initial conditions
        x_plot = []

        u_plot = []
        X_models = []
        X_0 = SX.sym('p_x', self.nd)  # This is a parameter that defines the Initial Conditions
        shrink   = SX.sym('p_shrink', self.N)
        x_ref    = SX.sym('p_ref', self.n_ref)
        thetas   = SX.sym('p_thetas', self.ntheta * NoModels)

        S_thetas = []
        for m in range(NoModels):
            S_thetas += [SX.sym('p_S_thetas_'+str(m), self.ntheta * self.ntheta)]

        if self.penalize_u:
            U_past = SX.sym('p_u', self.nu)  # This is a parameter that defines the Initial Conditions
            prev = U_past
        u_apply = []
        for m in range(NoModels):

            # Create a square matrix for the S_theta

            S_theta = SX.sym('S_theta', self.ntheta**2)
            w += [S_theta]
            lbw += [*(0 * np.ones([self.ntheta**2]))]
            # [*(0.8*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            #                                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(0*np.ones([self.ntheta]))]
            ubw += [*(1 * np.ones([self.ntheta**2]))]
            # [*(1.1*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(500*np.ones([self.ntheta]))]
            w0 += [*(0 * np.ones([self.ntheta**2]))]
            # [*(np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]
            g += [S_theta - S_thetas[m]]
            lbg += [*np.zeros([self.ntheta**2])]
            ubg += [*np.zeros([self.ntheta**2])]

            S_theta_reshaped = S_theta.reshape((self.ntheta, self.ntheta))

            Xk = SX.sym('X0', self.nd)
            w += [Xk]
            lbw += [*self.x_min]
            ubw += [*self.x_max]
            w0 += [*self.x_min]
            g += [Xk - X_0]
            lbg += [*np.zeros([self.nd])]
            ubg += [*np.zeros([self.nd])]
            x_plot += [Xk]
            X_his = []

            theta = SX.sym('theta', self.ntheta)
            w += [theta]
            lbw += [*(0 * np.ones([self.ntheta]))]
            # [*(0.8*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            #                                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(0*np.ones([self.ntheta]))]
            ubw += [*(1000 * np.ones([self.ntheta]))]
            # [*(1.1*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(500*np.ones([self.ntheta]))]
            w0 += [*(100 * np.ones([self.ntheta]))]
            # [*(np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]
            g += [theta - thetas[m * self.ntheta:(m + 1) * (self.ntheta)]]
            lbg += [*np.zeros([self.ntheta])]
            ubg += [*np.zeros([self.ntheta])]
            S = SX(0.001*np.eye(self.nd))
            for i in range(N):
                # Formulate the NLP
                # New NLP variable for the control
                if m == 0:
                    Uk = SX.sym('U_' + str(i), self.nu)
                    if self.penalize_u:
                        J += (Uk - prev).T @ self.R @ (Uk - prev) * shrink[i]
                        prev = Uk
                    w += [Uk]
                    lbw += [*self.u_min]
                    ubw += [*self.u_max]
                    w0 += [*(self.u_min)]
                    u_plot += [Uk]
                    u_apply += [Uk]

                # Integrate till the end of the interval
                auxiliary_vars = dc, self.nd, w, lbw, ubw, w0, \
                self.x_min, self.x_max, \
                D, i, C, dt, g, lbg, ubg, \
                shrink[i], x_plot, B, x_ref

                if N<1:

                    Xk, S, w, lbw, ubw, w0, g, lbg, ubg, _, x_plot = self.ukf1(self.f[m], Xk, S, theta, S_theta_reshaped,
                          self.hmeas[m], self.hmeas[m](Xk, u_apply[i]), self.Q, self.S_exp, u_apply[i], auxiliary_vars)
                else:
                    Xk, _, w, lbw, ubw, w0, g, lbg, ubg, _, x_plot = self.ukf1(self.f[m], Xk, S, theta, S_theta_reshaped,
                          self.hmeas[m], self.hmeas[m](Xk, u_apply[i]), self.Q, self.S_exp, u_apply[i], auxiliary_vars)

                for ig in range(self.ng):
                    g   += [self.gfcn(Xk, x_ref, u_apply[i])[ig]*shrink[i]]
                    lbg += [-inf]
                    ubg += [0.]
                X_his = vertcat(X_his,Xk.T)
            X_models += [X_his]
        J += -Criteria.HR(X_models)

        # J+= self.Obj_D(Xk, x_ref,  Uk) * shrink[i]
        # J +=  self.Obj_M(Xk, x_ref,  Uk)
        if self.penalize_u:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [U_past]
            p += [shrink]
            p += [thetas]
            for i in range(self.NoModels):
                p += [S_thetas[i]]

            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}
        else:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [shrink]
            p += [thetas]
            for i in range(self.NoModels):
                p += [S_thetas[i]]

            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}


        trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot)], ['w'], ['x','u'])



        solver = nlpsol('solver', 'ipopt', prob, self.opts)  # 'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
        self.solver, self.trajectories, self.w0, self.lbw, self.ubw, self.lbg, self.ubg = \
            solver, trajectories, w0, lbw, ubw, lbg, ubg

        return solver, trajectories, w0, lbw, ubw, lbg, ubg


    def MPC_construct_ukf_no_thetas(self, thetas, S_thetas):
        """
        ODEeq: Is the system of ODE
        gcn  : Is the inequality constraints
        ObjM : Is the mayer term of DO
        ObjL : Is the Lagrange term of DO
        Obj_D: The a discretized objective for each time
        :return:
        """
        N = self.N
        dc = self.dc
        dt = self.dt
        NoModels = self.NoModels

        C, D, B = construct_polynomials_basis(dc, 'radau')
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        Ts = []
        t = 0
        # "Lift" initial conditions
        x_plot = []

        u_plot = []
        X_models = []
        S_models = []

        X_0 = SX.sym('p_x', self.nd)  # This is a parameter that defines the Initial Conditions
        shrink   = SX.sym('p_shrink', self.N)
        x_ref    = SX.sym('p_ref', self.n_ref)
        # thetas   = SX.sym('p_thetas', self.ntheta * NoModels)

        # S_thetas = []
        # for m in range(NoModels):
        #     S_thetas += [SX.sym('p_S_thetas_'+str(m), self.ntheta * self.ntheta)]

        if self.penalize_u:
            U_past = SX.sym('p_u', self.nu)  # This is a parameter that defines the Initial Conditions
            prev = U_past
        u_apply = []
        for m in range(NoModels):

            # Create a square matrix for the S_theta

            S_theta = S_thetas[m]

            S_theta_reshaped = SX(S_theta.reshape((self.ntheta, self.ntheta)))

            Xk = SX.sym('X0', self.nd)
            w += [Xk]
            lbw += [*self.x_min]
            ubw += [*self.x_max]
            w0 += [*(self.x_min)]
            g += [Xk - X_0]
            lbg += [*np.zeros([self.nd])]
            ubg += [*np.zeros([self.nd])]
            x_plot += [Xk]
            X_his = []
            S_his = []
            # theta = SX.sym('theta', self.ntheta)
            # w += [theta]
            # lbw += [*(0 * np.ones([self.ntheta]))]
            # # [*(0.8*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            # #                                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(0*np.ones([self.ntheta]))]
            # ubw += [*(1000 * np.ones([self.ntheta]))]
            # # [*(1.1*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            # #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(500*np.ones([self.ntheta]))]
            # w0 += [*(100 * np.ones([self.ntheta]))]
            # # [*(np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            # #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]
            theta = SX(thetas[m])# * self.ntheta:(m + 1) * (self.ntheta)])
            # lbg += [*np.zeros([self.ntheta])]
            # ubg += [*np.zeros([self.ntheta])]
            S = SX(0.0000*np.eye(self.nd))
            for i in range(N):
                # Formulate the NLP
                # New NLP variable for the control
                if m == 0:
                    Uk = SX.sym('U_' + str(i), self.nu)
                    if self.penalize_u:
                        J += (Uk - prev).T @ self.R @ (Uk - prev) * shrink[i]
                        prev = Uk
                    w += [Uk]
                    lbw += [*self.u_min]
                    ubw += [*self.u_max]
                    w0 += [*((self.u_min+self.u_max)/2)]
                    u_plot += [Uk]
                    u_apply += [Uk]

                # Integrate till the end of the interval
                auxiliary_vars = dc, self.nd, w, lbw, ubw, w0, \
                self.x_min, self.x_max, \
                D, i, C, dt, g, lbg, ubg, \
                shrink[i], [], B, x_ref

                if i<4:

                    Xk, S, w, lbw, ubw, w0, g, lbg, ubg, _, _ = self.ukf1_regular(self.f[m], Xk, S, theta, S_theta_reshaped,
                          self.hmeas[m], self.hmeas[m](Xk, u_apply[i]), self.Q, self.S_exp, u_apply[i], auxiliary_vars)
                else:
                    Xk, _, w, lbw, ubw, w0, g, lbg, ubg, _, _ = self.ukf1_regular(self.f[m], Xk, S, theta, S_theta_reshaped,
                          self.hmeas[m], self.hmeas[m](Xk, u_apply[i]), self.Q, self.S_exp, u_apply[i], auxiliary_vars)
                x_plot += [Xk]
                for ig in range(self.ng):
                    g   += [self.gfcn(Xk, x_ref, u_apply[i])[ig]*shrink[i]]
                    lbg += [-inf]
                    ubg += [0.]
                X_his = vertcat(X_his,Xk.T)
                S_his += [S]

            X_models += [X_his]
            S_models += [S_his]

        J += -Criteria.BF(X_models, S_models, 0.000001*np.eye(self.nd))

        # J+= self.Obj_D(Xk, x_ref,  Uk) * shrink[i]
        # J +=  self.Obj_M(Xk, x_ref,  Uk)
        if self.penalize_u:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [U_past]
            p += [shrink]
            # p += [thetas]
            # for i in range(self.NoModels):
            #     p += [S_thetas[i]]

            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}
        else:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [shrink]
            # p += [thetas]
            # for i in range(self.NoModels):
            #     p += [S_thetas[i]]

            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}


        trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot)], ['w'], ['x','u'])



        solver = nlpsol('solver', 'ipopt', prob, self.opts)  # 'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
        self.solver, self.trajectories, self.w0, self.lbw, self.ubw, self.lbg, self.ubg = \
            solver, trajectories, w0, lbw, ubw, lbg, ubg

        return solver, trajectories, w0, lbw, ubw, lbg, ubg

    def MPC_construct_ukf_thetas(self, thetas, S_thetas):
        """
        ODEeq: Is the system of ODE
        gcn  : Is the inequality constraints
        ObjM : Is the mayer term of DO
        ObjL : Is the Lagrange term of DO
        Obj_D: The a discretized objective for each time
        :return:
        """
        N = self.N
        dc = self.dc
        dt = self.dt
        NoModels = self.NoModels

        C, D, B = construct_polynomials_basis(dc, 'radau')
        w = []
        w0 = []
        lbw = []
        ubw = []
        J = 0
        g = []
        lbg = []
        ubg = []
        Ts = []
        t = 0
        # "Lift" initial conditions
        x_plot = []

        u_plot = []
        X_models = []
        S_models = []

        X_0 = SX.sym('p_x', self.nd)  # This is a parameter that defines the Initial Conditions
        shrink   = np.ones(self.N)#SX.sym('p_shrink', self.N)
        x_ref    = SX.sym('p_ref', self.n_ref)
        # thetas   = SX.sym('p_thetas', self.ntheta * NoModels)

        # S_thetas = []
        # for m in range(NoModels):
        #     S_thetas += [SX.sym('p_S_thetas_'+str(m), self.ntheta * self.ntheta)]

        if self.penalize_u:
            U_past = SX.sym('p_u', self.nu)  # This is a parameter that defines the Initial Conditions
            prev = U_past
        u_apply = []
        S_plot = []
        # K = SX.sym('K_a_', self.nu*self.nd)
        # w += [K]
        # lbw += [*(-1000*np.ones(self.nu*self.nd))]
        # ubw += [*(1000*np.ones(self.nu*self.nd))]
        # w0 += [*(np.zeros(self.nu*self.nd))]
        # K_sq = K.reshape((self.nu,self.nd))
        for m in range(NoModels):

            # Create a square matrix for the S_theta

            S_theta = S_thetas[m]

            S_theta_reshaped = SX(S_theta.reshape((self.ntheta, self.ntheta)))

            Xk = SX.sym('X0', self.nd)
            w += [Xk]
            lbw += [*self.x_min]
            ubw += [*self.x_max]
            w0 += [*(self.x_min)]
            g += [Xk - X_0]
            lbg += [*np.zeros([self.nd])]
            ubg += [*np.zeros([self.nd])]
            x_plot += [Xk]
            X_his = []
            S_his = []
            # theta = SX.sym('theta', self.ntheta)
            # w += [theta]
            # lbw += [*(0 * np.ones([self.ntheta]))]
            # # [*(0.8*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            # #                                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(0*np.ones([self.ntheta]))]
            # ubw += [*(1000 * np.ones([self.ntheta]))]
            # # [*(1.1*np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            # #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]#[*(500*np.ones([self.ntheta]))]
            # w0 += [*(100 * np.ones([self.ntheta]))]
            # # [*(np.array( [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
            # #                 2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]))]
            theta = SX(thetas[m])# * self.ntheta:(m + 1) * (self.ntheta)])
            # lbg += [*np.zeros([self.ntheta])]
            # ubg += [*np.zeros([self.ntheta])]
            S = SX(0.0000*np.eye(self.nd))
            for i in range(N):
                # Formulate the NLP
                # New NLP variable for the control
                if m == 0:
                    Uk = SX.sym('U_' + str(i), self.nu)
                    if self.penalize_u:
                        J += (Uk - prev).T @ self.R @ (Uk - prev) * shrink[i]
                        prev = Uk
                    w += [Uk]
                    lbw += [*self.u_min]
                    ubw += [*self.u_max]
                    w0 += [*((self.u_min+self.u_max)/2)]
                    u_plot += [Uk]

                    u_apply += [Uk]# + K_sq @ (Xk)]

                # Integrate till the end of the interval
                auxiliary_vars = dc, self.nd, w, lbw, ubw, w0, \
                self.x_min, self.x_max, \
                D, i, C, dt, g, lbg, ubg, \
                shrink[i], [], B, x_ref

                if i<N:

                    Xk, S, w, lbw, ubw, w0, g, lbg, ubg, _, _ = self.ukf1_regular(self.f[m], Xk, S, theta, S_theta_reshaped,
                          self.hmeas[m], self.hmeas[m](Xk, u_apply[i]), self.Q, self.S_exp, u_apply[i], auxiliary_vars)
                else:
                    Xk, _, w, lbw, ubw, w0, g, lbg, ubg, _, _ = self.ukf1_regular(self.f[m], Xk, S, theta, S_theta_reshaped,
                          self.hmeas[m], self.hmeas[m](Xk, u_apply[i]), self.Q, self.S_exp, u_apply[i], auxiliary_vars)
                x_plot += [Xk]
                for ig in range(self.ng):
                    g   += [self.gfcn(Xk, x_ref, u_apply[i])[ig]]# + 4.35*sqrt(S[1,1])]
                    lbg += [-inf]
                    ubg += [0.]
                X_his = vertcat(X_his,(Xk).T)#/[14,800,1]
                S_his += [S]
                if m ==0:
                    S_plot+= [S.reshape((self.nd**2,1))]
            X_models += [X_his]
            S_models += [S_his]

        J += -log(Criteria.BF(X_models, S_models, self.S_exp)+1e-7)#(Criteria.AW(X_models, S_models, self.S_exp)+1e-7)

        # J+= self.Obj_D(Xk, x_ref,  Uk) * shrink[i]
        # J +=  self.Obj_M(Xk, x_ref,  Uk)
        if self.penalize_u:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [U_past]
            # p += [thetas]
            # for i in range(self.NoModels):
            #     p += [S_thetas[i]]

            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}
        else:
            p  = []
            p += [X_0]
            p += [x_ref]
            # p += [thetas]
            # for i in range(self.NoModels):
            #     p += [S_thetas[i]]

            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}


        trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot), horzcat(*S_plot)], ['w'], ['x','u','S'])



        solver = nlpsol('solver', 'ipopt', prob, self.opts)  # 'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
        self.solver, self.trajectories, self.w0, self.lbw, self.ubw, self.lbg, self.ubg = \
            solver, trajectories, w0, lbw, ubw, lbg, ubg

        return solver, trajectories, w0, lbw, ubw, lbg, ubg


    def solve_MPC(self, x, thetas, ref=None, u=None, t=0., S_theta=None):

        if self.n_ref>0:
            p0 = np.concatenate((x, np.array([ref]).reshape((-1,))))
        else:
            p0 = x

        if self.shrinking_horizon:
            if t==0.:
                shrink = np.ones([self.N])
                self.steps = self.N
            else:
                shrink = np.concatenate((np.ones([self.steps]), np.zeros([self.N-self.steps])))
        else:
            shrink = np.ones([self.N])

        if self.penalize_u:
            p0 = np.concatenate((p0,u))

        theta          = np.array(thetas)
        theta_reshaped = np.reshape(theta, self.ntheta*self.NoModels)
        p0             = np.concatenate((p0, shrink, theta_reshaped))
        #
        # # Add the parametric unc in the problem
        if self.ukf:
            for i in range(self.NoModels):
                S_theta_single = S_theta[i].reshape((self.ntheta**2))
                p0 = np.concatenate((p0, S_theta_single))


        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg,
                     p=p0)
        w_opt = sol['x'].full().flatten()
        x_opt, u_opt = self. trajectories(sol['x'])
        if self.solver.stats()['return_status'] != 'Solve_Succeeded':
            print('Opt failed')
        if self.shrinking_horizon:
            self.steps += - 1
        self.obj = sol['f'].full().flatten()
        return u_opt, x_opt, w_opt

    def solve_MPC_unc(self, x, ref=None, u=None, t=0.):

        if self.n_ref>0:
            p0 = np.concatenate((x, np.array([ref]).reshape((-1,))))
        else:
            p0 = x

        # if self.shrinking_horizon:
        #     if t==0.:
        #         shrink = np.ones([self.N])
        #         self.steps = self.N
        #     else:
        #         shrink = np.concatenate((np.ones([self.steps]), np.zeros([self.N-self.steps])))
        # else:
        #     shrink = np.ones([self.N])

        if self.penalize_u:
            p0 = np.concatenate((p0,u))

        # theta          = np.array(thetas)
        # theta_reshaped = np.reshape(theta, self.ntheta*self.NoModels)
        #p0             = np.concatenate((p0))#, theta_reshaped))
        #
        # # Add the parametric unc in the problem
        # if self.ukf:
        #     for i in range(self.NoModels):
        #         S_theta_single = S_theta[i].reshape((self.ntheta**2))
        #         p0 = np.concatenate((p0, S_theta_single))


        sol = self.solver(x0=self.w0, lbx=self.lbw, ubx=self.ubw, lbg=self.lbg, ubg=self.ubg,
                     p=p0)
        w_opt = sol['x'].full().flatten()
        x_opt, u_opt, S_opt = self. trajectories(sol['x'])
        if self.solver.stats()['return_status'] != 'Solve_Succeeded':
            print('Opt failed')
        # if self.shrinking_horizon:
        #     self.steps += - 1
        self.obj = sol['f'].full().flatten()
        return u_opt, x_opt, w_opt, S_opt




    def perform_orthogonal_collocation(self,d, nx, w, lbw, ubw, w0, lbx, ubx, D, Xk, s, C, f, Uk,
                                       h, g, lbg, ubg, shrink, x_plot, B, J, x_ref,unc_theta):
        """

        :return:
        """
        Xc = []

        for j in range(d):
            Xkj = SX.sym('X_' + str(s) + '_' + str(j), nx)
            Xc += [Xkj]
            w += [Xkj]
            lbw.extend(lbx)
            ubw.extend(ubx)
            w0.extend((self.x_min*1.2))
            x_plot+= [Xkj]
        # Loop over collocation points
        Xk_end = D[0] * Xk

        for j in range(1, d + 1):
            # Expression for the state derivative at the collocation point
            xp = C[0, j] * Xk
            for r in range(d):
                xp = xp + C[r + 1, j] * Xc[r]

            # Append collocation equations
            fj = f(Xc[j - 1], Uk, unc_theta) * shrink  #
            g += [(h * fj - xp)]
            lbg.extend([-1e-8] * nx)
            ubg.extend([1e-8] * nx)
            if not(self.ukf):
                for ig in range(self.ng):
                    g   += [self.gfcn(Xc[j-1], x_ref, Uk)[ig]*shrink]
                    lbg += [-inf]
                    ubg += [0.]
            # Add contribution to the end state
            Xk_end = Xk_end + D[j] * Xc[j - 1]
        #            if int(j1) < np.shape(t_meas)[0]:
        #                if np.real(k * T / N) == t_meas[j1]:
        #                    count[k] = 1
        #                    j1 += 1
        # Add contribution to quadrature function
            qj = 0.#self.Obj_L(Xc[j - 1], x_ref,Uk) * shrink  #

            J += B[j]*qj*h

        # New NLP variable for state at end of interval
        Xk = SX.sym('X_' + str(s + 1), nx)
        w += [Xk]
        lbw.extend(lbx)
        ubw.extend(ubx)
        w0.extend((self.x_min*1.2))

        # Add equality constraint
        g += [Xk_end - Xk]
        lbg.extend([0.] * nx)
        ubg.extend([0.] * nx)

        return w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot, J

    def ukf1(self, fstate, x, S, theta, S_theta, hmeas, z, Q, R, u, auxiliary_vars):

        dc, nd, w, lbw, ubw, w0,\
        x_min, x_max,\
        D, i, C, dt,g, lbg, ubg, \
        shrink, x_plot, B, x_ref = auxiliary_vars


        x_aug = vertcat(x, theta)
        S_aug = diagcat(S, S_theta)


        L = max(np.shape(x_aug))  # 2*len(x)+1
        m = z.shape[0]
        alpha = 1e-3
        ki = 0
        beta = 2
        lambda1 = alpha ** 2 * (L + ki) - L
        c = L + lambda1
        Wm = np.zeros(1 + 2 * L)
        Wm[0] = lambda1 / c
        Wm[1:] = 0.5 / c + np.zeros([1, 2 * L])
        Wc = Wm.copy()
        Wc[0] = Wc[0] + (1 - alpha ** 2 + beta)
        c = np.sqrt(c)
        # S[-4:,-4:]= 0.999**0.5 * S[-4:,-4:]
        X = self.sigmas(x_aug, S_aug, c)

        x1, X1, S1, X2, w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot = self.ut_with_orthogonal_collocation(
            fstate, X[:self.nd,:], X[self.nd:,:], Wm, Wc, nd, Q, u, auxiliary_vars)

        z1, Z1, S2, Z2 = self.ut(hmeas, X1, Wm, Wc, m, R, u)


        P12 = X2 @ np.diagflat(Wc) @ Z2.T
        # P12         = mtimes(mtimes(X2,np.diagflat(Wc)),Z2.T)
        K = mtimes(mtimes(P12, pinv(S2)), pinv(S2).T)  # .full()##P12 @np.linalg.pinv(S2)**2

        # K           = np.dot(np.dot(P12, np.linalg.pinv(S2.T)),np.linalg.pinv(S2)) #np.linalg.lstsq(S2.T,np.linalg.lstsq(S2, P12.T)[0].T)[0]
        # K1          = np.linalg.lstsq(S2.T, np.linalg.lstsq(S2, P12.T)[0].T)[0]
        x = x1 + K @ (z - z1)
        U = K @ S2.T
        for i in range(np.shape(z)[0]):
            S1 = self.cholupdate(S1, U[:, i], '-')
        S = S1

        return x, S, w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot


    def ukf1_regular(self, fstate, x, S, theta, S_theta, hmeas, z, Q, R, u, auxiliary_vars):

        dc, nd, w, lbw, ubw, w0,\
        x_min, x_max,\
        D, i, C, dt,g, lbg, ubg, \
        shrink, x_plot, B, x_ref = auxiliary_vars


        x_aug = vertcat(x, theta)
        S_aug = diagcat(S, S_theta)


        L = max(np.shape(x_aug))  # 2*len(x)+1
        m = z.shape[0]
        alpha = 1e-3
        ki = 0
        beta = 2
        lambda1 = 3 - L  # L*(alpha**2-1)#alpha**2*(L+ki)-L

        c = L + lambda1
        Wm = np.zeros(1 + 2 * L)
        Wm[0] = lambda1 / c
        Wm[1:] = 0.5 / c + np.zeros([1, 2 * L])
        Wc = Wm.copy()
        Wc[0] = Wc[0]# + (1 - alpha ** 2 + beta)
        #c = np.sqrt(c)
        # S[-4:,-4:]= 0.999**0.5 * S[-4:,-4:]
        X = self.sigmas_regular(x_aug, S_aug, c)

        x1, X1, S1, X2, w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot = self.ut_with_orthogonal_collocation_regular(
            fstate, X[:self.nd,:], X[self.nd:,:], Wm, Wc, nd, Q, u, auxiliary_vars)


        return x1, S1, w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot

    def ut_regular(self,f, X, Wm, Wc, n, R, u):

        L = X.shape[1]
        y = SX(np.zeros([n, ]))
        Y = SX(np.zeros([n, L]))
        for k in range(L):
            Y[:, k] = (f((X[:, k]), (u)))
            y += Wm[k] * Y[:, k]


        Sum_mean_matrix_m = []
        for i in range(L):
            Sum_mean_matrix_m = horzcat(Sum_mean_matrix_m, y)
        Y1 = (Y - Sum_mean_matrix_m)
        res = Y1 @ np.sqrt(np.diagflat(abs(Wc)))
        a = horzcat((Y1 @ sqrt(np.diagflat(abs(Wc))))[:, 1:L], SX(R)).T

        _, S = qr(a)
        if Wc[0] < 0:
            S1 = self.cholupdate(S, res[:, 0], '-')
        else:
            S1 = self.cholupdate(S, res[:, 0], '+')
        S = S1
        # P=Y1@np.diagflat(Wc) @ Y1.T+R
        return y, Y, S, Y1

    def ut_with_orthogonal_collocation_regular(self, f, X, theta, Wm, Wc, n, R, u, auxiliary_vars):


        dc, nd, w, lbw, ubw, w0,\
        x_min, x_max,\
        D, i, C, dt,g, lbg, ubg, \
        shrink, x_plot, B, x_ref = auxiliary_vars


        L = X.shape[1]
        y = SX(np.zeros([n, ]))
        Y = SX(np.zeros([n, L]))
        for k in range(L):
            w, lbw, ubw, w0, g, lbg, ubg, Xk, _, _ = self.perform_orthogonal_collocation(
                dc, self.nd, w, lbw, ubw, w0,
                self.x_min, self.x_max,
                D, X[:,k], i, C, f, u, dt,
                g, lbg, ubg, shrink, x_plot, B, 0, x_ref, theta[:,k])  # F1(x0=Xk, p=Uk, y=yk)#, DT=DTk)

            Y[:, k] = Xk
            y += Wm[k] * Y[:, k]

        Sum_mean_matrix_m = []
        for i in range(L):
            Sum_mean_matrix_m = horzcat(Sum_mean_matrix_m, y)
        Y1  = (Y - Sum_mean_matrix_m)
        # res = Y1 @ np.sqrt(np.diagflat(abs(Wc)))
        S   = Wc[0]*(Y[:,[0]]-y)@(Y[:,[0]]-y).T#Y1[:,[0]] @ Y1[:,[0]].T
        for i in range(1,L):
            S   += Wc[i]*(Y[:,[i]]-y)@(Y[:,[i]]-y).T#Wc[i]*Y1[:,[i]] @ Y1[:,[i]].T
        S +=1e-7*SX(np.eye(self.nd))

        # Sk = SX.sym('X0', self.nd**2)
        # w += [Sk]
        # lbS = -20*np.ones([self.nd, self.nd])+20*np.eye(self.nd)
        # lbw += [*lbS.reshape((self.nd**2,1))]
        # ubw += [*(np.zeros([self.nd**2])+20)]
        # w0 += [*((1e-7)*np.eye(self.nd).reshape((self.nd**2,1)))]
        # g += [Sk - S.reshape((self.nd**2,1))]
        # lbg += [*np.zeros([self.nd**2])]
        # ubg += [*np.zeros([self.nd**2])]
        #Sk.reshape((self.nd, self.nd))
        return y, Y, S, Y1, w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot

    def ut(self,f, X, Wm, Wc, n, R, u):

        L = X.shape[1]
        y = SX(np.zeros([n, ]))
        Y = SX(np.zeros([n, L]))
        for k in range(L):
            Y[:, k] = (f((X[:, k]), (u)))
            y += Wm[k] * Y[:, k]


        Sum_mean_matrix_m = []
        for i in range(L):
            Sum_mean_matrix_m = horzcat(Sum_mean_matrix_m, y)
        Y1 = (Y - Sum_mean_matrix_m)
        res = Y1 @ np.sqrt(np.diagflat(abs(Wc)))
        a = horzcat((Y1 @ sqrt(np.diagflat(abs(Wc))))[:, 1:L], SX(R)).T

        _, S = qr(a)
        if Wc[0] < 0:
            S1 = self.cholupdate(S, res[:, 0], '-')
        else:
            S1 = self.cholupdate(S, res[:, 0], '+')
        S = S1
        # P=Y1@np.diagflat(Wc) @ Y1.T+R
        return y, Y, S, Y1

    def ut_with_orthogonal_collocation(self, f, X, theta, Wm, Wc, n, R, u, auxiliary_vars):


        dc, nd, w, lbw, ubw, w0,\
        x_min, x_max,\
        D, i, C, dt,g, lbg, ubg, \
        shrink, x_plot, B, x_ref = auxiliary_vars


        L = X.shape[1]
        y = SX(np.zeros([n, ]))
        Y = SX(np.zeros([n, L]))
        for k in range(L):
            w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot, _ = self.perform_orthogonal_collocation(
                dc, self.nd, w, lbw, ubw, w0,
                self.x_min, self.x_max,
                D, X[:,k], i, C, f, u, dt,
                g, lbg, ubg, shrink, x_plot, B, 0, x_ref, theta[:,k])  # F1(x0=Xk, p=Uk, y=yk)#, DT=DTk)

            Y[:, k] = Xk
            y += Wm[k] * Y[:, k]

        Sum_mean_matrix_m = []
        for i in range(L):
            Sum_mean_matrix_m = horzcat(Sum_mean_matrix_m, y)
        Y1 = (Y - Sum_mean_matrix_m)
        res = Y1 @ np.sqrt(np.diagflat(abs(Wc)))
        a = horzcat((Y1 @ sqrt(np.diagflat(abs(Wc))))[:, 1:L], SX(R)).T

        _, S = qr(a)
        if Wc[0] < 0:
            S1 = self.cholupdate(S, res[:, 0], '-')
        else:
            S1 = self.cholupdate(S, res[:, 0], '+')
        S = S1
        # P=Y1@np.diagflat(Wc) @ Y1.T+R
        return y, Y, S, Y1, w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot



    def cholupdate(self,R, x, sign):
        p = max(np.shape(x))
        x = x.T
        for k in range(p):
            if sign == '+':
                r = sqrt(R[k, k] ** 2 + x[k] ** 2)
            elif sign == '-':
                r = sqrt(R[k, k] ** 2 - x[k] ** 2)
            c = r / R[k, k]
            s = x[k] / R[k, k]
            R[k, k] = r
            if k < p - 1:
                if sign == '+':
                    R[k, k + 1:p] = (R[k, k + 1:p] + s * x[k + 1:p]) / c
                elif sign == '-':
                    R[k, k + 1:p] = (R[k, k + 1:p] - s * x[k + 1:p]) / c

                x[k + 1:p] = c * x[k + 1:p] - s * R[k, k + 1:p]
        return R

    def sigmas(self,x, S, c):

        A = chol(c * S.T).T
        # Y = x[:,np.ones([1,len(x)])]
        n = x.shape[0]
        X =  horzcat(x.reshape((n, 1)), x.reshape((n, 1)) + A, x.reshape((n, 1)) - A)
        return X

    def sigmas_regular(self,x, S, c):

        A = chol(c * S.T).T
        # Y = x[:,np.ones([1,len(x)])]
        n = x.shape[0]
        X =  horzcat(x.reshape((n, 1)), x.reshape((n, 1)) + A, x.reshape((n, 1)) - A)
        return X

class cosntract_history:
    def __init__(self, model, N, store_u = True, set_point0 = 0.):
        #Define self vars
        dt, x0, Lsolver, c_code, specifications = model.specifications()

        xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, x_min, x_max, states, \
        algebraics, inputs, nd, na, nu, n_ref, nmp, modparval, ng, gfcn, Obj_M, \
        Obj_L, Obj_D, R = model.DAE_system()

        self.model   = model           # The model defined in terms of casadi
        self.N       = N               # Number of past data
        self.store_u = store_u
        self.nx      = nd
        self.nu      = nu
        self.nsp     = len(set_point0)
        self.u_min   = u_min
        self.u_max   = u_max
        state_0, e_sp0 = x0, x0-set_point0#model.reset(set_point0)
        # initialize history
        history_x = np.array([*state_0]*N).reshape((-1,1))
        history_sp = np.array([*e_sp0]*N).reshape((-1,1))

        if store_u:                  # If u are stored as history (simple RNN structure)
            history_u = np.array([0]*N*self.nu).reshape((-1,1))
            self.history = np.vstack((history_x,history_sp,history_u))
            self.size_states = N * (self.nu + self.nx + self.nsp)
        else:
            self.history = np.vstack((history_x,history_sp))
            self.size_states = N * (self.nx+self.nsp)

        self.history = self.history.reshape((-1,))
        # start counting the past values
        self.past = 1


    def append_history(self, new_state, u, e_sp):

        if self.store_u:
            n = self.nx+self.nu + self.nsp
            self.history[n:] = self.history[:n*(self.N-1)]
            aug_states = np.concatenate((new_state, e_sp, u))
            self.history[:n] = aug_states

        else:
            n = self.nx+ self.nsp
            self.history[n:] = self.history[:n*(self.N-1)]
            aug_states = np.concatenate((new_state, e_sp))

            self.history[:n] = aug_states
        self.past +=1

        return self.history

class Uncertainty_module:
    def __init__(self, Model_def, sensitivity=False):
        self.sensitivity = sensitivity
        dt, _, _, _, _ = Model_def().specifications()
        x, _, u, theta, ODEeq, _, u_min, u_max, x_min, x_max, _, \
        _, _, nx, _, nu, n_ref, ntheta, _, ng, gfcn, \
        Obj_M, Obj_L, Obj_D, R = Model_def().DAE_system(
            uncertain_parameters=True)  # Define the System

        xdot = vertcat(*ODEeq)
        x_p = SX.sym('xp', nx * ntheta)
        if sensitivity:
            xpdot = []
            for i in range(ntheta):
                    xpdot = vertcat(xpdot, jacobian(xdot, x) @ (x_p[nx * i: nx * i + nx])
                                    + jacobian(xdot, theta)[nx * i: nx * i + nx])
            f = Function('f', [x, u, theta, x_p], [xdot,  xpdot],
                                 ['x', 'u', 'theta', 'xp'], ['xdot', 'xpdot'])
        else:

            f = Function('f', [x, u, theta], [xdot],
                                 ['x', 'u', 'theta'], ['xdot'])


        self.f      = f
        self.nu     = nu
        self.nx     = nx
        self.ntheta = ntheta
        self.dt     = dt


    def integrator_model(self, embedded=True, sensitivity=True):
        """
        This function constructs the integrator to be suitable with casadi environment, for the equations of the model
        and the objective function with variable time step.
         inputs: model, sizes
         outputs: F: Function([x, u, dt]--> [xf, obj])
        """
        f      = self.f
        nu     = self.nu
        nx     = self.nx
        ntheta = self.ntheta
        dt     = self.dt
        M = 4  # RK4 steps per interval
        DT = dt#.sym('DT')
        DT1 = DT / M
        X0 = SX.sym('X0', nx)
        U = SX.sym('U', nu)
        theta = SX.sym('theta', ntheta)
        xp0 = SX.sym('xp', np.shape(X0)[0] * np.shape(theta)[0])
        X = X0
        Q = 0
        G = 0
        S = xp0
        if embedded:
            if sensitivity:
                xdot, xpdot = f(X, U, theta, xp0)
                dae = {'x': vertcat(X, xp0), 'p': vertcat(U, theta), 'ode': vertcat(xdot, xpdot)}
                opts = {'tf': dt}  # interval length
                F = integrator('F', 'cvodes', dae, opts)

            else:
                xdot = f(X, U, theta)
                dae  = {'x': vertcat(X), 'p': vertcat(U, theta), 'ode': vertcat(xdot)}
                opts = {'tf': dt}  # interval length
                F = integrator('F', 'cvodes', dae, opts)
        else:
            if sensitivity:

                for j in range(M):
                    k1, k1_a, k1_p = f(X, U, theta, S)
                    k2, k2_a, k2_p = f(X + DT1 / 2 * k1, U, theta, S + DT1 / 2 * k1_p)
                    k3, k3_a, k3_p = f(X + DT1 / 2 * k2, U, theta, S + DT1 / 2 * k2_p)
                    k4, k4_a, k4_p = f(X + DT1 * k3, U, theta, S + DT1 * k3_p)
                    X = X + DT1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                    G = G + DT1 / 6 * (k1_a + 2 * k2_a + 2 * k3_a + k4_a)
                    S = S + DT1 / 6 * (k1_p + 2 * k2_p + 2 * k3_p + k4_p)
                F = Function('F', [X0, U, theta, xp0], [X, G, S], ['x0', 'p', 'theta', 'xp0'], ['xf', 'g', 'xp'])
            else:
                for j in range(M):
                    k1,_ = f(X, U, theta)
                    k2,_ = f(X + DT1 / 2 * k1, U, theta)
                    k3,_ = f(X + DT1 / 2 * k2, U, theta)
                    k4,_ = f(X + DT1 * k3, U, theta)
                    X = X + DT1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
                F = Function('F', [X0, vertcat(U, theta)], [X], ['x0', 'p'], ['xf'])
        self.F = F
        return F

    def simulate_single_step(self,x0, u, theta, xp0):
        self.integrator_model(sensitivity=self.sensitivity)
        Fk = self.F(x0=vertcat(x0, xp0), p=vertcat(u, theta))

        x11 = Fk['xf'][0:self.nx]
        xp1 = Fk['xf'][self.nx:]
        return np.array(x11), np.array(xp1)

    def compute_FIM(self, x_initial, u_apply, theta, S_exp, criterion='D', prior = None):
        if prior is None:
            prior = 0

        steps = u_apply.shape[1]
        N_mc = 1
        Sum_of_FIM = 0
        for k in range(N_mc):
            x0 = x_initial
            xp0 = np.zeros(self.ntheta * self.nx)

            xp_reshaped = xp0.reshape((self.nx, self.ntheta))
            FIM = xp_reshaped.T @ S_exp @ xp_reshaped + prior
            for i in range(steps):
                x11, xp1 = self.simulate_single_step(x0, u_apply[:,i], theta, xp0)
                x0  = x11
                xp0 = xp1
                xp_reshaped = xp0.reshape((self.nx, self.ntheta))
                FIM += xp_reshaped.T @ np.linalg.pinv(S_exp) @ xp_reshaped

            if criterion == 'D':
                metric_FIM = log(det(FIM + 1e-8 * np.eye(self.ntheta)))
            elif criterion == 'A':
                metric_FIM = trace(FIM)

            else:
                raise Exception("Sorry, criterion " + criterion + " to be implemented")

            Sum_of_FIM += metric_FIM

        mean_FIM = Sum_of_FIM/N_mc
        return mean_FIM


    def compute_full_path(self, u_opt, N, x0, S0,
                                             theta, S_theta, z, Q, R):

        u_apply = u_opt.reshape((self.nu, N))
        x_his = np.array([])
        S_his = []

        for i in range(N):
            x1, S1 = self.ukf1_regular(x0, S0, theta, S_theta, z, Q, R, u_apply[:,i])
            x0     = x1
            S0     = S1
            if i == 0:
                x_his  =x1.T

            else:
                x_his  = np.vstack((x_his,x1.T))
            S_his += [S1]

        return x_his, S_his

    def ut_regular(self, X, theta, Wm, Wc, n, u):

        L = X.shape[1]
        y = (np.zeros([n, ]))
        Y = (np.zeros([n, L]))
        for k in range(L):
            if self.FIM_included:
                x11, xp1 = self.simulate_single_step(X[:self.nx,k], u, theta[:,k], X[self.nx:,k])
                Xk = np.hstack((x11, xp1))
            else:
                Xk, _ = self.simulate_single_step(X[:self.nx,k], u, theta[:,k], X[self.nx:,k])

            Y[:, k] = Xk.reshape((-1,))
            y += Wm[k] * Y[:, k]
        y = y.reshape((-1,1))
        Sum_mean_matrix_m = []
        for i in range(L):
            Sum_mean_matrix_m = horzcat(Sum_mean_matrix_m, y)
        Y1  = (Y - Sum_mean_matrix_m)

        S   = Wc[0]*(Y[:,[0]]-y)@(Y[:,[0]]-y).T#Y1[:,[0]] @ Y1[:,[0]].T
        for i in range(1,L):
            S   += Wc[i]*(Y[:,[i]]-y)@(Y[:,[i]]-y).T#Wc[i]*Y1[:,[i]] @ Y1[:,[i]].T
        S +=1e-7*(np.eye(self.nx))


        return y, Y, S, Y1


    def ukf1_regular(self, x, S, theta, S_theta, z, Q, R, u,FIM_included=False):
        self.FIM_included = FIM_included
        x     = x.reshape((-1,1))
        theta = theta.reshape((-1,1))
        x_aug = np.vstack((x, theta))
        S_aug = scipylinalg.block_diag(S, S_theta)


        L = max(np.shape(x_aug))  # 2*len(x)+1
        m = z.shape[0]
        alpha = 1e-3
        ki = 0
        beta = 2
        lambda1 = 3 - L  # L*(alpha**2-1)#alpha**2*(L+ki)-L

        c = L + lambda1
        Wm = np.zeros(1 + 2 * L)
        Wm[0] = lambda1 / c
        Wm[1:] = 0.5 / c + np.zeros([1, 2 * L])
        Wc = Wm.copy()
        Wc[0] = Wc[0]# + (1 - alpha ** 2 + beta)
        #c = np.sqrt(c)
        # S[-4:,-4:]= 0.999**0.5 * S[-4:,-4:]
        X = self.sigmas_regular(x_aug, S_aug, c)

        x1, X1, S1, X = self.ut_regular(X[:self.nx,:], X[self.nx:,:], Wm, Wc, self.nx,  u)


        return x1, S1



    def sigmas_regular(self,x, S, c):

        A = scipylinalg.sqrtm(c * S.T)
        # Y = x[:,np.ones([1,len(x)])]
        n = x.shape[0]
        X =  np.hstack((x.reshape((n, 1)), x.reshape((n, 1)) + A, x.reshape((n, 1)) - A))
        return X


