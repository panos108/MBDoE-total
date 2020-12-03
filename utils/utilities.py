from casadi import *
import numpy as np
csfp = os.path.abspath(os.path.dirname(__file__))
if csfp not in sys.path:
    sys.path.insert(0, csfp)
from utils.OrthogonalCollocation import construct_polynomials_basis

import Criteria
class MBDoE:

    def __init__(self, Model_Def, horizon, collocation_degree = 8, penalize_u=False,
                 ukf=False, theta_unc=None, S_exp = None):

        self.NoModels  = len(Model_Def)
        self.Model_def = []
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
                self.hmeas   += [Function('h1', [xd], [xd])]
                self.S_theta += [theta_unc[i]]


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
        opts["ipopt.print_level"] = 0
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
            raise NotImplementedError




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

        X_0 = SX.sym('p_x', self.nd)  # This is a parameter that defines the Initial Conditions
        shrink = SX.sym('p_shrink', self.N)
        x_ref = SX.sym('p_ref', self.n_ref)
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
            S       = SX.sym(np.eye(self.nd)*1e-8)
            S_theta = SX.sym(self.S_theta[m])
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
                auxiliary_vars = dc, self.nd, w, lbw, ubw, w0, \
                self.x_min, self.x_max, \
                D, i, C, dt, g, lbg, ubg, \
                shrink[i], x_plot, B, x_ref
                self.ukf1(self.f[m], Xk, S, self.thetak[m], S_theta,
                          self.hmeas[m], self.hmeas[m](Xk), self.Q, self.S_exp, u_apply[i], auxiliary_vars)

                w, lbw, ubw, w0, g, lbg, ubg, Xk, x_plot, _ = self.perform_orthogonal_collocation(
                                                     dc, self.nd, w, lbw, ubw, w0,
                                                     self.x_min, self.x_max,
                                                     D, Xk, i, C, self.f[m], u_apply[i], dt,
                                                     g, lbg, ubg, shrink[i], x_plot, B, J, x_ref)#F1(x0=Xk, p=Uk, y=yk)#, DT=DTk)

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
            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}
        else:
            p  = []
            p += [X_0]
            p += [x_ref]
            p += [shrink]
            prob = {'f': J, 'x': vertcat(*w),'p': vertcat(*p), 'g': vertcat(*g)}

        trajectories = Function('trajectories', [vertcat(*w)]
                                , [horzcat(*x_plot), horzcat(*u_plot)], ['w'], ['x','u'])



        solver = nlpsol('solver', 'ipopt', prob, self.opts)  # 'bonmin', prob, {"discrete": discrete})#'ipopt', prob, {'ipopt.output_file': 'error_on_fail'+str(ind)+'.txt'})#
        self.solver, self.trajectories, self.w0, self.lbw, self.ubw, self.lbg, self.ubg = \
            solver, trajectories, w0, lbw, ubw, lbg, ubg

        return solver, trajectories, w0, lbw, ubw, lbg, ubg


    def solve_MPC(self, x, thetas, ref=None, u=None, t=0.):

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
            w0.extend([1.] * nx)
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
        w0.extend([0] * nx)

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
        x_aug = horzcat(x, theta)
        S_aug = diagcat(S, S_theta)


        L = max(np.shape(x))  # 2*len(x)+1
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

        x1, X1, S1, X2 = self.ut(fstate, X, Wm, Wc, nd, Q, u)

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

        return x, S

    def ut(self,f, X, Wm, Wc, n, R, u):

        L = X.shape[1]
        y = SX(np.zeros([n, ]))
        Y = SX(np.zeros([n, L]))
        for k in range(L):
            Y[:, k] = (f(DM(X[:, k]).full(), DM(u).full()))
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

        A = c * S.T
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