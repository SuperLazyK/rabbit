import numpy as np
import scipy.optimize
from collections import namedtuple
from sympy import lambdify, diff, symbols, Matrix
import sys
#from gmres import gmres
from gmres_autogenu import gmres

def rk4(f, t, x, u, dt):
    k1 = f(x,             u)
    k2 = f(x + dt/2 * k1, u)
    k3 = f(x + dt/2 * k2, u)
    k4 = f(x + dt * k3,   u)
    return x + (k1 + 2*k2 + 2*k3 + k4)/6 * dt

class MPCModel():
    def __init__(self,
            t_, x_, u_,
            f_, phi_, L_,
            max_u, dw
            ):
        # dim
        n = len(x_)
        m = len(u_)
        self.n = len(x_)
        self.m = len(u_)
        #assert dw.size == m
        #assert max_u.size == m

        # symbolic calculation
        x_ = Matrix(x_)
        u_ = Matrix(u_)
        f_ = Matrix(f_)
        l_ = Matrix(symbols('l[0:%d]' %(n))) # lambda co-state symbols
        phi_ = Matrix([phi_])
        #dfdu_ = f_.jacobian(u_)
        #dfdx_ = f_.jacobian(x_)
        v_ = Matrix(symbols('v[0:%d]' %(m))) # slcak var symbols for dummy wait

        #box constraint
        C_ = Matrix([u_[i] ** 2 + v_[i] ** 2 - max_u[i] ** 2 for i in range(m)])
        r_ = Matrix(symbols('r[0:%d]' %(m))) # laglan-multiplier symbols for rho
        #dCdu_ = C_.jacobian(u_)
        #dCdv_ = C_.jacobian(v_)
        H_ = L_ + l_.dot(f_)
        H_ = H_ + r_.dot(C_)
        H_ = H_ - Matrix(dw).dot(v_)
        H_ = L_ + l_.dot(f_) + r_.dot(C_) - Matrix(dw).dot(v_)
        dHdx_ = Matrix([H_]).jacobian(x_)
        #dHdu_ = Matrix([H_]).jacobian(u_)
        dHduvr_ = Matrix([H_]).jacobian(Matrix([u_, v_, r_]))
        dphidx_ = phi_.jacobian(x_)

        # lambdify
        self.lambd_dphidx = lambdify([x_], dphidx_)
        self.lambd_dHduvr = lambdify([x_, l_, u_, v_, r_], dHduvr_)
        self.lambd_dHdx = lambdify([x_, l_, u_], dHdx_) # v, r is not used
        self.lambd_f = lambdify([x_, u_], f_)
        #self.lambd_C = lambdify([u_,v_], C_)
        self.dphidx = lambda x: self.lambd_dphidx(x)[0]
        self.dHduvr = lambda x,l,uvr: self.lambd_dHduvr(x,l,uvr[0:m], uvr[m:2*m], uvr[2*m:3*m])[0]
        self.dHdx   = lambda x,l,u: self.lambd_dHdx(x,l,u)[0]
        self.f      = lambda x,u: self.lambd_f(x,u)[:,0]

    def step(self, t, x, u, dt):
        return rk4(self.f, t, x, u, dt)


class MPCTracker():
    def __init__(self, model, T, N):
        self.model = model
        m = self.model.m
        n = self.model.n
        self.N = N
        self.T = T # prediction time

        #self.x = x0
        self.dUdt = np.zeros((N, 3*m))
        self.X = np.zeros((N+1, n))
        self.L = np.zeros((N+1, n)) # L[0] is not used

    def estimate_init_U(self, x0):
        l0 = self.model.dphidx(x0)
        m = self.model.m
        dHduvr_fixed_xl = lambda uvr: self.model.dHduvr(x0, l0, uvr)
        u_init = scipy.optimize.root(dHduvr_fixed_xl, np.zeros(m + m + m), method='lm').x # for u
        return np.array([u_init for i in range(self.N)]) # (N) x 3m <- uvr

    # 3 time step
    #  1.  dt: for state update (integration)
    #  2.  epsilon: for differential calculation
    #  3.  tau: sampling of prediction horizon
    def track_u(self, x, U, t, dt, epsilon=1.0e-08, k_max=5):
        rde = 1.0 / epsilon
        zeta = 1.0 / dt
        N = self.N
        m = self.model.m

        t1 = t+epsilon
        u0 = U[0,:m]
        U1 = U + self.dUdt * epsilon
        x1 = x + self.model.f(x, u0) * epsilon
        fonc0 = self.F(t, x, U)
        fonc1 = self.F(t1, x1, U)
        fonc2 = self.F(t1, x1, U1)
        b = ((rde - zeta) * fonc0 - fonc2 * rde)

        def funAx(dUdt):
            U1 = U + dUdt.reshape(N,-1) * epsilon
            fonc2 = self.F(t + epsilon, x1, U1)
            return (fonc2 - fonc1).reshape(-1) * rde

        print(b)
        self.dUdt = gmres(funAx, b.reshape(-1), self.dUdt.reshape(-1), k_max=k_max).reshape(N, -1)

        return rk4(self.model.f, t, x, U[0,0:m], dt), U + self.dUdt * dt, fonc0

    def F(self, t, x, U):
        N = self.N
        n = self.model.n
        m = self.model.m
        interval = self.T(t)
        dtau = interval/N
        X = self.X
        L = self.L

        # step1: forward calculation for x
        X[0,:] = x
        for i in range(N):
            X[i+1,:] = X[i,:] + self.model.f(X[i,:],U[i,0:m]) * dtau

        # step2: calc last lambda
        L[N,:] = self.model.dphidx(X[N,:]) # lambda_N

        # step3: backward calculation for lambda
        for i in range(N-1,0,-1):
            L[i,:] = L[i+1,:] + self.model.dHdx(X[i], L[i+1,:], U[i,:m]) * dtau

        return np.array([self.model.dHduvr(X[i], L[i+1], U[i]) for i in range(N)])

