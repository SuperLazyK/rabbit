import numpy as np
import scipy.optimize
import math

# euler lagrange equation
# x^* : x_ast
# x' = f(x, u)
# u : input
# U : descrete MPC input
# slack var : v
# C: u <= max_u <=> u^2 + v^2 = max_u^2
# J = phi(xf) + integral (L(x,u) - trick_coeff * v) dt
# lambda for f
# T(t) : T(t0) = 0
# v: slack variable for C to change eq from ineq
# rho: lagrange param for C : (m,)
# l : costate
# m : dim of u
# n : dim of x
def mpc_track_u(f, phi, L, dfdu, dfdx, dLdu, dLdx, x0, t0, t1,
        max_u, T, N, dt, max_u_penalty, dphidx):

    xi = int(1 / dt)
    m = max_u.shape[0]
    n = x0.shape[0]
    max_itr = int((t1 - t0) / dt)

    history_u   = np.zeros((max_itr,m))
    history_v   = np.zeros((max_itr,m))
    history_r   = np.zeros((max_itr,m))
    history_x   = np.zeros((max_itr,n))
    history_l   = np.zeros((max_itr,n))
    history_F0 = np.zeros(max_itr)

    C = lambda u,v : np.array([u[i] ** 2 + v[i] ** 2 - max_u[i] ** 2 for i in range(m)])
    dCdu = lambda u, v : 2 * np.diag(u)
    dCdv = lambda u, v :  -2 * np.diag(v)

    H = lambda x, l, u, v, rho: L(x, u) - max_u_penalty @ v + l @ f(x,u) + rho @ C(u, v)
    dHdx = lambda x, l, u: dLdx(x,u) - l @ dfdx(x,u)

    def dHdu(x, u, l, v, rho):
        return dLdu(x, u) + l @ dfdu(x, u) + rho @ dCdu(u, v)

    def dHduvr_fixed_xl (x, l):
        def _dHduvr(uvr):
            u = uvr[:m]
            v = uvr[m:2*m]
            rho =uvr[2*m:] 
            dHdv = -max_u_penalty + rho @ dCdu(u, v)
            dHdr = C(u, v)
            return np.concatenate([dHdu(x, u, l, v, rho), dHdv, dHdr])
        return _dHduvr

    # step0: calc U0
    l0 = dphidx(x0)
    dHduvr = dHduvr_fixed_xl(x0, l0)
    #sol = scipy.optimize.root(dHduvr, np.zeros(m + m + m), method='hybr') # failure
    sol = scipy.optimize.root(dHduvr, np.zeros(m + m + m), method='lm')
    U0 = sol.x
    U = np.array([U0 for i in range(N)]) # (N) x 3m
    X = np.zeros((N+1, n))
    L = np.zeros((N+1, n))
    L[0] = 0 # L[0] is not used
    x = x0


    for i in range(1,max_itr+1):
        print(f"ITERATION: {i}/{max_itr}")
        dtau = T(i*dt + t0)/N

        # step1: forward calculation for x
        X[0] = x
        for j in range(N):
            X[j+1] = X[j] + f(X[j],U[j,0:m]) * dtau

        # step2: calc last lambda
        # step3: backward calculation for lambda
        L[N] = dphidx(X[-1]) # lambda_N
        for j in range(N,1,-1):
            L[j-1] = L[j] + dHdx(X[j], U[j,0:m], L[j]) * dtau

        # step4: calc new U
        def F(U):
            return np.array([dHdu(X[i], U[i:,:m], L[i+1], U[i:,m:2*m], U[i:,2*m:]) for i in range(N)])

        f = xxxx # TODO
        dU = gmres(f, U) # TODO 
        U = U + dU * dt

        # measure x
        x = x + f(x,U) * dt
        history_u[i] = U[0]
        history_x[i] = x
        history_l[i] = L[0]



    return


def ex8_1():
    t0 = 0
    t1 = 20
    T = lambda t : (1 - exp (-0.5 * (t-t0)))
    f = lambda x, u : np.array([x[1], (1 - x[0]**2 - x[1]**2) * x[1] - x[0] + u])
    dfdu = lambda x, u : np.array([0, 1])
    dfdx = lambda x, u : np.array([[0, -2*x[0]*x[1]-1], [1, 1 - x[0]**2 - 3*x[1]**2]])

    phi = lambda x : x[0]**2 + x[1]**2
    dphidx = lambda x : np.array([2 * x[0], 2 * x[1]])

    L = lambda x, u : 1/2 * (x[0]**2 + x[1]**2 + u**2)
    dLdu = lambda x, u : u
    dLdx = lambda x, u : np.array([x[0], x[1]])

    max_u = np.array([0.5])
    dt = 0.01
    N = 10
    u0 = np.zeros(1)
    x0 = np.array([2, 0])
    max_u_penalty = np.array([0.01])
    us = mpc_track_u(f, phi, L, dfdu, dfdx, dLdu, dLdx, x0, u0, t0, t1,
            max_u, T, N, dt, max_u_penalty, dphidx)

if __name__ == '__main__':
    ex8_1()
