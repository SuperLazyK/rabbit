import numpy as np
import scipy.optimize

# euler lagrange equation
# x^* : x_ast
# x' = f(x, u)
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
def mpc_track_u(f, phi, L, dfdu, dLdu, x0, u0, t0, 
        max_u, T, dtau, xi, dt, max_u_penalty, dphidx=None):

    m = u0.shape[0]
    n = u0.shape[0]

    C = lambda u,v : np.array([u[i] ** 2 + v[i] ** 2 - max_u[i] ** 2 for i in range(m)])
    dCdu = lambda u, v : 2 * np.diag(u)
    dCdv = lambda u, v :  -2 * np.diag(v)

    H = lambda x, l, u, v, rho: L(x, u) - max_u_penalty @ v + l @ f(x,u) + rho @ C(u, v)

    def dHduvr_fixed_xl (x, l):
        def _dHduvr(uvr):
            u = uvr[:m]
            v = uvr[m:2*m]
            rho =uvr[2*m:] 
            dHdu = dLdu(x, u) + l @ dfdu(x, u) + rho @ dCdu(u, v)
            dHdv = -max_u_penalty + rho @ dCdu(u, v)
            dHdr = C(u, v)
            return np.concatenate([dHdu, dHdv, dHdr])
        return _dHduvr

    # failure
    ## step0: calc u0
    #if dphidx is not None:
    #    l0_est = dphidx(x0)
    #    dHduvr = dHduvr_fixed_xl(x0, l0_est)
    #    sol = scipy.optimize.root(dHduvr, np.zeros(m + m + m), method='hybr')
    #    print(sol)
    #    #u0 = sol[0:m]

    ## step1: forward calculation for x

    ## step2: calc last lambda

    ## step3: backward calculation for lambda

    ## step4: calc u

    return


def ex8_1():
    t0 = 0
    T = lambda t : (1 - exp (-0.5 * (t-t0)))
    f = lambda x, u : np.array([x[1], (1 - x[0]**2 - x[1]**2) * x[1] - x[0] + u])
    phi = lambda x : x[0]**2 + x[1]**2
    L = lambda x, u : 1/2 * (x[0]**2 + x[1]**2 + u**2)
    dLdu = lambda x, u : u
    dfdu = lambda x, u : np.array([0, 1])
    dphidx = lambda x : np.array([2 * x[0], 2 * x[1]])
    max_u = np.array([0.5])
    xi = 100
    dt = 0.01
    dtau = 0.01
    u0 = np.zeros(1)
    x0 = np.array([2, 0])
    max_u_penalty = np.array([0.01])
    us = mpc_track_u(f, phi, L, dfdu, dLdu, x0, u0, t0,
            max_u, T, dtau, xi, dt, max_u_penalty, dphidx)

if __name__ == '__main__':
    ex8_1()
