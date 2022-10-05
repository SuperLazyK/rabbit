import numpy as np

# euler lagrange equation
# x^* : x_ast
# x' = f(x, u)
# u <= max_u <=> u^2 + v^2 = max_u^2 with slack var : v
# J = phi(xf) + integral (L(x,u) - trick_coeff * v) dt
# lambda for f
# T(t) : T(t0) = 0

def mpc_track_u(f, phi, L, t0, max_u, T, N, dt=0.01, trick_coeff=0.01):
    dTau = T / N
    return


def ex8_1():
    t0 = 0
    T = lambda t : (1 - exp (-0.5 * (t-t0)))
    f = lambda x, u : np.array(x[1], (1 - x[0]**2 - x[1]**2) * x[1] - x[0] + u)
    phi = lambda x : x[0]**2 + x[1]**2
    L = lambda x, u : 1/2 * (x[0]**2 + x[1]**2 + u**2)
    max_u = np.array([0.5])
    #mpc_track_u()
    print(1)

if __name__ == '__main__':
    ex8_1()
