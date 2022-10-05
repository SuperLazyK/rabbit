
# euler lagrange equation
# x^* : x_ast
# x' = f(x, u)
# u <= max_u <=> u^2 + v^2 = max_u^2 with slack var : v
# J = phi(xf) + integral (L(x,u) - trick_coeff * v) dt

def mpc_track_u(f, phi, L, t0, max_u, T, N, trick_coeff=0.01):
    dTau = T / N
    return
