import numpy as np
import scipy
import scipy.linalg
import sys

def debug(x,y):
    print(x,y)

def gen_fun_Ax(A):
    return lambda x: A @ x

def extendMat(M):
    (m,n) = M.shape
    ret = np.eye(m+1, dtype=M.dtype)
    ret[:m, :n] = M
    return ret

# k = 0,1,2...
def givens(h, extOmega):
    rhosigma = extOmega[-2:,:] @ h
    rho = rhosigma[0]
    sigma = rhosigma[1]
    r = np.linalg.norm(rhosigma)
    return rho/r, sigma/r

def genG(n, s, c):
    ret = np.eye(n+2, dtype=np.float64)
    ret[n:,n:] = np.array([[c, s],[-s, c]])
    return ret

# k = 0,1,2...
def arnoldi(fun_Ax, Q, k, epsilon):
    h = np.zeros(k+2)
    q = fun_Ax(Q[:,k]) # a new Krylov Vector
    h[:k+1] = q @ Q[:,0:k+1]
    q = q - Q[:,0:k+1] @ h[:k+1]
    h[k+1] = np.linalg.norm(q);
    print(h[k+1])
    if abs(h[k+1]) < 1.0e-8:
        return  h, None
    else:
        q = q / h[k + 1]
        return h, q

# x0 = U'
# b = - xi F(U, x, t)
# fun_Ax(x) = dF(U') ~= dF/dU U' + dF/dx x' + dF/dt
# k : restart
def gmres(fun_Ax, b, x0, epsilon=0.001, k=None):

    (m, n) = A.shape

    if k is None:
        k =  min(n, m)

    b_norm = np.linalg.norm(b);

    if b_norm <= epsilon:
        return np.zeros_like(x0)

    invb = 1.0 / b_norm

    e1 = np.zeros(k+1, dtype=np.float64)
    e1[0] = 1

    r = b - fun_Ax(x0)
    r_norm = np.linalg.norm(r);
    beta = r_norm
    q1 = r / r_norm

    if beta * invb <= epsilon:
        return x0

    Q = np.zeros((n, k), dtype=np.float64)
    Q[:,0] = q1

    Htilda = np.zeros((k+1, k), dtype=np.float64)
    Omega  = np.array([[1]], dtype=np.float64)
    Rtilda = np.zeros((k+1 ,k), dtype=np.float64)

    # i = 0..k-1
    # i-th y : R^(i+1)
    # i-th Q : m x (i+1)
    # i-th Omega : (i+1) x (i+1)
    # i-th H~ : (i+2) x (i+1)
    # i-th R~ : (i+2) x (i+1)
    for i in range(k):
        debug("i =", i)

        # step1: calc i-th H colmn and (i+1)-th Q colmn
        h, q = arnoldi(fun_Ax, Q, i, epsilon)
        Htilda[:i+2, i] = h
        if q is not None:
            Q[:, i+1] = q

        debug("i-th: H = \n", Htilda[:i+2,:i+1])

        # step2: minimize ||beta * e1 - H~ @ y||
        extOmega = extendMat(Omega)
        c, s = givens(h, extOmega)
        G = genG(i, s, c)
        Omega = G @ extOmega # ith- Omega
        gtilda = Omega @ e1[:i+2]
        gamma = gtilda[-1]
        Rtilda[:i+2, i] = Omega @ h 

        debug("i-th Omega = \n", Omega)
        debug("i-th g~ = ", gtilda)
        debug("i-th gamma = ", gamma)
        debug("i-th R~ = \n", Rtilda[:i+2,:i+1])

        y = scipy.linalg.solve_triangular(Rtilda[:i+1,:i+1], gtilda[:-1])
        residual = beta *  abs(gamma) * invb
        debug("i-th y =", y)
        debug("i-th residual =", residual)
        if residual <= epsilon:
            break;
    x = x0 + Q[:, :i+1] @ y;
    debug("x =", x)
    debug("r = ", fun_Ax(x) - b)
    return x

if __name__ == '__main__':
    A = np.array([[3, 0, 0], [0, 2, 0], [0, 0, 1]])
    b = np.array([1, 2, 3])
    x0 = np.array([0,0,0])
    gmres(gen_fun_Ax(A), b, x0, k=3)
