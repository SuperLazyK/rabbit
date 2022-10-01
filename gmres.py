import numpy as np
import scipy

def normalize(v, epsilon):
    l = np.norm(v)
    if l < epsilon:
        return None, l
    return v / l, l

def gen_fun_Ax(A):
    return lambda x: A @ x

def min_y(H, b):
    pass

# x0 = U'
# b = - xi F(U, x, t)
# fun_Ax(x) = dF(U') ~= dF/dU U' + dF/dx x' + dF/dt
# k : restart
def gmres(fun_Ax, b, x0, epsilon=0.001, k=None):
    while True:
        r0 = b - fun_Ax(x0)
        V = np.zeros((), dtype=np.float64)
        H = np.zeros((k+1, k), dtype=np.float64)
        v0, l0 = normalize(r0, epsilon)
        if v0 is None:
            return x0
        V[:,0] = v0
        for i in range(k):
            projv = fun_Ax(V[:,i])
            for j in range(i):
                H[j, i] = projv @ V[:,j]
            new_v = projv - Sum_j (H[j,i] V[:,j])
            vi, li = normalize(new_v, epsilon)
            assert vi is not None, "fail to orthogonalize"
            y, ri = min_norm_with_y(H[], l0, e)
            xi = x0 + V y
            if ri < epsilon:
                return xi
        x0 = xi

