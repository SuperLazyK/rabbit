import numpy as np
import scipy
import sys

def gen_fun_Ax(A):
    return lambda x: A @ x

def arnoldi(fun_Ax, Q, k, epsilon):
    q = fun_Ax(Q[:,k]) # Krylov Vector

    #for i in range(k): # Modified Gram-Schmidt, keeping the Hessenberg matrix
    #    h(i) = transpose(q) * Q[:, i]
    #    q = q - h(i) * Q(:, i);

    H[0:k+1, k] = q @ Q[:,0:k+1]
    q = q - Q[:,0:k+1] @ H[0:k+1]
    h[k + 1] = np.linalg.norm(q);
    assert h[k + 1] > epsilon, "fail to orthogonalize"
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

    r = b - fun_Ax(x0)
    r_norm = np.linalg.norm(r);
    error = r_norm / b_norm

    if error <= epsilon:
        return x0

    sn = np.zeros(k)
    cs = np.zeros(k)
    e1 = np.zeros(k+1)
    e1[0] = 1

    Q = np.zeros((n, k), dtype=np.float64)
    Q[:,0] = r / r_norm;

    H = np.zeros((k+1, k), dtype=np.float64)
    beta = r_norm * e1

    for i in range(k):
        H[:i+1, i], Q[:, i+1] = arnoldi(fun_Ax, Q, i)

        H[:i+1, i], cs[i], sn[i] = apply_givens_rotation(H[:i+1,i], cs, sn, i)

        beta[i + 1] = -sn[i] * beta[i];
        beta[i]     = cs[i] * beta[i];

        error = abs(beta[i + 1]) / b_norm;
        if error <= epsilon:
            break;

    y = scipy.solve_tri(H[:i, :i], beta[:i])
    x = x0 + Q[:, :i] @ y;

    return xxx

if __name__ == '__main__':
    A = np.array([[1, 0], [0, 1]])
    b = np.array([1, 2])
    x0 = np.array([0,0])
    gmres(gen_fun_Ax(A), b, x0, k=2)
