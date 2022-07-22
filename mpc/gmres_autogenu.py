import numpy as np

def given_rot(vec, k, givens_c_vec, givens_s_vec):
    tmp1 = givens_c_vec[k] * vec[k] - givens_s_vec[k] * vec[k+1]
    tmp2 = givens_s_vec[k] * vec[k] + givens_c_vec[k] * vec[k+1]
    vec[k] = tmp1;
    vec[k+1] = tmp2;

# solve Ax = b
# x0 = U'
# b = - xi F(U, x, t)
# fun_Ax(x) = dF(U') ~= dF/dU U' + dF/dx x' + dF/dt
# k : restart
def gmres(fun_Ax, b, x0, epsilon=2.3e-16, k_max=None):

    dim = b.shape[0]
    ret = np.copy(x0)

    if k_max is None:
        k_max =  dim

    g_vec = np.zeros(k_max+1, dtype=np.float64)
    givens_c_vec = np.zeros(k_max+1, dtype=np.float64)
    givens_s_vec = np.zeros(k_max+1, dtype=np.float64)
    basis_mat = np.zeros((dim, k_max+1))
    hessenberg_mat = np.zeros((k_max+1, k_max+1))

    g_vec[0] = np.linalg.norm(b)
    assert g_vec[0] > 0
    basis_mat[:,0] = b / g_vec[0]

    k = 0
    while k < k_max:
        #print("HOGE k=", k)
        #print("g_vec", g_vec)
        #print("basis_mat\n", basis_mat)
        basis_mat[:,k+1] = fun_Ax(basis_mat[:,k])
        #print("Ax", basis_mat[:,k+1])
        for j in range(k+1):
            hessenberg_mat[k,j] = basis_mat[:,k+1].dot(basis_mat[:,j])
            basis_mat[:,k+1] -= hessenberg_mat[k, j] * basis_mat[:,j]

        hessenberg_mat[k,k+1] = np.linalg.norm(basis_mat[:,k+1])
        if np.abs(hessenberg_mat[k,k+1]) < epsilon:
            break

        basis_mat[:,k+1] /= hessenberg_mat[k,k+1]
        #print("basis_mat-div\n", basis_mat)

        for j in range(k): 
            given_rot(hessenberg_mat[k,:], j, givens_c_vec, givens_s_vec)

        nu = np.sqrt(hessenberg_mat[k,k]**2 + hessenberg_mat[k,k+1]**2)

        assert nu > epsilon

        givens_c_vec[k] = hessenberg_mat[k,k] / nu
        givens_s_vec[k] = - hessenberg_mat[k,k+1] / nu
        hessenberg_mat[k,k] = givens_c_vec[k] * hessenberg_mat[k, k] - givens_s_vec[k] * hessenberg_mat[k, k+1]
        hessenberg_mat[k,k+1] = 0
        #print(k)
        given_rot(g_vec, k, givens_c_vec, givens_s_vec)
        #print("g_vec\n", g_vec)
        k = k + 1

    #print("H~\n",hessenberg_mat )
    for i in range(k-1, -1, -1):
        tmp = g_vec[i]
        for j in range(i+1,k):
            tmp -= hessenberg_mat[j,i] * givens_c_vec[j]
        givens_c_vec[i] = tmp / hessenberg_mat[i, i]
        #print("cvec@", i, givens_c_vec[i])

    ret = np.zeros(dim)
    for i in range(dim):
        tmp = 0
        for j in range(k):
            tmp += basis_mat[i, j] * givens_c_vec[j]
        ret[i] = tmp + x0[i]

    return ret
