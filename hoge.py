import numpy as np
import scipy
import scipy.optimize

def f(x):
    return x**2 + x - 2

def fprime(x):
    return 2 * x + 1

root_guess = 0

print("Root:", scipy.optimize.newton(f, root_guess))
print("Root:", scipy.optimize.newton(f, root_guess, fprime))

root_guess = np.zeros(2)
print(root_guess)

#def fun(x):
#    return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
#            0.5 * (x[1] - x[0])**3 + x[1]]
#
#def jac(x):
#    return np.array([[1 + 1.5 * (x[0] - x[1])**2,
#                      -1.5 * (x[0] - x[1])**2],
#                     [-1.5 * (x[1] - x[0])**2,
#                      1 + 1.5 * (x[1] - x[0])**2]])
#
def fun(x):
    return [x[0]**2 + x[1]**2 - 1, x[0] - x[1]]

def jac(x):
    return np.array([[2*x[0], 2*x[1]], [1, -1]])

#print("Root:", scipy.optimize.newton(fv, root_guess))
sol = scipy.optimize.root(fun, [0, 0], jac=jac, method='hybr')
print(sol.x)
