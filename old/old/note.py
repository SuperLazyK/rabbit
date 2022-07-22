import dill as pickle
from sympy import symbols, Matrix, Function, Symbol, diff, solve, collect, Eq
from sympy import simplify
import sympy
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
from sympy import init_printing#, pprint
from pprint import pprint
import itertools

init_printing()

def save(filename, exp):
    with open(filename,'wb') as f:
        pickle.dump(exp,f)

def load(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

t = Symbol('t')

#----------
# const
#---------
l0,l1,l2 = symbols('l0 l1 l2')
m0,m1,m2 = symbols('m0 m1 m2')
g = symbols('g')


#----------
# sim symbol
#---------
symddx   = symbols("x0''")
symddy   = symbols("y0''")
symddth0 = symbols("th0''")
symddth1 = symbols("th1''")
symddth2 = symbols("th2''")


#-----------
# State
#-----------

x0 = Function('x0')(t)
y0 = Function('y0')(t)

th0 = Function('th0')(t)
th1 = Function('th1')(t)
th2 = Function('th2')(t)

#-----------
# ExtForce(Input)
#-----------
fx0 = symbols('fx0')
fy0 = symbols('fy0')

tau0 = symbols('tau0')
tau1 = symbols('tau1')
tau2 = symbols('tau2')

extF = Matrix([fx0, fy0, tau0, tau1, tau2])

#----------
# Kinematics
#----------

x1 = x0 + l0 * sympy.cos(th0)
y1 = y0 + l0 * sympy.sin(th0)

x2 = x1 + l1 * sympy.cos(th0 + th1)
y2 = y1 + l1 * sympy.sin(th0 + th1)


xc0 = x1
yc0 = y1

xc1 = x2
yc1 = y2

xc2 = x2 + l2 * sympy.cos(th0 + th1 + th2)
yc2 = y2 + l2 * sympy.sin(th0 + th1 + th2)

#----------
# dynamics
#----------
dth0 = diff(th0, t)
dth1 = diff(th1, t)
dth2 = diff(th2, t)

dx0 = diff(x0, t)
dy0 = diff(y0, t)

ddth0 = diff(dth0, t)
ddth1 = diff(dth1, t)
ddth2 = diff(dth2, t)

ddx0 = diff(dx0, t)
ddy0 = diff(dy0, t)

dxc0 = diff(xc0, t)
dyc0 = diff(yc0, t)

dxc1 = diff(xc1, t)
dyc1 = diff(yc1, t)

dxc2 = diff(xc2, t)
dyc2 = diff(yc2, t)

def n2(vx, vy):
    return vx * vx + vy * vy

dTh = [dth0, dth1, dth2]

#----------
# Lagrange
#----------


U0 = yc0 * m0 * g
K0_trs = 1/2 * m0 * n2(dxc0, dyc0)
K0_rot = 0


U1 = yc1 * m1 * g
K1_trs = 1/2 * m1 * n2(dxc1, dyc1)
K1_rot = 0


U2 = yc2 * m2 * g
K2_trs = 1/2 * m2 * n2(dxc2, dyc2)
K2_rot = 0

L = K0_trs + K0_rot + K1_trs + K1_rot + K2_trs + K2_rot - U0 - U1 - U2
X = Matrix([x0, y0, th0, th1, th2])
dX = Matrix([dx0, dy0, dth0, dth1, dth2])
ddX = Matrix([ddx0, ddy0, ddth0, ddth1, ddth2])

extF = diff(diff(L, dX), t) - diff(L, X)

def sprint(exp):
    #print(replace_sym(exp))
    pprint(replace_sym(exp))
    #pprint(replace_sym(exp), use_unicode=False)

def cached_simplify(filename, exp):
    if not os.path.isfile(filename):
        save(filename, simplify(exp))
    ret = load(filename)
    print("-----------")
    print(filename)
    print("-----------")
    sprint(ret)
    return ret


def sub_param(exp):
    return exp.subs(
        [ (l0,  1.  )
        , (l1,  1.  )
        , (l2,  1.  )
        , (m0,  1.  )
        , (m1,  1.  )
        , (m2,  2.  )
        , (g,   9.8  )
        ])

def sub_ddthsym(exp):
    return exp.subs(
        [ (ddth0, symddth0)
        , (ddth1, symddth1)
        , (ddth2, symddth2)
        , (ddx0 , 0)
        , (ddy0 , 0)
        ])

def sub_state(exp, vth0, vth1, vth2, vdth0, vdth1, vdth2):
    vx0 = 0
    vy0 = 0
    vdx0 = 0
    vdy0 = 0

    return exp.subs(
        [ (dx0,  vx0  )
        , (dy0,  vy0  )
        , (dth0, vdth0)
        , (dth1, vdth1)
        , (dth2, vdth2)
        , (vx0, vx0)
        , (vy0, vy0)
        , (sympy.cos(th0),         np.cos(vth0))
        , (sympy.cos(th1),         np.cos(vth1))
        , (sympy.cos(th2),         np.cos(vth2))
        , (sympy.cos(th0+th1),     np.cos(vth0+vth1))
        , (sympy.cos(th1+th2),     np.cos(vth1+vth2))
        , (sympy.cos(th0+th1+th2), np.cos(vth0+vth1+vth2))
        , (sympy.sin(th0),         np.sin(vth0))
        , (sympy.sin(th1),         np.sin(vth1))
        , (sympy.sin(th2),         np.sin(vth2))
        , (sympy.sin(th0+th1),     np.sin(vth0+vth1))
        , (sympy.sin(th1+th2),     np.sin(vth1+vth2))
        , (sympy.sin(th0+th1+th2), np.sin(vth0+vth1+vth2))
        , (x0, vx0)
        , (y0, vy0)
        , (th0, vth0)
        , (th1, vth1)
        , (th2, vth2)
        ])

def replace_sym(exp):

    return exp.subs(
        [ (ddx0,  symddx  )
        , (ddy0,  symddy  )
        , (ddth0, symddth0)
        , (ddth1, symddth1)
        , (ddth2, symddth2)
        , (dx0, symbols("dx0"))
        , (dy0, symbols("dy0"))
        , (dth0, symbols("dth0"))
        , (dth1, symbols("dth1"))
        , (dth2, symbols("dth2"))
        , (x0, symbols("x0"))
        , (y0, symbols("y0"))
        , (sympy.cos(th0), symbols("c0"))
        , (sympy.cos(th1), symbols("c1"))
        , (sympy.cos(th2), symbols("c2"))
        , (sympy.cos(th0+th1), symbols("c01"))
        , (sympy.cos(th0+th1+th2), symbols("c012"))
        , (sympy.cos(th1+th2), symbols("c12"))
        , (sympy.sin(th0), symbols("s0"))
        , (sympy.sin(th1), symbols("s1"))
        , (sympy.sin(th2), symbols("s2"))
        , (sympy.sin(th0+th1), symbols("s01"))
        , (sympy.sin(th0+th1+th2), symbols("s012"))
        , (sympy.sin(th1+th2), symbols("s12"))
        , (th0, symbols("th0"))
        , (th1, symbols("th1"))
        , (th2, symbols("th2"))
        ])

dx1 = diff(x1, t)
dy1 = diff(y1, t)

dx2 = diff(x2, t)
dy2 = diff(y2, t)

# only depnd on x0' y0'

extF = cached_simplify("extF.txt", extF)

#---------------------
# planning
#---------------------

# extF = M * ddX + C * dTh_C + B * dTh_B + G
# if the rabbit's toe is on the ground then x0''=0  y0''=0
# if the rabbit's toe is in the airthen f0''= 0

if False:
    M=extF.col(0).jacobian(ddX)
    M = cached_simplify("M.txt", M)
#print(M)

    remain = extF - M * ddX
    G = diff(remain, g) * g
    G = cached_simplify("G.txt", G)

    V = remain - G
    V = cached_simplify("V.txt", V)

#for Coriolis
    dTh_B = []
    B = []
    for d1,d2 in itertools.combinations(dTh, 2):
        dTh_B.append(d1*d2)
        b = []
        for i in range(V.shape[0]):
            b.append(collect(collect(V[i],d1).coeff(d1, 1), d2).coeff(d2, 1))
        B.append(b)
    dTh_B= Matrix(dTh_B)
    B = Matrix(B)
    B = cached_simplify("B.txt", B).T

#for Centrifugal
    dTh_C = []
    C = []
    for d in dTh:
        dTh_C.append(d*d)
        c = []
        for i in range(V.shape[0]):
            c.append(collect(V[i],d).coeff(d, 2))
        C.append(c)
    dTh_C= Matrix(dTh_C)
    C = Matrix(C)
    C = cached_simplify("C.txt", C).T


#---------------------
# simulation
#---------------------

if False:

# foot-contact--mode
    rhs = extF[2:]
    x = [ddth0, ddth1, ddth2]
    A = Matrix(rhs).col(0).jacobian(Matrix(x))
    b = Matrix(rhs) - A * Matrix(x) # - Matrix([tau0, tau1, tau2])
    print("==========")
    cached_simplify("A.txt", A)
    print("A", sub_state(sub_ddthsym(sub_param(A)), np.pi/4, np.pi/2, -np.pi/3, 0.01, 0, 0), Matrix([0, 0, 0]))
    print("==========")
    cached_simplify("b.txt", b)
    print("b", sub_state(sub_ddthsym(sub_param(b)), np.pi/4, np.pi/2, -np.pi/3, 0.01, 0, 0), Matrix([0, 0, 0]))
    print("==========")
    simeq = Eq(sub_state(sub_ddthsym(sub_param(A * Matrix(x) + b)), np.pi/4, np.pi/2, -np.pi/3, 0.01, 0, 0), Matrix([0, 0, 0]))
    #print("==========")
    #print(simeq)
    #print("==========")
    #print(x)
    #print("==========")
    print(solve(simeq, sub_ddthsym(Matrix(x))))
    #print("==========")
else:

# foot-in-the-air-mode
    rhs = extF
    #simeq_c = Eq(Matrix(rhs), Matrix(extF))
    x = [ddx0, ddy0, ddth0, ddth1, ddth2]
    A = Matrix(rhs).col(0).jacobian(Matrix(x))
    b = Matrix(rhs) - A * Matrix(x)
    print("==========")
    cached_simplify("A-2.txt", A)
    print("==========")
    cached_simplify("A-det", A.det())
    print("==========")
    print("==========")
    cached_simplify("b-2.txt", b)
    print("==========")
    #f_a_mode = f.subs([ (fx0, 0)
    #              , (fy0, 0)
    #              , (tau0, 0)
    #              ])

    #sprint(f_a_mode)
    #simulation_c_mode=sympy.solve(f_a_mode, ddX)
    #cached_simplify("simulation_a_mode.txt", simulation_c_mode)










