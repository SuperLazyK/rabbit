import dill as pickle
from sympy import symbols, Matrix, Function, Symbol, diff, solve, collect
from sympy import cos, sin, simplify
import sympy
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
from sympy import init_printing, pprint
import itertools

init_printing()

def save(filename, exp):
    with open(filename,'wb') as f:
        pickle.dump(exp,f)

def load(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)

def cached_simplify(filename, exp):
    if not os.path.isfile(filename):
        save(filename, simplify(exp))
    ret = load(filename)
    return ret

t = Symbol('t')

#----------
# const
#---------
l0,l1,l2 = symbols('l0 l1 l2')
m0,m1,m2 = symbols('m0 m1 m2')
g = symbols('g')

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

#----------
# Kinematics
#----------

x1 = x0 + l0 * cos(th0)
y1 = y0 + l0 * sin(th0)

x2 = x1 + l1 * cos(th0 + th1)
y2 = y1 + l1 * sin(th0 + th1)

xc0 = x1
yc0 = y1

xc1 = x2
yc1 = y2

xc2 = x2 + l2 * cos(th0 + th1 + th2)
yc2 = y2 + l2 * sin(th0 + th1 + th2)

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

tau = diff(diff(L, dX), t) - diff(L, X)

def replace_sym(exp):
    return exp.subs(
        [ (ddx0, symbols("x0''"))
        , (ddy0, symbols("y0''"))
        , (ddth0, symbols("th0''"))
        , (ddth1, symbols("th1''"))
        , (ddth2, symbols("th2''"))
        , (dx0, symbols("x0'"))
        , (dy0, symbols("y0'"))
        , (dth0, symbols("th0'"))
        , (dth1, symbols("th1'"))
        , (dth2, symbols("th2'"))
        , (x0, symbols("x0"))
        , (y0, symbols("y0"))
        , (cos(th0), symbols("c0"))
        , (cos(th1), symbols("c1"))
        , (cos(th2), symbols("c2"))
        , (cos(th0+th1), symbols("c01"))
        , (cos(th0+th1+th2), symbols("c012"))
        , (sin(th0), symbols("s0"))
        , (sin(th1), symbols("s1"))
        , (sin(th2), symbols("s2"))
        , (sin(th0+th1), symbols("s01"))
        , (sin(th0+th1+th2), symbols("s012"))
        , (th0, symbols("th0"))
        , (th1, symbols("th1"))
        , (th2, symbols("th2"))
        ])

def sprint(exp):
    print(replace_sym(exp))
    #pprint(replace_sym(exp), use_unicode=False)

tau = cached_simplify("extF.txt", tau)

M=tau.col(0).jacobian(ddX)
M = cached_simplify("M.txt", M)
#print(M)

remain = tau - M * ddX
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
B = cached_simplify("B.txt", B)
sprint(B)

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
C = cached_simplify("C.txt", C)
sprint(C)



#print(tau.col(0))
#print(ddX.shape)
#print(tau.col(0).shape)
#print(M.shape)
#print(M)

