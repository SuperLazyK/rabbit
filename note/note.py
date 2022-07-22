from sympy import symbols, Matrix, Function, Symbol, diff, solve
from sympy import cos, sin, simplify
import sympy
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
from sympy import init_printing, pprint
init_printing()


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

tau1 = Function('tau1')(t)
tau2 = Function('tau2')(t)

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

dxc0 = diff(xc0, t)
dyc0 = diff(yc0, t)

dxc1 = diff(xc1, t)
dyc1 = diff(yc1, t)

dxc2 = diff(xc2, t)
dyc2 = diff(yc2, t)

def n2(vx, vy):
    return vx * vx + vy * vy

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


X = Matrix([[th0], [th1], [th2]])
dX = Matrix([[dth0], [dth1], [dth2]])

tau = simplify(diff(diff(L, dX), t) - diff(L, X))

sym_th0 = symbols("th0")
sym_th1 = symbols("th1")
sym_th2 = symbols("th2")
symd_th0 = symbols("th0'")
symd_th1 = symbols("th1'")
symd_th2 = symbols("th2'")
symdd_th0 = symbols("th0''")
symdd_th1 = symbols("th1''")
symdd_th2 = symbols("th2''")

print(tau[0].subs(
        [ (ddth0, symdd_th0)
        , (ddth1, symdd_th1)
        , (ddth2, symdd_th2)
        , (dth0, symd_th0)
        , (dth1, symd_th1)
        , (dth2, symd_th2)
        , (th0, sym_th0)
        , (th1, sym_th1)
        , (th2, sym_th2)
        ]))
print(tau[1].subs(
        [ (ddth0, symdd_th0)
        , (ddth1, symdd_th1)
        , (ddth2, symdd_th2)
        , (dth0, symd_th0)
        , (dth1, symd_th1)
        , (dth2, symd_th2)
        , (th0, sym_th0)
        , (th1, sym_th1)
        , (th2, sym_th2)
        ]))
print(tau[2].subs(
        [ (ddth0, symdd_th0)
        , (ddth1, symdd_th1)
        , (ddth2, symdd_th2)
        , (dth0, symd_th0)
        , (dth1, symd_th1)
        , (dth2, symd_th2)
        , (th0, sym_th0)
        , (th1, sym_th1)
        , (th2, sym_th2)
        ]))


X = vector()
dX = vector()

#y_diff = y_diff.subs(u, opt_u)

