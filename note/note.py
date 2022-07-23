import dill as pickle
from sympy import symbols, Matrix, Function, Symbol, diff, solve
from sympy import cos, sin, simplify
import sympy
import numpy as np
import os
import tkinter as tk
import matplotlib.pyplot as plt
from sympy import init_printing, pprint

#init_printing()
#
#def save(filename, exp):
#    with open(filename,'wb') as f:
#        pickle.dump(exp,f)
#
#def load(filename):
#    with open(filename,'rb') as f:
#        return pickle.load(f)
#
#def cached_simplify(filename, exp):
#    if not os.path.isfile(filename):
#        save(filename, simplify(exp))
#    ret = load(filename)
#    return ret
#
#t = Symbol('t')
#
##----------
## const
##---------
#l0,l1,l2 = symbols('l0 l1 l2')
#m0,m1,m2 = symbols('m0 m1 m2')
#g = symbols('g')
#
##-----------
## State
##-----------
#
#x0 = Function('x0')(t)
#y0 = Function('y0')(t)
#
#th0 = Function('th0')(t)
#th1 = Function('th1')(t)
#th2 = Function('th2')(t)
#
##-----------
## ExtForce(Input)
##-----------
#
##----------
## Kinematics
##----------
#
#x1 = x0 + l0 * cos(th0)
#y1 = y0 + l0 * sin(th0)
#
#x2 = x1 + l1 * cos(th0 + th1)
#y2 = y1 + l1 * sin(th0 + th1)
#
#xc0 = x1
#yc0 = y1
#
#xc1 = x2
#yc1 = y2
#
#xc2 = x2 + l2 * cos(th0 + th1 + th2)
#yc2 = y2 + l2 * sin(th0 + th1 + th2)
#
##----------
## dynamics
##----------
#dth0 = diff(th0, t)
#dth1 = diff(th1, t)
#dth2 = diff(th2, t)
#
#dx0 = diff(x0, t)
#dy0 = diff(y0, t)
#
#ddth0 = diff(dth0, t)
#ddth1 = diff(dth1, t)
#ddth2 = diff(dth2, t)
#
#ddx0 = diff(dx0, t)
#ddy0 = diff(dy0, t)
#
#dxc0 = diff(xc0, t)
#dyc0 = diff(yc0, t)
#
#dxc1 = diff(xc1, t)
#dyc1 = diff(yc1, t)
#
#dxc2 = diff(xc2, t)
#dyc2 = diff(yc2, t)
#
#def n2(vx, vy):
#    return vx * vx + vy * vy
#
##----------
## Lagrange
##----------
#
#U0 = yc0 * m0 * g
#K0_trs = 1/2 * m0 * n2(dxc0, dyc0)
#K0_rot = 0
#
#
#U1 = yc1 * m1 * g
#K1_trs = 1/2 * m1 * n2(dxc1, dyc1)
#K1_rot = 0
#
#
#U2 = yc2 * m2 * g
#K2_trs = 1/2 * m2 * n2(dxc2, dyc2)
#K2_rot = 0
#
#L = K0_trs + K0_rot + K1_trs + K1_rot + K2_trs + K2_rot - U0 - U1 - U2
#X = Matrix([x0, y0, th0, th1, th2])
#dX = Matrix([dx0, dy0, dth0, dth1, dth2])
#ddX = Matrix([ddx0, ddy0, ddth0, ddth1, ddth2])
#dX_B = Matrix([dth0**2, dth1**2, dth2**2])
#dX_C = Matrix([dth0*dth1, dth0*dth2, dth1*dth2])
#
#tau = diff(diff(L, dX), t) - diff(L, X)
#
#def replace_sym(exp):
#    sym_x0 = symbols("x0")
#    sym_y0 = symbols("y0")
#    sym_th0 = symbols("th0")
#    sym_th1 = symbols("th1")
#    sym_th2 = symbols("th2")
#    symd_x0 = symbols("x0'")
#    symd_y0 = symbols("y0'")
#    symd_th0 = symbols("th0'")
#    symd_th1 = symbols("th1'")
#    symd_th2 = symbols("th2'")
#    symdd_x0 = symbols("x0''")
#    symdd_y0 = symbols("y0''")
#    symdd_th0 = symbols("th0''")
#    symdd_th1 = symbols("th1''")
#    symdd_th2 = symbols("th2''")
#    return exp.subs(
#        [ (ddx0, symdd_x0)
#        , (ddy0, symdd_y0)
#        , (ddth0, symdd_th0)
#        , (ddth1, symdd_th1)
#        , (ddth2, symdd_th2)
#        , (dx0, symd_x0)
#        , (dy0, symd_y0)
#        , (dth0, symd_th0)
#        , (dth1, symd_th1)
#        , (dth2, symd_th2)
#        , (x0, sym_x0)
#        , (y0, sym_y0)
#        , (th0, sym_th0)
#        , (th1, sym_th1)
#        , (th2, sym_th2)
#        ])
#
#tau = cached_simplify("extF.txt", tau)
#
#M=tau.col(0).jacobian(ddX)
#M = cached_simplify("M.txt", M)
#print(M)
#
#remain = tau - M * ddX
#G = diff(remain, g) * g
#G = cached_simplify("G.txt", G)
#
#V = remain - G
#V = cached_simplify("V.txt", V)
#
#B
#C
#print("G")
#print(G)
#print("V")
#print(replace_sym(V))
#
#
##print(tau.col(0))
##print(ddX.shape)
##print(tau.col(0).shape)
##print(M.shape)
##print(M)
#
