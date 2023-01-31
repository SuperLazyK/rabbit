import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpc_sympy import MPCTracker, MPCModel
from sympy import sin, cos, tan, exp, log, sinh, cosh, tanh, diff, sqrt, Symbol, symbols, Matrix


def rabbit_ground():
    t_ = Symbol('t')
    x_ = symbols('x[0:%d]' %(6)) #th0, th1, th2, dth0, dth1, dth2
    u_ = symbols('u[0:%d]' %(2))
    th0 = x_[0]
    th1 = x_[1]
    th2 = x_[2]
    dth0 = x_[3]
    dth1 = x_[4]
    dth2 = x_[5]
    tau1 = u_[0]
    tau2 = u_[1]
    umax = 2
    xg_ = 2.2727272727272727E-1*cos(th2+th1+th0)+6.818181818181818E-1*cos(th1+th0)+9.545454545454546E-1*cos(th0)
    yg_ = 2.2727272727272727E-1*sin(th2+th1+th0)+6.818181818181818E-1*sin(th1+th0)+9.545454545454546E-1*sin(th0)
    detA = -(9*cos(2*th2)+24*cos(2*th1)-63)/4.0E+2
    ddth0 = 1/detA*(1.3499999999999998E-2*dth2**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)+2.6999999999999997E-2*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)+2.6999999999999997E-2*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)+1.3499999999999998E-2*dth1**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)+2.6999999999999997E-2*dth0*dth1*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)+1.3499999999999998E-2*dth0**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-1.3499999999999998E-2*dth2**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)-2.6999999999999997E-2*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)-2.6999999999999997E-2*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)-1.3499999999999998E-2*dth1**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)-2.6999999999999997E-2*dth0*dth1*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)-1.3499999999999998E-2*dth0**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)-3.6000000000000007E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(2*th2+th1)+3.6000000000000007E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(2*th2+th1)-3.528E-1*cos(th1+th0)*cos(2*th2+th1)-4.5E-2*tau2*cos(2*th2+th1)+4.5E-2*tau1*cos(2*th2+th1)+3.6E-2*dth1**2*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)+7.2E-2*dth0*dth1*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)+3.6E-2*dth0**2*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)+3.6E-2*dth0**2*cos(th0)*cos(th2+th1)*sin(th2+th1+th0)-3.6E-2*dth1**2*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)-7.2E-2*dth0*dth1*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)-3.6E-2*dth0**2*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)-3.6E-2*dth0**2*cos(th0)*cos(th2-th1)*sin(th2+th1+th0)-1.734723475976807E-18*dth2**2*cos(th1+th0)*cos(2*th2)*sin(th2+th1+th0)-3.469446951953614E-18*dth1*dth2*cos(th1+th0)*cos(2*th2)*sin(th2+th1+th0)-3.469446951953614E-18*dth0*dth2*cos(th1+th0)*cos(2*th2)*sin(th2+th1+th0)-1.35E-2*dth2**2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)-2.7E-2*dth1*dth2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)-2.7E-2*dth0*dth2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)-1.35E-2*dth1**2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)-2.7E-2*dth0*dth1*cos(th0)*cos(2*th2)*sin(th2+th1+th0)-1.3499999999999998E-2*dth0**2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)-5.8499999999999999E-2*dth2**2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)-1.1699999999999999E-1*dth1*dth2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)-1.1699999999999999E-1*dth0*dth2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)-5.8499999999999999E-2*dth1**2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)-1.1699999999999999E-1*dth0*dth1*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)-5.8499999999999999E-2*dth0**2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+6.938893903907229E-18*dth2**2*cos(th1+th0)*sin(th2+th1+th0)+1.3877787807814458E-17*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)+1.3877787807814458E-17*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)+5.85E-2*dth2**2*cos(th0)*sin(th2+th1+th0)+1.17E-1*dth1*dth2*cos(th0)*sin(th2+th1+th0)+1.17E-1*dth0*dth2*cos(th0)*sin(th2+th1+th0)+5.85E-2*dth1**2*cos(th0)*sin(th2+th1+th0)+1.17E-1*dth0*dth1*cos(th0)*sin(th2+th1+th0)+5.8499999999999999E-2*dth0**2*cos(th0)*sin(th2+th1+th0)-3.6E-2*dth1**2*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)-7.2E-2*dth0*dth1*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)-3.6E-2*dth0**2*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)-3.6E-2*dth0**2*sin(th0)*cos(th2+th1)*cos(th2+th1+th0)+3.528E-1*cos(th2+th1)*cos(th2+th1+th0)+3.6E-2*dth1**2*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)+7.2E-2*dth0*dth1*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)+3.6E-2*dth0**2*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)+3.6E-2*dth0**2*sin(th0)*cos(th2-th1)*cos(th2+th1+th0)-3.528E-1*cos(th2-th1)*cos(th2+th1+th0)+1.734723475976807E-18*dth2**2*sin(th1+th0)*cos(2*th2)*cos(th2+th1+th0)+3.469446951953614E-18*dth1*dth2*sin(th1+th0)*cos(2*th2)*cos(th2+th1+th0)+3.469446951953614E-18*dth0*dth2*sin(th1+th0)*cos(2*th2)*cos(th2+th1+th0)+1.35E-2*dth2**2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)+2.7E-2*dth1*dth2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)+2.7E-2*dth0*dth2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)+1.35E-2*dth1**2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)+2.7E-2*dth0*dth1*sin(th0)*cos(2*th2)*cos(th2+th1+th0)+1.3499999999999998E-2*dth0**2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)+5.8499999999999999E-2*dth2**2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)+1.1699999999999999E-1*dth1*dth2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)+1.1699999999999999E-1*dth0*dth2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)+5.8499999999999999E-2*dth1**2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)+1.1699999999999999E-1*dth0*dth1*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)+5.8499999999999999E-2*dth0**2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-6.938893903907229E-18*dth2**2*sin(th1+th0)*cos(th2+th1+th0)-1.3877787807814458E-17*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)-1.3877787807814458E-17*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)-5.85E-2*dth2**2*sin(th0)*cos(th2+th1+th0)-1.17E-1*dth1*dth2*sin(th0)*cos(th2+th1+th0)-1.17E-1*dth0*dth2*sin(th0)*cos(th2+th1+th0)-5.85E-2*dth1**2*sin(th0)*cos(th2+th1+th0)-1.17E-1*dth0*dth1*sin(th0)*cos(th2+th1+th0)-5.8499999999999999E-2*dth0**2*sin(th0)*cos(th2+th1+th0)-1.2E-1*tau2*cos(th2+th1)+1.2E-1*tau2*cos(th2-th1)-3.6000000000000007E-2*dth1**2*cos(th0)*sin(th1+th0)*cos(2*th2)-7.200000000000001E-2*dth0*dth1*cos(th0)*sin(th1+th0)*cos(2*th2)-3.6000000000000007E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(2*th2)+3.6000000000000007E-2*dth1**2*sin(th0)*cos(th1+th0)*cos(2*th2)+7.200000000000001E-2*dth0*dth1*sin(th0)*cos(th1+th0)*cos(2*th2)+3.6000000000000007E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(2*th2)+5.733E-1*cos(th0)*cos(2*th2)+4.5E-2*tau1*cos(2*th2)+1.5600000000000004E-1*dth0**2*cos(th0)*cos(th1)*sin(th1+th0)+1.56E-1*dth1**2*cos(th0)*sin(th1+th0)+3.12E-1*dth0*dth1*cos(th0)*sin(th1+th0)+1.5600000000000004E-1*dth0**2*cos(th0)*sin(th1+th0)-1.5600000000000004E-1*dth0**2*sin(th0)*cos(th1)*cos(th1+th0)+1.5288E+0*cos(th1)*cos(th1+th0)-1.56E-1*dth1**2*sin(th0)*cos(th1+th0)-3.12E-1*dth0*dth1*sin(th0)*cos(th1+th0)-1.5600000000000004E-1*dth0**2*sin(th0)*cos(th1+th0)-2.220446049250313E-16*cos(th1+th0)+1.95E-1*tau2*cos(th1)-1.95E-1*tau1*cos(th1)-2.4843E+0*cos(th0)-1.95E-1*tau1)
    ddth1 = 1/detA*((-1.3499999999999998E-2*dth2**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1))-2.6999999999999997E-2*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)-2.6999999999999997E-2*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)-1.3499999999999998E-2*dth1**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)-2.6999999999999997E-2*dth0*dth1*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)-1.3499999999999998E-2*dth0**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)+1.3499999999999998E-2*dth2**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)+2.6999999999999997E-2*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)+2.6999999999999997E-2*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)+1.3499999999999998E-2*dth1**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)+2.6999999999999997E-2*dth0*dth1*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)+1.3499999999999998E-2*dth0**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)+3.6000000000000007E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(2*th2+2*th1)-3.6000000000000007E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(2*th2+2*th1)+3.528E-1*cos(th1+th0)*cos(2*th2+2*th1)+4.5E-2*tau2*cos(2*th2+2*th1)-4.5E-2*tau1*cos(2*th2+2*th1)-1.3499999999999997E-2*dth2**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-2.6999999999999994E-2*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-2.6999999999999994E-2*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-1.3499999999999998E-2*dth1**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-2.6999999999999997E-2*dth0*dth1*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-1.3499999999999998E-2*dth0**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)+1.35E-2*dth2**2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)+2.7E-2*dth1*dth2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)+2.7E-2*dth0*dth2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)+1.35E-2*dth1**2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)+2.7E-2*dth0*dth1*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)+1.3499999999999998E-2*dth0**2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)+1.3499999999999997E-2*dth2**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)+2.6999999999999994E-2*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)+2.6999999999999994E-2*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)+1.3499999999999998E-2*dth1**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)+2.6999999999999997E-2*dth0*dth1*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)+1.3499999999999998E-2*dth0**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)-1.35E-2*dth2**2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)-2.7E-2*dth1*dth2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)-2.7E-2*dth0*dth2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)-1.35E-2*dth1**2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)-2.7E-2*dth0*dth1*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)-1.3499999999999998E-2*dth0**2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)+3.6000000000000007E-2*dth1**2*cos(th0)*sin(th1+th0)*cos(2*th2+th1)+7.200000000000001E-2*dth0*dth1*cos(th0)*sin(th1+th0)*cos(2*th2+th1)+7.200000000000001E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(2*th2+th1)-3.6000000000000007E-2*dth1**2*sin(th0)*cos(th1+th0)*cos(2*th2+th1)-7.200000000000001E-2*dth0*dth1*sin(th0)*cos(th1+th0)*cos(2*th2+th1)-7.200000000000001E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(2*th2+th1)+3.528E-1*cos(th1+th0)*cos(2*th2+th1)-5.733E-1*cos(th0)*cos(2*th2+th1)+4.5E-2*tau2*cos(2*th2+th1)-9.0E-2*tau1*cos(2*th2+th1)-3.6E-2*dth1**2*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)-7.2E-2*dth0*dth1*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)-3.6E-2*dth0**2*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)-3.6E-2*dth0**2*cos(th0)*sin(th2+th1+th0)*cos(th2+2*th1)+3.6E-2*dth1**2*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)+7.2E-2*dth0*dth1*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)+3.6E-2*dth0**2*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)+3.6E-2*dth0**2*sin(th0)*cos(th2+th1+th0)*cos(th2+2*th1)-3.528E-1*cos(th2+th1+th0)*cos(th2+2*th1)+1.2E-1*tau2*cos(th2+2*th1)-3.6E-2*dth1**2*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)-7.2E-2*dth0*dth1*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)-3.6E-2*dth0**2*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)-3.6E-2*dth0**2*cos(th0)*cos(th2+th1)*sin(th2+th1+th0)+3.6E-2*dth1**2*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)+7.2E-2*dth0*dth1*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)+3.6E-2*dth0**2*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)+3.6E-2*dth0**2*cos(th0)*cos(th2-th1)*sin(th2+th1+th0)+1.734723475976807E-18*dth2**2*cos(th1+th0)*cos(2*th2)*sin(th2+th1+th0)+3.469446951953614E-18*dth1*dth2*cos(th1+th0)*cos(2*th2)*sin(th2+th1+th0)+3.469446951953614E-18*dth0*dth2*cos(th1+th0)*cos(2*th2)*sin(th2+th1+th0)+1.35E-2*dth2**2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)+2.7E-2*dth1*dth2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)+2.7E-2*dth0*dth2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)+1.35E-2*dth1**2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)+2.7E-2*dth0*dth1*cos(th0)*cos(2*th2)*sin(th2+th1+th0)+1.3499999999999998E-2*dth0**2*cos(th0)*cos(2*th2)*sin(th2+th1+th0)+8.1E-2*dth1**2*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)+1.62E-1*dth0*dth1*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)+8.1E-2*dth0**2*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)+8.1E-2*dth0**2*cos(th0)*cos(th2)*sin(th2+th1+th0)+5.849999999999999E-2*dth2**2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+1.1699999999999998E-1*dth1*dth2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+1.1699999999999998E-1*dth0*dth2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+5.8499999999999999E-2*dth1**2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+1.1699999999999999E-1*dth0*dth1*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+5.8499999999999999E-2*dth0**2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+1.0350000000000001E-1*dth2**2*cos(th1+th0)*sin(th2+th1+th0)+2.0700000000000003E-1*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)+2.0700000000000003E-1*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)+1.035E-1*dth1**2*cos(th1+th0)*sin(th2+th1+th0)+2.07E-1*dth0*dth1*cos(th1+th0)*sin(th2+th1+th0)+1.035E-1*dth0**2*cos(th1+th0)*sin(th2+th1+th0)-5.85E-2*dth2**2*cos(th0)*cos(th1)*sin(th2+th1+th0)-1.17E-1*dth1*dth2*cos(th0)*cos(th1)*sin(th2+th1+th0)-1.17E-1*dth0*dth2*cos(th0)*cos(th1)*sin(th2+th1+th0)-5.85E-2*dth1**2*cos(th0)*cos(th1)*sin(th2+th1+th0)-1.17E-1*dth0*dth1*cos(th0)*cos(th1)*sin(th2+th1+th0)-5.8499999999999999E-2*dth0**2*cos(th0)*cos(th1)*sin(th2+th1+th0)-5.85E-2*dth2**2*cos(th0)*sin(th2+th1+th0)-1.17E-1*dth1*dth2*cos(th0)*sin(th2+th1+th0)-1.17E-1*dth0*dth2*cos(th0)*sin(th2+th1+th0)-5.85E-2*dth1**2*cos(th0)*sin(th2+th1+th0)-1.17E-1*dth0*dth1*cos(th0)*sin(th2+th1+th0)-5.850000000000001E-2*dth0**2*cos(th0)*sin(th2+th1+th0)+3.6E-2*dth1**2*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)+7.2E-2*dth0*dth1*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)+3.6E-2*dth0**2*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)+3.6E-2*dth0**2*sin(th0)*cos(th2+th1)*cos(th2+th1+th0)-3.528E-1*cos(th2+th1)*cos(th2+th1+th0)-3.6E-2*dth1**2*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)-7.2E-2*dth0*dth1*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)-3.6E-2*dth0**2*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)-3.6E-2*dth0**2*sin(th0)*cos(th2-th1)*cos(th2+th1+th0)+3.528E-1*cos(th2-th1)*cos(th2+th1+th0)-1.734723475976807E-18*dth2**2*sin(th1+th0)*cos(2*th2)*cos(th2+th1+th0)-3.469446951953614E-18*dth1*dth2*sin(th1+th0)*cos(2*th2)*cos(th2+th1+th0)-3.469446951953614E-18*dth0*dth2*sin(th1+th0)*cos(2*th2)*cos(th2+th1+th0)-1.35E-2*dth2**2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)-2.7E-2*dth1*dth2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)-2.7E-2*dth0*dth2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)-1.35E-2*dth1**2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)-2.7E-2*dth0*dth1*sin(th0)*cos(2*th2)*cos(th2+th1+th0)-1.3499999999999998E-2*dth0**2*sin(th0)*cos(2*th2)*cos(th2+th1+th0)-8.1E-2*dth1**2*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)-1.62E-1*dth0*dth1*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)-8.1E-2*dth0**2*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)-8.1E-2*dth0**2*sin(th0)*cos(th2)*cos(th2+th1+th0)+7.938E-1*cos(th2)*cos(th2+th1+th0)-5.849999999999999E-2*dth2**2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-1.1699999999999998E-1*dth1*dth2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-1.1699999999999998E-1*dth0*dth2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-5.8499999999999999E-2*dth1**2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-1.1699999999999999E-1*dth0*dth1*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-5.8499999999999999E-2*dth0**2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-1.0350000000000001E-1*dth2**2*sin(th1+th0)*cos(th2+th1+th0)-2.0700000000000003E-1*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)-2.0700000000000003E-1*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)-1.035E-1*dth1**2*sin(th1+th0)*cos(th2+th1+th0)-2.07E-1*dth0*dth1*sin(th1+th0)*cos(th2+th1+th0)-1.035E-1*dth0**2*sin(th1+th0)*cos(th2+th1+th0)+5.85E-2*dth2**2*sin(th0)*cos(th1)*cos(th2+th1+th0)+1.17E-1*dth1*dth2*sin(th0)*cos(th1)*cos(th2+th1+th0)+1.17E-1*dth0*dth2*sin(th0)*cos(th1)*cos(th2+th1+th0)+5.85E-2*dth1**2*sin(th0)*cos(th1)*cos(th2+th1+th0)+1.17E-1*dth0*dth1*sin(th0)*cos(th1)*cos(th2+th1+th0)+5.8499999999999999E-2*dth0**2*sin(th0)*cos(th1)*cos(th2+th1+th0)+5.85E-2*dth2**2*sin(th0)*cos(th2+th1+th0)+1.17E-1*dth1*dth2*sin(th0)*cos(th2+th1+th0)+1.17E-1*dth0*dth2*sin(th0)*cos(th2+th1+th0)+5.85E-2*dth1**2*sin(th0)*cos(th2+th1+th0)+1.17E-1*dth0*dth1*sin(th0)*cos(th2+th1+th0)+5.850000000000001E-2*dth0**2*sin(th0)*cos(th2+th1+th0)+2.220446049250313E-16*cos(th2+th1+th0)+1.2E-1*tau2*cos(th2+th1)-1.2E-1*tau2*cos(th2-th1)+3.6000000000000007E-2*dth1**2*cos(th0)*sin(th1+th0)*cos(2*th2)+7.200000000000001E-2*dth0*dth1*cos(th0)*sin(th1+th0)*cos(2*th2)+3.6000000000000007E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(2*th2)-3.6000000000000007E-2*dth1**2*sin(th0)*cos(th1+th0)*cos(2*th2)-7.200000000000001E-2*dth0*dth1*sin(th0)*cos(th1+th0)*cos(2*th2)-3.6000000000000007E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(2*th2)-5.733E-1*cos(th0)*cos(2*th2)-4.5E-2*tau1*cos(2*th2)-2.7E-1*tau2*cos(th2)-1.56E-1*dth1**2*cos(th0)*cos(th1)*sin(th1+th0)-3.12E-1*dth0*dth1*cos(th0)*cos(th1)*sin(th1+th0)-3.1200000000000008E-1*dth0**2*cos(th0)*cos(th1)*sin(th1+th0)-1.56E-1*dth1**2*cos(th0)*sin(th1+th0)-3.12E-1*dth0*dth1*cos(th0)*sin(th1+th0)-4.3200000000000007E-1*dth0**2*cos(th0)*sin(th1+th0)+1.56E-1*dth1**2*sin(th0)*cos(th1)*cos(th1+th0)+3.12E-1*dth0*dth1*sin(th0)*cos(th1)*cos(th1+th0)+3.1200000000000008E-1*dth0**2*sin(th0)*cos(th1)*cos(th1+th0)-1.5287999999999998E+0*cos(th1)*cos(th1+th0)+1.56E-1*dth1**2*sin(th0)*cos(th1+th0)+3.12E-1*dth0*dth1*sin(th0)*cos(th1+th0)+4.3200000000000007E-1*dth0**2*sin(th0)*cos(th1+th0)-2.7047999999999998E+0*cos(th1+th0)+2.4843E+0*cos(th0)*cos(th1)-1.95E-1*tau2*cos(th1)+3.9E-1*tau1*cos(th1)+2.4843E+0*cos(th0)-3.4500000000000005E-1*tau2+5.4E-1*tau1)
    ddth2 = 1/detA*(1.3499999999999998E-2*dth2**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)+2.6999999999999997E-2*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)+2.6999999999999997E-2*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)+1.3499999999999998E-2*dth1**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)+2.6999999999999997E-2*dth0*dth1*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)+1.3499999999999998E-2*dth0**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+2*th1)-1.3499999999999998E-2*dth2**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)-2.6999999999999997E-2*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)-2.6999999999999997E-2*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)-1.3499999999999998E-2*dth1**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)-2.6999999999999997E-2*dth0*dth1*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)-1.3499999999999998E-2*dth0**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+2*th1)-3.6000000000000007E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(2*th2+2*th1)+3.6000000000000007E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(2*th2+2*th1)-3.528E-1*cos(th1+th0)*cos(2*th2+2*th1)-4.5E-2*tau2*cos(2*th2+2*th1)+4.5E-2*tau1*cos(2*th2+2*th1)-1.734723475976807E-18*dth2**2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-3.469446951953614E-18*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-3.469446951953614E-18*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(2*th2+th1)-1.35E-2*dth2**2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)-2.7E-2*dth1*dth2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)-2.7E-2*dth0*dth2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)-1.35E-2*dth1**2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)-2.7E-2*dth0*dth1*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)-1.3499999999999998E-2*dth0**2*cos(th0)*sin(th2+th1+th0)*cos(2*th2+th1)+1.734723475976807E-18*dth2**2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)+3.469446951953614E-18*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)+3.469446951953614E-18*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(2*th2+th1)+1.35E-2*dth2**2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)+2.7E-2*dth1*dth2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)+2.7E-2*dth0*dth2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)+1.35E-2*dth1**2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)+2.7E-2*dth0*dth1*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)+1.3499999999999998E-2*dth0**2*sin(th0)*cos(th2+th1+th0)*cos(2*th2+th1)-3.6000000000000007E-2*dth1**2*cos(th0)*sin(th1+th0)*cos(2*th2+th1)-7.200000000000001E-2*dth0*dth1*cos(th0)*sin(th1+th0)*cos(2*th2+th1)-3.6000000000000007E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(2*th2+th1)+3.6000000000000007E-2*dth1**2*sin(th0)*cos(th1+th0)*cos(2*th2+th1)+7.200000000000001E-2*dth0*dth1*sin(th0)*cos(th1+th0)*cos(2*th2+th1)+3.6000000000000007E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(2*th2+th1)+5.733E-1*cos(th0)*cos(2*th2+th1)+4.5E-2*tau1*cos(2*th2+th1)+3.6E-2*dth2**2*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)+7.2E-2*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)+7.2E-2*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)+7.2E-2*dth1**2*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)+1.44E-1*dth0*dth1*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)+7.2E-2*dth0**2*cos(th1+th0)*sin(th2+th1+th0)*cos(th2+2*th1)+3.6E-2*dth0**2*cos(th0)*sin(th2+th1+th0)*cos(th2+2*th1)-3.6E-2*dth2**2*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)-7.2E-2*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)-7.2E-2*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)-7.2E-2*dth1**2*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)-1.44E-1*dth0*dth1*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)-7.2E-2*dth0**2*sin(th1+th0)*cos(th2+th1+th0)*cos(th2+2*th1)-3.6E-2*dth0**2*sin(th0)*cos(th2+th1+th0)*cos(th2+2*th1)+3.528E-1*cos(th2+th1+th0)*cos(th2+2*th1)-9.600000000000002E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(th2+2*th1)+9.600000000000002E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(th2+2*th1)-9.408E-1*cos(th1+th0)*cos(th2+2*th1)-2.4E-1*tau2*cos(th2+2*th1)+1.2E-1*tau1*cos(th2+2*th1)-6.938893903907229E-18*dth2**2*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)-1.3877787807814458E-17*dth1*dth2*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)-1.3877787807814458E-17*dth0*dth2*cos(th1+th0)*cos(th2+th1)*sin(th2+th1+th0)-3.6000000000000007E-2*dth2**2*cos(th0)*cos(th2+th1)*sin(th2+th1+th0)-7.200000000000001E-2*dth1*dth2*cos(th0)*cos(th2+th1)*sin(th2+th1+th0)-7.200000000000001E-2*dth0*dth2*cos(th0)*cos(th2+th1)*sin(th2+th1+th0)-3.6000000000000007E-2*dth1**2*cos(th0)*cos(th2+th1)*sin(th2+th1+th0)-7.200000000000001E-2*dth0*dth1*cos(th0)*cos(th2+th1)*sin(th2+th1+th0)-3.6E-2*dth0**2*cos(th0)*cos(th2+th1)*sin(th2+th1+th0)+6.938893903907229E-18*dth2**2*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)+1.3877787807814458E-17*dth1*dth2*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)+1.3877787807814458E-17*dth0*dth2*cos(th1+th0)*cos(th2-th1)*sin(th2+th1+th0)+3.6000000000000007E-2*dth2**2*cos(th0)*cos(th2-th1)*sin(th2+th1+th0)+7.200000000000001E-2*dth1*dth2*cos(th0)*cos(th2-th1)*sin(th2+th1+th0)+7.200000000000001E-2*dth0*dth2*cos(th0)*cos(th2-th1)*sin(th2+th1+th0)+3.6000000000000007E-2*dth1**2*cos(th0)*cos(th2-th1)*sin(th2+th1+th0)+7.200000000000001E-2*dth0*dth1*cos(th0)*cos(th2-th1)*sin(th2+th1+th0)+3.6E-2*dth0**2*cos(th0)*cos(th2-th1)*sin(th2+th1+th0)-8.1E-2*dth2**2*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)-1.62E-1*dth1*dth2*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)-1.62E-1*dth0*dth2*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)-1.62E-1*dth1**2*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)-3.24E-1*dth0*dth1*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)-1.62E-1*dth0**2*cos(th1+th0)*cos(th2)*sin(th2+th1+th0)-8.1E-2*dth0**2*cos(th0)*cos(th2)*sin(th2+th1+th0)+9.6E-2*dth1**2*cos(2*th1)*cos(th1+th0)*sin(th2+th1+th0)+1.92E-1*dth0*dth1*cos(2*th1)*cos(th1+th0)*sin(th2+th1+th0)+9.6E-2*dth0**2*cos(2*th1)*cos(th1+th0)*sin(th2+th1+th0)+6.938893903907229E-18*dth2**2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+1.3877787807814458E-17*dth1*dth2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)+1.3877787807814458E-17*dth0*dth2*cos(th1)*cos(th1+th0)*sin(th2+th1+th0)-1.035E-1*dth2**2*cos(th1+th0)*sin(th2+th1+th0)-2.07E-1*dth1*dth2*cos(th1+th0)*sin(th2+th1+th0)-2.07E-1*dth0*dth2*cos(th1+th0)*sin(th2+th1+th0)-3.195E-1*dth1**2*cos(th1+th0)*sin(th2+th1+th0)-6.39E-1*dth0*dth1*cos(th1+th0)*sin(th2+th1+th0)-3.195E-1*dth0**2*cos(th1+th0)*sin(th2+th1+th0)+9.6E-2*dth0**2*cos(th0)*cos(2*th1)*sin(th2+th1+th0)+5.85E-2*dth2**2*cos(th0)*cos(th1)*sin(th2+th1+th0)+1.17E-1*dth1*dth2*cos(th0)*cos(th1)*sin(th2+th1+th0)+1.17E-1*dth0*dth2*cos(th0)*cos(th1)*sin(th2+th1+th0)+5.85E-2*dth1**2*cos(th0)*cos(th1)*sin(th2+th1+th0)+1.17E-1*dth0*dth1*cos(th0)*cos(th1)*sin(th2+th1+th0)+5.8499999999999999E-2*dth0**2*cos(th0)*cos(th1)*sin(th2+th1+th0)-2.1600000000000003E-1*dth0**2*cos(th0)*sin(th2+th1+th0)+6.938893903907229E-18*dth2**2*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)+1.3877787807814458E-17*dth1*dth2*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)+1.3877787807814458E-17*dth0*dth2*sin(th1+th0)*cos(th2+th1)*cos(th2+th1+th0)+3.6000000000000007E-2*dth2**2*sin(th0)*cos(th2+th1)*cos(th2+th1+th0)+7.200000000000001E-2*dth1*dth2*sin(th0)*cos(th2+th1)*cos(th2+th1+th0)+7.200000000000001E-2*dth0*dth2*sin(th0)*cos(th2+th1)*cos(th2+th1+th0)+3.6000000000000007E-2*dth1**2*sin(th0)*cos(th2+th1)*cos(th2+th1+th0)+7.200000000000001E-2*dth0*dth1*sin(th0)*cos(th2+th1)*cos(th2+th1+th0)+3.6E-2*dth0**2*sin(th0)*cos(th2+th1)*cos(th2+th1+th0)-6.938893903907229E-18*dth2**2*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)-1.3877787807814458E-17*dth1*dth2*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)-1.3877787807814458E-17*dth0*dth2*sin(th1+th0)*cos(th2-th1)*cos(th2+th1+th0)-3.6000000000000007E-2*dth2**2*sin(th0)*cos(th2-th1)*cos(th2+th1+th0)-7.200000000000001E-2*dth1*dth2*sin(th0)*cos(th2-th1)*cos(th2+th1+th0)-7.200000000000001E-2*dth0*dth2*sin(th0)*cos(th2-th1)*cos(th2+th1+th0)-3.6000000000000007E-2*dth1**2*sin(th0)*cos(th2-th1)*cos(th2+th1+th0)-7.200000000000001E-2*dth0*dth1*sin(th0)*cos(th2-th1)*cos(th2+th1+th0)-3.6E-2*dth0**2*sin(th0)*cos(th2-th1)*cos(th2+th1+th0)+8.1E-2*dth2**2*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)+1.62E-1*dth1*dth2*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)+1.62E-1*dth0*dth2*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)+1.62E-1*dth1**2*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)+3.24E-1*dth0*dth1*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)+1.62E-1*dth0**2*sin(th1+th0)*cos(th2)*cos(th2+th1+th0)+8.1E-2*dth0**2*sin(th0)*cos(th2)*cos(th2+th1+th0)-7.938E-1*cos(th2)*cos(th2+th1+th0)-9.6E-2*dth1**2*cos(2*th1)*sin(th1+th0)*cos(th2+th1+th0)-1.92E-1*dth0*dth1*cos(2*th1)*sin(th1+th0)*cos(th2+th1+th0)-9.6E-2*dth0**2*cos(2*th1)*sin(th1+th0)*cos(th2+th1+th0)-6.938893903907229E-18*dth2**2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-1.3877787807814458E-17*dth1*dth2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)-1.3877787807814458E-17*dth0*dth2*cos(th1)*sin(th1+th0)*cos(th2+th1+th0)+1.035E-1*dth2**2*sin(th1+th0)*cos(th2+th1+th0)+2.07E-1*dth1*dth2*sin(th1+th0)*cos(th2+th1+th0)+2.07E-1*dth0*dth2*sin(th1+th0)*cos(th2+th1+th0)+3.195E-1*dth1**2*sin(th1+th0)*cos(th2+th1+th0)+6.39E-1*dth0*dth1*sin(th1+th0)*cos(th2+th1+th0)+3.195E-1*dth0**2*sin(th1+th0)*cos(th2+th1+th0)-9.6E-2*dth0**2*sin(th0)*cos(2*th1)*cos(th2+th1+th0)+9.408E-1*cos(2*th1)*cos(th2+th1+th0)-5.85E-2*dth2**2*sin(th0)*cos(th1)*cos(th2+th1+th0)-1.17E-1*dth1*dth2*sin(th0)*cos(th1)*cos(th2+th1+th0)-1.17E-1*dth0*dth2*sin(th0)*cos(th1)*cos(th2+th1+th0)-5.85E-2*dth1**2*sin(th0)*cos(th1)*cos(th2+th1+th0)-1.17E-1*dth0*dth1*sin(th0)*cos(th1)*cos(th2+th1+th0)-5.8499999999999999E-2*dth0**2*sin(th0)*cos(th1)*cos(th2+th1+th0)+2.1600000000000003E-1*dth0**2*sin(th0)*cos(th2+th1+th0)-2.1168E+0*cos(th2+th1+th0)-9.6E-2*dth1**2*cos(th0)*sin(th1+th0)*cos(th2+th1)-1.92E-1*dth0*dth1*cos(th0)*sin(th1+th0)*cos(th2+th1)-9.600000000000002E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(th2+th1)+9.6E-2*dth1**2*sin(th0)*cos(th1+th0)*cos(th2+th1)+1.92E-1*dth0*dth1*sin(th0)*cos(th1+th0)*cos(th2+th1)+9.600000000000002E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(th2+th1)+1.1102230246251566E-16*cos(th1+th0)*cos(th2+th1)+1.5288000000000002E+0*cos(th0)*cos(th2+th1)+1.2E-1*tau1*cos(th2+th1)+9.6E-2*dth1**2*cos(th0)*sin(th1+th0)*cos(th2-th1)+1.92E-1*dth0*dth1*cos(th0)*sin(th1+th0)*cos(th2-th1)+9.600000000000002E-2*dth0**2*cos(th0)*sin(th1+th0)*cos(th2-th1)-9.6E-2*dth1**2*sin(th0)*cos(th1+th0)*cos(th2-th1)-1.92E-1*dth0*dth1*sin(th0)*cos(th1+th0)*cos(th2-th1)-9.600000000000002E-2*dth0**2*sin(th0)*cos(th1+th0)*cos(th2-th1)-1.1102230246251566E-16*cos(th1+th0)*cos(th2-th1)-1.5288000000000002E+0*cos(th0)*cos(th2-th1)-1.2E-1*tau1*cos(th2-th1)+2.1600000000000003E-1*dth0**2*cos(th0)*sin(th1+th0)*cos(th2)-2.1600000000000003E-1*dth0**2*sin(th0)*cos(th1+th0)*cos(th2)+2.1168E+0*cos(th1+th0)*cos(th2)+5.4E-1*tau2*cos(th2)-2.7E-1*tau1*cos(th2)+1.56E-1*dth1**2*cos(th0)*cos(th1)*sin(th1+th0)+3.12E-1*dth0*dth1*cos(th0)*cos(th1)*sin(th1+th0)+1.5600000000000004E-1*dth0**2*cos(th0)*cos(th1)*sin(th1+th0)+2.76E-1*dth0**2*cos(th0)*sin(th1+th0)-1.56E-1*dth1**2*sin(th0)*cos(th1)*cos(th1+th0)-3.12E-1*dth0*dth1*sin(th0)*cos(th1)*cos(th1+th0)-1.5600000000000004E-1*dth0**2*sin(th0)*cos(th1)*cos(th1+th0)-2.220446049250313E-16*cos(th1)*cos(th1+th0)-2.76E-1*dth0**2*sin(th0)*cos(th1+th0)+2.7048E+0*cos(th1+th0)-3.2E-1*tau2*cos(2*th1)-2.4843E+0*cos(th0)*cos(th1)-1.95E-1*tau1*cos(th1)+1.065E+0*tau2-3.4500000000000005E-1*tau1)
    f_ = [ x_[3]
         , x_[4]
         , x_[5]
         , ddth0
         , ddth1
         , ddth2
        ]

    # xG = 0
    # yG = n
    #phi_ = 1/2 * ((xg_) ** 2 + (yg_ - 1) ** 2)
    phi_ = 1/2 * ((xg_) ** 2 + (yg_ - 1) ** 2)
    L_ = phi_ + 0.1 * 1/2 * (u_[0] ** 2 + u_[1] ** 2)

    t0 = 0
    t1 = 20
    x = np.array([ np.pi/4
                 , np.pi*3/4 - np.pi/4
                 , np.pi*5/12 - np.pi*3/4
                 , 0
                 , 0
                 , 0
                 ])

    model = MPCModel(
            t_, x_, u_,
            f_, phi_, L_,
            np.array([umax, umax]), np.array([0.01, 0.01])
            )
    T = lambda t : 1 - np.exp (-0.5 * (t-t0))
    N = 10
    tracker = MPCTracker(model, T, N)
    dt = 0.01
    t_start = 0
    t_end = 20
    max_itr = int((t_end - t_start) / dt)

    m = model.m
    n = model.n

    U = tracker.estimate_init_U(x)

    #---------------------
    # output
    #---------------------
    history= {}
    history["u"] = np.zeros((max_itr,m))
    history["v"] = np.zeros((max_itr,m))
    history["r"] = np.zeros((max_itr,m))
    history["x"] = np.zeros((max_itr,n))
    history["c"] = np.zeros((max_itr,2))
    history["error"] = np.zeros(max_itr)

    fig = plt.figure()
    ax1 = fig.add_subplot(321)
    ax2 = fig.add_subplot(322)
    ax3 = fig.add_subplot(323)
    ax4 = fig.add_subplot(324)
    ax5 = fig.add_subplot(325)
    ax6 = fig.add_subplot(326)
    ax1.set_xlim(-np.pi, np.pi)
    ax2.set_xlim(-np.pi, np.pi)
    ax3.set_xlim(-np.pi, np.pi)
    ax4.set_xlim(-0.5, 0.5)
    ax5.set_xlim(-0.5, 0.5)

    # tracking loop
    for step in range(max_itr):
        print(f"ITERATION: {step}/{max_itr}")
        t = t_start + step*dt
        x, U, error = tracker.track_u(x, U, t, dt=0.01)

        history["u"][step] = np.copy(U[0,:m])
        history["v"][step] = np.copy(U[0,m:2*m])
        history["r"][step] = np.copy(U[0,2*m:])
        history["x"][step] = np.copy(x)
        history["c"][step] = np.array([ 2.2727272727272727E-1*cos(x[2]+x[1]+x[0])+6.818181818181818E-1*cos(x[1]+x[0])+9.545454545454546E-1*cos(x[0])
                                      , 2.2727272727272727E-1*sin(x[2]+x[1]+x[0])+6.818181818181818E-1*sin(x[1]+x[0])+9.545454545454546E-1*sin(x[0])
                                      ])
        history["error"][step] = -np.log(np.linalg.norm(error,"fro"))

        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()
        ax5.cla()
        ax6.cla()

        #ax1.plot(history["x"][:step+1,0])
        #ax2.plot(history["x"][:step+1,1])
        #ax3.plot(history["x"][:step+1,2])
        ax1.plot(history["c"][:step+1,0])
        ax2.plot(history["c"][:step+1,1])
        ax4.plot(history["u"][:step+1,0])
        ax5.plot(history["u"][:step+1,1])
        ax6.plot(history["error"][:step+1])
        plt.pause(0.01) 


if __name__ == '__main__':
    rabbit_ground()