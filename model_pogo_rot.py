import numpy as np
from numpy import sin, cos, abs
import sys


# Pogo-rotation-knee
max_z = 0.4

#ref_max_th0 = np.deg2rad(10)
ref_max_th0 = np.deg2rad(80)
ref_min_th0 = np.deg2rad(5)
ref_max_th1 = np.deg2rad(80)
ref_min_th1 = np.deg2rad(0)

max_th0 = np.deg2rad(90)
min_th0 = np.deg2rad(0)
max_th1 = np.deg2rad(100)
min_th1 = np.deg2rad(-10)

z0 =  0.4
l0 =  0.4
l1 =  0.6
#knee weight is 0
m0 = 10
m1 = 25
m2 = 35
g  = 9.8
#g  = 0
#T/4=0.15
k  = 20000 # mgh = 1/2 k x^2 -> T=2*pi sqrt(m/k)
#c = 100
c = 50
#c = 0.
# knee 50-250deg/sec

#MAX_TORQUE0=800 # knee can keep 100kg weight at pi/2 + arm
MAX_TORQUE0=1200 # knee can keep 100kg weight at pi/2 + arm
MAX_TORQUE1=300

#Kp = 1200
Kp = 1200
Kd = Kp * (0.1)

#-----------------
# State
#-----------------

IDX_xr   = 0
IDX_yr   = 1
IDX_thr  = 2
IDX_z    = 3
IDX_th0  = 4
IDX_th1  = 5
IDX_dx   = 6
IDX_dy   = 7
IDX_dthr = 8
IDX_dz   = 9
IDX_dth0 = 10
IDX_dth1 = 11
IDX_MAX = 12

def reset_state(np_random=None):
    s = np.zeros(IDX_MAX, dtype=np.float64)
    #s[IDX_yr]  = 1
    s[IDX_yr]  = 0.1
    s[IDX_dy]  = -5
    #s[IDX_yr]  = 0.3
    #s[IDX_dth1]  = 5
    #s[IDX_th0]  = np.pi/4
    s[IDX_th0]  = np.deg2rad(5)

    #s[IDX_th0]  = np.pi/4
    #s[IDX_th1]  = np.pi*3/4 - np.pi/4
    return s

def state_dict(s, d={}):
    d["xr  "] = s[IDX_xr ]
    d["yr  "] = s[IDX_yr ]
    d["thr "] = s[IDX_thr]
    d["z   "] = s[IDX_z  ]
    d["th0 "] = s[IDX_th0]
    d["th1 "] = s[IDX_th1]
    d["dx  "] = s[IDX_dx ]
    d["dy  "] = s[IDX_dy ]
    d["dthr"] = s[IDX_dthr]
    d["dz  "] = s[IDX_dz ]
    d["dth0"] = s[IDX_dth0]
    d["dth1"] = s[IDX_dth1]
    return d

def print_state(s):
    print(f"xr:  {s[IDX_xr ]:.3f}", end=", ")
    print(f"yr:  {s[IDX_yr ]:.3f}", end=", ")
    print(f"thr: {s[IDX_thr]:.3f}", end=", ")
    print(f"z :  {s[IDX_z  ]:.3f}", end=", ")
    print(f"th0: {s[IDX_th0]:.3f}", end=", ")
    print(f"th1: {s[IDX_th1]:.3f}")
    print(f"dx:  {s[IDX_dx  ]:.3f}", end=", ")
    print(f"dy:  {s[IDX_dy  ]:.3f}", end=", ")
    print(f"dthr:{s[IDX_dthr]:.3f}", end=", ")
    print(f"dz:  {s[IDX_dz  ]:.3f}", end=", ")
    print(f"dth0:{s[IDX_dth0]:.3f}", end=", ")
    print(f"dth1:{s[IDX_dth1]:.3f}")

#-----------------
# Kinematics
#-----------------

def check_invariant(s):
    if abs(s[IDX_z]) > max_z:
        print("|z| is too big")
        return False
    if min_th0 > s[IDX_th0] or max_th0 < s[IDX_th0]:
        print("th0 is out-of-range")
        return False
    if min_th1 > s[IDX_th1] or max_th1 < s[IDX_th1]:
        print("th1 is out-of-range")
        return False
    pr, p0, pk, p1, p2 = node_pos(s)
    if p0[1] <= 0.001:
        print(f"GAME OVER p0={p0:}")
        return False
    if pk[1] <= 0.001:
        print(f"GAME OVER pk={pk:}")
        return False
    if p1[1] <= 0.001:
        print(f"GAME OVER p1={p1:}")
        return False
    if p2[1] <= 0.001:
        print(f"GAME OVER p2={p2:}")
        return False
    return True

def postprocess(s):
    new_s = s.copy()
    # normalize_angle
    new_s[IDX_thr] = ((s[IDX_thr] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th0] = ((s[IDX_th0] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th1] = ((s[IDX_th1] + np.pi)% (2*np.pi)) - np.pi
    return new_s

def node_pos(s):
    thr = s[IDX_thr]
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    z   = s[IDX_z]
    d   = 2 * l0 * cos(th0)

    pr = s[IDX_xr:IDX_yr+1].copy()
    p0 = pr + (z0 + z) * np.array([np.sin(thr), np.cos(thr)])
    pk = p0 + l0 * np.array([np.sin(thr+th0), np.cos(thr+th0)])
    p1 = p0 + d * np.array([np.sin(thr), np.cos(thr)])
    p2 = p1 + l1 * np.array([np.sin(thr+th1), np.cos(thr+th1)])

    return pr, p0, pk, p1, p2

def node_vel(s):
    z   = s[IDX_z]
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    thr = s[IDX_thr]
    dz   = s[IDX_dz]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dthr = s[IDX_dthr]

    vr = s[IDX_dx:IDX_dy+1]
    v0 = vr + np.array([sin(thr), cos(thr)]) * dz + np.array([cos(thr), -sin(thr)]) * (z+z0) * dthr
    v1 = v0 + np.array([cos(thr), -sin(thr)]) * 2*dthr*l0*cos(th0) - np.array([sin(thr), cos(thr)]) * 2*dth0*l0*sin(th0)
    v2 = v1 + np.array([cos(th0+th1), -sin(th0+th1)]) * l1 * (dth0+dth1)

    return vr, v0, v1, v2

#-----------------
# Dynamics Util
#-----------------
def rk4(f, t, s, u, params, dt):

    k1 = f(t,        s,             u, params)
    k2 = f(t + dt/2, s + dt/2 * k1, u, params)
    k3 = f(t + dt/2, s + dt/2 * k2, u, params)
    k4 = f(t + dt,   s + dt * k3,   u, params)

    return (k1 + 2*k2 + 2*k3 + k4)/6


#----------------------------
# Dynamics
#----------------------------
def max_u():
    return np.array([MAX_TORQUE0, MAX_TORQUE1])
    #return np.array([MAX_TORQUE0, 0])

def torq_limit(s, u):
    m = max_u()
    ret = np.clip(u, -m, m)
    return ret

def _calcAb44(s, u):
    z    = s[IDX_z]
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    dz   = s[IDX_dz  ]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    fz   = 0
    taur = 0
    tau0 = u[0]
    tau1 = u[1]
    extf = np.array([fz, taur, tau0, tau1]).reshape(4,1)

    A = np.zeros((4,4), dtype=np.float64)
    A[0][0] = m2+m1+m0
    A[0][1] = -l1*m2*sin(th1)
    A[0][2] = ((-2*l0*m2)-2*l0*m1)*sin(th0)
    A[0][3] = -l1*m2*sin(th1)
    A[1][0] = -l1*m2*sin(th1)
    A[1][1] = (m2+m1+m0)*z0**2+((2*m2+2*m1+2*m0)*z+2*l1*m2*cos(th1)+(4*l0*m2+4*l0*m1)*cos(th0))*z0+(m2+m1+m0)*z**2+(2*l1*m2*cos(th1)+(4*l0*m2+4*l0*m1)*cos(th0))*z+2*l0*l1*m2*cos(th1+th0)+2*l0*l1*m2*cos(th1-th0)+(2*l0**2*m2+2*l0**2*m1)*cos(2*th0)+(l1**2+2*l0**2)*m2+2*l0**2*m1
    A[1][2] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[1][3] = l1*m2*cos(th1)*z0+l1*m2*cos(th1)*z+l0*l1*m2*cos(th1+th0)+l0*l1*m2*cos(th1-th0)+l1**2*m2
    A[2][0] = ((-2*l0*m2)-2*l0*m1)*sin(th0)
    A[2][1] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[2][2] = ((-2*l0**2*m2)-2*l0**2*m1)*cos(2*th0)+2*l0**2*m2+2*l0**2*m1
    A[2][3] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[3][0] = -l1*m2*sin(th1)
    A[3][1] = l1*m2*cos(th1)*z0+l1*m2*cos(th1)*z+l0*l1*m2*cos(th1+th0)+l0*l1*m2*cos(th1-th0)+l1**2*m2
    A[3][2] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[3][3] = l1**2*m2

    b = np.zeros((4,1), dtype=np.float64)
    b[0] = (-dthr**2*m2*sin(thr)**2*z0)-dthr**2*m1*sin(thr)**2*z0-dthr**2*m0*sin(thr)**2*z0-dthr**2*m2*cos(thr)**2*z0-dthr**2*m1*cos(thr)**2*z0-dthr**2*m0*cos(thr)**2*z0-dthr**2*m2*sin(thr)**2*z-dthr**2*m1*sin(thr)**2*z-dthr**2*m0*sin(thr)**2*z-dthr**2*m2*cos(thr)**2*z-dthr**2*m1*cos(thr)**2*z-dthr**2*m0*cos(thr)**2*z+k*z-dthr**2*l1*m2*sin(thr)*sin(thr+th1)-2*dth1*dthr*l1*m2*sin(thr)*sin(thr+th1)-dth1**2*l1*m2*sin(thr)*sin(thr+th1)-dthr**2*l1*m2*cos(thr)*cos(thr+th1)-2*dth1*dthr*l1*m2*cos(thr)*cos(thr+th1)-dth1**2*l1*m2*cos(thr)*cos(thr+th1)-2*dthr**2*l0*m2*cos(th0)*sin(thr)**2-2*dth0**2*l0*m2*cos(th0)*sin(thr)**2-2*dthr**2*l0*m1*cos(th0)*sin(thr)**2-2*dth0**2*l0*m1*cos(th0)*sin(thr)**2-2*dthr**2*l0*m2*cos(th0)*cos(thr)**2-2*dth0**2*l0*m2*cos(th0)*cos(thr)**2-2*dthr**2*l0*m1*cos(th0)*cos(thr)**2-2*dth0**2*l0*m1*cos(th0)*cos(thr)**2+g*m2*cos(thr)+g*m1*cos(thr)+g*m0*cos(thr)+c*dz
    b[1] = (-2*dth1*dthr*l1*m2*cos(thr)*sin(thr+th1)*z0)-dth1**2*l1*m2*cos(thr)*sin(thr+th1)*z0+2*dth1*dthr*l1*m2*sin(thr)*cos(thr+th1)*z0+dth1**2*l1*m2*sin(thr)*cos(thr+th1)*z0-4*dth0*dthr*l0*m2*sin(th0)*sin(thr)**2*z0-4*dth0*dthr*l0*m1*sin(th0)*sin(thr)**2*z0+2*dthr*dz*m2*sin(thr)**2*z0+2*dthr*dz*m1*sin(thr)**2*z0+2*dthr*dz*m0*sin(thr)**2*z0-g*m2*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0-4*dth0*dthr*l0*m2*sin(th0)*cos(thr)**2*z0-4*dth0*dthr*l0*m1*sin(th0)*cos(thr)**2*z0+2*dthr*dz*m2*cos(thr)**2*z0+2*dthr*dz*m1*cos(thr)**2*z0+2*dthr*dz*m0*cos(thr)**2*z0-2*dth1*dthr*l1*m2*cos(thr)*sin(thr+th1)*z-dth1**2*l1*m2*cos(thr)*sin(thr+th1)*z+2*dth1*dthr*l1*m2*sin(thr)*cos(thr+th1)*z+dth1**2*l1*m2*sin(thr)*cos(thr+th1)*z-4*dth0*dthr*l0*m2*sin(th0)*sin(thr)**2*z-4*dth0*dthr*l0*m1*sin(th0)*sin(thr)**2*z+2*dthr*dz*m2*sin(thr)**2*z+2*dthr*dz*m1*sin(thr)**2*z+2*dthr*dz*m0*sin(thr)**2*z-g*m2*sin(thr)*z-g*m1*sin(thr)*z-g*m0*sin(thr)*z-4*dth0*dthr*l0*m2*sin(th0)*cos(thr)**2*z-4*dth0*dthr*l0*m1*sin(th0)*cos(thr)**2*z+2*dthr*dz*m2*cos(thr)**2*z+2*dthr*dz*m1*cos(thr)**2*z+2*dthr*dz*m0*cos(thr)**2*z-4*dth0*dthr*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+2*dthr*dz*l1*m2*sin(thr)*sin(thr+th1)-4*dth1*dthr*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)-2*dth1**2*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)+2*dth0**2*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)-g*l1*m2*sin(thr+th1)+4*dth1*dthr*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)+2*dth1**2*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)-2*dth0**2*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)-4*dth0*dthr*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)+2*dthr*dz*l1*m2*cos(thr)*cos(thr+th1)-8*dth0*dthr*l0**2*m2*cos(th0)*sin(th0)*sin(thr)**2-8*dth0*dthr*l0**2*m1*cos(th0)*sin(th0)*sin(thr)**2+4*dthr*dz*l0*m2*cos(th0)*sin(thr)**2+4*dthr*dz*l0*m1*cos(th0)*sin(thr)**2-2*g*l0*m2*cos(th0)*sin(thr)-2*g*l0*m1*cos(th0)*sin(thr)-8*dth0*dthr*l0**2*m2*cos(th0)*sin(th0)*cos(thr)**2-8*dth0*dthr*l0**2*m1*cos(th0)*sin(th0)*cos(thr)**2+4*dthr*dz*l0*m2*cos(th0)*cos(thr)**2+4*dthr*dz*l0*m1*cos(th0)*cos(thr)**2
    b[2] = 2*dthr**2*l0*m2*sin(th0)*sin(thr)**2*z0+2*dthr**2*l0*m1*sin(th0)*sin(thr)**2*z0+2*dthr**2*l0*m2*sin(th0)*cos(thr)**2*z0+2*dthr**2*l0*m1*sin(th0)*cos(thr)**2*z0+2*dthr**2*l0*m2*sin(th0)*sin(thr)**2*z+2*dthr**2*l0*m1*sin(th0)*sin(thr)**2*z+2*dthr**2*l0*m2*sin(th0)*cos(thr)**2*z+2*dthr**2*l0*m1*sin(th0)*cos(thr)**2*z+2*dthr**2*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+4*dth1*dthr*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+2*dth1**2*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+2*dthr**2*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)+4*dth1*dthr*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)+2*dth1**2*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)+4*dthr**2*l0**2*m2*cos(th0)*sin(th0)*sin(thr)**2+4*dth0**2*l0**2*m2*cos(th0)*sin(th0)*sin(thr)**2+4*dthr**2*l0**2*m1*cos(th0)*sin(th0)*sin(thr)**2+4*dth0**2*l0**2*m1*cos(th0)*sin(th0)*sin(thr)**2+4*dthr**2*l0**2*m2*cos(th0)*sin(th0)*cos(thr)**2+4*dth0**2*l0**2*m2*cos(th0)*sin(th0)*cos(thr)**2+4*dthr**2*l0**2*m1*cos(th0)*sin(th0)*cos(thr)**2+4*dth0**2*l0**2*m1*cos(th0)*sin(th0)*cos(thr)**2-2*g*l0*m2*sin(th0)*cos(thr)-2*g*l0*m1*sin(th0)*cos(thr)
    b[3] = dthr**2*l1*m2*cos(thr)*sin(thr+th1)*z0-dthr**2*l1*m2*sin(thr)*cos(thr+th1)*z0+dthr**2*l1*m2*cos(thr)*sin(thr+th1)*z-dthr**2*l1*m2*sin(thr)*cos(thr+th1)*z-4*dth0*dthr*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+2*dthr*dz*l1*m2*sin(thr)*sin(thr+th1)+2*dthr**2*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)+2*dth0**2*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)-g*l1*m2*sin(thr+th1)-2*dthr**2*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)-2*dth0**2*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)-4*dth0*dthr*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)+2*dthr*dz*l1*m2*cos(thr)*cos(thr+th1)
    return A, b, extf

def _calcAb55(s, u):

    xr   = s[IDX_xr]
    yr   = s[IDX_yr]
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    tau0 = u[0]
    tau1 = u[1]
    extf = np.array([0, 0, 0, tau0, tau1]).reshape(5,1)

    A = np.zeros((5,5), dtype=np.float64)
    A[0][0] = m2+m1+m0
    A[0][1] = 0
    A[0][2] = (m2+m1+m0)*cos(thr)*z0+l1*m2*cos(thr+th1)+(l0*m2+l0*m1)*cos(thr+th0)+(l0*m2+l0*m1)*cos(thr-th0)
    A[0][3] = (l0*m2+l0*m1)*cos(thr+th0)+((-l0*m2)-l0*m1)*cos(thr-th0)
    A[0][4] = l1*m2*cos(thr+th1)
    A[1][0] = 0
    A[1][1] = m2+m1+m0
    A[1][2] = ((-m2)-m1-m0)*sin(thr)*z0-l1*m2*sin(thr+th1)+((-l0*m2)-l0*m1)*sin(thr+th0)+((-l0*m2)-l0*m1)*sin(thr-th0)
    A[1][3] = ((-l0*m2)-l0*m1)*sin(thr+th0)+(l0*m2+l0*m1)*sin(thr-th0)
    A[1][4] = -l1*m2*sin(thr+th1)
    A[2][0] = (m2+m1+m0)*cos(thr)*z0+l1*m2*cos(thr+th1)+(l0*m2+l0*m1)*cos(thr+th0)+(l0*m2+l0*m1)*cos(thr-th0)
    A[2][1] = ((-m2)-m1-m0)*sin(thr)*z0-l1*m2*sin(thr+th1)+((-l0*m2)-l0*m1)*sin(thr+th0)+((-l0*m2)-l0*m1)*sin(thr-th0)
    A[2][2] = (m2+m1+m0)*z0**2+(2*l1*m2*cos(th1)+(4*l0*m2+4*l0*m1)*cos(th0))*z0+2*l0*l1*m2*cos(th1+th0)+2*l0*l1*m2*cos(th1-th0)+(2*l0**2*m2+2*l0**2*m1)*cos(2*th0)+(l1**2+2*l0**2)*m2+2*l0**2*m1
    A[2][3] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[2][4] = l1*m2*cos(th1)*z0+l0*l1*m2*cos(th1+th0)+l0*l1*m2*cos(th1-th0)+l1**2*m2
    A[3][0] = (l0*m2+l0*m1)*cos(thr+th0)+((-l0*m2)-l0*m1)*cos(thr-th0)
    A[3][1] = ((-l0*m2)-l0*m1)*sin(thr+th0)+(l0*m2+l0*m1)*sin(thr-th0)
    A[3][2] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[3][3] = ((-2*l0**2*m2)-2*l0**2*m1)*cos(2*th0)+2*l0**2*m2+2*l0**2*m1
    A[3][4] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[4][0] = l1*m2*cos(thr+th1)
    A[4][1] = -l1*m2*sin(thr+th1)
    A[4][2] = l1*m2*cos(th1)*z0+l0*l1*m2*cos(th1+th0)+l0*l1*m2*cos(th1-th0)+l1**2*m2
    A[4][3] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[4][4] = l1**2*m2

    b = np.zeros((5,1), dtype=np.float64)
    b[0] = (-dthr**2*m2*sin(thr)*z0)-dthr**2*m1*sin(thr)*z0-dthr**2*m0*sin(thr)*z0-dthr**2*l1*m2*sin(thr+th1)-2*dth1*dthr*l1*m2*sin(thr+th1)-dth1**2*l1*m2*sin(thr+th1)-2*dthr**2*l0*m2*cos(th0)*sin(thr)-2*dth0**2*l0*m2*cos(th0)*sin(thr)-2*dthr**2*l0*m1*cos(th0)*sin(thr)-2*dth0**2*l0*m1*cos(th0)*sin(thr)-4*dth0*dthr*l0*m2*sin(th0)*cos(thr)-4*dth0*dthr*l0*m1*sin(th0)*cos(thr)
    b[1] = (-dthr**2*m2*cos(thr)*z0)-dthr**2*m1*cos(thr)*z0-dthr**2*m0*cos(thr)*z0-dthr**2*l1*m2*cos(thr+th1)-2*dth1*dthr*l1*m2*cos(thr+th1)-dth1**2*l1*m2*cos(thr+th1)+4*dth0*dthr*l0*m2*sin(th0)*sin(thr)+4*dth0*dthr*l0*m1*sin(th0)*sin(thr)-2*dthr**2*l0*m2*cos(th0)*cos(thr)-2*dth0**2*l0*m2*cos(th0)*cos(thr)-2*dthr**2*l0*m1*cos(th0)*cos(thr)-2*dth0**2*l0*m1*cos(th0)*cos(thr)+g*m2+g*m1+g*m0
    b[2] = (-2*dth1*dthr*l1*m2*cos(thr)*sin(thr+th1)*z0)-dth1**2*l1*m2*cos(thr)*sin(thr+th1)*z0+2*dth1*dthr*l1*m2*sin(thr)*cos(thr+th1)*z0+dth1**2*l1*m2*sin(thr)*cos(thr+th1)*z0-4*dth0*dthr*l0*m2*sin(th0)*sin(thr)**2*z0-4*dth0*dthr*l0*m1*sin(th0)*sin(thr)**2*z0-g*m2*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0-4*dth0*dthr*l0*m2*sin(th0)*cos(thr)**2*z0-4*dth0*dthr*l0*m1*sin(th0)*cos(thr)**2*z0-4*dth0*dthr*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)-4*dth1*dthr*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)-2*dth1**2*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)+2*dth0**2*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)-g*l1*m2*sin(thr+th1)+4*dth1*dthr*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)+2*dth1**2*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)-2*dth0**2*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)-4*dth0*dthr*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)-8*dth0*dthr*l0**2*m2*cos(th0)*sin(th0)*sin(thr)**2-8*dth0*dthr*l0**2*m1*cos(th0)*sin(th0)*sin(thr)**2-2*g*l0*m2*cos(th0)*sin(thr)-2*g*l0*m1*cos(th0)*sin(thr)-8*dth0*dthr*l0**2*m2*cos(th0)*sin(th0)*cos(thr)**2-8*dth0*dthr*l0**2*m1*cos(th0)*sin(th0)*cos(thr)**2
    b[3] = 2*dthr**2*l0*m2*sin(th0)*sin(thr)**2*z0+2*dthr**2*l0*m1*sin(th0)*sin(thr)**2*z0+2*dthr**2*l0*m2*sin(th0)*cos(thr)**2*z0+2*dthr**2*l0*m1*sin(th0)*cos(thr)**2*z0+2*dthr**2*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+4*dth1*dthr*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+2*dth1**2*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+2*dthr**2*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)+4*dth1*dthr*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)+2*dth1**2*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)+4*dthr**2*l0**2*m2*cos(th0)*sin(th0)*sin(thr)**2+4*dth0**2*l0**2*m2*cos(th0)*sin(th0)*sin(thr)**2+4*dthr**2*l0**2*m1*cos(th0)*sin(th0)*sin(thr)**2+4*dth0**2*l0**2*m1*cos(th0)*sin(th0)*sin(thr)**2+4*dthr**2*l0**2*m2*cos(th0)*sin(th0)*cos(thr)**2+4*dth0**2*l0**2*m2*cos(th0)*sin(th0)*cos(thr)**2+4*dthr**2*l0**2*m1*cos(th0)*sin(th0)*cos(thr)**2+4*dth0**2*l0**2*m1*cos(th0)*sin(th0)*cos(thr)**2-2*g*l0*m2*sin(th0)*cos(thr)-2*g*l0*m1*sin(th0)*cos(thr)
    b[4] = dthr**2*l1*m2*cos(thr)*sin(thr+th1)*z0-dthr**2*l1*m2*sin(thr)*cos(thr+th1)*z0-4*dth0*dthr*l0*l1*m2*sin(th0)*sin(thr)*sin(thr+th1)+2*dthr**2*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)+2*dth0**2*l0*l1*m2*cos(th0)*cos(thr)*sin(thr+th1)-g*l1*m2*sin(thr+th1)-2*dthr**2*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)-2*dth0**2*l0*l1*m2*cos(th0)*sin(thr)*cos(thr+th1)-4*dth0*dthr*l0*l1*m2*sin(th0)*cos(thr)*cos(thr+th1)

    return A, b, extf

def f_air(t, s, u, params={}):
    A, b, extf = _calcAb55(s, u)
    assert np.linalg.matrix_rank(A) == 5, (s, A)
    #Ax + b = extf
    #print(np.linalg.det(A))
    y = np.linalg.solve(A, extf-b).reshape(5)

    ds = np.zeros_like(s)
    ds[IDX_xr] = s[IDX_dx]
    ds[IDX_yr] = s[IDX_dy]
    ds[IDX_z]  = 0
    ds[IDX_thr] = s[IDX_dthr]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_dx]   = y[0]
    ds[IDX_dy]   = y[1]
    ds[IDX_dthr] = y[2]
    ds[IDX_dz]   = 0
    ds[IDX_dth0] = y[3]
    ds[IDX_dth1] = y[4]

    return ds

def f_ground(t, s, u, params={}):
    A, b, extf = _calcAb44(s, u)
    assert np.linalg.matrix_rank(A) == 4
    y = np.linalg.solve(A, extf-b).reshape(4)

    ds = np.zeros_like(s)
    ds[IDX_xr] = 0
    ds[IDX_yr] = 0
    ds[IDX_z]   = s[IDX_dz]
    ds[IDX_thr] = s[IDX_dthr]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_dx]   = 0
    ds[IDX_dy]   = 0
    ds[IDX_dz]   = y[0]
    ds[IDX_dthr] = y[1]
    ds[IDX_dth0] = y[2]
    ds[IDX_dth1] = y[3]

    return ds


# only change velocity
# compensate trans velocity change with rotation change with f_air
def jump_time(s, ds, dt): # z = 0
    s1 = s + ds * dt
    z = s[IDX_z]
    z1 = s1[IDX_z]
    if z > 0 and z1 <= 0 or z < 0 and z1 >= 0:
        return z * dt / (z - z1)
    else:
        return None


def collision_time(s, ds, dt): # y0 = 0
    s1 = s + ds * dt
    yr = s[IDX_yr]
    yr1 = s1[IDX_yr]
    if yr > 0 and yr1 <= 0:
        return yr * dt / (yr - yr1)
    else:
        return None

def impulse_jump(s):
    new_s = s.copy()
    new_s[IDX_dx] = s[IDX_dz] * np.sin(s[IDX_thr])
    new_s[IDX_dy] = s[IDX_dz] * np.cos(s[IDX_thr])
    new_s[IDX_z]  = 0
    new_s[IDX_dz] = 0
    new_s[IDX_yr] = 0
    return new_s

def impulse_collision(s):
    new_s = s.copy()
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    z = s[IDX_z]
    dthr = s[IDX_dthr]
    thr = s[IDX_thr]
    dxr = s[IDX_dx]
    dyr = s[IDX_dy]
    A = np.zeros((2,2), dtype=np.float64)
    A[0][0] = ((-m2)-m1-m0)*sin(thr)**2+((-m2)-m1-m0)*cos(thr)**2
    A[0][1] = l1*m2*cos(thr)*sin(thr+th1)-l1*m2*sin(thr)*cos(thr+th1)
    A[1][0] = 0
    A[1][1] = (((-m2)-m1-m0)*sin(thr)**2+((-m2)-m1-m0)*cos(thr)**2)*z0+(((-m2)-m1-m0)*sin(thr)**2+((-m2)-m1-m0)*cos(thr)**2)*z-l1*m2*sin(thr)*sin(thr+th1)-l1*m2*cos(thr)*cos(thr+th1)+((-2*l0*m2)-2*l0*m1)*cos(th0)*sin(thr)**2+((-2*l0*m2)-2*l0*m1)*cos(th0)*cos(thr)**2

    b = np.zeros((2,1), dtype=np.float64)
    b[0] = (-dthr*l1*m2*cos(thr)*sin(thr+th1))+dthr*l1*m2*sin(thr)*cos(thr+th1)+dxr*m2*sin(thr)+dxr*m1*sin(thr)+dxr*m0*sin(thr)+dyr*m2*cos(thr)+dyr*m1*cos(thr)+dyr*m0*cos(thr)
    b[1] = dthr*m2*sin(thr)**2*z0+dthr*m1*sin(thr)**2*z0+dthr*m0*sin(thr)**2*z0+dthr*m2*cos(thr)**2*z0+dthr*m1*cos(thr)**2*z0+dthr*m0*cos(thr)**2*z0+dthr*l1*m2*sin(thr)*sin(thr+th1)+dthr*l1*m2*cos(thr)*cos(thr+th1)+2*dthr*l0*m2*cos(th0)*sin(thr)**2+2*dthr*l0*m1*cos(th0)*sin(thr)**2-dyr*m2*sin(thr)-dyr*m1*sin(thr)-dyr*m0*sin(thr)+2*dthr*l0*m2*cos(th0)*cos(thr)**2+2*dthr*l0*m1*cos(th0)*cos(thr)**2+dxr*m2*cos(thr)+dxr*m1*cos(thr)+dxr*m0*cos(thr)

    d = np.linalg.solve(A, -b).reshape(2)
    new_s[IDX_dz]   = d[0]
    new_s[IDX_dthr] = d[1]
    new_s[IDX_dx] = 0
    new_s[IDX_dy] = 0
    new_s[IDX_yr] = 0

    return new_s

def step(t, s, u, dt):
    u = torq_limit(s, u)
    if s[IDX_yr] == 0 and s[IDX_dx] == 0 and s[IDX_dy] == 0: # already on the ground
        assert -np.pi/2 < s[IDX_thr] and s[IDX_thr] < np.pi/2, state_dict(s)
        ds = rk4(f_ground, t, s, u, {}, dt)
        jmpt = jump_time(s, ds, dt)
        if jmpt is not None:
            ds = rk4(f_ground, t, s, u, {}, jmpt)
            s = s + ds * jmpt
            s = impulse_jump(s)
            return "jump", t + jmpt, postprocess(s)
        else:
            return "ground", t + dt, postprocess(s + ds * dt)
    elif s[IDX_yr] >= 0: # already in the air
        assert s[IDX_z] == 0 and s[IDX_dz] == 0, print_state(s)
        ds = rk4(f_air, t, s, u, {}, dt)
        colt = collision_time(s, ds, dt)
        if colt is not None:
            ds = rk4(f_air, t, s, u, {}, colt)
            s = s + ds * colt
            s = impulse_collision(s)
            return  "collision", t + colt, postprocess(s)
        else:
            return "air", t + dt, postprocess(s + ds * dt)
    else:
        assert False, s


def energyU(s):
    pr, p0, pk, p1, p2 = node_pos(s)
    return p0[1] * m0 * g + p1[1] * m1 * g + p2[1] * m2 * g + 1/2 * k * s[IDX_z] ** 2

def energyT(s):
    vr, v0, v1, v2 = node_vel(s)
    return m0 * v0 @ v0 / 2 + m1 * v1 @ v1 / 2 + m2 * v2 @ v2 / 2

def energy(s):
    return energyU(s) + energyT(s)

#------------
# control
#------------
def calc_thr(s):
    return s[IDX_thr]

def ref_clip(ref):
    return np.clip(ref, np.array([ref_min_th0, ref_min_th1]), np.array([ref_max_th0, ref_max_th1]))

def init_ref(s):
    return np.array([s[IDX_th0], s[IDX_th1]])

def inverse_model_d_th_ground(s, ddth):
    z   = s[IDX_z]
    thr = s[IDX_thr]
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    dz   = s[IDX_dz  ]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    ddth0 = ddth[0]
    ddth1 = ddth[1]

    A = np.zeros((4,4), dtype=np.float64)
    A[0][0] = m2+m1+m0
    A[0][1] = -l1*m2*sin(th1)
    A[0][2] = 0
    A[0][3] = 0
    A[1][0] = -l1*m2*sin(th1)
    A[1][1] = (m2+m1+m0)*z0**2+((2*m2+2*m1+2*m0)*z+2*l1*m2*cos(th1)+(4*l0*m2+4*l0*m1)*cos(th0))*z0+(m2+m1+m0)*z**2+(2*l1*m2*cos(th1)+(4*l0*m2+4*l0*m1)*cos(th0))*z+2*l0*l1*m2*cos(th1+th0)+2*l0*l1*m2*cos(th1-th0)+(2*l0**2*m2+2*l0**2*m1)*cos(2*th0)+(l1**2+2*l0**2)*m2+2*l0**2*m1
    A[1][2] = 0
    A[1][3] = 0
    A[2][0] = ((-2*l0*m2)-2*l0*m1)*sin(th0)
    A[2][1] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[2][2] = -1
    A[2][3] = 0
    A[3][0] = -l1*m2*sin(th1)
    A[3][1] = l1*m2*cos(th1)*z0+l1*m2*cos(th1)*z+l0*l1*m2*cos(th1+th0)+l0*l1*m2*cos(th1-th0)+l1**2*m2
    A[3][2] = 0
    A[3][3] = -1

    b = np.zeros((4,1), dtype=np.float64)
    b[0] = ((-dthr**2*m2)-dthr**2*m1-dthr**2*m0)*z0+((-dthr**2*m2)-dthr**2*m1-dthr**2*m0+k)*z+m2*(g*cos(thr)+l1*((-ddth1*sin(th1))-dthr**2*cos(th1)-2*dth1*dthr*cos(th1)-dth1**2*cos(th1))+l0*((-2*ddth0*sin(th0))-2*dthr**2*cos(th0)-2*dth0**2*cos(th0)))+m1*(g*cos(thr)+l0*((-2*ddth0*sin(th0))-2*dthr**2*cos(th0)-2*dth0**2*cos(th0)))+g*m0*cos(thr)+c*dz
    b[1] = (m2*((-g*sin(thr))+l1*((-2*dth1*dthr*sin(th1))-dth1**2*sin(th1)+ddth1*cos(th1))-4*dth0*dthr*l0*sin(th0)+2*dthr*dz)+m1*((-g*sin(thr))-4*dth0*dthr*l0*sin(th0)+2*dthr*dz)+m0*(2*dthr*dz-g*sin(thr)))*z0+(m2*((-g*sin(thr))+l1*((-2*dth1*dthr*sin(th1))-dth1**2*sin(th1)+ddth1*cos(th1))-4*dth0*dthr*l0*sin(th0)+2*dthr*dz)+m1*((-g*sin(thr))-4*dth0*dthr*l0*sin(th0)+2*dthr*dz)+m0*(2*dthr*dz-g*sin(thr)))*z+m2*(l1*((-g*sin(thr+th1))+l0*(dth0**2*(sin(th1+th0)+sin(th1-th0))+dth1**2*((-sin(th1+th0))-sin(th1-th0))+dthr*(dth0*(2*sin(th1-th0)-2*sin(th1+th0))+dth1*((-2*sin(th1+th0))-2*sin(th1-th0)))+ddth1*(cos(th1+th0)+cos(th1-th0))+ddth0*(cos(th1-th0)-cos(th1+th0)))+2*dthr*dz*cos(th1))+l0*(g*((-sin(thr+th0))-sin(thr-th0))+4*dthr*dz*cos(th0))-4*dth0*dthr*l0**2*sin(2*th0)+ddth1*l1**2)+m1*(l0*(g*((-sin(thr+th0))-sin(thr-th0))+4*dthr*dz*cos(th0))-4*dth0*dthr*l0**2*sin(2*th0))
    b[2] = (2*dthr**2*l0*m2*sin(th0)+2*dthr**2*l0*m1*sin(th0))*z0+(2*dthr**2*l0*m2*sin(th0)+2*dthr**2*l0*m1*sin(th0))*z+m2*(g*l0*(sin(thr-th0)-sin(thr+th0))+l0*l1*(dth1*dthr*(2*sin(th1+th0)-2*sin(th1-th0))+dthr**2*(sin(th1+th0)-sin(th1-th0))+dth1**2*(sin(th1+th0)-sin(th1-th0))+ddth1*(cos(th1-th0)-cos(th1+th0)))+l0**2*(2*dthr**2*sin(2*th0)+2*dth0**2*sin(2*th0)+ddth0*(2-2*cos(2*th0))))+m1*(g*l0*(sin(thr-th0)-sin(thr+th0))+l0**2*(2*dthr**2*sin(2*th0)+2*dth0**2*sin(2*th0)+ddth0*(2-2*cos(2*th0))))
    b[3] = dthr**2*l1*m2*sin(th1)*z0+dthr**2*l1*m2*sin(th1)*z+m2*(l1*((-g*sin(thr+th1))+l0*(dthr**2*(sin(th1+th0)+sin(th1-th0))+dth0**2*(sin(th1+th0)+sin(th1-th0))+dth0*dthr*(2*sin(th1-th0)-2*sin(th1+th0))+ddth0*(cos(th1-th0)-cos(th1+th0)))+2*dthr*dz*cos(th1))+ddth1*l1**2)

    #y : symbolilze([z'', th0'', tau0, tau1])$
    y = np.linalg.solve(A, -b).reshape(4)
    return np.array([y[2], y[3]])

def inverse_model_d_th_air(s, ddth):
    xr   = s[IDX_xr]
    yr   = s[IDX_yr]
    thr  = s[IDX_thr]
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    ddth0 = ddth[0]
    ddth1 = ddth[1]
    A = np.zeros((5,5), dtype=np.float64)
    A[0][0] = m2+m1+m0
    A[0][1] = 0
    A[0][2] = (m2+m1+m0)*cos(thr)*z0+l1*m2*cos(thr+th1)+(l0*m2+l0*m1)*cos(thr+th0)+(l0*m2+l0*m1)*cos(thr-th0)
    A[0][3] = 0
    A[0][4] = 0
    A[1][0] = 0
    A[1][1] = m2+m1+m0
    A[1][2] = ((-m2)-m1-m0)*sin(thr)*z0-l1*m2*sin(thr+th1)+((-l0*m2)-l0*m1)*sin(thr+th0)+((-l0*m2)-l0*m1)*sin(thr-th0)
    A[1][3] = 0
    A[1][4] = 0
    A[2][0] = (m2+m1+m0)*cos(thr)*z0+l1*m2*cos(thr+th1)+(l0*m2+l0*m1)*cos(thr+th0)+(l0*m2+l0*m1)*cos(thr-th0)
    A[2][1] = ((-m2)-m1-m0)*sin(thr)*z0-l1*m2*sin(thr+th1)+((-l0*m2)-l0*m1)*sin(thr+th0)+((-l0*m2)-l0*m1)*sin(thr-th0)
    A[2][2] = (m2+m1+m0)*z0**2+(2*l1*m2*cos(th1)+(4*l0*m2+4*l0*m1)*cos(th0))*z0+2*l0*l1*m2*cos(th1+th0)+2*l0*l1*m2*cos(th1-th0)+(2*l0**2*m2+2*l0**2*m1)*cos(2*th0)+(l1**2+2*l0**2)*m2+2*l0**2*m1
    A[2][3] = 0
    A[2][4] = 0
    A[3][0] = (l0*m2+l0*m1)*cos(thr+th0)+((-l0*m2)-l0*m1)*cos(thr-th0)
    A[3][1] = ((-l0*m2)-l0*m1)*sin(thr+th0)+(l0*m2+l0*m1)*sin(thr-th0)
    A[3][2] = l0*l1*m2*cos(th1-th0)-l0*l1*m2*cos(th1+th0)
    A[3][3] = -1
    A[3][4] = 0
    A[4][0] = l1*m2*cos(thr+th1)
    A[4][1] = -l1*m2*sin(thr+th1)
    A[4][2] = l1*m2*cos(th1)*z0+l0*l1*m2*cos(th1+th0)+l0*l1*m2*cos(th1-th0)+l1**2*m2
    A[4][3] = 0
    A[4][4] = -1

    b = np.zeros((5,1), dtype=np.float64)
    b[0] = ((-dthr**2*m2*sin(thr))-dthr**2*m1*sin(thr)-dthr**2*m0*sin(thr))*z0+m2*(l1*((-dthr**2*sin(thr+th1))-2*dth1*dthr*sin(thr+th1)-dth1**2*sin(thr+th1)+ddth1*cos(thr+th1))+l0*(dthr**2*((-sin(thr+th0))-sin(thr-th0))+dth0**2*((-sin(thr+th0))-sin(thr-th0))+dth0*dthr*(2*sin(thr-th0)-2*sin(thr+th0))+ddth0*(cos(thr+th0)-cos(thr-th0))))+l0*m1*(dthr**2*((-sin(thr+th0))-sin(thr-th0))+dth0**2*((-sin(thr+th0))-sin(thr-th0))+dth0*dthr*(2*sin(thr-th0)-2*sin(thr+th0))+ddth0*(cos(thr+th0)-cos(thr-th0)))
    b[1] = ((-dthr**2*m2*cos(thr))-dthr**2*m1*cos(thr)-dthr**2*m0*cos(thr))*z0+m2*(l1*((-ddth1*sin(thr+th1))-dthr**2*cos(thr+th1)-2*dth1*dthr*cos(thr+th1)-dth1**2*cos(thr+th1))+l0*(ddth0*(sin(thr-th0)-sin(thr+th0))+dthr**2*((-cos(thr+th0))-cos(thr-th0))+dth0**2*((-cos(thr+th0))-cos(thr-th0))+dth0*dthr*(2*cos(thr-th0)-2*cos(thr+th0)))+g)+m1*(l0*(ddth0*(sin(thr-th0)-sin(thr+th0))+dthr**2*((-cos(thr+th0))-cos(thr-th0))+dth0**2*((-cos(thr+th0))-cos(thr-th0))+dth0*dthr*(2*cos(thr-th0)-2*cos(thr+th0)))+g)+g*m0
    b[2] = (m2*((-g*sin(thr))+l1*((-2*dth1*dthr*sin(th1))-dth1**2*sin(th1)+ddth1*cos(th1))-4*dth0*dthr*l0*sin(th0))+m1*((-g*sin(thr))-4*dth0*dthr*l0*sin(th0))-g*m0*sin(thr))*z0+m2*(l1*(l0*(dth0**2*(sin(th1+th0)+sin(th1-th0))+dth1**2*((-sin(th1+th0))-sin(th1-th0))+dthr*(dth0*(2*sin(th1-th0)-2*sin(th1+th0))+dth1*((-2*sin(th1+th0))-2*sin(th1-th0)))+ddth1*(cos(th1+th0)+cos(th1-th0))+ddth0*(cos(th1-th0)-cos(th1+th0)))-g*sin(thr+th1))+g*l0*((-sin(thr+th0))-sin(thr-th0))-4*dth0*dthr*l0**2*sin(2*th0)+ddth1*l1**2)+m1*(g*l0*((-sin(thr+th0))-sin(thr-th0))-4*dth0*dthr*l0**2*sin(2*th0))
    b[3] = (2*dthr**2*l0*m2*sin(th0)+2*dthr**2*l0*m1*sin(th0))*z0+m2*(g*l0*(sin(thr-th0)-sin(thr+th0))+l0*l1*(dth1*dthr*(2*sin(th1+th0)-2*sin(th1-th0))+dthr**2*(sin(th1+th0)-sin(th1-th0))+dth1**2*(sin(th1+th0)-sin(th1-th0))+ddth1*(cos(th1-th0)-cos(th1+th0)))+l0**2*(2*dthr**2*sin(2*th0)+2*dth0**2*sin(2*th0)+ddth0*(2-2*cos(2*th0))))+m1*(g*l0*(sin(thr-th0)-sin(thr+th0))+l0**2*(2*dthr**2*sin(2*th0)+2*dth0**2*sin(2*th0)+ddth0*(2-2*cos(2*th0))))
    b[4] = dthr**2*l1*m2*sin(th1)*z0+m2*(l1*(l0*(dthr**2*(sin(th1+th0)+sin(th1-th0))+dth0**2*(sin(th1+th0)+sin(th1-th0))+dth0*dthr*(2*sin(th1-th0)-2*sin(th1+th0))+ddth0*(cos(th1-th0)-cos(th1+th0)))-g*sin(thr+th1))+ddth1*l1**2)

    #y : symbolilze([xr'', yr'', thr'', tau0, tau1])$
    y = np.linalg.solve(A, -b).reshape(5)
    return np.array([y[3], y[4]])

# (d, th1) --PD--> u
def pdcontrol_th(s, ref):
    dob  = np.array([s[IDX_dth0], s[IDX_dth1]])
    ob = np.array([s[IDX_th0], s[IDX_th1]])
    err = ref - ob
    ddth = err * Kp - Kd * dob

    if s[IDX_yr] == 0 and s[IDX_dx] == 0 and s[IDX_dy] == 0: # already on the ground
        assert -np.pi/2 < s[IDX_thr] and s[IDX_thr] < np.pi/2, state_dict(s)
        u = inverse_model_d_th_ground(s, ddth)
    elif s[IDX_yr] >= 0: # already in the air
        assert s[IDX_z] == 0 and s[IDX_dz] == 0, state_dict(s)
        u = inverse_model_d_th_air(s, ddth)
    else:
        assert False, state_dict(s)
    print("pd-input: P,D = ", err * Kp, - Kd * dob, u)
    return u

