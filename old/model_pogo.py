import numpy as np
from numpy import sin, cos, abs
import sys


# Pogo
max_z = 0.4
max_d = 0.3
z0 =  0.4
l0 =  0.8 - max_d
l1 =  0.6

m0 = 10
m1 = 25
m2 = 35
g  = 9.8
#g  = 0
#T/4=0.15
k  = 20000 # (2*pi * T) ** 2 * M
# knee 50-250deg/sec

MAX_FORCE_D=3000   # knee flex 200Nm  ext 300Nm
MAX_TORQUE1=200

#Kp = 800
Kp = 12000
Kd = Kp * (0.1)

#-----------------
# State
#-----------------

IDX_xr   = 0
IDX_yr   = 1
IDX_z    = 2
IDX_d    = 3
IDX_th0  = 4
IDX_th1  = 5
IDX_dx   = 6
IDX_dy   = 7
IDX_dz   = 8
IDX_dd   = 9
IDX_dth0 = 10
IDX_dth1 = 11
IDX_MAX = 12

def reset_state(np_random=None):
    s = np.zeros(IDX_MAX, dtype=np.float64)
    #s[IDX_yr]  = 1
    s[IDX_yr]  = 0.01
    #s[IDX_dy]  = -5
    #s[IDX_yr]  = 0.3
    #s[IDX_dth1]  = 5
    s[IDX_th0]  = np.pi/2

    #s[IDX_th0]  = np.pi/4
    #s[IDX_th1]  = np.pi*3/4 - np.pi/4
    return s

def state_dict(s):
    return { "xr  " :  s[IDX_xr ]
           , "yr  " :  s[IDX_yr ]
           , "z   " :  s[IDX_z  ]
           , "d   " :  s[IDX_d  ]
           , "th0 " :  s[IDX_th0]
           , "th1 " :  s[IDX_th1]
           , "dx  " :  s[IDX_dx ]
           , "dy  " :  s[IDX_dy ]
           , "dz  " :  s[IDX_dz ]
           , "dd  " :  s[IDX_dd ]
           , "dth0" :  s[IDX_dth0]
           , "dth1" :  s[IDX_dth1]
           }

def print_state(s):
    print(f"xr:  {s[IDX_xr ]:.3f}", end=", ")
    print(f"yr:  {s[IDX_yr ]:.3f}", end=", ")
    print(f"z :  {s[IDX_z  ]:.3f}", end=", ")
    print(f"d :  {s[IDX_d  ]:.6f}", end=", ")
    print(f"th0: {s[IDX_th0]:.3f}", end=", ")
    print(f"th1: {s[IDX_th1]:.3f}")
    print(f"dx:  {s[IDX_dx  ]:.3f}", end=", ")
    print(f"dy:  {s[IDX_dy  ]:.3f}", end=", ")
    print(f"dz:  {s[IDX_dz  ]:.3f}", end=", ")
    print(f"dd:  {s[IDX_dd  ]:.6f}", end=", ")
    print(f"dth0:{s[IDX_dth0]:.3f}", end=", ")
    print(f"dth1:{s[IDX_dth1]:.3f}")

#-----------------
# Kinematics
#-----------------

def check_invariant(s):
    if abs(s[IDX_z]) > max_z:
        print("|z| is too big")
        return False
    if abs(s[IDX_d]) > max_d:
        print("|d| is too big")
        return False
    pr, p0, p1, p2 = node_pos(s)
    if p0[1] < 0:
        print(f"GAME OVER p0={p0:}")
        return False
    if p1[1] < 0:
        print(f"GAME OVER p1={p1:}")
        return False
    if p2[1] < 0:
        print(f"GAME OVER p2={p2:}")
        return False
    return True

def clip_z(s, check=False):
    new_s = s.copy()
    if check:
        assert new_s[IDX_z] == s[IDX_z], (new_s[IDX_z], s[IDX_z])
    return new_s

def postprocess(s):
    new_s = s.copy()
    # normalize_angle
    new_s[IDX_th0] = ((s[IDX_th0] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th1] = ((s[IDX_th1] + np.pi)% (2*np.pi)) - np.pi
    # max_z_abs
    return clip_z(new_s)

def node_pos(s):
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    d   = s[IDX_d]
    z   = s[IDX_z]

    pr = s[IDX_xr:IDX_yr+1].copy()
    p0 = pr + (z0 + z) * np.array([np.cos(th0), np.sin(th0)])
    p1 = p0 + (l0 + d) * np.array([np.cos(th0), np.sin(th0)])
    p2 = p1 + l1 * np.array([np.cos(th0+th1), np.sin(th0+th1)])

    return pr, p0, p1, p2

def node_vel(s):
    z   = s[IDX_z]
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    d   = s[IDX_d]
    dz   = s[IDX_dz]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dd   = s[IDX_dd]

    vr = s[IDX_dx:IDX_dy+1]
    v0 = vr + np.array([cos(th0), sin(th0)]) * dz + np.array([-sin(th0), cos(th0)]) * (z+z0) * dth0
    v1 = v0 + np.array([cos(th0), sin(th0)]) * dd + np.array([-sin(th0), cos(th0)]) * (d+l0) * dth0
    v2 = v1 + np.array([-sin(th0+th1), cos(th0+th1)]) * l1 * (dth0+dth1)

    return vr, v0, v1, v2

#-----------------
# Dynamics Util
#-----------------
def rk4(f, t, s, u, params, dt):

    #k1 = f(t,        s,             u, params)
    #k2 = f(t + dt/2, s + dt/2 * k1, u, params)
    #k3 = f(t + dt/2, s + dt/2 * k2, u, params)
    #k4 = f(t + dt,   s + dt * k3,   u, params)
    k1 = f(t,        clip_z(s            ), u, params)
    k2 = f(t + dt/2, clip_z(s + dt/2 * k1), u, params)
    k3 = f(t + dt/2, clip_z(s + dt/2 * k2), u, params)
    k4 = f(t + dt,   clip_z(s + dt * k3  ), u, params)

    return (k1 + 2*k2 + 2*k3 + k4)/6


#----------------------------
# Dynamics
#----------------------------
def max_u():
    return np.array([MAX_FORCE_D, 0])

def torq_limit(s, u):
    m = max_u()
    ret = np.clip(u, -m, m)
    #if (ret != u).any():
    #    print("clipped!", ret, u)
    return ret

def _calcAb44(s, u):
    z    = s[IDX_z]
    d    = s[IDX_d]
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    dz   = s[IDX_dz  ]
    dd   = s[IDX_dd  ]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    fz0  = 0
    fd   = u[0]
    tau0 = 0
    tau1 = u[1]
    extf = np.array([fz0, fd, tau0, tau1]).reshape(4,1)

    A = np.zeros((4,4), dtype=np.float64)
    A[0][0] = m2+m1+m0
    A[0][1] = m2+m1
    A[0][2] = -l1*m2*sin(th1)
    A[0][3] = -l1*m2*sin(th1)
    A[1][0] = m2+m1
    A[1][1] = m2+m1
    A[1][2] = -l1*m2*sin(th1)
    A[1][3] = -l1*m2*sin(th1)
    A[2][0] = -l1*m2*sin(th1)
    A[2][1] = -l1*m2*sin(th1)
    A[2][2] = (m2+m1+m0)*z0**2+((2*m2+2*m1+2*m0)*z+2*l1*m2*cos(th1)+(2*l0+2*d)*m2+(2*l0+2*d)*m1)*z0+(m2+m1+m0)*z**2+(2*l1*m2*cos(th1)+(2*l0+2*d)*m2+(2*l0+2*d)*m1)*z+(2*l0+2*d)*l1*m2*cos(th1)+(l1**2+l0**2+2*d*l0+d**2)*m2+(l0**2+2*d*l0+d**2)*m1
    A[2][3] = l1*m2*cos(th1)*z0+l1*m2*cos(th1)*z+(l0+d)*l1*m2*cos(th1)+l1**2*m2
    A[3][0] = -l1*m2*sin(th1)
    A[3][1] = -l1*m2*sin(th1)
    A[3][2] = l1*m2*cos(th1)*z0+l1*m2*cos(th1)*z+(l0+d)*l1*m2*cos(th1)+l1**2*m2
    A[3][3] = l1**2*m2

    b = np.zeros((4,1), dtype=np.float64)
    b[0] = (-dth0**2*m2*sin(th0)**2*z0)-dth0**2*m1*sin(th0)**2*z0-dth0**2*m0*sin(th0)**2*z0-dth0**2*m2*cos(th0)**2*z0-dth0**2*m1*cos(th0)**2*z0-dth0**2*m0*cos(th0)**2*z0-dth0**2*m2*sin(th0)**2*z-dth0**2*m1*sin(th0)**2*z-dth0**2*m0*sin(th0)**2*z-dth0**2*m2*cos(th0)**2*z-dth0**2*m1*cos(th0)**2*z-dth0**2*m0*cos(th0)**2*z+k*z-dth1**2*l1*m2*sin(th0)*sin(th1+th0)-2*dth0*dth1*l1*m2*sin(th0)*sin(th1+th0)-dth0**2*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l1*m2*cos(th0)*cos(th1+th0)-2*dth0*dth1*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l0*m2*sin(th0)**2-d*dth0**2*m2*sin(th0)**2-dth0**2*l0*m1*sin(th0)**2-d*dth0**2*m1*sin(th0)**2+g*m2*sin(th0)+g*m1*sin(th0)+g*m0*sin(th0)-dth0**2*l0*m2*cos(th0)**2-d*dth0**2*m2*cos(th0)**2-dth0**2*l0*m1*cos(th0)**2-d*dth0**2*m1*cos(th0)**2
    b[1] = (-dth0**2*m2*sin(th0)**2*z0)-dth0**2*m1*sin(th0)**2*z0-dth0**2*m2*cos(th0)**2*z0-dth0**2*m1*cos(th0)**2*z0-dth0**2*m2*sin(th0)**2*z-dth0**2*m1*sin(th0)**2*z-dth0**2*m2*cos(th0)**2*z-dth0**2*m1*cos(th0)**2*z-dth1**2*l1*m2*sin(th0)*sin(th1+th0)-2*dth0*dth1*l1*m2*sin(th0)*sin(th1+th0)-dth0**2*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l1*m2*cos(th0)*cos(th1+th0)-2*dth0*dth1*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l0*m2*sin(th0)**2-d*dth0**2*m2*sin(th0)**2-dth0**2*l0*m1*sin(th0)**2-d*dth0**2*m1*sin(th0)**2+g*m2*sin(th0)+g*m1*sin(th0)-dth0**2*l0*m2*cos(th0)**2-d*dth0**2*m2*cos(th0)**2-dth0**2*l0*m1*cos(th0)**2-d*dth0**2*m1*cos(th0)**2
    b[2] = (-dth1**2*l1*m2*cos(th0)*sin(th1+th0)*z0)-2*dth0*dth1*l1*m2*cos(th0)*sin(th1+th0)*z0+dth1**2*l1*m2*sin(th0)*cos(th1+th0)*z0+2*dth0*dth1*l1*m2*sin(th0)*cos(th1+th0)*z0+2*dth0*dz*m2*sin(th0)**2*z0+2*dd*dth0*m2*sin(th0)**2*z0+2*dth0*dz*m1*sin(th0)**2*z0+2*dd*dth0*m1*sin(th0)**2*z0+2*dth0*dz*m0*sin(th0)**2*z0+2*dth0*dz*m2*cos(th0)**2*z0+2*dd*dth0*m2*cos(th0)**2*z0+2*dth0*dz*m1*cos(th0)**2*z0+2*dd*dth0*m1*cos(th0)**2*z0+2*dth0*dz*m0*cos(th0)**2*z0+g*m2*cos(th0)*z0+g*m1*cos(th0)*z0+g*m0*cos(th0)*z0-dth1**2*l1*m2*cos(th0)*sin(th1+th0)*z-2*dth0*dth1*l1*m2*cos(th0)*sin(th1+th0)*z+dth1**2*l1*m2*sin(th0)*cos(th1+th0)*z+2*dth0*dth1*l1*m2*sin(th0)*cos(th1+th0)*z+2*dth0*dz*m2*sin(th0)**2*z+2*dd*dth0*m2*sin(th0)**2*z+2*dth0*dz*m1*sin(th0)**2*z+2*dd*dth0*m1*sin(th0)**2*z+2*dth0*dz*m0*sin(th0)**2*z+2*dth0*dz*m2*cos(th0)**2*z+2*dd*dth0*m2*cos(th0)**2*z+2*dth0*dz*m1*cos(th0)**2*z+2*dd*dth0*m1*cos(th0)**2*z+2*dth0*dz*m0*cos(th0)**2*z+g*m2*cos(th0)*z+g*m1*cos(th0)*z+g*m0*cos(th0)*z+2*dth0*dz*l1*m2*sin(th0)*sin(th1+th0)+2*dd*dth0*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l0*l1*m2*cos(th0)*sin(th1+th0)-2*dth0*dth1*l0*l1*m2*cos(th0)*sin(th1+th0)-d*dth1**2*l1*m2*cos(th0)*sin(th1+th0)-2*d*dth0*dth1*l1*m2*cos(th0)*sin(th1+th0)+dth1**2*l0*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dth1*l0*l1*m2*sin(th0)*cos(th1+th0)+d*dth1**2*l1*m2*sin(th0)*cos(th1+th0)+2*d*dth0*dth1*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dz*l1*m2*cos(th0)*cos(th1+th0)+2*dd*dth0*l1*m2*cos(th0)*cos(th1+th0)+g*l1*m2*cos(th1+th0)+2*dth0*dz*l0*m2*sin(th0)**2+2*dd*dth0*l0*m2*sin(th0)**2+2*d*dth0*dz*m2*sin(th0)**2+2*d*dd*dth0*m2*sin(th0)**2+2*dth0*dz*l0*m1*sin(th0)**2+2*dd*dth0*l0*m1*sin(th0)**2+2*d*dth0*dz*m1*sin(th0)**2+2*d*dd*dth0*m1*sin(th0)**2+2*dth0*dz*l0*m2*cos(th0)**2+2*dd*dth0*l0*m2*cos(th0)**2+2*d*dth0*dz*m2*cos(th0)**2+2*d*dd*dth0*m2*cos(th0)**2+2*dth0*dz*l0*m1*cos(th0)**2+2*dd*dth0*l0*m1*cos(th0)**2+2*d*dth0*dz*m1*cos(th0)**2+2*d*dd*dth0*m1*cos(th0)**2+g*l0*m2*cos(th0)+d*g*m2*cos(th0)+g*l0*m1*cos(th0)+d*g*m1*cos(th0)
    b[3] = dth0**2*l1*m2*cos(th0)*sin(th1+th0)*z0-dth0**2*l1*m2*sin(th0)*cos(th1+th0)*z0+dth0**2*l1*m2*cos(th0)*sin(th1+th0)*z-dth0**2*l1*m2*sin(th0)*cos(th1+th0)*z+2*dth0*dz*l1*m2*sin(th0)*sin(th1+th0)+2*dd*dth0*l1*m2*sin(th0)*sin(th1+th0)+dth0**2*l0*l1*m2*cos(th0)*sin(th1+th0)+d*dth0**2*l1*m2*cos(th0)*sin(th1+th0)-dth0**2*l0*l1*m2*sin(th0)*cos(th1+th0)-d*dth0**2*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dz*l1*m2*cos(th0)*cos(th1+th0)+2*dd*dth0*l1*m2*cos(th0)*cos(th1+th0)+g*l1*m2*cos(th1+th0)
    return A, b, extf

def _calcAb55(s, u):

    xr   = s[IDX_xr]
    yr   = s[IDX_yr]
    d    = s[IDX_d]
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    dd   = s[IDX_dd  ]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    fd   = u[0]
    tau0 = 0
    tau1 = u[1]
    extf = np.array([0, 0, fd, tau0, tau1]).reshape(5,1)

    A = np.zeros((5,5), dtype=np.float64)
    A[0][0] = m2+m1+m0
    A[0][1] = 0
    A[0][2] = (m2+m1)*cos(th0)
    A[0][3] = ((-m2)-m1-m0)*sin(th0)*z0-l1*m2*sin(th1+th0)+(((-l0)-d)*m2+((-l0)-d)*m1)*sin(th0)
    A[0][4] = -l1*m2*sin(th1+th0)
    A[1][0] = 0
    A[1][1] = m2+m1+m0
    A[1][2] = (m2+m1)*sin(th0)
    A[1][3] = (m2+m1+m0)*cos(th0)*z0+l1*m2*cos(th1+th0)+((l0+d)*m2+(l0+d)*m1)*cos(th0)
    A[1][4] = l1*m2*cos(th1+th0)
    A[2][0] = (m2+m1)*cos(th0)
    A[2][1] = (m2+m1)*sin(th0)
    A[2][2] = m2+m1
    A[2][3] = -l1*m2*sin(th1)
    A[2][4] = -l1*m2*sin(th1)
    A[3][0] = ((-m2)-m1-m0)*sin(th0)*z0-l1*m2*sin(th1+th0)+(((-l0)-d)*m2+((-l0)-d)*m1)*sin(th0)
    A[3][1] = (m2+m1+m0)*cos(th0)*z0+l1*m2*cos(th1+th0)+((l0+d)*m2+(l0+d)*m1)*cos(th0)
    A[3][2] = -l1*m2*sin(th1)
    A[3][3] = (m2+m1+m0)*z0**2+(2*l1*m2*cos(th1)+(2*l0+2*d)*m2+(2*l0+2*d)*m1)*z0+(2*l0+2*d)*l1*m2*cos(th1)+(l1**2+l0**2+2*d*l0+d**2)*m2+(l0**2+2*d*l0+d**2)*m1
    A[3][4] = l1*m2*cos(th1)*z0+(l0+d)*l1*m2*cos(th1)+l1**2*m2
    A[4][0] = -l1*m2*sin(th1+th0)
    A[4][1] = l1*m2*cos(th1+th0)
    A[4][2] = -l1*m2*sin(th1)
    A[4][3] = l1*m2*cos(th1)*z0+(l0+d)*l1*m2*cos(th1)+l1**2*m2
    A[4][4] = l1**2*m2

    b = np.zeros((5,1), dtype=np.float64)
    b[0] = (-dth0**2*m2*cos(th0)*z0)-dth0**2*m1*cos(th0)*z0-dth0**2*m0*cos(th0)*z0-dth1**2*l1*m2*cos(th1+th0)-2*dth0*dth1*l1*m2*cos(th1+th0)-dth0**2*l1*m2*cos(th1+th0)-2*dd*dth0*m2*sin(th0)-2*dd*dth0*m1*sin(th0)-dth0**2*l0*m2*cos(th0)-d*dth0**2*m2*cos(th0)-dth0**2*l0*m1*cos(th0)-d*dth0**2*m1*cos(th0)
    b[1] = (-dth0**2*m2*sin(th0)*z0)-dth0**2*m1*sin(th0)*z0-dth0**2*m0*sin(th0)*z0-dth1**2*l1*m2*sin(th1+th0)-2*dth0*dth1*l1*m2*sin(th1+th0)-dth0**2*l1*m2*sin(th1+th0)-dth0**2*l0*m2*sin(th0)-d*dth0**2*m2*sin(th0)-dth0**2*l0*m1*sin(th0)-d*dth0**2*m1*sin(th0)+2*dd*dth0*m2*cos(th0)+2*dd*dth0*m1*cos(th0)+g*m2+g*m1+g*m0
    b[2] = (-dth0**2*m2*sin(th0)**2*z0)-dth0**2*m1*sin(th0)**2*z0-dth0**2*m2*cos(th0)**2*z0-dth0**2*m1*cos(th0)**2*z0-dth1**2*l1*m2*sin(th0)*sin(th1+th0)-2*dth0*dth1*l1*m2*sin(th0)*sin(th1+th0)-dth0**2*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l1*m2*cos(th0)*cos(th1+th0)-2*dth0*dth1*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l0*m2*sin(th0)**2-d*dth0**2*m2*sin(th0)**2-dth0**2*l0*m1*sin(th0)**2-d*dth0**2*m1*sin(th0)**2+g*m2*sin(th0)+g*m1*sin(th0)-dth0**2*l0*m2*cos(th0)**2-d*dth0**2*m2*cos(th0)**2-dth0**2*l0*m1*cos(th0)**2-d*dth0**2*m1*cos(th0)**2
    b[3] = (-dth1**2*l1*m2*cos(th0)*sin(th1+th0)*z0)-2*dth0*dth1*l1*m2*cos(th0)*sin(th1+th0)*z0+dth1**2*l1*m2*sin(th0)*cos(th1+th0)*z0+2*dth0*dth1*l1*m2*sin(th0)*cos(th1+th0)*z0+2*dd*dth0*m2*sin(th0)**2*z0+2*dd*dth0*m1*sin(th0)**2*z0+2*dd*dth0*m2*cos(th0)**2*z0+2*dd*dth0*m1*cos(th0)**2*z0+g*m2*cos(th0)*z0+g*m1*cos(th0)*z0+g*m0*cos(th0)*z0+2*dd*dth0*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l0*l1*m2*cos(th0)*sin(th1+th0)-2*dth0*dth1*l0*l1*m2*cos(th0)*sin(th1+th0)-d*dth1**2*l1*m2*cos(th0)*sin(th1+th0)-2*d*dth0*dth1*l1*m2*cos(th0)*sin(th1+th0)+dth1**2*l0*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dth1*l0*l1*m2*sin(th0)*cos(th1+th0)+d*dth1**2*l1*m2*sin(th0)*cos(th1+th0)+2*d*dth0*dth1*l1*m2*sin(th0)*cos(th1+th0)+2*dd*dth0*l1*m2*cos(th0)*cos(th1+th0)+g*l1*m2*cos(th1+th0)+2*dd*dth0*l0*m2*sin(th0)**2+2*d*dd*dth0*m2*sin(th0)**2+2*dd*dth0*l0*m1*sin(th0)**2+2*d*dd*dth0*m1*sin(th0)**2+2*dd*dth0*l0*m2*cos(th0)**2+2*d*dd*dth0*m2*cos(th0)**2+2*dd*dth0*l0*m1*cos(th0)**2+2*d*dd*dth0*m1*cos(th0)**2+g*l0*m2*cos(th0)+d*g*m2*cos(th0)+g*l0*m1*cos(th0)+d*g*m1*cos(th0)
    b[4] = dth0**2*l1*m2*cos(th0)*sin(th1+th0)*z0-dth0**2*l1*m2*sin(th0)*cos(th1+th0)*z0+2*dd*dth0*l1*m2*sin(th0)*sin(th1+th0)+dth0**2*l0*l1*m2*cos(th0)*sin(th1+th0)+d*dth0**2*l1*m2*cos(th0)*sin(th1+th0)-dth0**2*l0*l1*m2*sin(th0)*cos(th1+th0)-d*dth0**2*l1*m2*sin(th0)*cos(th1+th0)+2*dd*dth0*l1*m2*cos(th0)*cos(th1+th0)+g*l1*m2*cos(th1+th0)

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
    ds[IDX_d]   = s[IDX_dd]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_dx]   = y[0]
    ds[IDX_dy]   = y[1]
    ds[IDX_dd]   = y[2]
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
    ds[IDX_d]   = s[IDX_dd]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_dx]   = 0
    ds[IDX_dy]   = 0
    ds[IDX_dz]   = y[0]
    ds[IDX_dd]   = y[1]
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
    new_s[IDX_dx] = s[IDX_dz] * np.cos(s[IDX_th0])
    new_s[IDX_dy] = s[IDX_dz] * np.sin(s[IDX_th0])
    new_s[IDX_z]  = 0
    new_s[IDX_dz] = 0
    new_s[IDX_yr] = 0
    return new_s

def impulse_collision(s):
    new_s = s.copy()
    b = np.zeros((2,1), dtype=np.float64)
    dth0 = s[IDX_dth0]
    th0 = s[IDX_th0]
    dxr = s[IDX_dx]
    dyr = s[IDX_dy]
    b[0] = (-dth0*sin(th0)*(z0))  + dxr
    b[1] =  dth0*cos(th0)*(z0)    + dyr
    A = np.zeros((2,2), dtype=np.float64)
    A[0][0] = cos(th0)
    A[0][1] = -sin(th0) * z0 
    A[1][0] = sin(th0)
    A[1][1] = cos(th0)*(z0)
    d = np.linalg.solve(A, b).reshape(2)
    new_s[IDX_dz]   = d[0]
    new_s[IDX_dth0] = d[1]

    new_s[IDX_dx] = 0
    new_s[IDX_dy] = 0
    new_s[IDX_yr] = 0
    return new_s

def step(t, s, u, dt):
    u = torq_limit(s, u)
    if s[IDX_yr] == 0 and s[IDX_dx] == 0 and s[IDX_dy] == 0: # already on the ground
        assert 0 < s[IDX_th0] and s[IDX_th0] < np.pi, print_state(s)
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
    pr, p0, p1, p2 = node_pos(s)
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
    return s[IDX_th0]

def init_ref(s):
    return np.array([s[IDX_d], s[IDX_th1]])

def inverse_model_d_th_ground(s, dddth1):
    z    = s[IDX_z]
    d    = s[IDX_d]
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    dz   = s[IDX_dz  ]
    dd   = s[IDX_dd  ]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    ddd   = dddth1[0]
    ddth1 = dddth1[1]

    A = np.zeros((4,4), dtype=np.float64)
    A[0][0] = m2+m1+m0
    A[0][1] = -l1*m2*sin(th1)
    A[0][2] = 0
    A[0][3] = 0
    A[1][0] = m2+m1
    A[1][1] = -l1*m2*sin(th1)
    A[1][2] = -1
    A[1][3] = 0
    A[2][0] = -l1*m2*sin(th1)
    A[2][1] = (m2+m1+m0)*z0**2+((2*m2+2*m1+2*m0)*z+2*l1*m2*cos(th1)+(2*l0+2*d)*m2+(2*l0+2*d)*m1)*z0+(m2+m1+m0)*z**2+(2*l1*m2*cos(th1)+(2*l0+2*d)*m2+(2*l0+2*d)*m1)*z+(2*l0+2*d)*l1*m2*cos(th1)+(l1**2+l0**2+2*d*l0+d**2)*m2+(l0**2+2*d*l0+d**2)*m1
    A[2][2] = 0
    A[2][3] = 0
    A[3][0] = -l1*m2*sin(th1)
    A[3][1] = l1*m2*cos(th1)*z0+l1*m2*cos(th1)*z+(l0+d)*l1*m2*cos(th1)+l1**2*m2
    A[3][2] = 0
    A[3][3] = -1

    b = np.zeros((4,1), dtype=np.float64)
    b[0] = ((-dth0**2*m2)-dth0**2*m1-dth0**2*m0)*z0+((-dth0**2*m2)-dth0**2*m1-dth0**2*m0+k)*z+m2*(l1*((-ddth1*sin(th1))-dth1**2*cos(th1)-2*dth0*dth1*cos(th1)-dth0**2*cos(th1))+g*sin(th0)-dth0**2*l0-d*dth0**2+ddd)+m1*(g*sin(th0)-dth0**2*l0-d*dth0**2+ddd)+g*m0*sin(th0)
    b[1] = ((-dth0**2*m2)-dth0**2*m1)*z0+((-dth0**2*m2)-dth0**2*m1)*z+m2*(l1*((-ddth1*sin(th1))-dth1**2*cos(th1)-2*dth0*dth1*cos(th1)-dth0**2*cos(th1))+g*sin(th0)-dth0**2*l0-d*dth0**2+ddd)+m1*(g*sin(th0)-dth0**2*l0-d*dth0**2+ddd)
    b[2] = (m2*(l1*((-dth1**2*sin(th1))-2*dth0*dth1*sin(th1)+ddth1*cos(th1))+g*cos(th0)+dth0*(2*dz+2*dd))+m1*(g*cos(th0)+dth0*(2*dz+2*dd))+m0*(g*cos(th0)+2*dth0*dz))*z0+(m2*(l1*((-dth1**2*sin(th1))-2*dth0*dth1*sin(th1)+ddth1*cos(th1))+g*cos(th0)+dth0*(2*dz+2*dd))+m1*(g*cos(th0)+dth0*(2*dz+2*dd))+m0*(g*cos(th0)+2*dth0*dz))*z+m2*(l1*(g*cos(th1+th0)+l0*((-dth1**2*sin(th1))-2*dth0*dth1*sin(th1)+ddth1*cos(th1))-d*dth1**2*sin(th1)-2*d*dth0*dth1*sin(th1)-ddd*sin(th1)+dth0*(2*dz*cos(th1)+2*dd*cos(th1))+d*ddth1*cos(th1))+l0*(g*cos(th0)+dth0*(2*dz+2*dd))+d*g*cos(th0)+ddth1*l1**2+d*dth0*(2*dz+2*dd))+m1*(l0*(g*cos(th0)+dth0*(2*dz+2*dd))+d*g*cos(th0)+d*dth0*(2*dz+2*dd))
    b[3] = dth0**2*l1*m2*sin(th1)*z0+dth0**2*l1*m2*sin(th1)*z+m2*(l1*(g*cos(th1+th0)+dth0**2*l0*sin(th1)+d*dth0**2*sin(th1)-ddd*sin(th1)+dth0*(2*dz*cos(th1)+2*dd*cos(th1)))+ddth1*l1**2)

    #y : symbolilze([z'', th0'', fd, tau1])$
    y = np.linalg.solve(A, -b).reshape(4)
    return np.array([y[2], y[3]])

def inverse_model_d_th_air(s, dddth1):
    xr   = s[IDX_xr]
    yr   = s[IDX_yr]
    d    = s[IDX_d]
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    dd   = s[IDX_dd  ]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    ddd   = dddth1[0]
    ddth1 = dddth1[1]
    A = np.zeros((5,5), dtype=np.float64)
    A[0][0] = m2+m1+m0
    A[0][1] = 0
    A[0][2] = ((-m2)-m1-m0)*sin(th0)*z0-l1*m2*sin(th1+th0)+(((-l0)-d)*m2+((-l0)-d)*m1)*sin(th0)
    A[0][3] = 0
    A[0][4] = 0
    A[1][0] = 0
    A[1][1] = m2+m1+m0
    A[1][2] = (m2+m1+m0)*cos(th0)*z0+l1*m2*cos(th1+th0)+((l0+d)*m2+(l0+d)*m1)*cos(th0)
    A[1][3] = 0
    A[1][4] = 0
    A[2][0] = (m2+m1)*cos(th0)
    A[2][1] = (m2+m1)*sin(th0)
    A[2][2] = -l1*m2*sin(th1)
    A[2][3] = -1
    A[2][4] = 0
    A[3][0] = ((-m2)-m1-m0)*sin(th0)*z0-l1*m2*sin(th1+th0)+(((-l0)-d)*m2+((-l0)-d)*m1)*sin(th0)
    A[3][1] = (m2+m1+m0)*cos(th0)*z0+l1*m2*cos(th1+th0)+((l0+d)*m2+(l0+d)*m1)*cos(th0)
    A[3][2] = (m2+m1+m0)*z0**2+(2*l1*m2*cos(th1)+(2*l0+2*d)*m2+(2*l0+2*d)*m1)*z0+(2*l0+2*d)*l1*m2*cos(th1)+(l1**2+l0**2+2*d*l0+d**2)*m2+(l0**2+2*d*l0+d**2)*m1
    A[3][3] = 0
    A[3][4] = 0
    A[4][0] = -l1*m2*sin(th1+th0)
    A[4][1] = l1*m2*cos(th1+th0)
    A[4][2] = l1*m2*cos(th1)*z0+(l0+d)*l1*m2*cos(th1)+l1**2*m2
    A[4][3] = 0
    A[4][4] = -1

    b = np.zeros((5,1), dtype=np.float64)
    b[0] = ((-dth0**2*m2*cos(th0))-dth0**2*m1*cos(th0)-dth0**2*m0*cos(th0))*z0+m2*(l1*((-ddth1*sin(th1+th0))-dth1**2*cos(th1+th0)-2*dth0*dth1*cos(th1+th0)-dth0**2*cos(th1+th0))-2*dd*dth0*sin(th0)-dth0**2*l0*cos(th0)-d*dth0**2*cos(th0)+ddd*cos(th0))+m1*((-2*dd*dth0*sin(th0))-dth0**2*l0*cos(th0)-d*dth0**2*cos(th0)+ddd*cos(th0))
    b[1] = ((-dth0**2*m2*sin(th0))-dth0**2*m1*sin(th0)-dth0**2*m0*sin(th0))*z0+m2*(l1*((-dth1**2*sin(th1+th0))-2*dth0*dth1*sin(th1+th0)-dth0**2*sin(th1+th0)+ddth1*cos(th1+th0))-dth0**2*l0*sin(th0)-d*dth0**2*sin(th0)+ddd*sin(th0)+2*dd*dth0*cos(th0)+g)+m1*((-dth0**2*l0*sin(th0))-d*dth0**2*sin(th0)+ddd*sin(th0)+2*dd*dth0*cos(th0)+g)+g*m0
    b[2] = ((-dth0**2*m2)-dth0**2*m1)*z0+m2*(l1*((-ddth1*sin(th1))-dth1**2*cos(th1)-2*dth0*dth1*cos(th1)-dth0**2*cos(th1))+g*sin(th0)-dth0**2*l0-d*dth0**2+ddd)+m1*(g*sin(th0)-dth0**2*l0-d*dth0**2+ddd)
    b[3] = (m2*(l1*((-dth1**2*sin(th1))-2*dth0*dth1*sin(th1)+ddth1*cos(th1))+g*cos(th0)+2*dd*dth0)+m1*(g*cos(th0)+2*dd*dth0)+g*m0*cos(th0))*z0+m2*(l1*(g*cos(th1+th0)+l0*((-dth1**2*sin(th1))-2*dth0*dth1*sin(th1)+ddth1*cos(th1))-d*dth1**2*sin(th1)-2*d*dth0*dth1*sin(th1)-ddd*sin(th1)+2*dd*dth0*cos(th1)+d*ddth1*cos(th1))+l0*(g*cos(th0)+2*dd*dth0)+d*g*cos(th0)+ddth1*l1**2+2*d*dd*dth0)+m1*(l0*(g*cos(th0)+2*dd*dth0)+d*g*cos(th0)+2*d*dd*dth0)
    b[4] = dth0**2*l1*m2*sin(th1)*z0+m2*(l1*(g*cos(th1+th0)+dth0**2*l0*sin(th1)+d*dth0**2*sin(th1)-ddd*sin(th1)+2*dd*dth0*cos(th1))+ddth1*l1**2)

    #y : symbolilze([xr'', yr'', th0'', fd, tau1])$
    y = np.linalg.solve(A, -b).reshape(5)
    return np.array([y[3], y[4]])

# (d, th1) --PD--> u
def pdcontrol_d_th1(s, ref):
    dob  = np.array([s[IDX_dd], s[IDX_dth1]])
    ob = np.array([s[IDX_d], s[IDX_th1]])
    err = ref - ob
    dddth1 = err * Kp - Kd * dob

    if s[IDX_yr] == 0 and s[IDX_dx] == 0 and s[IDX_dy] == 0: # already on the ground
        assert 0 < s[IDX_th0] and s[IDX_th0] < np.pi, state_dict(s)
        u = inverse_model_d_th_ground(s, dddth1)
    elif s[IDX_yr] >= 0: # already in the air
        assert s[IDX_z] == 0 and s[IDX_dz] == 0, state_dict(s)
        u = inverse_model_d_th_air(s, dddth1)
    else:
        assert False, s
    return u

#def calc_r(s):
#    th1 = s[IDX_th1]
#    d = s[IDX_d]
#    return (l1*m2*cos(th1)+(l0+d)*m2+(l0+d)*m1)/(m2+m1+m0)
#
#def calc_dr(s):
#    th1 = s[IDX_th1]
#    dth1 = s[IDX_dth1]
#    dd = s[IDX_dd]
#    return -(dth1*l1*m2*sin(th1)-dd*m2-dd*m1)/(m2+m1+m0)
#
#def inverse_model_r_th_ground(s, ddrth0):
#    z    = s[IDX_z]
#    d    = s[IDX_d]
#    th0 = s[IDX_th0]
#    th1 = s[IDX_th1]
#    dz   = s[IDX_dz  ]
#    dd   = s[IDX_dd  ]
#    dth0 = s[IDX_dth0]
#    dth1 = s[IDX_dth1]
#    ddr   = ddrth0[0]
#    ddth0 = ddrth0[1]
#
#    A = np.zeros((5,5), dtype=np.float64)
#    A[0][0] = m2+m1+m0
#    A[0][1] = m2+m1
#    A[0][2] = -l1*m2*sin(th1)
#    A[0][3] = 0
#    A[0][4] = 0
#    A[1][0] = m2+m1
#    A[1][1] = m2+m1
#    A[1][2] = -l1*m2*sin(th1)
#    A[1][3] = -1
#    A[1][4] = 0
#    A[2][0] = -l1*m2*sin(th1)
#    A[2][1] = -l1*m2*sin(th1)
#    A[2][2] = l1*m2*cos(th1)*z0+l1*m2*cos(th1)*z+(l0+d)*l1*m2*cos(th1)+l1**2*m2
#    A[2][3] = 0
#    A[2][4] = 0
#    A[3][0] = -l1*m2*sin(th1)
#    A[3][1] = -l1*m2*sin(th1)
#    A[3][2] = l1**2*m2
#    A[3][3] = 0
#    A[3][4] = -1
#    A[4][0] = 0
#    A[4][1] = (m2+m1)/(m2+m1+m0)
#    A[4][2] = -(l1*m2*sin(th1))/(m2+m1+m0)
#    A[4][3] = 0
#    A[4][4] = 0
#
#    b = np.zeros((5,1), dtype=np.float64)
#    b[0] = ((-dth0**2*m2)-dth0**2*m1-dth0**2*m0)*z0+((-dth0**2*m2)-dth0**2*m1-dth0**2*m0+k)*z-ddth0*l1*m2*sin(th1)+((-dth1**2)-2*dth0*dth1-dth0**2)*l1*m2*cos(th1)+(g*m2+g*m1+g*m0)*sin(th0)+((-dth0**2*l0)-d*dth0**2)*m2+((-dth0**2*l0)-d*dth0**2)*m1
#    b[1] = ((-dth0**2*m2)-dth0**2*m1)*z0+((-dth0**2*m2)-dth0**2*m1)*z-ddth0*l1*m2*sin(th1)+((-dth1**2)-2*dth0*dth1-dth0**2)*l1*m2*cos(th1)+(g*m2+g*m1)*sin(th0)+((-dth0**2*l0)-d*dth0**2)*m2+((-dth0**2*l0)-d*dth0**2)*m1
#    b[2] = (ddth0*m2+ddth0*m1+ddth0*m0)*z0**2+((2*ddth0*m2+2*ddth0*m1+2*ddth0*m0)*z+((-dth1**2)-2*dth0*dth1)*l1*m2*sin(th1)+2*ddth0*l1*m2*cos(th1)+(g*m2+g*m1+g*m0)*cos(th0)+(2*ddth0*l0+2*dth0*dz+2*dd*dth0+2*d*ddth0)*m2+(2*ddth0*l0+2*dth0*dz+2*dd*dth0+2*d*ddth0)*m1+2*dth0*dz*m0)*z0+(ddth0*m2+ddth0*m1+ddth0*m0)*z**2+(((-dth1**2)-2*dth0*dth1)*l1*m2*sin(th1)+2*ddth0*l1*m2*cos(th1)+(g*m2+g*m1+g*m0)*cos(th0)+(2*ddth0*l0+2*dth0*dz+2*dd*dth0+2*d*ddth0)*m2+(2*ddth0*l0+2*dth0*dz+2*dd*dth0+2*d*ddth0)*m1+2*dth0*dz*m0)*z+g*l1*m2*cos(th1+th0)+(((-dth1**2)-2*dth0*dth1)*l0-d*dth1**2-2*d*dth0*dth1)*l1*m2*sin(th1)+(2*ddth0*l0+2*dth0*dz+2*dd*dth0+2*d*ddth0)*l1*m2*cos(th1)+((g*l0+d*g)*m2+(g*l0+d*g)*m1)*cos(th0)+(ddth0*l1**2+ddth0*l0**2+(2*dth0*dz+2*dd*dth0+2*d*ddth0)*l0+2*d*dth0*dz+2*d*dd*dth0+d**2*ddth0)*m2+(ddth0*l0**2+(2*dth0*dz+2*dd*dth0+2*d*ddth0)*l0+2*d*dth0*dz+2*d*dd*dth0+d**2*ddth0)*m1
#    b[3] = (dth0**2*l1*m2*sin(th1)+ddth0*l1*m2*cos(th1))*z0+(dth0**2*l1*m2*sin(th1)+ddth0*l1*m2*cos(th1))*z+g*l1*m2*cos(th1+th0)+(dth0**2*l0+d*dth0**2)*l1*m2*sin(th1)+(ddth0*l0+2*dth0*dz+2*dd*dth0+d*ddth0)*l1*m2*cos(th1)+ddth0*l1**2*m2
#    b[4] = -(dth1**2*l1*m2*cos(th1)+ddr*m2+ddr*m1+ddr*m0)/(m2+m1+m0)
#
#    #y : symbolilze([z'', d'', th1'', fd, tau1])$
#    y = np.linalg.solve(A, -b).reshape(5)
#    return np.array([y[3], y[4]])
#
#
#def inverse_model_r_th_air(s, ddrth0):
#    xr   = s[IDX_xr]
#    yr   = s[IDX_yr]
#    d    = s[IDX_d]
#    th0 = s[IDX_th0]
#    th1 = s[IDX_th1]
#    dd   = s[IDX_dd  ]
#    dth0 = s[IDX_dth0]
#    dth1 = s[IDX_dth1]
#    ddr   = ddrth0[0]
#    ddth0 = ddrth0[1]
#
#    A = np.zeros((6,6), dtype=np.float64)
#    A[0][0] = m2+m1+m0
#    A[0][1] = 0
#    A[0][2] = (m2+m1)*cos(th0)
#    A[0][3] = -l1*m2*sin(th1+th0)
#    A[0][4] = 0
#    A[0][5] = 0
#    A[1][0] = 0
#    A[1][1] = m2+m1+m0
#    A[1][2] = (m2+m1)*sin(th0)
#    A[1][3] = l1*m2*cos(th1+th0)
#    A[1][4] = 0
#    A[1][5] = 0
#    A[2][0] = (m2+m1)*cos(th0)
#    A[2][1] = (m2+m1)*sin(th0)
#    A[2][2] = m2+m1
#    A[2][3] = -l1*m2*sin(th1)
#    A[2][4] = -1
#    A[2][5] = 0
#    A[3][0] = ((-m2)-m1-m0)*sin(th0)*z0-l1*m2*sin(th1+th0)+(((-l0)-d)*m2+((-l0)-d)*m1)*sin(th0)
#    A[3][1] = (m2+m1+m0)*cos(th0)*z0+l1*m2*cos(th1+th0)+((l0+d)*m2+(l0+d)*m1)*cos(th0)
#    A[3][2] = -l1*m2*sin(th1)
#    A[3][3] = l1*m2*cos(th1)*z0+(l0+d)*l1*m2*cos(th1)+l1**2*m2
#    A[3][4] = 0
#    A[3][5] = 0
#    A[4][0] = -l1*m2*sin(th1+th0)
#    A[4][1] = l1*m2*cos(th1+th0)
#    A[4][2] = -l1*m2*sin(th1)
#    A[4][3] = l1**2*m2
#    A[4][4] = 0
#    A[4][5] = -1
#    A[5][0] = 0
#    A[5][1] = 0
#    A[5][2] = (m2+m1)/(m2+m1+m0)
#    A[5][3] = -(l1*m2*sin(th1))/(m2+m1+m0)
#    A[5][4] = 0
#    A[5][5] = 0
#
#    b = np.zeros((6,1), dtype=np.float64)
#    b[0] = (((-ddth0*m2)-ddth0*m1-ddth0*m0)*sin(th0)+((-dth0**2*m2)-dth0**2*m1-dth0**2*m0)*cos(th0))*z0-ddth0*l1*m2*sin(th1+th0)+((-dth1**2)-2*dth0*dth1-dth0**2)*l1*m2*cos(th1+th0)+(((-ddth0*l0)-2*dd*dth0-d*ddth0)*m2+((-ddth0*l0)-2*dd*dth0-d*ddth0)*m1)*sin(th0)+(((-dth0**2*l0)-d*dth0**2)*m2+((-dth0**2*l0)-d*dth0**2)*m1)*cos(th0)
#    b[1] = (((-dth0**2*m2)-dth0**2*m1-dth0**2*m0)*sin(th0)+(ddth0*m2+ddth0*m1+ddth0*m0)*cos(th0))*z0+((-dth1**2)-2*dth0*dth1-dth0**2)*l1*m2*sin(th1+th0)+ddth0*l1*m2*cos(th1+th0)+(((-dth0**2*l0)-d*dth0**2)*m2+((-dth0**2*l0)-d*dth0**2)*m1)*sin(th0)+((ddth0*l0+2*dd*dth0+d*ddth0)*m2+(ddth0*l0+2*dd*dth0+d*ddth0)*m1)*cos(th0)+g*m2+g*m1+g*m0
#    b[2] = ((-dth0**2*m2)-dth0**2*m1)*z0-ddth0*l1*m2*sin(th1)+((-dth1**2)-2*dth0*dth1-dth0**2)*l1*m2*cos(th1)+(g*m2+g*m1)*sin(th0)+((-dth0**2*l0)-d*dth0**2)*m2+((-dth0**2*l0)-d*dth0**2)*m1
#    b[3] = (ddth0*m2+ddth0*m1+ddth0*m0)*z0**2+(((-dth1**2)-2*dth0*dth1)*l1*m2*sin(th1)+2*ddth0*l1*m2*cos(th1)+(g*m2+g*m1+g*m0)*cos(th0)+(2*ddth0*l0+2*dd*dth0+2*d*ddth0)*m2+(2*ddth0*l0+2*dd*dth0+2*d*ddth0)*m1)*z0+g*l1*m2*cos(th1+th0)+(((-dth1**2)-2*dth0*dth1)*l0-d*dth1**2-2*d*dth0*dth1)*l1*m2*sin(th1)+(2*ddth0*l0+2*dd*dth0+2*d*ddth0)*l1*m2*cos(th1)+((g*l0+d*g)*m2+(g*l0+d*g)*m1)*cos(th0)+(ddth0*l1**2+ddth0*l0**2+(2*dd*dth0+2*d*ddth0)*l0+2*d*dd*dth0+d**2*ddth0)*m2+(ddth0*l0**2+(2*dd*dth0+2*d*ddth0)*l0+2*d*dd*dth0+d**2*ddth0)*m1
#    b[4] = (dth0**2*l1*m2*sin(th1)+ddth0*l1*m2*cos(th1))*z0+g*l1*m2*cos(th1+th0)+(dth0**2*l0+d*dth0**2)*l1*m2*sin(th1)+(ddth0*l0+2*dd*dth0+d*ddth0)*l1*m2*cos(th1)+ddth0*l1**2*m2
#    b[5] = -(dth1**2*l1*m2*cos(th1)+ddr*m2+ddr*m1+ddr*m0)/(m2+m1+m0)
#
#    #y : symbolilze([xr'', yr'', d'', th1'', fd, tau1])$
#    y = np.linalg.solve(A, -b).reshape(6)
#    return np.array([y[4], y[5]])
#
#
## (r, th0) --PD--> (r'', th0'') --inverse-model--> u
#def pdcontrol_r_th(s, ref):
#    dr  = np.array([calc_dr(s), s[IDX_dth0]])
#    ob = np.array([calc_r(s), s[IDX_th0]])
#    err = ref - ob
#    ddrth0 = err * Kp - Kd * dr
#
#    print(f"d_r: {err[0]} r': {dr[0]} r'': {ddrth0[0]}")
#    print(f"d_th0: {err[1]} th0': {dr[1]} th0'': {ddrth0[1]}")
#
#    if s[IDX_yr] == 0 and s[IDX_dx] == 0 and s[IDX_dy] == 0: # already on the ground
#        assert 0 < s[IDX_th0] and s[IDX_th0] < np.pi, state_dict(s)
#        u = inverse_model_r_th_ground(s, ddrth0)
#    elif s[IDX_yr] >= 0: # already in the air
#        assert s[IDX_z] == 0 and s[IDX_dz] == 0, state_dict(s)
#        u = inverse_model_r_th_air(s, ddrth0)
#    else:
#        assert False, s
#    return u
