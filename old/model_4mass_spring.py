import numpy as np
from numpy import sin, cos
import sys


l0 =  0.4
l1 =  0.4
l2 =  0.4
max_z = 0.1

m0 = 1
m1 = 10
m2 = 10
m3 = 10
g  = 9.8
#g  = 0
k  = 200000

#-----------------
# State
#-----------------

IDX_x0   = 0
IDX_y0   = 1
IDX_z    = 2
IDX_th0  = 3
IDX_th1  = 4
IDX_th2  = 5
IDX_dx   = 6
IDX_dy   = 7
IDX_dz   = 8
IDX_dth0 = 9
IDX_dth1 = 10
IDX_dth2 = 11
IDX_MAX = 12

def reset_state(np_random=None):
    s = np.zeros(IDX_MAX, dtype=np.float64)
    s[IDX_y0]  = 1
    s[IDX_th0]  = np.pi/2
    s[IDX_z]  = 0.1
    #s[IDX_th0]  = np.pi/4
    #s[IDX_th1]  = np.pi*3/4 - np.pi/4
    #s[IDX_th2]  = np.pi*5/12 - np.pi*3/4
    return s

def print_state(s):
    print("")
    print(f"x0:  {s[IDX_x0 ]:.2f} ")
    print(f"y0:  {s[IDX_y0 ]:.2f} ")
    print(f"z :  {s[IDX_z  ]:.2f} ")
    print(f"th0: {s[IDX_th0]:.2f} ")
    print(f"th1: {s[IDX_th1]:.2f} ")
    print(f"th2: {s[IDX_th2]:.2f} ")
    print("")
    print(f"dx:  {s[IDX_dx  ]:.2f}")
    print(f"dy:  {s[IDX_dy  ]:.2f}")
    print(f"dz:  {s[IDX_dz  ]:.2f}")
    print(f"dth0:{s[IDX_dth0]:.2f}")
    print(f"dth1:{s[IDX_dth1]:.2f}")
    print(f"dth2:{s[IDX_dth2]:.2f}")

#-----------------
# Kinematics
#-----------------

def clip_z(s, check=False):
    new_s = s.copy()
    new_s[IDX_z] = np.clip(s[IDX_z], -max_z, max_z)
    if check:
        assert new_s[IDX_z] == s[IDX_z], (new_s[IDX_z], s[IDX_z])
    return new_s

def postprocess(s):
    new_s = s.copy()
    # normalize_angle
    new_s[IDX_th0] = ((s[IDX_th0] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th1] = ((s[IDX_th1] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th2] = ((s[IDX_th2] + np.pi)% (2*np.pi)) - np.pi
    # max_z_abs
    return clip_z(new_s)


def node_angle(s):
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    th2 = s[IDX_th2]
    return th0, th1, th2

def node_omega(s):
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dth2 = s[IDX_dth2]
    return dth0, dth1, dth2

def node_pos(s):
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    th2 = s[IDX_th2]
    z   = s[IDX_z]

    p0 = s[IDX_x0:IDX_y0+1].copy()
    p1 = p0 + (l0 + z) * np.array([np.cos(th0), np.sin(th0)])
    p2 = p1 + l1 * np.array([np.cos(th0+th1), np.sin(th0+th1)])
    p3 = p2 + l2 * np.array([np.cos(th0+th1+th2), np.sin(th0+th1+th2)])

    return p0, p1, p2, p3

def node_vel(s):
    z   = s[IDX_z]
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    th2 = s[IDX_th2]
    dz   = s[IDX_dz]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dth2 = s[IDX_dth2]
    v0 = s[IDX_dx:IDX_dy+1]
    v1 = v0 + np.array([cos(th0), sin(th0)]) * dz + np.array([-sin(th0), cos(th0)]) * (z+l0) * dth0
    v2 = v1 + np.array([-sin(th0+th1), cos(th0+th1)]) * l1 * (dth0+dth1)
    v3 = v2 + np.array([-sin(th0+th1+th2), cos(th0+th1+th2)]) * l2 * (dth0+dth1+dth2)

    return v0, v1, v2, v3

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

def torq_limit(s, u):
    return u

def _calcAb44(s, u):
    x0   = s[IDX_x0]
    y0   = s[IDX_y0]
    z    = s[IDX_z]
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    th2  = s[IDX_th2]
    dx   = s[IDX_dx  ]
    dy   = s[IDX_dy  ]
    dz   = s[IDX_dz  ]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dth2 = s[IDX_dth2]
    fx0  = 0
    fy0  = 0
    fz0  = 0
    tau0 = 0
    tau1 = u[0]
    tau2 = u[1]
    extf = np.array([fz0, tau0, tau1, tau2]).reshape(4,1)

    A = np.zeros((4,4), dtype=np.float64)
    A[0][0] = m3+m2+m1
    A[0][1] = ((-l1*m3)-l1*m2)*sin(th1)-l2*m3*sin(th2+th1)
    A[0][2] = ((-l1*m3)-l1*m2)*sin(th1)-l2*m3*sin(th2+th1)
    A[0][3] = -l2*m3*sin(th2+th1)
    A[1][0] = ((-l1*m3)-l1*m2)*sin(th1)-l2*m3*sin(th2+th1)
    A[1][1] = (m3+m2+m1)*z**2+(2*l2*m3*cos(th2+th1)+(2*l1*m3+2*l1*m2)*cos(th1)+2*l0*m3+2*l0*m2+2*l0*m1)*z+2*l0*l2*m3*cos(th2+th1)+2*l1*l2*m3*cos(th2)+(2*l0*l1*m3+2*l0*l1*m2)*cos(th1)+(l2**2+l1**2+l0**2)*m3+(l1**2+l0**2)*m2+l0**2*m1
    A[1][2] = (l2*m3*cos(th2+th1)+(l1*m3+l1*m2)*cos(th1))*z+l0*l2*m3*cos(th2+th1)+2*l1*l2*m3*cos(th2)+(l0*l1*m3+l0*l1*m2)*cos(th1)+(l2**2+l1**2)*m3+l1**2*m2
    A[1][3] = l2*m3*cos(th2+th1)*z+l0*l2*m3*cos(th2+th1)+l1*l2*m3*cos(th2)+l2**2*m3
    A[2][0] = ((-l1*m3)-l1*m2)*sin(th1)-l2*m3*sin(th2+th1)
    A[2][1] = (l2*m3*cos(th2+th1)+(l1*m3+l1*m2)*cos(th1))*z+l0*l2*m3*cos(th2+th1)+2*l1*l2*m3*cos(th2)+(l0*l1*m3+l0*l1*m2)*cos(th1)+(l2**2+l1**2)*m3+l1**2*m2
    A[2][2] = 2*l1*l2*m3*cos(th2)+(l2**2+l1**2)*m3+l1**2*m2
    A[2][3] = l1*l2*m3*cos(th2)+l2**2*m3
    A[3][0] = -l2*m3*sin(th2+th1)
    A[3][1] = l2*m3*cos(th2+th1)*z+l0*l2*m3*cos(th2+th1)+l1*l2*m3*cos(th2)+l2**2*m3
    A[3][2] = l1*l2*m3*cos(th2)+l2**2*m3
    A[3][3] = l2**2*m3

    b = np.zeros((4,1), dtype=np.float64)
    b[0] = (-dth0**2*m3*sin(th0)**2*z)-dth0**2*m2*sin(th0)**2*z-dth0**2*m1*sin(th0)**2*z-dth0**2*m3*cos(th0)**2*z-dth0**2*m2*cos(th0)**2*z-dth0**2*m1*cos(th0)**2*z+k*z-dth2**2*l2*m3*sin(th0)*sin(th2+th1+th0)-2*dth1*dth2*l2*m3*sin(th0)*sin(th2+th1+th0)-2*dth0*dth2*l2*m3*sin(th0)*sin(th2+th1+th0)-dth1**2*l2*m3*sin(th0)*sin(th2+th1+th0)-2*dth0*dth1*l2*m3*sin(th0)*sin(th2+th1+th0)-dth0**2*l2*m3*sin(th0)*sin(th2+th1+th0)-dth2**2*l2*m3*cos(th0)*cos(th2+th1+th0)-2*dth1*dth2*l2*m3*cos(th0)*cos(th2+th1+th0)-2*dth0*dth2*l2*m3*cos(th0)*cos(th2+th1+th0)-dth1**2*l2*m3*cos(th0)*cos(th2+th1+th0)-2*dth0*dth1*l2*m3*cos(th0)*cos(th2+th1+th0)-dth0**2*l2*m3*cos(th0)*cos(th2+th1+th0)-dth1**2*l1*m3*sin(th0)*sin(th1+th0)-2*dth0*dth1*l1*m3*sin(th0)*sin(th1+th0)-dth0**2*l1*m3*sin(th0)*sin(th1+th0)-dth1**2*l1*m2*sin(th0)*sin(th1+th0)-2*dth0*dth1*l1*m2*sin(th0)*sin(th1+th0)-dth0**2*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l1*m3*cos(th0)*cos(th1+th0)-2*dth0*dth1*l1*m3*cos(th0)*cos(th1+th0)-dth0**2*l1*m3*cos(th0)*cos(th1+th0)-dth1**2*l1*m2*cos(th0)*cos(th1+th0)-2*dth0*dth1*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l0*m3*sin(th0)**2-dth0**2*l0*m2*sin(th0)**2-dth0**2*l0*m1*sin(th0)**2+g*m3*sin(th0)+g*m2*sin(th0)+g*m1*sin(th0)-dth0**2*l0*m3*cos(th0)**2-dth0**2*l0*m2*cos(th0)**2-dth0**2*l0*m1*cos(th0)**2
    b[1] = (-dth2**2*l2*m3*cos(th0)*sin(th2+th1+th0)*z)-2*dth1*dth2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-2*dth0*dth2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-dth1**2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-2*dth0*dth1*l2*m3*cos(th0)*sin(th2+th1+th0)*z+dth2**2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+2*dth1*dth2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+2*dth0*dth2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+dth1**2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+2*dth0*dth1*l2*m3*sin(th0)*cos(th2+th1+th0)*z-dth1**2*l1*m3*cos(th0)*sin(th1+th0)*z-2*dth0*dth1*l1*m3*cos(th0)*sin(th1+th0)*z-dth1**2*l1*m2*cos(th0)*sin(th1+th0)*z-2*dth0*dth1*l1*m2*cos(th0)*sin(th1+th0)*z+dth1**2*l1*m3*sin(th0)*cos(th1+th0)*z+2*dth0*dth1*l1*m3*sin(th0)*cos(th1+th0)*z+dth1**2*l1*m2*sin(th0)*cos(th1+th0)*z+2*dth0*dth1*l1*m2*sin(th0)*cos(th1+th0)*z+2*dth0*dz*m3*sin(th0)**2*z+2*dth0*dz*m2*sin(th0)**2*z+2*dth0*dz*m1*sin(th0)**2*z+2*dth0*dz*m3*cos(th0)**2*z+2*dth0*dz*m2*cos(th0)**2*z+2*dth0*dz*m1*cos(th0)**2*z+g*m3*cos(th0)*z+g*m2*cos(th0)*z+g*m1*cos(th0)*z-dth2**2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)-2*dth1*dth2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)-2*dth0*dth2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+2*dth0*dz*l2*m3*sin(th0)*sin(th2+th1+th0)-dth2**2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-2*dth1*dth2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-2*dth0*dth2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-dth1**2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-2*dth0*dth1*l0*l2*m3*cos(th0)*sin(th2+th1+th0)+dth2**2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+2*dth1*dth2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+2*dth0*dth2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+dth2**2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth1*dth2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dth2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+dth1**2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dth1*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dz*l2*m3*cos(th0)*cos(th2+th1+th0)+g*l2*m3*cos(th2+th1+th0)+2*dth0*dz*l1*m3*sin(th0)*sin(th1+th0)+2*dth0*dz*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l0*l1*m3*cos(th0)*sin(th1+th0)-2*dth0*dth1*l0*l1*m3*cos(th0)*sin(th1+th0)-dth1**2*l0*l1*m2*cos(th0)*sin(th1+th0)-2*dth0*dth1*l0*l1*m2*cos(th0)*sin(th1+th0)+dth1**2*l0*l1*m3*sin(th0)*cos(th1+th0)+2*dth0*dth1*l0*l1*m3*sin(th0)*cos(th1+th0)+dth1**2*l0*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dth1*l0*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dz*l1*m3*cos(th0)*cos(th1+th0)+2*dth0*dz*l1*m2*cos(th0)*cos(th1+th0)+g*l1*m3*cos(th1+th0)+g*l1*m2*cos(th1+th0)+2*dth0*dz*l0*m3*sin(th0)**2+2*dth0*dz*l0*m2*sin(th0)**2+2*dth0*dz*l0*m1*sin(th0)**2+2*dth0*dz*l0*m3*cos(th0)**2+2*dth0*dz*l0*m2*cos(th0)**2+2*dth0*dz*l0*m1*cos(th0)**2+g*l0*m3*cos(th0)+g*l0*m2*cos(th0)+g*l0*m1*cos(th0)
    b[2] = dth0**2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-dth0**2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+dth0**2*l1*m3*cos(th0)*sin(th1+th0)*z+dth0**2*l1*m2*cos(th0)*sin(th1+th0)*z-dth0**2*l1*m3*sin(th0)*cos(th1+th0)*z-dth0**2*l1*m2*sin(th0)*cos(th1+th0)*z-dth2**2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)-2*dth1*dth2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)-2*dth0*dth2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+2*dth0*dz*l2*m3*sin(th0)*sin(th2+th1+th0)+dth0**2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)+dth2**2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+2*dth1*dth2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+2*dth0*dth2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)-dth0**2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dz*l2*m3*cos(th0)*cos(th2+th1+th0)+g*l2*m3*cos(th2+th1+th0)+2*dth0*dz*l1*m3*sin(th0)*sin(th1+th0)+2*dth0*dz*l1*m2*sin(th0)*sin(th1+th0)+dth0**2*l0*l1*m3*cos(th0)*sin(th1+th0)+dth0**2*l0*l1*m2*cos(th0)*sin(th1+th0)-dth0**2*l0*l1*m3*sin(th0)*cos(th1+th0)-dth0**2*l0*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dz*l1*m3*cos(th0)*cos(th1+th0)+2*dth0*dz*l1*m2*cos(th0)*cos(th1+th0)+g*l1*m3*cos(th1+th0)+g*l1*m2*cos(th1+th0)
    b[3] = dth0**2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-dth0**2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+dth1**2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+2*dth0*dth1*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+dth0**2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+2*dth0*dz*l2*m3*sin(th0)*sin(th2+th1+th0)+dth0**2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-dth1**2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)-2*dth0*dth1*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)-dth0**2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)-dth0**2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dz*l2*m3*cos(th0)*cos(th2+th1+th0)+g*l2*m3*cos(th2+th1+th0)
    return A, b, extf

def _calcAb66(s, u):

    x0   = s[IDX_x0]
    y0   = s[IDX_y0]
    z    = s[IDX_z]
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    th2  = s[IDX_th2]
    dx   = s[IDX_dx  ]
    dy   = s[IDX_dy  ]
    dz   = s[IDX_dz  ]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dth2 = s[IDX_dth2]
    fx0  = 0
    fy0  = 0
    fz0  = 0
    tau0 = 0
    tau1 = u[0]
    tau2 = u[1]
    extf = np.array([fx0, fy0, fz0, tau0, tau1, tau2]).reshape(6,1)

    A = np.zeros((6,6), dtype=np.float64)
    A[0][0] = m3+m2+m1+m0
    A[0][1] = 0
    A[0][2] = (m3+m2+m1)*cos(th0)
    A[0][3] = ((-m3)-m2-m1)*sin(th0)*z-l2*m3*sin(th2+th1+th0)+((-l1*m3)-l1*m2)*sin(th1+th0)+((-l0*m3)-l0*m2-l0*m1)*sin(th0)
    A[0][4] = ((-l1*m3)-l1*m2)*sin(th1+th0)-l2*m3*sin(th2+th1+th0)
    A[0][5] = -l2*m3*sin(th2+th1+th0)
    A[1][0] = 0
    A[1][1] = m3+m2+m1+m0
    A[1][2] = (m3+m2+m1)*sin(th0)
    A[1][3] = (m3+m2+m1)*cos(th0)*z+l2*m3*cos(th2+th1+th0)+(l1*m3+l1*m2)*cos(th1+th0)+(l0*m3+l0*m2+l0*m1)*cos(th0)
    A[1][4] = l2*m3*cos(th2+th1+th0)+(l1*m3+l1*m2)*cos(th1+th0)
    A[1][5] = l2*m3*cos(th2+th1+th0)
    A[2][0] = (m3+m2+m1)*cos(th0)
    A[2][1] = (m3+m2+m1)*sin(th0)
    A[2][2] = m3+m2+m1
    A[2][3] = ((-l1*m3)-l1*m2)*sin(th1)-l2*m3*sin(th2+th1)
    A[2][4] = ((-l1*m3)-l1*m2)*sin(th1)-l2*m3*sin(th2+th1)
    A[2][5] = -l2*m3*sin(th2+th1)
    A[3][0] = ((-m3)-m2-m1)*sin(th0)*z-l2*m3*sin(th2+th1+th0)+((-l1*m3)-l1*m2)*sin(th1+th0)+((-l0*m3)-l0*m2-l0*m1)*sin(th0)
    A[3][1] = (m3+m2+m1)*cos(th0)*z+l2*m3*cos(th2+th1+th0)+(l1*m3+l1*m2)*cos(th1+th0)+(l0*m3+l0*m2+l0*m1)*cos(th0)
    A[3][2] = ((-l1*m3)-l1*m2)*sin(th1)-l2*m3*sin(th2+th1)
    A[3][3] = (m3+m2+m1)*z**2+(2*l2*m3*cos(th2+th1)+(2*l1*m3+2*l1*m2)*cos(th1)+2*l0*m3+2*l0*m2+2*l0*m1)*z+2*l0*l2*m3*cos(th2+th1)+2*l1*l2*m3*cos(th2)+(2*l0*l1*m3+2*l0*l1*m2)*cos(th1)+(l2**2+l1**2+l0**2)*m3+(l1**2+l0**2)*m2+l0**2*m1
    A[3][4] = (l2*m3*cos(th2+th1)+(l1*m3+l1*m2)*cos(th1))*z+l0*l2*m3*cos(th2+th1)+2*l1*l2*m3*cos(th2)+(l0*l1*m3+l0*l1*m2)*cos(th1)+(l2**2+l1**2)*m3+l1**2*m2
    A[3][5] = l2*m3*cos(th2+th1)*z+l0*l2*m3*cos(th2+th1)+l1*l2*m3*cos(th2)+l2**2*m3
    A[4][0] = ((-l1*m3)-l1*m2)*sin(th1+th0)-l2*m3*sin(th2+th1+th0)
    A[4][1] = l2*m3*cos(th2+th1+th0)+(l1*m3+l1*m2)*cos(th1+th0)
    A[4][2] = ((-l1*m3)-l1*m2)*sin(th1)-l2*m3*sin(th2+th1)
    A[4][3] = (l2*m3*cos(th2+th1)+(l1*m3+l1*m2)*cos(th1))*z+l0*l2*m3*cos(th2+th1)+2*l1*l2*m3*cos(th2)+(l0*l1*m3+l0*l1*m2)*cos(th1)+(l2**2+l1**2)*m3+l1**2*m2
    A[4][4] = 2*l1*l2*m3*cos(th2)+(l2**2+l1**2)*m3+l1**2*m2
    A[4][5] = l1*l2*m3*cos(th2)+l2**2*m3
    A[5][0] = -l2*m3*sin(th2+th1+th0)
    A[5][1] = l2*m3*cos(th2+th1+th0)
    A[5][2] = -l2*m3*sin(th2+th1)
    A[5][3] = l2*m3*cos(th2+th1)*z+l0*l2*m3*cos(th2+th1)+l1*l2*m3*cos(th2)+l2**2*m3
    A[5][4] = l1*l2*m3*cos(th2)+l2**2*m3
    A[5][5] = l2**2*m3

    b = np.zeros((6,1), dtype=np.float64)
    b[0] = (-dth0**2*m3*cos(th0)*z)-dth0**2*m2*cos(th0)*z-dth0**2*m1*cos(th0)*z-dth2**2*l2*m3*cos(th2+th1+th0)-2*dth1*dth2*l2*m3*cos(th2+th1+th0)-2*dth0*dth2*l2*m3*cos(th2+th1+th0)-dth1**2*l2*m3*cos(th2+th1+th0)-2*dth0*dth1*l2*m3*cos(th2+th1+th0)-dth0**2*l2*m3*cos(th2+th1+th0)-dth1**2*l1*m3*cos(th1+th0)-2*dth0*dth1*l1*m3*cos(th1+th0)-dth0**2*l1*m3*cos(th1+th0)-dth1**2*l1*m2*cos(th1+th0)-2*dth0*dth1*l1*m2*cos(th1+th0)-dth0**2*l1*m2*cos(th1+th0)-2*dth0*dz*m3*sin(th0)-2*dth0*dz*m2*sin(th0)-2*dth0*dz*m1*sin(th0)-dth0**2*l0*m3*cos(th0)-dth0**2*l0*m2*cos(th0)-dth0**2*l0*m1*cos(th0)
    b[1] = (-dth0**2*m3*sin(th0)*z)-dth0**2*m2*sin(th0)*z-dth0**2*m1*sin(th0)*z-dth2**2*l2*m3*sin(th2+th1+th0)-2*dth1*dth2*l2*m3*sin(th2+th1+th0)-2*dth0*dth2*l2*m3*sin(th2+th1+th0)-dth1**2*l2*m3*sin(th2+th1+th0)-2*dth0*dth1*l2*m3*sin(th2+th1+th0)-dth0**2*l2*m3*sin(th2+th1+th0)-dth1**2*l1*m3*sin(th1+th0)-2*dth0*dth1*l1*m3*sin(th1+th0)-dth0**2*l1*m3*sin(th1+th0)-dth1**2*l1*m2*sin(th1+th0)-2*dth0*dth1*l1*m2*sin(th1+th0)-dth0**2*l1*m2*sin(th1+th0)-dth0**2*l0*m3*sin(th0)-dth0**2*l0*m2*sin(th0)-dth0**2*l0*m1*sin(th0)+2*dth0*dz*m3*cos(th0)+2*dth0*dz*m2*cos(th0)+2*dth0*dz*m1*cos(th0)+g*m3+g*m2+g*m1+g*m0
    b[2] = (-dth0**2*m3*sin(th0)**2*z)-dth0**2*m2*sin(th0)**2*z-dth0**2*m1*sin(th0)**2*z-dth0**2*m3*cos(th0)**2*z-dth0**2*m2*cos(th0)**2*z-dth0**2*m1*cos(th0)**2*z+k*z-dth2**2*l2*m3*sin(th0)*sin(th2+th1+th0)-2*dth1*dth2*l2*m3*sin(th0)*sin(th2+th1+th0)-2*dth0*dth2*l2*m3*sin(th0)*sin(th2+th1+th0)-dth1**2*l2*m3*sin(th0)*sin(th2+th1+th0)-2*dth0*dth1*l2*m3*sin(th0)*sin(th2+th1+th0)-dth0**2*l2*m3*sin(th0)*sin(th2+th1+th0)-dth2**2*l2*m3*cos(th0)*cos(th2+th1+th0)-2*dth1*dth2*l2*m3*cos(th0)*cos(th2+th1+th0)-2*dth0*dth2*l2*m3*cos(th0)*cos(th2+th1+th0)-dth1**2*l2*m3*cos(th0)*cos(th2+th1+th0)-2*dth0*dth1*l2*m3*cos(th0)*cos(th2+th1+th0)-dth0**2*l2*m3*cos(th0)*cos(th2+th1+th0)-dth1**2*l1*m3*sin(th0)*sin(th1+th0)-2*dth0*dth1*l1*m3*sin(th0)*sin(th1+th0)-dth0**2*l1*m3*sin(th0)*sin(th1+th0)-dth1**2*l1*m2*sin(th0)*sin(th1+th0)-2*dth0*dth1*l1*m2*sin(th0)*sin(th1+th0)-dth0**2*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l1*m3*cos(th0)*cos(th1+th0)-2*dth0*dth1*l1*m3*cos(th0)*cos(th1+th0)-dth0**2*l1*m3*cos(th0)*cos(th1+th0)-dth1**2*l1*m2*cos(th0)*cos(th1+th0)-2*dth0*dth1*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l1*m2*cos(th0)*cos(th1+th0)-dth0**2*l0*m3*sin(th0)**2-dth0**2*l0*m2*sin(th0)**2-dth0**2*l0*m1*sin(th0)**2+g*m3*sin(th0)+g*m2*sin(th0)+g*m1*sin(th0)-dth0**2*l0*m3*cos(th0)**2-dth0**2*l0*m2*cos(th0)**2-dth0**2*l0*m1*cos(th0)**2
    b[3] = (-dth2**2*l2*m3*cos(th0)*sin(th2+th1+th0)*z)-2*dth1*dth2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-2*dth0*dth2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-dth1**2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-2*dth0*dth1*l2*m3*cos(th0)*sin(th2+th1+th0)*z+dth2**2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+2*dth1*dth2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+2*dth0*dth2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+dth1**2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+2*dth0*dth1*l2*m3*sin(th0)*cos(th2+th1+th0)*z-dth1**2*l1*m3*cos(th0)*sin(th1+th0)*z-2*dth0*dth1*l1*m3*cos(th0)*sin(th1+th0)*z-dth1**2*l1*m2*cos(th0)*sin(th1+th0)*z-2*dth0*dth1*l1*m2*cos(th0)*sin(th1+th0)*z+dth1**2*l1*m3*sin(th0)*cos(th1+th0)*z+2*dth0*dth1*l1*m3*sin(th0)*cos(th1+th0)*z+dth1**2*l1*m2*sin(th0)*cos(th1+th0)*z+2*dth0*dth1*l1*m2*sin(th0)*cos(th1+th0)*z+2*dth0*dz*m3*sin(th0)**2*z+2*dth0*dz*m2*sin(th0)**2*z+2*dth0*dz*m1*sin(th0)**2*z+2*dth0*dz*m3*cos(th0)**2*z+2*dth0*dz*m2*cos(th0)**2*z+2*dth0*dz*m1*cos(th0)**2*z+g*m3*cos(th0)*z+g*m2*cos(th0)*z+g*m1*cos(th0)*z-dth2**2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)-2*dth1*dth2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)-2*dth0*dth2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+2*dth0*dz*l2*m3*sin(th0)*sin(th2+th1+th0)-dth2**2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-2*dth1*dth2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-2*dth0*dth2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-dth1**2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-2*dth0*dth1*l0*l2*m3*cos(th0)*sin(th2+th1+th0)+dth2**2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+2*dth1*dth2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+2*dth0*dth2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+dth2**2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth1*dth2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dth2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+dth1**2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dth1*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dz*l2*m3*cos(th0)*cos(th2+th1+th0)+g*l2*m3*cos(th2+th1+th0)+2*dth0*dz*l1*m3*sin(th0)*sin(th1+th0)+2*dth0*dz*l1*m2*sin(th0)*sin(th1+th0)-dth1**2*l0*l1*m3*cos(th0)*sin(th1+th0)-2*dth0*dth1*l0*l1*m3*cos(th0)*sin(th1+th0)-dth1**2*l0*l1*m2*cos(th0)*sin(th1+th0)-2*dth0*dth1*l0*l1*m2*cos(th0)*sin(th1+th0)+dth1**2*l0*l1*m3*sin(th0)*cos(th1+th0)+2*dth0*dth1*l0*l1*m3*sin(th0)*cos(th1+th0)+dth1**2*l0*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dth1*l0*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dz*l1*m3*cos(th0)*cos(th1+th0)+2*dth0*dz*l1*m2*cos(th0)*cos(th1+th0)+g*l1*m3*cos(th1+th0)+g*l1*m2*cos(th1+th0)+2*dth0*dz*l0*m3*sin(th0)**2+2*dth0*dz*l0*m2*sin(th0)**2+2*dth0*dz*l0*m1*sin(th0)**2+2*dth0*dz*l0*m3*cos(th0)**2+2*dth0*dz*l0*m2*cos(th0)**2+2*dth0*dz*l0*m1*cos(th0)**2+g*l0*m3*cos(th0)+g*l0*m2*cos(th0)+g*l0*m1*cos(th0)
    b[4] = dth0**2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-dth0**2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+dth0**2*l1*m3*cos(th0)*sin(th1+th0)*z+dth0**2*l1*m2*cos(th0)*sin(th1+th0)*z-dth0**2*l1*m3*sin(th0)*cos(th1+th0)*z-dth0**2*l1*m2*sin(th0)*cos(th1+th0)*z-dth2**2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)-2*dth1*dth2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)-2*dth0*dth2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+2*dth0*dz*l2*m3*sin(th0)*sin(th2+th1+th0)+dth0**2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)+dth2**2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+2*dth1*dth2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)+2*dth0*dth2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)-dth0**2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dz*l2*m3*cos(th0)*cos(th2+th1+th0)+g*l2*m3*cos(th2+th1+th0)+2*dth0*dz*l1*m3*sin(th0)*sin(th1+th0)+2*dth0*dz*l1*m2*sin(th0)*sin(th1+th0)+dth0**2*l0*l1*m3*cos(th0)*sin(th1+th0)+dth0**2*l0*l1*m2*cos(th0)*sin(th1+th0)-dth0**2*l0*l1*m3*sin(th0)*cos(th1+th0)-dth0**2*l0*l1*m2*sin(th0)*cos(th1+th0)+2*dth0*dz*l1*m3*cos(th0)*cos(th1+th0)+2*dth0*dz*l1*m2*cos(th0)*cos(th1+th0)+g*l1*m3*cos(th1+th0)+g*l1*m2*cos(th1+th0)
    b[5] = dth0**2*l2*m3*cos(th0)*sin(th2+th1+th0)*z-dth0**2*l2*m3*sin(th0)*cos(th2+th1+th0)*z+dth1**2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+2*dth0*dth1*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+dth0**2*l1*l2*m3*cos(th1+th0)*sin(th2+th1+th0)+2*dth0*dz*l2*m3*sin(th0)*sin(th2+th1+th0)+dth0**2*l0*l2*m3*cos(th0)*sin(th2+th1+th0)-dth1**2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)-2*dth0*dth1*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)-dth0**2*l1*l2*m3*sin(th1+th0)*cos(th2+th1+th0)-dth0**2*l0*l2*m3*sin(th0)*cos(th2+th1+th0)+2*dth0*dz*l2*m3*cos(th0)*cos(th2+th1+th0)+g*l2*m3*cos(th2+th1+th0)

    return A, b, extf

def f_air(t, s, u, params={}):
    A, b, extf = _calcAb66(s, u)
    assert np.linalg.matrix_rank(A) == 6, (s, A)
    #Ax + b = extf
    #print(np.linalg.det(A))
    dd = np.linalg.solve(A, extf-b).reshape(6)
    print("A", A)
    print("b", extf-b)
    print("x", dd)

    ds = np.zeros_like(s)
    ds[IDX_x0] = s[IDX_dx]
    ds[IDX_y0] = s[IDX_dy]
    ds[IDX_z]  = s[IDX_dz]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]
    ds[IDX_dx]   = dd[0]
    ds[IDX_dy]   = dd[1]
    ds[IDX_dz]   = dd[2]
    ds[IDX_dth0] = dd[3]
    ds[IDX_dth1] = dd[4]
    ds[IDX_dth2] = dd[5]

    return ds

def f_ground(t, s, u, params={}):
    A, b, extf = _calcAb44(s, u)
    assert np.linalg.matrix_rank(A) == 4
    dd = np.linalg.solve(A, extf-b).reshape(4)

    ds = np.zeros_like(s)
    ds[IDX_x0] = 0
    ds[IDX_y0] = 0
    ds[IDX_z]   = s[IDX_dz]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]
    ds[IDX_dx]   = 0
    ds[IDX_dy]   = 0
    ds[IDX_dz]   = dd[0]
    ds[IDX_dth0] = dd[1]
    ds[IDX_dth1] = dd[2]
    ds[IDX_dth2] = dd[3]

    return ds


# only change velocity
# compensate trans velocity change with rotation change with f_air
def f_collision_impulse(s, u):

    ddxdt = - s[IDX_dx]
    ddydt = - s[IDX_dy]
    A, b, extf = _calcAb66(s, u)

    offset = A[2:,:2] @ np.array([[ddxdt],[ddydt]])
    assert np.linalg.matrix_rank(A[2:,2:]) == 4
    dd = np.linalg.solve(A[2:,2:], -offset).reshape(4)

    new_s = s.copy()
    new_s[IDX_dx]   = 0
    new_s[IDX_dy]   = 0
    new_s[IDX_dz]   = s[IDX_dz]   + dd[0]
    new_s[IDX_dth0] = s[IDX_dth0] + dd[1]
    new_s[IDX_dth1] = s[IDX_dth1] + dd[2]
    new_s[IDX_dth2] = s[IDX_dth2] + dd[3]

    return new_s


def collision_time(s, ds, dt):
    s1 = s + ds * dt
    y0 = s[IDX_y0]
    y1 = s1[IDX_y0]
    if y0 > 0 and y1 <= 0:
        return y0 * dt / (y0 - y1)
    else:
        return None

def step(t, s, u, dt):
    u = torq_limit(s, u)

    #ds = rk4(f_ground, t, s, u, {}, dt)
    #return "constraint", t + dt, postprocess(s + ds * dt)
    ds = rk4(f_air, t, s, u, {}, dt)
    #return "jump", t + dt, postprocess(s + ds * dt)
    if s[IDX_y0] == 0: # already on the ground
        if ds[IDX_y0] > 0:
            return "jump", t + dt, postprocess(s + ds * dt)
        else:
            ds = rk4(f_ground, t, s, u, {}, dt)
            return "constraint", t + dt, postprocess(s + ds * dt)
    elif s[IDX_y0] > 0: # already in the air
        colt = collision_time(s, ds, dt)
        if colt is not None:
            ds = rk4(f_air, t, s, u, {}, colt)
            s = s + ds * colt
            s[IDX_y0] = 0
            s = f_collision_impulse(s, u)
            return  "collision", t + colt, postprocess(s)
        else:
            return "free", t + dt, postprocess(s + ds * dt)
    else:
        assert False


def energyU(s):
    p0, p1, p2, p3 = node_pos(s)
    return p0[1] * m0 * g + p1[1] * m1 * g + p2[1] * m2 * g + p3[1] * m3 * g + 1/2 * k * s[IDX_z] ** 2

def energyT(s):
    v0, v1, v2, v3 = node_vel(s)
    return m0 * v0 @ v0 / 2 + m1 * v1 @ v1 / 2 + m2 * v2 @ v2 / 2 + m3 * v3 @ v3 / 2

def energy(s):
    return energyU(s) + energyT(s)

#------------
# control
#------------
def calc_input(s, dds):
    pass

# x -> u
def pdcontrol(s, pos_ref):
    pass




