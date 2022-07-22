import numpy as np
from numpy import sin, cos
import sys

#-----------------
# State
#-----------------

IDX_x0   = 0
IDX_y0   = 1
IDX_th0  = 2
IDX_th1  = 3
IDX_th2  = 4
IDX_dx   = 5
IDX_dy   = 6
IDX_dth0 = 7
IDX_dth1 = 8
IDX_dth2 = 9

def reset_state(np_random=None):
    s = np.zeros(10, dtype=np.float32)
    #s[IDX_x0]   = -10.
    s[IDX_x0]   = 0.
    s[IDX_y0]   = 1.
    s[IDX_th0]  = np.pi/4
    s[IDX_th1]  = np.pi*3/4 - np.pi/4
    s[IDX_th2]  = np.pi*5/12 - np.pi*3/4
    s[IDX_dx  ] = 0.
    s[IDX_dy  ] = 0.
    s[IDX_dth0] = 0
    s[IDX_dth1] = 0
    s[IDX_dth2] = 0

    #if np_random is not None:
    #    s[IDX_th0] = s[IDX_th0] + np_random.uniform(low=-np.pi/10, high=np.pi/10)
    #    s[IDX_th1] = s[IDX_th1] + np_random.uniform(low=-np.pi/10, high=np.pi/10)
    #    s[IDX_th2] = s[IDX_th2] + np_random.uniform(low=-np.pi/4, high=np.pi/4)

    return s

def print_state(s):
    print("")
    print(f"x0:  {s[IDX_x0 ]:.2f} ")
    print(f"y0:  {s[IDX_y0 ]:.2f} ")
    print(f"th0: {s[IDX_th0]:.2f} ")
    print(f"th1: {s[IDX_th1]:.2f} ")
    print(f"th2: {s[IDX_th2]:.2f} ")
    print("")
    print(f"dx:  {s[IDX_dx  ]:.2f}")
    print(f"dy:  {s[IDX_dy  ]:.2f}")
    print(f"dth0:{s[IDX_dth0]:.2f}")
    print(f"dth1:{s[IDX_dth1]:.2f}")
    print(f"dth2:{s[IDX_dth2]:.2f}")

#-----------------
# Kinematics
#-----------------

l0 =  1.
l1 =  1.
l2 =  1.

def normalize_angle(s):
    new_s = s.copy()
    new_s[IDX_th0] = ((s[IDX_th0] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th1] = ((s[IDX_th1] + np.pi)% (2*np.pi)) - np.pi
    new_s[IDX_th2] = ((s[IDX_th2] + np.pi)% (2*np.pi)) - np.pi
    return new_s
    #return s

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

    p0 = s[IDX_x0:IDX_y0+1].copy()
    p1 = p0 + l0 * np.array([np.cos(th0), np.sin(th0)])
    p2 = p1 + l1 * np.array([np.cos(th0+th1), np.sin(th0+th1)])
    p3 = p2 + l2 * np.array([np.cos(th0+th1+th2), np.sin(th0+th1+th2)])

    return p0, p1, p2, p3

def node_vel(s):
    th0 = s[IDX_th0]
    th1 = s[IDX_th1]
    th2 = s[IDX_th2]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dth2 = s[IDX_dth2]
    v0 = s[IDX_dx:IDX_dy+1]
    v1 = v0 + np.array([-sin(th0), cos(th0)]) * l0 * dth0
    v2 = v1 + np.array([-sin(th0+th1), cos(th0+th1)]) * l1 * (dth0+dth1)
    v3 = v2 + np.array([-sin(th0+th1+th2), cos(th0+th1+th2)]) * l2 * (dth0+dth1+dth2)

    return v0, v1, v2, v3

#-----------------
# Dynamics Util
#-----------------
def rk4(f, t, s, u, params, dt):

    k1 = f(t,        s,             u, params)
    k2 = f(t + dt/2, s + dt/2 * k1, u, params)
    k3 = f(t + dt/2, s + dt/2 * k2, u, params)
    k4 = f(t + dt,   s + dt * k3,   u, params)

    return (k1 + 2*k2 + 2*k3 + k4)/6


def collision_time(s, ds, dt):
    s1 = s + ds * dt
    y0 = s[IDX_y0]
    y1 = s1[IDX_y0]
    if y0 > 0 and y1 <= 0:
        y0 = s[IDX_y0]
        y1 = s1[IDX_y0]
        return y0 * dt / (y0 - y1)
    else:
        return None

def step_impl(t, s, u, dt, f_air, f_ground, f_collision_impulse):
    ds = rk4(f_air, t, s, u, {}, dt)
    if s[IDX_y0] == 0: # already on the ground
        if ds[IDX_y0] > 0:
            return "jump", t + dt, normalize_angle(s + ds * dt)
        else:
            ds = rk4(f_ground, t, s, u, {}, dt)
            return "constraint", t + dt, normalize_angle(s + ds * dt)
    elif s[IDX_y0] > 0: # already in the air
        colt = collision_time(s, ds, dt)
        if colt is not None:
            ds = rk4(f_air, t, s, u, {}, colt)
            s = s + ds * colt
            s[IDX_y0] = 0
            s = f_collision_impulse(s, u)
            return  "collision", t + colt, normalize_angle(s)
        else:
            return "free", t + dt, normalize_angle(s + ds * dt)
    else:
        assert False

#----------------------------
# Dynamics
#----------------------------

m0 = 0.1
m1 = 0.5
m2 = 0.5
m3 = 0.3
g  = 9.8

def _calcAb(s, u):

    x0   = s[IDX_x0]
    y0   = s[IDX_y0]
    th0  = s[IDX_th0]
    th1  = s[IDX_th1]
    th2  = s[IDX_th2]
    dx   = s[IDX_dx  ]
    dy   = s[IDX_dy  ]
    dth0 = s[IDX_dth0]
    dth1 = s[IDX_dth1]
    dth2 = s[IDX_dth2]
    fx0  = 0
    fy0  = 0
    tau0 = 0
    tau1 = u[0]
    tau2 = u[1]
    extf = np.array([fx0, fy0, tau0, tau1, tau2]).reshape(5,1)

    A = np.array([ [m3+m2+m1+m0,0,(-l2*m3*sin(th2+th1+th0))-l1*m3*sin(th1+th0)-l1*m2*sin(th1+th0)-l0*m3*sin(th0)-l0*m2*sin(th0)-l0*m1*sin(th0),(-l2*m3*sin(th2+th1+th0))-l1*m3*sin(th1+th0)-l1*m2*sin(th1+th0),-l2*m3*sin(th2+th1+th0)]
                 , [0,m3+m2+m1+m0,l2*m3*cos(th2+th1+th0)+l1*m3*cos(th1+th0)+l1*m2*cos(th1+th0)+l0*m3*cos(th0)+l0*m2*cos(th0)+l0*m1*cos(th0),l2*m3*cos(th2+th1+th0)+l1*m3*cos(th1+th0)+l1*m2*cos(th1+th0),l2*m3*cos(th2+th1+th0)]
                 , [(-l2*m3*sin(th2+th1+th0))-l1*m3*sin(th1+th0)-l1*m2*sin(th1+th0)-l0*m3*sin(th0)-l0*m2*sin(th0)-l0*m1*sin(th0),l2*m3*cos(th2+th1+th0)+l1*m3*cos(th1+th0)+l1*m2*cos(th1+th0)+l0*m3*cos(th0)+l0*m2*cos(th0)+l0*m1*cos(th0),2*l1*l2*m3*sin(th1+th0)*sin(th2+th1+th0)+2*l0*l2*m3*sin(th0)*sin(th2+th1+th0)+2*l1*l2*m3*cos(th1+th0)*cos(th2+th1+th0)+2*l0*l2*m3*cos(th0)*cos(th2+th1+th0)+2*l0*l1*m3*sin(th0)*sin(th1+th0)+2*l0*l1*m2*sin(th0)*sin(th1+th0)+2*l0*l1*m3*cos(th0)*cos(th1+th0)+2*l0*l1*m2*cos(th0)*cos(th1+th0)+l2**2*m3+l1**2*m3+l0**2*m3+l1**2*m2+l0**2*m2+l0**2*m1,2*l1*l2*m3*sin(th1+th0)*sin(th2+th1+th0)+l0*l2*m3*sin(th0)*sin(th2+th1+th0)+2*l1*l2*m3*cos(th1+th0)*cos(th2+th1+th0)+l0*l2*m3*cos(th0)*cos(th2+th1+th0)+l0*l1*m3*sin(th0)*sin(th1+th0)+l0*l1*m2*sin(th0)*sin(th1+th0)+l0*l1*m3*cos(th0)*cos(th1+th0)+l0*l1*m2*cos(th0)*cos(th1+th0)+l2**2*m3+l1**2*m3+l1**2*m2,l1*l2*m3*sin(th1+th0)*sin(th2+th1+th0)+l0*l2*m3*sin(th0)*sin(th2+th1+th0)+l1*l2*m3*cos(th1+th0)*cos(th2+th1+th0)+l0*l2*m3*cos(th0)*cos(th2+th1+th0)+l2**2*m3]
                 , [(-l2*m3*sin(th2+th1+th0))-l1*m3*sin(th1+th0)-l1*m2*sin(th1+th0),l2*m3*cos(th2+th1+th0)+l1*m3*cos(th1+th0)+l1*m2*cos(th1+th0),2*l1*l2*m3*sin(th1+th0)*sin(th2+th1+th0)+l0*l2*m3*sin(th0)*sin(th2+th1+th0)+2*l1*l2*m3*cos(th1+th0)*cos(th2+th1+th0)+l0*l2*m3*cos(th0)*cos(th2+th1+th0)+l0*l1*m3*sin(th0)*sin(th1+th0)+l0*l1*m2*sin(th0)*sin(th1+th0)+l0*l1*m3*cos(th0)*cos(th1+th0)+l0*l1*m2*cos(th0)*cos(th1+th0)+l2**2*m3+l1**2*m3+l1**2*m2,2*l1*l2*m3*sin(th1+th0)*sin(th2+th1+th0)+2*l1*l2*m3*cos(th1+th0)*cos(th2+th1+th0)+l2**2*m3+l1**2*m3+l1**2*m2,l1*l2*m3*sin(th1+th0)*sin(th2+th1+th0)+l1*l2*m3*cos(th1+th0)*cos(th2+th1+th0)+l2**2*m3]
                 , [-l2*m3*sin(th2+th1+th0),l2*m3*cos(th2+th1+th0),l1*l2*m3*sin(th1+th0)*sin(th2+th1+th0)+l0*l2*m3*sin(th0)*sin(th2+th1+th0)+l1*l2*m3*cos(th1+th0)*cos(th2+th1+th0)+l0*l2*m3*cos(th0)*cos(th2+th1+th0)+l2**2*m3,l1*l2*m3*sin(th1+th0)*sin(th2+th1+th0)+l1*l2*m3*cos(th1+th0)*cos(th2+th1+th0)+l2**2*m3,l2**2*m3]
                 ])

    b = np.array([ [(-dth2**2*l2*m3*cos(th2+th1+th0))-2*dth1*dth2*l2*m3*cos(th2+th1+th0)-2*dth0*dth2*l2*m3*cos(th2+th1+th0)-dth1**2*l2*m3*cos(th2+th1+th0)-2*dth0*dth1*l2*m3*cos(th2+th1+th0)-dth0**2*l2*m3*cos(th2+th1+th0)-dth1**2*l1*m3*cos(th1+th0)-2*dth0*dth1*l1*m3*cos(th1+th0)-dth0**2*l1*m3*cos(th1+th0)-dth1**2*l1*m2*cos(th1+th0)-2*dth0*dth1*l1*m2*cos(th1+th0)-dth0**2*l1*m2*cos(th1+th0)-dth0**2*l0*m3*cos(th0)-dth0**2*l0*m2*cos(th0)-dth0**2*l0*m1*cos(th0)]
                 , [(-dth2**2*l2*m3*sin(th2+th1+th0))-2*dth1*dth2*l2*m3*sin(th2+th1+th0)-2*dth0*dth2*l2*m3*sin(th2+th1+th0)-dth1**2*l2*m3*sin(th2+th1+th0)-2*dth0*dth1*l2*m3*sin(th2+th1+th0)-dth0**2*l2*m3*sin(th2+th1+th0)-dth1**2*l1*m3*sin(th1+th0)-2*dth0*dth1*l1*m3*sin(th1+th0)-dth0**2*l1*m3*sin(th1+th0)-dth1**2*l1*m2*sin(th1+th0)-2*dth0*dth1*l1*m2*sin(th1+th0)-dth0**2*l1*m2*sin(th1+th0)-dth0**2*l0*m3*sin(th0)-dth0**2*l0*m2*sin(th0)-dth0**2*l0*m1*sin(th0)+g*m3+g*m2+g*m1+g*m0]
                 , [g*l2*m3*cos(th2+th1+th0)-dth2**2*l0*l2*m3*sin(th2+th1)-2*dth1*dth2*l0*l2*m3*sin(th2+th1)-2*dth0*dth2*l0*l2*m3*sin(th2+th1)-dth1**2*l0*l2*m3*sin(th2+th1)-2*dth0*dth1*l0*l2*m3*sin(th2+th1)-dth2**2*l1*l2*m3*sin(th2)-2*dth1*dth2*l1*l2*m3*sin(th2)-2*dth0*dth2*l1*l2*m3*sin(th2)+g*l1*m3*cos(th1+th0)+g*l1*m2*cos(th1+th0)-dth1**2*l0*l1*m3*sin(th1)-2*dth0*dth1*l0*l1*m3*sin(th1)-dth1**2*l0*l1*m2*sin(th1)-2*dth0*dth1*l0*l1*m2*sin(th1)+g*l0*m3*cos(th0)+g*l0*m2*cos(th0)+g*l0*m1*cos(th0)]
                 , [g*l2*m3*cos(th2+th1+th0)+dth0**2*l0*l2*m3*sin(th2+th1)-dth2**2*l1*l2*m3*sin(th2)-2*dth1*dth2*l1*l2*m3*sin(th2)-2*dth0*dth2*l1*l2*m3*sin(th2)+g*l1*m3*cos(th1+th0)+g*l1*m2*cos(th1+th0)+dth0**2*l0*l1*m3*sin(th1)+dth0**2*l0*l1*m2*sin(th1)]
                 , [g*l2*m3*cos(th2+th1+th0)+dth0**2*l0*l2*m3*sin(th2+th1)+dth1**2*l1*l2*m3*sin(th2)+2*dth0*dth1*l1*l2*m3*sin(th2)+dth0**2*l1*l2*m3*sin(th2)]
                 ])

    return A, b, extf


def f_air_4mass(t, s, u, params={}):

    A, b, extf = _calcAb(s, u)

    ds = np.zeros_like(s)
    ds[IDX_x0] = s[IDX_dx]
    ds[IDX_y0] = s[IDX_dy]
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]

    assert np.linalg.matrix_rank(A) == 5, (s, A)

    dd = np.linalg.solve(A, extf-b).reshape(5)

    ds[IDX_dx]   = dd[0]
    ds[IDX_dy]   = dd[1]
    ds[IDX_dth0] = dd[2]
    ds[IDX_dth1] = dd[3]
    ds[IDX_dth2] = dd[4]

    return ds

def f_ground_4mass(t, s, u, params={}):

    A, b, extf = _calcAb(s, u)

    ds = np.zeros_like(s)
    ds[IDX_x0] = 0
    ds[IDX_y0] = 0
    ds[IDX_th0] = s[IDX_dth0]
    ds[IDX_th1] = s[IDX_dth1]
    ds[IDX_th2] = s[IDX_dth2]

    assert np.linalg.matrix_rank(A[2:,2:]) == 3
    ddtheta = np.linalg.solve(A[2:,2:], extf[2:]-b[2:]).reshape(3)

    ds[IDX_dx]   = 0
    ds[IDX_dy]   = 0
    ds[IDX_dth0] = ddtheta[0]
    ds[IDX_dth1] = ddtheta[1]
    ds[IDX_dth2] = ddtheta[2]

    return ds


# only change velocity
# compensate trans velocity change with rotation change with f_air
def f_collision_impulse_4mass(s, u):
    ddxdt = - s[IDX_dx]
    ddydt = - s[IDX_dy]
    A, b, extf = _calcAb(s, u)

    offset = A[2:,:2] @ np.array([[ddxdt],[ddydt]])

    assert np.linalg.matrix_rank(A[2:,2:]) == 3
    ddthetadt = np.linalg.solve(A[2:,2:], -offset).reshape(3)

    new_s = s.copy()
    new_s[IDX_dx]   = 0
    new_s[IDX_dy]   = 0
    new_s[IDX_dth0] = s[IDX_dth0] + ddthetadt[0]
    new_s[IDX_dth1] = s[IDX_dth1] + ddthetadt[1]
    new_s[IDX_dth2] = s[IDX_dth2] + ddthetadt[2]

    return new_s


def step(t, s, u, dt):
    return step_impl(t, s, u, dt, f_air_4mass, f_ground_4mass, f_collision_impulse_4mass)

def energyU(s):
    p0, p1, p2, p3 = node_pos(s)
    return p0[1] * m0 * g + p1[1] * m1 * g + p2[1] * m2 * g + p3[1] * m3 * g

def energyT(s):
    v0, v1, v2, v3 = node_vel(s)
    return m0 * v0 @ v0 / 2 + m1 * v1 @ v1 / 2 + m2 * v2 @ v2 / 2 + m3 * v3 @ v3 / 2

def energy(s):
    return energyU(s) + energyT(s)

