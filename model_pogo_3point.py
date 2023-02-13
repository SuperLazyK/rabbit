import numpy as np
from math import atan2, acos, sqrt
from numpy import sin, cos, abs
import sys
import scipy

debug=0
np.set_printoptions(linewidth=np.inf)

def debug_print(x):
    if debug:
        print(x)

def normalize_angle(x):
    return (x + np. pi) % (2 * np.pi) - np.pi

# Pogo-rotation-knee-phy
max_z = 0.55
z0 = 0.55
lt = 0.83
mr = 0
m0 = 20
m1 = 50
mt = 7
M=np.array([mr, m0, m1, mt])
l = 1.2
g  = 0
#g  = 9.8
#g  = -9.8
K  = 15000 # mgh = 1/2 k x^2 -> T=2*pi sqrt(m/k)
c = 0
#c = 10


# ccw is positive
ref_min_th0 = np.deg2rad(0)
ref_max_th0 = np.deg2rad(20)
ref_min_thk = np.deg2rad(0)
ref_max_thk = np.deg2rad(30)
REF_MIN = np.array([ref_min_th0, ref_min_thk])
REF_MAX = np.array([ref_max_th0, ref_max_thk])

limit_min_th0 = np.deg2rad(-10)
limit_max_th0 = np.deg2rad(40)
limit_min_thk = np.deg2rad(-10)
limit_max_thk = np.deg2rad(40)

MAX_ROT_SPEED=100
MAX_SPEED=100
MAX_TORQUEK=300 # arm
MAX_TORQUE0=800 # arm

inf = float('inf')
#Kp = np.array([4000, 13000])
Kp = 5*np.array([400, 800])
#Kp = np.array([400, 800])
#Kp = np.array([400, 400, 800])
#Kd = Kp * (0.01)
Kd = Kp * (0.1)

#-----------------
# State
#-----------------

IDX_xr   = 0
IDX_yr   = 1
IDX_thr  = 2
IDX_z    = 3
IDX_th0  = 4
IDX_thk  = 5
IDX_MAX = 12

def reset_state(pr, thr, th0, thk, vr = np.array([0,0]), dthr=0, dth0=0, dthk=0, z = 0, dz = 0):
    s = np.zeros(2*IDX_MAX)
    s[IDX_xr ] = pr[0]
    s[IDX_yr ] = pr[1]
    s[IDX_thr] = thr
    s[IDX_z  ] = z
    s[IDX_th0] = th0
    s[IDX_thk] = thk

    s[IDX_xr  + IDX_MAX] = vr[0]
    s[IDX_yr  + IDX_MAX] = vr[1]
    s[IDX_thr + IDX_MAX] = dthr
    s[IDX_z   + IDX_MAX] = dz
    s[IDX_th0 + IDX_MAX] = dth0
    s[IDX_thk + IDX_MAX] = dthk
    return s


def print_state(s, titlePrefix="", fieldPrefix=""):
    ps = node_pos(s)
    for i in range(len(ps)):
        print(f"{titlePrefix} OBJ{i:}: P{i}: {ps[i]:.2f}")

def calc_joint_property(s):
    d = {}
    d['xr'] = s[IDX_xr ]
    d['yr'] = s[IDX_yr ]
    d['thr'] = s[IDX_thr]
    d['z'] = s[IDX_z  ]
    d['th0'] = s[IDX_th0]
    d['thk'] = s[IDX_thk]
    return d

def max_u():
    return np.array([MAX_TORQUE0, MAX_TORQUEK])

def torq_limit(s, u):
    m = max_u()
    ret = np.clip(u, -m, m)
    return ret

def ref_clip(ref):
    return np.clip(ref, REF_MIN, REF_MAX)

def node_pos(s):
    pr = np.array([s[IDX_xr], s[IDX_yr]])
    thr = s[IDX_thr]
    z   = s[IDX_z  ]
    th0 = s[IDX_th0]
    thk = s[IDX_thk]

    dir_thr = np.array([-np.sin(thr), np.cos(thr)])
    dir_thr0 = np.array([-np.sin(thr+th0), np.cos(thr+th0)])

    l1 = l * np.cos(thk)
    p0 = pr + (z0 + z) * dir_thr
    pt = p0 + lt * dir_thr
    p1 = p0 + l * dir_thr0

    return pr, p0, p1, pt


def node_vel(s):
    thr = s[IDX_thr]
    z   = s[IDX_z  ]
    th0 = s[IDX_th0]
    thk = s[IDX_thk]
    dir_thr = np.array([-np.sin(thr), np.cos(thr)])
    dir_thr0 = np.array([-np.sin(thr+th0), np.cos(thr+th0)])
    dir_dthr = np.array([-np.cos(thr), -np.sin(thr)])
    dir_dthr0 = np.array([-np.cos(thr+th0), -np.sin(thr+th0)])
    vr = np.array([s[IDX_xr  + IDX_MAX], s[IDX_yr  + IDX_MAX]])
    dthr = s[IDX_thr + IDX_MAX]
    dz   = s[IDX_z   + IDX_MAX]
    dth0 = s[IDX_th0 + IDX_MAX]
    dthk = s[IDX_thk + IDX_MAX]

    v0 = vr + dz*dir_thr + dthr*(z0+z)*dir_dthr
    vt = v0 + dthr*lt*dir_dthr
    v1 = v0 - dthk*l*sin(thk)*dir_thr + (dthr+dth0)*l*cos(thk)*dir_dthr0
    return vr, v0, v1, vt

OBS_MIN = np.array([0, limit_min_th0, limit_min_thk, -max_z, -MAX_SPEED, -MAX_SPEED, -MAX_ROT_SPEED, -MAX_ROT_SPEED, -MAX_SPEED])
OBS_MAX = np.array([5, limit_max_th0, limit_max_thk,   0, MAX_SPEED , MAX_SPEED , MAX_ROT_SPEED, MAX_ROT_SPEED, MAX_SPEED  ])

def obs(s):
    o = s.copy()
    o[0] = 1 if ground(s) else 0
    return s[1:]

def reward(s):
    return 0

def init_ref(s):
    return np.array([s[IDX_th0], s[IDX_thk]])

def check_invariant(s):
    ps = list(node_pos(s))
    for i in range(1,len(ps)):
        if ps[i][1] <= 0.001:
            reason = f"GAME OVER @ p{i}={ps[i]:}"
            return False, reason
    return True, ""

def energyS(s):
    return 1/2 * K * s[IDX_z] ** 2

def energyU(s):
    ps = list(node_pos(s))
    return sum([g * ps[i][1] * M[i] for i in range(len(ps))]) + energyS(s)

def energyTy(s):
    vs = list(node_vel(s))
    return sum([1/2 * (vs[i][1] ** 2) * M[i] for i in range(len(vs))])

def energyT(s):
    vs = list(node_vel(s))
    print(vs)
    return sum([1/2 * (vs[i] @ vs[i]) * M[i] for i in range(len(vs))])

def energy(s):
    return energyU(s) + energyT(s)

def cog(s):
    ps = list(node_pos(s))
    p = sum([M[i]*ps[i] for i in range(len(ps))])/sum(M)
    return p

def dcog(s):
    vs = list(node_vel(s))
    v = sum([M[i]*vs[i] for i in range(len(vs))])/sum(M)
    return v

def moment(s):
    vs = list(node_vel(s))
    tm = sum([M[i] * vs[i] for i in range(len(vs))])
    ps = list(node_pos(s))
    am = sum([M[i]*np.cross(vs[i]-vs[0], vs[i]-ps[0]) for i in range(len(vs))])
    return tm[0], tm[1], am

def pdcontrol(s, ref):
    dob  = np.array([s[IDX_MAX + IDX_th0], s[IDX_MAX + IDX_thk]])
    ob = np.array([s[IDX_th0], s[IDX_thk]])
    err = ref - ob
    debug_print(f"PD-ref: {np.rad2deg(ref[0])} {ref[1]}")
    debug_print(f"PD-obs: {np.rad2deg(ob[0])}  {ob[1]}")
    debug_print(f"PD-vel: {dob}")
    ret = err * Kp - Kd * dob
    return ret


def ground(s):
    return s[IDX_yr] == 0 and s[IDX_xr+IDX_MAX] == 0 and s[IDX_yr+IDX_MAX] == 0

def jumpup(s):
    s[IDX_MAX + IDX_xr] = - s[IDX_dz] * np.sin(s[IDX_thr])
    s[IDX_MAX + IDX_yr] = s[IDX_dz] * np.cos(s[IDX_thr])
    s[IDX_z] = 0
    s[IDX_MAX + IDX_z] = 0

def land(s):
    s[IDX_z] = 0
    s[IDX_MAX + IDX_z] = sqrt(s[IDX_MAX + IDX_xr]**2 + s[IDX_MAX + IDX_xr]**2)
    s[IDX_MAX + IDX_xr] = 0
    s[IDX_MAX + IDX_yr] = 0
    s[IDX_yr] = 0

def groundAb(s, u):
    z    = s[IDX_z]
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    thk  = s[IDX_thk]
    dz   = s[IDX_dz  ]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dthk = s[IDX_dthk]
    fz   = 0
    taur = 0
    tau0 = u[0]
    tauk = u[1]
    extf = np.array([fz, taur, tau0, tauk]).reshape(4,1)

    A = np.zeros((4,4))
    A[0][0] = mt+m1+m0
    A[0][1] = -l*m1*sin(th0)*cos(thk)
    A[0][2] = -l*m1*sin(th0)*cos(thk)
    A[0][3] = -l*m1*cos(th0)*sin(thk)
    A[1][0] = -l*m1*sin(th0)*cos(thk)
    A[1][1] = (mt+m1+m0)*z0**2+((2*mt+2*m1+2*m0)*z+2*l*m1*cos(th0)*cos(thk)+2*lt*mt)*z0+(mt+m1+m0)*z**2+(2*l*m1*cos(th0)*cos(thk)+2*lt*mt)*z+l**2*m1*cos(thk)**2+lt**2*mt
    A[1][2] = l*m1*cos(th0)*cos(thk)*z0+l*m1*cos(th0)*cos(thk)*z+l**2*m1*cos(thk)**2
    A[1][3] = (-l*m1*sin(th0)*sin(thk)*z0)-l*m1*sin(th0)*sin(thk)*z
    A[2][0] = -l*m1*sin(th0)*cos(thk)
    A[2][1] = l*m1*cos(th0)*cos(thk)*z0+l*m1*cos(th0)*cos(thk)*z+l**2*m1*cos(thk)**2
    A[2][2] = l**2*m1*cos(thk)**2
    A[2][3] = 0
    A[3][0] = -l*m1*cos(th0)*sin(thk)
    A[3][1] = (-l*m1*sin(th0)*sin(thk)*z0)-l*m1*sin(th0)*sin(thk)*z
    A[3][2] = 0
    A[3][3] = l**2*m1*sin(thk)**2

    b = np.zeros((4,1))
    b[0] = (-dthr**2*mt*sin(thr)**2*z0)-dthr**2*m1*sin(thr)**2*z0-dthr**2*m0*sin(thr)**2*z0-dthr**2*mt*cos(thr)**2*z0-dthr**2*m1*cos(thr)**2*z0-dthr**2*m0*cos(thr)**2*z0-dthr**2*mt*sin(thr)**2*z-dthr**2*m1*sin(thr)**2*z-dthr**2*m0*sin(thr)**2*z-dthr**2*mt*cos(thr)**2*z-dthr**2*m1*cos(thr)**2*z-dthr**2*m0*cos(thr)**2*z+k*z-dthr**2*l*m1*cos(thk)*sin(thr)*sin(thr+th0)-2*dth0*dthr*l*m1*cos(thk)*sin(thr)*sin(thr+th0)-dthk**2*l*m1*cos(thk)*sin(thr)*sin(thr+th0)-dth0**2*l*m1*cos(thk)*sin(thr)*sin(thr+th0)+2*dthk*dthr*l*m1*sin(thk)*cos(thr)*sin(thr+th0)+2*dth0*dthk*l*m1*sin(thk)*cos(thr)*sin(thr+th0)-2*dthk*dthr*l*m1*sin(thk)*sin(thr)*cos(thr+th0)-2*dth0*dthk*l*m1*sin(thk)*sin(thr)*cos(thr+th0)-dthr**2*l*m1*cos(thk)*cos(thr)*cos(thr+th0)-2*dth0*dthr*l*m1*cos(thk)*cos(thr)*cos(thr+th0)-dthk**2*l*m1*cos(thk)*cos(thr)*cos(thr+th0)-dth0**2*l*m1*cos(thk)*cos(thr)*cos(thr+th0)-dthr**2*lt*mt*sin(thr)**2-dthr**2*lt*mt*cos(thr)**2+g*mt*cos(thr)+g*m1*cos(thr)+g*m0*cos(thr)
    b[1] = (-2*dthk*dthr*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0)-2*dth0*dthk*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0-2*dth0*dthr*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dthk**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dth0**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0+2*dth0*dthr*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0+dthk**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0+dth0**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0-2*dthk*dthr*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0-2*dth0*dthk*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0+2*dthr*dz*mt*sin(thr)**2*z0+2*dthr*dz*m1*sin(thr)**2*z0+2*dthr*dz*m0*sin(thr)**2*z0-g*mt*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0+2*dthr*dz*mt*cos(thr)**2*z0+2*dthr*dz*m1*cos(thr)**2*z0+2*dthr*dz*m0*cos(thr)**2*z0-2*dthk*dthr*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z-2*dth0*dthk*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z-2*dth0*dthr*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z-dthk**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z-dth0**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z+2*dth0*dthr*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z+dthk**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z+dth0**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z-2*dthk*dthr*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z-2*dth0*dthk*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z+2*dthr*dz*mt*sin(thr)**2*z+2*dthr*dz*m1*sin(thr)**2*z+2*dthr*dz*m0*sin(thr)**2*z-g*mt*sin(thr)*z-g*m1*sin(thr)*z-g*m0*sin(thr)*z+2*dthr*dz*mt*cos(thr)**2*z+2*dthr*dz*m1*cos(thr)**2*z+2*dthr*dz*m0*cos(thr)**2*z-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+2*dthr*dz*l*m1*cos(thk)*sin(thr)*sin(thr+th0)-g*l*m1*cos(thk)*sin(thr+th0)-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+2*dthr*dz*l*m1*cos(thk)*cos(thr)*cos(thr+th0)+2*dthr*dz*lt*mt*sin(thr)**2-g*lt*mt*sin(thr)+2*dthr*dz*lt*mt*cos(thr)**2
    b[2] = dthr**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dthr**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0+dthr**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z-dthr**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+2*dthr*dz*l*m1*cos(thk)*sin(thr)*sin(thr+th0)-g*l*m1*cos(thk)*sin(thr+th0)-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+2*dthr*dz*l*m1*cos(thk)*cos(thr)*cos(thr+th0)
    b[3] = dthr**2*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0+dthr**2*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0+dthr**2*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z+dthr**2*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z+dthr**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+2*dth0*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+dthk**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+dth0**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-2*dthr*dz*l*m1*sin(thk)*cos(thr)*sin(thr+th0)+dthr**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+2*dth0*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+dthk**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+dth0**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+2*dthr*dz*l*m1*sin(thk)*sin(thr)*cos(thr+th0)-g*l*m1*sin(thk)*cos(thr+th0)
    return A, b, extf


def airAb(s, u):
    xr   = s[IDX_xr]
    yr   = s[IDX_yr]
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    thk  = s[IDX_thk]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dthk = s[IDX_dthk]
    tau0 = u[0]
    tau1 = u[1]
    extf = np.array([0, 0, 0, tau0, tau1]).reshape(5,1)

    A = np.zeros((5,5))
    A[0][0] = mt+m1+m0
    A[0][1] = 0
    A[0][2] = ((-mt)-m1-m0)*cos(thr)*z0+l*m1*sin(th0)*cos(thk)*sin(thr)+((-l*m1*cos(th0)*cos(thk))-lt*mt)*cos(thr)
    A[0][3] = l*m1*sin(th0)*cos(thk)*sin(thr)-l*m1*cos(th0)*cos(thk)*cos(thr)
    A[0][4] = l*m1*cos(th0)*sin(thk)*sin(thr)+l*m1*sin(th0)*sin(thk)*cos(thr)
    A[1][0] = 0
    A[1][1] = mt+m1+m0
    A[1][2] = ((-mt)-m1-m0)*sin(thr)*z0+((-l*m1*cos(th0)*cos(thk))-lt*mt)*sin(thr)-l*m1*sin(th0)*cos(thk)*cos(thr)
    A[1][3] = (-l*m1*cos(th0)*cos(thk)*sin(thr))-l*m1*sin(th0)*cos(thk)*cos(thr)
    A[1][4] = l*m1*sin(th0)*sin(thk)*sin(thr)-l*m1*cos(th0)*sin(thk)*cos(thr)
    A[2][0] = ((-mt)-m1-m0)*cos(thr)*z0+l*m1*sin(th0)*cos(thk)*sin(thr)+((-l*m1*cos(th0)*cos(thk))-lt*mt)*cos(thr)
    A[2][1] = ((-mt)-m1-m0)*sin(thr)*z0+((-l*m1*cos(th0)*cos(thk))-lt*mt)*sin(thr)-l*m1*sin(th0)*cos(thk)*cos(thr)
    A[2][2] = (mt+m1+m0)*z0**2+(2*l*m1*cos(th0)*cos(thk)+2*lt*mt)*z0+l**2*m1*cos(thk)**2+lt**2*mt
    A[2][3] = l*m1*cos(th0)*cos(thk)*z0+l**2*m1*cos(thk)**2
    A[2][4] = -l*m1*sin(th0)*sin(thk)*z0
    A[3][0] = l*m1*sin(th0)*cos(thk)*sin(thr)-l*m1*cos(th0)*cos(thk)*cos(thr)
    A[3][1] = (-l*m1*cos(th0)*cos(thk)*sin(thr))-l*m1*sin(th0)*cos(thk)*cos(thr)
    A[3][2] = l*m1*cos(th0)*cos(thk)*z0+l**2*m1*cos(thk)**2
    A[3][3] = l**2*m1*cos(thk)**2
    A[3][4] = 0
    A[4][0] = l*m1*cos(th0)*sin(thk)*sin(thr)+l*m1*sin(th0)*sin(thk)*cos(thr)
    A[4][1] = l*m1*sin(th0)*sin(thk)*sin(thr)-l*m1*cos(th0)*sin(thk)*cos(thr)
    A[4][2] = -l*m1*sin(th0)*sin(thk)*z0
    A[4][3] = 0
    A[4][4] = l**2*m1*sin(thk)**2

    b = np.zeros((5,1))
    b[0] = dthr**2*mt*sin(thr)*z0+dthr**2*m1*sin(thr)*z0+dthr**2*m0*sin(thr)*z0+dthr**2*l*m1*cos(thk)*sin(thr+th0)+2*dth0*dthr*l*m1*cos(thk)*sin(thr+th0)+dthk**2*l*m1*cos(thk)*sin(thr+th0)+dth0**2*l*m1*cos(thk)*sin(thr+th0)+2*dthk*dthr*l*m1*sin(thk)*cos(thr+th0)+2*dth0*dthk*l*m1*sin(thk)*cos(thr+th0)+dthr**2*lt*mt*sin(thr)-ddz*mt*sin(thr)-ddz*m1*sin(thr)-ddz*m0*sin(thr)
    b[1] = (-dthr**2*mt*cos(thr)*z0)-dthr**2*m1*cos(thr)*z0-dthr**2*m0*cos(thr)*z0+2*dthk*dthr*l*m1*sin(thk)*sin(thr+th0)+2*dth0*dthk*l*m1*sin(thk)*sin(thr+th0)-dthr**2*l*m1*cos(thk)*cos(thr+th0)-2*dth0*dthr*l*m1*cos(thk)*cos(thr+th0)-dthk**2*l*m1*cos(thk)*cos(thr+th0)-dth0**2*l*m1*cos(thk)*cos(thr+th0)-dthr**2*lt*mt*cos(thr)+ddz*mt*cos(thr)+ddz*m1*cos(thr)+ddz*m0*cos(thr)+g*mt+g*m1+g*m0
    b[2] = (-2*dthk*dthr*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0)-2*dth0*dthk*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0-2*dth0*dthr*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dthk**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dth0**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0+2*dth0*dthr*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0+dthk**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0+dth0**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0-2*dthk*dthr*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0-2*dth0*dthk*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0-g*mt*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-ddz*l*m1*cos(thk)*cos(thr)*sin(thr+th0)-g*l*m1*cos(thk)*sin(thr+th0)-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+ddz*l*m1*cos(thk)*sin(thr)*cos(thr+th0)-g*lt*mt*sin(thr)
    b[3] = dthr**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dthr**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-ddz*l*m1*cos(thk)*cos(thr)*sin(thr+th0)-g*l*m1*cos(thk)*sin(thr+th0)-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+ddz*l*m1*cos(thk)*sin(thr)*cos(thr+th0)
    b[4] = dthr**2*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0+dthr**2*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0+dthr**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+2*dth0*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+dthk**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+dth0**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-ddz*l*m1*sin(thk)*sin(thr)*sin(thr+th0)+dthr**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+2*dth0*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+dthk**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+dth0**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-ddz*l*m1*sin(thk)*cos(thr)*cos(thr+th0)-g*l*m1*sin(thk)*cos(thr+th0)

    return A, b, extf

def f_ground(s, u):
    A, b, extf = groundAb(s, u)
    assert np.linalg.matrix_rank(A) == 4
    y = np.linalg.solve(A, extf-b).reshape(4)
    f = np.zeros(IDX_MAX)
    f[IDX_dx]   = 0
    f[IDX_dy]   = 0
    f[IDX_dz]   = y[0]
    f[IDX_dthr] = y[1]
    f[IDX_dth0] = y[2]
    f[IDX_dthk] = y[3]
    return f

def f_air(s, u):
    A, b, extf = airAb(s, u)
    assert np.linalg.matrix_rank(A) == 5, (s, A)
    y = np.linalg.solve(A, extf-b).reshape(5)
    f = np.zeros(IDX_MAX)
    f[IDX_dx]   = y[0]
    f[IDX_dy]   = y[1]
    f[IDX_dthr] = y[2]
    f[IDX_dz]   = 0
    f[IDX_dth0] = y[3]
    f[IDX_dthk] = y[4]
    return f


def step(t, s, u, dt):
    u = torq_limit(s, u)
    ret = np.zeros(2*IDX_MAX)
    if ground(s):
        impulse = f_ground(s, u) * dt
        ret[IDX_MAX:] = s[IDX_MAX:] + impulse
        ret[:IDX_MAX] = s[:IDX_MAX] + ret[IDX_MAX:] * dt
        if ret[IDX_z] >= 0:
            return "jump", t + dt, jumpup(ret)
        else:
            return "ground", t + dt, ret
    else:
        impulse = f_air(s, u) * dt
        ret[IDX_MAX:] = s[IDX_MAX:] + impulse
        ret[:IDX_MAX] = s[:IDX_MAX] + ret[IDX_MAX:] * dt
        if ret[IDX_yr] <= 0:
            return "land", t + dt, land(ret)
        else:
            return "air", t + dt, ret

