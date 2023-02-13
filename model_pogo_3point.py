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
M=[mr, m0, m1, mt]
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
    s = np.zeros(IDX_MAX, dtype=np.float64)
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
        print(f"{titlePrefix}OBJ{i:}: P{i}: {ps[i]}:.2f}")

def max_u():
    return np.array([MAX_TORQUE0, MAX_TORQUEK])

def torq_limit(s, u):
    m = max_u()
    ret = np.clip(u, -m, m)
    return ret

def ref_clip(ref):
    return np.clip(ref, REF_MIN, REF_MAX)

def node_pos(s):
    pr = np.array([s[IDX_xr], s[IDX_yr ])
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
    dir_thr = np.array([-np.sin(thr), np.cos(thr)])
    dir_thr0 = np.array([-np.sin(thr+th0), np.cos(thr+th0)])
    dir_dthr = np.array([-np.cos(thr), -np.sin(thr)])
    dir_dthr0 = np.array([-np.cos(thr+th0), -np.sin(thr+th0)])
    vr = np.array(s[IDX_xr  + IDX_MAX], s[IDX_yr  + IDX_MAX])
    dthr = s[IDX_thr + IDX_MAX]
    dz   = s[IDX_z   + IDX_MAX]
    dth0 = s[IDX_th0 + IDX_MAX]
    dthk = s[IDX_thk + IDX_MAX]

    v0 = vr + dz*dir_thr + dthr*(z0+z)*dir_dthr
    vt = v0 + dthr*lt*dir_dthr
    v1 = v0 + dd*dir_thr0 + (dthr+dth0)*a*dir_dthr0

    return vr, v0, v1, vt

def ground(s):
    return s[IDX_yr] == 0

OBS_MIN = np.array([0, limit_min_th0, limit_min_thk, -z0, -MAX_SPEED, -MAX_SPEED, -MAX_ROT_SPEED, -MAX_ROT_SPEED, -MAX_SPEED])
OBS_MAX = np.array([5, limit_max_th0, limit_max_thk,   0, MAX_SPEED , MAX_SPEED , MAX_ROT_SPEED, MAX_ROT_SPEED, MAX_SPEED  ])

def obs(s):
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

def step(t, s, u, dt):
    pass

def energyS(s):
    return 1/2 * K * z[IDX_z] ** 2

def energyU(s):
    ps = list(node_pos(s))
    return sum([g * ps[i][1] * M[i] for i in range(len(ps))]) + energyS(s)

def energyTy(s):
    vs = list(node_vel(s))
    return sum([1/2 * (vs[i][1] ** 2) * M[i] for i in range(len(vs))])

def energyT(s):
    vs = list(node_vel(s))
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
    ob = np.array([s[IDX_th0], s[IDX_thk])
    err = ref - ob
    debug_print(f"PD-ref: {np.rad2deg(ref[0])} {ref[1]}")
    debug_print(f"PD-obs: {np.rad2deg(ob[0])}  {ob[1]}")
    debug_print(f"PD-vel: {dob}")
    ret = err * Kp - Kd * dob
    return ret


