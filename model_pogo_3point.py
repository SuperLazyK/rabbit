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
mk = 0
mw = 20
m1 = 30
mt = 7
M=np.array([mr, m0, mk, mw, m1, mt])
IDX_pr=0
IDX_p0=1
IDX_pk=2
IDX_pw=3
IDX_p1=4
IDX_pt=5
#M=np.array([mr, m0, mk, m1, mt])
l0 = 0.4
l1 = 0.5
l2 = 0.3
#l = 1.2
#g  = 0
g  = 9.8
#g  = -9.8
k  = 15000 # mgh = 1/2 k x**2 -> T=2*pi sqrt(m/k)
c = 0
#c = 10


# ccw is positive
ref_min_th0 = np.deg2rad(-30)
ref_max_th0 = np.deg2rad(20)
ref_min_thk = np.deg2rad(1)
ref_max_thk = np.deg2rad(60)
ref_min_thw = np.deg2rad(-50)
ref_max_thw = np.deg2rad(0)
REF_MIN = np.array([ref_min_th0, ref_min_thk, ref_min_thw])
REF_MAX = np.array([ref_max_th0, ref_max_thk, ref_max_thw])
DIM_U = REF_MIN.shape
DEFAULT_U = np.zeros(DIM_U)
limit_min_thr = np.deg2rad(-90)
limit_max_thr = np.deg2rad(90)

limit_min_th0 = np.deg2rad(-45)
limit_max_th0 = np.deg2rad(45)

limit_min_thk = np.deg2rad(0)
limit_max_thk = np.deg2rad(90)
limit_min_thw = np.deg2rad(-90)
limit_max_thw = np.deg2rad(0)

MAX_ROT_SPEED=100
MAX_SPEED=100
#MAX_TORQUEK=3000 # arm
#MAX_TORQUE0=8000 # arm
MAX_TORQUEK=400 # knee(400Nm) + arm(800N * 0.5m)
MAX_TORQUEW=400 # knee(400Nm) + arm(800N * 0.5m)
MAX_TORQUE0=800 # arm(800N x 1m)

inf = float('inf')
#Kp = np.array([4000, 13000])
#Kp = 20*np.array([400, 800])
Kp = 10*np.array([400, 800, 800])
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
IDX_thw  = 6
IDX_MAX = 7
IDX_dxr   = IDX_MAX+IDX_xr 
IDX_dyr   = IDX_MAX+IDX_yr 
IDX_dthr  = IDX_MAX+IDX_thr
IDX_dz    = IDX_MAX+IDX_z  
IDX_dth0  = IDX_MAX+IDX_th0
IDX_dthk  = IDX_MAX+IDX_thk
IDX_dthw  = IDX_MAX+IDX_thw

def reset_state(pr, thr, th0, thk, thw, vr = np.array([0,0]), dthr=0, dth0=0, dthk=0, dthw=0, z = 0, dz = 0):
    s = np.zeros(2*IDX_MAX)
    s[IDX_xr ] = pr[0]
    s[IDX_yr ] = pr[1]
    s[IDX_thr] = thr
    s[IDX_z  ] = z
    s[IDX_th0] = th0
    s[IDX_thk] = thk
    s[IDX_thw] = thw

    s[IDX_dxr ] = vr[0]
    s[IDX_dyr ] = vr[1]
    s[IDX_dthr] = dthr
    s[IDX_dz  ] = dz
    s[IDX_dth0] = dth0
    s[IDX_dthk] = dthk
    s[IDX_dthw] = dthw
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
    d['thw'] = s[IDX_thw]
    d['dxr'] = s[IDX_dxr ]
    d['dyr'] = s[IDX_dyr ]
    d['dthr'] = s[IDX_dthr]
    d['dz'] = s[IDX_dz  ]
    d['dth0'] = s[IDX_dth0]
    d['dthk'] = s[IDX_dthk]
    d['dthw'] = s[IDX_dthw]
    return d

def max_u():
    return np.array([MAX_TORQUE0, MAX_TORQUEK, MAX_TORQUEW])

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
    thw = s[IDX_thw]

    dir_thr = np.array([-np.sin(thr), np.cos(thr)])
    dir_thr0 = np.array([-np.sin(thr+th0), np.cos(thr+th0)])
    dir_thr0k = np.array([-np.sin(thr+th0+thk), np.cos(thr+th0+thk)])
    dir_thr0kw = np.array([-np.sin(thr+th0+thk+thw), np.cos(thr+th0+thk+thw)])

    p0 = pr + (z0 + z) * dir_thr
    pk = p0 + l0 * dir_thr0
    pw = pk + l1 * dir_thr0k
    p1 = pw + l2 * dir_thr0kw
    pt = p0 + lt * dir_thr

    return pr, p0, pk, pw, p1, pt


def node_vel(s):
    thr = s[IDX_thr]
    z   = s[IDX_z  ]
    th0 = s[IDX_th0]
    thk = s[IDX_thk]
    thw = s[IDX_thw]
    dir_thr = np.array([-np.sin(thr), np.cos(thr)])
    dir_thr0 = np.array([-np.sin(thr+th0), np.cos(thr+th0)])
    dir_thr0k = np.array([-np.sin(thr+th0+thk), np.cos(thr+th0+thk)])
    dir_thr0kw = np.array([-np.sin(thr+th0+thk+thw), np.cos(thr+th0+thk+thw)])
    dir_dthr = np.array([-np.cos(thr), -np.sin(thr)])
    dir_dthr0 = np.array([-np.cos(thr+th0), -np.sin(thr+th0)])
    dir_dthr0k = np.array([-np.cos(thr+th0+thk), -np.sin(thr+th0+thk)])
    dir_dthr0kw = np.array([-np.cos(thr+th0+thk+thw), -np.sin(thr+th0+thk+thw)])
    vr = np.array([s[IDX_dxr], s[IDX_dyr]])
    dthr = s[IDX_dthr]
    dz   = s[IDX_dz  ]
    dth0 = s[IDX_dth0]
    dthk = s[IDX_dthk]
    dthw = s[IDX_dthw]

    v0 = vr + dz*dir_thr + dthr*(z0+z)*dir_dthr
    vk = v0 + (dthr+dth0) * l0 * dir_dthr0
    vw = vk + (dthr+dth0+dthk) * l1 * dir_dthr0k
    v1 = vw + (dthr+dth0+dthk+dthw) * l2 * dir_dthr0kw
    vt = v0 + dthr*lt*dir_dthr
    return vr, v0, vk, vw, v1, vt

OBS_MIN = np.array([0, 0, limit_min_thr, -max_z, limit_min_th0, limit_min_thk, limit_min_thw,
                   -MAX_SPEED, -MAX_SPEED, -MAX_ROT_SPEED, -MAX_SPEED, -MAX_ROT_SPEED, -MAX_ROT_SPEED, -MAX_ROT_SPEED])
OBS_MAX = np.array([1, 5, limit_max_thr,      0, limit_max_th0, limit_max_thk, limit_max_thw,
                    MAX_SPEED , MAX_SPEED , MAX_ROT_SPEED, MAX_SPEED, MAX_ROT_SPEED, MAX_ROT_SPEED, MAX_ROT_SPEED])

def obs(s):
    o = s.copy()
    o[0] = 1 if ground(s) else 0
    return o

def reward(s):
    pcog = cog(s)
    r_y = (energyU(s) + energyTy(s))/2000
    r_thr = -abs(s[IDX_thr])*2/np.pi
    r_cogx = -abs(pcog[0]-s[IDX_xr])

    return np.exp(r_y + r_thr + r_cogx)

def init_ref(s):
    return np.array([s[IDX_th0], s[IDX_thk], s[IDX_thw]])

def check_invariant(s):
    ps = list(node_pos(s))
    for i in range(1,len(ps)):
        if ps[i][1] <= 0.001:
            reason = f"GAME OVER @ p{i}={ps[i]:}"
            return False, reason
    if s[IDX_th0] < limit_min_th0 or s[IDX_th0] > limit_max_th0:
            reason = f"GAME OVER @ range error th0={np.rad2deg(s[IDX_th0]):}"
            return False, reason
    if s[IDX_thk] < limit_min_thk or s[IDX_thk] > limit_max_thk:
            reason = f"GAME OVER @ range error thk={np.rad2deg(s[IDX_thk]):}"
            return False, reason
    if s[IDX_thw] < limit_min_thw or s[IDX_thw] > limit_max_thw:
            reason = f"GAME OVER @ range error thw={np.rad2deg(s[IDX_thw]):}"
            return False, reason

    vec_0t = ps[IDX_pt] - ps[IDX_p0]
    vec_0w = ps[IDX_pw] - ps[IDX_p0]

    if  np.cross(vec_0t, vec_0w) < 0:
            reason = f"GAME OVER @ line-w1 < line-0t"
            return False, reason

    pc = cog(s)
    if pc[1] < 0.4:
            reason = f"GAME OVER @ cog too low cogy={pc[1]:}"
            return False, reason
    return True, ""

def energyS(s):
    return 1/2 * k * s[IDX_z] ** 2

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
    dob  = np.array([s[IDX_dth0], s[IDX_dthk], s[IDX_dthw]])
    ob = np.array([s[IDX_th0], s[IDX_thk], s[IDX_thw]])
    err = ref - ob
    debug_print(f"PD-ref: {np.rad2deg(ref[0])} {np.rad2deg(ref[1])} {np.rad2deg(ref[2])} ")
    debug_print(f"PD-obs: {np.rad2deg(ob[0])}  {np.rad2deg(ob[1])}  {np.rad2deg(ob[2])} ")
    debug_print(f"PD-vel: {dob}")
    ret = err * Kp - Kd * dob
    return ret


def ground(s):
    return s[IDX_yr] == 0 and s[IDX_dxr] == 0 and s[IDX_dyr] == 0

def jumpup(s):
    ret = s.copy()
    ret[IDX_dxr] = - s[IDX_dz] * np.sin(s[IDX_thr])
    ret[IDX_dyr] = s[IDX_dz] * np.cos(s[IDX_thr])
    ret[IDX_z] = 0
    ret[IDX_dz] = 0
    return ret

def impulse_collision(s):

    z    = s[IDX_z]
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    thk  = s[IDX_thk]
    thw  = s[IDX_thw]

    A21 = np.zeros((5,2))
    A21[0][0] = ((-mw)-mt-m1-m0)*sin(thr)
    A21[0][1] = (mw+mt+m1+m0)*cos(thr)
    A21[1][0] = ((-mw)-mt-m1-m0)*cos(thr)*z0+((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk)+(l0*mw+l0*m1)*sin(th0))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0)-lt*mt)*cos(thr)
    A21[1][1] = ((-mw)-mt-m1-m0)*sin(thr)*z0+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0)-lt*mt)*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0))*cos(thr)
    A21[2][0] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk)+(l0*mw+l0*m1)*sin(th0))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0))*cos(thr)
    A21[2][1] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0))*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0))*cos(thr)
    A21[3][0] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk))*cos(thr)
    A21[3][1] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk))*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk))*cos(thr)
    A21[4][0] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)
    A21[4][1] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)

    A22 = np.zeros((5,5))
    A22[0][0] = mw+mt+m1+m0
    A22[0][1] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0)
    A22[0][2] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0)
    A22[0][3] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)
    A22[0][4] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)
    A22[1][0] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0)
    A22[1][1] = (mw+mt+m1+m0)*z0**2+(((-2*l2*m1*cos(th0)*sin(thk))-2*l2*m1*sin(th0)*cos(thk))*sin(thw)+(2*l2*m1*cos(th0)*cos(thk)-2*l2*m1*sin(th0)*sin(thk))*cos(thw)+((-2*l1*mw)-2*l1*m1)*sin(th0)*sin(thk)+(2*l1*mw+2*l1*m1)*cos(th0)*cos(thk)+(2*l0*mw+2*l0*m1)*cos(th0)+2*lt*mt)*z0-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+lt**2*mt+(l2**2+l1**2+l0**2)*m1
    A22[1][2] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk)+(l0*mw+l0*m1)*cos(th0))*z0-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A22[1][3] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk))*z0-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A22[1][4] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw))*z0-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A22[2][0] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0)
    A22[2][1] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk)+(l0*mw+l0*m1)*cos(th0))*z0-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A22[2][2] = (-2*l0*l2*m1*sin(thk)*sin(thw))+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A22[2][3] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A22[2][4] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A22[3][0] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)
    A22[3][1] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk))*z0-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A22[3][2] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A22[3][3] = 2*l1*l2*m1*cos(thw)+l1**2*mw+(l2**2+l1**2)*m1
    A22[3][4] = l1*l2*m1*cos(thw)+l2**2*m1
    A22[4][0] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)
    A22[4][1] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw))*z0-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A22[4][2] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A22[4][3] = l1*l2*m1*cos(thw)+l2**2*m1
    A22[4][4] = l2**2*m1
    if np.linalg.matrix_rank(A22) < 5:
        raise Exception("rank")

    #solve  A22 y(z,thr,th0,thk) = A21 (dx,dy)
    y = np.linalg.solve(A22, A21 @ s[IDX_dxr:IDX_dyr+1]).reshape(5)

    d = np.zeros(IDX_MAX)
    d[IDX_xr] = 0
    d[IDX_yr] = 0
    d[IDX_z]   = y[0]
    d[IDX_thr] = y[1]
    d[IDX_th0] = y[2]
    d[IDX_thk] = y[3]
    d[IDX_thw] = y[4]
    return d

def land(s):
    ret = s.copy()
    d = impulse_collision(s)
    ret[IDX_yr] = 0
    ret[IDX_z] = 0
    ret[IDX_MAX:] = d
    #print("land-before", calc_joint_property(s))
    #print("land-after", calc_joint_property(ret))
    return ret

def groundAb(s, u):
    z    = s[IDX_z]
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    thk  = s[IDX_thk]
    thw  = s[IDX_thw]
    dz   = s[IDX_dz  ]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dthk = s[IDX_dthk]
    dthw = s[IDX_dthw]
    fz   = 0
    taur = 0
    tau0 = u[0]
    tauk = u[1]
    tauw = u[2]
    extf = np.array([fz, taur, tau0, tauk, tauw]).reshape(5,1)

    A = np.zeros((5,5))
    A[0][0] = mw+mt+m1+m0
    A[0][1] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0)
    A[0][2] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0)
    A[0][3] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)
    A[0][4] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)
    A[1][0] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0)
    A[1][1] = (mw+mt+m1+m0)*z0**2+((2*mw+2*mt+2*m1+2*m0)*z+((-2*l2*m1*cos(th0)*sin(thk))-2*l2*m1*sin(th0)*cos(thk))*sin(thw)+(2*l2*m1*cos(th0)*cos(thk)-2*l2*m1*sin(th0)*sin(thk))*cos(thw)+((-2*l1*mw)-2*l1*m1)*sin(th0)*sin(thk)+(2*l1*mw+2*l1*m1)*cos(th0)*cos(thk)+(2*l0*mw+2*l0*m1)*cos(th0)+2*lt*mt)*z0+(mw+mt+m1+m0)*z**2+(((-2*l2*m1*cos(th0)*sin(thk))-2*l2*m1*sin(th0)*cos(thk))*sin(thw)+(2*l2*m1*cos(th0)*cos(thk)-2*l2*m1*sin(th0)*sin(thk))*cos(thw)+((-2*l1*mw)-2*l1*m1)*sin(th0)*sin(thk)+(2*l1*mw+2*l1*m1)*cos(th0)*cos(thk)+(2*l0*mw+2*l0*m1)*cos(th0)+2*lt*mt)*z-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+lt**2*mt+(l2**2+l1**2+l0**2)*m1
    A[1][2] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk)+(l0*mw+l0*m1)*cos(th0))*z0+(((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk)+(l0*mw+l0*m1)*cos(th0))*z-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A[1][3] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk))*z0+(((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk))*z-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A[1][4] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw))*z0+(((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw))*z-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A[2][0] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0)
    A[2][1] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk)+(l0*mw+l0*m1)*cos(th0))*z0+(((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk)+(l0*mw+l0*m1)*cos(th0))*z-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A[2][2] = (-2*l0*l2*m1*sin(thk)*sin(thw))+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A[2][3] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A[2][4] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A[3][0] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)+((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)
    A[3][1] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk))*z0+(((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk))*z-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A[3][2] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A[3][3] = 2*l1*l2*m1*cos(thw)+l1**2*mw+(l2**2+l1**2)*m1
    A[3][4] = l1*l2*m1*cos(thw)+l2**2*m1
    A[4][0] = (l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thw)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thw)
    A[4][1] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw))*z0+(((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw))*z-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A[4][2] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A[4][3] = l1*l2*m1*cos(thw)+l2**2*m1
    A[4][4] = l2**2*m1

    b = np.zeros((5,1))
    b[0] = (-dthr**2*mw*sin(thr)**2*z0)-dthr**2*mt*sin(thr)**2*z0-dthr**2*m1*sin(thr)**2*z0-dthr**2*m0*sin(thr)**2*z0-dthr**2*mw*cos(thr)**2*z0-dthr**2*mt*cos(thr)**2*z0-dthr**2*m1*cos(thr)**2*z0-dthr**2*m0*cos(thr)**2*z0-dthr**2*mw*sin(thr)**2*z-dthr**2*mt*sin(thr)**2*z-dthr**2*m1*sin(thr)**2*z-dthr**2*m0*sin(thr)**2*z-dthr**2*mw*cos(thr)**2*z-dthr**2*mt*cos(thr)**2*z-dthr**2*m1*cos(thr)**2*z-dthr**2*m0*cos(thr)**2*z+k*z-dthw**2*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-2*dthr*dthw*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-2*dthk*dthw*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-2*dth0*dthw*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-dthr**2*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-2*dthk*dthr*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-2*dth0*dthr*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-dthk**2*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-2*dth0*dthk*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-dth0**2*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-dthw**2*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-2*dthr*dthw*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-2*dthk*dthw*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-2*dth0*dthw*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-dthr**2*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-2*dthk*dthr*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-2*dth0*dthr*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-dthk**2*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-2*dth0*dthk*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-dth0**2*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-dthr**2*l1*mw*sin(thr)*sin(thr+thk+th0)-2*dthk*dthr*l1*mw*sin(thr)*sin(thr+thk+th0)-2*dth0*dthr*l1*mw*sin(thr)*sin(thr+thk+th0)-dthk**2*l1*mw*sin(thr)*sin(thr+thk+th0)-2*dth0*dthk*l1*mw*sin(thr)*sin(thr+thk+th0)-dth0**2*l1*mw*sin(thr)*sin(thr+thk+th0)-dthr**2*l1*m1*sin(thr)*sin(thr+thk+th0)-2*dthk*dthr*l1*m1*sin(thr)*sin(thr+thk+th0)-2*dth0*dthr*l1*m1*sin(thr)*sin(thr+thk+th0)-dthk**2*l1*m1*sin(thr)*sin(thr+thk+th0)-2*dth0*dthk*l1*m1*sin(thr)*sin(thr+thk+th0)-dth0**2*l1*m1*sin(thr)*sin(thr+thk+th0)-dthr**2*l1*mw*cos(thr)*cos(thr+thk+th0)-2*dthk*dthr*l1*mw*cos(thr)*cos(thr+thk+th0)-2*dth0*dthr*l1*mw*cos(thr)*cos(thr+thk+th0)-dthk**2*l1*mw*cos(thr)*cos(thr+thk+th0)-2*dth0*dthk*l1*mw*cos(thr)*cos(thr+thk+th0)-dth0**2*l1*mw*cos(thr)*cos(thr+thk+th0)-dthr**2*l1*m1*cos(thr)*cos(thr+thk+th0)-2*dthk*dthr*l1*m1*cos(thr)*cos(thr+thk+th0)-2*dth0*dthr*l1*m1*cos(thr)*cos(thr+thk+th0)-dthk**2*l1*m1*cos(thr)*cos(thr+thk+th0)-2*dth0*dthk*l1*m1*cos(thr)*cos(thr+thk+th0)-dth0**2*l1*m1*cos(thr)*cos(thr+thk+th0)-dthr**2*l0*mw*sin(thr)*sin(thr+th0)-2*dth0*dthr*l0*mw*sin(thr)*sin(thr+th0)-dth0**2*l0*mw*sin(thr)*sin(thr+th0)-dthr**2*l0*m1*sin(thr)*sin(thr+th0)-2*dth0*dthr*l0*m1*sin(thr)*sin(thr+th0)-dth0**2*l0*m1*sin(thr)*sin(thr+th0)-dthr**2*l0*mw*cos(thr)*cos(thr+th0)-2*dth0*dthr*l0*mw*cos(thr)*cos(thr+th0)-dth0**2*l0*mw*cos(thr)*cos(thr+th0)-dthr**2*l0*m1*cos(thr)*cos(thr+th0)-2*dth0*dthr*l0*m1*cos(thr)*cos(thr+th0)-dth0**2*l0*m1*cos(thr)*cos(thr+th0)-dthr**2*lt*mt*sin(thr)**2-dthr**2*lt*mt*cos(thr)**2+g*mw*cos(thr)+g*mt*cos(thr)+g*m1*cos(thr)+g*m0*cos(thr)
    b[1] = (-dthw**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0)-2*dthr*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dthk*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dth0*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dthk*dthr*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dth0*dthr*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dthk**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dth0*dthk*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dth0**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0+dthw**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dthr*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dthk*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dth0*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dthk*dthr*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dth0*dthr*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dthk**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dth0*dthk*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dth0**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0-2*dthk*dthr*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-2*dth0*dthr*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-dthk**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-2*dth0*dthk*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-dth0**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-2*dthk*dthr*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-2*dth0*dthr*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-dthk**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-2*dth0*dthk*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-dth0**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z0+2*dthk*dthr*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+2*dth0*dthr*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+dthk**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+2*dth0*dthk*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+dth0**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+2*dthk*dthr*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+2*dth0*dthr*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+dthk**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+2*dth0*dthk*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+dth0**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z0-2*dth0*dthr*l0*mw*cos(thr)*sin(thr+th0)*z0-dth0**2*l0*mw*cos(thr)*sin(thr+th0)*z0-2*dth0*dthr*l0*m1*cos(thr)*sin(thr+th0)*z0-dth0**2*l0*m1*cos(thr)*sin(thr+th0)*z0+2*dth0*dthr*l0*mw*sin(thr)*cos(thr+th0)*z0+dth0**2*l0*mw*sin(thr)*cos(thr+th0)*z0+2*dth0*dthr*l0*m1*sin(thr)*cos(thr+th0)*z0+dth0**2*l0*m1*sin(thr)*cos(thr+th0)*z0+2*dthr*dz*mw*sin(thr)**2*z0+2*dthr*dz*mt*sin(thr)**2*z0+2*dthr*dz*m1*sin(thr)**2*z0+2*dthr*dz*m0*sin(thr)**2*z0-g*mw*sin(thr)*z0-g*mt*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0+2*dthr*dz*mw*cos(thr)**2*z0+2*dthr*dz*mt*cos(thr)**2*z0+2*dthr*dz*m1*cos(thr)**2*z0+2*dthr*dz*m0*cos(thr)**2*z0-dthw**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-2*dthr*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-2*dthk*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-2*dth0*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-2*dthk*dthr*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-2*dth0*dthr*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-dthk**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-2*dth0*dthk*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-dth0**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z+dthw**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+2*dthr*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+2*dthk*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+2*dth0*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+2*dthk*dthr*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+2*dth0*dthr*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+dthk**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+2*dth0*dthk*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+dth0**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z-2*dthk*dthr*l1*mw*cos(thr)*sin(thr+thk+th0)*z-2*dth0*dthr*l1*mw*cos(thr)*sin(thr+thk+th0)*z-dthk**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z-2*dth0*dthk*l1*mw*cos(thr)*sin(thr+thk+th0)*z-dth0**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z-2*dthk*dthr*l1*m1*cos(thr)*sin(thr+thk+th0)*z-2*dth0*dthr*l1*m1*cos(thr)*sin(thr+thk+th0)*z-dthk**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z-2*dth0*dthk*l1*m1*cos(thr)*sin(thr+thk+th0)*z-dth0**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z+2*dthk*dthr*l1*mw*sin(thr)*cos(thr+thk+th0)*z+2*dth0*dthr*l1*mw*sin(thr)*cos(thr+thk+th0)*z+dthk**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z+2*dth0*dthk*l1*mw*sin(thr)*cos(thr+thk+th0)*z+dth0**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z+2*dthk*dthr*l1*m1*sin(thr)*cos(thr+thk+th0)*z+2*dth0*dthr*l1*m1*sin(thr)*cos(thr+thk+th0)*z+dthk**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z+2*dth0*dthk*l1*m1*sin(thr)*cos(thr+thk+th0)*z+dth0**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z-2*dth0*dthr*l0*mw*cos(thr)*sin(thr+th0)*z-dth0**2*l0*mw*cos(thr)*sin(thr+th0)*z-2*dth0*dthr*l0*m1*cos(thr)*sin(thr+th0)*z-dth0**2*l0*m1*cos(thr)*sin(thr+th0)*z+2*dth0*dthr*l0*mw*sin(thr)*cos(thr+th0)*z+dth0**2*l0*mw*sin(thr)*cos(thr+th0)*z+2*dth0*dthr*l0*m1*sin(thr)*cos(thr+th0)*z+dth0**2*l0*m1*sin(thr)*cos(thr+th0)*z+2*dthr*dz*mw*sin(thr)**2*z+2*dthr*dz*mt*sin(thr)**2*z+2*dthr*dz*m1*sin(thr)**2*z+2*dthr*dz*m0*sin(thr)**2*z-g*mw*sin(thr)*z-g*mt*sin(thr)*z-g*m1*sin(thr)*z-g*m0*sin(thr)*z+2*dthr*dz*mw*cos(thr)**2*z+2*dthr*dz*mt*cos(thr)**2*z+2*dthr*dz*m1*cos(thr)**2*z+2*dthr*dz*m0*cos(thr)**2*z-dthw**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-dthw**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthk*dthr*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-dthk**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dth0*dthk*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+2*dthr*dz*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-g*l2*m1*sin(thw+thr+thk+th0)+dthw**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+dthw**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthk*dthr*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+dthk**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dth0*dthk*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthr*dz*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-2*dthk*dthr*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-dthk**2*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-2*dth0*dthk*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-2*dthk*dthr*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-dthk**2*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-2*dth0*dthk*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)+2*dthr*dz*l1*mw*sin(thr)*sin(thr+thk+th0)+2*dthr*dz*l1*m1*sin(thr)*sin(thr+thk+th0)-g*l1*mw*sin(thr+thk+th0)-g*l1*m1*sin(thr+thk+th0)+2*dthk*dthr*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+dthk**2*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+2*dth0*dthk*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+2*dthk*dthr*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+dthk**2*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+2*dth0*dthk*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+2*dthr*dz*l1*mw*cos(thr)*cos(thr+thk+th0)+2*dthr*dz*l1*m1*cos(thr)*cos(thr+thk+th0)+2*dthr*dz*l0*mw*sin(thr)*sin(thr+th0)+2*dthr*dz*l0*m1*sin(thr)*sin(thr+th0)-g*l0*mw*sin(thr+th0)-g*l0*m1*sin(thr+th0)+2*dthr*dz*l0*mw*cos(thr)*cos(thr+th0)+2*dthr*dz*l0*m1*cos(thr)*cos(thr+th0)+2*dthr*dz*lt*mt*sin(thr)**2-g*lt*mt*sin(thr)+2*dthr*dz*lt*mt*cos(thr)**2
    b[2] = dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dthr**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z0+dthr**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-dthr**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z0-dthr**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+dthr**2*l0*mw*cos(thr)*sin(thr+th0)*z0+dthr**2*l0*m1*cos(thr)*sin(thr+th0)*z0-dthr**2*l0*mw*sin(thr)*cos(thr+th0)*z0-dthr**2*l0*m1*sin(thr)*cos(thr+th0)*z0+dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+dthr**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z+dthr**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z-dthr**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z-dthr**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z+dthr**2*l0*mw*cos(thr)*sin(thr+th0)*z+dthr**2*l0*m1*cos(thr)*sin(thr+th0)*z-dthr**2*l0*mw*sin(thr)*cos(thr+th0)*z-dthr**2*l0*m1*sin(thr)*cos(thr+th0)*z-dthw**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-dthw**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthk*dthr*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-dthk**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dth0*dthk*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+2*dthr*dz*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-g*l2*m1*sin(thw+thr+thk+th0)+dthw**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+dthw**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthk*dthr*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+dthk**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dth0*dthk*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthr*dz*l2*m1*cos(thr)*cos(thw+thr+thk+th0)-2*dthk*dthr*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-dthk**2*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-2*dth0*dthk*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-2*dthk*dthr*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-dthk**2*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-2*dth0*dthk*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)+2*dthr*dz*l1*mw*sin(thr)*sin(thr+thk+th0)+2*dthr*dz*l1*m1*sin(thr)*sin(thr+thk+th0)-g*l1*mw*sin(thr+thk+th0)-g*l1*m1*sin(thr+thk+th0)+2*dthk*dthr*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+dthk**2*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+2*dth0*dthk*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+2*dthk*dthr*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+dthk**2*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+2*dth0*dthk*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+2*dthr*dz*l1*mw*cos(thr)*cos(thr+thk+th0)+2*dthr*dz*l1*m1*cos(thr)*cos(thr+thk+th0)+2*dthr*dz*l0*mw*sin(thr)*sin(thr+th0)+2*dthr*dz*l0*m1*sin(thr)*sin(thr+th0)-g*l0*mw*sin(thr+th0)-g*l0*m1*sin(thr+th0)+2*dthr*dz*l0*mw*cos(thr)*cos(thr+th0)+2*dthr*dz*l0*m1*cos(thr)*cos(thr+th0)
    b[3] = dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dthr**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z0+dthr**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-dthr**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z0-dthr**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+dthr**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z+dthr**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z-dthr**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z-dthr**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z-dthw**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+dthr**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+2*dth0*dthr*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+dth0**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+2*dthr*dz*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-g*l2*m1*sin(thw+thr+thk+th0)+dthw**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-dthr**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-2*dth0*dthr*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-dth0**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthr*dz*l2*m1*cos(thr)*cos(thw+thr+thk+th0)+dthr**2*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)+2*dth0*dthr*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)+dth0**2*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)+dthr**2*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)+2*dth0*dthr*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)+dth0**2*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)+2*dthr*dz*l1*mw*sin(thr)*sin(thr+thk+th0)+2*dthr*dz*l1*m1*sin(thr)*sin(thr+thk+th0)-g*l1*mw*sin(thr+thk+th0)-g*l1*m1*sin(thr+thk+th0)-dthr**2*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)-2*dth0*dthr*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)-dth0**2*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)-dthr**2*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)-2*dth0*dthr*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)-dth0**2*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+2*dthr*dz*l1*mw*cos(thr)*cos(thr+thk+th0)+2*dthr*dz*l1*m1*cos(thr)*cos(thr+thk+th0)
    b[4] = dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z+dthr**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+2*dthk*dthr*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+2*dth0*dthr*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+dthk**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+2*dth0*dthk*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+dth0**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+dthr**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+2*dth0*dthr*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+dth0**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+2*dthr*dz*l2*m1*sin(thr)*sin(thw+thr+thk+th0)-g*l2*m1*sin(thw+thr+thk+th0)-dthr**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-2*dthk*dthr*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-2*dth0*dthr*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-dthk**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-2*dth0*dthk*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-dth0**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-dthr**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-2*dth0*dthr*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-dth0**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthr*dz*l2*m1*cos(thr)*cos(thw+thr+thk+th0)
    return A, b, extf


def airAb(s, u):
    xr   = s[IDX_xr]
    yr   = s[IDX_yr]
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    thk  = s[IDX_thk]
    thw  = s[IDX_thw]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dthk = s[IDX_dthk]
    dthw = s[IDX_dthw]
    tau0 = u[0]
    tau1 = u[1]
    tau2 = u[2]
    extf = np.array([0, 0, 0, tau0, tau1, tau2]).reshape(6,1)

    A = np.zeros((6,6))
    A[0][0] = mw+mt+m1+m0
    A[0][1] = 0
    A[0][2] = ((-mw)-mt-m1-m0)*cos(thr)*z0+((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk)+(l0*mw+l0*m1)*sin(th0))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0)-lt*mt)*cos(thr)
    A[0][3] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk)+(l0*mw+l0*m1)*sin(th0))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0))*cos(thr)
    A[0][4] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk))*cos(thr)
    A[0][5] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)
    A[1][0] = 0
    A[1][1] = mw+mt+m1+m0
    A[1][2] = ((-mw)-mt-m1-m0)*sin(thr)*z0+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0)-lt*mt)*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0))*cos(thr)
    A[1][3] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0))*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0))*cos(thr)
    A[1][4] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk))*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk))*cos(thr)
    A[1][5] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)
    A[2][0] = ((-mw)-mt-m1-m0)*cos(thr)*z0+((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk)+(l0*mw+l0*m1)*sin(th0))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0)-lt*mt)*cos(thr)
    A[2][1] = ((-mw)-mt-m1-m0)*sin(thr)*z0+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0)-lt*mt)*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0))*cos(thr)
    A[2][2] = (mw+mt+m1+m0)*z0**2+(((-2*l2*m1*cos(th0)*sin(thk))-2*l2*m1*sin(th0)*cos(thk))*sin(thw)+(2*l2*m1*cos(th0)*cos(thk)-2*l2*m1*sin(th0)*sin(thk))*cos(thw)+((-2*l1*mw)-2*l1*m1)*sin(th0)*sin(thk)+(2*l1*mw+2*l1*m1)*cos(th0)*cos(thk)+(2*l0*mw+2*l0*m1)*cos(th0)+2*lt*mt)*z0-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+lt**2*mt+(l2**2+l1**2+l0**2)*m1
    A[2][3] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk)+(l0*mw+l0*m1)*cos(th0))*z0-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A[2][4] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk))*z0-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A[2][5] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw))*z0-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A[3][0] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk)+(l0*mw+l0*m1)*sin(th0))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0))*cos(thr)
    A[3][1] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk)+((-l0*mw)-l0*m1)*cos(th0))*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk)+((-l0*mw)-l0*m1)*sin(th0))*cos(thr)
    A[3][2] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk)+(l0*mw+l0*m1)*cos(th0))*z0-2*l0*l2*m1*sin(thk)*sin(thw)+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A[3][3] = (-2*l0*l2*m1*sin(thk)*sin(thw))+(2*l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(2*l0*l1*mw+2*l0*l1*m1)*cos(thk)+(l1**2+l0**2)*mw+(l2**2+l1**2+l0**2)*m1
    A[3][4] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A[3][5] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A[4][0] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*cos(th0)*sin(thk)+(l1*mw+l1*m1)*sin(th0)*cos(thk))*sin(thr)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk))*cos(thr)
    A[4][1] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)+((l1*mw+l1*m1)*sin(th0)*sin(thk)+((-l1*mw)-l1*m1)*cos(th0)*cos(thk))*sin(thr)+(((-l1*mw)-l1*m1)*cos(th0)*sin(thk)+((-l1*mw)-l1*m1)*sin(th0)*cos(thk))*cos(thr)
    A[4][2] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw)+((-l1*mw)-l1*m1)*sin(th0)*sin(thk)+(l1*mw+l1*m1)*cos(th0)*cos(thk))*z0-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A[4][3] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+2*l1*l2*m1)*cos(thw)+(l0*l1*mw+l0*l1*m1)*cos(thk)+l1**2*mw+(l2**2+l1**2)*m1
    A[4][4] = 2*l1*l2*m1*cos(thw)+l1**2*mw+(l2**2+l1**2)*m1
    A[4][5] = l1*l2*m1*cos(thw)+l2**2*m1
    A[5][0] = ((l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*sin(thr)+(l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*cos(thw)
    A[5][1] = ((l2*m1*cos(th0)*sin(thk)+l2*m1*sin(th0)*cos(thk))*sin(thr)+(l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*cos(thr))*sin(thw)+((l2*m1*sin(th0)*sin(thk)-l2*m1*cos(th0)*cos(thk))*sin(thr)+((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*cos(thr))*cos(thw)
    A[5][2] = (((-l2*m1*cos(th0)*sin(thk))-l2*m1*sin(th0)*cos(thk))*sin(thw)+(l2*m1*cos(th0)*cos(thk)-l2*m1*sin(th0)*sin(thk))*cos(thw))*z0-l0*l2*m1*sin(thk)*sin(thw)+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A[5][3] = (-l0*l2*m1*sin(thk)*sin(thw))+(l0*l2*m1*cos(thk)+l1*l2*m1)*cos(thw)+l2**2*m1
    A[5][4] = l1*l2*m1*cos(thw)+l2**2*m1
    A[5][5] = l2**2*m1

    b = np.zeros((6,1))
    b[0] = dthr**2*mw*sin(thr)*z0+dthr**2*mt*sin(thr)*z0+dthr**2*m1*sin(thr)*z0+dthr**2*m0*sin(thr)*z0+dthw**2*l2*m1*sin(thw+thr+thk+th0)+2*dthr*dthw*l2*m1*sin(thw+thr+thk+th0)+2*dthk*dthw*l2*m1*sin(thw+thr+thk+th0)+2*dth0*dthw*l2*m1*sin(thw+thr+thk+th0)+dthr**2*l2*m1*sin(thw+thr+thk+th0)+2*dthk*dthr*l2*m1*sin(thw+thr+thk+th0)+2*dth0*dthr*l2*m1*sin(thw+thr+thk+th0)+dthk**2*l2*m1*sin(thw+thr+thk+th0)+2*dth0*dthk*l2*m1*sin(thw+thr+thk+th0)+dth0**2*l2*m1*sin(thw+thr+thk+th0)+dthr**2*l1*mw*sin(thr+thk+th0)+2*dthk*dthr*l1*mw*sin(thr+thk+th0)+2*dth0*dthr*l1*mw*sin(thr+thk+th0)+dthk**2*l1*mw*sin(thr+thk+th0)+2*dth0*dthk*l1*mw*sin(thr+thk+th0)+dth0**2*l1*mw*sin(thr+thk+th0)+dthr**2*l1*m1*sin(thr+thk+th0)+2*dthk*dthr*l1*m1*sin(thr+thk+th0)+2*dth0*dthr*l1*m1*sin(thr+thk+th0)+dthk**2*l1*m1*sin(thr+thk+th0)+2*dth0*dthk*l1*m1*sin(thr+thk+th0)+dth0**2*l1*m1*sin(thr+thk+th0)+dthr**2*l0*mw*sin(thr+th0)+2*dth0*dthr*l0*mw*sin(thr+th0)+dth0**2*l0*mw*sin(thr+th0)+dthr**2*l0*m1*sin(thr+th0)+2*dth0*dthr*l0*m1*sin(thr+th0)+dth0**2*l0*m1*sin(thr+th0)+dthr**2*lt*mt*sin(thr)
    b[1] = (-dthr**2*mw*cos(thr)*z0)-dthr**2*mt*cos(thr)*z0-dthr**2*m1*cos(thr)*z0-dthr**2*m0*cos(thr)*z0-dthw**2*l2*m1*cos(thw+thr+thk+th0)-2*dthr*dthw*l2*m1*cos(thw+thr+thk+th0)-2*dthk*dthw*l2*m1*cos(thw+thr+thk+th0)-2*dth0*dthw*l2*m1*cos(thw+thr+thk+th0)-dthr**2*l2*m1*cos(thw+thr+thk+th0)-2*dthk*dthr*l2*m1*cos(thw+thr+thk+th0)-2*dth0*dthr*l2*m1*cos(thw+thr+thk+th0)-dthk**2*l2*m1*cos(thw+thr+thk+th0)-2*dth0*dthk*l2*m1*cos(thw+thr+thk+th0)-dth0**2*l2*m1*cos(thw+thr+thk+th0)-dthr**2*l1*mw*cos(thr+thk+th0)-2*dthk*dthr*l1*mw*cos(thr+thk+th0)-2*dth0*dthr*l1*mw*cos(thr+thk+th0)-dthk**2*l1*mw*cos(thr+thk+th0)-2*dth0*dthk*l1*mw*cos(thr+thk+th0)-dth0**2*l1*mw*cos(thr+thk+th0)-dthr**2*l1*m1*cos(thr+thk+th0)-2*dthk*dthr*l1*m1*cos(thr+thk+th0)-2*dth0*dthr*l1*m1*cos(thr+thk+th0)-dthk**2*l1*m1*cos(thr+thk+th0)-2*dth0*dthk*l1*m1*cos(thr+thk+th0)-dth0**2*l1*m1*cos(thr+thk+th0)-dthr**2*l0*mw*cos(thr+th0)-2*dth0*dthr*l0*mw*cos(thr+th0)-dth0**2*l0*mw*cos(thr+th0)-dthr**2*l0*m1*cos(thr+th0)-2*dth0*dthr*l0*m1*cos(thr+th0)-dth0**2*l0*m1*cos(thr+th0)-dthr**2*lt*mt*cos(thr)+g*mw+g*mt+g*m1+g*m0
    b[2] = (-dthw**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0)-2*dthr*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dthk*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dth0*dthw*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dthk*dthr*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dth0*dthr*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dthk**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-2*dth0*dthk*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dth0**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0+dthw**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dthr*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dthk*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dth0*dthw*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dthk*dthr*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dth0*dthr*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dthk**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+2*dth0*dthk*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dth0**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0-2*dthk*dthr*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-2*dth0*dthr*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-dthk**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-2*dth0*dthk*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-dth0**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z0-2*dthk*dthr*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-2*dth0*dthr*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-dthk**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-2*dth0*dthk*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-dth0**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z0+2*dthk*dthr*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+2*dth0*dthr*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+dthk**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+2*dth0*dthk*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+dth0**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z0+2*dthk*dthr*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+2*dth0*dthr*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+dthk**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+2*dth0*dthk*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+dth0**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z0-2*dth0*dthr*l0*mw*cos(thr)*sin(thr+th0)*z0-dth0**2*l0*mw*cos(thr)*sin(thr+th0)*z0-2*dth0*dthr*l0*m1*cos(thr)*sin(thr+th0)*z0-dth0**2*l0*m1*cos(thr)*sin(thr+th0)*z0+2*dth0*dthr*l0*mw*sin(thr)*cos(thr+th0)*z0+dth0**2*l0*mw*sin(thr)*cos(thr+th0)*z0+2*dth0*dthr*l0*m1*sin(thr)*cos(thr+th0)*z0+dth0**2*l0*m1*sin(thr)*cos(thr+th0)*z0-g*mw*sin(thr)*z0-g*mt*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0-dthw**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-dthw**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthk*dthr*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-dthk**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dth0*dthk*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-g*l2*m1*sin(thw+thr+thk+th0)+dthw**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+dthw**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthk*dthr*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+dthk**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dth0*dthk*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-2*dthk*dthr*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-dthk**2*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-2*dth0*dthk*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-2*dthk*dthr*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-dthk**2*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-2*dth0*dthk*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-g*l1*mw*sin(thr+thk+th0)-g*l1*m1*sin(thr+thk+th0)+2*dthk*dthr*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+dthk**2*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+2*dth0*dthk*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+2*dthk*dthr*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+dthk**2*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+2*dth0*dthk*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)-g*l0*mw*sin(thr+th0)-g*l0*m1*sin(thr+th0)-g*lt*mt*sin(thr)
    b[3] = dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dthr**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z0+dthr**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-dthr**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z0-dthr**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z0+dthr**2*l0*mw*cos(thr)*sin(thr+th0)*z0+dthr**2*l0*m1*cos(thr)*sin(thr+th0)*z0-dthr**2*l0*mw*sin(thr)*cos(thr+th0)*z0-dthr**2*l0*m1*sin(thr)*cos(thr+th0)*z0-dthw**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-dthw**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dthk*dthr*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-dthk**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-2*dth0*dthk*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-g*l2*m1*sin(thw+thr+thk+th0)+dthw**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+dthw**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dthk*dthr*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+dthk**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+2*dth0*dthk*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-2*dthk*dthr*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-dthk**2*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-2*dth0*dthk*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)-2*dthk*dthr*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-dthk**2*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-2*dth0*dthk*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-g*l1*mw*sin(thr+thk+th0)-g*l1*m1*sin(thr+thk+th0)+2*dthk*dthr*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+dthk**2*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+2*dth0*dthk*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)+2*dthk*dthr*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+dthk**2*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)+2*dth0*dthk*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)-g*l0*mw*sin(thr+th0)-g*l0*m1*sin(thr+th0)
    b[4] = dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dthr**2*l1*mw*cos(thr)*sin(thr+thk+th0)*z0+dthr**2*l1*m1*cos(thr)*sin(thr+thk+th0)*z0-dthr**2*l1*mw*sin(thr)*cos(thr+thk+th0)*z0-dthr**2*l1*m1*sin(thr)*cos(thr+thk+th0)*z0-dthw**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthr*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dthk*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)-2*dth0*dthw*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+dthr**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+2*dth0*dthr*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+dth0**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-g*l2*m1*sin(thw+thr+thk+th0)+dthw**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthr*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dthk*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)+2*dth0*dthw*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-dthr**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-2*dth0*dthr*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-dth0**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)+dthr**2*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)+2*dth0*dthr*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)+dth0**2*l0*l1*mw*cos(thr+th0)*sin(thr+thk+th0)+dthr**2*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)+2*dth0*dthr*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)+dth0**2*l0*l1*m1*cos(thr+th0)*sin(thr+thk+th0)-g*l1*mw*sin(thr+thk+th0)-g*l1*m1*sin(thr+thk+th0)-dthr**2*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)-2*dth0*dthr*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)-dth0**2*l0*l1*mw*sin(thr+th0)*cos(thr+thk+th0)-dthr**2*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)-2*dth0*dthr*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)-dth0**2*l0*l1*m1*sin(thr+th0)*cos(thr+thk+th0)
    b[5] = dthr**2*l2*m1*cos(thr)*sin(thw+thr+thk+th0)*z0-dthr**2*l2*m1*sin(thr)*cos(thw+thr+thk+th0)*z0+dthr**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+2*dthk*dthr*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+2*dth0*dthr*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+dthk**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+2*dth0*dthk*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+dth0**2*l1*l2*m1*cos(thr+thk+th0)*sin(thw+thr+thk+th0)+dthr**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+2*dth0*dthr*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)+dth0**2*l0*l2*m1*cos(thr+th0)*sin(thw+thr+thk+th0)-g*l2*m1*sin(thw+thr+thk+th0)-dthr**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-2*dthk*dthr*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-2*dth0*dthr*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-dthk**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-2*dth0*dthk*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-dth0**2*l1*l2*m1*sin(thr+thk+th0)*cos(thw+thr+thk+th0)-dthr**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-2*dth0*dthr*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)-dth0**2*l0*l2*m1*sin(thr+th0)*cos(thw+thr+thk+th0)

    return A, b, extf

def f_ground(s, u):
    A, b, extf = groundAb(s, u)
    if np.linalg.matrix_rank(A) < 5:
        print(np.linalg.det(A))
        print(A)
        raise Exception("rank")
    y = np.linalg.solve(A, extf-b).reshape(5)
    f = np.zeros(IDX_MAX)
    f[IDX_xr]   = 0
    f[IDX_yr]   = 0
    f[IDX_z]   = y[0]
    f[IDX_thr] = y[1]
    f[IDX_th0] = y[2]
    f[IDX_thk] = y[3]
    f[IDX_thw] = y[4]
    return f

def f_air(s, u):
    A, b, extf = airAb(s, u)
    if np.linalg.matrix_rank(A) < 6:
        raise Exception("rank")
    y = np.linalg.solve(A, extf-b).reshape(6)
    f = np.zeros(IDX_MAX)
    f[IDX_xr]   = y[0]
    f[IDX_yr]   = y[1]
    f[IDX_thr] = y[2]
    f[IDX_z]   = 0
    f[IDX_th0] = y[3]
    f[IDX_thk] = y[4]
    f[IDX_thw] = y[5]
    return f

def step(t, s, u, dt):
    try:
        u = torq_limit(s, u)
        ret = np.zeros(2*IDX_MAX)
        if ground(s):
            impulse = f_ground(s, u) * dt
            ret[IDX_MAX:] = s[IDX_MAX:] + impulse
            ret[:IDX_MAX] = s[:IDX_MAX] + ret[IDX_MAX:] * dt
            if ret[IDX_z] >= 0:
                mode = "jump"
                ret = jumpup(ret)
            else:
                mode = "ground"
        else:
            impulse = f_air(s, u) * dt
            ret[IDX_MAX:] = s[IDX_MAX:] + impulse
            ret[:IDX_MAX] = s[:IDX_MAX] + ret[IDX_MAX:] * dt
            if ret[IDX_yr] <= 0:
                mode = "land"
                ret = land(ret)
            else:
                mode ="air"
        return True, 1 if ground(ret) else 0, t+dt, ret
    except:
        return False, 0, t, s

