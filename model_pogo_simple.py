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
lt = 0.93
mr = 0
m0 = 20
m1 = 50
mt = 7

M=np.array([mr, m0, m1, mt])
IDX_pr=0
IDX_p0=1
IDX_p1=2
IDX_pt=3

g  = 0
#g  = 9.8
k  = 15000 # mgh = 1/2 k x**2 -> T=2*pi sqrt(m/k)
c = 0
#c = 10

# ccw is positive
ref_min_th0 = np.deg2rad(-5)
ref_max_th0 = np.deg2rad(20)
ref_min_thr = np.deg2rad(-20)
ref_max_thr = np.deg2rad(20)
ref_min_r = 0.6
ref_max_r = 1
REF_MIN = np.array([ref_min_thr, ref_min_th0, ref_min_r])
REF_MAX = np.array([ref_max_thr, ref_max_th0, ref_max_r])
DIM_U = REF_MIN.shape
DEFAULT_U = np.zeros(DIM_U)
limit_min_thr = np.deg2rad(-180)
limit_max_thr = np.deg2rad(180)
limit_min_th0 = np.deg2rad(-45)
limit_max_th0 = np.deg2rad(45)
limit_min_r = 0.4
limit_max_r = 1.2

MAX_ROT_SPEED=100
MAX_SPEED=100
#MAX_TORQUEK=3000 # arm
#MAX_TORQUE0=8000 # arm
MAX_TORQUER=(lt+z0)*800
MAX_TORQUE0=700 # knee(400) + west(300)
MAX_FORCER=1300 # [N]
MAX_FORCE = np.array([MAX_TORQUER, MAX_TORQUE0, MAX_FORCER])

inf = float('inf')
#Kp = np.array([4000, 13000])
#Kp = 20*np.array([400, 800])
Kp = 10*np.array([400, 800, 800])
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
IDX_r    = 5
IDX_MAX = 6
IDX_dxr   = IDX_MAX+IDX_xr 
IDX_dyr   = IDX_MAX+IDX_yr 
IDX_dthr  = IDX_MAX+IDX_thr
IDX_dz    = IDX_MAX+IDX_z  
IDX_dth0  = IDX_MAX+IDX_th0
IDX_dr    = IDX_MAX+IDX_r

def reset_state(d = {}):
    s = np.zeros(2*IDX_MAX)
    s[IDX_xr ] = d.get('prx', 0)
    s[IDX_yr ] = d.get('pry', 0.4)
    s[IDX_thr] = d.get('thr', 0)
    s[IDX_z  ] = d.get('z', 0)
    s[IDX_th0] = d.get('th0', np.deg2rad(-10))
    s[IDX_r]   = d.get('r', 0.8)
    s[IDX_dxr ] = d.get('dprx', 0)
    s[IDX_dyr ] = d.get('dpry', 0)
    s[IDX_dthr] = d.get('dthr', 0)
    s[IDX_dz  ] = d.get('dz'  , 0)  
    s[IDX_dth0] = d.get('dthr', 0)
    s[IDX_dr] = d.get('dr', 0)

    return s


def print_state(s, titlePrefix="", fieldPrefix=""):
    d = calc_joint_property(s)
    for i in d:
        print(f"{titlePrefix} {i:}: {d[i]:.2f}")

def calc_joint_property(s, d = {}):
    d['prx'] = s[IDX_xr ]
    d['pry'] = s[IDX_yr ]
    d['thr'] = s[IDX_thr]
    d['z'] = s[IDX_z  ]
    d['th0'] = s[IDX_th0]
    d['r'] = s[IDX_r]
    d['dprx'] = s[IDX_dxr ]
    d['dpry'] = s[IDX_dyr ]
    d['dthr'] = s[IDX_dthr]
    d['dz'] = s[IDX_dz  ]
    d['dth0'] = s[IDX_dth0]
    d['dr'] = s[IDX_dr]
    return d

def max_u():
    return MAX_FORCE

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
    r = s[IDX_r]

    dir_thr = np.array([-np.sin(thr), np.cos(thr)])
    dir_thr0 = np.array([-np.sin(thr+th0), np.cos(thr+th0)])

    p0 = pr + (z0 + z) * dir_thr
    p1 = p0 + r * dir_thr0
    pt = p0 + lt * dir_thr

    return pr, p0, p1, pt


def node_vel(s):
    thr = s[IDX_thr]
    z   = s[IDX_z  ]
    th0 = s[IDX_th0]
    r = s[IDX_r]
    dir_thr = np.array([-np.sin(thr), np.cos(thr)])
    dir_thr0 = np.array([-np.sin(thr+th0), np.cos(thr+th0)])
    dir_dthr = np.array([-np.cos(thr), -np.sin(thr)])
    dir_dthr0 = np.array([-np.cos(thr+th0), -np.sin(thr+th0)])
    vr = np.array([s[IDX_dxr], s[IDX_dyr]])
    dthr = s[IDX_dthr]
    dz   = s[IDX_dz  ]
    dth0 = s[IDX_dth0]
    dr = s[IDX_dr]

    v0 = vr + dz*dir_thr + dthr*(z0+z)*dir_dthr
    v1 = v0 + dr*dir_thr0 + (dthr+dth0)*r*dir_dthr0
    vt = v0 + dthr*lt*dir_dthr
    return vr, v0, v1, vt

OBS_MIN = np.array([0, 0, limit_min_thr, -max_z, limit_min_th0, limit_min_r,
                   -MAX_SPEED, -MAX_SPEED, -MAX_ROT_SPEED, -MAX_SPEED, -MAX_ROT_SPEED, -MAX_SPEED])
OBS_MAX = np.array([1, 5, limit_max_thr,      0, limit_max_th0, limit_max_r,
                    MAX_SPEED , MAX_SPEED , MAX_ROT_SPEED, MAX_SPEED, MAX_ROT_SPEED, MAX_SPEED])

def obs(s):
    o = s.copy()
    o[0] = 1 if ground(s) else 0
    return o

def reward(s):
    pcog = cog(s)
    r_y = (energyU(s) + energyTy(s))/3000
    r_thr = -abs(s[IDX_thr])*2/np.pi
    r_cogx = -abs(pcog[0]-s[IDX_xr])
    r = max(np.exp(r_y + r_thr + r_cogx), 0.1)
    if r > 1000:
        print("TOO MUCH REWARD")
        print(r_y, r_thr, r_cogx)
        print(s)
        sys.exit(0)
    return r

def init_ref(s):
    return np.array([s[IDX_thr], s[IDX_th0], s[IDX_r]])

def check_invariant(s):
    ps = list(node_pos(s))
    for i in range(1,len(ps)):
        if ps[i][1] <= 0.001:
            reason = f"GAME OVER @ p{i}={ps[i]:}"
            return False, reason
    if s[IDX_th0] < limit_min_th0 or s[IDX_th0] > limit_max_th0:
            reason = f"GAME OVER @ range error th0={np.rad2deg(s[IDX_th0]):}"
            return False, reason
    if s[IDX_r] < limit_min_r or s[IDX_r] > limit_max_r:
            reason = f"GAME OVER @ range error r={s[IDX_thw]:}"
            return False, reason

    if ground(s) and abs(s[IDX_thr]) > np.deg2rad(45):
        reason = f"GAME OVER @ thr is too big on ground"
        return False, reason

    if energy(s) > 5000:
        reason = f"GAME OVER @ energy is too big"
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
    dob  = np.array([s[IDX_dthr], s[IDX_dth0], s[IDX_dr]])
    ob = np.array([s[IDX_thr], s[IDX_th0], s[IDX_r]])
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
    thr  = s[IDX_r]

    A21 = np.zeros((4,2))
    A[0][0] = ((-mt)-m1-m0)*sin(thr)
    A[0][1] = (mt+m1+m0)*cos(thr)
    A[1][0] = ((-mt)-m1-m0)*cos(thr)*z0+m1*r*sin(th0)*sin(thr)+((-m1*r*cos(th0))-lt*mt)*cos(thr)
    A[1][1] = ((-mt)-m1-m0)*sin(thr)*z0+((-m1*r*cos(th0))-lt*mt)*sin(thr)-m1*r*sin(th0)*cos(thr)
    A[2][0] = m1*r*sin(th0)*sin(thr)-m1*r*cos(th0)*cos(thr)
    A[2][1] = (-m1*r*cos(th0)*sin(thr))-m1*r*sin(th0)*cos(thr)
    A[3][0] = (-m1*cos(th0)*sin(thr))-m1*sin(th0)*cos(thr)
    A[3][1] = m1*cos(th0)*cos(thr)-m1*sin(th0)*sin(thr)

    A22 = np.zeros((4,4))
    A[0][0] = mt+m1+m0
    A[0][1] = -m1*r*sin(th0)
    A[0][2] = -m1*r*sin(th0)
    A[0][3] = m1*cos(th0)
    A[1][0] = -m1*r*sin(th0)
    A[1][1] = (mt+m1+m0)*z0**2+(2*m1*r*cos(th0)+2*lt*mt)*z0+m1*r**2+lt**2*mt
    A[1][2] = m1*r*cos(th0)*z0+m1*r**2
    A[1][3] = m1*sin(th0)*z0
    A[2][0] = -m1*r*sin(th0)
    A[2][1] = m1*r*cos(th0)*z0+m1*r**2
    A[2][2] = m1*r**2
    A[2][3] = 0
    A[3][0] = m1*cos(th0)
    A[3][1] = m1*sin(th0)*z0
    A[3][2] = 0
    A[3][3] = m1

    if np.linalg.matrix_rank(A22) < 4:
        print("collision", A22)
        print("collision", np.linalg.matrix_rank(A22))
        raise Exception("rank")

    #solve  A22 y(z,thr,th0,thk) = A21 (dx,dy)
    y = np.linalg.solve(A22, A21 @ s[IDX_dxr:IDX_dyr+1]).reshape(4)

    d = np.zeros(IDX_MAX)
    d[IDX_xr] = 0
    d[IDX_yr] = 0
    d[IDX_z]   = y[0]
    d[IDX_thr] = y[1]
    d[IDX_th0] = y[2]
    d[IDX_r]   = y[3]
    return d

def land(s):
    ret = s.copy()
    d = impulse_collision(s)
    ret[IDX_yr] = 0
    ret[IDX_z] = 0
    ret[IDX_MAX:] = d
    return ret

def groundAb(s, u):
    z    = s[IDX_z]
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    r  = s[IDX_r]
    dz   = s[IDX_dz  ]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dr = s[IDX_dr]

    fz   = 0
    taur = u[0]
    tau0 = u[1]
    fr   = u[2]
    extf = np.array([fz, taur, tau0, fr]).reshape(4,1)

    A = np.zeros((4,4))
    A[0][0] = mt+m1+m0
    A[0][1] = -m1*r*sin(th0)
    A[0][2] = -m1*r*sin(th0)
    A[0][3] = m1*cos(th0)
    A[1][0] = -m1*r*sin(th0)
    A[1][1] = (mt+m1+m0)*z0**2+((2*mt+2*m1+2*m0)*z+2*m1*r*cos(th0)+2*lt*mt)*z0+(mt+m1+m0)*z**2+(2*m1*r*cos(th0)+2*lt*mt)*z+m1*r**2+lt**2*mt
    A[1][2] = m1*r*cos(th0)*z0+m1*r*cos(th0)*z+m1*r**2
    A[1][3] = m1*sin(th0)*z0+m1*sin(th0)*z
    A[2][0] = -m1*r*sin(th0)
    A[2][1] = m1*r*cos(th0)*z0+m1*r*cos(th0)*z+m1*r**2
    A[2][2] = m1*r**2
    A[2][3] = 0
    A[3][0] = m1*cos(th0)
    A[3][1] = m1*sin(th0)*z0+m1*sin(th0)*z
    A[3][2] = 0
    A[3][3] = m1

    b = np.zeros((4,1))
    b[0] = (-dthr**2*mt*sin(thr)**2*z0)-dthr**2*m1*sin(thr)**2*z0-dthr**2*m0*sin(thr)**2*z0-dthr**2*mt*cos(thr)**2*z0-dthr**2*m1*cos(thr)**2*z0-dthr**2*m0*cos(thr)**2*z0-dthr**2*mt*sin(thr)**2*z-dthr**2*m1*sin(thr)**2*z-dthr**2*m0*sin(thr)**2*z-dthr**2*mt*cos(thr)**2*z-dthr**2*m1*cos(thr)**2*z-dthr**2*m0*cos(thr)**2*z+k*z-dthr**2*m1*r*sin(thr)*sin(thr+th0)-2*dth0*dthr*m1*r*sin(thr)*sin(thr+th0)-dth0**2*m1*r*sin(thr)*sin(thr+th0)-2*dr*dthr*m1*cos(thr)*sin(thr+th0)-2*dr*dth0*m1*cos(thr)*sin(thr+th0)+2*dr*dthr*m1*sin(thr)*cos(thr+th0)+2*dr*dth0*m1*sin(thr)*cos(thr+th0)-dthr**2*m1*r*cos(thr)*cos(thr+th0)-2*dth0*dthr*m1*r*cos(thr)*cos(thr+th0)-dth0**2*m1*r*cos(thr)*cos(thr+th0)-dthr**2*lt*mt*sin(thr)**2-dthr**2*lt*mt*cos(thr)**2+g*mt*cos(thr)+g*m1*cos(thr)+g*m0*cos(thr)
    b[1] = 2*dr*dthr*m1*sin(thr)*sin(thr+th0)*z0+2*dr*dth0*m1*sin(thr)*sin(thr+th0)*z0-2*dth0*dthr*m1*r*cos(thr)*sin(thr+th0)*z0-dth0**2*m1*r*cos(thr)*sin(thr+th0)*z0+2*dth0*dthr*m1*r*sin(thr)*cos(thr+th0)*z0+dth0**2*m1*r*sin(thr)*cos(thr+th0)*z0+2*dr*dthr*m1*cos(thr)*cos(thr+th0)*z0+2*dr*dth0*m1*cos(thr)*cos(thr+th0)*z0+2*dthr*dz*mt*sin(thr)**2*z0+2*dthr*dz*m1*sin(thr)**2*z0+2*dthr*dz*m0*sin(thr)**2*z0-g*mt*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0+2*dthr*dz*mt*cos(thr)**2*z0+2*dthr*dz*m1*cos(thr)**2*z0+2*dthr*dz*m0*cos(thr)**2*z0+2*dr*dthr*m1*sin(thr)*sin(thr+th0)*z+2*dr*dth0*m1*sin(thr)*sin(thr+th0)*z-2*dth0*dthr*m1*r*cos(thr)*sin(thr+th0)*z-dth0**2*m1*r*cos(thr)*sin(thr+th0)*z+2*dth0*dthr*m1*r*sin(thr)*cos(thr+th0)*z+dth0**2*m1*r*sin(thr)*cos(thr+th0)*z+2*dr*dthr*m1*cos(thr)*cos(thr+th0)*z+2*dr*dth0*m1*cos(thr)*cos(thr+th0)*z+2*dthr*dz*mt*sin(thr)**2*z+2*dthr*dz*m1*sin(thr)**2*z+2*dthr*dz*m0*sin(thr)**2*z-g*mt*sin(thr)*z-g*m1*sin(thr)*z-g*m0*sin(thr)*z+2*dthr*dz*mt*cos(thr)**2*z+2*dthr*dz*m1*cos(thr)**2*z+2*dthr*dz*m0*cos(thr)**2*z+2*dr*dthr*m1*r*sin(thr+th0)**2+2*dr*dth0*m1*r*sin(thr+th0)**2+2*dthr*dz*m1*r*sin(thr)*sin(thr+th0)-g*m1*r*sin(thr+th0)+2*dr*dthr*m1*r*cos(thr+th0)**2+2*dr*dth0*m1*r*cos(thr+th0)**2+2*dthr*dz*m1*r*cos(thr)*cos(thr+th0)+2*dthr*dz*lt*mt*sin(thr)**2-g*lt*mt*sin(thr)+2*dthr*dz*lt*mt*cos(thr)**2
    b[2] = dthr**2*m1*r*cos(thr)*sin(thr+th0)*z0-dthr**2*m1*r*sin(thr)*cos(thr+th0)*z0+dthr**2*m1*r*cos(thr)*sin(thr+th0)*z-dthr**2*m1*r*sin(thr)*cos(thr+th0)*z+2*dr*dthr*m1*r*sin(thr+th0)**2+2*dr*dth0*m1*r*sin(thr+th0)**2+2*dthr*dz*m1*r*sin(thr)*sin(thr+th0)-g*m1*r*sin(thr+th0)+2*dr*dthr*m1*r*cos(thr+th0)**2+2*dr*dth0*m1*r*cos(thr+th0)**2+2*dthr*dz*m1*r*cos(thr)*cos(thr+th0)
    b[3] = (-dthr**2*m1*sin(thr)*sin(thr+th0)*z0)-dthr**2*m1*cos(thr)*cos(thr+th0)*z0-dthr**2*m1*sin(thr)*sin(thr+th0)*z-dthr**2*m1*cos(thr)*cos(thr+th0)*z-dthr**2*m1*r*sin(thr+th0)**2-2*dth0*dthr*m1*r*sin(thr+th0)**2-dth0**2*m1*r*sin(thr+th0)**2+2*dthr*dz*m1*cos(thr)*sin(thr+th0)-dthr**2*m1*r*cos(thr+th0)**2-2*dth0*dthr*m1*r*cos(thr+th0)**2-dth0**2*m1*r*cos(thr+th0)**2-2*dthr*dz*m1*sin(thr)*cos(thr+th0)+g*m1*cos(thr+th0)
    return A, b, extf


def airAb(s, u):
    thr  = s[IDX_thr]
    th0  = s[IDX_th0]
    r  = s[IDX_r]
    dthr = s[IDX_dthr]
    dth0 = s[IDX_dth0]
    dr = s[IDX_dr]
    taur = u[0]
    tau0 = u[1]
    fr   = u[2]
    extf = np.array([0, 0, taur, tau0, fr]).reshape(5,1)

    A = np.zeros((5,5))
    A[0][0] = mt+m1+m0
    A[0][1] = 0
    A[0][2] = ((-mt)-m1-m0)*cos(thr)*z0+m1*r*sin(th0)*sin(thr)+((-m1*r*cos(th0))-lt*mt)*cos(thr)
    A[0][3] = m1*r*sin(th0)*sin(thr)-m1*r*cos(th0)*cos(thr)
    A[0][4] = (-m1*cos(th0)*sin(thr))-m1*sin(th0)*cos(thr)
    A[1][0] = 0
    A[1][1] = mt+m1+m0
    A[1][2] = ((-mt)-m1-m0)*sin(thr)*z0+((-m1*r*cos(th0))-lt*mt)*sin(thr)-m1*r*sin(th0)*cos(thr)
    A[1][3] = (-m1*r*cos(th0)*sin(thr))-m1*r*sin(th0)*cos(thr)
    A[1][4] = m1*cos(th0)*cos(thr)-m1*sin(th0)*sin(thr)
    A[2][0] = ((-mt)-m1-m0)*cos(thr)*z0+m1*r*sin(th0)*sin(thr)+((-m1*r*cos(th0))-lt*mt)*cos(thr)
    A[2][1] = ((-mt)-m1-m0)*sin(thr)*z0+((-m1*r*cos(th0))-lt*mt)*sin(thr)-m1*r*sin(th0)*cos(thr)
    A[2][2] = (mt+m1+m0)*z0**2+(2*m1*r*cos(th0)+2*lt*mt)*z0+m1*r**2+lt**2*mt
    A[2][3] = m1*r*cos(th0)*z0+m1*r**2
    A[2][4] = m1*sin(th0)*z0
    A[3][0] = m1*r*sin(th0)*sin(thr)-m1*r*cos(th0)*cos(thr)
    A[3][1] = (-m1*r*cos(th0)*sin(thr))-m1*r*sin(th0)*cos(thr)
    A[3][2] = m1*r*cos(th0)*z0+m1*r**2
    A[3][3] = m1*r**2
    A[3][4] = 0
    A[4][0] = (-m1*cos(th0)*sin(thr))-m1*sin(th0)*cos(thr)
    A[4][1] = m1*cos(th0)*cos(thr)-m1*sin(th0)*sin(thr)
    A[4][2] = m1*sin(th0)*z0
    A[4][3] = 0
    A[4][4] = m1

    b = np.zeros((5,1))
    b[0] = dthr**2*mt*sin(thr)*z0+dthr**2*m1*sin(thr)*z0+dthr**2*m0*sin(thr)*z0+dthr**2*m1*r*sin(thr+th0)+2*dth0*dthr*m1*r*sin(thr+th0)+dth0**2*m1*r*sin(thr+th0)-2*dr*dthr*m1*cos(thr+th0)-2*dr*dth0*m1*cos(thr+th0)+dthr**2*lt*mt*sin(thr)
    b[1] = (-dthr**2*mt*cos(thr)*z0)-dthr**2*m1*cos(thr)*z0-dthr**2*m0*cos(thr)*z0-2*dr*dthr*m1*sin(thr+th0)-2*dr*dth0*m1*sin(thr+th0)-dthr**2*m1*r*cos(thr+th0)-2*dth0*dthr*m1*r*cos(thr+th0)-dth0**2*m1*r*cos(thr+th0)-dthr**2*lt*mt*cos(thr)+g*mt+g*m1+g*m0
    b[2] = 2*dr*dthr*m1*sin(thr)*sin(thr+th0)*z0+2*dr*dth0*m1*sin(thr)*sin(thr+th0)*z0-2*dth0*dthr*m1*r*cos(thr)*sin(thr+th0)*z0-dth0**2*m1*r*cos(thr)*sin(thr+th0)*z0+2*dth0*dthr*m1*r*sin(thr)*cos(thr+th0)*z0+dth0**2*m1*r*sin(thr)*cos(thr+th0)*z0+2*dr*dthr*m1*cos(thr)*cos(thr+th0)*z0+2*dr*dth0*m1*cos(thr)*cos(thr+th0)*z0-g*mt*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0+2*dr*dthr*m1*r*sin(thr+th0)**2+2*dr*dth0*m1*r*sin(thr+th0)**2-g*m1*r*sin(thr+th0)+2*dr*dthr*m1*r*cos(thr+th0)**2+2*dr*dth0*m1*r*cos(thr+th0)**2-g*lt*mt*sin(thr)
    b[3] = dthr**2*m1*r*cos(thr)*sin(thr+th0)*z0-dthr**2*m1*r*sin(thr)*cos(thr+th0)*z0+2*dr*dthr*m1*r*sin(thr+th0)**2+2*dr*dth0*m1*r*sin(thr+th0)**2-g*m1*r*sin(thr+th0)+2*dr*dthr*m1*r*cos(thr+th0)**2+2*dr*dth0*m1*r*cos(thr+th0)**2
    b[4] = (-dthr**2*m1*sin(thr)*sin(thr+th0)*z0)-dthr**2*m1*cos(thr)*cos(thr+th0)*z0-dthr**2*m1*r*sin(thr+th0)**2-2*dth0*dthr*m1*r*sin(thr+th0)**2-dth0**2*m1*r*sin(thr+th0)**2-dthr**2*m1*r*cos(thr+th0)**2-2*dth0*dthr*m1*r*cos(thr+th0)**2-dth0**2*m1*r*cos(thr+th0)**2+g*m1*cos(thr+th0)

    return A, b, extf

def f_ground(s, u):
    A, b, extf = groundAb(s, u)
    if np.linalg.matrix_rank(A) < 4:
        print("ground", np.linalg.det(A))
        print("ground", A)
        raise Exception("rank")
    y = np.linalg.solve(A, extf-b).reshape(4)
    f = np.zeros(IDX_MAX)
    f[IDX_xr]   = 0
    f[IDX_yr]   = 0
    f[IDX_z]   = y[0]
    f[IDX_thr] = y[1]
    f[IDX_th0] = y[2]
    f[IDX_r]   = y[3]
    return f

def f_air(s, u):
    A, b, extf = airAb(s, u)
    if np.linalg.matrix_rank(A) < 5:
        print("air", np.linalg.det(A))
        print("air", A)
        raise Exception("rank")
    y = np.linalg.solve(A, extf-b).reshape(5)
    f = np.zeros(IDX_MAX)
    f[IDX_xr]   = y[0]
    f[IDX_yr]   = y[1]
    f[IDX_thr] = y[2]
    f[IDX_z]   = 0
    f[IDX_th0] = y[3]
    f[IDX_r] = y[4]
    return f

def step(t, s, u, dt):
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
    ret[IDX_thr] = normalize_angle(ret[IDX_thr])
    return True, 1 if ground(ret) else 0, t+dt, ret

