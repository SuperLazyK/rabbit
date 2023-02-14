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
mw = 0
m1 = 50
mt = 7
M=np.array([mr, m0, mk, mw, m1, mt])
#M=np.array([mr, m0, mk, m1, mt])
lw1 = 0.3
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
ref_max_thk = np.deg2rad(30)
ref_min_thw = np.deg2rad(-50)
ref_max_thw = np.deg2rad(0)
REF_MIN = np.array([ref_min_th0, ref_min_thk, ref_min_thw])
REF_MAX = np.array([ref_max_th0, ref_max_thk, ref_max_thw])

limit_min_thr = np.deg2rad(-90)
limit_max_thr = np.deg2rad(90)
limit_min_th0 = np.deg2rad(45)
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
Kp = 10*np.array([400, 800])
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
            reason = f"GAME OVER @ range error th0={s[IDX_th0]:}"
            return False, reason
    if s[IDX_thk] < limit_min_thk or s[IDX_thk] > limit_max_thk:
            reason = f"GAME OVER @ range error thk={s[IDX_thk]:}"
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

    A21 = np.zeros((4,2))
    A21[0][0]= ((-mt)-m1-m0)*sin(thr)
    A21[0][1]= (mt+m1+m0)*cos(thr)
    A21[1][0]= ((-mt)-m1-m0)*cos(thr)*z0+((-mt)-m1-m0)*cos(thr)*z+l*m1*sin(th0)*cos(thk)*sin(thr)+((-l*m1*cos(th0)*cos(thk))-lt*mt)*cos(thr)
    A21[1][1]= ((-mt)-m1-m0)*sin(thr)*z0+((-mt)-m1-m0)*sin(thr)*z+((-l*m1*cos(th0)*cos(thk))-lt*mt)*sin(thr)-l*m1*sin(th0)*cos(thk)*cos(thr)
    A21[2][0]= l*m1*sin(th0)*cos(thk)*sin(thr)-l*m1*cos(th0)*cos(thk)*cos(thr)
    A21[2][1]= (-l*m1*cos(th0)*cos(thk)*sin(thr))-l*m1*sin(th0)*cos(thk)*cos(thr)
    A21[3][0]= l*m1*cos(th0)*sin(thk)*sin(thr)+l*m1*sin(th0)*sin(thk)*cos(thr)
    A21[3][1]= l*m1*sin(th0)*sin(thk)*sin(thr)-l*m1*cos(th0)*sin(thk)*cos(thr)

    A22 = np.zeros((4,4))
    A22[0][0]= mt+m1+m0
    A22[0][1]= -l*m1*sin(th0)*cos(thk)
    A22[0][2]= -l*m1*sin(th0)*cos(thk)
    A22[0][3]= -l*m1*cos(th0)*sin(thk)
    A22[1][0]= -l*m1*sin(th0)*cos(thk)
    A22[1][1]= (mt+m1+m0)*z0**2+((2*mt+2*m1+2*m0)*z+2*l*m1*cos(th0)*cos(thk)+2*lt*mt)*z0+(mt+m1+m0)*z**2+(2*l*m1*cos(th0)*cos(thk)+2*lt*mt)*z+l**2*m1*cos(thk)**2+lt**2*mt
    A22[1][2]= l*m1*cos(th0)*cos(thk)*z0+l*m1*cos(th0)*cos(thk)*z+l**2*m1*cos(thk)**2
    A22[1][3]= (-l*m1*sin(th0)*sin(thk)*z0)-l*m1*sin(th0)*sin(thk)*z
    A22[2][0]= -l*m1*sin(th0)*cos(thk)
    A22[2][1]= l*m1*cos(th0)*cos(thk)*z0+l*m1*cos(th0)*cos(thk)*z+l**2*m1*cos(thk)**2
    A22[2][2]= l**2*m1*cos(thk)**2
    A22[2][3]= 0
    A22[3][0]= -l*m1*cos(th0)*sin(thk)
    A22[3][1]= (-l*m1*sin(th0)*sin(thk)*z0)-l*m1*sin(th0)*sin(thk)*z
    A22[3][2]= 0
    A22[3][3]= l**2*m1*sin(thk)**2

    #solve  A22 y(z,thr,th0,thk) = A21 (dx,dy)
    y = np.linalg.solve(A22, A21 @ s[IDX_dxr:IDX_dyr+1]).reshape(4)

    d = np.zeros(IDX_MAX)
    d[IDX_xr] = 0
    d[IDX_yr] = 0
    d[IDX_z]   = y[0]
    d[IDX_thr] = y[1]
    d[IDX_th0] = y[2]
    d[IDX_thk] = y[3]
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
    b[0] = dthr**2*mt*sin(thr)*z0+dthr**2*m1*sin(thr)*z0+dthr**2*m0*sin(thr)*z0+dthr**2*l*m1*cos(thk)*sin(thr+th0)+2*dth0*dthr*l*m1*cos(thk)*sin(thr+th0)+dthk**2*l*m1*cos(thk)*sin(thr+th0)+dth0**2*l*m1*cos(thk)*sin(thr+th0)+2*dthk*dthr*l*m1*sin(thk)*cos(thr+th0)+2*dth0*dthk*l*m1*sin(thk)*cos(thr+th0)+dthr**2*lt*mt*sin(thr)
    b[1] = (-dthr**2*mt*cos(thr)*z0)-dthr**2*m1*cos(thr)*z0-dthr**2*m0*cos(thr)*z0+2*dthk*dthr*l*m1*sin(thk)*sin(thr+th0)+2*dth0*dthk*l*m1*sin(thk)*sin(thr+th0)-dthr**2*l*m1*cos(thk)*cos(thr+th0)-2*dth0*dthr*l*m1*cos(thk)*cos(thr+th0)-dthk**2*l*m1*cos(thk)*cos(thr+th0)-dth0**2*l*m1*cos(thk)*cos(thr+th0)-dthr**2*lt*mt*cos(thr)+g*mt+g*m1+g*m0
    b[2] = (-2*dthk*dthr*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0)-2*dth0*dthk*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0-2*dth0*dthr*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dthk**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dth0**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0+2*dth0*dthr*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0+dthk**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0+dth0**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0-2*dthk*dthr*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0-2*dth0*dthk*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0-g*mt*sin(thr)*z0-g*m1*sin(thr)*z0-g*m0*sin(thr)*z0-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-g*l*m1*cos(thk)*sin(thr+th0)-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-g*lt*mt*sin(thr)
    b[3] = dthr**2*l*m1*cos(thk)*cos(thr)*sin(thr+th0)*z0-dthr**2*l*m1*cos(thk)*sin(thr)*cos(thr+th0)*z0-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2-g*l*m1*cos(thk)*sin(thr+th0)-2*dthk*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-2*dth0*dthk*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2
    b[4] = dthr**2*l*m1*sin(thk)*sin(thr)*sin(thr+th0)*z0+dthr**2*l*m1*sin(thk)*cos(thr)*cos(thr+th0)*z0+dthr**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+2*dth0*dthr*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+dthk**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+dth0**2*l**2*m1*cos(thk)*sin(thk)*sin(thr+th0)**2+dthr**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+2*dth0*dthr*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+dthk**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2+dth0**2*l**2*m1*cos(thk)*sin(thk)*cos(thr+th0)**2-g*l*m1*sin(thk)*cos(thr+th0)

    return A, b, extf

def f_ground(s, u):
    A, b, extf = groundAb(s, u)
    assert np.linalg.matrix_rank(A) == 4
    y = np.linalg.solve(A, extf-b).reshape(4)
    f = np.zeros(IDX_MAX)
    f[IDX_xr]   = 0
    f[IDX_yr]   = 0
    f[IDX_z]   = y[0]
    f[IDX_thr] = y[1]
    f[IDX_th0] = y[2]
    f[IDX_thk] = y[3]
    return f

def f_air(s, u):
    A, b, extf = airAb(s, u)
    assert np.linalg.matrix_rank(A) == 5, (s, A)
    y = np.linalg.solve(A, extf-b).reshape(5)
    f = np.zeros(IDX_MAX)
    f[IDX_xr]   = y[0]
    f[IDX_yr]   = y[1]
    f[IDX_thr] = y[2]
    f[IDX_z]   = 0
    f[IDX_th0] = y[3]
    f[IDX_thk] = y[4]
    return f

def jump_time(s, s1, dt): # z = 0
    z = s[IDX_z]
    z1 = s1[IDX_z]
    if z > 0 and z1 <= 0 or z < 0 and z1 >= 0:
        if abs(z - z1) > epsilon:
            return z * dt / (z - z1)
        else:
            return dt
    else:
        return None

def collision_time(s, s1, dt): # y0 = 0
    yr = s[IDX_yr]
    yr1 = s1[IDX_yr]
    if yr > 0 and yr1 <= 0:
        return yr * dt / (yr - yr1)
    else:
        return None

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
    return 1 if ground(ret) else 0, t+dt, ret

