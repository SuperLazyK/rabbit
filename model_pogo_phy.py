import numpy as np
from math import atan2, acos, sqrt
from numpy import sin, cos, abs
import sys
import scipy

MAX_IMPULSE=10
debug=0
np.set_printoptions(linewidth=np.inf)

def debug_print(x):
    if debug:
        print(x)

def normalize_angle(x):
    return (x + np. pi) % (2 * np.pi) - np.pi

# Pogo-rotation-knee-phy
max_z = 0.55

# ccw is positive
ref_min_tht = np.deg2rad(3)
ref_max_tht = np.deg2rad(45)
ref_min_d = 0.10
ref_max_d = 0.45
REF_MIN = np.array([ref_min_tht, ref_min_d])
REF_MAX = np.array([ref_max_tht, ref_max_d])

limit_min_tht = np.deg2rad(-10)
limit_max_tht = np.deg2rad(120)
limit_min_d = 0.00
limit_max_d = 0.55

MAX_ROT_SPEED=100
MAX_SPEED=100

z0 = 0.55
lt = 0.83
mr = 1
m0 = 15
m2 = 55
mt = 7
g  = 0
#g  = 9.8
#g  = -9.8
K  = 15000 # mgh = 1/2 k x^2 -> T=2*pi sqrt(m/k)
c = 0
#c = 10

MAX_TORQUE2=300 # arm
MAX_FORCE=800 # arm [N]

inf = float('inf')
#Kp = np.array([2000, 8000])
Kp = np.array([400, 800])
#Kp = np.array([400, 400, 800])
#Kd = Kp * (0.01)
Kd = Kp * (0.1)

M = np.array([mr, mr, m0, m0, m2, m2, mt, mt])
invM = np.diag(1. / M)
#M[0] = 0
#M[1] = 0
#invM[0,0] = 0
#invM[1,1] = 0

#-----------------
# State
#-----------------
NUM_OF_MASS_POINTS = 4
IDX_VEL = NUM_OF_MASS_POINTS * 2
IDX_MAX = NUM_OF_MASS_POINTS * 2 * 2

IDX_r   = 0
IDX_0   = 1
IDX_2   = 2
IDX_t   = 3

IDX_xr, IDX_yr, IDX_dxr, IDX_dyr = IDX_r*2, IDX_r*2+1, IDX_VEL+IDX_r*2, IDX_VEL+IDX_r*2+1
IDX_x0, IDX_y0, IDX_dx0, IDX_dy0 = IDX_0*2, IDX_0*2+1, IDX_VEL+IDX_0*2, IDX_VEL+IDX_0*2+1
IDX_x2, IDX_y2, IDX_dx2, IDX_dy2 = IDX_2*2, IDX_2*2+1, IDX_VEL+IDX_2*2, IDX_VEL+IDX_2*2+1
IDX_xt, IDX_yt, IDX_dxt, IDX_dyt = IDX_t*2, IDX_t*2+1, IDX_VEL+IDX_t*2, IDX_VEL+IDX_t*2+1

def reset_state(pr, thr, tht, d, vr = np.array([0,0]), dthr=0, dtht=0, dd=0, z = 0, dz = 0):

    dir_thr = np.array([-np.sin(thr), np.cos(thr)])
    dir_thr0t = np.array([-np.sin(thr+tht), np.cos(thr+tht)])
    dir_dthr = np.array([-np.cos(thr), -np.sin(thr)])
    dir_dthr0t = np.array([-np.cos(thr+tht), -np.sin(thr+tht)])
    p0 = pr + (z0 + z) * dir_thr
    pt = p0 + lt * dir_thr
    p2 = pt + d * dir_thr0t

    v0 = vr + dz*dir_thr + dthr*(z0+z)*dir_dthr
    vt = v0 + dthr*lt*dir_dthr
    v2 = vt + dd*dir_thr0t + (dthr+dtht)*d*dir_dthr0t

    s = np.zeros(IDX_MAX, dtype=np.float64)
    s[IDX_xr:IDX_yr+1]  = pr
    s[IDX_x0:IDX_y0+1]  = p0
    s[IDX_x2:IDX_y2+1]  = p2
    s[IDX_xt:IDX_yt+1]  = pt
    s[IDX_dxr:IDX_dyr+1]  = vr
    s[IDX_dx0:IDX_dy0+1]  = v0
    s[IDX_dx2:IDX_dy2+1]  = v2
    s[IDX_dxt:IDX_dyt+1]  = vt

    return s


def print_state(s, titlePrefix="", fieldPrefix=""):
    ps = node_pos(s)
    for i in range(len(ps)):
        print(f"{titlePrefix}OBJ{i:}:P {s[2*i]:.2f},{s[2*i+1]:.2f} :V {s[IDX_VEL+2*i]:.4f},{s[IDX_VEL+2*i+1]:.4f}")
        #print(f"{titlePrefix}OBJ{i:}:{fieldPrefix}P {s[2*i+1]} :{fieldPrefix}V {s[IDX_VEL+2*i+1]}")

def max_u():
    return np.array([MAX_TORQUE2, MAX_FORCE])

def torq_limit(s, u):
    m = max_u()
    ret = np.clip(u, -m, m)
    return ret

def ref_clip(ref):
    return np.clip(ref, REF_MIN, REF_MAX)

def node_pos(s):
    return s[IDX_xr:IDX_yr+1], s[IDX_x0:IDX_y0+1], s[IDX_x2:IDX_y2+1], s[IDX_xt:IDX_yt+1]

#def head_pos(s):
#    return s[IDX_x2:IDX_y2+1] + normalize(s[IDX_x2:IDX_y2+1] - s[IDX_x1:IDX_y1+1]) * lh

def node_vel(s):
    return s[IDX_dxr:IDX_dyr+1], s[IDX_dx0:IDX_dy0+1], s[IDX_dx2:IDX_dy2+1], s[IDX_dxt:IDX_dyt+1]

def normalize(v):
    return v / np.linalg.norm(v)

def normal(v):
    return np.array([-v[1], v[0]]) # cross( ) z axis for 3D

def vec2rad(v1,v2): # angle from v1 to v2
    s = np.cross(v1, v2)
    c = v1 @ v2
    return atan2(s,c)

def calc_joint_property(s):
    ret = {}
    pr = s[IDX_xr:IDX_yr+1]
    p0 = s[IDX_x0:IDX_y0+1]
    p2 = s[IDX_x2:IDX_y2+1]
    pt = s[IDX_xt:IDX_yt+1]
    pr0 = p0 - pr
    p0t = pt - p0
    pt2 = p2 - pt
    lr0 = np.linalg.norm(pr0)
    ret['d'] = d = max(limit_min_d, np.linalg.norm(pt2))
    ret['z'] = lr0 -z0
    ret['thr'] = atan2(pr0[1], pr0[0]) - np.pi/2
    ret['tht'] = vec2rad(p0t/lt, pt2/d)

    vr = s[IDX_dxr:IDX_dyr+1]
    v0 = s[IDX_dx0:IDX_dy0+1]
    v2 = s[IDX_dx2:IDX_dy2+1]
    vt = s[IDX_dxt:IDX_dyt+1]

    ret['dd'] = (pt2/d) @ (v2 -vt)
    ret['dz'] = (pr0/lr0) @ (v0 - vr)
    ret['dthr'] = dthr = np.cross(pr0, v0-vr)/lr0**2
    ret['dtht'] = dtht = np.cross(pt2, v2-vt)/d**2 - dthr

    ret['prx'] = s[IDX_xr]
    ret['pry'] = s[IDX_yr]
    ret['vrx'] = s[IDX_dxr]
    ret['vry'] = s[IDX_dyr]
    return ret


def ground(s):
    return s[IDX_yr] < 0.05 or cog(s)[1] < 0.8

OBS_MIN = np.array([0, -MAX_SPEED, -np.pi,     0, limit_min_tht, limit_min_d])
OBS_MAX = np.array([5,  MAX_SPEED,  np.pi, max_z, limit_max_tht, limit_max_d])

def obs(s):
    vcog = dcog(s)
    pcog = cog(s)
    prop = calc_joint_property(s)
    return np.array([ pcog[1]
                    , prop['thr']
                    , prop['tht']
                    , prop['d']
                    , vcog[0]
                    #, prop['dz']
                    ])

def reward_imitation_jump(s, t):
    return 0

def reward_imitation_flip(s, t):
    return 0

def reward(s):
    vcog = dcog(s)
    dir0r = np.array([s[IDX_xr] - s[IDX_x0], s[IDX_yr]-s[IDX_y0]])

    pcog = cog(s)
    r_y = (energyU(s) + energyTy(s))/600
    r_thr = 0
    r_cogx = 0

    if ground(s):
        mode = "ground"
        r_cogx = np.exp(-(pcog[0]-s[IDX_xr])**2)*3
        r_thr = (2-np.linalg.norm(normalize(dir0r) - np.array([0, -1])))/2
    else:
        mode = "air"
        if vcog[1] < 0:
            r_thr = (2-np.linalg.norm(normalize(dir0r) - normalize(vcog))) * (3/(1+pcog[1]))
    debug_print((mode, r_y, r_thr, r_cogx))
    return r_y + r_thr + r_cogx

def init_ref(s):
    prop = calc_joint_property(s)
    return np.array([prop['tht'], prop['d']])

def check_invariant(s):
    ps = list(node_pos(s))
    for i in range(1,len(ps)):
        if ps[i][1] <= 0.001:
            reason = f"GAME OVER @ p{i}={ps[i]:}"
            return False, reason
    return True, ""

def force_gravity(s):
    f = np.zeros_like(M)
    f[1::2] = -g
    return M * f

#def force_vfric(s, idx):
#    v = s[IDX_VEL+2*idx:IDX_VEL+2*idx+2]
#    f = np.zeros_like(M)
#    f[2*idx:2*idx+2] = -v * 90. * M[2*idx:2*idx+2] # 0.9 * 1/dt
#    return f

def force_linear(s, idx0, idx1, u, umin, umax, flip=1):
    p0 = s[2*idx0:2*idx0+2]
    p1 = s[2*idx1:2*idx1+2]
    v = normalize(p1-p0)
    f = np.zeros_like(M)
    f[2*idx0:2*idx0+2] = -flip * v * u
    f[2*idx1:2*idx1+2] = flip * v * u
    return f

def force_spring(s, d, k, idx0, idx1):
    p0 = s[2*idx0:2*idx0+2]
    p1 = s[2*idx1:2*idx1+2]
    v0 = s[IDX_VEL+2*idx0:IDX_VEL+2*idx0+2]
    v1 = s[IDX_VEL+2*idx1:IDX_VEL+2*idx1+2]
    p01 = p1 - p0
    l01 = np.linalg.norm(p0 - p1)
    u01 = p01/l01
    v01 = v0 - v1
    f = np.zeros_like(M)
    fspring = k * (l01 - d)
    fric = c * (v01 @ u01) * u01
    f[2*idx0:2*idx0+2] = fspring * u01 - fric
    f[2*idx1:2*idx1+2] = -fspring * u01 + fric
    return f

def force_motor(s, idx0, idx1, idx2, u, umin, umax, ccw=1):
    f = np.zeros_like(M)
    p0 = s[2*idx0:2*idx0+2]
    p1 = s[2*idx1:2*idx1+2]
    p2 = s[2*idx2:2*idx2+2]
    l10 = np.linalg.norm(p0 - p1)
    l12 = np.linalg.norm(p2 - p1)
    u10 = (p0 - p1)/l10
    u12 = (p2 - p1)/l12
    torq = np.clip(u, umin, umax)
    f[2*idx0:2*idx0+2] = -ccw * normal(u10) * torq / l10
    f[2*idx2:2*idx2+2] = ccw * normal(u12) * torq / l12
    f[2*idx1:2*idx1+2] = -f[2*idx0:2*idx0+2] - f[2*idx2:2*idx2+2]
    return f

# dc/dt = - beta c(t) 
# non-elastic collistion
def constraint_ground_penetration(s, idx, y, dt, beta, pred):
    py = s[2*idx+1]
    C = y-py 
    b = beta*C/dt
    #dCdt = -vy
    j = np.zeros(IDX_VEL)
    j[2*idx+1] = -1
    return j, b, pred(C)

# dc/dt = - beta c(t) 
# non-elastic collistion
# line01 point idx2
def constraint_point_line_penetration(s, idx0, idx1, idx2, dt, beta, pred):
    p0 = s[2*idx0:2*idx0+2]
    p1 = s[2*idx1:2*idx1+2]
    p2 = s[2*idx2:2*idx2+2]
    p01=p1-p0
    p02=p2-p0
    l01=np.linalg.norm(p01)
    l02=np.linalg.norm(p02)
    C = vec2rad(p01/l01, p02/l02)
    b = beta*C/dt
    j = np.zeros(IDX_VEL)
    x0 = p0[0]
    y0 = p0[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    j[2*idx1]   = (y1-y0)/l01**2
    j[2*idx1+1] = -(x1-x0)/l01**2
    j[2*idx2]   = -(y2-y0)/l02**2
    j[2*idx2+1] = (x2-x0)/l02**2
    j[2*idx0]   = -j[2*idx1] - j[2*idx2]
    j[2*idx0+1] = -j[2*idx1+1] - j[2*idx2+1]
    return j, b, pred(C)

def constraint_ground_friction(s, idx, y, dt):
    py = s[2*idx+1]
    active = py < y + 0.005
    #dCdt = -vx
    j = np.zeros(IDX_VEL)
    j[2*idx] = -1
    return j, 0, active

def constraint_fixed_point_fric(s, idx, point, dt):
    p = s[2*idx:2*idx+2]
    v = p - point
    l = np.linalg.norm(v)
    j = np.zeros(IDX_VEL)
    if l < 0.0001:
        return j, 0, False
    else:
        j[2*idx:2*idx+2] = normal(v/l)
        return j, 0, True

def constraint_angleR(s, idx0, idx1, idx2, th, dt, beta, pred):
    p0 = s[2*idx0:2*idx0+2]
    p1 = s[2*idx1:2*idx1+2]
    p2 = s[2*idx2:2*idx2+2]
    p10 = p0-p1
    p12 = p2-p1
    l10 = np.linalg.norm(p10)
    l12 = np.linalg.norm(p12)
    th102 = vec2rad(p10/l10, p12/l12)
    C = th102 - th
    b = beta*C/dt
    j = np.zeros(IDX_VEL)
    j[2*idx0]   = -p10[1]/l10**2
    j[2*idx0+1] =  p10[0]/l10**2
    j[2*idx2]   = -p12[1]/l12**2
    j[2*idx2+1] =  p12[0]/l12**2
    j[2*idx1]   = -j[2*idx0] - j[2*idx2]
    j[2*idx1+1] = -j[2*idx0+1] - j[2*idx2+1]
    return j, b, pred(C)

# TODO use deadband
def constraint_fixed_point_distant(s, idx0, p, l, dt, beta, pred):
    p0 = s[2*idx0:2*idx0+2]
    C = np.linalg.norm(p0 - p) - l
    j = np.zeros(IDX_VEL)
    j[2*idx0:2*idx0+2] = (p0-p)/ np.linalg.norm(p0-p)
    b = beta*C/dt
    return j, b, pred(C)

def constraint_distant(s, idx0, idx1, l, dt, beta, pred):
    p0 = s[2*idx0:2*idx0+2]
    p1 = s[2*idx1:2*idx1+2]
    C = np.linalg.norm(p0 - p1) - l
    b = beta*C/dt
    j = np.zeros(IDX_VEL)
    r = np.linalg.norm(p0-p1)
    j[2*idx0:2*idx0+2] = (p0-p1)/np.linalg.norm(p0-p1)
    j[2*idx1:2*idx1+2] = -j[2*idx0:2*idx0+2]
    return j, b, pred(C)

# d(p/|p|)/dt[1] = -(n0x*n0y*v0y+(n0x**2-1)*v0x)/l0
# d(p/|p|)/dt[2] = -((n0y**2-1)*v0y+n0x*n0y*v0x)/l0
def constraint_angle(s, idx0, idx1, idx2, th, dt, beta, pred):
    p0 = s[2*idx0:2*idx0+2]
    p1 = s[2*idx1:2*idx1+2]
    p2 = s[2*idx2:2*idx2+2]
    p01 = p1-p0
    p12 = p2-p1
    l01 = np.linalg.norm(p01)
    l12 = np.linalg.norm(p12)
    th012 = vec2rad(p01/l01, p12/l12)
    C = th012 - th
    b = beta*C/dt
    j = np.zeros(IDX_VEL)
    x0 = p0[0]
    y0 = p0[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    j[2*idx0]   = -(y1-y0)/l01**2
    j[2*idx0+1] = (x1-x0)/l01**2
    j[2*idx2]   = -(y2-y1)/l12**2
    j[2*idx2+1] = (x2-x1)/l12**2
    j[2*idx1]   = -j[2*idx0] - j[2*idx2]
    j[2*idx1+1] = -j[2*idx0+1] - j[2*idx2+1]
    return j, b, pred(C)

def pred_gt0(C):
    return C>0

def pred_lt0(C):
    return C<0

def pred_ne0(C):
    return C!=0

extforce = [ ("g", lambda t, s, u: force_gravity(s))
           , ("s", lambda t, s, u: force_spring(s, z0, K, IDX_r, IDX_0))
           , ("mt-rev", lambda t, s, u: force_motor(s, IDX_0, IDX_t, IDX_2, u[0], -MAX_TORQUE2, MAX_TORQUE2)) # trick
           , ("mt-linear", lambda t, s, u: force_linear(s, IDX_2, IDX_t, u[1], -MAX_FORCE, MAX_FORCE))
           ]

constraints = [ ("ground-pen", lambda s, dt: constraint_ground_penetration(s, IDX_r, 0, dt, 0.1, pred_gt0), (-inf, 0))
              , ("ground-fric", lambda s, dt: constraint_ground_friction(s, IDX_r, 0, dt), (-inf, inf))
              , ("line-r0t", lambda s, dt: constraint_angle(s, IDX_r, IDX_0, IDX_t, 0, dt, 0.1, pred_ne0), (-inf, inf))
              #, ("fixed-pointer0", lambda s, dt: constraint_fixed_point_distant(s, IDX_0, np.array([0, 1+z0]), 0, dt, 0.1, pred_ne0), (-inf, inf))
              #, ("fixed-pointert", lambda s, dt: constraint_fixed_point_distant(s, IDX_t, np.array([0, 1 + z0+lt]), 0, dt, 0.1, pred_ne0), (-inf, inf))
              , ("dist-0t", lambda s, dt: constraint_distant(s, IDX_0, IDX_t, lt, dt, 0.3, pred_ne0), (-inf, inf))
              #, ("limit-0t2-min", lambda s, dt: constraint_angleR(s, IDX_0, IDX_t, IDX_2, limit_min_tht, dt, 0.1, pred_lt0), (0, inf))
              #, ("limit-0t2-max", lambda s, dt: constraint_angleR(s, IDX_0, IDX_t, IDX_2, limit_max_tht, dt, 0.1, pred_gt0), (-inf, 0))
              #, ("limit-2t-min", lambda s, dt: constraint_distant(s, IDX_2, IDX_t, limit_min_d, dt, 0.1, pred_lt0), (0, inf))
              #, ("limit-2t-max", lambda s, dt: constraint_distant(s, IDX_2, IDX_t, limit_max_d, dt, 0.1, pred_gt0), (-inf, 0))
              #, ("limit-r02-min", lambda s, dt: constraint_angleR(s, IDX_r, IDX_0, IDX_2, np.deg2rad(-45), dt, 0.1, pred_lt0), (0, inf))
              #, ("limit-r02-max", lambda s, dt: constraint_angleR(s, IDX_r, IDX_0, IDX_2, 0, dt, 0.1, pred_gt0), (-inf, 0))
              ]

optional_constraints = []
optional_extforce = []

def set_fixed_constraint_0t(arg_s):
    global optional_constraints
    optional_constraints = [ ("fixed-pointer0", lambda s, dt: constraint_fixed_point_distant(s, IDX_0, arg_s[2*IDX_0:2*IDX_0+2], 0, dt, 0.1, pred_ne0), (-inf, inf))
                           , ("fixed-pointert", lambda s, dt: constraint_fixed_point_distant(s, IDX_t, arg_s[2*IDX_t:2*IDX_t+2], 0, dt, 0.1, pred_ne0), (-inf, inf))
                           ]

def set_fixed_constraint(idx,  point):
    global optional_constraints
    global optional_extforce
    if point is None:
        optional_constraints = []
        #optional_extforce = []
    else:
        optional_constraints = [ ("fixed-pointer", lambda s, dt: constraint_fixed_point_distant(s, idx, point, 0, dt, 0.1, pred_ne0), (-inf, inf))
                               , ("fixed-pointer-fric", lambda s, dt: constraint_fixed_point_fric(s, idx, point, dt), (-inf, inf))
                               ]
        #optional_extforce = [("fixed-pointer-vfric", lambda t, s, u: force_vfric(s, idx))]

def calc_constraint_impulse(s, fext, dt):
    names, js, bs, cmin, cmax = [], [], [], [], []
    for name, f, (l, u) in constraints + optional_constraints:
        j, pf, active = f(s, dt)
        if active:
            names.append(name)
            js.append(j)
            bs.append(pf)
            cmin.append(l)
            cmax.append(u)
    if len(js) == 0:
        return np.zeros(IDX_VEL)
    J = np.array(js)
    b = np.array(bs)
    v = s[IDX_VEL:]

    debug_print(names)
    K = J @ invM @ J.T
    #debug_print(("check K", K))
    #debug_print(("check det(K)", np.linalg.det(K)))
    #debug_print(("check J", J))
    r = -b - J @ (v  + invM @ fext * dt)
    #debug_print(("check r", r))
    try:
        lmd = np.linalg.solve(K, r)
    except np.linalg.LinAlgError as err:
        lmd, exit_code = scipy.sparse.linalg.cg(K, r, maxiter=K.shape[0]*2)
        if exit_code != 0:
            print("not converged")
    lmd = np.clip(lmd, np.array(cmin), np.array(cmax))
    #debug_print(("check lmd", lmd))
    impulse = J.T @ lmd
    debug_print(("check impulse", impulse))
    return np.clip(impulse, -MAX_IMPULSE, MAX_IMPULSE)

def calc_ext_force(t, s, u, dt):
    fext = np.zeros(2*NUM_OF_MASS_POINTS)
    for name, ff in extforce + optional_extforce:
        f = ff(t, s, u)
        fext = fext + f
        debug_print(("ext-force impulse", name, f*dt))
    return fext

def step(t, s, u, dt):
    try:
        prev_tx, prev_ty, prev_a = moment(s)
        new_s = s.copy()
        fext = calc_ext_force(t, s, u, dt)
        pc = calc_constraint_impulse(new_s, fext, dt)
        pe = fext * dt
        new_s[IDX_VEL:] = new_s[IDX_VEL:] + invM @ (pc + pe)
        new_s[0:IDX_VEL] = new_s[0:IDX_VEL] + dt * new_s[IDX_VEL:]
        return True, t+dt, new_s
    except np.linalg.LinAlgError as err:
        print(err)
        return False, t+dt, s


def energyS(s):
    pr = s[2*IDX_r:2*IDX_r+2]
    p0 = s[2*IDX_0:2*IDX_0+2]
    z = np.linalg.norm(pr - p0) - z0
    return 1/2 * K * z ** 2

def energyU(s):
    ps = list(node_pos(s))
    return sum([g * ps[i][1] * M[2*i] for i in range(len(ps))]) + energyS(s)

def energyTy(s):
    vs = list(node_vel(s))
    return sum([1/2 * (vs[i][1] ** 2) * M[2*i] for i in range(len(vs))])

def energyT(s):
    vs = list(node_vel(s))
    return sum([1/2 * (vs[i] @ vs[i]) * M[2*i] for i in range(len(vs))])

def energy(s):
    return energyU(s) + energyT(s)

def cog(s):
    ps = list(node_pos(s))
    p = sum([M[2*i]*ps[i] for i in range(len(ps))])/sum(M[0::2])
    return p

def dcog(s):
    vs = list(node_vel(s))
    v = sum([M[2*i]*vs[i] for i in range(len(vs))])/sum(M[0::2])
    return v

def moment(s):
    vs = list(node_vel(s))
    tm = sum([M[2*i] * vs[i] for i in range(len(vs))])
    ps = list(node_pos(s))
    am = sum([M[2*i]*np.cross(vs[i]-vs[0], vs[i]-ps[0]) for i in range(len(vs))])
    return tm[0], tm[1], am

def pdcontrol(s, ref):
    prop = calc_joint_property(s)
    dob  = np.array([prop['dtht'], prop['dd']])
    ob = np.array([prop['tht'], prop['d']])
    err = ref - ob
    debug_print(f"PD-ref: {np.rad2deg(ref[0])} {ref[1]}")
    debug_print(f"PD-obs: {np.rad2deg(ob[0])}  {ob[1]}")
    debug_print(f"PD-vel: {dob}")
    ret = err * Kp - Kd * dob
    return ret


