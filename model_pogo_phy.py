import numpy as np
from math import atan2, acos, sqrt
from numpy import sin, cos, abs
import sys

# Pogo-rotation-knee-phy
max_z = 0.55

# ccw is positive
ref_min_thk = np.deg2rad(-140)
ref_max_thk = np.deg2rad(-20)
ref_min_th1 = np.deg2rad(0)
ref_max_th1 = np.deg2rad(90)
ref_min_a = 0.05
ref_max_a = 0.45

z0 = 0.55
l0 = 0.4
l1 = 0.4
l2 = 0.4
lh = 0.2
lt = 0.9
mr = 1
m0 = 10
mk = 10
m1 = 20
m2 = 30
mt = 2
g  = 0
#g  = 9.8
#g  = -9.8
K  = 10000 # mgh = 1/2 k x^2 -> T=2*pi sqrt(m/k)
c = 1

MAX_TORQUE0=600 # knee can keep 100kg weight at pi/2 + arm
MAX_TORQUE1=800 #west
MAX_FORCE=800 # arm [N]

inf = float('inf')
Kp = np.array([1200, 1200, 400])
#Kd = Kp * (0.01)
Kd = Kp * (0.1)

M = np.array([mr, mr, m0, m0, mk, mk, m1, m1, m2, m2, mt, mt])
invM = np.diag(1. / M)
#M[0] = 0
#M[1] = 0
#invM[0,0] = 0
#invM[1,1] = 0

#-----------------
# State
#-----------------
NUM_OF_MASS_POINTS = 6
IDX_VEL = NUM_OF_MASS_POINTS * 2
IDX_MAX = NUM_OF_MASS_POINTS * 2 * 2

IDX_r   = 0
IDX_0   = 1
IDX_k   = 2
IDX_1   = 3
IDX_2   = 4
IDX_t   = 5

IDX_xr, IDX_yr, IDX_dxr, IDX_dyr = IDX_r*2, IDX_r*2+1, IDX_VEL+IDX_r*2, IDX_VEL+IDX_r*2+1
IDX_x0, IDX_y0, IDX_dx0, IDX_dy0 = IDX_0*2, IDX_0*2+1, IDX_VEL+IDX_0*2, IDX_VEL+IDX_0*2+1
IDX_xk, IDX_yk, IDX_dxk, IDX_dyk = IDX_k*2, IDX_k*2+1, IDX_VEL+IDX_k*2, IDX_VEL+IDX_k*2+1
IDX_x1, IDX_y1, IDX_dx1, IDX_dy1 = IDX_1*2, IDX_1*2+1, IDX_VEL+IDX_1*2, IDX_VEL+IDX_1*2+1
IDX_x2, IDX_y2, IDX_dx2, IDX_dy2 = IDX_2*2, IDX_2*2+1, IDX_VEL+IDX_2*2, IDX_VEL+IDX_2*2+1
IDX_xt, IDX_yt, IDX_dxt, IDX_dyt = IDX_t*2, IDX_t*2+1, IDX_VEL+IDX_t*2, IDX_VEL+IDX_t*2+1

def reset_state(np_random=None):
    thr = 0
    th0 = np.deg2rad(30)
    thk = np.deg2rad(-90)
    th1 = np.deg2rad(74)
    z   = 0.
    pr = np.array([0, 1])
    #pr = np.array([0, 0.001])
    #pr = np.array([0, 0.1])
    #pr = np.array([0, 0])
    p0 = pr + (z0 + z) * np.array([-np.sin(thr), np.cos(thr)])
    pk = p0 + l0 * np.array([-np.sin(thr+th0), np.cos(thr+th0)])
    p1 = pk + l1 * np.array([-np.sin(thr+th0+thk), np.cos(thr+th0+thk)])
    p2 = p1 + l2 * np.array([-np.sin(thr+th0+thk+th1), np.cos(thr+th0+thk+th1)])
    pt = p0 + lt * np.array([-np.sin(thr), np.cos(thr)])

    s = np.zeros(IDX_MAX, dtype=np.float64)
    s[IDX_xr:IDX_yr+1]  = pr
    s[IDX_x0:IDX_y0+1]  = p0
    s[IDX_xk:IDX_yk+1]  = pk
    s[IDX_x1:IDX_y1+1]  = p1
    s[IDX_x2:IDX_y2+1]  = p2
    s[IDX_xt:IDX_yt+1]  = pt

    #vr = np.array([-1, -1])
    #v0 = np.array([-1, -1])
    #vk = np.array([-1, -1])
    #s[IDX_dxr:IDX_dyr+1]  = vr
    #s[IDX_dx0:IDX_dy0+1]  = v0
    #s[IDX_dxk:IDX_dyk+1]  = vk
    return s

def print_state(s, titlePrefix="", fieldPrefix=""):
    ps = node_pos(s)
    for i in range(len(ps)):
        #print(f"{titlePrefix}OBJ{i:}:P {s[2*i]:.2f},{s[2*i+1]:.2f} :V {s[IDX_VEL+2*i]:.4f},{s[IDX_VEL+2*i+1]:.4f}")
        print(f"{titlePrefix}OBJ{i:}:{fieldPrefix}P {s[2*i+1]} :{fieldPrefix}V {s[IDX_VEL+2*i+1]}")
    #pr = s[IDX_xr:IDX_yr+1]
    #p0 = s[IDX_x0:IDX_y0+1]
    #vr = s[IDX_dxr:IDX_dyr+1]
    #v0 = s[IDX_dx0:IDX_dy0+1]
    #z = np.linalg.norm(pr - p0) - z0
    #dz = np.linalg.norm(vr - v0) - z0
    #print(f"z:{fieldPrefix}P {z} :{fieldPrefix}V {dz}")

def max_u():
    return np.array([MAX_TORQUE0, MAX_TORQUE1, MAX_FORCE])

def torq_limit(s, u):
    m = max_u()
    ret = np.clip(u, -m, m)
    return ret

def ref_clip(ref):
    return np.clip(ref, np.array([ref_min_thk, ref_min_th1, ref_min_a]), np.array([ref_max_thk, ref_max_th1, ref_max_a]))

def node_pos(s):
    #return s[IDX_xr:IDX_yr+1],
    #return s[IDX_xr:IDX_yr+1], s[IDX_x0:IDX_y0+1], s[IDX_xk:IDX_yk+1]
    #return s[IDX_xr:IDX_yr+1], s[IDX_x0:IDX_y0+1]
    return s[IDX_xr:IDX_yr+1], s[IDX_x0:IDX_y0+1], s[IDX_xk:IDX_yk+1], s[IDX_x1:IDX_y1+1], s[IDX_x2:IDX_y2+1], s[IDX_xt:IDX_yt+1]

def head_pos(s):
    return s[IDX_x2:IDX_y2+1] + normalize(s[IDX_x2:IDX_y2+1] - s[IDX_x1:IDX_y1+1]) * lh

def node_vel(s):
    #return s[IDX_dxr:IDX_dyr+1],
    #return s[IDX_dxr:IDX_dyr+1], s[IDX_dx0:IDX_dy0+1], s[IDX_dxk:IDX_dyk+1] 
    return s[IDX_dxr:IDX_dyr+1], s[IDX_dx0:IDX_dy0+1]
    #return s[IDX_dxr:IDX_dyr+1], s[IDX_dx0:IDX_dy0+1], s[IDX_dxk:IDX_dyk+1], s[IDX_dx1:IDX_dy1+1], s[IDX_dx2:IDX_dy2+1], s[IDX_dxt:IDX_dyt+1]

def normalize(v):
    return v / np.linalg.norm(v)

def normal(v):
    return np.array([-v[1], v[0]]) # cross( ) z axis for 3D

def vec2rad(v1,v2): # angle from v1 to v2
    s = np.cross(v1, v2)
    c = v1 @ v2
    return atan2(s,c)

def calc_joint_property(s):
    pr = s[IDX_xr:IDX_yr+1]
    p0 = s[IDX_x0:IDX_y0+1]
    pk = s[IDX_xk:IDX_yk+1]
    p1 = s[IDX_x1:IDX_y1+1]
    p2 = s[IDX_x2:IDX_y2+1]
    pt = s[IDX_xt:IDX_yt+1]
    pr0 = p0 - pr
    p01 = p1 - p0
    p0k = pk - p0
    pk1 = p1 - pk
    p12 = p2 - p1
    p2t = pt - p2
    lr0 = np.linalg.norm(pr0)
    l0k = np.linalg.norm(p0k)
    lk1 = np.linalg.norm(pk1)
    l12 = np.linalg.norm(p12)
    d = np.linalg.norm(p2t)

    thr = atan2(pr0[1], pr0[0]) - np.pi/2
    thk = vec2rad(p0k/l0k, pk1/lk1)
    th1 = vec2rad(pk1/lk1, p12/l12)

    vr = s[IDX_dxr:IDX_dyr+1]
    v0 = s[IDX_dx0:IDX_dy0+1]
    vk = s[IDX_dxk:IDX_dyk+1]
    v1 = s[IDX_dx1:IDX_dy1+1]
    v2 = s[IDX_dx2:IDX_dy2+1]
    vt = s[IDX_dxt:IDX_dyt+1]
    dthr0 = np.cross(pr0, v0-vr)/lr0**2
    dth0k = np.cross(p0k, vk-v0)/l0k**2
    dthk1 = np.cross(pk1, v1-vk)/lk1**2
    dth12 = np.cross(p12, v2-v1)/l12**2
    dd = (p2t/d) @ (vt -v2)

    dthr = dthr0
    dth1 = dth12 - dthr
    dthk = dthk1 - dth0k

    return thk, th1, d, dthk, dth1, dd

def init_ref(s):
    thk, th1, d, _, _, _ = calc_joint_property(s)
    return np.array([thk, th1, d])

def check_invariant(s):
    ps = list(node_pos(s))
    for i in range(1,len(ps)):
        if ps[i][1] <= 0.001:
            print(f"GAME OVER p{i}={ps[i]:}")
            return False
    return True

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
    j[2*idx0]   = (l01**2*y2-l02**2*y1+(l02**2-l01**2)*y0)/(l01**2*l02**2)
    j[2*idx0+1] = -(l01**2*x2-l02**2*x1+(l02**2-l01**2)*x0)/(l01**2*l02**2)
    j[2*idx1]   = (y1-y0)/l01**2
    j[2*idx1+1] = -(x1-x0)/l01**2
    j[2*idx2]   = -(y2-y0)/l02**2
    j[2*idx2+1] = (x2-x0)/l02**2
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
    x0 = p0[0]
    y0 = p0[1]
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]
    j = np.zeros(IDX_VEL)
    j[2*idx0]   = -(y1-y0)/l01**2
    j[2*idx0+1] = (x1-x0)/l01**2
    j[2*idx1]   = (l01**2*y2+(l12**2-l01**2)*y1-l12**2*y0)/(l01**2*l12**2)
    j[2*idx1+1] = -(l01**2*x2+(l12**2-l01**2)*x1-l12**2*x0)/(l01**2*l12**2)
    j[2*idx2]   = -(y2-y1)/l12**2
    j[2*idx2+1] = (x2-x1)/l12**2
    return j, b, pred(C)

def pred_gt0(C):
    return C>0

def pred_lt0(C):
    return C<0

def pred_ne0(C):
    return C!=0

extforce = [ ("g", lambda t, s, u: force_gravity(s))
           , ("s", lambda t, s, u: force_spring(s, z0, K, IDX_r, IDX_0))
           , ("m0", lambda t, s, u: force_motor(s, IDX_0, IDX_k, IDX_1, u[0], -MAX_TORQUE0, MAX_TORQUE0))
           , ("m1", lambda t, s, u: force_motor(s, IDX_k, IDX_1, IDX_2, u[1], -MAX_TORQUE1, MAX_TORQUE1))
           , ("m2", lambda t, s, u: force_linear(s, IDX_2, IDX_t, u[2], -MAX_FORCE, MAX_FORCE))
           ]
constraints = [ ("ground-pen", lambda s, dt: constraint_ground_penetration(s, IDX_r, 0, dt, 0.1, pred_gt0), (-inf, 0))
              , ("ground-fric", lambda s, dt: constraint_ground_friction(s, IDX_r, 0, dt), (-inf, inf))
              , ("line-r0t", lambda s, dt: constraint_angle(s, IDX_r, IDX_0, IDX_t, 0, dt, 0.1, pred_ne0), (-inf, inf))
              , ("dist-0k", lambda s, dt: constraint_distant(s, IDX_0, IDX_k, l0, dt, 0.3, pred_ne0), (-inf, inf))
              , ("dist-k1", lambda s, dt: constraint_distant(s, IDX_k, IDX_1, l1, dt, 0.3, pred_ne0), (-inf, inf))
              , ("dist-12", lambda s, dt: constraint_distant(s, IDX_1, IDX_2, l2, dt, 0.3, pred_ne0), (-inf, inf))
              , ("dist-0t", lambda s, dt: constraint_distant(s, IDX_0, IDX_t, lt, dt, 0.3, pred_ne0), (-inf, inf))
              , ("limit-0k1-min", lambda s, dt: constraint_angle(s, IDX_0, IDX_k, IDX_1, np.deg2rad(-150), dt, 0.1, pred_lt0), (0, inf))
              , ("limit-0k1-max", lambda s, dt: constraint_angle(s, IDX_0, IDX_k, IDX_1, np.deg2rad(-10), dt, 0.1, pred_gt0), (-inf, 0))
              , ("limit-k12-min", lambda s, dt: constraint_angle(s, IDX_k, IDX_1, IDX_2, np.deg2rad(-10), dt, 0.1, pred_lt0), (0, inf))
              , ("limit-k12-max", lambda s, dt: constraint_angle(s, IDX_k, IDX_1, IDX_2, np.deg2rad(130), dt, 0.1, pred_gt0), (-inf, 0))
              , ("limit-12t-min", lambda s, dt: constraint_angle(s, IDX_1, IDX_2, IDX_t, np.deg2rad(10), dt, 0.1, pred_lt0), (0, inf))
              , ("limit-12t-max", lambda s, dt: constraint_angle(s, IDX_1, IDX_2, IDX_t, np.deg2rad(170), dt, 0.1, pred_gt0), (-inf, 0))
              , ("limit-2t-min", lambda s, dt: constraint_distant(s, IDX_2, IDX_t, 0.0, dt, 0.1, pred_lt0), (0, inf))
              , ("limit-2t-max", lambda s, dt: constraint_distant(s, IDX_2, IDX_t, 0.5, dt, 0.1, pred_gt0), (-inf, 0))
              , ("stick < hip", lambda s, dt: constraint_point_line_penetration(s, IDX_0, IDX_t, IDX_1, dt, 0.1, pred_gt0), (-inf, 0))
              ]

optional_constraints = []
optional_extforce = []

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

    print(names)
    K = J @ invM @ J.T
    r = -b - J @ (v  + invM @ fext * dt)
    lmd = np.linalg.solve(K, r)
    #print("check lmd", lmd)
    lmd = np.clip(lmd, np.array(cmin), np.array(cmax))
    #print("check lmd", lmd)
    impulse = J.T @ lmd
    print("check impulse", impulse)
    return impulse

def calc_ext_force(t, s, u, dt):
    fext = np.zeros(2*NUM_OF_MASS_POINTS)
    for name, ff in extforce + optional_extforce:
        f = ff(t, s, u)
        fext = fext + f
        print("ext-force impulse", name, f*dt)
    return fext

def step(t, s, u, dt):
    prev_tx, prev_ty, prev_a = moment(s)
    new_s = s.copy()
    fext = calc_ext_force(t, s, u, dt)
    pc = calc_constraint_impulse(new_s, fext, dt)
    pe = fext * dt
    new_s[IDX_VEL:] = new_s[IDX_VEL:] + invM @ (pc + pe)
    new_s[0:IDX_VEL] = new_s[0:IDX_VEL] + dt * new_s[IDX_VEL:]
    return "normal", t+dt, new_s

def energyS(s):
    pr = s[2*IDX_r:2*IDX_r+2]
    p0 = s[2*IDX_0:2*IDX_0+2]
    z = np.linalg.norm(pr - p0) - z0
    return 1/2 * k * z ** 2

def energyU(s):
    ps = list(node_pos(s))
    return sum([g * ps[i][1] * M[2*i] for i in range(len(ps))])# + energyS(s)

def energyT(s):
    vs = list(node_vel(s))
    return sum([1/2 * (vs[i] @ vs[i]) * M[2*i] for i in range(len(vs))])

def energy(s):
    return energyU(s) + energyT(s)

def cog(s):
    ps = list(node_pos(s))
    p = sum([M[2*i]*ps[i] for i in range(len(ps))])/sum(M[0::2])
    return p


def moment(s):
    vs = list(node_vel(s))
    tm = sum([M[2*i] * vs[i] for i in range(len(vs))])
    ps = list(node_pos(s))
    am = sum([M[2*i]*np.cross(vs[i]-vs[0], vs[i]-ps[0]) for i in range(len(vs))])
    return tm[0], tm[1], am

def pdcontrol(s, ref):
    thk, th1, d, dthk, dth1, dd = calc_joint_property(s)
    dob  = np.array([dthk, dth1, dd])
    ob = np.array([thk, th1, d])
    err = ref - ob
    #print(f"PD-ref: {np.rad2deg(ref[0])} {np.rad2deg(ref[1])} {ref[2]}")
    #print(f"PD-obs: {np.rad2deg(thk)} {np.rad2deg(th1)} {d}")
    ret = err * Kp - Kd * dob
    return ret


def test_dynamics():
    s = reset_state()
    dt = 0.001
    t = 0
    for i in range(5000):
        print("----------")
        print("t", t)
        print("----------")
        print_state(s)
        _, t, s = step(t, s, np.zeros(2), dt)

if __name__ == '__main__':
    test_dynamics()

