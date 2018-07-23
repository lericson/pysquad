"Multicopter dynamics"

import numpy as np
import numba as nb

from .. import quat
from ..utis import env_param, polyval, cross

# Common type definitions
from numba.types import Tuple, none
StateT     = nb.float64[::1]
ActionT    = nb.float64[::1]
TimeDeltaT = nb.float64
StateArrayT  = nb.float64[:, ::1]
ActionArrayT = nb.float64[:, ::1]
DerivativesT = Tuple((StateArrayT, ActionArrayT))

eps = 1e-9
radius = 0.2  # m
mass = 1.3  # kg
num_rotors = 4
rpm_max = 20000
rotor_rpm_coeff = 6e1
friction_force = 1e-2
friction_torque = 1e-5
gravity = np.r_[0.0, 0.0, 9.82]

# I = mk^2 where k is the radius of gyration
inertia = mass*(0.275)**2
arm_length = radius
rotor_angles = 2*np.pi*(np.r_[0:num_rotors] - 1/2)/num_rotors
rotor_positions = arm_length*np.c_[np.cos(rotor_angles), np.sin(rotor_angles), np.zeros(num_rotors)]
rotor_axes = np.repeat(np.c_[0, 0, 1], num_rotors, axis=0)
rotor_directions = (-1)**np.r_[0:num_rotors]
# Thrust coefficients from https://www.bitcraze.io/2015/02/measuring-propeller-rpm-part-3/
thrust_rpm_max = (mass + 5)*9.82
thrust_coeffs = np.r_[+1.0942e-07, -2.1059e-04, 0.0]
thrust_coeffs *= thrust_rpm_max/np.polyval(thrust_coeffs, rpm_max)
z_torque_per_rpm = 5e-4/rpm_max

(ipx, ipy, ipz, iqi, iqj, iqk, iqr, ivx, ivy, ivz, iwx, iwy, iwz, iw0, iw1,
 iw2, iw3, imax) = range(13+num_rotors+1)

num_substeps = env_param('num_substeps', default=2, cast=int)

@nb.njit(none(TimeDeltaT, StateT, ActionT, StateT), cache=True, nogil=True)
def x_dot_out(t, x, u, out):
    x[3:7] /= np.linalg.norm(x[3:7]) + eps
    attq, linvel, angvel, rpms = x[3:7], x[7:10], x[10:13], x[13:]
    rpms = u*rpm_max
    force_bf = np.zeros(3)
    torque   = np.zeros(3)
    for i in range(num_rotors):
        rotor_force = polyval(thrust_coeffs, rpms[i])*rotor_axes[i, :]
        force_bf += rotor_force
        torque += cross(rotor_positions[i, :], rotor_force)
        torque += z_torque_per_rpm*rotor_directions[i]*(rpms[i]**2)*rotor_axes[i, :]
    torque -= angvel*friction_torque
    # Disabled because cross(av, v) = 0 for scalar a and vector v. In cases
    # where moment of inertia isn't scalar, this is needed.
    #torque -= np.cross(angvel, inertia*angvel)
    force_if  = quat.vecdot(attq, force_bf)
    force_if -= mass*gravity
    force_if -= linvel*friction_force
    out[0:3]   = linvel
    out[3:7]   = 0.5*quat.dot(attq, quat.from_vec(angvel))
    out[7:10]  = force_if/mass
    out[10:13] = torque/inertia
    out[13:]   = rotor_rpm_coeff*(u*rpm_max - x[13:])

@nb.njit(StateT(TimeDeltaT, StateT, ActionT), cache=True, nogil=True)
def x_dot(t, x, u):
    out = np.empty_like(x)
    x_dot_out(t, x, u, out)
    return out

@nb.njit(none(StateT, ActionT, TimeDeltaT), cache=True, nogil=True)
def step_eul_inplace(x, u, dt):
    dt_substep = dt/num_substeps
    dx = np.empty_like(x)
    for j in range(num_substeps):
        x_dot_out(0.0, x, u, dx)
        x += dt_substep*dx

@nb.njit(StateT(StateT, ActionT, TimeDeltaT), cache=True, nogil=True)
def step_eul(x0, u, dt):
    x = x0.copy()
    step_eul_inplace(x, u, dt)
    return x

@nb.njit(StateArrayT(StateT, ActionArrayT, TimeDeltaT), cache=True, nogil=True)
def step_array(x0, U, dt=1e-2):
    X = np.empty((U.shape[0] + 1, x0.shape[0]))
    X[0, :] = x0
    for i in range(U.shape[0]):
        X[i+1, :] = X[i, :]
        step_eul_inplace(X[i+1, :], U[i, :], dt)
    return X

@nb.njit(DerivativesT(StateT, ActionT, TimeDeltaT, *DerivativesT), cache=True, nogil=True)
def derivatives(x, u, dt, fx, fu):
    rp = rotor_positions
    t_a, t_b, t_c = thrust_coeffs
    x[3:7] /= np.linalg.norm(x[3:7])
    px, py, pz, qi, qj, qk, qr, vx, vy, vz, wx, wy, wz, rpm0, rpm1, rpm2, rpm3 = x
    rpms = u*rpm_max
    acc_bf = polyval(thrust_coeffs, rpms).sum()/mass
    thrust_derivs = 2*rpms*t_a + t_b

    for i in range(x.shape[0]):
        for j in range(x.shape[0]):
            if i == j:
                fx[i, j] = 1.0
            else:
                fx[i, j] = 0.0
        for j in range(u.shape[0]):
            fu[i, j] = 0.0

    # position wrt velocity
    fx[0:3, 7:10] = dt*np.eye(3)
    # attitude wrt attitude
    fx[3:7, 3:7] += dt*0.5*np.array(([0.0,  wz, -wy,  wx],
                                     [-wz, 0.0,  wx,  wy],
                                     [ wy, -wx, 0.0,  wz],
                                     [-wx, -wy, -wz, 0.0]))
    # attitude wrt angular velocity
    fx[3:7, 10:13] += dt*0.5*np.array(([ qr, -qk,  qj],
                                       [ qk,  qr, -qi],
                                       [-qj,  qi,  qr],
                                       [-qi, -qj, -qk]))
    # linear velocity wrt attitude
    fx[7:10, 3:7] += dt*2*acc_bf*np.array(([ qk,  qr,  qi,  qj],
                                           [-qr,  qk,  qj, -qi],
                                           [-qi, -qj,  qk,  qr]))

    # linear and angular velocity drag
    fx[7:10, 7:10]   -= dt*friction_force*np.eye(3)/mass
    fx[10:13, 10:13] -= dt*friction_torque*np.eye(3)/inertia
    # rpms
    fx[13:, 13:] -= dt*rotor_rpm_coeff*np.eye(num_rotors)

    fu[7, :] = dt*rpm_max*thrust_derivs/mass*2*(qi*qk + qj*qr)
    fu[8, :] = dt*rpm_max*thrust_derivs/mass*2*(qj*qk - qi*qr)
    fu[9, :] = dt*rpm_max*thrust_derivs/mass*(qk**2 + qr**2 - qi**2 - qj**2)
    fu[10, :] =  dt*rp[:, 1]*rpm_max*thrust_derivs/inertia
    fu[11, :] = -dt*rp[:, 0]*rpm_max*thrust_derivs/inertia
    fu[12, :] = dt*rotor_directions*z_torque_per_rpm*2*rpm_max**2*u/inertia
    fu[13:, :] = dt*rotor_rpm_coeff*rpm_max*np.eye(num_rotors)

    return fx, fu

@nb.njit(DerivativesT(StateT, ActionT, TimeDeltaT), cache=True, nogil=True)
def derivatives_num(x, u, dt):
    h = 1e-11/dt
    N = 13 + num_rotors
    K = num_rotors
    fx = np.empty((N, N))
    fu = np.empty((N, K))

    for i in range(N):
        x[i] -= h/2.0
        x_0 = x_dot(0.0, x, u)
        #x_0[3:7] /= np.linalg.norm(x_0[3:7])
        x[i] += h
        x_1 = x_dot(0.0, x, u)
        #x_1[3:7] /= np.linalg.norm(x_1[3:7])
        x[i] -= h/2.0
        fx[:, i]  = dt*(x_1 - x_0)/h
        fx[i, i] += 1.0

    for i in range(K):
        u[i] -= h/2.0
        x_0 = x_dot(0.0, x, u)
        u[i] += h
        x_1 = x_dot(0.0, x, u)
        u[i] -= h/2.0
        #f1[3:7] /= np.linalg.norm(f1[3:7])
        fu[:, i] = dt*(x_1 - x_0)/h

    return fx, fu

def test_derivs(dt=1/400):
    from models.mc import Quad
    m = Quad()
    N, = m.state_shape
    K, = m.action_shape

    def num_derivs(x, u, dt, h=1e-7):
        fx = np.zeros((N, N))
        fu = np.zeros((N, K))

        for i in range(N):
            x[i] -= h/2.0
            x_0 = x_dot(0.0, x, u)
            #x_0[3:7] /= np.linalg.norm(x_0[3:7])
            x[i] += h
            x_1 = x_dot(0.0, x, u)
            #x_1[3:7] /= np.linalg.norm(x_1[3:7])
            x[i] -= h/2.0
            fx[:, i]  = dt*(x_1 - x_0)/h
            fx[i, i] += 1.0

        for i in range(K):
            u[i] -= h/2.0
            x_0 = x_dot(0.0, x, u)
            u[i] += h
            x_1 = x_dot(0.0, x, u)
            u[i] -= h/2.0
            #f1[3:7] /= np.linalg.norm(f1[3:7])
            fu[:, i] = dt*(x_1 - x_0)/h

        return fx, fu, None, None, None, None

    def matcompare(A, B=None, check_cells=True, tolerance=1e-5,
                   row_names=None, col_names=None):
        if row_names is None:
            row_names = list(range(A.shape[0]))
        if col_names is None:
            col_names = list(range(A.shape[1]))
        sdiff = 0.0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                diff = A[i, j] - B[i, j]
                sdiff += diff**2
                if diff**2 > tolerance and check_cells:
                    print(f'∆{row_names[i]},{col_names[j]}:{diff:-11.5g}. '
                          f'A:{A[i,j]:-11.5g}. '
                          f'B:{B[i,j]:-11.5g}')
        print('mse:', sdiff/A.shape[0]/A.shape[1])

    px, py, pz, qi, qj, qk, qr, vx, vy, vz, wx, wy, wz, r0, r1, r2, r3 = range(N)
    state_dim_names = 'px py pz qi qj qk qr vx vy vz wx wy wz r0 r1 r2 r3'.split()
    action_dim_names = 'u0 u1 u2 u3'.split()
    for i in range(20):
        x = np.random.normal(0.0, 2e0, size=N)
        #x[3:6]=0.0
        x[qk:qr+1]*=5
        #x[qi:qr+1]=(0,0,0,1)
        x[qi:qr+1] /= np.linalg.norm(x[3:7])
        x = m.state()
        u = 0.5 + 0*m.sample_actions()[0]
        fx, fu = num_derivs(x.copy(), u.copy(), dt)
        gx, gu = np.empty_like(fx), np.empty_like(fu)
        derivatives(x.copy(), u.copy(), dt, gx, gu)
        print(m.state_rep(x))
        print('x:', x)
        print('u:', u)
        print('fx')
        matcompare(fx, gx, row_names=state_dim_names, col_names=state_dim_names)
        print('fu')
        matcompare(fu, gu, row_names=state_dim_names, col_names=action_dim_names)
        x_ = x + dt*x_dot(0.0, x, u)
        print(((fx@x + fu@u) - x_))
        print(((gx@x + gu@u) - x_))
        print(m.state_rep((fx - gx)@x))

    #x = model.sample_state()
    #u = np.random.uniform(0, 1, size=(n, *model.action_shape))
    #u = 0.5*np.ones((n, *model.action_shape))
    #matcompare(f, g)
    #raise SystemExit

if __name__ == "__main__":
    test_derivs()
