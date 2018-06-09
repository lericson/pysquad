import numpy as np
import numba as nb
from scipy.stats import norm as normal, uniform

import stats
from utis import selfdot
from .base import ODEModel

class PointMass(ODEModel):
    """2nd order point mass system

    x' = Ax + Bu, A = (0 0 0 1 0 0  , B = (0 0 0
                       0 0 0 0 1 0         0 0 0
                       0 0 0 0 0 1         0 0 0
                       0 0 0 0 0 0         1 0 0
                       0 0 0 0 0 0         0 1 0
                       0 0 0 0 0 0)        0 0 1)
    """

    theta_shape = 2,
    u_scale = 1.0
    u_lower = 5.0*np.r_[-u_scale, -u_scale, -u_scale]
    u_upper = 5.0*np.r_[+u_scale, +u_scale, +u_scale]
    x_sample_mean = np.r_[0, 0, 0, 0, 0, 0]
    x_sample_std  = np.r_[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    state_dist = normal(x_sample_mean, x_sample_std)
    action_dist = uniform(u_lower, u_upper)
    gravity = np.r_[0.0, 0.0, 1.0]

    x_sample_lower = -1.0
    x_sample_upper = +1.0

    def __init__(self, *args, **kwds):
        super().__init__(*args, **kwds)
        self.step_array = self._mksteparray()

    #              x  y  z vx vy vz
    A = np.array(((0, 0, 0, 1, 0, 0),
                  (0, 0, 0, 0, 1, 0),
                  (0, 0, 0, 0, 0, 1),
                  (0, 0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0, 0),
                  (0, 0, 0, 0, 0, 0)), dtype=np.float64)
    B = np.array(((0, 0, 0),
                  (0, 0, 0),
                  (0, 0, 0),
                  (1, 0, 0),
                  (0, 1, 0),
                  (0, 0, 1)), dtype=np.float64)/u_scale
    state_shape = A.shape[0],
    action_shape = B.shape[1],

    def rewards_cafvi(model, X, X_):
        c = np.r_[-1e4, 7.5e2, -1]
        a1, a2 = np.r_[14, 1e4]
        ε = 0.05
        return c @ (selfdot(X_[:, 0:3], axis=1),
                    0*a1*(np.linalg.norm(model.F(X_), axis=1) < ε),
                    a2*(X_[:, 2] < 0))

    def _rewards_2d_simple(model, X, X_, tol=1e-2):
        pos, vel = X_[:, 0:3], X_[:, 3:6]
        speed = np.linalg.norm(vel, axis=1)
        dist = np.linalg.norm(pos, axis=1)
        return 5.0*np.exp(-dist) + 1.0*np.exp(-speed) - 1.0
        #return (5.0 - np.clip(dist, 0, 5.0))**2 + (1.0 - np.clip(speed, 0, 1.0))**2

    def _rewards_2d_screwy(model, X, X_, tol=1e-2):
        # Cosine similarity of position and velocity is positive going away
        # from the origin, negative going towards the origin and zero
        # orthogonally.
        pos, vel = X_[:, 0:3], X_[:, 3:6]
        speed = np.linalg.norm(vel, axis=1)
        dist = np.linalg.norm(pos, axis=1)
        cos_sim = np.sum(vel*pos, axis=1)/speed/dist
        r = ((dist > tol)*-1 +        # Punish BEING away from origin
             (dist > tol)*-cos_sim +  # Punish GOING away from origin
             (dist < tol)*5 +         # Reward being CLOSE to origin
             np.fmin(50, (dist < tol)*1/speed))   # Reward inverse velocities at goal
        return r

    _rewards_2d = _rewards_2d_simple

    def F(model, x):
        X = np.atleast_2d(x)
        F = np.c_[selfdot(X[:, 0:3], axis=1),
                  selfdot(X[:, 3:6], axis=1)]
        if x.ndim == 1:
            F = F[0, :]
        return F

    def x_dot(m, t, x, u):
        if np.any(u < m.u_lower) or np.any(m.u_upper < u):
            raise ValueError(u)
        return m.A@x + m.B@(u/m.u_scale - m.gravity)

    def x_jac(model, t, x, u):
        return model.A
    use_x_jac = False

    @stats.measure_func
    def step_analytic(m, x, u, t=0, dt=0.1):
        x = x.copy()
        acc = np.clip(u, m.u_lower, m.u_upper)/m.u_scale - m.gravity
        x_ = x + np.r_[x[3:6]*dt + acc*dt*dt/2, acc*dt]
        r = m.rewards(x, x_, dt=dt)
        return t + dt, x_, r

    step = step_analytic

    def _mksteparray(m):
        u_scale = m.u_scale
        u_lower = m.u_lower
        u_upper = m.u_upper
        gravity = m.gravity

        @nb.njit('f8[:, :](f8[:], f8[:, :], f8)', nogil=True, cache=True)
        def step_array(x0, u, dt=0.1):
            x = np.empty((u.shape[0]+1, x0.shape[0]))
            x[0, 0:6] = x0
            for i in range(u.shape[0]):
                ui = np.array([min(max(u[i, 0], u_lower[0]), u_upper[0]),
                               min(max(u[i, 1], u_lower[1]), u_upper[1]),
                               min(max(u[i, 2], u_lower[2]), u_upper[2])])
                acc = ui/u_scale - gravity
                x[i+1, 0:6]  = x[i, 0:6]
                x[i+1, 0:3] += dt*(x[i, 3:6] + acc*dt/2)
                x[i+1, 3:6] += dt*acc
            return x

        return step_array

    def state(self, position=np.r_[0.0, 0.0, 0.0], velocity=np.r_[0.0, 0.0, 0.0]):
        return np.r_[position, velocity]

def p_ctrl(*, model=None, kpp=1.0, kpv=1.0):
    model = PointMass() if model is None else model
    u_scale = getattr(model, 'u_scale', 1.0)
    def policy(x):
        vel_sp = -kpp*x[0:3]
        acc_sp = kpv*(vel_sp - x[3:6])
        u = u_scale*(acc_sp + model.gravity)
        return np.clip(u, model.u_lower, model.u_upper)
    return policy
