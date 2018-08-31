# State representation is 3D but limited to the XZ plane, where Z is the
# gravitational axis. The 13-tuple of state is
#   px py pz,
#   qi qj qk qr,
#   vx vy vz,
#   wx wy wz,
# where
#   py = vy = wx = wz = qi = qk = 0, and
#   u0 = u1, u2 = u3,
# meaning the only axis of rotation is the Y axis, and the only axes of
# movement are the XZ axes, and that only pitch is controllable.

import numpy as np
import numba as nb

from . import quadfast as qf
from .quadfast import ipy, ivy, iwx, iwz, iqi, iqk, iqr, eps

iwx00, iwx01 = np.arange(qf.num_rotors)[qf.rotor_positions[:, 0] > 0]
iwx10, iwx11 = np.arange(qf.num_rotors)[qf.rotor_positions[:, 0] < 0]

@nb.njit(getattr(qf.x_dot_out, 'nopython_signatures', None), cache=True, nogil=True)
def x_dot_out(t, x, u, x_):
    u[iwx01] = u[iwx00]
    u[iwx11] = u[iwx10]
    qf.x_dot_out(t, x, u, x_)
    x_[ivy] = x_[iwx] = x_[iwz] = x_[iqi] = x_[iqk] = 0.0
    x_[iqi:iqr+1] /= np.linalg.norm(x[iqi:iqr+1]) + eps

from .quadfast import (StateT, ActionT, StateArrayT, ActionArrayT, TimeDeltaT,
                       none, num_substeps)
## COPY&PASTE FROM quadfast.py ##

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
