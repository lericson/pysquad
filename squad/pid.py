"Fast but crappy PID control!"

import numpy as np
import numba as nb

from .utis import clip_norm, clip_abs

@nb.jitclass([('coeffs', nb.float64[:, ::1]),
              ('integral', nb.float64[:]),
              ('prev', nb.float64[:]),
              ('max_abs', nb.float64[::1]),
              ('max_abs', nb.float64[::1]),
              ('max_norm', nb.float64),
              ('max_integral_abs', nb.float64[::1]),
              ('max_integral_norm', nb.float64)])
class Controller():
    def __init__(ctrl, kp, ki=None, kd=None,
                 max_abs=np.inf, max_integral_abs=np.inf,
                 max_norm=np.inf, max_integral_norm=np.inf):
        ctrl.coeffs = np.vstack((kp, ki, kd))
        ctrl.integral = np.zeros_like(kp)
        ctrl.prev = np.zeros_like(kp)
        ctrl.max_abs = max_abs
        ctrl.max_norm = max_norm
        ctrl.max_integral_abs = max_integral_abs
        ctrl.max_integral_norm = max_integral_norm

    @property
    def state(ctrl):
        return np.vstack((ctrl.integral, ctrl.prev))

    @state.setter
    def state(ctrl, state):
        ctrl.integral = state[:ctrl.integral.shape[0]]
        ctrl.prev = state[ctrl.integral.shape[0]:]

    def reset(ctrl):
        ctrl.integral *= 0.0
        ctrl.prev *= 0.0

    def feed(ctrl, sp, pv):
        err = sp - pv
        ctrl.integral[:] += err
        if ctrl.max_integral_norm < np.inf:
            ctrl.integral[:] = clip_norm(ctrl.integral, ctrl.max_integral_norm)
        if np.any(ctrl.max_integral_abs < np.inf):
            ctrl.integral[:] = clip_abs(ctrl.integral, ctrl.max_integral_abs)
        diff = err - ctrl.prev
        ctrl.prev[:] = err
        #signal = np.diag(ctrl.coeffs.T @ ((err, ctrl.integral, diff)))
        signal  = ctrl.coeffs[0, :] * err
        signal += ctrl.coeffs[1, :] * ctrl.integral
        signal += ctrl.coeffs[2, :] * diff
        if ctrl.max_norm < np.inf:
            signal = clip_norm(signal, ctrl.max_norm)
        if np.any(ctrl.max_abs < np.inf):
            signal = clip_abs(signal, ctrl.max_abs)
        return signal

def new(kp, ki=None, kd=None,
        max_abs=np.inf, max_integral_abs=np.inf,
        max_norm=np.inf, max_integral_norm=np.inf):
    kp = np.atleast_1d(kp)
    ki = np.zeros_like(kp) if ki is None else np.atleast_1d(ki)
    kd = np.zeros_like(kp) if kd is None else np.atleast_1d(kd)
    max_abs = np.atleast_1d(max_abs)
    max_integral_abs = np.atleast_1d(max_integral_abs)
    return Controller(kp, ki, kd, max_abs, max_integral_abs, max_norm, max_integral_norm)
