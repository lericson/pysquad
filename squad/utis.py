import os
from warnings import warn
from collections import namedtuple

import numpy as np
import numba as nb

def import_name(name):
    modname, attname = name.rsplit('.', 1)
    return getattr(__import__(modname, fromlist=[attname]), attname)

@nb.njit('float64[:](float64[:], float64)', cache=True, nogil=True)
def clip_norm(v, max_norm):
    v_norm = np.linalg.norm(v)
    if v_norm == 0.0:
        return v
    return v*np.fmin(max_norm/v_norm, 1.0)

_typeme = np.empty((0,), dtype=np.float64)
_typeme.flags.writeable = False
ro_float64 = nb.typeof(_typeme)
@nb.njit([nb.float64[:](nb.float64[:], ro_float64, ro_float64)], cache=True, nogil=True)
def clip(a, a_min, a_max):
    out = a.copy()
    for i in range(a.shape[0]):
        if a[i] < a_min[i]:
            out[i] = a_min[i]
        elif a[i] > a_max[i]:
            out[i] = a_max[i]
    return out

@nb.njit('f8[:](f8[:], f8, f8)', cache=True, nogil=True)
def clip_all(a, a_min, a_max):
    out = a.copy()
    for i in range(a.shape[0]):
        if a[i] < a_min:
            out[i] = a_min
        elif a[i] > a_max:
            out[i] = a_max
    return out

@nb.njit('f8(f8, f8, f8)', cache=True, nogil=True)
def clip_one(a, a_min, a_max):
    if a < a_min:
        return a_min
    elif a > a_max:
        return a_max
    return a

@nb.njit(cache=True, nogil=True)
def clip_abs(v, absv):
    return clip(v, -absv, +absv)

def R(theta):
    return np.array([(+np.cos(theta), -np.sin(theta)),
                     (+np.sin(theta), +np.cos(theta))])

@nb.njit(cache=True, nogil=True)
def selfdot(v, axis=-1):
    return (v**2).sum(axis=axis)

@nb.njit(cache=True, nogil=True)
def polyval(coeffs, x):
    y = np.zeros_like(x)
    n = coeffs.shape[0]
    for i in range(n):
        y += coeffs[i]*x**(n-i-1)
    return y

@nb.njit(cache=True, nogil=True)
def cross(xs, ys):
    zs = np.empty_like(xs)
    zs[0] = xs[1]*ys[2] - ys[1]*xs[2]
    zs[1] = xs[2]*ys[0] - ys[2]*xs[0]
    zs[2] = xs[0]*ys[1] - ys[0]*xs[1]
    return zs

def env_param(key, *, default, empty_is_default=True, cast=str):
    """Get *key* as parameter from environment.

    *default* can be any value. If *empty_is_default*, it means an empty
    environment variable is interpreted as if it wasn't set at all. *cast* is a
    function that is applied to any value retrieved from the environment (i.e.
    not applied for *default*.)
    """
    value = os.environ.get(key, '')
    if key not in os.environ or (empty_is_default and not value.strip()):
        if key in {k.lower() for k in os.environ if k != key}:
            warn(f'environment variable {key!r} is unset, '
                 f'but other casing exists. is it a mistake?')
        return default
    return cast(os.environ[key].strip())

def str2bool(value):
    value = str(value).lower()
    if value.isdigit():
        return bool(int(value, 10))
    elif value in {'y', 'yes', 'f', 'true', 'on'}:
        return True
    elif value in {'n', 'no', 't', 'false', 'off'}:
        return False
    raise ValueError(value)

@nb.njit(['f8(f8[:, ::], f8[:])'], nogil=True, cache=True)
def qf(Q, x):
    "Evaluate quadratic form $y = x^T Q x$."
    return x.T @ Q @ x

def assert_qf_isclose(Q, x, v):
    assert np.isclose(qf(Q, x), v), f'{x@Q@x} ≠ {v}, x:\n{x}'
