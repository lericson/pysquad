import numpy as np
import numba as nb
from . import trans

dot = trans.quaternion_multiply
inv = trans.quaternion_inverse
conj = trans.quaternion_conjugate
from_axis_angle = trans.quaternion_about_axis

@nb.njit('f8[:](f8[:, :])', cache=True, nogil=True)
def from_rotmat(R):
    q = np.empty(4, dtype=np.float64)
    t = np.trace(R) + 1
    if t > 1:
        q[3] = t
        q[2] = R[1, 0] - R[0, 1]
        q[1] = R[0, 2] - R[2, 0]
        q[0] = R[2, 1] - R[1, 2]
    else:
        i, j, k = 0, 1, 2
        if R[1, 1] > R[0, 0]:
            i, j, k = 1, 2, 0
        if R[2, 2] > R[i, i]:
            i, j, k = 2, 0, 1
        t = R[i, i] - (R[j, j] + R[k, k]) + 1
        q[i] = t
        q[j] = R[i, j] + R[j, i]
        q[k] = R[k, i] + R[i, k]
        q[3] = R[k, j] - R[j, k]
    q *= 0.5 / np.sqrt(t)
    return q

@nb.njit
def vecdot(q, v):
    qv = from_vec(v)
    return dot(dot(q, qv), inv(q))[:-1]

@nb.njit
def from_vec(v):
    qv = np.zeros(4)
    qv[0:3] = v
    return qv

rotmat = trans.quaternion_matrix

euler_rpy = trans.euler_from_quaternion
from_rpy = trans.quaternion_from_euler

assert np.allclose(vecdot(from_axis_angle(np.pi, (1, 0, 0)), (1, 0, 0)), (1, 0, 0))
assert np.allclose(vecdot(from_axis_angle(np.pi, (1, 0, 0)), (0, 1, 0)), (0, -1, 0))
