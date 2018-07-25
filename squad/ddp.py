import time
import logging
from collections import namedtuple

import numpy as np
import numba as nb

from . import px4, traj
from .dyn import mc
from .utis import clip, env_param, qf
#from models.pointmass import PointMass
model = mc.Quad()

TrajTuple = namedtuple('TrajTuple', 'states actions')

log = logging.getLogger(__name__)
DT = env_param('dt', default=1/400, cast=float)
Tl = env_param('ddp_lookahead', default=2000, cast=int)

@nb.njit(cache=True, nogil=True)
def inv_reg(A, λ):
    "Regularized inversion of A."
    # Method 1: plain old inversion.
    # Variant 1a: plain old inversion.
    #A_reg = A.copy()
    #for j in range(A.shape[0]):
    #    A_reg += λ
    #    A_reg += λ*A[j, j]
    #return np.linalg.inv(reg)

    # Method 2: eigendecomposition
    #eigvals, eigvecs = np.linalg.eig(A.astype(np.complex128))
    #eigvals[eigvals.real < 1e-9] = 0.0
    #eigvals[eigvals.imag != 0.0] = 0.0
    #eigvals  += λ
    #eigvalinv = np.diag(1.0/eigvals)
    #A_inv   = (eigvecs @ eigvalinv @ eigvecs.T).real
    #return A_inv

    # Method 3: SVD, by far faster than eigendecomposition
    U, s, V = np.linalg.svd(A)
    for j in range(A.shape[0]):
        s[j] = 1.0/(s[j] + λ)
        #s[j] = 1.0/(s[j] + λ**2/s[j] + eps)
    return U@(s*V)

@nb.jitclass([('Qf', nb.float64[:, ::1]),
              ('Q', nb.float64[:, ::1]),
              ('R', nb.float64[:, ::1])])
class QuadraticCost():
    "l(x, u) = x^T Q x + u^T R u"

    def __init__(self, Qf, Q, R):
        self.Qf = Qf
        self.Q = Q
        self.R = R

    def trajectory(self, X, U, x_goal):
        cost = 0.0
        for i in range(U.shape[0]):
            cost += self.cost(X[i], U[i], x_goal)
        cost += self.final(X[i+1], x_goal)
        return cost

    def cost(self, x, u, x_goal):
        err = x - x_goal
        return err.T @ self.Q @ err + u.T @ self.R @ u

    # lx  = np.empty(state_shape)
    # lu  = np.empty(action_shape)
    # lxx = np.empty(state_shape + state_shape)
    # lxu = np.empty(action_shape + state_shape)
    # luu = np.empty(action_shape + action_shape)
    # lux = np.empty(action_shape + state_shape)
    def derivatives(self, x, u, x_goal, lx, lu, lxx, lxu, luu, lux):
        #lx, lu, lxx, lxu, luu, lux = out
        err = x - x_goal
        np.dot(err.T, self.Q, out=lx)
        lx *= 2.0
        np.dot(u.T, self.R, out=lu)
        lu *= 2.0
        lxx[:, :] = self.Q
        lxx *= 2.0
        luu[:, :] = self.R
        luu *= 2.0
        lxu[:, :] = 0.0
        lux[:, :] = 0.0
        return lx, lu, lxx, lxu, luu, lux

    def final(self, x, x_goal):
        err = x - x_goal
        return err.T @ self.Qf @ err

    def final_derivatives(self, x, x_goal):
        err = x - x_goal
        lx = err.T @ self.Qf + err.T @ self.Qf.T
        lxx = self.Qf + self.Qf.T
        return lx, lxx

class NumericalDynamics():
    def __init__(self, model, h=1e-9):
        x_dot = model.x_dot
        N, K = *model.state_shape, *model.action_shape

        @nb.njit(nogil=True)
        def derivatives(x, u, dt):
            fx = np.empty((N, N))
            fu = np.empty((N, K))
            fxx = None  # np.zeros((N, N, N))
            fxu = None  # np.zeros((N, N, K))
            fuu = None  # np.zeros((N, K, K))
            fux = None  # np.zeros((N, K, N))

            for i in range(N):
                dx = np.zeros(N)
                dx[i] = h
                df0 = x_dot(0.0, x-dx, u)
                df1 = x_dot(0.0, x+dx, u)
                fx[:, i] = (dx + dt*(df1 - df0)/2.0)/h

            for i in range(K):
                du = np.zeros(K)
                du[i] = h
                df0 = x_dot(0.0, x, u-du)
                df1 = x_dot(0.0, x, u+du)
                fu[:, i] = dt*(df1 - df0)/2.0/h

            return fx, fu, fxx, fxu, fuu, fux

        self.derivatives = derivatives

def make(*, model, cost, x_goal, dt=DT, T=Tl, initial=None,
         dynamics='analytical'):

    step_eul = model.step_eul
    u_upper = model.u_upper
    u_lower = model.u_lower
    state_shape = model.state_shape
    action_shape = model.action_shape

    def initial_traj(x0, *, T):
        policy = initial()
        U = np.empty((T,   *model.action_shape))
        X = np.empty((T+1, *model.state_shape))
        X[0] = x0
        for i in range(T):
            U[i] = policy(X[i])
            U[i] = clip(U[i], model.u_lower, model.u_upper)
            X[i+1]  = X[i]
            X[i+1] += dt*model.x_dot(0.0, X[i], U[i])
        return TrajTuple(X, U)

    if dynamics == 'analytical':
        dyn = model.derivatives
    elif dynamics == 'numerical':
        dyn = NumericalDynamics(model).derivatives
    elif callable(dynamics):
        dyn = dynamics
    else:
        raise ValueError(dynamics)

    @nb.njit(cache=True, nogil=True)
    def _calc_derivatives(cost, X, U, x_goal, fx, fu, lx, lu, lxx, lxu, luu, lux):
        T = U.shape[0]
        V_x, V_xx = cost.final_derivatives(X[T], x_goal)
        for i in range(T):
            #fx[i], fu[i], _, _, _, _ = dyn(X[i], U[i], dt=dt)
            #lx[i], lu[i], lxx[i], lxu[i], luu[i], lux[i] = cost.derivatives(X[i], U[i], x_goal)
            dyn(X[i], U[i], dt, fx[i], fu[i])
            cost.derivatives(X[i], U[i], x_goal, lx[i], lu[i], lxx[i], lxu[i], luu[i], lux[i])
        return V_x, V_xx, fx, fu, lx, lu, lxx, lxu, luu, lux

    @nb.njit('f8[:](f8[:])', cache=True, nogil=True)
    def clip_action(u):
        for i in range(action_shape[0]):
            if u[i] < u_lower[i]:
                u[i] = u_lower[i]
            if u_upper[i] < u[i]:
                u[i] = u_upper[i]
        return u

    @nb.njit(cache=True, nogil=True)
    def forward_pass(K, k, X, U):
        X_new = np.empty_like(X)
        U_new = np.empty_like(U)
        X_new[0] = X[0]
        for i in range(U.shape[0]):
            U_new[i] = clip_action(U[i] + k[i] + K[i] @ (X_new[i] - X[i]))
            X_new[i+1] = step_eul(X_new[i], U_new[i], dt=dt)
            #_, X_new[i+1], _ = model.step(X_new[i], U_new[i], dt=dt)
        return X_new, U_new

    @nb.njit(cache=True, nogil=True)
    def backward_pass(K, k, V_x, V_xx, fx, fu, lx, lu, lxx, lxu, luu, lux, λ):
        T = fx.shape[0]
        Qx  = np.empty(state_shape)
        Qu  = np.empty(action_shape)
        Qxx = np.empty(state_shape + state_shape)
        Quu = np.empty(action_shape + action_shape)
        Qux = np.empty(action_shape + state_shape)
        V_x  = V_x.copy()
        V_xx = V_xx.copy()
        for i in range(T-1, -1, -1):
            # Calculate Qx, Qu, Qxx, Quu, Qxu.
            np.dot(fx[i].T, V_x, out=Qx); Qx += lx[i]
            np.dot(fu[i].T, V_x, out=Qu); Qu += lu[i]
            fxTV_xx = np.dot(fx[i].T, V_xx)
            fuTV_xx = np.dot(fu[i].T, V_xx)
            np.dot(fxTV_xx, fx[i], out=Qxx); Qxx += lxx[i]
            np.dot(fuTV_xx, fu[i], out=Quu); Quu += luu[i]
            np.dot(fuTV_xx, fx[i], out=Qux); Qux += lux[i]

            # Inversion of Quu. Very important stuff.
            neg_Quu_inv  = inv_reg(Quu, λ)
            neg_Quu_inv *= -1

            # Compute k, K, Vx, Vxx.
            np.dot(neg_Quu_inv, Qu,  out=k[i])
            np.dot(neg_Quu_inv, Qux, out=K[i])
            neg_KTQuu  = np.dot(K[i].T, Quu)
            neg_KTQuu *= -1
            np.dot(neg_KTQuu, k[i], out=V_x); V_x += Qx
            np.dot(neg_KTQuu, K[i], out=V_xx); V_xx += Qxx
        return K, k

    #@nb.njit(cache=True, nogil=True)
    def ddp(x0, x_goal, U, ln_λ=-5, ln_λ_max=55, λ_base=1.5,
            iter_max=2000, atol=5e-1, callback=None, return_controllers=False):
        t0     = time.time()
        T      = U.shape[0]
        X      = model.step_array(x0, U, dt=dt)
        k_new  = np.empty((T,) + action_shape)
        K_new  = np.empty((T,) + action_shape + state_shape)
        fx     = np.empty((T,) + state_shape + state_shape)
        fu     = np.empty((T,) + state_shape + action_shape)
        lx     = np.empty((T,) + state_shape)
        lu     = np.empty((T,) + action_shape)
        lxx    = np.empty((T,) + state_shape + state_shape)
        lxu    = np.empty((T,) + state_shape + action_shape)
        luu    = np.empty((T,) + action_shape + action_shape)
        lux    = np.empty((T,) + action_shape + state_shape)
        c      = cost.trajectory(X, U, x_goal)
        c_init = c
        derivs = _calc_derivatives(cost, X, U, x_goal, fx, fu, lx, lu, lxx, lxu, luu, lux)
        n_updt = 0

        if callback is not None:
            callback(U)

        ln_λ_start = ln_λ
        line_search_failed = True
        for i in range(iter_max):
            #if not all(np.all(np.isfinite(d)) for d in derivs):
            #    log.warn('non-finite derivatives:\n%s', derivs)
            try:
                backward_pass(K_new, k_new, *derivs, λ_base**ln_λ)
            except np.linalg.LinAlgError as e:
                #log.debug('rej, min(c): %.5g, ln λ: %d, overflow', c, ln_λ)
                ln_λ += 1
                continue
            X_new, U_new = forward_pass(K_new, k_new, X, U)
            c_new = cost.trajectory(X_new, U_new, x_goal)
            change = c - c_new
            if change > atol or (change > 0 and n_updt == 0):
                log.debug('acc, min(c): %.5g, c: %.5g, ln λ: %d, change %.3g', c, c_new, ln_λ, change)
                c, X, U, K, k = c_new, X_new, U_new, K_new.copy(), k_new.copy()
                derivs = _calc_derivatives(cost, X, U, x_goal, fx, fu, lx, lu, lxx, lxu, luu, lux)
                n_updt += 1
                ln_λ -= 1
                line_search_failed = False
                if callback is not None:
                    callback(U)
            else:
                #log.debug('rej, min(c): %.5g, c: %.5g, ln λ: %d, change %.3g', c, c_new, ln_λ, change)
                ln_λ += 1
                if ln_λ > ln_λ_max:
                    if line_search_failed:
                        break
                    log.debug('resetting λ')
                    ln_λ = ln_λ_start
                    line_search_failed = True

        t1 = time.time()
        log.info('ddp done. %2d upd %3d iters %5.3g s, c_init: %.5g, c: %.5g, %%ch: %.5g',
                 n_updt, i, t1 - t0, c_init, c, (c_init - c)/abs(c_init))
        trj = TrajTuple(X, U)

        if return_controllers:
            return trj, K, k
        else:
            return trj

    def solver(x0, *, initial=None, **kw):
        assert ((initial is None) ^ (T is None)) or (initial.shape[0] == T)
        if initial is None:
            _, initial = initial_traj(x0, T=T)
        return ddp(x0, x_goal, initial, **kw)

    solver.initial_traj = initial_traj
    solver.forward_pass = forward_pass
    return solver

def make_quad(**kw):
    ipx, ipy, ipz, iqi, iqj, iqk, iqr, ivx, ivy, ivz, iwx, iwy, iwz, irpm = range(14)
    Qf = np.zeros(model.state_shape)
    Qf[ipx] = Qf[ipy] = 4e1
    Qf[ipz]           = 5e1
    Qf[ivx:ivz+1] = 2e1
    Qf[iwx:iwz+1] = 5e0

    # RPMs should be low since rpm^2 is the power usage. Bit too hard to
    # optimize; better to just minimize control signal.
    #Qf[irpm:] = 1e-5*np.eye(model.num_rotors)*model.thrust_coeffs[0]

    # Here L is the sum of projections of body axes onto inertial frame, plus
    # extra for Z.
    q_axis_x, q_axis_y, q_axis_z = 6e0, 6e0, 1e1
    Qf[iqi] = -q_axis_x + q_axis_y + q_axis_z
    Qf[iqj] = +q_axis_x - q_axis_y + q_axis_z
    Qf[iqk] = +q_axis_x + q_axis_y - q_axis_z
    Qf[iqr] = -q_axis_x - q_axis_y - q_axis_z

    Qf = np.diag(Qf)

    # r·v is larger for velocities going away from the origin.
    #Qf[ipx, ivx] = Qf[ivx, ipx] = 5e0
    #Qf[ipy, ivy] = Qf[ivy, ipy] = 5e0
    #Qf[ipz, ivz] = Qf[ivz, ipz] = 5e0

    #Q = np.zeros((*model.state_shape, *model.state_shape))
    #Q[0:3, 0:3] = 1e0*np.eye(3)
    Qf *= 1e-1
    Q = 5e-1*Qf.copy()

    #R = 0*np.eye(*model.action_shape)
    R = np.diag(1e-3*np.ones(*model.action_shape))

    # NOTE attq for the goal is zeroed!
    x_goal = 0.0*model.state(position=(0, 0, 0))

    cost = QuadraticCost(Qf=Qf, Q=Q, R=R)
    solver = make(model=model, cost=cost, initial=px4.Agent, x_goal=x_goal, **kw)
    solver.cost_matrices = Qf, Q, R
    return solver

def make_quad_policy(*, dt=DT, T=Tl, tolerance=2e0, **kw):
    solver = make_quad(dt=dt, T=T, **kw)
    Qf, Q, R = solver.cost_matrices
    X, U, Ks, ks, th, t = None, None, None, None, None, 0

    import threading
    lock = threading.Lock()

    def cb(x0, U):
        if hasattr(action, 'tracker'):
            action.tracker.set_trajectory(TrajTuple(model.step_array(x0, U, dt), U))

    def action(x0):
        nonlocal X, U, Ks, ks, th, t

        q = qf(Q, (X[t] - x0)) if X is not None else np.inf

        if q < tolerance:
            dists = np.linalg.norm(X[t:t+10, 0:3] - x0[0:3], axis=1)
            t = t + dists.argmin()

            # Re-run controllers from x_t
            with lock:
                X[t] = x0
                X[t:], U[t:] = solver.forward_pass(Ks[t:], 0*ks[t:], X[t:], U[t:])

                if X.shape[0] < 5:
                    X = None

            if th is None or not th.is_alive():
                #X, U, Ks, ks, t = X[t:], U[t:], Ks[t:], ks[t:], 0
                th = threading.Thread(target=update, args=(t+100,))
                th.start()

            return U[t]

        else:
            if X is not None:
                log.warn('re-optimizing trajectory mid-flight!')
            else:
                log.info('optimizing trajectory')

            # Throw away current update step, as it is irrelevant anyway.
            if th is not None:
                th.join()

            (X, U), Ks, ks = solver(x0, initial=None, callback=lambda U: cb(x0, U),
                                    atol=5e0, λ_base=3.0, ln_λ=-5, ln_λ_max=20,
                                    iter_max=50, return_controllers=True)
            t = 0
            return U[t]

    def update(n):
        "Update current trajectory estimate from point n."
        nonlocal X, U, Ks, ks

        # New set of actions U_ is an extension of U into the future.
        U_ = U
        if U_ is not None:
            U_ = U_[n:][:T]
            filler = np.repeat(U_[-1][None], T-U_.shape[0], axis=0)
            U_ = np.vstack((U_, filler))

        trj, Ks_, ks_ = solver(X[n], initial=U_, callback=lambda U: cb(X[n], U),
                               atol=5e0, λ_base=3.0, ln_λ=-5, ln_λ_max=20,
                               iter_max=50, return_controllers=True)

        # Safely update shared variables
        with lock:
            X = np.vstack((X[:n], trj.states))
            U = np.vstack((U[:n], trj.actions))
            Ks = np.vstack((Ks[:n], Ks_))
            ks = np.vstack((ks[:n], ks_))

    return action

def main(dt=DT, T=Tl, Tl=Tl, Tstep=Tl, n=10):
    import logcolor
    logcolor.basic_config(level=logging.DEBUG)

    solver = make_quad(dt=dt, T=T)

    initials = []
    optimizeds = []

    for i, x0 in enumerate(model.sample_states(n)):
        log.info('sample %d, x0:\n%s', i, model.state_rep(x0))

        states, actions = solver.initial_traj(x0, T=T)
        initials.append(traj.Trajectory(dt=dt, states=states.copy(), actions=actions.copy()))

        initial = solver.initial_traj(x0, T=Tl).actions
        for j in range(0, T, Tstep):
            k = j + Tstep
            states_j, actions_j = solver(states[j], initial=initial,
                                         atol=5e0, λ_base=3.0, ln_λ=-5,
                                         ln_λ_max=20, iter_max=2000)
            states[j:k+1], actions[j:k] = states_j[:Tstep+1], actions_j[:Tstep]
            #filler = np.zeros((Tstep, *model.action_shape))
            #filler = solver.initial_traj(states[k], T=Tstep).actions
            filler = np.repeat(actions_j[-1][None], Tstep, axis=0)
            #filler = np.random.uniform(u_lower, u_upper, size=(Tstep, *model.action_shape))
            initial = np.vstack((actions_j[Tstep:], filler))

        #states = model.step_array(x0, actions, dt=dt)
        log.info('x_final:\n%s', model.state_rep(states[-1]))
        optimizeds.append(traj.Trajectory(dt=dt, states=states, actions=actions))

    traj.save_many('pid_trajs.npz', initials)
    traj.save_many('ddp_trajs.npz', optimizeds)

if __name__ == "__main__":
    main()
