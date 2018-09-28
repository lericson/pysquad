import time
import logging
import threading

import numpy as np
import numba as nb

from . import px4, traj
from .dyn import mc
from .utis import clip, env_param, qf
#from models.pointmass import PointMass

class Solution(traj.Trajectory):
    def sample(self, *, n, sigma2):
        # Future work: sample trajectories simultaneously by replacing the
        # state and action vectors by matrices.
        return [self._sample_one(sigma2) for i in range(n)]

    def rerun(self, *, state=None, start=0):
        "Re-run controllers from time step *start*."
        if state is not None:
            self.states[start] = state
        return self._sample_one(start=start)

    def _sample_one(self, sigma2=0.0, start=0):
        X_new = np.empty_like(self.states)
        U_new = np.empty_like(self.actions)
        X_new[start] = self.states[start]
        for i in range(start, self.actions.shape[0]):
            xi, ui = self.states[i], self.actions[i]
            Ki, ki = self.controllers[i]
            ui_new = ui + ki + Ki @ (X_new[i] - xi)
            U_new[i] = self._clip_action(ui_new)
            if sigma2 > 0.0:
                ui_new = np.random.normal(ui_new, sigma2)
            ui_new = self._clip_action(ui_new)
            X_new[i+1] = self.model.step_eul(X_new[i], ui_new, dt=self.dt)
            #_, X_new[i+1], _ = model.step(X_new[i], U_new[i], dt=dt)
        trj = traj.Trajectory(dt=self.dt, states=X_new, actions=U_new)
        trj.cost = self._costf.trajectory(X_new, U_new, self._x_goal)
        return trj

    def _sample_one_old(self, sigma2=0.0, start=0):
        X_new = np.empty_like(self.states)
        U_new = np.empty_like(self.actions)
        X_new[0] = self.states[0]
        for i in range(self.actions.shape[0]):
            Ki, ki = self.controllers[i]
            U_new[i] = self.actions[i] + ki + Ki @ (X_new[i] - self.states[i])
            if sigma2 > 0.0:
                U_new[i] = np.random.normal(U_new[i], sigma2)
            U_new[i] = self._clip_action(U_new[i])
            X_new[i+1] = self.model.step_eul(X_new[i], U_new[i], dt=self.dt)
            #_, X_new[i+1], _ = model.step(X_new[i], U_new[i], dt=dt)
        return traj.Trajectory(dt=self.dt, states=X_new, actions=U_new)

    def _clip_action(self, u):
        return np.clip(u, self.model.u_lower, self.model.u_upper)

log = logging.getLogger(__name__)
DT = env_param('dt', default=1/400, cast=float)
Tl = env_param('ddp_lookahead', default=2000, cast=int)

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
            clip_action(U[i])
            X[i+1] = step_eul(X[i], U[i], dt=dt)
        return traj.Trajectory(dt=dt, states=X, actions=U)

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

    @nb.njit('none(f8[:])', cache=True, nogil=True)
    def clip_action(u):
        for i in range(action_shape[0]):
            if u[i] < u_lower[i]:
                u[i] = u_lower[i]
            if u_upper[i] < u[i]:
                u[i] = u_upper[i]

    @nb.njit(cache=True, nogil=True)
    def forward_pass(K, k, X, U, α, X_new, U_new):
        X_new[0] = X[0]
        for i in range(U.shape[0]):
            U_new[i]  = U[i]
            U_new[i] += α*k[i]
            U_new[i] += α*K[i] @ (X_new[i] - X[i])
            clip_action(U_new[i])
            X_new[i+1] = step_eul(X_new[i], U_new[i], dt=dt)
            #_, X_new[i+1], _ = model.step(X_new[i], U_new[i], dt=dt)

    @nb.njit(cache=True, nogil=True)
    def is_finite(a):
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if not np.isfinite(a[i, j]):
                    return False
        return True

    @nb.njit(cache=True, nogil=True)
    def backward_pass(K, k, V_x, V_xx, fx, fu, lx, lu, lxx, lxu, luu, lux,
                      Qx, Qu, Qxx, Quu, Qux, λ):
        T = fx.shape[0]
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

            # Manually check for infinities, because if this happens inside of
            # the Numba wrapper of numpy.linalg.svd, then memory is leaked.
            if not is_finite(Quu):
                raise np.linalg.LinAlgError('Quu has become infinite')

            # Inversion of Quu. Very important stuff.
            U, s, V = np.linalg.svd(Quu, full_matrices=False)
            for j in range(Quu.shape[0]):
                s[j] = 1.0/(s[j] + λ)
                #s[j] = 1.0/(s[j] + λ**2/s[j] + eps)
            neg_Quu_inv = -U@(s*V)

            # Compute k, K, Vx, Vxx.
            np.dot(neg_Quu_inv, Qu,  out=k[i])
            np.dot(neg_Quu_inv, Qux, out=K[i])
            neg_KTQuu  = np.dot(K[i].T, Quu)
            neg_KTQuu *= -1
            np.dot(neg_KTQuu, k[i], out=V_x); V_x += Qx
            np.dot(neg_KTQuu, K[i], out=V_xx); V_xx += Qxx

    #@nb.njit(cache=True, nogil=True)
    def ddp(x0, x_goal, U, ln_λ=-5, ln_λ_max=55, λ_base=1.5, num_α=5,
            iter_max=1000, atol=5e-1, callback=None, require_convergence=True):
        t0     = time.time()
        T      = U.shape[0]
        X      = model.step_array(x0, U, dt=dt)
        X_new  = np.empty_like(X)
        U_new  = np.empty_like(U)
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
        Qx     = np.empty(state_shape)
        Qu     = np.empty(action_shape)
        Qxx    = np.empty(state_shape + state_shape)
        Quu    = np.empty(action_shape + action_shape)
        Qux    = np.empty(action_shape + state_shape)
        c      = cost.trajectory(X, U, x_goal)
        c_init = c
        derivs = _calc_derivatives(cost, X, U, x_goal, fx, fu, lx, lu, lxx, lxu, luu, lux)
        n_updt = 0

        if callback is not None:
            callback(X, U)

        ln_λ_start = ln_λ
        λ_found = False

        for i in range(iter_max):

            try:
                backward_pass(K_new, k_new, *derivs, Qx, Qu, Qxx, Quu, Qux,
                              λ_base**ln_λ)
            except np.linalg.LinAlgError as e:
                #log.debug('rej, min(c): %.5g, ln λ: %d, overflow', c, ln_λ)
                ln_λ += 1
                continue

            for α in np.linspace(1.0, 0.0, num_α, endpoint=False):

                forward_pass(K_new, k_new, X, U, α, X_new, U_new)
                c_new = cost.trajectory(X_new, U_new, x_goal)
                change = c - c_new

                if change > atol or (change > 0 and n_updt == 0):
                    log.debug('acc, min(c): %.5g, c: %.5g, ln λ: %d, α: %.2f'
                              ' change %.3g', c, c_new, ln_λ, α, change)
                    X_ref, U_ref = X, U
                    c = c_new
                    X, U = X_new.copy(), U_new.copy()
                    K, k = α*K_new.copy(), α*k_new.copy()
                    derivs = _calc_derivatives(cost, X, U, x_goal, fx, fu, lx, lu, lxx, lxu, luu, lux)
                    n_updt += 1
                    ln_λ -= 1
                    λ_found = True
                    if callback is not None:
                        callback(X, U)
                    break

            else:
                #log.debug('rej, min(c): %.5g, c: %.5g, ln λ: %d, change %.3g', c, c_new, ln_λ, change)
                ln_λ += 1
                if ln_λ > ln_λ_max:
                    if not λ_found:
                        break
                    log.debug('resetting λ')
                    ln_λ = ln_λ_start
                    λ_found = False

        else:
            if require_convergence:
                raise ValueError(f'no convergence in {i} iterations')

        if n_updt == 0:
            raise ValueError('line search for regularization term failed')

        t1 = time.time()
        log.info('ddp done. %2d upd %3d iters %5.3g s, c_init: %.5g, c: %.5g, %%ch: %.5g',
                 n_updt, i, t1 - t0, c_init, c, (c_init - c)/abs(c_init))

        trj = Solution(dt=dt, states=X_ref, actions=U_ref)
        trj._costf = cost
        trj._x_goal = x_goal
        trj.cost = c
        trj.model = model
        trj.controllers = list(zip(K, k))
        return trj

    def solver(x0, *, initial=None, **kw):
        if initial is None:
            initial = initial_traj(x0, T=T).actions
        return ddp(x0, x_goal, initial, **kw)

    solver.initial_traj = initial_traj
    solver.x_goal = x_goal
    solver.dt = dt
    solver.T = T
    return solver

def make_quad(*, model=None, **kw):
    if model is None:
        model = mc.Quad()

    ipx, ipy, ipz, iqi, iqj, iqk, iqr, ivx, ivy, ivz, iwx, iwy, iwz, irpm = range(14)
    Qf = np.zeros(model.state_shape)
    Qf[ipx] = Qf[ipy] = 4e1
    Qf[ipz]           = 5e1
    Qf[ivx:ivz+1] = 1e1
    Qf[iwx:iwz+1] = 5e0

    # RPMs should be low since rpm^2 is the power usage. Bit too hard to
    # optimize; better to just minimize control signal.
    #Qf[irpm:] = 1e-5*np.eye(model.num_rotors)*model.thrust_coeffs[0]

    # Here L is the sum of projections of body axes onto inertial frame, plus
    # extra for Z.
    #q_axis_x, q_axis_y, q_axis_z = 3e0, 3e0, 5e0
    #Qf[iqi] = -q_axis_x + q_axis_y + q_axis_z
    #Qf[iqj] = +q_axis_x - q_axis_y + q_axis_z
    #Qf[iqk] = +q_axis_x + q_axis_y - q_axis_z
    #Qf[iqr] = -q_axis_x - q_axis_y - q_axis_z

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
    solver.cost = cost
    solver.model = model
    return solver

class LinearTodayPolicy():
    # How many timesteps into the future to start optimization. A too low value
    # means the optimizer will never finish before its result is needed. A too
    # high value means the result is so far into the future that it's not
    # certain the result will ever be useful. Ideally it should be just enough
    # that the result is done a little before it's used.
    num_optimize_ahead = 100

    # Force re-optimization of trajectory if only this many controllers left.
    # This isn't something that should happen, but if the optimization takes
    # too long, it might.
    num_controllers_min = 5

    # How many future states to measure distance to. Solely for optimization.
    max_time_jump = 10

    def __init__(self, *, tolerance, solver):
        self.solver = solver
        self.tolerance = tolerance
        self.lock = threading.Lock()
        _, self.Q, _ = solver.cost_matrices
        self.sol = None
        self.tracker = None
        self._update_thread = None
        self.t = 0

    def _cb(self, trj):
        if self.tracker is not None:
            self.tracker.set_trajectory(trj)

    def __call__(self, x0):
        q = qf(self.Q, (self.sol.states[self.t] - x0)) if self.sol is not None else np.inf

        if q < self.tolerance:
            return self._local_control(x0)
        else:
            return self._traj_opt(x0)

    def _local_control(self, x0):
        next_states = self.sol.states[self.t:self.t+self.max_time_jump]
        dists = np.linalg.norm(next_states[:, 0:3] - x0[0:3], axis=1)
        self.t += dists.argmin()

        # Re-run controllers from x_t. At a first glance, it might seem as if
        # we would only need to calculate the action for this time step;
        # however, if there have been deviations, our re-optimization needs to
        # start from a state that we will actually (probably) end up in, or
        # close to. Thus we recompute the remainder of the current trajectory.
        with self.lock:
            self.sol = self.sol.rerun(state=x0, start=self.t)

        action = self.sol.actions[self.t]

        if len(self.sol.controllers[self.t:]) < self.num_controllers_min:
            self.sol = None

        if self._update_thread is None or not self._update_thread.is_alive():
            self._update_thread = threading.Thread(target=self._update,
                                                   args=(self.t + self.num_optimize_ahead,))
            self._update_thread.start()

        return action

    def _traj_opt(self, x0):
        if self.sol is not None:
            log.warning('re-optimizing trajectory mid-flight!')
        else:
            log.info('optimizing trajectory')

        # Throw away current update step, as it is irrelevant anyway.
        if self._update_thread is not None and self._update_thread.is_alive():
            log.warning('waiting for current optimization to finish')
            self._update_thread.join()

        sol = self.solver(x0, initial=None, callback=self._cb,
                          atol=5e0, λ_base=3.0, ln_λ=-5,
                          ln_λ_max=20, iter_max=50)
        self.sol = sol
        self.t = 0
        return sol.actions[0]

    def _update(self, n):
        "Update current trajectory estimate from point n."
        # New set of actions U_ is an extension of U into the future.
        U_ = self.actions
        if U_ is not None:
            U_ = U_[n:][:self.solver.T]
            filler = np.repeat(U_[-1][None], self.solver.T-U_.shape[0], axis=0)
            U_ = np.vstack((U_, filler))

        sol = self.solver(self.states[n], initial=U_, callback=self._cb,
                          atol=5e0, λ_base=3.0, ln_λ=-5, ln_λ_max=20,
                          iter_max=50)

        with self.lock:
            self.sol = sol

def make_quad_policy(*, dt=DT, T=Tl, tolerance=2e0, **kw):
    solver = make_quad(dt=dt, T=T, **kw)
    return LinearTodayPolicy(solver=solver, tolerance=tolerance)

def main(dt=DT, T=Tl, Tl=Tl, Tstep=Tl, n=10):
    import logcolor
    logcolor.basic_config(level=logging.DEBUG)

    solver = make_quad(dt=dt, T=T)
    model = solver.model

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
