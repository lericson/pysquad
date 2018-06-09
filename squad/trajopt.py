"Optimize trajectory using Monte Carlo sampling"

import sys
import time
import logging
import threading
from datetime import datetime

import logcolor
import numpy as np
import numba as nb
from scipy import optimize

from . import px4, ddp, traj, tpool
from .ddp import TrajTuple
from .dyn import mc
from .utis import env_param, qf
model = mc.Quad()
init_policy = px4.Agent

Xnorm = np.eye(*model.state_shape)
Xnorm[13:, 13:] = 0.0

log = logging.getLogger(__name__)

NUM_LOOKAHEAD = env_param('traj_length', default=2000, cast=int)
DYNAMICS = env_param('dynamics', default='analytical', cast=str)
METHOD = env_param('method', default='ddp', cast=str)
DT = env_param('dt', default=1/400., cast=float)

OPTIMIZER_OPTIONS = {'tnc': dict(maxiter=200, disp=False),
                     #'l-bfgs-b': dict(maxfun=100000, ftol=5e-2, maxls=20, disp=False),
                     'l-bfgs-b': dict(disp=True),
                     'slsqp': dict(maxiter=1000),
                     'de': dict(maxiter=1000, disp=False),
                     'basinhopping': dict(T=100.0, niter=200, niter_success=20)}

_cost_compile_lock = threading.Lock()

def make(*, model=model, n=NUM_LOOKAHEAD, dt=DT,
         callback_part=None, callback_opt=None, expensive_initial_guess=False,
         dynamics=DYNAMICS, method=METHOD, single_pass=True, **opt_kws):

    opt_opts = dict(OPTIMIZER_OPTIONS.get(method.lower(), {}), **opt_kws)
    bound_l = np.repeat(model.u_lower[None, :], n, axis=0)
    bound_u = np.repeat(model.u_upper[None, :], n, axis=0)
    bounds = np.c_[bound_l.flatten(), bound_u.flatten()]

    if expensive_initial_guess:
        def initial(x0, *, n, initial=None):
            return ddp_solve(x0, T=n, dt=dt, initial=initial, noise=False)
    else:
        def initial(x0, *, n=n):
            policy = init_policy()
            #model.steps(x0, policy, steps=10, dt=dt)
            t, X, U, R = model.steps(x0, policy, steps=n, dt=dt)
            return TrajTuple(X, U)

    ddp_solve = ddp.load(dynamics=dynamics)

    state_shape = model.state_shape
    action_shape = model.action_shape
    step_array = model.step_array

    # Lock circumvents race condition in numba eager compilation.
    with _cost_compile_lock:
        log.debug('compiling cost function')
        Qf, Q, R = ddp.cost_matrices()
        x_goal = 0.0*model.state()  # attq needs to be zero

        @nb.njit('f8(f8[::1], f8[::1])', cache=True, nogil=True)
        def cost(u_, x0):
            n = u_.shape[0] // action_shape[0]
            U = u_.reshape((n,) + action_shape)
            costf = ddp.QuadraticCost(Qf, Q, R)
            X = step_array(x0, U, dt=dt)
            return costf.trajectory(X, U, x_goal)

        @nb.njit('f8[::1](f8[::1], f8[::1])', cache=True, nogil=True)
        def num_grad(u_, x):
            h = 1e-12
            grad = np.empty_like(u_)
            c0 = cost(u_, x)
            for i in range(grad.shape[0]):
                u_[i] += h
                c1 = cost(u_, x)
                u_[i] -= h
                grad[i] = (c1 - c0)/h
            return grad

        dyn = model.derivatives

        @nb.njit('f8[::1](f8[::1], f8[::1])', cache=True, nogil=True)
        def analyt_grad(U_, x0):
            n = U_.shape[0] // action_shape[0]
            U = U_.reshape((n,) + action_shape)
            costf = ddp.QuadraticCost(Qf, Q, R)
            grad = np.zeros_like(U)
            X = step_array(x0, U, dt=dt)
            fx  = np.empty(state_shape + state_shape)
            fu  = np.empty(state_shape + action_shape)
            lx  = np.empty(state_shape)
            lu  = np.empty(action_shape)
            lxx = np.empty(state_shape + state_shape)
            lxu = np.empty(action_shape + state_shape)
            luu = np.empty(action_shape + action_shape)
            lux = np.empty(action_shape + state_shape)
            # TODO Use out= parameter of derivative functions to save memory
            for i in range(n):
                costf.derivatives(X[i], U[i], x_goal, lx, lu, lxx, lxu, luu, lux)
                dyn(X[i], U[i], dt, fx, fu)
                grad[i] = lu
                t = fu
                for j in range(i+1, n):
                    costf.derivatives(X[j], U[j], x_goal, lx, lu, lxx, lxu, luu, lux)
                    dyn(X[j], U[j], dt, fx, fu)
                    grad[i] += lx@t
                    t = fx@t
                lx, _ = costf.final_derivatives(X[n], x_goal)
                grad[i] += lx@t

            #print('mean gradient error num:   ', np.mean((num_grad(U_, x0)    - grad.flatten())**2))
            #print('mean gradient error analyt:', np.mean((analyt_grad(U_, x0) - grad.flatten())**2))

            return grad.flatten()

    def matcompare(A, B, check_cells=False, tolerance=1e-5):
        print(A-B)
        #print(B)
        sdiff = 0.0
        for i in range(A.shape[0]):
            for j in range(A.shape[1]):
                diff = A[i, j] - B[i, j]
                sdiff += diff**2
                if diff**2 > tolerance and check_cells:
                    print(f'∆{i},{j}: {diff:.5g}. Aij: {A[i,j]:.5g}. Bij: {B[i,j]:.5g}')
        print('mse:', sdiff/A.shape[0]/A.shape[1])

    #x0 = model.sample_state()
    #u = np.random.uniform(model.u_lower, model.u_upper, size=(n, *model.action_shape))
    #log.info('num_grad calculation')
    #f = num_grad(u.flatten(), x0).reshape(u.shape)
    #log.info('analyt_grad calculation')
    #g = analyt_grad(u.flatten(), x0).reshape(u.shape)
    #log.info('grad_err calculation')
    #graderr = np.mean((f-g)**2)
    #if graderr > 1.0:
    #    log.warn('numeric and analytic gradient do not match, '
    #             'error: %.5g', graderr)

    jac = {'analytical': analyt_grad, 'numerical': num_grad}[dynamics]

    # scipy.optimize interface to the DDP minimizer
    def ddp_minimizer(cost, u, args, *a, callback=None, **kw):
        n = u.shape[0] // action_shape[0]
        U = u.reshape((n,) + action_shape)
        (x0,) = args
        try:
            X, U = ddp_solve(x0, dt=dt, callback=callback, initial=U)
        except:
            log.exception('ddp solve failed')
        return optimize.OptimizeResult(x=U.flatten(), fun=cost(U.flatten(), x0), success=True)

    def mcddp(x0, u_init, callback):
        "Monte-Carlo DDP"
        rtol = 1e-2
        niter_same = 0
        niter_same_max = 2
        c_best = cost(u_init.flatten(), x0)
        u_best = u_init
        niter_max = 10
        for i in range(niter_max):
            #log.info('mcddp adding noise')
            std = 1e-5/dt*np.sqrt(model.u_upper - model.u_lower)
            u_i = u_best.copy()
            u_i = np.random.normal(u_best, std, size=u_i.shape)
            u_i = np.clip(u_i, model.u_lower, model.u_upper)
            #log.info('mcddp solving')
            _, u_i = ddp_solve(x0, initial=u_i, dt=dt, callback=callback)
            #log.info('mcddp final state:\n%s', model.state_rep(step_array(x0, u_i, dt=dt)[-1]))
            c = cost(u_i.flatten(), x0)
            #log.info('mcddp final cost: %.5g', c)
            niter_same += 1
            if c < c_best:
                #log.info('mcddp improved best solution, i: %d, c: %.5g, '
                #         '%%ch: %.5g, std: %s', i, c, (c_best-c)/abs(c_best), std)
                if (c_best - c)/abs(c_best) > rtol:
                    niter_same = 0
                c_best, u_best = c, u_i.copy()
                if callback_opt:
                    callback_opt(x0, u_best)
            if niter_same >= niter_same_max:
                break
        return optimize.OptimizeResult(success=True, x=u_best)

    def gd(x0, u, callback=None):
        u_new = np.empty_like(u)
        n = u.shape[0] // action_shape[0]
        for i in range(500):
            print('calculating gradient', end='...', flush=True)
            df = jac(u, x0)
            c_prev = c_best = cost(u, x0)
            st_exp = -30
            print('done')
            print('line searching step size', end='...', flush=True)
            for j in range(100):
                u_new[:] = u
                u_new -= df*(2**st_exp)
                c_new = cost(u_new, x0)
                if c_new < c_best:
                    print('stexp', st_exp, 'c', c_new, '%', (c_best - c_new)/c_best)
                    u[:] = u_new
                    c_best = c_new
                    st_exp += 1
                    callback(u)
                else:
                    st_exp -= 1
                    if st_exp < -45:
                        break
            if c_prev <= c_best or (i % 10) == 0:
                _, u_new = ddp_solve(x0, initial=u_new.reshape((n,) + action_shape), dt=dt, noise=True, callback=callback)
                u_new = u_new.flatten()
                c_new = cost(u_new, x0)
                if c_new < c_best:
                    u[:] = u_new
                    c_best = c_new
                    print('ddp improved', c_new, '%', (c_best - c_new)/c_best)
                else:
                    print('ddp rejected', c_new, '%', (c_best - c_new)/c_best)
            if c_prev <= c_best:
                break
        return u

    def minimize_cost(x0, u_init, callback_opt=callback_opt):
        if callback_opt:
            callback_opt(x0, u_init)

        if method == 'DE':
            cb = (lambda uk, convergence=None: callback_opt(x0, uk)) if callback_opt else None
            result = optimize.differential_evolution(cost, args=(x0,), callback=cb, bounds=bounds, **opt_opts)

        elif method == 'basinhopping':
            cb = (lambda uk, f=None, accept=True: callback_opt(x0, uk)) if callback_opt else None
            result = optimize.basinhopping(cost, u_init.flatten(),
                                           minimizer_kwargs={'method': 'L-BFGS-B',
                                                             'bounds': bounds,
                                                             'jac': jac,
                                                             'callback': cb,
                                                             'args': (x0,)},
                                           #minimizer_kwargs={'method': ddp_minimizer,
                                           #                  'args': (x0,)},
                                           callback=cb, **opt_opts)
            result.success = True

        elif method == 'least_squares':
            result = optimize.least_squares(cost, u_init.flatten(), jac=jac, args=(x0,), bounds=list(bounds.T))

        elif method == 'mcddp':
            cb = (lambda uk: callback_opt(x0, uk)) if callback_opt else None
            result = mcddp(x0, u_init, callback=cb)

        elif method == 'ddp':
            cb = (lambda uk: callback_opt(x0, uk)) if callback_opt else None
            _, u_opt = ddp_solve(x0, initial=u_init, dt=dt, callback=cb, noise=False, rtol=1e-3, λ_base=3.0, ln_λ=-5, ln_λ_max=20, iter_max=80)
            _, u_opt = ddp_solve(x0, initial=u_opt,  dt=dt, callback=cb, noise=False, rtol=1e-3, λ_base=2.0, ln_λ=0,  ln_λ_max=30, iter_max=120)
            result = optimize.OptimizeResult(success=True, x=u_opt)

        elif method == 'gd':
            cb = (lambda uk: callback_opt(x0, uk)) if callback_opt else None
            u_opt = gd(x0, u_init.flatten(), callback=cb)
            result = optimize.OptimizeResult(success=True, x=u_opt)

        elif method == 'none':
            result = optimize.OptimizeResult(success=True, x=u_init)

        else:
            cb = (lambda uk: callback_opt(x0, uk)) if callback_opt else None
            result = optimize.minimize(cost, u_init.flatten(), jac=jac, args=(x0,),
                                       bounds=bounds, method=method,
                                       callback=cb, options=opt_opts)

        if not result.success:
            log.warn('trajectory optimization incomplete: %s', result.message)

        u_opt = result.x.reshape((n, *model.action_shape))
        if callback_opt:
            callback_opt(x0, u_opt)

        # We could use more high-fidelity simulation here but it becomes
        # confusing if the optimization finally returns with state estimates
        # that are not those optimized to.
        x_est = model.step_array(x0, u_opt, dt=dt)
        #_, x_est, u_opt, _ = model.steps(x0, policy=u_opt, dt=dt)

        return x_est, u_opt

    def solve_multi(x0, parts=30, reuse=15,
                    callback_part=callback_part, callback_opt=callback_opt):
        log.info('multi optimizing trajectory x0:\n%s', model.state_rep(x0))
        x_est = np.empty(((parts-1)*reuse+n+1, *model.state_shape))
        u_opt = np.empty(((parts-1)*reuse+n,   *model.action_shape))
        x_est[0] = x0
        for i in range(parts):
            t0 = time.time()
            start = i*reuse
            _, xi, _ = model.step_many(x=x0, policy=u_opt[:start], dt=dt)
            xε = qf(Xnorm, xi - x_est[start])
            if xε > 1e-2:
                log.warn('estimated state not very close to actual, xε: %.5g', xε)
            x_est[start:start+n+1], u_opt[start:start+n] = initial(xi, n=n)
            c_init = cost(u_opt[start:][:n].flatten(), xi)
            x_est_i, u_opt_i = minimize_cost(xi, u_init=u_opt[start:][:n],
                                             callback_opt=callback_opt)
            x_est[start:start+n+1] = x_est_i
            u_opt[start:start+n]   = u_opt_i
            c_opt  = cost(u_opt[start:][:n].flatten(), xi)
            log.info('%s done part %d. %5.3g s, c_init: %.5g, c_opt: %.5g, %%ch: %.5g',
                     method, i, time.time() - t0, c_init, c_opt, (c_init - c_opt)/abs(c_init))
            if callback_part:
                callback_part(x0, u_opt[:start+n])
        log.info('multi optimization finished')
        return TrajTuple(x_est, u_opt)

    def solve(x0, callback_part=callback_part, callback_opt=callback_opt):
        t0 = time.time()
        x_est, u_init = initial(x0, n=n)
        c_init = cost(u_init.flatten(), x0)
        x_est, u_opt = minimize_cost(x0, u_init=u_init,
                                     callback_opt=callback_opt)
        c_opt = cost(u_opt.flatten(), x0)
        log.info('%s done. %5.3g s, c_init: %.5g, c_opt: %.5g, %%ch: %.5g, f10r: %.5g',
                 method, time.time() - t0, c_init, c_opt, (c_init - c_opt)/abs(c_init),
                 np.sum((u_init[:10] - u_opt[:10])**2))
        if callback_part:
            callback_part(x0, u_opt)
        return TrajTuple(x_est, u_opt)

    return solve if single_pass else solve_multi

def _solve_one(x0, solve, kwargs={}):
    base_fname = f'traj_{datetime.now():%Y%m%d_%H%M%S}'
    # Optimized trajectory
    Xopt, Uopt = solve(x0, **kwargs)
    # Reference trajectory from safe policy
    _, Xpid, Upid, Rpid = model.steps(x0, init_policy(), steps=Uopt.shape[0], dt=DT)
    trajs = [traj.from_parts(dt=DT, states=Xopt, actions=Uopt),
             traj.from_parts(dt=DT, states=Xpid, actions=Upid, rewards=Rpid)]
    traj.save_many(f'{base_fname}.npz', trajs)

def main(args=sys.argv[1:]):
    logcolor.basic_config(level=logging.DEBUG)
    if ''.join(args[:1]) == 'cli':
        cli()
    else:
        gui()

def cli():
    solve = make(dt=DT)
    x0 = np.r_[0.1, 0.1, 0.0,
               0.0, 0.0, 0.0, 1.0,
               4.0, 4.0, 0.0,
               0.0, 0.0, 0.0,
               0.0, 0.0, 0.0, 0.0]
    x0s = np.vstack((x0, model.sample_states(20)))
    for x0 in x0s:
        _solve_one(x0, solve)

def gui():
    #solve(model.sample_states(1)[0, :])
    x0s = np.array([model.state(position=(0.1, 0.1, 0.1), velocity=(4.0, 4.0, 0.4)),
                    model.state(position=(-0.1, -0.1, 0.0), velocity=(0.1, 0.1, 0.0))])

    if model.state_shape == (6,):
        x0s = np.hstack((x0s[:, 0:3], x0s[:, 7:10]))

    from . import traj
    viewer = traj.Viewer()
    x0s = np.vstack((x0s, model.sample_states(20)))
    #x0s = model.sample_states(20)

    class TrajLines():
        def __init__(self, viewer, x0, opt_line_color='#fff', part_line_color='g'):
            self.viewer = viewer
            self.lines_part = viewer.plot_traj([(0.0, x0, None, None, None)])
            self.lines_opt = viewer.plot_traj([(0.0, x0, None, None, None)])
            self.opt_line_color = opt_line_color
            self.part_line_color = part_line_color

        def _update(self, lines, x0, U_, **kw):
            U = U_.reshape((-1,) + model.action_shape)
            #t, X, U, R = model.steps(x0, U, dt=DT)
            X = model.step_array(x0, U, dt=DT)
            history = traj.from_parts(states=X, actions=U, dt=DT)
            lines.setData(**self.viewer.plot_traj_kwds(history, **kw))

        def callback_part(self, x0, u):
            self.lines_opt.pos = None
            self._update(self.lines_part, x0, u, width=2.0, color=self.part_line_color)

        def callback_opt(self, x0, u, finished=False):
            self._update(self.lines_opt, x0, u, width=0.5, color=self.opt_line_color)

    ts = []

    #jobs = []
    #solve = make(dynamics='numerical', dt=DT)
    #for x0 in x0s:
    #    lines = TrajLines(viewer, x0)
    #    kwargs = dict(callback_part=lines.callback_part,
    #                  callback_opt=lines.callback_opt)
    #    jobs.append((x0, solve, kwargs))
    #ts += tpool.run_async(_solve_one, jobs)

    jobs = []
    solve = make(dt=DT)
    for x0 in x0s:
        lines = TrajLines(viewer, x0, opt_line_color='#e05', part_line_color='b')
        kwargs = dict(callback_part=lines.callback_part,
                      callback_opt=lines.callback_opt)
        jobs.append((x0, solve, kwargs))
    ts += tpool.run_async(_solve_one, jobs)

    viewer.show()
    try:
        tpool.wait(ts)
    except KeyboardInterrupt:
        sys.exit(1)

if __name__ == "__main__":
    main()
