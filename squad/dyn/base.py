import logging

import numpy as np
try:
    from scipy.integrate import solve_ivp
except ImportError:
    solve_ivp = None

log = logging.getLogger(__name__)

class BaseModel():
    "Base of models, can be an adapted model or a real one."

    def sample_states(model, n=1):
        return model.state_dist.rvs((n,) + model.state_shape)

    def sample_state(model):
        return model.sample_states(1)[0]

    def sample_actions(model, n=1):
        return model.action_dist.rvs((n,) + model.action_shape)

    def unroll(model, *, policy, x0=None, q_min=1e-3, q_max=1e7,
               t_min=1.0, t_max=60.0, t=0.0, dt=1/400, interruptable=True,
               callback=None):
        log.debug('starting simulation')
        if x0 is not None:
            x = x0
        else:
            x = model.sample_states()[0, :]
            log.debug('sampled initial state %s', x)
        print_every = int(5/dt) if t_max >= 10 else int(t_max/4/dt)
        history, q = [], (q_min + q_max)/2.0
        try:
            while t < t_min or (q_min < q < q_max and t < t_max):
                u = np.clip(policy(x), model.u_lower, model.u_upper)
                t, x, r = model.step(x, u, t=t, dt=dt)
                history.append((t, x, None, u, r))
                if callback is not None:
                    callback(history)
                q = np.linalg.norm(x[0:3])
                if (len(history) % print_every) == 1:
                    log.info('simulation at % 6.3g s, q: %.5g', t, q)
        except KeyboardInterrupt:
            if not interruptable:
                raise
            log.warn('simulation interrupted by keyboard')
        else:
            log.info('terminated at % 6.3f s, q: %.5g', t, q)
        return x, history

    def step_many(model, *a, **kw):
        t, X, U, R = model.steps(*a, **kw)
        return t, X[-1], R.sum()

    def steps(model, x, policy, *, steps=None, **kw):
        steps = policy.shape[0] if steps is None else steps
        X = np.empty((steps+1, *model.state_shape))
        U = np.empty((steps, *model.action_shape))
        R = np.empty((steps,))
        t, X[0] = 0.0, x
        for i in range(steps):
            if callable(policy):
                U[i] = policy(X[i])
            elif hasattr(policy, 'ndim') and policy.ndim == 2:
                U[i] = policy[i]
            else:
                U[i] = policy
            t, X[i+1], R[i] = model.step(X[i], U[i], t=t, **kw)
        return t, X, U, R

    def rewards(model, x, x_, *, dt, **kw):
        X = np.atleast_2d(x)
        X_ = np.atleast_2d(x_)
        R = dt*model._rewards_2d(X, X_, **kw)
        if x.ndim < 2:
            return R.reshape(x.shape[:-1])
        return R

    def state_rep(self, x):
        return str(np.asarray(x))

class ODEModel(BaseModel):
    "A model with dynamics in the form of ODEs."
    ivp_solver_opts = {}

    def step_ivp(m, x, u, t=0, dt=0.1):
        use_x_jac = hasattr(m, 'x_jac') and getattr(m, 'use_x_jac', True)
        kw = m.ivp_solver_opts.copy()
        if use_x_jac:
            kw['jac'] = lambda t, x: m.x_jac(t, x, u)
        result = solve_ivp(lambda t, x: m.x_dot(t, x, u),
                           (t, t + dt), x.copy(), **kw)
        if not result.success:
            raise RuntimeError('ivp solve failed: {}'.format(result.message))
        t_, x_ = result.t[-1], result.y[:, -1]
        r = m.rewards(x, x_, dt=t_ - t)
        return t_, x_, r

    def step_ivp_euler(m, x, u, t=0, dt=0.1):
        x_ = x + dt*m.x_dot(t, x, u)
        r = m.rewards(x, x_, dt=dt)
        return t + dt, x_, r

class AdaptedModel(BaseModel):
    "An adaptation of another model given as a constructor argument."

    def __init__(self, model):
        self.model = model

    def __getattr__(self, attname):
        return getattr(self.model, attname)
