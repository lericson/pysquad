import logging
from itertools import takewhile
from contextlib import contextmanager

import numpy as np
from squad import tpool, nblm
from squad.traj import Trajectory


log = logging.getLogger(__name__)
nblm.start()


class Expert():
    def actions(self, states):
        raise NotImplementedError


class QuadLQRExpert(Expert):
    atol = 5e0
    lambda_base = 3.0
    ln_lambda_init = 0
    ln_lambda_max = 15
    num_iter_max = 1000
    num_samples = 5
    action_sigma2 = 1e-2
    advice_stop = 200
    advice_stride = 4
    opt_stop = None
    opt_stride = 10
    opt_show_full_trajectory = True
    warn_cost_var = 1e4
    rerun_previous = False

    def __init__(self, *, tracker=None, num_threads=tpool.NUM_THREADS, **kw):
        from squad import ddp
        self._solver = ddp.make_quad(**kw)
        self._costf = self._solver.cost
        self.model = self._solver.model
        self.num_threads = num_threads
        self.tracker = tracker

    def unroll(self, states):
        sol_traj_sets = tpool.map(self._opt_ivp_track, states,
                                  num_threads=self.num_threads)
        if self.tracker is not None:
            self.tracker.clear()
        return [trj for sol_trajs in sol_traj_sets for trj in sol_trajs]

    def annotate_cost(self, trajs):
        for trj in trajs:
            trj.cost = self._costf.trajectory(trj.states, trj.actions,
                                              self._solver.x_goal)

    def _pred(self, state):
        state_cost = state.T @ self._solver.cost.Q @ state
        return state_cost < 2e2

    def advice(self, trajs):
        sol_states  = np.empty((0, *self.model.state_shape))
        sol_actions = np.empty((0, *self.model.action_shape))
        for trj in trajs:
            log.info('optimizing from initial state:\n%s',
                     self.model.state_rep(trj.states[0]))
            with self._display_trajectory(trj):
                states = trj.states[:self.opt_stop:self.opt_stride]
                states = takewhile(self._pred, states)
                sol_traj_sets = tpool.map(self._opt_ivp_track, states,
                                          break_on_error=True,
                                          num_threads=self.num_threads)
            for sol_trajs in sol_traj_sets:
                stop, step = self.advice_stop, self.advice_stride
                sol_states  = np.vstack((sol_states,  *(sol_trj.states[:stop:step]  for sol_trj in sol_trajs)))
                sol_actions = np.vstack((sol_actions, *(sol_trj.actions[:stop:step] for sol_trj in sol_trajs)))
            del sol_traj_sets
        return sol_states, sol_actions

    @contextmanager
    def _display_trajectory(self, trj):
        if self.tracker is not None:
            self.tracker.set_trajectory(trj)
        yield
        if self.tracker is not None:
            self.tracker.clear()

    def _opt_ivp_track(self, *args, **kwds):
        trk = self.tracker.alt_lru() if self.tracker is not None else None
        cb = lambda X, U: self._cb(X, U, tracker=trk)
        return self._opt_ivp(*args, callback=cb, **kwds)

    def _opt_ivp(self, x0, **kwds):
        kwds = dict({'atol': self.atol,
                     'λ_base': self.lambda_base,
                     'ln_λ': self.ln_lambda_init,
                     'ln_λ''_max': self.ln_lambda_max,
                     'iter_max': self.num_iter_max},
                    **kwds)

        sol = self._solver(x0, **kwds)
        trajs = sol.sample(n=self.num_samples,
                           sigma2=self.action_sigma2)

        if self.tracker:
            for trj in trajs:
                self.tracker.alt_lru().set_trajectory(trj)

        # Make sure returned trajectories are similar to the solution
        cost_vars = [(trj.cost - sol.cost)**2 for trj in trajs]
        if any(cost_var > self.warn_cost_var for cost_var in cost_vars):
            log.error('cost variance high for some trajectories, likely '
                      'due to sampled trajectory deviating too far from '
                      'the reference trajectory, costs: %s',
                      [trj.cost for trj in trajs])

        trajs = [trj for trj, cost_var in zip(trajs, cost_vars)
                 if cost_var < self.warn_cost_var]
        cost_vars = [c for c in cost_vars if c < self.warn_cost_var]

        if not trajs:
            raise ValueError('sampled trajectories deviate from reference')

        log.info('sampled %d trajectories, cost variance: %.3g',
                 len(trajs), np.mean(cost_vars))

        return trajs

    def _cb(self, X, U, tracker=None):
        if tracker is None:
            return
        if not self.opt_show_full_trajectory:
            X = X[:self.advice_ahead_stop:self.advice_ahead_stride]
            U = U[:self.advice_ahead_stop:self.advice_ahead_stride][:-1]
        trj = Trajectory(dt=self._solver.dt, states=X, actions=U)
        tracker.set_trajectory(trj)


class QuadLQRExpertOnline(QuadLQRExpert):
    def advice(self, trajs):
        states, actions = [], []
        for trj in trajs:
            sol, sol_trajs = self._opt_ivp_track(trj.states[0],
                                                 initial=trj.actions)
            for sol_trj in sol_trajs:
                states.extend(sol_trj.states[:-1][:self.advice_ahead_stop:self.advice_ahead_stride])
                actions.extend(sol_trj.actions[:self.advice_ahead_stop:self.advice_ahead_stride])
        if self.tracker is not None:
            self.tracker.clear()
        return states, actions
