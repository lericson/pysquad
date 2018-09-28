import sys
import logging

import numpy as np
from squad import stats
from squad.utis import env_param, forward_exceptions

from . import configs


log = logging.getLogger(__name__)

default_config_name = env_param('config', default='full', cast=str)
default_config_cls = getattr(configs, default_config_name)


class RejectEpoch(Exception):
    pass


class Dagger():
    print_stats = True
    costs_csv_filename = 'costs.csv'
    costs_csv_delimiter = '\t'

    def __init__(self, config, tracker=None):
        log.info('initializing model class %s', config.model_class)
        model = config.model_class()

        log.info('initializing expert class %s', config.expert_class)
        expert = config.expert_class(model=model, tracker=tracker,
                                     T=config.T_ddp, dt=config.dt)

        log.info('initializing learner class %s', config.learner_class)
        learner = config.learner_class(state_shape=(13,), model=model,
                                       T=config.T, dt=config.dt)
        # Remove last four dimensions, the RPMs
        learner.proj_matrix = np.eye(17)[:, :13]

        self.config, self.model = config, model
        self.expert, self.learner = expert, learner
        self.tracker = tracker

    def _sample_initial_trajectories(self):
        config, model, expert = self.config, self.model, self.expert

        log.info('sampling %d initial trajectories', config.num_initial)
        trajs = expert.unroll(model.sample_states(self.config.num_initial))

        #: States and actions in aggregated training dataset
        self.states_agg  = np.asarray([xt for trj in trajs[:-config.num_validation] for xt in trj.states[:-1]])
        self.actions_agg = np.asarray([ut for trj in trajs[:-config.num_validation] for ut in trj.actions])

        #: States and actions in validation dataset
        self.states_val  = np.asarray([xt for trj in trajs[config.num_validation:] for xt in trj.states[:-1]])
        self.actions_val = np.asarray([ut for trj in trajs[config.num_validation:] for ut in trj.actions])

    def _fit(self):
        for j in range(self.config.num_epochs):
            with self.learner.transaction() as rollback:
                try:
                    self._fit_epoch(epoch=j)
                except RejectEpoch:
                    rollback()
                    log.warning('epoch parameters rolled back')

    def _fit_epoch(self, *, epoch):
        val_prev = self._val_prev
        loss = self.learner.fit_epoch(self.states_agg, self.actions_agg, epoch=epoch)
        val  = self.learner.validate(self.states_val, self.actions_val)
        val_chg     = val - val_prev
        val_chg_rel = val_chg/val_prev if not np.isnan(val_prev) else 0.0
        log.info('epoch %s training loss: %.3g', epoch, loss)
        log.info('epoch %d val loss: %.3g%% (%.3g, chg: %.3g)',
                 epoch, 100*val_chg_rel, val, val_chg)
        if val_chg_rel > 5e-2 and epoch > 0:
            raise RejectEpoch
        else:
            self._val_prev = val_prev = val

    def _evaluate(self):
        model, expert, learner = self.model, self.expert, self.learner

        trajs = learner.unroll(model.sample_states(self.config.num_eval))
        end_states = np.array([trj.states[-1] for trj in trajs])
        log.info('mean end state:\n%s', model.state_rep(end_states.mean(axis=0)))
        log.info('std end state:\n%s', model.state_rep(end_states.std(axis=0)))

        if self.tracker:
            for trj in trajs[-len(self.tracker.alts):]:
                self.tracker.alt_lru().set_trajectory(trj)

        expert.annotate_cost(trajs)

        trajs.sort(key=lambda trj: trj.cost)

        log.info('mean cost: %.3g', np.mean([trj.cost for trj in trajs]))
        log.info('std cost: %.3g', np.std([trj.cost for trj in trajs]))
        log.info('min cost: %.3g', trajs[0].cost)
        log.info('max cost: %.3g', trajs[-1].cost)

        return trajs

    def _sample_learner_trajs(self, *, trajs=None):
        config = self.config
        if config.sample_method == 'worst':
            log.info('sampling %d worst trajectories', config.num_learner)
            return trajs[-config.num_learner:]
        elif config.sample_method == 'best_and_worst':
            N_best = config.num_learner//2
            N_worst = config.num_learner - N_best
            log.info('sampling %d+%d best+worst trajectories', N_best, N_worst)
            return trajs[:N_best] + trajs[-N_worst:]
        elif config.sample_method == 'random':
            log.info('sampling %d random trajectories', config.num_learner)
            #log.info('#%d theta:\n%s', i, learner.theta)
            states = self.model.sample_states(config.num_learner)
            return self.learner.unroll(states)
        else:
            raise ValueError(config.sample_method)

    def run(self):
        config = self.config

        self._sample_initial_trajectories()

        #: Previous epoch's validation loss
        self._val_prev = np.nan

        for i in range(config.num_iters):
            log.info('#%d fitting %d samples', i, len(self.states_agg))
            self._fit()

            trajs = None
            if (max(1, i) % config.eval_interval) == 0:
                log.info('#%d evaluating on %d trajectories', i, config.num_eval)
                trajs = self._evaluate()
                with open(self.costs_csv_filename, 'a') as fileobj:
                    fields = map(str, (i, *[trj.cost for trj in trajs]))
                    print(self.costs_csv_delimiter.join(fields), file=fileobj)

            trajs = self._sample_learner_trajs(trajs=trajs)

            log.info('#%d expert solving %d trajectories', i, len(trajs))
            states, actions = self.expert.advice(trajs)

            log.info('expert returned %d samples', len(states))
            self.states_agg  = np.vstack((self.states_agg, states))[-config.max_aggregated:]
            self.actions_agg = np.vstack((self.actions_agg, actions))[-config.max_aggregated:]

            if self.print_stats:
                import numba.runtime as nrt
                sys._debugmallocstats()
                st = nrt.rtsys.get_allocation_stats()
                print('NRT allocations:', st.alloc - st.free, file=sys.stderr)
                stats.print_summary(file=sys.stderr)


def main(*, config=None):
    import logcolor
    logcolor.basic_config(level=logging.INFO)
    if config is None:
        config = default_config_cls()
    log.info('loading from config %r', config)

    # This trickery runs DAgger in a separate thread, and makes sure the thread
    # dies with the main thread. This is simply because the Qt GUI library must
    # run in the main thread on at least macOS, due to Apple engineering - and
    # who knows, maybe they have good reasons.
    if config.show_tracker:
        from squad.traj import Tracker
        from threading import Thread
        tracker = Tracker(plot_opts=dict(color='ff0000'), autopan=False,
                          elevation=0, azimuth=-90, distance=100.0)
        d = Dagger(config=config, tracker=tracker)
        t = Thread(target=d.run, daemon=True)
        t.start()
        try:
            with forward_exceptions(t):
                tracker.show()
                if t.is_alive():
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            log.warning('keyboard interrupt, quitting')
    else:
        d = Dagger(config=config)
        d.run()


if __name__ == "__main__":
    main()
