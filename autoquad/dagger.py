import sys
import logging

import numpy as np
from squad.utis import env_param, str2bool, forward_exceptions
from squad.dyn import mc

from .learners import TwoLayerPerceptron
from .experts import QuadLQRExpert

log = logging.getLogger(__name__)


class configs:
    class base:
        #: Simulation time step.
        dt = 1/400

        #: Learner rollout length when optimizing and evaluating. Note that
        # this doesn't necessarily mean the expert has to optimize all time
        # steps.
        T = 1000

        #: DDP expert trajectory length for use during optimization.
        T_ddp = 1000

        #: Number of learner rollouts to optimize each iteration.
        num_learner = 2

        #: Number of epochs for learner to fit current dataset
        num_epochs = 10

        #: Number of DAgger iterations
        num_iters = 1000

        #: Learner evaluation is done every Nth iteration
        eval_interval = 1

        #: Number of learner rollouts during evaluation.
        num_eval = 500

        #: Limit of aggregated dataset size.
        max_aggregated = 1_000_000

        expert_class = QuadLQRExpert
        learner_class = TwoLayerPerceptron

    class full(base):
        num_initial = 50
        num_validation = 30
        two_dimensional = env_param('two_d', default=False, cast=str2bool)
        model_class = mc.Quad2D if two_dimensional else mc.Quad
        show_tracker = env_param('show_tracker', default=False, cast=str2bool)

    class quick(base):
        num_initial = 4
        num_validation = 2
        num_eval = 100
        two_dimensional = env_param('two_d', default=True, cast=str2bool)
        model_class = mc.Quad2D if two_dimensional else mc.Quad
        show_tracker = env_param('show_tracker', default=True, cast=str2bool)


config = getattr(configs, env_param('config', default='full', cast=str))


def DaggerThread(*, config, tracker=None):
    log.info('initializing model class %s', config.model_class)
    model = config.model_class()

    log.info('initializing expert class %s', config.expert_class)
    expert = config.expert_class(model=model, tracker=tracker,
                                 T=config.T_ddp, dt=config.dt)

    log.info('initializing learner class %s', config.learner_class)
    #learner = PolynomialLeastSquares(model=model, T=T, dt=dt)
    #learner = MultiLayerPerceptron(num_units=(13*1+1, 64, 4), order=1,
    #                               model=model, T=T, dt=dt)
    learner = config.learner_class(state_shape=(13,), model=model,
                                   T=config.T, dt=config.dt)
    # Cut away last four dimensions, the RPMs
    learner.proj_matrix = np.eye(17)[:, :13]

    def run():

        log.info('sampling %d initial trajectories', config.num_initial)
        trajs = expert.unroll(model.sample_states(config.num_initial))

        #: States and actions in aggregated training dataset
        states_agg  = np.asarray([xt for trj in trajs[:-config.num_validation] for xt in trj.states[:-1]])
        actions_agg = np.asarray([ut for trj in trajs[:-config.num_validation] for ut in trj.actions])

        #: States and actions in validation dataset
        states_val  = np.asarray([xt for trj in trajs[config.num_validation:] for xt in trj.states[:-1]])
        actions_val = np.asarray([ut for trj in trajs[config.num_validation:] for ut in trj.actions])

        #: Previous epoch's validation loss
        val_prev = 0.0

        for i in range(config.num_iters):
            log.info('#%d fitting %d transitions', i, len(states_agg))
            for j in range(config.num_epochs):
                with learner.transaction() as rollback:
                    loss = learner.fit_epoch(states_agg, actions_agg, epoch=j)
                    val  = learner.validate(states_val, actions_val)
                    val_chg     = val - val_prev
                    val_chg_rel = val_chg/val_prev if val_prev != 0.0 else 0.0
                    log.info('epoch #%d.%d training loss: %.3g', i, j, loss)
                    log.info('epoch #%d.%d val loss: %.3g%% (%.3g, chg: %.3g)',
                             i, j, 100*val_chg_rel, val, val_chg)
                    if val_chg_rel > 5e-2 and j > 0:
                        rollback()
                        log.warn('epoch #%d.%d update rejected', i, j)
                        #break
                        val = val_prev
                    val_prev = val

            if (max(1, i) % config.eval_interval) == 0:
                log.info('#%d evaluating on %d trajectories', i, config.num_eval)
                trajs = learner.unroll(model.sample_states(config.num_eval))
                end_states = np.array([trj.states[-1] for trj in trajs])
                log.info('mean end state:\n%s', model.state_rep(end_states.mean(axis=0)))
                log.info('std end state:\n%s', model.state_rep(end_states.std(axis=0)))

                if tracker:
                    for trj in trajs[-len(tracker.alts):]:
                        tracker.alt_lru().set_trajectory(trj)

                expert.annotate_cost(trajs)
                with open('costs.csv', 'a') as fileobj:
                    costs = [trj.cost for trj in trajs]
                    print('\t'.join(map(str, (i, *costs))), file=fileobj)

                log.info('mean cost: %.3g', np.mean([trj.cost for trj in trajs]))
                log.info('std cost: %.3g', np.std([trj.cost for trj in trajs]))
                log.info('#%d using %d worst trajectories in evaluation',
                         i, config.num_learner)
                trajs.sort(key=lambda trj: -trj.cost)
                trajs = trajs[:config.num_learner]

            else:
                log.info('#%d sampling %d learner trajectories', i, config.num_learner)
                #log.info('#%d theta:\n%s', i, learner.theta)
                trajs = learner.unroll(model.sample_states(config.num_learner))

            log.info('#%d expert solving %d trajectories', i, len(trajs))
            states, actions = expert.advice(trajs)
            log.info('#%d expert gave %d samples', i, len(states))
            states_agg  = np.vstack((states_agg, states))[-config.max_aggregated:]
            actions_agg = np.vstack((actions_agg, actions))[-config.max_aggregated:]

            if True:
                import numba.runtime as nrt
                sys._debugmallocstats()
                st = nrt.rtsys.get_allocation_stats()
                print('NRT allocations:', st.alloc - st.free, file=sys.stderr)

    import threading
    return threading.Thread(target=run)

def main(*, config=config):
    import logcolor
    logcolor.basic_config(level=logging.INFO)
    log.info('loading configuration %r', config.__name__)

    # This trickery runs DAgger in a separate thread, and makes sure the thread
    # dies with the main thread. This is simply because the Qt GUI library must
    # run in the main thread on at least macOS, due to Apple engineering - and
    # who knows, maybe they have good reasons.
    if config.show_tracker:
        from squad.traj import Tracker
        tracker = Tracker(plot_opts=dict(color='ff0000'), autopan=False,
                          elevation=0, azimuth=-90, distance=100.0)
        t = DaggerThread(config=config, tracker=tracker)
        t.daemon = True
        t.start()
        try:
            with forward_exceptions(t):
                tracker.show()
                if t.is_alive():
                    raise KeyboardInterrupt
        except KeyboardInterrupt:
            log.warn('keyboard interrupt, quitting')
    else:
        t = DaggerThread(config=config)
        t.run()

if __name__ == "__main__":
    main()
