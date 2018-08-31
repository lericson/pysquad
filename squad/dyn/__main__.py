import sys
import logging
from threading import Thread
from functools import wraps

import logcolor
import numpy as np

from .. import traj, plots
from ..utis import env_param, import_name, forward_exceptions

log = logging.getLogger(__name__)

class ReturnThread(Thread):
    def run(self):
        try:
            self._return = self._target(*self._args, **self._kwargs)
        finally:
            # Avoid a refcycle if the thread is running a function with
            # an argument that has a member that points to the thread.
            del self._target, self._args, self._kwargs

    def join(self):
        super().join()
        return self._return

def threadwrap(f, autostart=True, **opts):
    @wraps(f)
    def inner(*a, **kw):
        t = ReturnThread(target=f, args=a, kwargs=kw, **opts)
        if autostart:
            t.start()
        return t
    return inner

def main(args=sys.argv[1:]):
    logcolor.basic_config(level=logging.INFO)
    model_spec, policy_spec, *init_state = args
    tracker = traj.Tracker()
    model = import_name(model_spec)()
    policy = import_name(policy_spec)()
    num_samples = env_param('num_samples', default=1, cast=int)
    try:
        policy.tracker = tracker.alt(color='g')
    except Exception as e:
        log.warn('could not set tracker attribute: %s', e)
    if init_state:
        vals = list(map(eval, init_state))
        x0s = np.array(vals, dtype=np.float64)
        x0s = x0s.reshape((num_samples, *model.state_shape))
    else:
        x0s = model.sample_states(num_samples)
    log.info('initial states:\n%s', model.state_rep(x0s))
    dt = env_param('dt', default=1/400, cast=float)
    policy_noise = lambda x: np.random.normal(policy(x), 1e-2)

    thread_unroll = threadwrap(model.unroll)

    ts = [thread_unroll(policy=policy_noise, x0=x0, dt=dt,
                        callback=tracker.alt().set_history,
                        t_min=env_param('t_min', default=1.0, cast=float),
                        t_max=env_param('t_max', default=60.0, cast=float),
                        q_min=env_param('q_min', default=1e-4, cast=float))
          for x0 in x0s]

    tracker.autopan = (len(ts) == 1)

    tracker.show()
    with forward_exceptions(*ts):
        for t in ts:
            x, history = t.join()
            log.info('final state:\n%s', model.state_rep(x))

    trj = traj.Trajectory.from_history(history)
    trj.save('.last.npz')

if __name__ == "__main__":
    main()
