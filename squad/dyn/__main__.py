import sys
import logging
from threading import Thread
from functools import wraps
from contextlib import contextmanager

import logcolor
import numpy as np

from .. import traj, plots
from ..utis import env_param, import_name

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

@contextmanager
def mad(*thread_objs):
    "One for all, all for one"
    import ctypes
    try:
        yield
    except:
        exc = ctypes.py_object(sys.exc_info()[0])
        for t in thread_objs:
            tid = ctypes.c_long(t._ident)
            ret = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, exc)
            if ret != 1:
                log.warn('mad return for %s: %d', t, ret)
        raise

def main(args=sys.argv[1:]):
    logcolor.basic_config(level=logging.INFO)
    model_spec, policy_spec, *init_state = args
    tracker = traj.Tracker()
    model = import_name(model_spec)()
    policy = import_name(policy_spec)()
    try:
        policy.tracker = tracker.alt(color='g')
    except Exception as e:
        log.warn('could not set tracker attribute: %s', e)
    if init_state:
        x0 = np.array(list(map(eval, init_state)), dtype=np.float64)
    else:
        x0 = model.sample_states()[0, :]
    #x0 = np.r_[1, 1, 1, 1, 1, 1]
    log.info('initial state:\n%s', model.state_rep(x0))
    dt = env_param('dt', default=1/400, cast=float)
    policy_noise = lambda x: np.random.normal(policy(x), 1e-2)

    thread_unroll = threadwrap(model.unroll)
    t = thread_unroll(policy=policy_noise, x0=x0, dt=dt,
                      callback=tracker.set_history,
                      t_min=env_param('t_min', default=1.0, cast=float),
                      t_max=env_param('t_max', default=60.0, cast=float),
                      q_min=env_param('q_min', default=1e-4, cast=float))
    tracker.show()
    with mad(t):
        x, history = t.join()
    log.info('final state:\n%s', model.state_rep(x))

    trj = traj.Trajectory.from_history(history)
    trj.save('.last.npz')

if __name__ == "__main__":
    main()
