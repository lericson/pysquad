import threading
from itertools import chain
from .utis import env_param

NUM_THREADS = env_param('num_threads', default=4, cast=int)

def worker(l, q, f):
    while True:
        try:
            with l:
                args = next(q)
        except StopIteration:
            break
        f(*args)

def run(*a, **kw):
    ts = run_async(*a, **kw)
    wait(ts)

def run_async(f, *qs, num_threads=NUM_THREADS):
    ts = []
    l = threading.Lock()
    q = chain(*qs)
    for i in range(num_threads):
        th = threading.Thread(target=worker, args=(l, q, f), daemon=True)
        th.start()
        ts.append(th)
    return ts

def wait(ts):
    for th in ts:
        th.join()
