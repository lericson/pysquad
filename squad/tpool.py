"Simple thread pooling"

import logging
import threading
from itertools import chain
from collections import namedtuple
from multiprocessing import cpu_count

from .utis import env_param, forward_exceptions, raise_in_threads

NUM_THREADS = env_param('num_threads', default=cpu_count()-1, cast=int)

log = logging.getLogger(__name__)
R = namedtuple('R', 'ok num retval')


class Stop(Exception):
    pass

class WorkerSet():
    # If enabled, the workers will stop processing items if any job raises an
    # exception. Note that the workers stop after the current job is done,
    # unless preemption is enabled.
    stop_on_error = True

    # Preemptive stopping raises an exception inside the current job to stop it
    # more immediately, in a similar way to how KeyboardInterrupt will raise an
    # exception in the currently executing frame.
    preemptive_stop = True

    def __init__(self, func, *queues, num_threads=NUM_THREADS):
        self.func = func
        self._lock = threading.Lock()
        self._queue = enumerate(chain(*queues))
        self._workers = [threading.Thread(target=self._worker, daemon=True)
                         for i in range(num_threads)]
        self._results = []

    def start(self):
        for w in self._workers:
            w.start()

    def join(self):
        with forward_exceptions(*self._workers):
            for w in self._workers:
                w.join()

    def stop(self):
        self._queue = iter(())
        if self.preemptive_stop:
            raise_in_threads(Stop, *self._workers)

    def results(self):
        self._results.sort(key=lambda r: r.num)
        for r in self._results:
            if not r.ok:
                try:
                    raise r.retval
                except Stop:
                    pass
            else:
                yield r.retval

    def _worker(self):
        name = str(self._workers.index(threading.current_thread()))
        while True:
            try:
                with self._lock:
                    num, job = next(self._queue)
            except StopIteration:
                break
            try:
                retval = self.func(job)
            except Stop as e:
                self._results.append(R(False, num, e))
                log.warning('[%s] job stopped', name)
            except Exception as e:
                self._results.append(R(False, num, e))
                log.error('[%s] job failed: %s\nfunc: %s\njob: %s',
                          name, e, self.func, job)
                if self.stop_on_error:
                    self.stop()
            else:
                self._results.append(R(True, num, retval))

def run_async(f, *qs, num_threads=NUM_THREADS):
    ws = WorkerSet(f, *qs, num_threads=num_threads)
    ws.start()
    return ws

def wait(ws):
    ws.join()

def map(f, *qs, num_threads=NUM_THREADS, break_on_error=False):
    ws = WorkerSet(f, *qs, num_threads=NUM_THREADS)
    ws.start()
    ws.join()
    L = []
    try:
        L.extend(ws.results())
    except Exception as e:
        if not break_on_error:
            raise
        log.error('map failed: %s', e)
    return L

run = map
