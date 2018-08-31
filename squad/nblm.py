"Numba Leak Monitor"

import time
import logging
import threading
import numpy as np

log = logging.getLogger(__name__)

class LeakMonitor():
    interval = 30.0
    num_history = 10

    def __init__(self):
        self._thread = threading.Thread(target=self._thread, daemon=True)
        self._imbalance = []

    def start(self):
        self._run = True
        self._thread.start()

    def stop(self):
        self._run = False

    def _thread(self):
        import numba.runtime as nrt
        while self._run:
            st = nrt.rtsys.get_allocation_stats()
            self._imbalance.append(st.alloc - st.free)
            self._imbalance = self._imbalance[-self.num_history:]
            if len(self._imbalance) == self.num_history:
                self._check_leaks()
            time.sleep(self.interval)

    def _check_leaks(self):
        if np.all(np.diff(self._imbalance) > 0):
            log.warn('seems like numba is leaking memory! '
                     '%d unfreed allocations', self._imbalance[-1])

_lm = LeakMonitor()
start = _lm.start
stop  = _lm.stop
def set_interval(n):
    _lm.interval = n
