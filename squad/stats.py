import logging
from functools import wraps
from contextlib import contextmanager
from time import time as clock
from threading import current_thread

log = logging.getLogger(__name__)

# Implement our own thread local scheme because we want to be able to print
# summaries over all threads.
_currs = {}
_roots = {}
_modules = set()

def _curr():
    tid = current_thread().ident
    if tid not in _currs:
        curr = {'.name': f'thread #{len(_currs)}'}
        _currs[tid] = curr
        _roots[tid] = curr
    return _currs[tid]

@contextmanager
def _push(name):
    tid = current_thread().ident
    prev = _curr()
    _currs[tid] = prev.setdefault(name, {'.name': name})
    try:
        yield _currs[tid]
    finally:
        _currs[tid] = prev

@contextmanager
def timed(name):
    t0 = clock()
    with _push(name) as curr:
        curr.setdefault('.firsttime', t0)
        try:
            yield
        finally:
            t1 = clock()
            curr['.lasttime'] = (t1 - t0)
            curr['.cumtime'] = curr.get('.cumtime', 0) + (t1 - t0)
            curr['.n_calls'] = curr.get('.n_calls', 0) + 1

def measure_func(f):
    name = f'{f.__module__}.{f.__name__}'
    _modules.add(f.__module__)
    @wraps(f)
    def measure_wrap(*a, **k):
        with timed(name):
            return f(*a, **k)
    return measure_wrap

def timed_iter(it, *, name):
    """Just like wrapping a for loop in with timed() but updates the statistics
    on each iteration."""
    with _push(name) as curr:
        t0 = clock()
        curr.setdefault('.firsttime', t0)
        cumtime0 = curr.get('.cumtime', 0.0)
        n_calls0 = curr.get('.n_calls', 0)
        for i, v in enumerate(it):
            try:
                yield v
            finally:
                t1 = clock()
                curr['.lasttime'] = (t1 - t0)
                curr['.cumtime'] = cumtime0 + (t1 - t0)
                curr['.n_calls'] = n_calls0 + i

def warn_slow(f, max_time, msg='function running slow'):
    name = f'{f.__module__}.{f.__name__}'
    dt = _curr().get(name, {}).get('.lasttime')
    if dt is not None and dt > max_time:
        log.warn(f'{msg} (%.5gs)', dt)

def print_summary(curr=_roots, indent='', file=None):
    subkeys = {k for k in curr if not str(k).startswith('.')}
    n = len(subkeys)
    sk_it = sorted(subkeys,
                   key=lambda d: curr[d].get('.cumtime', -1),
                   reverse=True)
    for i, key in enumerate(sk_it):
        val = curr[key]
        name = val['.name']
        if len(_modules) <= 1:
            name = name.split('.', 1)[-1]
        if '.name' in curr:
            ldc = '├┌└ '[int(i==0) + 2*int(i==n-1)] + ' '
            subindent = indent + ('│ ' if n-i > 1 else '  ')
        else:
            subindent, ldc = indent, ''
        if '.cumtime' in val:
            print(f'{indent}{ldc}{name} ({val[".cumtime"]:.3g}s, {val[".n_calls"]}#)', file=file)
        else:
            print(f'{indent}{ldc}{name}', file=file)
        print_summary(curr=val, indent=subindent, file=file)

def clear(local=True):
    if local:
        tid = current_thread().ident
        del _roots[tid]
        del _currs[tid]
    else:
        _roots.clear()
        _currs.clear()
