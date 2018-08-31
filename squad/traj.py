import sys
import logging
from collections import namedtuple

import numpy as np
import logcolor

from . import plots

log = logging.getLogger(__name__)

class Trajectory(namedtuple('TrajectoryBase', 'dt states obs actions rewards')):
    def __new__(cls, *, dt, states, actions, obs=None, rewards=None):
        return super().__new__(cls, dt, states, obs, actions, rewards)

    def save(self, fn, **kw):
        np.savez(fn, dt=self.dt, states=self.states, obs=self.obs,
                 actions=self.actions, rewards=self.rewards, **kw)

    @classmethod
    def load(cls, fn):
        return cls(**np.load(fn))

    @classmethod
    def from_history(cls, history):
        return cls.from_parts(*to_parts(history))

    @classmethod
    def from_parts(cls, ts, states, obs, actions, rewards):
        dts = np.diff(ts)
        assert dts.std() < 1e-9
        dt = dts.mean()
        return cls(dt, states, obs, actions, rewards)

    def to_history(self):
        return from_parts(dt=self.dt, states=self.states, obses=self.obses,
                          actions=self.actions, rewards=self.rewards)

def load(fn):
    items = np.load(fn)
    #states = items['states']
    #actions = items['actions']
    #ts = np.arange(states.shape[0])*items['dt']
    #rewards = items['rewards'] if 'rewards' in items else np.zeros_like(ts)
    #obses = [None for i in range(states.shape[0])]
    return from_parts(**items)

def load_many(fn):
    items = np.load(fn)
    if 'states' in items:
        yield from_parts(**items)
    else:
        i = 0
        while f't{i}_states' in items:
            traj_items = {k.split('_', 1)[-1]: items[k]
                          for k in items
                          if k.startswith(f't{i}_')}
            yield from_parts(**traj_items)
            i += 1

def save_many(fn, histories):
    all_items = {}
    for i, history in enumerate(histories):
        if hasattr(history, 'states'):
            trj = history
        else:
            trj = Trajectory.from_history(history)
        items = dict(dt=trj.dt, states=trj.states, obs=trj.obs,
                     actions=trj.actions, rewards=trj.rewards)
        all_items.update({f't{i}_{k}': v for k, v in items.items() if v is not None})
    np.savez(fn, **all_items)

def from_parts(*, t0=0.0, dt, states, obses=None, actions=None, rewards=None, **kw):
    states = np.asarray(states, dtype=np.float64)
    n = states.shape[0]
    ts = t0 + dt*np.arange(n)
    obses = obses if obses is not None else np.zeros((n, 0))
    actions = actions if actions is not None else np.zeros((n, 0))
    rewards = rewards if rewards is not None else np.zeros((n,))
    return list(zip(ts, states, obses, actions, rewards))

def to_parts(history):
    ts, states, obses, actions, rewards = map(np.asarray, zip(*history))
    return ts, states, obses, actions, rewards

class Viewer():
    def __init__(self, *, distance=1.5, elevation=30.0, azimuth=10.0):
        from pyqtgraph.Qt import QtGui, QtCore
        self.app = QtGui.QApplication([])
        self.app.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)
        from .trajview import GLViewWidget, AxisItem
        self.w = w = GLViewWidget()
        w.opts['fov'] = 90.0
        w.opts['elevation'] = elevation
        w.opts['azimuth'] = azimuth
        w.opts['distance'] = distance
        w.setHome()
        w.setBackgroundColor('#ccc')
        w.addItem(AxisItem(antialias=True))
        w.update()

    def plot_traj_kwds(self, history=None, states=None, color='#fff',
                       width=0.5, alternate=50):
        import pyqtgraph as pg
        if states is None and history:
            _, states, *_ = to_parts(history)
        if states is None or states.shape[0] <= 1:
            segs, colors = None, None
        else:
            edges = np.arange(states.shape[0]) - 1
            segs = np.vstack([states[(dst, edges[dst]), 0:3]
                              for dst in range(states.shape[0])
                              if edges[dst] >= 0])
            colors = np.empty((segs.shape[0], 4))
            colors[:, :] = pg.glColor(color)
            for i in range(0*alternate, 1*alternate):
                colors[i::2*alternate, 3] *= 1.0
            for i in range(1*alternate, 2*alternate):
                colors[i::2*alternate,   3] *= 0.25
                # Inverted colors in RGB space
                #colors[i::2*alternate, 0:3] -= 1.0
                #colors[i::2*alternate, 0:3] *= -1.0
        return dict(pos=segs, color=colors, antialias=True, mode='lines')

    def plot_traj(self, history=None, **kw):
        import pyqtgraph.opengl as gl
        #import pyqtgraph as pg
        #_, x0, *_ = history[0]
        #start_point = gl.GLScatterPlotItem(pos=x0[0:3][None, :], pxMode=False,
        #                                   color=(1.0, 1.0, 1.0, 1.0), size=0.02)
        #self.w.addItem(start_point)
        lines = gl.GLLinePlotItem(**self.plot_traj_kwds(history, **kw))
        self.w.addItem(lines)
        return lines

    def pan_start(self, history):
        _, x0, *_ = history[0]
        _, x1, *_ = history[-1]
        self.pan(*(x0[0:3] + x1[0:3])/2.0)

    def pan(self, x, y, z):
        from pyqtgraph import Vector
        self.w.opts['center'] = Vector(x, y, z)

    def show(self):
        from pyqtgraph.Qt import QtGui
        self.w.show()
        QtGui.QApplication.instance().exec_()

class Tracker(Viewer):
    # Copied from matplotlib
    autocolors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                  '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    def __init__(self, *args, autopan=True, axis=True,
                 parent=None, plot_opts={}, num_history=100, **kwds):
        if parent is None:
            super().__init__(*args, **kwds)
        else:
            self.app = parent.app
            self.w = parent.w
        self.autopan = autopan
        from .trajview import AxisItem
        if axis:
            self.axis = AxisItem(antialias=True)
            self.axis.setSize(x=0.1, y=0.1, z=0.1)
        else:
            self.axis = None
        self.w.addItem(self.axis)
        autocolor = plot_opts.pop('autocolor', None)
        if autocolor is not None:
            color_idx = autocolor % len(self.autocolors)
            plot_opts['color'] = self.autocolors[color_idx]
        self.plot_opts = plot_opts
        self.line_plot = self.plot_traj([], **self.plot_opts)
        self.alts = [self.alt(autocolor=i)
                     for i in range(num_history)] if parent is None else []
        self._lru_ptr = 0

    def set_history(self, history):
        self.line_plot.setData(**self.plot_traj_kwds(history, **self.plot_opts))
        self.line_plot.setVisible(True)
        _, x1, *_ = history[-1]
        if self.autopan:
            self.pan(*x1[0:3])
        if self.axis:
            self.axis.position = x1[0:3]
            self.axis.orientation = x1[3:7]
            self.axis.setVisible(True)

    def clear(self):
        self.line_plot.setVisible(False)
        if self.axis:
            self.axis.setVisible(False)
        for alt in self.alts:
            alt.clear()

    def set_trajectory(self, trj):
        self.set_history([(None, x, u, None, None) for x, u in zip(trj.states, trj.actions)])

    def alt(self, **kwds):
        return type(self)(parent=self, autopan=False, plot_opts=kwds)

    def alt_lru(self):
        if not self.alts:
            return None
        alt = self.alts[self._lru_ptr]
        self._lru_ptr += 1
        self._lru_ptr %= len(self.alts)
        return alt

def main(args=sys.argv[1:]):
    logcolor.basic_config()
    histories = []
    while args:
        arg = args.pop(0)
        if arg == 'load':
            fn = args.pop(0)
            history = load(fn)
            histories.append(history)
            log.info("loaded '%s' (%d steps)", fn, len(history))
        elif arg == 'loadmany':
            fn = args.pop(0)
            histories_fn = list(load_many(fn))
            histories.extend(histories_fn)
            log.info("loaded '%s' (%d trajectories)", fn, len(histories_fn))
            for i, history in enumerate(histories_fn):
                log.info("- trajectory %d (%d steps)", i, len(history))
        elif arg == 'select':
            num = int(args.pop(0))
            history = histories[num]
        elif arg == 'savemany':
            fn = args.pop(0)
            save_many(fn, histories)
            log.info("saving to '%s' (%d trajectories)", fn, len(histories))
        elif arg == 'animate' or arg == 'video':
            plots.animation_frames(*histories)
        elif arg == 'plot' or arg == 'evaluate':
            plots.evaluation(history)
        elif arg == 'showtraj':
            plots.trajectory(*histories)
        elif arg == 'viewinit':
            viewer = Viewer()
        elif arg == 'viewtraj':
            for history in histories:
                viewer.plot_traj(history)
        elif arg == 'viewtraj2':
            viewer.plot_traj(history, color='g', width=0.2)
        elif arg == 'viewpan':
            viewer.pan_start(history)
        elif arg == 'viewshow':
            viewer.show()
        else:
            log.error('unknown command: %s', arg)

if __name__ == '__main__':
    main()
