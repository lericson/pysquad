import os
import numpy as np
import logging

from . import quat
from .utis import env_param

log = logging.getLogger(__name__)

output_file = env_param('plot_output', default=None, cast=str)
figures_dir = env_param('OUTDIR', default=None)
def figdir(*p):
    return os.path.join(figures_dir, *p)

fps = env_param('fps', default=30, cast=int)
plot_format = env_param('plot_format', default='-')

def _s(s):
    if s.shape[-1] == 6:
        s = np.r_[s[0:3], 0, 0, 0, 1, s[3:6], 0, 0, 0]
    return s

def evaluation(history, fmt=plot_format):
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import pyplot as plt
    from scipy import signal

    plt.rc('legend', loc='upper right')
    fig, ((ax11, ax12), (ax21, ax22), (ax31, ax32), (ax41, ax42), (ax51, ax52)) = plt.subplots(5, 2, sharex=True)
    ts = np.array([t for t, state, obs, action, reward in history])
    states = np.array([_s(state) for t, state, obs, action, reward in history])
    actions = np.array([action for t, state, obs, action, reward in history])
    rewards = np.array([reward for t, state, obs, action, reward in history])
    rpys = np.array([quat.euler_rpy(state[3:7]) for state in states])

    for i, label in enumerate('$r_x$ $r_y$'.split()):
        ax11.plot(ts, states[:, i], fmt, label=label)
    ax11.legend()
    ax11.grid()
    ax11.set_ylabel('Position (m)')
    #ax11.set_xlabel('Time (s)')

    for i, label in enumerate('$v_x$ $v_y$'.split()):
        ax12.plot(ts, states[:, 3+4+i], fmt, label=label)
    ax12.legend()
    ax12.grid()
    ax12.set_ylabel('Velocity (m/s)')
    #ax12.set_xlabel('Time (s)')

    for i, label in enumerate('$r_z$'.split()):
        ax21.plot(ts, states[:, 2+i], fmt, label=label)
    ax21.legend()
    ax21.grid()
    ax21.set_ylabel('Altitude (m)')
    #ax21.set_xlabel('Time (s)')

    for i, label in enumerate('$v_z$'.split()):
        ax22.plot(ts, states[:, 3+4+2+i], fmt, label=label)
    ax22.legend()
    ax22.grid()
    ax22.set_ylabel('Velocity (m/s)')
    #ax22.set_xlabel('Time (s)')

    for i, label in enumerate(r'$\theta_x$ $\theta_y$ $\theta_z$'.split()):
        ax31.plot(ts, rpys[:, i], fmt, label=label)
    ax31.legend()
    ax31.grid()
    ax31.set_ylabel('Orientation (rad)')
    #ax31.set_xlabel('Time (s)')

    for i, label in enumerate('$\omega_x$ $\omega_y$ $\omega_z$'.split()):
        ax32.plot(ts, states[:, 3+4+3+i], fmt, label=label)
    ax32.legend()
    ax32.grid()
    ax32.set_ylabel('Velocity (rad/s)')
    #ax32.set_xlabel('Time (s)')

    for i in range(states.shape[1] - (3+4+3+3)):
        ax41.plot(ts, states[:, 3+4+3+3+i], fmt, label=r'$\alpha_{}$'.format(i))
    ax41.legend()
    ax41.grid()
    ax41.set_ylabel('Rotor speed (RPM)')
    #ax41.set_xlabel('Time (s)')

    for i in range(actions.shape[1]):
        ax42.plot(ts, actions[:, i], fmt, label='$a_{}$'.format(i))
    ax42.legend()
    ax42.grid()
    ax42.set_ylim([0, 1])
    ax42.set_ylabel('PWM signal level')
    #ax42.set_xlabel('Time (s)')

    ax51.plot(ts, rewards.cumsum())
    ax51.grid()
    ax51.set_ylabel('Return')
    ax51.set_xlabel('Time (s)')

    if len(ts) > 24:
        f, t, Sxx = signal.spectrogram(actions.mean(axis=1), 1/np.diff(ts).mean(),
                                       axis=0, nperseg=24, noverlap=16,
                                       scaling='spectrum', mode='magnitude')
        ax52.pcolormesh(t, f, Sxx)
    ax52.set_ylabel('Frequency (Hz)')
    ax52.set_xlabel('Time (s)')

    plt.show()

def _render_line_seg(ax, p0, p1, **kw):
    p0 = np.atleast_2d(p0)
    p1 = np.atleast_2d(p1)
    ax.quiver(p0[:, 0], p0[:, 1], p0[:, 2],
              p1[:, 0], p1[:, 1], p1[:, 2],
              arrow_length_ratio=0.0, **kw)

_bounds = None
def _render_animation_frames(a, target=np.r_[0.0, 0.0, 0.0]):
    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import pyplot as plt
    num, frames = a
    ts, states, obses, actions, rewards = zip(*frames)
    plt.ioff()
    fig = plt.gcf()
    ax1 = fig.gca(projection='3d')
    ax1.set_title(f'Simulation ($t = {ts[0]:.3g}$)')
    for (t, state, obs, action, reward) in frames:
        state = _s(state)
        #ax1.plot(world.lines[(0, 2), :], world.lines[(1, 3), :])
        # Plot X, Y and Z axes in each agent's frame to show orientation.
        R = 0.25*quat.rotmat(state[3:7])
        _render_line_seg(ax1, state[0:3], R[:, 0], color='C3')
        _render_line_seg(ax1, state[0:3], R[:, 1], color='C2')
        _render_line_seg(ax1, state[0:3], R[:, 2], color='C0')
        _render_line_seg(ax1, state[0:3], state[7:10], color='purple')
        ax1.scatter(state[None, 0], state[None, 1], state[None, 2], marker='o', s=30)
    ax1.scatter(target[None, 0], target[None, 1], target[None, 2], marker='x', s=20)
    ax1.set_xlabel('X (m)')
    ax1.set_xlim(_bounds[:, 0])
    ax1.set_ylabel('Y (m)')
    ax1.set_ylim(_bounds[:, 1])
    ax1.set_zlabel('Z (m)')
    ax1.set_zlim(_bounds[:, 2])
    plt.tight_layout()
    plt.savefig(figdir('e{:03d}-f{:03d}.png'.format(0, num)))
    if len(ts) == 1:
        rewards, actions = rewards[0], actions[0]
    log.info('saved frame %d, r: %s, action: %s', num, rewards, actions)
    plt.clf()

def default_bounds(*histories):
    states = np.array([state
                       for history in histories
                       for (t, state, obs, action, reward) in history])
    bounds = np.array([states[:, 0:3].min(axis=0),
                       states[:, 0:3].max(axis=0)])
    midpoints = bounds.mean(axis=0)
    lengths = np.diff(bounds, axis=0)
    length = lengths.max()
    mcoef = env_param('plot_margin_coef', default=1.0, cast=float)
    return np.array([midpoints - length/2*mcoef,
                     midpoints + length/2*mcoef])

def animation_frames(*histories, bounds=None, fps=fps):
    global _bounds

    # Step trajectory samples to achieve *fps*
    if len(histories[0]) > 10:
        fps_traj = 1/np.mean(np.diff([t for t, *_ in histories[0]]))
        step = max(int(fps_traj//fps), 1)
        log.info('fps_traj: %.3g Hz, fps: %.3g Hz, step: %d',
                 fps_traj, fps, step)
        histories = [history[::step] for history in histories]

    fps_traj = 1/np.mean(np.diff([t for t, *_ in histories[0]]))

    with open(figdir('fps.txt'), 'w') as fh:
        fh.write(str(fps_traj))

    if bounds is None:
        bounds = default_bounds(*histories)

    _bounds = bounds
    log.info('%d frames to render', len(histories[0]))
    import multiprocessing
    pool = multiprocessing.Pool()
    pool.map(_render_animation_frames, list(enumerate(zip(*histories))))

def trajectory(*histories, bounds=None, target=np.r_[0.0, 0.0, 0.0],
               elevation=env_param('plot_elevation', default=10.0, cast=float),
               azimuth=env_param('plot_azimuth', default=-70.0, cast=float)):
    if bounds is None:
        bounds = default_bounds(*histories)

    from mpl_toolkits.mplot3d import axes3d
    from matplotlib import pyplot as plt
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.gca(projection='3d')
    ax1.view_init(elevation, azimuth)

    rgba = lambda h: tuple(((int(h[1:], 16) >> i*8) & 0xff)/0xff for i in range(3, -1, -1))
    mist, stone, shadow, autumn = '#90afc5', '#336b87', '#2a3132', '#763626'
    gridcolor = rgba(f'{shadow}ff')
    panecolors = rgba(f'{mist}8c'), rgba(f'{mist}80'), rgba(f'{mist}86')
    #panecolors = rgba('#ff000080'), rgba('#00ff0080'), rgba('#0000ff80')
    #panecolors = rgba('#1f77b420'), rgba('#2ca02c30'), rgba('#d6272828')
    for axis, c in zip((ax1.xaxis, ax1.yaxis, ax1.zaxis), panecolors):
        #axis.set_pane_color(c)
        axis._axinfo['grid']['color'] = gridcolor
        axis.init3d()

    sample_rate = 1.0/np.mean(np.diff([t for t, *_ in histories[0]]))
    log.info('sample_rate: %.2f', sample_rate)

    # How often to plot with new visual style to indicate trajectory speed
    num_step = int(0.5*sample_rate)

    seq = [t for t in (dict(linestyle='-', color='C0'),
                       dict(linestyle='-', color='C2'),
                       dict(linestyle='-', color='C3')) for i in range(num_step)]

    for history in histories:
        states = np.array([state for (t, state, obs, action, reward) in history])
        for i in range(0, states.shape[0], num_step):
            j = i + num_step
            kw = seq[i % len(seq)]
            ax1.plot(*states[i:j, 0:3].T, **kw)
            #ax1.plot(*np.array([x0[0:3], x1[0:3]]).T, style)
            #_render_line_seg(ax1, x0[0:3], x1[0:3] - x0[0:3], style, color=color)
        #for i, x0 in enumerate(states[::num_step]):
        #    ax1.scatter(x0[None, 0], x0[None, 1], x0[None, 2], marker='o', color=colorseq[i*num_step % len(colorseq)], s=30)

    ax1.scatter(states[0:1, 0], states[0:1, 1], states[0:1, 2], marker='o', color='k', s=30)
    ax1.scatter(target[None, 0], target[None, 1], target[None, 2], marker='x', color='k', s=40)

    ax1.set_xlabel('X (m)')
    ax1.set_xlim(bounds[:, 0])
    ax1.set_ylabel('Y (m)')
    ax1.set_ylim(bounds[:, 1])
    ax1.set_zlabel('Z (m)')
    ax1.set_zlim(bounds[:, 2])
    #plt.axis('square')
    ax1.set_aspect('equal')
    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file)
