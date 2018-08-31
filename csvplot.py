import sys
import argparse
from contextlib import contextmanager

import numpy as np
from squad.utis import env_param
from matplotlib import pyplot as plt

output_file = env_param('plot_output', default=None, cast=str)
title = env_param('plot_title', default=None, cast=str)

@contextmanager
def single_plot(figsize=None, xyz=False, xlabel=None, ylabel=None, zlabel=None,
                xlim=None, ylim=None, zlim=None, tight_layout=True,
                legend=False, title=title, output_file=output_file):
    fig = plt.figure(figsize=figsize)
    if not xyz:
        ax = plt.gca()
    else:
        __import__('mpl_toolkits.mplot3d')
        ax = fig.gca(projection='3d')

    yield fig, ax

    if xlabel is not None: ax.set_xlabel(xlabel)
    if ylabel is not None: ax.set_ylabel(ylabel)
    if zlabel is not None: ax.set_zlabel(zlabel)

    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if zlim is not None: ax.set_zlim(zlim)

    if title is not None: ax.set_title(title)

    if tight_layout: fig.tight_layout()
    if legend: ax.legend()

    if output_file is not None:
        fig.savefig(output_file)
    else:
        plt.show()

def sizestr(v):
    w, h = v.split(',')
    return float(w), float(h)

def limitspec(spec):
    if not spec:
        return slice(None, None, None)
    elif ':' not in spec:
        raise ValueError('limit must include a colon')
    else:
        start, stop = spec.split(':', 1)
        if not start:
            return slice(None, int(stop), None)
        elif not stop:
            return slice(int(start), None, None)
        else:
            return slice(int(start), int(stop), None)

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='input csv filename, or -')
parser.add_argument('-x', metavar='COL', default=0,
                    help='plot column COL as the x axis', type=int)
parser.add_argument('-y', metavar='COL', action='append',
                    help='plot column COL as the y axis', type=int)
parser.add_argument('--rows', help='range of row numbers to print in the form '
                    'of start:stop, both optional.', metavar='RANGE',
                    type=limitspec, default=slice(None, None, None))
parser.add_argument('--limit', '-l', metavar='NUM', help='maximum number of rows', type=int)
parser.add_argument('--delimiter', '-d', help='field delimiter, default: whitespace',
                    metavar='DELIM')
parser.add_argument('--skiprows', metavar='NUM', help='skip NUM rows',
                    type=int, default=0)
parser.add_argument('--figsize', help='figure size', type=sizestr)
parser.add_argument('--x-label', help='label for x axis', metavar='LABEL')
parser.add_argument('--y-label', help='label for y axis', metavar='LABEL')
parser.add_argument('--title', help='plot title')
parser.add_argument('--output', metavar='FILE', help='save plot to FILE')
parser.add_argument('--x-grid-minor-period', default=25, type=int, metavar='NUM',
                    help='periodicity of minor grid along x axis')
parser.add_argument('--x-grid-num-major', default=5, type=int, metavar='NUM',
                    help='number of major grid along x axis')
parser.add_argument('--y-log', action='store_true', help='make y axis logarithmic')
parser.add_argument('--no-plot', help='do not plot main plot',
                    action='store_true', default=False)
parser.add_argument('--mean', action='store_true', help='plot average of minor tick')
parser.add_argument('--lowpass', help='plot lowpass-filtered signal',
                    metavar='RC', type=float)

def main():
    args = parser.parse_args()

    filename = args.filename if args.filename != '-' else sys.stdin
    vals = np.loadtxt(filename,
                      delimiter=args.delimiter,
                      skiprows=args.skiprows)
    vals = vals[args.rows][:args.limit]
    n_rows = vals.shape[0]
    print('loaded', n_rows, 'rows', file=sys.stderr)
    xs = vals[:, args.x]
    xlim = [xs[0] - 1e-3, xs[-1] + 1e-3]
    with single_plot(xlabel=args.x_label, ylabel=args.y_label, xlim=xlim,
                     figsize=args.figsize, title=args.title,
                     output_file=args.output) as (fig, ax):
        n_maj_ticks = args.x_grid_num_major
        min_tick_period = args.x_grid_minor_period
        n_min_ticks = n_rows//min_tick_period
        maj_tick_period = int(min_tick_period*max(1, n_min_ticks//n_maj_ticks))
        if args.y is None:
            args.y = [i for i in range(vals.shape[1]) if i != args.x]
        for y in args.y:
            ys = vals[:, y]
            if not args.no_plot:
                ax.plot(xs, ys, linewidth=1.2)
            xs_min_period = xs.astype(int) % min_tick_period
            if args.mean:
                istart = xs_min_period[:min_tick_period].argmin()
                iend = n_rows - min_tick_period + xs_min_period[-min_tick_period:].argmin()
                if n_rows - iend >= min_tick_period:
                    iend += min_tick_period
                ys_mean = ys[istart:iend].reshape((-1, min_tick_period)).mean(axis=1)
                xs_mean = xs[istart:iend].reshape((-1, min_tick_period)).mean(axis=1)
                ax.plot(xs_mean, ys_mean, '--o', markersize=3.0, linewidth=1.2)
            if args.lowpass:
                α = 1.0/(args.lowpass + 1.0)
                zs = np.empty_like(ys)
                zprev = ys[0]
                for i in range(zs.shape[0]):
                    if xs_min_period[i] == 0:
                        zprev = ys[i]
                    zs[i] = zprev = α*ys[i] + (1 - α)*zprev
                #ys_ma = (ys[3:] + ys[2:-1] + ys[1:-2] + ys[:-3])/4.0
                #xs_ma = xs[3:]
                ax.plot(xs, zs, '-', linewidth=1.2)
        ax.xaxis.grid(True, which='both', linewidth=1.0, alpha=0.9)
        ax.xaxis.grid(True, which='minor', linewidth=0.4, alpha=0.8)
        ax.xaxis.set_ticks([x for x in xs if (x % maj_tick_period) == 0], minor=False)
        ax.xaxis.set_ticks([x for x in xs if (x % min_tick_period) == 0], minor=True)
        if args.y_log:
            ax.set_yscale('log')

if __name__ == "__main__":
    main()
