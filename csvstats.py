import re
import sys
import argparse
import numpy as np
from itertools import count

# {{{ Statistical functions registry
_stat_func_resolver_reg = []
def stat_func_resolver(sre):
    sre = re.compile(sre)
    def deco(resolver):
        _stat_func_resolver_reg.append((resolver, sre))
        return resolver
    return deco

def resolve_stat_func(key):
    for resolver, sre in _stat_func_resolver_reg:
        mo = sre.match(key)
        if mo is not None:
            return resolver(key, *mo.groups())
    else:
        raise KeyError(key)

def resolve_stat_func_spec(spec):
    field_name_sets, sfs = list(zip(*map(resolve_stat_func, spec.split(','))))
    field_names = [fn for field_names in field_name_sets for fn in field_names]
    return field_names, sfs
# }}}

@stat_func_resolver(r'lineno|linenum|linenr')
def lineno_resolver(key):
    counter = count()
    return [key], lambda row: next(counter)

@stat_func_resolver(r'mean|min|max|std|var')
def np_func_resolver(key):
    return [key], getattr(np, key)

@stat_func_resolver(r'(asc|first|top|desc|last|bot(?:tom)?)(\d+)(st|rd|th)?')
def sorted_n(key, order, n, ordinal_suffix):
    n = int(n)
    definite = bool(ordinal_suffix)
    sign = -1 if order in {'desc', 'last', 'bot', 'bottom'} else +1
    if not definite:
        start, stop = 0, n
    else:
        start, stop = n, n + 1
    field_names = [f'{order}[{sign*i}]' for i in range(start, stop)]
    return field_names, lambda row: sign*np.sort(sign*row)[start:stop]

@stat_func_resolver(r'(\d+(?:\.\d+)?)%')
def percentile_resolver(key, perc):
    perc = float(perc)
    return [key], lambda row: np.percentile(row, [perc])

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
parser.add_argument('--rows', help='range of row numbers to print in the form '
                    'of start:stop, both optional.', metavar='RANGE',
                    type=limitspec, default=slice(None, None, None))
parser.add_argument('--cols', help='range of column numbers to print in the '
                    'form of start:stop, both optional.', metavar='RANGE',
                    type=limitspec, default=slice(None, None, None))
parser.add_argument('--limit', '-l', metavar='NUM', help='maximum number of rows', type=int)
parser.add_argument('--delimiter', '-d', help='field delimiter, default: whitespace',
                    metavar='DELIM')
parser.add_argument('--skiprows', metavar='NUM', help='skip NUM rows',
                    type=int, default=0)
parser.add_argument('--stats', help='comma-separated stats to compute. '
                    'available: mean, min, max, std, var, x%',
                    default='5%,mean,95%')
parser.add_argument('--header', action='store_true',
                    help='print field names as first row')

def main():
    args = parser.parse_args()
    filename = args.filename if args.filename != '-' else sys.stdin
    vals = np.loadtxt(filename,
                      delimiter=args.delimiter,
                      skiprows=args.skiprows)
    vals = vals[args.rows, args.cols]
    vals = vals[:args.limit]
    delim_out = args.delimiter if args.delimiter else '\t'
    field_names, stat_funcs = resolve_stat_func_spec(args.stats)
    if args.header:
        print(delim_out.join(field_names))
    for row in vals:
        out_row = [v for f in stat_funcs for v in np.atleast_1d(f(row))]
        print(delim_out.join(map(str, out_row)))

if __name__ == "__main__":
    main()
