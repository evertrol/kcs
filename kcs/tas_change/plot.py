"""

Usage example:

$ python -m kcs.tas_change.plot  distribution-percentiles.csv cmip6.png \
    --xrange 1950 2100    --ylabel 'Temperature change [${}^{\circ}$]' \
    --title 'Global year temperature change'  --smooth 7 --yrange -1 6

"""

import sys
from datetime import datetime
import logging
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cftime
from ..utils.logging import setup as setup_logging
from ..utils.argparse import parser as kcs_parser
from ..config import read_config


# If we run as a runnable module, use a more appropriate logger name
logname = 'tas-change-plot' if __name__ == '__main__' else __name__
logger = logging.getLogger(logname)


def num2date(coord, index=None):
    """DUMMY DOCSTRING"""
    if index is None:
        return cftime.num2date(coord.points, str(coord.units), coord.units.calendar)
    return cftime.num2date(coord.points[index], str(coord.units),
                           coord.units.calendar)


def plot(figure, percs, dataset=None, xlabel=None, ylabel=None,
         legend=False, title=None, xrange=None, yrange=None):
    """Plot the percentile distribution as function of time

    Parameters
    ----------
    - figure: matplotlib.figure.Figure

    Returns
    -------
    matplotlib.figure.Figure

    """
    logger.info("Plotting")

    if dataset is not None:
        for item in dataset.itertuples():
            # Iris's plotting functionality has some problems when plotting
            # graphs with multiple calendar types on top of each other
            # We (attempt to) convert the dates to Python datetimes,
            # which should be fine, given that we're using yearly averages
            # We're only doing this for plotting; the actual data is not changed
            dates = num2date(item.cube.coord('time'))
            dates = np.array([datetime(date.year, date.month, date.day) for date in dates],
                             dtype='datetime64')
            plt.plot(dates, item.cube.data, alpha=0.3, zorder=1,
                     label=f"{item.model}-{item.realization}-{item.rcp}")

    if percs is not None:
        dates = percs.index
        plt.fill_between(dates,
                         percs['5'].to_numpy(dtype=np.float),
                         percs['95'].to_numpy(dtype=np.float),
                         color='#bbbbbb', alpha=0.8, zorder=2, label='5% - 95%')
        plt.fill_between(dates,
                         percs['10'].to_numpy(dtype=np.float),
                         percs['90'].to_numpy(dtype=np.float),
                         color='#888888', alpha=0.4, zorder=3, label='10% - 90%')
        plt.fill_between(dates,
                         percs['25'].to_numpy(dtype=np.float),
                         percs['75'].to_numpy(dtype=np.float),
                         color='#555555', alpha=0.2, zorder=4, label='25% - 75%')
        plt.plot(dates, percs['mean'], color='#000000', lw=2, zorder=5, label='mean')
        axes = plt.gca()
        axes.yaxis.set_ticks_position('both')


    if legend:
        plt.legend()
    if xrange is None:
        xrange = [datetime(1950, 1, 1), datetime(2100, 1, 1)]
    if isinstance(xrange[0], (int, float)):
        year = int(xrange[0])
        mon = int(12 * (xrange[0] - year)) + 1
        day = int(28 * (xrange[0] - year - (mon-1)/12)) + 1   # Simple, safe estimate
        xrange = [datetime(year, mon, day), xrange[1]]
    if isinstance(xrange[1], (int, float)):
        year = int(xrange[1])
        mon = int(12 * (xrange[1] - year)) + 1
        day = int(28 * (xrange[1] - year - (mon-1)/12)) + 1   # Simple, safe estimate
        xrange = [xrange[0], datetime(year, mon, day)]
    if yrange:
        plt.axis([xrange[0], xrange[1], yrange[0], yrange[1]])
    plt.xticks([datetime(year, 1, 1) for year in np.arange(1950, 2100, 20)])
    return figure


def finish(xlabel=None, ylabel=None, title=None, legend=True, grid=True):
    """Add labels, a legend and a grid"""

    if not xlabel:
        xlabel = 'Year'
    plt.xlabel(xlabel, fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18)
    if title:
        plt.title(title, fontsize=22)
    if grid:
        plt.grid()
    if legend:
        plt.legend()


def run(percentiles, outfile, dataset=None, xlabel=None, ylabel=None,
        xrange=None, yrange=None, title=None, grid=True, legend=True,
        smooth=None):
    figure = plt.figure(figsize=(12, 8))
    if smooth:
        percentiles = percentiles.rolling(window=smooth, center=True).mean()
    plot(figure, percentiles, xlabel=xlabel, ylabel=ylabel,
         xrange=xrange, yrange=yrange, title=title)

    finish(xlabel=xlabel, ylabel=ylabel, title=title, grid=grid, legend=legend)

    plt.tight_layout()
    plt.savefig(outfile)


def parse_args():
    """DUMMY DOCSTRING"""
    parser = argparse.ArgumentParser(parents=[kcs_parser],
                                     conflict_handler='resolve')
    parser.add_argument('infile', help="Input CSV file with distribution percentiles")
    parser.add_argument('outfile', help="Output figure filename. The extension determines "
                        "the file type.")
    parser.add_argument('--xlabel', default='Year')
    parser.add_argument('--ylabel')
    parser.add_argument('--xrange', type=float, nargs=2)
    parser.add_argument('--yrange', type=float, nargs=2)
    parser.add_argument('--title')
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--smooth', type=int, nargs='?', const=10)

    args = parser.parse_args()
    setup_logging(args.verbosity)
    read_config(args.config)

    return args


def main():
    """DUMMY DOCSTRING"""
    args = parse_args()
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    percentiles = pd.read_csv(args.infile, index_col=0)
    percentiles.index = pd.to_datetime(percentiles.index)
    run(percentiles, args.outfile, xlabel=args.xlabel, ylabel=args.ylabel,
        xrange=args.xrange, yrange=args.yrange, title=args.title,
        grid=args.grid, legend=args.legend, smooth=args.smooth)
    logger.info("Done processing")


if __name__ == '__main__':
    main()
