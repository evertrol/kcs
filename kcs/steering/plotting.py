"""Overplot EC-EARTH scenario matches on top of CMIP tas distribution"""

import sys
from datetime import datetime
import logging
import argparse
import itertools
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import iris
import kcs.utils.logging
import kcs.utils.argparse
from kcs.cmip.plotting import plot as cmipplot
from kcs.utils.atlist import atlist
from . import read_data, normalize_average_dataset, num2date


REFERENCE_PERIOD = (1981, 2010)

# If we run as a runnable module, use a more appropriate logger name
logname = 'kcs.steering.plotting' if __name__ == '__main__' else __name__
logger = logging.getLogger(logname)


def plot_ecearth(ecearth_data, relative=False, reference_period=REFERENCE_PERIOD,
                 smooth=None, years=None):
    """Plot the averaged EC-EARTH dataset"""

    paths = list(itertools.chain.from_iterable(atlist(path) for path in ecearth_data))
    dataset = read_data(paths)
    cube = normalize_average_dataset(dataset['cube'], relative=relative,
                                     reference_period=reference_period)

    if years:
        constraint = iris.Constraint(year=lambda cell: cell.point in years)
        cube = constraint.extract(cube)

    if smooth:
        cube = cube.rolling_window('year', iris.analysis.MEAN, smooth)

    dates = num2date(cube.coord('time'))
    dates = np.array([datetime(date.year, date.month, date.day) for date in dates],
                     dtype='datetime64')
    plt.plot(dates, cube.data, zorder=6, label='EC-EARTH', color='#669955')


def plot_scenarios(scenarios, reference_epoch=None):
    """DUMMY DOCSTRING"""
    if reference_epoch:
        plt.scatter(datetime(reference_epoch, 1, 1), 0, s=100, marker='s', color='black', zorder=6)
    for scenario in scenarios:
        date = datetime(scenario['epoch'], 1, 1)
        temp = scenario['percentile']
        plt.scatter(date, temp, s=100, marker='o', color='green', zorder=6)


def run(percentiles, steering_table, outfile, reference_epoch=None,
        xlabel=None, ylabel=None, xrange=None, yrange=None, title=None, smooth=None,
        ecearth_data=None, relative=False, reference_period=REFERENCE_PERIOD):
    """DUMMY DOCSTRING"""

    figure = plt.figure(figsize=(12, 8))
    if smooth:
        percentiles = percentiles.rolling(window=smooth, center=True).mean()
    figure = cmipplot(figure, percentiles, xlabel=xlabel, ylabel=ylabel,
                      xrange=xrange, yrange=yrange, title=title)

    if ecearth_data:
        years = [dt.year for dt in percentiles.index]
        plot_ecearth(ecearth_data, relative=relative, reference_period=reference_period,
                     smooth=smooth, years=years)

    scenarios = steering_table.to_dict('records')
    # This assumes the percentiles in the scenarios are present (as
    # strings) in the percentiles table.
    for scenario in scenarios:
        epoch = datetime(scenario['epoch'], 1, 1)
        percentile = str(scenario['percentile'])
        scenario['percentile'] = percentiles.loc[epoch, percentile]

    plot_scenarios(scenarios, reference_epoch=reference_epoch)

    plt.tight_layout()
    plt.savefig(outfile)


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser(parents=[kcs.utils.argparse.parser],
                                     conflict_handler='resolve')
    parser.add_argument('percentiles', help="Input CSV file with CMIP distribution percentiles")
    parser.add_argument('steering_table', help="Input steering table CSV file")
    parser.add_argument('outfile', help="Output figure filename. The extension determines "
                        "the file type.")
    parser.add_argument('--ecearth-data', nargs='+', help="EC-EARTH data files. Using this "
                        "option will overplot the average of the input data.")
    parser.add_argument('--relative', action='store_true', help="Indicate the EC-EARTH data "
                        "should be calculated as relative change, as opposed to absolute change.")
    parser.add_argument('--reference-period', type=int, nargs=2, default=REFERENCE_PERIOD,
                        help="Reference period to normalize the EC-EARTH data to (if input). "
                        "Note that this can be different than the reference epoch, that is, "
                        "the middle of the reference-period does *not* have to correspond "
                        "to the reference epoch.")
    parser.add_argument('--reference-epoch', type=int, help="Mid-reference period year to "
                        "which all CMIP data were normalized. If this option is not used, no "
                        "indicator of the reference point is plotted.")
    parser.add_argument('--xlabel', default='Year')
    parser.add_argument('--ylabel')
    parser.add_argument('--xrange', type=float, nargs=2)
    parser.add_argument('--yrange', type=float, nargs=2)
    parser.add_argument('--title')
    parser.add_argument('--smooth', type=int, nargs='?', const=10)
    args = parser.parse_args()
    if args.ecearth_data:
        args.ecearth_data = [pathlib.Path(filename) for filename in args.ecearth_data]
    return args


def main():
    """Starting point when running as a script/module"""
    args = parse_args()
    kcs.utils.logging.setup(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    percentiles = pd.read_csv(args.percentiles, index_col=0)
    percentiles.index = pd.to_datetime(percentiles.index)
    steering_table = pd.read_csv(args.steering_table)
    steering_table['period'] = steering_table['period'].apply(
        lambda x: tuple(map(int, x.strip('()').split(','))))

    run(percentiles, steering_table, args.outfile, reference_epoch=args.reference_epoch,
        xlabel=args.xlabel, ylabel=args.ylabel,
        xrange=args.xrange, yrange=args.yrange, title=args.title, smooth=args.smooth,
        ecearth_data=args.ecearth_data, relative=args.relative,
        reference_period=args.reference_period)
    logger.info("Done processing")


if __name__ == '__main__':
    main()
