r"""Overplot extra model scenario matches on top of CMIP tas distribution

Usage example:

$ python -m kcs.steering.plot distribution-percentiles.csv steering.csv \
    --outfile output.png --extra-data @extra-tas-global.list --reference-epoch 1995 \
    --ylabel 'Temperature increase [${}^{\circ}$]'  --smooth 10

"""

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
from ..config import default_config, read_config
from ..utils.logging import setup as setup_logging
from ..utils.argparse import parser as kcs_parser
from ..utils.attributes import get as get_attrs
from ..tas_change.plot import plot as cmipplot
from ..tas_change.plot import finish as plot_finish
from ..utils.atlist import atlist
from .core import normalize_average_dataset, num2date


# If we run as a runnable module, use a more appropriate logger name
logname = 'steering-plot' if __name__ == '__main__' else __name__  # pylint: disable=invalid-name
logger = logging.getLogger(logname)  # pylint: disable=invalid-name


def read_data(paths, info_from=('attributes', 'filename'),
              attributes=None, filename_pattern=None):
    """DUMMY DOC-STRING"""
    cubes = [iris.load_cube(str(path)) for path in paths]

    # Get the attributes, and create a dataframe with cubes & attributes
    dataset = get_attrs(
        cubes, paths, info_from=info_from,
        attributes=attributes, filename_pattern=filename_pattern)

    return dataset


def plot_extra(cube, smooth=None, years=None, label=''):
    """Plot the averaged extra dataset

    The input should be a single Iris cube.

    """

    if years:
        constraint = iris.Constraint(year=lambda cell: cell.point in years)
        cube = constraint.extract(cube)

    if smooth:
        cube = cube.rolling_window('year', iris.analysis.MEAN, smooth)

    dates = num2date(cube.coord('time'))
    dates = np.array([datetime(date.year, date.month, date.day) for date in dates],
                     dtype='datetime64')
    plt.plot(dates, cube.data, zorder=6, label=label, color='#669955')


def plot_scenarios(scenarios, reference_epoch=None):
    """DUMMY DOCSTRING"""

    if reference_epoch:
        plt.scatter(datetime(reference_epoch, 1, 1), 0, s=100, marker='s', color='black', zorder=6)
    for scenario in scenarios:
        date = datetime(scenario['epoch'], 1, 1)
        temp = scenario['percentile']
        plt.scatter(date, temp, s=100, marker='o', color='green', zorder=6)


def run(percentiles, steering_table, outfile, reference_epoch=None,
        xlabel=None, ylabel=None, xrange=None, yrange=None, title=None,
        grid=True, legend=True, smooth=None,
        extra_data=None, extra_label=''):
    """DUMMY DOCSTRING"""

    figure = plt.figure(figsize=(12, 8))
    if smooth:
        percentiles = percentiles.rolling(window=smooth, center=True).mean()
    figure = cmipplot(figure, percentiles, xrange=xrange, yrange=yrange)

    if extra_data:
        years = [dt.year for dt in percentiles.index]
        plot_extra(extra_data, smooth=smooth, years=years, label=extra_label)

    scenarios = steering_table.to_dict('records')
    # This assumes the percentiles in the scenarios are present (as
    # strings) in the percentiles table.
    for scenario in scenarios:
        epoch = datetime(scenario['epoch'], 1, 1)
        percentile = str(scenario['percentile'])
        scenario['percentile'] = percentiles.loc[epoch, percentile]

    plot_scenarios(scenarios, reference_epoch=reference_epoch)

    plot_finish(xlabel=xlabel, ylabel=ylabel, title=title, grid=grid, legend=legend)

    plt.tight_layout()
    plt.savefig(outfile)


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser(parents=[kcs_parser],
                                     conflict_handler='resolve')
    parser.add_argument('percentiles', help="Input CSV file with CMIP distribution percentiles")
    parser.add_argument('steering_table', help="Input steering table CSV file")
    parser.add_argument('--outfile', required=True,
                        help="Output figure filename. The extension determines the file type.")
    parser.add_argument('--extra-data', nargs='+', help="Model of interest data files. Using this "
                        "option will overplot the average of the input data.")
    parser.add_argument('--relative', action='store_true', help="Indicate the model data "
                        "should be calculated as relative change, as opposed to absolute change.")
    parser.add_argument('--reference-period', type=int, nargs=2,
                        help="Reference period to normalize the extra data to (if input). "
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
    parser.add_argument('--legend', action='store_true')
    parser.add_argument('--grid', action='store_true')
    parser.add_argument('--extra-label', help="Label to indicate the extra model data.")
    parser.add_argument('--smooth', type=int, nargs='?', const=10)

    args = parser.parse_args()
    read_config(args.config)
    setup_logging(args.verbosity)

    if args.extra_data:
        args.extra_data = map(pathlib.Path, args.extra_data)
    if not args.reference_period:
        args.reference_period = default_config['data']['extra']['control_period']
    return args


def main():
    """Starting point when running as a script/module"""
    args = parse_args()
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    percentiles = pd.read_csv(args.percentiles, index_col=0)
    percentiles.index = pd.to_datetime(percentiles.index)
    steering_table = pd.read_csv(args.steering_table)
    steering_table['period'] = steering_table['period'].apply(
        lambda x: tuple(map(int, x.strip('()').split(','))))

    if args.extra_data:
        paths = list(itertools.chain.from_iterable(atlist(path) for path in args.extra_data))
        dataset = read_data(paths)
        extra_data = normalize_average_dataset(dataset['cube'], relative=args.relative,
                                               reference_period=args.reference_period)

    run(percentiles, steering_table, args.outfile, xlabel=args.xlabel, ylabel=args.ylabel,
        xrange=args.xrange, yrange=args.yrange, title=args.title, smooth=args.smooth,
        extra_data=extra_data, extra_label=args.extra_label,
        reference_epoch=args.reference_epoch, grid=args.grid, legend=args.legend)
    logger.info("Done processing")


if __name__ == '__main__':
    main()
