"""Calculate the "steering" table

Calculates the ranges for a given dataset, where it matches the CMIP
distribution for yearly temperature change for a given scenario(s).

For example, if the scenario is 90% (percentile), 2050, it will yield
a year-range for the given input data, where the average of that
year-range matches the 90% value in 2050 of the CMIP data. It will
also produce a scale factor to indicate how far off it is, which
should be close to 1. (This value may be significant, since the
year-range is rounded to years, or one may have to go beyond the range
of input data, if e.g. the requested range is large.)

Example usage:

$ python -m kcs.steering  tas_change.csv  @extra-tas-global-averaged.list \
      --scenario G 2050 10 --scenario W 2050 90 \
      --scenario G 2085 10 --scenario W 2085 90 \
       --rolling-mean 10 --outfile steering.csv

"""

import sys
import argparse
import logging
import pathlib
import itertools
import pandas as pd
import iris
from ..config import read_config, default_config
from ..utils.argparse import parser as kcs_parser
from ..utils.logging import setup as setup_logging
from ..utils.atlist import atlist
from ..utils.attributes import get as get_attrs
from .core import run


logger = logging.getLogger('steering')  # pylint: disable=invalid-name


def read_data(paths, info_from=('attributes', 'filename'),
              attributes=None, filename_pattern=None):
    """DUMMY DOC-STRING"""
    cubes = [iris.load_cube(str(path)) for path in paths]

    # Get the attributes, and create a dataframe with cubes & attributes
    dataset = get_attrs(
        cubes, paths, info_from=info_from,
        attributes=attributes, filename_pattern=filename_pattern)

    return dataset


def parse_args():
    """DUMMY DOC-STRING"""
    parser = argparse.ArgumentParser(parents=[kcs_parser],
                                     conflict_handler='resolve')
    parser.add_argument('csv', help="CSV file with distribution percentiles.")
    parser.add_argument('files', nargs='+', help="model of interest datasets")
    parser.add_argument('--scenario', required=True, nargs=3, action='append',
                        help="Specify a scenario. Takes three arguments: name, "
                        "epoch and percentile. This option is required, and can "
                        "be used multiple times. Examples would '--scenario G 2050 10', "
                        "'--scenario H 2085 90'.")
    parser.add_argument('--outfile', help="Output CSV file to write the steering "
                        "table to. If not given, write to standard output.")
    parser.add_argument('--timespan', type=int, default=30,
                        help="Timespan around epoch(s) given in the scenario(s), "
                        "in years. Default is 30 years.")
    parser.add_argument('--rolling-mean', default=0, type=int,
                        help="Apply a rolling mean to the percentile distribution "
                        "before computing the scenario temperature increase match. "
                        "Takes one argument, the window size (in years).")
    parser.add_argument('--rounding', type=float,
                        help="Round the matched temperature increase to a multiple "
                        "of this value, which should be a positive floating point "
                        "value. Default is not to round")
    parser.add_argument('--reference-period', nargs=2, type=int,
                        help="Reference period (to normalize our model of interestdata): "
                        "start and end year. Years are inclusive (i.e., Jan 1 of 'start' "
                        "up to and including Dec 31 of 'end').")

    args = parser.parse_args()
    setup_logging(args.verbosity)
    read_config(args.config)

    if not args.reference_period:
        args.reference_period = default_config['data']['extra']['control_period']
    args.paths = [pathlib.Path(filename) for filename in args.files]
    args.scenarios = [dict(zip(('name', 'epoch', 'percentile'), scenario))
                      for scenario in args.scenario]
    if args.rounding is not None:
        if args.rounding <= 0:
            raise ValueError('--rounding should be a positive number')
    return args


def main():
    """DUMMY DOCSTRING"""
    args = parse_args()
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    paths = list(itertools.chain.from_iterable(atlist(path) for path in args.paths))
    dataset = read_data(paths)

    percentiles = pd.read_csv(args.csv, index_col=0)
    percentiles.index = pd.to_datetime(percentiles.index)

    steering = run(dataset, percentiles, args.scenarios, timespan=args.timespan,
                   rolling_mean=args.rolling_mean, rounding=args.rounding,
                   reference_period=args.reference_period)
    steering = pd.DataFrame(steering)

    if args.outfile:
        steering.to_csv(args.outfile, index=False)
    else:
        print(steering)
    logger.info("Done processing: steering table = %s", steering)


if __name__ == '__main__':
    main()
