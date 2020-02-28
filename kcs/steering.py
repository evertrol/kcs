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

"""

import sys
import argparse
import logging
import pathlib
from datetime import datetime, timedelta
import itertools
import numpy as np
import pandas as pd
import iris
from iris.experimental.equalise_cubes import equalise_attributes
import cftime
import kcs.utils.argparse
import kcs.utils.logging
from kcs.utils.atlist import atlist
import kcs.utils.attributes
import kcs.utils.constraints


REFERENCE_PERIOD = (1981, 2010)

# If we run as a runnable module, use a more appropriate logger name
logname = 'kcs.distribution' if __name__ == '__main__' else __name__
logger = logging.getLogger(logname)


def num2date(coord, index=None):
    """DUMMY DOCSTRING"""
    if index is None:
        return cftime.num2date(coord.points, str(coord.units), coord.units.calendar)
    return cftime.num2date(coord.points[index], str(coord.units),
                           coord.units.calendar)


def read_data(paths, info_from=('attributes', 'filename'),
              attributes=None, filename_pattern=None):
    """DUMMY DOC-STRING"""
    cubes = [iris.load_cube(str(path)) for path in paths]

    # Get the attributes, and create a dataframe with cubes & attributes
    dataset = kcs.utils.attributes.get(
        cubes, paths, info_from=info_from,
        attributes=attributes, filename_pattern=filename_pattern)

    return dataset


def extract_season(cubes, season):
    """DUMMY DOC-STRING"""
    constraint = iris.Constraint(season=kcs.utils.constraints.EqualConstraint(season))
    logger.info("Extracting season %s", season)
    for cube in cubes:
        if not cube.coords('season'):
            iris.coord_categorisation.add_season(cube, 'time')
    cubes = list(map(constraint.extract, cubes))
    return cubes


def average_year_cube(cube, season=None):
    """DUMMY DOC-STRING"""
    if season:
        if not cube.coords('season_year'):
            iris.coord_categorisation.add_season_year(cube, 'time')
        mean = cube.aggregated_by('season_year', iris.analysis.MEAN)
    else:
        if not cube.coords('year'):
            iris.coord_categorisation.add_year(cube, 'time')
        mean = cube.aggregated_by('year', iris.analysis.MEAN)
    return mean


def average_year(cubes, season=None):
    """DUMMY DOC-STRING"""
    logger.info("Calculating %s averages", season if season else 'yearly')
    averages = []
    for cube in cubes:
        if season:
            if not cube.coords('season_year'):
                iris.coord_categorisation.add_season_year(cube, 'time')
            average = cube.aggregated_by('season_year', iris.analysis.MEAN)
        else:
            if not cube.coords('year'):
                iris.coord_categorisation.add_year(cube, 'time')
            average = cube.aggregated_by('year', iris.analysis.MEAN)
        averages.append(average)
    return averages


def calc_reference_values(dataset, reference_period, normby='run'):
    """Calculate the reference values for each cube

    If normby is not 'run', the individual reference values are averaged, so that each run
    will later be normalized by the same value.

    """

    constraint = iris.Constraint(year=kcs.utils.constraints.RangeConstraint(*reference_period))
    values = []
    for cube in dataset['cube']:
        if not cube.coords('year'):
            iris.coord_categorisation.add_year(cube, 'time')
        mean = constraint.extract(cube)
        mean = mean.collapsed('time', iris.analysis.MEAN)
        values.append(mean.data)
    values = np.array(values)
    if normby != 'run':
        values = np.full(values.shape, values.mean(), dtype=values.dtype)
    return values


def normalize(cubes, refvalues, relative):
    """Normalize a single cube to its reference value

    If the value is a relative value, i.e., a percentual change, set
    the 'relative' parameter to `True`.

    """

    for cube, refvalue in zip(cubes, refvalues):
        cube.data -= refvalue
        if relative:
            cube.data /= refvalue
            cube.data *= 100
            cube.units = '%'
    return cubes


def calc(dataset, distribution, scenarios, rolling_mean=0, rounding=0,
         timespan=30, maxepoch=2100):
    """Parameters
    ----------
    - dataset: Iris.cube.Cube

        Input data. Historical and future experiment data should be
        concatenated into one cube.  Data should be yearly-averaged
        and normalized. If there are more realizations, these should
        be merged (averaged) into a single cube.

    - distribution: Pandas DataFrame

        CMIP percentile distribution. The index is years, the columns
        are 'mean' and percentiles of interest (e.g., 5, 10, 25, 50,
        75, 90, 95). Calculated with `kcs.cmip.distribution.calc_percentiles`.

    - scenarios: list of dicts
      The dict should contain a name, year and percentile, e.g.
        scenarios=[{'name': 'G', 'epoch': 2050, 'percentile': 90},
                   {'name': 'L', 'epoch': 2050, 'percentile': 10},]
      The name should be unique.

    """

    if rolling_mean > 1:
        distribution = distribution.rolling(rolling_mean, center=True).mean().dropna()

    # Limit the dataset to 2085, so we don't try and calculate beyond 2100
    maxyear = distribution.index.max().year
    cube = iris.Constraint(year=lambda year: year <= maxyear).extract(dataset)

    for i, scenario in enumerate(scenarios):
        epoch = datetime(int(scenario['epoch']), 1, 1)
        percentile = scenario['percentile']
        delta_t = distribution.loc[epoch, percentile]
        if rounding:  # nearest multiple of `round`
            rem = delta_t % rounding
            if rounding - rem > rem:
                # Round down
                delta_t -= rem
            else:
                # Round up
                delta_t += rounding - rem

        index = np.argmin(np.abs((cube.data - delta_t).data))
        date = num2date(cube[index].coord('time'))[0]
        # Fix below to use proper years
        period = ((date - timedelta(timespan/2*365.24)).year,
                  (date + timedelta(timespan/2*365.24)).year)
        if period[1] > maxepoch:
            period = maxepoch - timespan, maxepoch
            epoch = datetime(maxepoch - timespan//2, 1, 1)
            delta = np.array(num2date(cube.coord('time')) - epoch, dtype=np.timedelta64)
            index = np.argmin(np.abs(delta))
        scenarios[i]['cmip_delta_t'] = delta_t
        # Correct for the fact that our previous calculations were all on January 1.
        # We simply equate that to Dec 12 of the previous year, and thus make the
        # end-year of the period inclusive
        scenarios[i]['period'] = period[0], period[1]-1
        model_delta_t = cube[index].data
        scenarios[i]['model_delta_t'] = model_delta_t
        scenarios[i]['factor'] = delta_t / model_delta_t

    return scenarios


def run(dataset, percentiles, scenarios, season=None, average_years=True,
        relative=False, reference_period=REFERENCE_PERIOD, normby='run',
        timespan=30, rolling_mean=0, rounding=None):
    """Calculate the percentile yearly change distribution for the input data

    Also performs extracting of season (optional), averaging of years
    (optional) and normalization to a common reference period (needed
    for a better inter-model comparison), before the percentiles are
    calculated.

    Returns
      2-tuple of

      - Percentiles, as Pandas DataFrame

      - Input dataset, but with possibly extracted seasons and
        averaged years, and normalized data

    """

    if season:
        dataset['cube'] = extract_season(dataset['cube'], season)
    if average_years:
        dataset['cube'] = average_year(dataset['cube'], season=season)
    refvalues = calc_reference_values(
        dataset, reference_period=reference_period, normby=normby)
    dataset['cubes'] = normalize(dataset['cube'], refvalues, relative=relative)

    cubes = iris.cube.CubeList(dataset['cubes'])
    equalise_attributes(cubes)
    cube2d = cubes.merge_cube()
    mean = cube2d.collapsed('realization', iris.analysis.MEAN)
    mean = mean.aggregated_by('year', iris.analysis.MEAN)

    steering = calc(mean, percentiles, scenarios, timespan=timespan,
                    rolling_mean=rolling_mean, rounding=rounding)
    return steering


def parse_args():
    """DUMMY DOC-STRING"""
    parser = argparse.ArgumentParser(parents=[kcs.utils.argparse.parser],
                                     conflict_handler='resolve')
    parser.add_argument('csv', help="CSV file with distribution percentiles.")
    parser.add_argument('files', nargs='+', help="EC-EARTH datasets")
    parser.add_argument('--scenario', required=True, nargs=3, action='append',
                        help="Specify a scenario. Takes three arguments: name, "
                        "epoch and percentile. This option is required, and can "
                        "be used multiple times")
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
    args = parser.parse_args()
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
    kcs.utils.logging.setup(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    paths = list(itertools.chain.from_iterable(atlist(path) for path in args.paths))
    dataset = read_data(paths)

    percentiles = pd.read_csv(args.csv, index_col=0)
    percentiles.index = pd.to_datetime(percentiles.index)

    steering = run(dataset, percentiles, args.scenarios, timespan=args.timespan,
                   rolling_mean=args.rolling_mean, rounding=args.rounding)
    steering = pd.DataFrame(steering)

    if args.outfile:
        steering.to_csv(args.outfile, index=False)
    else:
        print(steering)
    logger.info("Done processing: steering table = %s", steering)


if __name__ == '__main__':
    main()
