"""Calculate the percentile distribution for the input datasets.

The input datasets should correspond to a single variable, e.g., the
air temperature ('tas').

"""

import sys
import argparse
import logging
import warnings
import pathlib
import functools
from datetime import datetime
import re
import multiprocessing
import iris
import iris.analysis
import numpy as np
import pandas as pd
import kcs.utils.date
import kcs.utils.logging
import kcs.utils.argparse
import kcs.utils.constraints


HISTORICAL_KEY = 'historical'
REFERENCE_PERIOD = (1981, 2010)
MINDATA = {'historical': 20, 'future': 4}
PERC_PERIOD = (1950, 2100)

# If we run as a runnable module, use a more appropriate logger name
logname = 'kcs' if __name__ == '__main__' else __name__
logger = logging.getLogger(logname)


def read_single(path, experiments, historical=HISTORICAL_KEY, lazy=True):
    """TO DO; TO DO"""
    exp = None
    for experiment in experiments:
        if experiment in path.stem:
            exp = experiment
            break
    else:
        if historical in path.stem:
            exp = historical
    if not exp:
        logger.warning("No suitable experiment found in %s. Skipping dataset", path)
        return None
    experiment = exp

    cube = iris.load_cube(str(path))
    attrs = cube.attributes
    model = attrs['model_id']
    attr = attrs.get('experiment_id')
    if not attr:
        attr = attrs.get('experiment')
    if attr and attr != experiment:
        logger.warning("Experiment as found in attributes does not match that found "
                       "from the file name (%s). Skipping dataset %s", experiment, path)
        return None

    pattern = r'r(?P<realization>\d+)i(?P<initialization_method>\d+)p(?P<physics_version>\d+)'
    match = re.search(pattern, path.stem)
    if not match:
        logger.warning("RIP not found in %s. Skipping dataset", path)
        return None
    rip = []
    for key in ('realization', 'initialization_method', 'physics_version'):
        try:
            value = int(attrs[key])
            rip.append(value)
        except KeyError:
            logger.warning("Could not find '%s' in attributes of %s", key, path)
            continue
        except ValueError:
            logger.warning("Incorrect '%s' in attributes of %s: %s", key, path, attrs[key])
            continue
        if int(value) != int(match.group(key)):
            logger.warning("%s found in attributes does not match the one "
                           "deduced from the file name %s", key, path)
    rip = tuple(rip)
    #strings = path.stem.split("-")
    #rcp, realization = strings[-2:]
    #if rcp == 'historical':
    #    rcp = RCP_HISTORICAL
    #else:
    #    rcp = int(rcp[3:])
    #model = "-".join(strings[:-2])
    #logger.info('Reading %s (RCP %d - %s)', model, rcp, realization)
    #cube = iris.load_cube(str(path))

    if not lazy:     # Read the actual data (by 'touching' it), e.g., if
        cube.data    # we're dealing with multiprocessing
    return {'path': path, 'model': model, 'experiment': experiment, 'rip': rip, 'cube': cube}


def read_data(paths, experiments, historical=HISTORICAL_KEY, nproc=1):
    """TO DO; TO DO"""
    read_partial = functools.partial(read_single, experiments=experiments,
                                     historical=historical, lazy=(nproc == 1))
    if nproc == 1:
        data = list(filter(None, map(read_partial, paths)))
    else:
        with multiprocessing.Pool(nproc) as pool:
            data = list(filter(None, pool.map(read_partial, paths)))

    df = pd.DataFrame(data)
    return df


def extract_season(cubes, season, nproc=1):
    """TO DO; TO DO"""
    # Use a class instead of a lambda function, so we can pass the
    # constraint to multiprocessing (which doesn't handle lambdas).
    constraint = iris.Constraint(season=kcs.utils.constraints.EqualConstraint(season))
    logger.info("Extracting season %s", season)
    if nproc == 1:
        cubes = list(map(constraint.extract, cubes))
    else:
        with multiprocessing.Pool(nproc, maxtasksperchild=1) as pool:
            cubes = pool.map(constraint.extract, cubes)
    return cubes


def average_year_cube(cube):
    """TO DO; TO DO"""
    return cube.aggregated_by('year', iris.analysis.MEAN)


def average_year(cubes, nproc=1):
    """TO DO; TO DO"""
    logger.info("Calculating yearly averages")
    if nproc == 1:
        return list(map(average_year_cube, cubes))

    with multiprocessing.Pool(nproc, maxtasksperchild=1) as pool:
        cubes = pool.map(average_year_cube, cubes)
    return cubes


class ModelReferencePointCalculation:
    """TO DO; TO DO"""

    def __init__(self, data, historical_key=HISTORICAL_KEY, yearly=True, season=False,
                 reference_period=REFERENCE_PERIOD,
                 ):
        self.data = data
        self.historical_key = historical_key
        if yearly:
            self.mindata = MINDATA
        elif season in ['djf', 'mam', 'jja', 'son']:
            # Three months a year
            self.mindata = {key: 3*value for key, value in MINDATA.items()}
        else:
            # Twelve months a year
            self.mindata = {key: 12*value for key, value in MINDATA.items()}
        self.constraint = kcs.utils.date.make_year_constraint_all_calendars(*reference_period)
        #self.constraint = {'hist': make_year_constraint_all_calendars(1981, 2005),
        #                   'rcp': make_year_constraint_all_calendars(2006, 2010)}

    def __call__(self, model):
        """TO DO; TO DO"""
        data = self.data[self.data['model'] == model]

        avs = {}
        ndata = {}
        mean = {}
        cubes = data.loc[self.data['experiment'] == self.historical_key, 'cube']
        avs['historical'] = self.calc_mean(cubes, self.constraint,
                                           self.mindata['historical'], model)
        cubes = self.data.loc[self.data['experiment'] != self.historical_key, 'cube']
        avs['future'] = self.calc_mean(cubes, self.constraint,
                                       self.mindata['future'], model)

        if not avs['historical'] or not avs['future']:  # Too few data to calculate a decent bias
            logging.warning("%s does not have enough data to compute a reference", model)
            return None

        for key, values in avs.items():
            n = len(values)
            # Weighted time means for each section
            ndata[key] = sum(value[1] for value in values) / n
            mean[key] = sum(value[0].data for value in values) / n
        reference = ((mean['historical'] * ndata['historical'] +
                      mean['future'] * ndata['future']) /
                     (ndata['historical'] + ndata['future']))

        return reference

    def calc_mean(self, cubes, constraint, mindata, model):
        """TO DO; TO DO"""
        averages = []
        for cube in cubes:
            calendar = cube.coord('time').units.calendar
            excube = constraint[calendar].extract(cube)
            if excube is None:
                logging.warning("A cube of %s does not support time range: %s",
                                model, cube.coord('time'))
                continue
            ndata = len(excube.coord('time').points)
            # Less than 20 years of data? Ignore
            if ndata < mindata:
                logging.warning("A cube of %s has only %d data points for its historical "
                                "time range", model, ndata)
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning,
                    message="Collapsing a non-contiguous coordinate. "
                    "Metadata may not be fully descriptive for 'year'")
                averages.append((excube.collapsed('time', iris.analysis.MEAN), ndata))
        return averages


def calc_reference_values(data, yearly=False, season=False,
                          historical_key=HISTORICAL_KEY,
                          reference_period=REFERENCE_PERIOD,
                          nproc=1):
    """Calculate reference values *per model*, so that each realization is
    scaled to the reference period following the average *model* reference
    value

    """

    logger.info("Calculating reference values (period = %s)", reference_period)
    calculation = ModelReferencePointCalculation(
        data, yearly=yearly, season=season, historical_key=historical_key,
        reference_period=reference_period)

    models = data['model'].unique()
    if nproc == 1:
        reference_values = filter(None, map(calculation, models))
    else:
        with multiprocessing.Pool(nproc, maxtasksperchild=1) as pool:
            reference_values = filter(None, pool.map(calculation, models))

    return dict(zip(models, reference_values))


def correct_cube(item, relative=False):
    """TO DO; TO DO"""
    cube, zeropoint = item
    cube.data -= zeropoint
    if relative:
        cube.data /= zeropoint
        cube.data *= 100
        #cube.attributes = cube.attributes
        cube.units = '%'
    return cube


def correct(cubes, reference_values, relative=False):
    """TO DO; TO DO"""
    logger.info("Correcting data for zeropoint")
    cubes = zip(cubes, reference_values)
    correct_partial = functools.partial(correct_cube, relative=relative)
    cubes = list(map(correct_partial, cubes))
    return cubes


def calc_percentile_year(cubes, year):
    """Calculate the percentile distribution of the cubes for a given year"""

    constraint = iris.Constraint(year=lambda point: point == year)
    cubes = filter(None, map(constraint.extract, cubes))
    data = np.array([cube.data for cube in cubes])
    mean = data.mean()
    percs = np.percentile(data, [5, 10, 25, 50, 75, 90, 95], overwrite_input=True)
    return dict(zip(['mean', '5', '10', '25', '50', '75', '90', '95'],
                    [mean] + percs.tolist()))


def calc_percentiles(cubes, period=PERC_PERIOD, nproc=1):
    """TO DO; TO DO"""
    logger.info("Calculating percentiles")
    years = list(range(*period))
    func = functools.partial(calc_percentile_year, cubes)
    if nproc == 1:
        percs = list(map(func, years))
    else:
        with multiprocessing.Pool(nproc, maxtasksperchild=1) as pool:
            percs = pool.map(func, years)
    return pd.DataFrame(
        percs, index=pd.DatetimeIndex([datetime(year, 1, 1) for year in years]))


def run(paths, experiments, historical_key, season=None, average_years=True,
        relative=False, reference_period=REFERENCE_PERIOD,
        percentile_period=PERC_PERIOD, nproc=1):
    """TO DO; TO DO"""
    df = read_data(paths, experiments, historical_key, nproc=nproc)
    df['fname'] = df['path'].apply(lambda x: x.stem)
    df['cubesumm'] = df['cube'].apply(lambda x: x.summary(shorten=True))
    df['l'] = df['cube'].apply(lambda x: len(x.coord('time').points))

    df['lex'] = -1
    if season:
        df['cube'] = extract_season(df['cube'], season, nproc=nproc)
        df['lex'] = df['cube'].apply(lambda x: len(x.coord('time').points))

    df['lav'] = -1
    if average_years:
        df['cube'] = average_year(df['cube'], nproc=nproc)
        df['lav'] = df['cube'].apply(lambda x: len(x.coord('time').points))

    reference_values = calc_reference_values(
        df, yearly=average_years, season=season,
        historical_key=historical_key,
        reference_period=reference_period,
        nproc=nproc)

    df['reference_values'] = df['model'].map(reference_values)
    df['cube'] = correct(df['cube'], df['reference_values'], relative=relative)

    percentiles = calc_percentiles(df['cube'], period=percentile_period, nproc=nproc)

    return percentiles


def parse_args():
    """TO DO; TO DO"""
    parser = argparse.ArgumentParser(parents=[kcs.utils.argparse.parser],
                                     conflict_handler='resolve')
    parser.add_argument('files', nargs='+', help="Input file paths")
    parser.add_argument('--experiment', action='append', required=True,
                        help="Experiment identifier (e.g., 'rcp45', 'ssp78'). "
                        "This option can be used multiple times.")
    parser.add_argument('--historical', default=HISTORICAL_KEY,
                        help="Identifier for the historical experiment "
                        f"(default '{HISTORICAL_KEY}').")
    parser.add_argument('--percentile-period', nargs=2, type=int, default=PERC_PERIOD,
                        help="Period (years, inclusive) for which to calculate "
                        "the percentiles. Default is 1950-2100.")
    parser.add_argument('--relative', action='store_true',
                        help="Calculate relative change (values will be "
                        "a percentage change between future and reference period")
    parser.add_argument('--season', choices=['djf', 'mam', 'jja', 'son'],
                        help="Season to extract / use. Leave blank to "
                        "use full years")
    parser.add_argument('--no-year-average', action='store_true',
                        help="Do not use yearly averages")
    parser.add_argument('--reference-period', nargs=2, type=int,
                        default=list(REFERENCE_PERIOD),
                        help="Reference period: start and end year. Years are "
                        "inclusive (i.e., Jan 1 of 'start' up to and "
                        f"including Dec 31 of 'end'). Default {REFERENCE_PERIOD}.")
    args = parser.parse_args()
    args.paths = [pathlib.Path(filename) for filename in args.files]
    args.average_years = not args.no_year_average
    return args


def main():
    """TO DO; TO DO"""
    args = parse_args()
    kcs.utils.logging.setup(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)
    result = run(args.paths, args.experiment, historical_key=args.historical,
                 season=args.season, average_years=args.average_years,
                 relative=args.relative, reference_period=args.reference_period,
                 percentile_period=args.percentile_period,
                 nproc=args.nproc)
    logger.info("Done processing: percentiles = %s", result)


if __name__ == '__main__':
    main()
