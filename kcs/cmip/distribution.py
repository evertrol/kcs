"""Calculate the (CMIP) percentile distribution for the given input
datasets.

The input datasets should correspond to a single variable, e.g., the
air temperature ('tas').

"""

import sys
import argparse
import logging
import warnings
import pathlib
import functools
from itertools import chain
from datetime import datetime
import multiprocessing
from pprint import pformat
import iris
import iris.cube
import iris.analysis
import iris.util
import iris.experimental.equalise_cubes
import iris.coord_categorisation
import iris.exceptions
import numpy as np
import pandas as pd
import kcs.utils.date
import kcs.utils.logging
import kcs.utils.argparse
import kcs.utils.constraints
import kcs.utils.attributes
import kcs.utils.matching
from kcs.utils.atlist import atlist


HISTORICAL_KEY = 'historical'
REFERENCE_PERIOD = (1981, 2010)
MINDATA = {'historical': 20, 'future': 4}
PERC_PERIOD = (1950, 2100)


# If we run as a runnable module, use a more appropriate logger name
logname = 'kcs.distribution' if __name__ == '__main__' else __name__
logger = logging.getLogger(logname)


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
    # Use a class instead of a lambda function, so we can pass the
    # constraint to multiprocessing (which doesn't handle lambdas).
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


def average_year(cubes, season=None, nproc=1):
    """DUMMY DOC-STRING"""
    if season:
        logger.info("Calculating %s averages", season)
    else:
        logger.info("Calculating yearly averages")
    func = functools.partial(average_year_cube, season=season)
    if nproc > 1:
        with multiprocessing.Pool(nproc) as pool:
            cubes = pool.map(func, cubes)
    else:
        cubes = list(map(func, cubes))
    return cubes


class ModelReferencePointCalculation:
    """DUMMY DOC-STRING"""

    def __init__(self, dataset, historical_key=HISTORICAL_KEY, yearly=True, season=None,
                 reference_period=REFERENCE_PERIOD, normby='run'):
        self.dataset = dataset
        self.historical_key = historical_key
        self.normby = normby
        if yearly:
            self.mindata = MINDATA
        elif season:  # in ['djf', 'mam', 'jja', 'son']:
            # Three months a year
            self.mindata = {key: 3*value for key, value in MINDATA.items()}
        else:
            # Twelve months a year
            self.mindata = {key: 12*value for key, value in MINDATA.items()}
        self.constraint = kcs.utils.date.make_year_constraint_all_calendars(*reference_period)

    def __call__(self, model):
        """DUMMY DOC-STRING"""
        dataset = self.dataset[self.dataset['model'] == model]

        if self.normby == 'model':
            cubes = dataset['cube']
            histcubes = dataset['match_historical_run']

            value = self.calc_refvalue(cubes, histcubes, model)
        elif self.normby == 'experiment':
            value = {}
            for exp, group in dataset.groupby('experiment'):
                cubes = group['cube']
                histcubes = group['match_historical_run']
                value[exp] = self.calc_refvalue(cubes, histcubes, model)
        else:
            value = {}
            for index, row in dataset.iterrows():
                cubes = [row['cube']]
                histcubes = [row['match_historical_run']]
                value[index] = self.calc_refvalue(cubes, histcubes, model)

        return value

    def calc_refvalue(self, cubes, histcubes, model):
        """DUMMY DOC-STRING"""
        avs = {}
        avs['historical'] = self.calc_mean(histcubes, self.mindata['historical'], model)
        avs['future'] = self.calc_mean(cubes, self.mindata['future'], model)
        if not avs['historical'] or not avs['future']:  # Too few data to calculate a decent bias
            logger.warning("%s does not have enough data to compute a reference", model)
            return None
        logger.info("Calculating time-weighted reference value")
        ndata = {}
        mean = {}
        for key, values in avs.items():
            n = len(values)
            # Weighted time means for each section
            ndata[key] = sum(value[1] for value in values) / n
            mean[key] = sum(value[0].data for value in values) / n
        logger.debug("Reference data values: %s , with weights %s", pformat(mean), pformat(ndata))
        value = ((mean['historical'] * ndata['historical'] +
                  mean['future'] * ndata['future']) /
                 (ndata['historical'] + ndata['future']))

        return value

    def calc_mean(self, cubes, mindata, model):
        """DUMMY DOC-STRING"""
        averages = []
        for cube in cubes:
            calendar = cube.coord('time').units.calendar
            excube = self.constraint[calendar].extract(cube)
            if excube is None:
                logger.warning("A cube of %s does not support time range: %s",
                               model, cube.coord('time'))
                continue
            ndata = len(excube.coord('time').points)
            # Not enough of data? Ignore
            if ndata < mindata:
                logger.warning("A cube of %s has only %d data points for its time range",
                               model, ndata)
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=UserWarning,
                    message="Collapsing a non-contiguous coordinate. "
                    "Metadata may not be fully descriptive for 'year'")
                averages.append((excube.collapsed('time', iris.analysis.MEAN), ndata))
        return averages


def calc_reference_values(dataset, yearly=False, season=None,
                          historical_key=HISTORICAL_KEY,
                          reference_period=REFERENCE_PERIOD,
                          normby='run'):
    """Calculate reference values

    model:  *per model*, so that each realization is
    scaled to the reference period following the average *model* reference
    value

    """

    logger.info("Calculating reference values (period = %s)", reference_period)
    index = dataset['index_match_run']
    hindex = index[index > -1]
    cubes = dataset.loc[hindex, 'cube'].array
    future_data = dataset[index > -1].copy()
    future_data['match_historical_run'] = cubes

    calculation = ModelReferencePointCalculation(
        future_data, yearly=yearly, season=season, historical_key=historical_key,
        reference_period=reference_period, normby=normby)

    models = dataset['model'].unique()
    reference_values = filter(None, map(calculation, models))

    if normby == 'model':
        ref_values = dataset['model'].map(dict(zip(models, reference_values)))
    elif normby == 'experiment':
        ref_values = []
        for model, values in zip(models, reference_values):
            sel = (dataset['model'] == model)
            experiments = dataset.loc[sel, 'experiment']
            ref_values.append(experiments.map(values))
        ref_values = pd.concat(ref_values)
    else:
        ref_values = pd.concat([pd.Series(values) for values in reference_values])

    return ref_values


def normalize_cube(item, relative=False):
    """Normalize a single cube to its reference value

    If the value is a relative value, i.e., a percentual change, set
    the 'relative' parameter to `True`.

    """

    cube, ref_value = item
    cube.data -= ref_value
    if relative:
        cube.data /= ref_value
        cube.data *= 100
        cube.units = '%'
    return cube


def normalize(dataset, relative=False, normby='run'):
    """Normalize cubes to their reference values

    If the value is a relative value, i.e., a percentual change, set
    the 'relative' parameter to `True`.

    This returns a new, modified dataset.

    """

    dataset['matched_exp'] = ''
    if normby != 'model':
        # Add the (double/triple/etc) historical runs for the future experiments,
        # and get rid of the old historical runs
        # We do this by
        # - selecting the indices & reference values for the matching runs
        # - copy the relevant data from the dataset
        #   We need to copy the Iris cubes, otherwise we'll have
        #   multiple references to the same cube
        # - Set the reference values and the matching future indices
        #   for the newly copied historical data
        # - Concatenate the current and new datasets, and drop all
        #   rows that don't have a reference value (original
        #   historical rows)

        sel = dataset['index_match_run'] > -1
        indices = dataset.loc[sel, 'index_match_run']

        #match = dataset.loc[sel].index
        reference_values = dataset.loc[sel, 'reference_value']
        hist_data = dataset.loc[indices, :]
        hist_data['cube'] = hist_data['cube'].apply(lambda cube: cube.copy())
        hist_data['reference_value'] = reference_values.array
        hist_data['index_match_run'] = dataset.loc[sel].index.array
        hist_data['matched_exp'] = dataset.loc[sel, 'experiment'].array
        dataset = pd.concat([dataset, hist_data])
        dataset.dropna(axis=0, subset=['reference_value'], inplace=True)
    logger.info("Normalizing data to the reference period")
    # Tempting, but it's not possible to simply do dataset['cube'] =
    # dataset['cube'] / data['reference_value']
    func = functools.partial(normalize_cube, relative=relative)
    data = zip(dataset['cube'], dataset['reference_value'])
    dataset['cube'] = list(map(func, data))
    return dataset


def calc_percentile_year(dataset, year, average_experiments=False):
    """Calculate the percentile distribution of the cubes for a given year"""

    constraint = iris.Constraint(year=kcs.utils.constraints.EqualConstraint(year))

    #for cube in dataset['cube']:
    #    if not cube.coords('year'):
    #        iris.coord_categorisation.add_year(cube, 'time')

    if average_experiments:
        data = []
        for _, group in dataset.groupby(['model', 'experiment', 'matched_exp']):
            cubes = list(filter(None, map(constraint.extract, group['cube'])))
            if cubes:
                data.append(np.mean([cube.data for cube in cubes]))
    else:
        cubes = list(filter(None, map(constraint.extract, dataset['cube'])))
        data = [cube.data for cube in cubes]
    mean = np.mean(data)
    percs = np.percentile(data, [5, 10, 25, 50, 75, 90, 95], overwrite_input=True)
    return dict(zip(['mean', '5', '10', '25', '50', '75', '90', '95'],
                    [mean] + percs.tolist()))


def calc_percentiles(dataset, period=PERC_PERIOD, average_experiments=False, nproc=1):
    """DUMMY DOC-STRING"""

    logger.info("Calculating percentiles")

    years = list(range(*period))
    func = functools.partial(calc_percentile_year, dataset, average_experiments=average_experiments)
    if nproc == 1:
        percs = list(map(func, years))
    else:
        with multiprocessing.Pool(nproc, maxtasksperchild=1) as pool:
            percs = pool.map(func, years)
    return pd.DataFrame(
        percs, index=pd.DatetimeIndex([datetime(year, 1, 1) for year in years]))


def run(dataset, historical_key, season=None, average_years=True,
        relative=False, reference_period=REFERENCE_PERIOD,
        period=PERC_PERIOD, normby='run', average_experiments=False):
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

    reference_values = calc_reference_values(
        dataset, yearly=average_years, season=season,
        historical_key=historical_key,
        reference_period=reference_period, normby=normby)
    dataset['reference_value'] = reference_values
    dataset = normalize(dataset, relative=relative, normby=normby)

    ## Print statement for verification
    #with pd.option_context('max_rows', 999):
    #    for _, group in dataset.groupby(['model', 'experiment', 'matched_exp']):
    #        columns = ['model', 'experiment', 'realization', 'initialization', 'physics']
    #        print(group.sort_values(columns)[
    #            columns + ['prip', 'index_match_run', 'reference_value', 'matched_exp']])

    percentiles = calc_percentiles(dataset, period=period,
                                   average_experiments=average_experiments)

    return percentiles, dataset


def parse_args():
    """DUMMY DOC-STRING"""
    parser = argparse.ArgumentParser(parents=[kcs.utils.argparse.parser],
                                     conflict_handler='resolve')
    parser.add_argument('files', nargs='+', help="Input file paths")
    parser.add_argument('--outfile', default="distribution-percentiles.csv",
                        help="Output CSV file with the distribution "
                        "percentiles-versus-year table. "
                        "The default is 'distribution-percentiles.csv'")
    parser.add_argument('--historical-key', default=HISTORICAL_KEY,
                        help="Identifier for the historical experiment "
                        f"(default '{HISTORICAL_KEY}').")
    parser.add_argument('--period', nargs=2, type=int, default=PERC_PERIOD,
                        help="Period (years, inclusive) for which to calculate "
                        "the percentile distribution. Default is 1950-2100.")
    parser.add_argument('--relative', action='store_true',
                        help="Calculate relative change (values will be "
                        "a percentage change between future and reference period")
    parser.add_argument('--season', choices=['djf', 'mam', 'jja', 'son'],
                        help="Season to extract / use. Leave blank to "
                        "use full years")
    parser.add_argument('--no-year-average', action='store_true',
                        help="Do not use yearly/seasonal averages")
    parser.add_argument('--reference-period', nargs=2, type=int,
                        default=list(REFERENCE_PERIOD),
                        help="Reference period: start and end year. Years are "
                        "inclusive (i.e., Jan 1 of 'start' up to and "
                        f"including Dec 31 of 'end'). Default {REFERENCE_PERIOD}.")
    parser.add_argument('--norm-by', choices=['model', 'experiment', 'run'],
                        default='run',
                        help="Normalize data (to --reference-period) per 'model', "
                        "'experiment' or 'run'. These normalization choices are from "
                        "'wide' (large spread) to 'narrow' (little spread).")
    parser.add_argument('--match-by', choices=['model', 'ensemble'], default='ensemble',
                        help="Match future and historical runs by model (very generic) "
                        "or ensemble (very specific). Default is 'ensemble'.")
    parser.add_argument('--match-info-from', choices=['attributes', 'filename'], nargs='+',
                        default=['attributes', 'filename'],
                        help="Where to get the information from to match runs. 'attributes' "
                        "will use the NetCDF attributes, 'filename' will attempt to deduce "
                        "them from the filename. Both can be given, in order of importance, "
                        "that is, the second then serves as a fallback. "
                        "Default is 'attributes filename'. "
                        "Important attributes are 'parent-experiment-rip', 'realization', "
                        "'physics_version', 'initialization_method', 'experiment', 'model_id'")
    parser.add_argument('--on-no-match', choices=['error', 'remove', 'randomrun', 'random'],
                        default='error', help="What to do with a (future) experiment run that "
                        "has no matching run. 'error' raise an exception, 'remove' removes the "
                        "run. 'randomrun' picks a random history run with the same 'physics' and "
                        "'initialization' values (but a different 'realization' value), while "
                        "'random' picks a random history run from all ensembles for that model.")
    parser.add_argument('--average-experiments', action='store_true', help="Average ensemble "
                        "runs over their model-experiment, before calculating percentiles.")
    args = parser.parse_args()
    args.paths = [pathlib.Path(filename) for filename in args.files]
    args.average_years = not args.no_year_average
    return args


def main():
    """DUMMY DOC-STRING"""
    args = parse_args()
    kcs.utils.logging.setup(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    paths = list(chain.from_iterable(atlist(path) for path in args.paths))
    dataset = read_data(paths)

    dataset = kcs.utils.matching.match(
        dataset, match_by=args.match_by, on_no_match=args.on_no_match,
        historical_key=args.historical_key)

    result, _ = run(dataset, historical_key=args.historical_key,
                    season=args.season, average_years=args.average_years,
                    relative=args.relative, reference_period=args.reference_period,
                    period=args.period, normby=args.norm_by,
                    average_experiments=args.average_experiments)
    result.to_csv(args.outfile, index_label="date")
    logger.info("Done processing: percentiles = %s", result)


if __name__ == '__main__':
    main()
