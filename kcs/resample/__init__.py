"""DUMMY DOCSTRING"""

import math
import re
import pathlib
import glob
import itertools
import functools
from datetime import datetime
import multiprocessing
import logging
import numpy as np
import pandas as pd
import iris
from ..config import default_config


ALLSEASONS = ['djf', 'mam', 'jja', 'son']
STATS = ['mean', '5', '10', '25', '50', '75', '90', '95']


logger = logging.getLogger(__name__)   # pylint: disable=invalid-name


def segment_data(cubes, period, control_period, nsections, seasons=None):
    """Given a list of cubes (or CubeList), return a dict with periods and seasons extracted"""

    if seasons is None:
        seasons = ALLSEASONS

    data = {season: {} for season in seasons}
    # pragma pylint: disable=cell-var-from-loop
    for key, years in zip(['control', 'future'], [control_period, period]):
        # Chop each cube into n-year segments
        span = (years[1] - years[0] + 1) // nsections
        constraint = iris.Constraint(year=lambda point: years[0] <= point <= years[1])
        logger.debug("Extraction %s period %s for all datasets", key, years)
        excubes = [constraint.extract(cube) for cube in cubes]
        for season in seasons:
            logger.debug("Extraction season %s for all datasets", season)
            constraint = iris.Constraint(season=lambda point: point == season)
            season_cubes = [constraint.extract(cube) for cube in excubes]
            logger.debug("Extracting %d-year segments for all datasets", span)
            data[season][key] = np.array([
                [iris.Constraint(year=lambda point: year <= point < year+span).extract(cube)
                 for year in range(years[0], years[1], span)]
                for cube in season_cubes], dtype=np.object)
    # pragma pylint: enable=cell-var-from-loop

    return data


def create_indices(nsets, nsections):
    """Create an array of indices for all possible resamples

    This results in an array with a shape of (nsets^nsections,
    nsections). Each row is a separate resample, indexing the relevant
    segment from a dataset i.

    """

    indices = list([range(nsets) for _ in range(nsections)])
    indices = np.array(list(itertools.product(*indices)))
    assert len(indices) == nsets**nsections, "incorrect number of generated indices"

    return indices


class Calculation:
    """Class to calculate the difference between a calculated precipiation
    change, and a target value

    We use a class, so it can be used with multiprocessing: fixed
    arguments are passed to the constructor, while the variable
    argument (the subselection of the data) is passed to the call
    method.

    An alternative is the use of functools.partial.

    """

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cols = np.arange(data.shape[1])

    def __call__(self, index):
        return np.abs(self.data[index, self.cols].mean() - self.target)


def calc_means(data, seasons=None):
    """Calculate the averages of all n-year segments, and store them in a
    dataframe

    The dataframe columns correspond to the n-year segments (simply
    numbered 0 to ndata-1).

    The dataframe index consists of the variable, season and period
    (control or future), and the individual datasets (runs) as a
    multi-index.

    The values then, of course, are the averages for that specific
    n-year segment, variable, season, period and dataset.

    """

    if seasons is None:
        seasons = ALLSEASONS
    means = {}
    std = {}
    for var in data:
        for season in seasons:
            for period in ['control', 'future']:
                item = data[var][season][period]
                for i, cubelist in enumerate(item):
                    means[(var, season, period, i)] = [cube.data.mean() for cube in cubelist]
                    std[(var, season, period, i)] = [cube.data.std() for cube in cubelist]

    df_mean = pd.DataFrame.from_dict(means, orient='index')
    df_mean.index = pd.MultiIndex.from_tuples(means.keys(),
                                              names=['var', 'season', 'period', 'run'])
    df_mean = df_mean.sort_index()

    df_std = pd.DataFrame.from_dict(std, orient='index')
    df_std.index = pd.MultiIndex.from_tuples(means.keys(),
                                             names=['var', 'season', 'period', 'run'])
    df_std = df_std.sort_index()

    return df_mean, df_std


def prepare_data(dataset, variables, period, control_period, nsections):
    """Prepare the data for a scenario

    - segment the data into nsections.

    - calculate the means for variables and seasons of interest, for
      all individual ensemble runs, for each n-year period.

    """

    data = {}
    ndata = set()
    for var in variables:
        cubes = dataset.loc[dataset['var'] == var, 'cube']
        logger.debug("Segmenting %s data into %d sections", var, nsections)
        data[var] = segment_data(cubes, period, control_period, nsections)
        season = list(data[var].keys())[0]
        segment = data[var][season]['future']
        ndata.add(len(segment))

    assert len(ndata) == 1, "Datasets are not the same length for different variables"
    ndata = ndata.pop()
    indices = create_indices(ndata, nsections)

    logger.debug("start")
    means, _ = calc_means(data)
    logger.debug("stop")

    return data, indices, means, ndata


def calculate_s1(means, indices, target, var='pr', season='djf', nproc=None):
    """Calculate the subset S1: winter precipitation change equals `target`"""

    if not nproc:
        nsections = default_config['resampling']['nproc']

    # Calculate the procentual change in winter precipitation for all sections individually,
    # With respect to an overall-averaged control period
    controlmean = means.loc[(var, season, 'control'), :]
    controlmean = controlmean.mean().mean()
    data = means.loc[(var, season, 'future'), :].values
    data = 100 * (data - controlmean) / controlmean

    calculation = Calculation(data, target)
    tstart = datetime.now()
    if nproc == 1:
        values = list(map(calculation, indices))
    else:
        with multiprocessing.Pool(nproc) as pool:
            values = pool.map(calculation, indices)
    logger.debug("time(calculation) = %s", datetime.now() - tstart)
    values = np.array(values)

    tstart = datetime.now()
    order = np.argsort(values)

    selind = {'future': indices[order]}

    data = means.loc[(var, season, 'control'), :].values
    data = data - controlmean
    calculation = Calculation(data, 0)
    tstart = datetime.now()
    if nproc == 1:
        values = list(map(calculation, indices))
    else:
        with multiprocessing.Pool(nproc) as pool:
            values = pool.map(calculation, indices)
    logger.debug("time(calculation) = %s", datetime.now() - tstart)
    values = np.array(values)

    tstart = datetime.now()
    order = np.argsort(values)

    selind['control'] = indices[order]

    return selind


def calculate_s2(means, indices, scenarios):
    """Calculate the subset S2: select percentile ranges for average
    precipitation and temperatures"""

    s2_indices = {}
    columns = np.arange(indices['control'].shape[1])
    for period in ['control', 'future']:
        selection = np.ones(indices[period].shape[0], dtype=np.bool)
        s2_indices[period] = indices[period][selection]
        for scenario in scenarios:
            var = scenario['var']
            season = scenario['season']
            logger.debug("Subsetting with percentiles for %s, %s, %s", var, season, period)
            values = means.loc[(var, season, period), :].values[s2_indices[period], columns]
            # Calculate mean along the columns, i.e., one 30-year period
            mean = values.mean(axis=1)
            logger.debug("Min, max, mean, median values: %f  %f  %f  %f",
                         mean.min(), mean.max(), mean.mean(), np.median(mean))
            p_range = scenario[period]
            logger.debug("Percentile range = %s", p_range)
            low, high = np.percentile(mean, p_range)
            logger.debug("Percentile values = %f  --  %f", low, high)
            selection = (low <= mean) & (mean <= high)
            s2_indices[period] = s2_indices[period][selection]
            logger.debug("Subsetting down to %d samples", selection.sum())


    return s2_indices


def calculate_s3(indices_dict, penalties, nstep3=None, nsample=None,
                 minimum_penalty=None):
    """Calculate the subset S3: find a subset with the least re-use of
    segments, using random sampling"""

    if not nstep3:
        nstep3 = default_config['resampling']['nstep3']
    if not nsample:
        nsample = default_config['resampling']['nsample']
    if minimum_penalty is None:
        minimum_penalty = np.finfo(float).eps

    s3_indices = {}
    rng = np.random.default_rng()   # pylint: disable=no-member
    for period, indices in indices_dict.items():
        n = len(indices)  # pylint: disable=invalid-name
        m = math.factorial(n) // math.factorial(nstep3) // math.factorial(n - nstep3)
        logger.debug("Number of combinations for the %s period = %d", period, m)
        best = np.inf
        for i in range(nsample):
            choice = rng.choice(indices, size=nstep3, replace=False)
            penalty = 0
            for column in choice.T:
                _, counts = np.unique(column, return_counts=True)
                for count in counts:
                    penalty += penalties[count]
            if penalty <= best:
                best = penalty
                s3_indices[period] = choice
            if best < minimum_penalty:
                logger.debug("Minimum penalty reached after %d iterations", i)
                break
        logger.info("Best sample for %s (penalty: %f) out of %d samples: %s",
                    period, best, nsample, s3_indices[period])

    return s3_indices


def find_resamples(indices, means, precip_change, ranges, penalties,
                   nstep1=None, nstep3=None, nsample=None, nproc=None):
    """Find the (best) resamples

    This does the actual work:

    - run step one (for both future and control), limiting samples by
      required precipitation change

    - run step two, limiting samples by requiring their mean values
      to be in certain percentile change compared to the overall mean
      values of the remaining resamples.

    - run step three, by randomly selecting samples, keeping the
      number of duplicate segments to a minimum.

    Note that the actual data cubes are not required: we have already
    calculated the averages, and those are used in step one and two
    (step three doesn't require any actual data).

    """
    logger.debug("Calculating S1")
    logger.debug("Precipitation change: %.1f", precip_change)
    if not nstep1:
        nstep1 = default_config['resampling']['nstep1']
    if not nstep3:
        nstep3 = default_config['resampling']['nstep3']
    if not nsample:
        nsample = default_config['resampling']['nsample']
    if not nproc:
        nsections = default_config['resampling']['nproc']

    indices = calculate_s1(means, indices, precip_change, nproc=nproc)
    indices['control'] = indices['control'][:nstep1]
    indices['future'] = indices['future'][:nstep1]

    indices = calculate_s2(means, indices, ranges)
    logger.debug("The S2 subset has %d & %d indices for the control & future periods, resp.",
                 len(indices['control']), len(indices['future']))

    indices = calculate_s3(indices, penalties, nstep3=nstep3, nsample=nsample)

    return indices


def resample(indices, data, variables, seasons, relative, means):
    """Perform the actual resampling of data, given the resampled indices"""

    percs = list(map(float, STATS[1:]))

    diffs = {}
    for key, value in indices.items():
        diffs[key] = {}
        tempindices = value['data']
        for var in variables:
            diffs[key][var] = {}
            for season in seasons:
                stats = {}
                for period in ['control', 'future']:
                    cubes = data[key][var][season][period]
                    columns = np.arange(cubes.shape[1])
                    resampled = cubes[tempindices[period], columns]
                    stats[period] = pd.DataFrame({'mean': 0, '5': 0, '10': 0, '25': 0, '50': 0,
                                                  '75': 0, '90': 0, '95': 0},
                                                 index=np.arange(len(resampled)))
                    for i, cubeset in enumerate(resampled):
                        block = np.hstack([cube.data for cube in cubeset])
                        stats[period].loc[i, 'mean'] = block.mean()
                        stats[period].loc[i, STATS[1:]] = np.percentile(block, percs)

                diff = pd.DataFrame()
                for col in STATS:
                    diff[col] = stats['future'][col] - stats['control'][col]
                    if var in relative:
                        diff[col] = 100 * diff[col] / stats['control'][col]
                diffs[key][var][season] = diff
    return diffs


def run(dataset, steering_table, ranges, penalties,
        nstep1=None, nstep3=None, nsample=None,
        nsections=None, reference_period=None,
        relative=None, nproc=None):
    """DUMMY DOCSTRING"""

    if relative is None:
        relative = default_config['variables']['relative']
    if reference_period is None:
        reference_period = default_config['data.extra']['control_period']
    if not nstep1:
        nstep1 = default_config['resampling']['nstep1']
    if not nstep3:
        nstep3 = default_config['resampling']['nstep3']
    if not nsample:
        nsample = default_config['resampling']['nsample']
    if not nsections:
        nsections = default_config['resampling']['nsections']
    if not nproc:
        nsections = default_config['resampling']['nproc']

    data = {}
    indices = {}
    means = {}
    variables = dataset['var'].unique()

    for _, row in steering_table.iterrows():
        period = tuple(map(int, row['period']))
        scenario = row['scenario']
        subscenario = row['subscenario']
        epoch = row['epoch']
        mainkey = (str(epoch), scenario, subscenario)
        logger.info("Preparing data for %s_%s - %s %s", scenario, subscenario, epoch, period)
        data[mainkey], indices[mainkey], means[mainkey], _ = prepare_data(
            dataset, variables, period, reference_period, nsections)

    all_indices = {}
    for _, row in steering_table.iterrows():
        period = tuple(map(int, row['period']))
        scenario = row['scenario']
        subscenario = row['subscenario']
        epoch = row['epoch']
        mainkey = (str(epoch), scenario, subscenario)
        precip_change = row['precip_change']

        rrange = ranges[scenario][subscenario][epoch]
        logger.info("Processing %s_%s - %s %s, %.2f pr", scenario, subscenario, epoch,
                    period, precip_change)
        final_indices = find_resamples(indices[mainkey], means[mainkey],
                                       precip_change, rrange, penalties,
                                       nstep1, nstep3, nsample, nproc)

        attrs = {
            'scenario': scenario, 'subscenario': subscenario, 'epoch': epoch,
            'period': period, 'reference-period': reference_period,
            'ranges': rrange, 'winter-precip-change': precip_change}
        all_indices[mainkey] = {'data': final_indices, 'meta': attrs}

    diffs = resample(all_indices, data, variables, seasons=['djf', 'mam', 'jja', 'son'],
                     relative=relative, means=means)

    return all_indices, diffs
