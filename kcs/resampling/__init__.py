import math
import re
import pathlib
import glob
import itertools
import functools
from datetime import datetime
import multiprocessing
import json
import argparse
import logging
from pprint import pformat
import numpy as np
import pandas as pd
import h5py
import iris
import ecearth


NPROC = 1
ALLSEASONS = ['djf', 'mam', 'jja', 'son']
N1 = 1000
N2 = 50
N3 = 9
NSAMPLE = 10_000
STATS = ['mean', '5', '10', '25', '50', '75', '90', '95']


logger = logging.getLogger(__name__)


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
    """Class to calculate the difference between a calculated precipiation change, and a target value

    We use a class, so it can be used with multiprocessing: fixed
    arguments are passed to the constructor, while the variable
    argument (the subselection of the data) is passed to the call
    method.

    An alternative is the use of functools.partial

    """

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.cols = np.arange(data.shape[1])

    def __call__(self, index):
        return np.abs(self.data[index, self.cols].mean() - self.target)


def calc_means(data, seasons=None):
    """Calculate the averages of all 5-year segments, and store them in a dataframe"""
    if seasons is None:
        seasons = ALLSEASONS
    periods = ['control', 'future']
    means = {}
    std = {}
    for var in data:
        for season in seasons:
            for period in ['control', 'future']:
                item = data[var][season][period]
                for i, cubelist in enumerate(item):
                    #if var == 'pr' and season == 'djf' and period == 'control':
                    #    print([(cube.data.mean(), cube.data.std()) for cube in cubelist])
                    means[(var, season, period, i)] = [cube.data.mean() for cube in cubelist]
                    #print(var, season, period)
                    #print('5 year means:', means[(var, season, period, i)])
                    std[(var, season, period, i)] = [cube.data.std() for cube in cubelist]
                    #print('5 year std:', std[(var, season, period, i)])

    df = pd.DataFrame.from_dict(means, orient='index')
    df.index = pd.MultiIndex.from_tuples(means.keys(), names=['var', 'season', 'period', 'run'])
    df = df.sort_index()
    return df


def calculate_s1(means, indices, target, var='pr', season='djf', nproc=NPROC):
    """Calculate the subset S1: winter precipitation change equals `target`"""

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
    logger.info("time(calculation) = %s", datetime.now() - tstart)
    values = np.array(values)

    tstart = datetime.now()
    order = np.argsort(values)
    logger.info("time(np.argsort) = %s", datetime.now() - tstart)

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
    logger.info("time(calculation) = %s", datetime.now() - tstart)
    values = np.array(values)

    tstart = datetime.now()
    order = np.argsort(values)
    logger.info("time(np.argsort) = %s", datetime.now() - tstart)

    selind['control'] = indices[order]

    return selind


def calculate_s2(means, indices, ranges):
    """Calculate the subset S2: select percentile ranges for average precipitation and temperatures"""

    if isinstance(ranges, dict):
        newranges = []
        for key1, value1 in ranges.items():
            for key2, value2 in value1.items():
                newranges.append({key1: {key2: value2}})
        ranges = newranges

    s2_indices = {}
    columns = np.arange(indices['control'].shape[1])
    for period in ['control', 'future']:
        selection = np.ones(indices[period].shape[0], dtype=np.bool)
        s2_indices[period] = indices[period][selection]
        for item in ranges:
            for var in item:
                for season in item[var]:
                    logger.debug("Subsetting with percentiles for %s, %s, %s", period, var, season)
                    values = means.loc[(var, season, period), :].values[s2_indices[period], columns]
                    # Calculate mean along the columns, i.e., one 30-year period
                    mean = values.mean(axis=1)
                    print('VALUES, INDICES, MEAN SHAPE =', means.loc[(var, season, period), :].values.shape,
                          values.shape, s2_indices[period].shape, mean.shape)
                    logger.debug("Min, max, mean, median values: %f  %f  %f  %f",
                                 mean.min(), mean.max(), mean.mean(), np.median(mean))
                    p_range = item[var][season][period]
                    logger.debug("Percentile range = %s", p_range)
                    low, high = np.percentile(mean, p_range)
                    logger.debug("Percentile values = %f  --  %f", low, high)
                    selection = (low <= mean) & (mean <= high)
                    s2_indices[period] = s2_indices[period][selection]
                    logger.debug("Subsetting down to %d samples", selection.sum())
        for item in ranges:
            for var in item:
                for season in item[var]:
                    values = means.loc[(var, season, period), :].values[s2_indices[period], columns]
                    mean = values.mean(axis=1)
                    logger.debug("%s %s %s subset min, max, median: %g, %g, %g", period, var, season,
                                 mean.min(), mean.max(), np.median(mean))
                    values = means.loc[(var, season, period), :].values[indices[period], columns]
                    mean = values.mean(axis=1)
                    logger.debug("%s %s %s total min, max, median: %g, %g, %g", period, var, season,
                                 mean.min(), mean.max(), np.median(mean))

    values = means.loc[('pr', 'djf', 'control'), :].values[s2_indices['control'], columns]
    meanc = values.mean(axis=1)
    pc = np.percentile(values, [5, 50, 95], axis=1)
    values = means.loc[('pr', 'djf', 'future'), :].values[s2_indices['future'], columns]
    meanf = values.mean(axis=1)
    pf = np.percentile(values, [5, 50, 95], axis=1)
    m = (meanf.mean() - meanc.mean()) / meanc.mean() * 100
    p = (pf.mean(axis=1) - pc.mean(axis=1)) / pc.mean(axis=1) * 100
    print('PR/DJF =', m, p)
    #values = means.loc[('tas', 'djf', 'control'), :].values[s2_indices['control'], columns]
    #meanc = values.mean(axis=1)
    #pc = np.percentile(values, [5, 50, 95], axis=1)
    #values = means.loc[('tas', 'djf', 'future'), :].values[s2_indices['future'], columns]
    #meanf = values.mean(axis=1)
    #pf = np.percentile(values, [5, 50, 95], axis=1)
    #m = (meanf.mean() - meanc.mean()) / meanc.mean() * 100
    #p = (pf.mean(axis=1) - pc.mean(axis=1)) / pc.mean(axis=1) * 100
    #print('tas/djf =', m, p)
    #values = means.loc[('tas', 'jja', 'control'), :].values[s2_indices['control'], columns]
    #meanc = values.mean(axis=1)
    #pc = np.percentile(values, [5, 50, 95], axis=1)
    #values = means.loc[('tas', 'jja', 'future'), :].values[s2_indices['future'], columns]
    #meanf = values.mean(axis=1)
    #pf = np.percentile(values, [5, 50, 95], axis=1)
    #m = (meanf.mean() - meanc.mean()) / meanc.mean() * 100
    #p = (pf.mean(axis=1) - pc.mean(axis=1)) / pc.mean(axis=1) * 100
    #print('tas/jja =', m, p)


    return s2_indices


def calculate_s3(means, indices_dict, penalties, n3=N3, nsample=NSAMPLE,
                 minimum_penalty=None):
    """Calculate the subset S3: find a subset with the least re-use of segments, using random sampling"""

    if minimum_penalty is None:
        minimum_penalty = np.finfo(float).eps
    s3_indices = {}
    rng = np.random.default_rng()
    for period, indices in indices_dict.items():
        n = len(indices)
        m = math.factorial(n) // math.factorial(n3) // math.factorial(n - n3)
        logger.debug("Number of combinations for the %s period = %d", period, m)
        best = np.inf
        for i in range(nsample):
            choice = rng.choice(indices, size=n3, replace=False)

            penalty = 0
            for column in choice.T:
                values, counts = np.unique(column, return_counts=True)
                for c in counts:
                    penalty += penalties[c]
            if penalty <= best:
                best = penalty
                s3_indices[period] = choice
            if best < minimum_penalty:
                logger.debug("Minimum penalty reached after %d iterations", i)
                break
        logger.info("Best sample for %s (penalty: %f) out of %d samples: %s",
                    period, best, nsample, s3_indices[period])

    return s3_indices


def save_indices(filename, indices, meta=None):
    data = {key: value.tolist() for key, value in indices.items()}

    if meta:
        data = {'meta': meta, 'indices': data}
    with open(filename, 'w') as fh:
        json.dump(data, fh)


def save_indices_h5(filename, indices):
    """Save the (resampled) array of indices in a HDF5 file"""
    h5file = h5py.File('indices.h5', 'a')
    for key, value in indices.items():
        name = "/".join(key)
        try:
            group = h5file[name]
        except KeyError:
            group = h5file.create_group(name)
        for k, v in value['meta'].items():
            group.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        if 'control' in group:
            del group['control']
        group.create_dataset('control', data=value['data']['control'])
        if 'future' in group:
            del group['future']
        group.create_dataset('future', data=value['data']['future'])


def resample(indices, data, variables, relative, use_means=False, controlmeans=None):
    """Perform the actual resampling of data, given the resampled indices"""
    percs = list(map(float, STATS[1:]))

    diffs = {}
    #from pprint import pprint; pprint(indices)
    for key, value in indices.items():
        diffs[key] = {}
        tempindices = value['data']
        for var in variables:
            diffs[key][var] = {}
            for season in ['djf', 'jja']:  #['djf', 'mam', 'jja', 'son']:
                stats = {}
                for period in ['control', 'future']:
                    cubes = data[key][var][season][period]
                    columns = np.arange(cubes.shape[1])
                    print()
                    print(key, season, period, tempindices[period])
                    #print('>>!>>', tempindices[period].shape, columns.shape)
                    resampled = cubes[tempindices[period], columns]
                    #print('>!>', resampled.shape)
                    stats[period] = pd.DataFrame({'mean': 0, '5': 0, '10': 0, '25': 0, '50': 0,
                                                  '75': 0, '90': 0, '95': 0},
                                                 index=np.arange(len(resampled)))
                    aves = []
                    for i, cubeset in enumerate(resampled):
                        block = np.hstack([cube.data for cube in cubeset])
                        stats[period].loc[i, STATS[1:]] = np.percentile(block, percs)
                        stats[period].loc[i, 'mean'] = block.mean()
                        if period == 'control':
                            aves.append(block.mean())
                        print(f"{block.mean():g}")
                    with pd.option_context('display.float_format', '{:g}'.format):
                        print(stats[period].loc[:, 'mean'])
                    if aves:
                        print('cm =', np.mean(aves), controlmeans)
                    if use_means and period == 'control':
                        controlmean = []
                        for cubeset in cubes:
                            controlmean.append(np.array([cube.data.mean() for cube in cubeset]))
                        controlmean = np.array(controlmean)
                        print('!!', controlmean.shape)

                        stats[period].loc[i, STATS[1:]] = np.percentile(controlmean, percs)
                        stats[period].loc[i, 'mean'] = controlmean.mean()

                #with pd.option_context('display.float_format', '{:g}'.format):
                #    print(stats['control'])
                #    print(stats['future'])
                #    if relative[var]:
                #        print(stats['control']['mean'])
                #        print(stats['future']['mean'])
                #        print((stats['future']['mean'] - stats['control']['mean']) / stats['control']['mean'] * 100)

                diff = pd.DataFrame()
                for col in STATS:
                    diff[col] = stats['future'][col] - stats['control'][col]
                    if relative[var]:
                        diff[col] = 100 * diff[col] / stats['control'][col]
                diffs[key][var][season] = diff
                print('>>', key, var, season)
                print(diff)
                print(diff.mean(axis=0))
    return diffs


def save_resamples(filename, diffs):
    """Save the resampled data, that is, the differences, in a HDF5 file"""
    h5file = h5py.File('resamples.h5', 'a')
    for key, value in diffs.items():
        group = h5file
        name = "/".join(key)
        print('A', key, name)
        for k in key:
            if k not in group:
                group.create_group(k)
            group = group[k]
        for var, value2 in value.items():
            name2 = f"{name}/{var}"
            print('B', var, name2)
            if var not in group:
                group.create_group(var)
            group2 = group[var]
            for season, diff in value2.items():
                name3 = f"{name2}/{season}"
                print('C', season, name3)
                if season not in group2:
                    group2.create_group(season)
                group3 = group2[season]

                # Remove existing datasets, to avoid problems
                # (we probably could overwrite them; this'll work just as easily)
                for k in {'diff', 'mean', 'std', 'keys'}:
                    if k in group3:
                        del group3[k]
                group3.create_dataset('diff', data=diff.values)
                group3.create_dataset('mean', data=diff.mean(axis=0))
                group3.create_dataset('std', data=diff.std(axis=0))
                ds = group3.create_dataset('keys', (len(STATS),), dtype=h5py.string_dtype())
                ds[:] = STATS

                assert len(STATS) == len(diff.mean(axis=0))
