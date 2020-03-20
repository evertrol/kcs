"""DUMMY DOCSTRING"""

import logging
import numpy as np
import pandas as pd
import iris
from ..utils.constraints import EqualConstraint
from ..config import default_config


STATS = ['mean', '5', '10', '25', '50', '75', '90', '95']

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def calc_percentiles(dataset, period, relative=False, reference_period=None):
    """DUMMY DOCSTRING"""

    logger.info("Calculating percentiles")
    if reference_period is None:
        reference_period = default_config['data']['cmip']['control_period']

    refconstraint = iris.Constraint(year=lambda year:
                                    reference_period[0] <= year <= reference_period[1])
    constraint = iris.Constraint(year=lambda year: period[0] <= year <= period[1])

    stats = []
    # This loop could easily be done in parallel However, attempts
    # with multiprocessing.Pool thus far failed: quite often, the
    # process got stuck on a lock. I have not been able to deduce the
    # cause; it could be caused by memory issues due to the large
    # memory overhead and copying of data, but memory should all be
    # well within system limits. It might even be a bug in Python's
    # multiprocessing.
    # Possibly, underlying routines are not thread-safe (e.g.,
    # Iris extract), though I'd expect Python's multiprocessing
    # to not suffer from this, since these would be individual
    # processes.
    for _, row in dataset.iterrows():
        cube = row['cube']
        ref_cube = refconstraint.extract(cube)
        future_cube = constraint.extract(cube)
        if ref_cube is None:
            logger.warning("Reference data not found for %s", row['model'])
            continue
        if future_cube is None:
            logger.warning("Future data not found for %s", row['model'])
            continue

        stat = {
            'model': row['model'],
            'experiment': row['experiment'],
            'ensemble': f"r{row['realization']}i{row['initialization']}p{row['physics']}",
            'ref-mean': ref_cube.data.mean(),
            'fut-mean': future_cube.data.mean()
        }
        basepercs = np.percentile(ref_cube.data, list(map(int, STATS[1:])))
        percs = np.percentile(future_cube.data, list(map(int, STATS[1:])))
        for baseperc, perc, col in zip(basepercs, percs, STATS[1:]):
            stat[f'ref-{col}'] = baseperc
            stat[f'fut-{col}'] = perc
        stats.append(stat)
    stats = pd.DataFrame(stats)

    # Calculate the difference between reference and main epochs
    for col in STATS:
        stats[f'{col}'] = stats[f'fut-{col}'] - stats[f'ref-{col}']
        if relative:
            stats[f'{col}'] = (stats[f'{col}'] / stats[f'ref-{col}']) * 100
    return stats


def calc_percentile_distribution(dataset):
    """DUMMY DOCSTRING"""
    percs = pd.DataFrame({'mean': 0, '5': 0, '10': 0, '25': 0, '50': 0,
                          '75': 0, '90': 0, '95': 0}, index=STATS)
    percentiles = list(map(int, STATS[1:]))
    for key in percs.index:
        p = np.percentile(dataset.loc[:, key], percentiles)  # pylint: disable=invalid-name
        percs.loc[key, STATS[1:]] = p
        percs.loc[key, 'mean'] = dataset.loc[:, key].mean()
    return percs


def extract_season(cubes, season):
    """DUMMY DOC-STRING"""
    constraint = iris.Constraint(season=EqualConstraint(season))
    logger.info("Extracting season %s", season)
    for cube in cubes:
        if not cube.coords('season'):
            iris.coord_categorisation.add_season(cube, 'time')
    cubes = list(map(constraint.extract, cubes))
    return cubes


def run(dataset, season, period, relative=False, reference_period=None):
    """DUMMY DOCSTRING"""
    if reference_period is None:
        reference_period = default_config['data']['cmip']['control_period']
    if season != 'year':
        dataset['cube'] = extract_season(dataset['cube'], season)
    percs = calc_percentiles(dataset, period=period, reference_period=reference_period,
                             relative=relative)

    perc_distr = calc_percentile_distribution(percs)
    return perc_distr, percs
