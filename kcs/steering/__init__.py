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

import logging
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import iris
from iris.experimental.equalise_cubes import equalise_attributes
import cftime
import kcs.utils.logging
import kcs.utils.attributes
import kcs.utils.constraints


REFERENCE_PERIOD = (1981, 2010)

logger = logging.getLogger(__name__)


def read_data(paths, info_from=('attributes', 'filename'),
              attributes=None, filename_pattern=None):
    """DUMMY DOC-STRING"""
    cubes = [iris.load_cube(str(path)) for path in paths]

    # Get the attributes, and create a dataframe with cubes & attributes
    dataset = kcs.utils.attributes.get(
        cubes, paths, info_from=info_from,
        attributes=attributes, filename_pattern=filename_pattern)

    return dataset


def num2date(coord, index=None):
    """DUMMY DOCSTRING"""
    if index is None:
        return cftime.num2date(coord.points, str(coord.units), coord.units.calendar)
    return cftime.num2date(coord.points[index], str(coord.units),
                           coord.units.calendar)


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


def calc_reference_values(cubes, reference_period, normby='run'):
    """Calculate the reference values for each cube

    If normby is not 'run', the individual reference values are averaged, so that each run
    will later be normalized by the same value.

    """

    constraint = iris.Constraint(year=kcs.utils.constraints.RangeConstraint(*reference_period))
    values = []
    for cube in cubes:
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


def normalize_average_dataset(cubes, season=None, average_years=True, relative=False,
                              reference_period=REFERENCE_PERIOD):
    """Normalize and average a given iterable of cubes

    The dataset is normalized by, and averaged across, its individual
    ensemble runs. Thus, this works best (only), if the dataset
    belongs to the same model and experiment, and has no other
    ensemble variations other than its realization.

    The dataset should already be concatenated across the historical
    and future experiment, if necessary.

    Each Iris cube inside the dataset should have a scalar realization
    coordinate, with its value given the realization number. If not,
    these are added on the fly, equal to the iteration index of the
    cubes.

    """

    if season:
        cubes = extract_season(dataset, season)
    if average_years:
        cubes = average_year(cubes, season=season)
    refvalues = calc_reference_values(
        cubes, reference_period=reference_period, normby='run')
    cubes = normalize(cubes, refvalues, relative=relative)

    for i, cube in enumerate(cubes):
        if not cube.coords('realization'):
            coord = iris.coords.AuxCoord(
                i, standard_name='realization', long_name='realization',
                var_name='realization')
            cube.add_aux_coord(coord)

    cubes = iris.cube.CubeList(cubes)
    equalise_attributes(cubes)
    cube2d = cubes.merge_cube()
    mean = cube2d.collapsed('realization', iris.analysis.MEAN)
    print(mean.coord('year'))
    return mean


def run(dataset, percentiles, scenarios, season=None, average_years=True,
        relative=False, reference_period=REFERENCE_PERIOD,
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

    mean = normalize_average_dataset(dataset['cube'], season, average_years,
                                     relative=relative, reference_period=reference_period)

    steering = calc(mean, percentiles, scenarios, timespan=timespan,
                    rolling_mean=rolling_mean, rounding=rounding)
    return steering
