"""
Example usage
python3 plotcmip5.py --var tas --percentiles-rolling-mean 10 --yearly-average --plot --area global --season year --plot-ylabel 'Temperature change [K]' --plot-title 'Global year temperature' --plot-scenario-points --nproc=4 --save-percentiles --save-zeropoints
"""

import os
from pathlib import Path
from collections import namedtuple
import functools
from datetime import datetime, timedelta
import json
import multiprocessing as mp
import argparse
import warnings
import logging
import numpy as np
import pandas as pd
import iris
from iris.cube import CubeList
from iris.experimental.equalise_cubes import equalise_attributes
import cftime
import matplotlib.pyplot as plt
from ksc.utils.date import make_year_constraint_all_calendars


RCP_HISTORICAL = -1
MINDATA = {'hist': 20, 'rcp': 4}

Item = namedtuple('Item', ['path', 'model', 'rcp', 'realization', 'cube', 'corrected'])


class EqualConstraint:
    def __init__(self, value):
        self.value = value
    def __call__(self, value):
        return self.value == value


logger = logging.getLogger(__name__)


def setup_logging(verbosity=0):
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[max(0, min(verbosity, len(levels)))]
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt="%Y-%m-%dT%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def read(path):
    strings = path.stem.split("-")
    rcp, realization = strings[-2:]
    if rcp == 'historical':
        rcp = RCP_HISTORICAL
    else:
        rcp = int(rcp[3:])
    model = "-".join(strings[:-2])
    logger.info('Reading %s (RCP %d - %s)', model, rcp, realization)
    cube = iris.load_cube(str(path))

    return {'model': model, 'rcp': rcp, 'realization': realization, 'cube': cube}


def read_data(datadir, var, area, models=None, nproc=1):
    dirname = datadir / f"{var}-{area}-area-averaged"
    if models is None:
        paths = list(dirname.glob("[A-Za-z]*.nc"))
    else:
        paths = [list(dirname.glob(f"{model}-*.nc")) for model in models]
        paths = [path for pathlist in paths for path in pathlist]
    if nproc == 1:
        data = list(map(read, paths))
    else:
        with mp.Pool(nproc) as pool:
            data = pool.map(read, paths)
    data = [Item(path, item['model'], item['rcp'], item['realization'], item['cube'], None)
            for path, item in zip(paths, data)]
    return data


class ModelZeropointCalculation:

    def __init__(self, data, yearly=False, season=False):
        self.data = data
        if yearly:
            self.mindata = MINDATA
        elif season in ['djf', 'mam', 'jja', 'son']:
            # Three months a year
            self.mindata = {key: 3*value for key, value in MINDATA.items()}
        else:
            # Twelve months a year
            self.mindata = {key: 12*value for key, value in MINDATA.items()}
        self.constraint = {'hist': make_year_constraint_all_calendars(1981, 2005),
                           'rcp': make_year_constraint_all_calendars(2006, 2010)}

    def __call__(self, model):
        items = [item for item in self.data if item.model == model]
        data = {'hist': [item for item in items if item.rcp == RCP_HISTORICAL],
                'rcp': [item for item in items if item.rcp != RCP_HISTORICAL]}

        avs = {}
        ndata = {}
        mean = {}
        for key, value in data.items():
            avs[key] = self.calc_mean(value, self.constraint[key], self.mindata[key], model)
        if not avs['hist'] or not avs['rcp']:  # Too few data to calculate a decent bias
            logging.warning("%s does not have enough data to compute a zeropoint", model)
            return None

        for key, values in avs.items():
            n = len(values)
            # Weighted time means for each section
            ndata[key] = sum(value[1] for value in values) / n
            mean[key] = sum(value[0].data for value in values) / n
        zeropoint = ((mean['hist'] * ndata['hist'] + mean['rcp'] * ndata['rcp']) /
                     (ndata['hist'] + ndata['rcp']))

        return zeropoint

    def calc_mean(self, data, constraint, mindata, model):
        averages = []
        for item in data:
            calendar = item.cube.coord('time').units.calendar
            cube = constraint[calendar].extract(item.cube)
            if cube is None:
                logging.warning("A cube of %s does not support time range: %s",
                                model, item.cube.coord('time'))
                continue
            ndata = len(cube.coord('time').points)
            # Less than 20 years of data? Ignore
            if ndata < mindata:
                logging.warning("A cube of %s has only %d data points for its historical "
                                "time range", model, ndata)
                continue
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                averages.append((cube.collapsed('time', iris.analysis.MEAN), ndata))
        return averages


def average_year_cube(item):
    cube = item.cube.aggregated_by('year', iris.analysis.MEAN)
    return Item(item.path, item.model, item.rcp, item.realization, cube, None)


def average_year(data, nproc=1):
    logger.info("Calculating yearly averages")
    if nproc == 1:
        return list(map(average_year_cube, data))

    with mp.Pool(nproc, maxtasksperchild=1) as pool:
        data = pool.map(average_year_cube, data)
    return data


def calczeropoints(data, yearly=False, season=False, nproc=1):
    calculation = ModelZeropointCalculation(data, yearly=yearly, season=season)
    models = {item.model for item in data}
    if nproc == 1:
        zeropoints = map(calculation, models)
    else:
        with mp.Pool(nproc, maxtasksperchild=1) as pool:
            zeropoints = pool.map(calculation, models)
    zeropoints = dict(zip(models, zeropoints))
    return zeropoints


def extract_season_cube(item, season):
    constraint = EqualConstraint(season)
    cube = iris.Constraint(season=constraint).extract(item.cube)
    return Item(item.path, item.model, item.rcp, item.realization, cube, None)


def extract_season(data, season, nproc=1):
    if season == 'year':
        return data
    logger.info("Extracting season %s", season)
    func = functools.partial(extract_season_cube, season=season)
    if nproc == 1:
        return list(map(func, data))

    with mp.Pool(nproc, maxtasksperchild=1) as pool:
        data = pool.map(func, data)
    return data


def correct_cube(item, relative=False):
    item, zeropoint = item
    cube = item.cube
    cube.data -= zeropoint
    if relative:
        cube.data /= zeropoint
        cube.data *= 100
        #cube.attributes = item.cube.attributes
        cube.units = '%'
    return Item(item.path, item.model, item.rcp, item.realization, item.cube, cube)


def correct(data, zeropoints, relative=False):
    logger.info("Correcting data for zeropoint")
    data = [(item, zeropoints[item.model]) for item in data
            if data is not None and zeropoints[item.model] is not None]
    correct_partial = functools.partial(correct_cube, relative=relative)
    data = list(map(correct_partial, data))
    return data


def save_corrected(item, area, season, var):
    model = item.model
    rcp = item.rcp
    realization = item.realization
    cube = item.corrected
    if cube is None:
        return
    dirname = f"{var}-{area}-{season}-zpcorrected"
    os.makedirs(dirname, exist_ok=True)
    if rcp == RCP_HISTORICAL:
        fname = f"{model}-historical-{realization}.nc"
    else:
        fname = f"{model}-rcp{rcp}-{realization}.nc"
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        iris.save(cube, os.path.join(dirname, fname))


class YearPercentileCalculation:

    def __init__(self, data):
        self.data = data

    def __call__(self, year):
        #constraint = EqualConstraint(year)
        constraint = iris.Constraint(year=lambda point: point == year)
        cubes = filter(None, [constraint.extract(item.corrected) for item in self.data])
        data = np.array([cube.data for cube in cubes])
        mean = data.mean()
        percs = np.percentile(data, [5, 10, 25, 50, 75, 90, 95], overwrite_input=True)
        return dict(zip(['mean', '5', '10', '25', '50', '75', '90', '95'],
                        [mean] + percs.tolist()))


def calc_percentiles(data, nproc=1):
    logger.info("Calculating percentiles")
    years = list(range(1950, 2100))
    calculation = YearPercentileCalculation(data)
    if nproc == 1:
        percs = list(map(calculation, years))
    else:
        with mp.Pool(nproc, maxtasksperchild=1) as pool:
            percs = pool.map(calculation, years)
    return pd.DataFrame(
        percs, index=pd.DatetimeIndex([datetime(year, 1, 1) for year in years]))


def num2date(coord, index=None):
    if index is None:
        return cftime.num2date(coord.points, str(coord.units), coord.units.calendar)
    return cftime.num2date(coord.points[index], str(coord.units),
                           coord.units.calendar)


def plot(filename, data, percs=None, ylabel=None, legend=False, title=None,
         xrange=None, yrange=None, scenarios=None, ecearth=None):
    logger.info("Plotting")

    plt.figure(figsize=(12, 8))
    if data is not None:
        for item in data:
            # Iris's plotting functionality has some problems when plotting
            # graphs with multiple calendar types on top of each other
            # We (attempt to) convert the dates to Python datetimes,
            # which should be fine, given that we're using yearly averages
            # We're only doing this for plotting; the actual data is not changed
            dates = num2date(item.corrected.coord('time'))
            dates = np.array([datetime(date.year, date.month, date.day) for date in dates],
                             dtype='datetime64')
            plt.plot(dates, item.corrected.data, alpha=0.3, zorder=1,
                     label=f"{item.model}-{item.realization}-{item.rcp}")

    if percs is not None:
        dates = percs.index
        plt.fill_between(dates,
                         percs['5'].to_numpy(dtype=np.float),
                         percs['95'].to_numpy(dtype=np.float),
                         color='#bbbbbb', alpha=0.8, zorder=2, label='5% - 95%')
        plt.fill_between(dates,
                         percs['10'].to_numpy(dtype=np.float),
                         percs['90'].to_numpy(dtype=np.float),
                         color='#888888', alpha=0.4, zorder=3, label='10% - 90%')
        plt.fill_between(dates,
                         percs['25'].to_numpy(dtype=np.float),
                         percs['75'].to_numpy(dtype=np.float),
                         color='#555555', alpha=0.2, zorder=4, label='25% - 75%')
        plt.plot(dates, percs['mean'], color='#000000', lw=2, zorder=5, label='mean')
        ax = plt.gca()
        ax.yaxis.set_ticks_position('both')

    if ecearth is not None:
        if isinstance(ecearth, CubeList):
            for cube in ecearth:
                dates = num2date(cube.coord('time'))
                dates = np.array([datetime(date.year, date.month, date.day) for date in dates],
                                 dtype='datetime64')
                plt.plot(dates, cube.data, zorder=6)
        else:
            dates = num2date(ecearth.coord('time'))
            dates = np.array([datetime(date.year, date.month, date.day) for date in dates],
                             dtype='datetime64')
            plt.plot(dates, ecearth.data, zorder=6, label='EC-EARTH')


    if scenarios['plot'] and percs is not None:
        year = scenarios['base']
        plt.scatter(datetime(year, 1, 1), 0, s=100, marker='s', color='black', zorder=6)
        for year in scenarios['years']:
            date = datetime(year, 1, 1)
            low, high = scenarios['percentiles']
            low = percs.loc[date, low]
            high = percs.loc[date, high]
            plt.scatter(datetime(year, 1, 1), low, s=100, marker='o', color='green', zorder=6)
            plt.scatter(datetime(year, 1, 1), high, s=100, marker='o', color='green', zorder=6)

    if legend:
        plt.legend()
    if xrange is None:
        xrange = [datetime(1950, 1, 1), datetime(2100, 1, 1)]
    plt.axis([xrange[0], xrange[1], yrange[0], yrange[1]])
    plt.xticks([datetime(year, 1, 1) for year in np.arange(1950, 2100, 20)])
    plt.xlabel('Year', fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18)
    if title:
        plt.title(title, fontsize=22)
    plt.grid()
    plt.savefig(filename)


def load_ecearth(datadir, var, area, season='year', relative=False, nproc=1):
    logger.info('Reading EC-EARTH data')
    dirname = datadir / f"{var}-ecearth-{area}-area-averaged/"
    paths = [str(path) for path in dirname.glob(f"{var}*.nc")]
    cubes = iris.load(paths)
    data = [Item(path, 'EC-EARTH-16', 85, i, cube, None)
            for i, (path, cube) in enumerate(zip(paths, cubes))]
    data = extract_season(data, season, nproc=nproc)
    cubes = CubeList([item.cube.aggregated_by('year', iris.analysis.MEAN) for item in data])

    equalise_attributes(cubes)
    cube2d = cubes.merge_cube()
    mean = cube2d.collapsed('realization', iris.analysis.MEAN)
    mean = mean.aggregated_by('year', iris.analysis.MEAN)

    begin, end = cftime.DatetimeGregorian(1981, 1, 1), cftime.DatetimeGregorian(2010, 12, 31)
    constraint = iris.Constraint(time=lambda cell: begin <= cell.point <= end)
    zeropoint = constraint.extract(mean)
    zeropoint = zeropoint.collapsed('time', iris.analysis.MEAN)
    zeropoint = zeropoint.data

    for cube in cubes:
        cube.data -= zeropoint
        if relative:
            cube.data /= zeropoint
            cube.data *= 100
            #cube.attributes = cube.attributes
            cube.units = '%'
    mean.data -= zeropoint
    if relative:
        mean.data /= zeropoint
        mean.data *= 100
        #cube.attributes = mean.attributes
        mean.units = '%'

    return cubes, mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='*')
    parser.add_argument('-v', '--verbosity', action='count',
                        default=0, help="Verbosity level")
    parser.add_argument('--nproc', type=int, default=1)
    parser.add_argument('--basedir', default='.',
                        help="Main directory containing subdirectories with extracted data")
    parser.add_argument('--var', default='tas')
    parser.add_argument('--season', default='year', choices=['year', 'djf', 'mam', 'jja', 'son'])
    parser.add_argument('--area', default='global',
                        choices=['global', 'nlbox', 'weurbox', 'nlpoint'])
    parser.add_argument('--relative', action='store_true',
                        help="Calculate the relative change with respect to "
                        "the reference period.")
    parser.add_argument('--save-corrected', action='store_true')
    parser.add_argument('--save-zeropoints', action='store_true')
    parser.add_argument('--save-percentiles', action='store_true')
    parser.add_argument('--yearly-average', action='store_true')
    parser.add_argument('--percentiles-rolling-mean', type=int, default=0)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--plot-outputfile', default="")
    parser.add_argument('--plot-no-data', action='store_true')
    parser.add_argument('--plot-no-percentiles', action='store_true')
    parser.add_argument('--plot-ylabel', default="")#'Temperature [\u212a]')
    parser.add_argument('--plot-title', default="DEFAULT")
    parser.add_argument('--plot-legend', action='store_true')
    parser.add_argument('--plot-xrange', type=str, nargs=2, default=['1950-1-1', '2100-1-1'])
    parser.add_argument('--plot-yrange', type=float, default=[None, None], nargs=2)
    parser.add_argument('--plot-scenario-points', action='store_true')
    parser.add_argument('--scenario-base-year', type=int, default=1996)
    parser.add_argument('--scenario-years', type=int, nargs='+', default=[2050, 2085])
    parser.add_argument('--scenario-percentiles', nargs=2, default=['10', '90'])
    parser.add_argument('--ecearth', action='store_true', help="Read EC-EARTH "
                        "data and print Ec-EARTH resample period and scaling factors")
    parser.add_argument('--plot-ecearth', action='store_true')
    parser.add_argument('--ecearth-average', action='store_true')

    args = parser.parse_args()
    args.models = args.models or None
    args.basedir = Path(args.basedir)
    args.plot_xrange = [datetime.strptime(args.plot_xrange[0], '%Y-%m-%d'),
                        datetime.strptime(args.plot_xrange[1], '%Y-%m-%d')]
    return args


def run(datadir, models, var, season, area, options, nproc=1):
    relative = options.get('relative', False)
    data = read_data(datadir, var, area, models=models, nproc=nproc)
    if season != 'year':
        data = extract_season(data, season, nproc=nproc)
    if options['yearly-average']:
        data = average_year(data, nproc=nproc)
    zeropoints = calczeropoints(data, yearly=options['yearly-average'], season=season, nproc=nproc)

    if options['save-zeropoints']:
        with open(f"zeropoints-{var}-{area}-{season}.json", 'w') as fp:
            json.dump(zeropoints, fp)

    data = correct(data, zeropoints, relative)

    if options['save-corrected']:
        save_partial = functools.partial(save_corrected, area=area, season=season, var=var)
        list(map(save_partial, data))

    percs = calc_percentiles(data, nproc=nproc)

    n = options['percentiles-rolling-mean']
    if n and n > 1:
        percs = percs.rolling(n, center=True).mean().dropna()
    if options['save-percentiles']:
        percs.to_csv(f'percentiles-{var}-{area}-{season}.csv', index_label='year')

    steering = []
    for year in options['scenarios']['years']:
        date = datetime(year, 1, 1)
        steering.append({'year': year, 'scenario': 'G',
                         'delta-T': percs.loc[date, '10']})
        steering.append({'year': year, 'scenario': 'W',
                         'delta-T': percs.loc[date, '90']})
    steering = pd.DataFrame(steering)

    eccalc = options['ecearth']['calculate']
    ecearth, ecmean = None, None
    if options['ecearth']['plot'] or eccalc:
        ecearth, ecmean = load_ecearth(datadir, var, area, season=season, relative=relative,
                                       nproc=nproc)
        if eccalc:
            # Limit EC-EARTH data to 2085, so we don't try and calculate beyond 2100
            cube = iris.Constraint(year=lambda year: year <= 2085).extract(ecmean)
            # Find indices and dates for the EC-EARTH data nearest
            # to the steering delta-T's
            indices = [np.argmin(np.abs((cube.data - t).data))
                       for t in steering['delta-T']]
            dates = num2date(cube[indices].coord('time'))
            steering['resampling period'] = [
                ((date - timedelta(15*365.24)).year, (date + timedelta(15*365.24)).year)
                for date in dates]
            steering['actual delta-t'] = cube[indices].data.data
            steering['factor'] = steering['delta-T'] / steering['actual delta-t']

        if options['ecearth'].get('average'):
            ecearth = ecmean

    if options['plot']:
        filename = options['plot']['outputfile']
        if not filename:
            filename = f'{var}-{area}-{season}-zpcorrected.png'
        title = options['plot']['title']
        if title == 'DEFAULT':
            title = f"{var} - {area} - {season}"
        ylabel = options['plot']['ylabel']
        xrange = options['plot']['xrange']
        #ylabel = 'Temperature [\u212a]'
        yrange = options['plot']['yrange']
        if options['plot']['no-data']:
            data = None
        if options['plot']['no-percentiles']:
            percs = None
        legend = options['plot']['legend']

        plot(filename, data=data, percs=percs, scenarios=options['scenarios'],
             title=title, ylabel=ylabel, xrange=xrange, yrange=yrange, legend=legend,
             ecearth=ecearth)

    print(steering)


def main():
    args = parse_args()
    setup_logging(args.verbosity)
    options = {
        'relative': args.relative,
        'save-corrected': args.save_corrected,
        'save-zeropoints': args.save_zeropoints,
        'save-percentiles': args.save_percentiles,
        'yearly-average': args.yearly_average,
        'percentiles-rolling-mean': args.percentiles_rolling_mean,
        'ecearth': {'calculate': args.ecearth,
                    'plot': args.plot_ecearth,
                    'average': args.ecearth_average}
    }
    options['plot'] = {}
    if args.plot:
        options['plot'] = {
            'outputfile': args.plot_outputfile,
            'no-data': args.plot_no_data,
            'no-percentiles': args.plot_no_percentiles,
            'ylabel': args.plot_ylabel,
            'title': args.plot_title,
            'legend': args.plot_legend,
            'xrange': args.plot_xrange,
            'yrange': args.plot_yrange,
        }
    scenarios = {
        'plot': args.plot_scenario_points,
        'base': args.scenario_base_year,
        'years': args.scenario_years,
        'percentiles': args.scenario_percentiles,
    }
    options['scenarios'] = scenarios

    run(args.basedir, args.models, args.var, args.season, args.area, options=options,
        nproc=args.nproc)


if __name__ == "__main__":
    main()
