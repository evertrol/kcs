import sys
import os
from pathlib import Path
import argparse
from collections import namedtuple
import functools
import re
import multiprocessing as mp
import warnings
import logging
from pprint import pprint
import numpy as np
import pandas as pd
import iris
from iris.cube import CubeList
import ksc.utils.coord
from ksc.utils.io import load_cube
from ksc.utils.constraints import CoordConstraint
from ksc.utils.date import make_year_constraint_all_calendars, months_coord_to_days_coord


Data = namedtuple('Data', ['path', 'realization', 'cubes'])

RCP_HISTORICAL = -1
AREAS = {
    'rhinebasin': {'w': 6, 'e': 9, 'n': 52, 's': 47},
    'nl': dict(w=3.3, e=7.1, s=50.0, n=53.6),
    'nlbox': dict(w=4.5, e=8, s=50.5, n=53),
    'weurbox': dict(w=4, e=14, s=47, n=53),
    'global': None,
    'nlpoint': dict(latitude=51.25, longitude=6.25),
}

logger = logging.getLogger(__name__)


def process_single(path, var, areas, regrid=True, targetgrid=None, save_result=True,
                   average_area=True, gridscheme='area'):
    regex = re.search('\_(?P<realization>\d+)\.nc$', path.name)
    if regex:
        realization = int(regex.group('realization'))
    else:
        raise ValueError(f'realization not found: {path.name}')
    logger.info('Reading %s', path)
    cube = iris.load_cube(str(path))
    logger.info('Fixing coordinates')
    cube = ksc.utils.coord.fixcoords(cube, realization)
    logger.info('Extracting areas')
    cubes = ksc.utils.coord.extract_areas(cube, var=var, areas=areas, targetgrid=targetgrid,
                                          average_area=average_area, gridscheme=gridscheme)
    assert len(cubes) == len(areas)

    if save_result:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for area_name, cube in cubes.items():
                fname = f"{var}-ecearth-{area_name}-area-averaged/{path.name}"
                logger.info("Saving area %s, realization %d in '%s'", area_name, realization, fname)
                iris.save(cube, fname)

    cubes = dict(zip(areas, cubes))
    return Data(path, realization, cubes)


def process(paths, var, areas, regrid=True, save_result=True, average_area=True,
            gridscheme='area', nproc=1):
    targetgrid = None
    if regrid:
        targetgrid = ksc.utils.coord.create_grid()
    if save_result:
        for area in areas:
            dirname = f"{var}-ecearth-{area}-area-averaged"
            logger.info("Creating directory '%s' to save extracted results",
                        dirname)
            os.makedirs(dirname, exist_ok=True)

    func = functools.partial(process_single, var=var, areas=areas, targetgrid=targetgrid,
                             save_result=save_result, average_area=average_area,
                             gridscheme=gridscheme)
    if nproc == 1:
        data = list(map(func, paths))
    else:
        with mp.Pool(nproc, maxtasksperchild=1) as pool:
            data = pool.map(func, paths)
            input('>>? ')
    return data


def parse_args():
    class NotADirectoryError(ValueError):
        pass
    class ListAreas(argparse.Action):
        def __call__(self, parser, namespace, values, option_string):
            print("\n".join(AREAS))
            parser.exit()
    def dir_path(string):
        if os.path.isdir(string):
            return string
        else:
            raise NotADirectoryError(string)
    parser = argparse.ArgumentParser()
    parser.add_argument('variable')
    parser.add_argument('area', nargs='+', help="One or more area names")
    parser.add_argument('-v', '--verbosity', action='count',
                        default=0, help="Verbosity level")
    parser.add_argument('-P', '--nproc', type=int, default=1,
                        help="Number of simultaneous processes")
    parser.add_argument('--list-areas', action=ListAreas, nargs=0,
                        help="List availabe areas and quit")
    parser.add_argument('--datadir', type=dir_path, default=".",
                        help="Directory with EC-EARTH data")
    parser.add_argument('--no-regrid', action='store_true')
    parser.add_argument('--no-save-results', action='store_true')
    parser.add_argument('--no-average-area', action='store_true')
    args = parser.parse_args()
    args.regrid = not args.no_regrid
    args.save_result = not args.no_save_results
    args.average_area = not args.no_average_area
    args.datadir = Path(args.datadir)
    args.area = {name: AREAS[name] for name in args.area}
    return args


def setup_logging(verbosity=0):
    levels = [logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[max(0, min(verbosity, len(levels)))]
    logger = logging.getLogger('ec-earth')
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt="%Y-%m-%dT%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def run(var, areas, datadir, regrid=True, save_result=True, average_area=True, nproc=1):
    #var = "pr"
    #areas = ['global', 'nlpoint', 'nlbox', 'weurbox']
    #dirpath = Path("/mnt/data/data1/thredds/ecearth16")
    paths = datadir.glob(f"{var}_Amon_ECEARTH23_rcp85_186001-210012_*.nc")
    paths = list(paths)
    data = process(paths, var, areas, regrid=True, save_result=True, average_area=True, nproc=nproc)
    pprint(data)


def main():
    args = parse_args()
    setup_logging(args.verbosity)
    run(args.variable, args.area, args.datadir, regrid=args.regrid,
        save_result=args.save_result, average_area=args.average_area,
        nproc=args.nproc)


if __name__ == "__main__":
    main()
