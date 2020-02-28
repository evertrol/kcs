"""Module to extract data from CMIP NetCDF data files.

This module wraps the area extraction funcionality from
`kcs.utils.coord`. It can run multiple processes in
parallel. Extracted datasets can be saved (by default) to disk, in
subdirectoriees named after the variable and area (given by a template
that follows Python formatted strings with variable names; the default
is given in the `TEMPLATE` constant).

"""

import os
import pathlib
from pprint import pformat
from tempfile import NamedTemporaryFile
from collections import namedtuple, defaultdict
import itertools
import functools
import re
import multiprocessing
import warnings
import logging
import iris
from kcs.types import Data
import kcs.utils.io
import kcs.utils.coord
from kcs.utils.date import months_coord_to_days_coord
import kcs.config


# Allowed template substitution variable names: var, area, filename
# (filename refers to the filename part of the input file, not the full path)
TEMPLATE = "ecearth/{var}-{area}-averaged/{filename}"

logger = logging.getLogger(__name__)


def get_realization_from_path(path):
    regex = re.search(r'\_(?P<realization>\d+)\.nc$', path.name)
    if not regex:
        regex = re.search(r'r(?P<realization>\d+)i\d+p\d+', path.name)
    if regex:
        realization = int(regex.group('realization'))
    else:
        raise ValueError(f'realization not found: {path.name}')
    return realization


def get_varname(path):
    match = re.search('^(?P<var>[a-z]+)_.*$', path.name)
    if match:
        return match.group('var')


def process_single(path, areas, targetgrid=None, save_result=True,
                   average_area=True, gridscheme='area', template=TEMPLATE,
                   mp=False, tempdir=None, ignore_common_warnings=False):
    if isinstance(path, list):
        realizations = set(get_realization_from_path(p) for p in path)
        if len(realizations) > 1:
            raise ValueError("multiple realizations inside subdir "
                             "{path[0].parent}")
        realization = realizations.pop()
    else:
        realization = get_realization_from_path(path)
    logger.debug("Realization %d for %s", realization, path)

    varname = get_varname(path)
    with warnings.catch_warnings():
        if ignore_common_warnings:
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message="Missing CF-netCDF measure variable 'areacella', "
                                    "referenced by netCDF variable")
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message="Using DEFAULT_SPHERICAL_EARTH_RADIUS")
        logger.info('Reading %s', path)
        cube = kcs.utils.io.load_cube(path, variable_name=varname)

    logger.info('Fixing coordinates')
    # EC-EARTH data has only 'months since', not 'days since'; iris doesn't accept that.
    try:
        day_coord = months_coord_to_days_coord(cube.coord('time'))
        cube.remove_coord('time')
        cube.add_dim_coord(day_coord, 0)
    except ValueError as exc:
        if str(exc) == "units step is not months":
            pass
        else:
            raise
    cube = kcs.utils.coord.fixcoords(cube, realization)

    logger.info('Extracting areas')
    with warnings.catch_warnings():
        if ignore_common_warnings:
            warnings.filterwarnings("ignore", category=UserWarning,
                                    message="Using DEFAULT_SPHERICAL_EARTH_RADIUS")
        cubes = kcs.utils.coord.extract_areas(cube, areas=areas, targetgrid=targetgrid,
                                              average_area=average_area, gridscheme=gridscheme)
    assert len(cubes) == len(areas)

    data = []
    if save_result:
        with warnings.catch_warnings():
            #warnings.filterwarnings("ignore", category=UserWarning)
            for area, cube in cubes.items():
                var = cube.var_name
                # os.path.commonprefix is good enough for our purpose:
                # the filename convention has the last part changing
                # most often
                filename = (os.path.commonprefix([p.name for p in path])
                            if isinstance(path, (set, list)) else path.name)
                # Strip off .nc extension; any other extension will be left as is
                filename = filename[:-3] if filename.endswith(".nc") else filename
                outpath = pathlib.Path(template)
                dirname = str(outpath.parent).format(var=var, area=area, filename=filename)
                os.makedirs(dirname, exist_ok=True)
                outpath = str(outpath).format(var=var, area=area, filename=filename)
                logger.info("Saving area %s, realization %d in '%s'",
                            area, realization, outpath)
                iris.save(cube, outpath)
                data.append(Data(outpath, realization, area, cube))
    elif mp:  # We're using multiple processes in separate threads
        # We're saving the output to a temporary file
        # The data is generally too large to pass directly to the main thread,
        # so we need a workaround, and use disk space as intermediate storage
        # Not overly efficient, but with large data files, more efficient solutions
        # likely require a lot of extra code (such as passing the data structure
        # in parts).
        for area, cube in cubes.items():
            var = cube.var_name
            fh = NamedTemporaryFile(suffix=".nc", prefix="extract-ecearth",
                                    dir=tempdir, delete=False)
            outpath = fh.name
            logger.debug("Saving cube to temporary file %s", outpath)
            iris.save(cube, outpath)
            data.append(Data(outpath, realization, area, None))
    else:
        data = [Data(None, realization, area, cube) for area, cube in cubes.items()]

    if mp:  # Ensure no cubes are passed back to the main thread
        data = [Data(item.path, item.realization, item.area, None) for item in data]

    return data


def process(paths, areas, regrid=False, save_result=True, average_area=True,
            gridscheme='area', nproc=1, template=TEMPLATE, tempdir=None,
            subdir_per_realization=False, ignore_common_warnings=False):
    if subdir_per_realization:
        pathlist = defaultdict(list)
        for path in paths:
            pathlist[path.parent].append(path)
        paths = list(pathlist.values())

    targetgrid = None
    if regrid:
        targetgrid = kcs.utils.coord.create_grid()

    func = functools.partial(process_single, areas=areas, targetgrid=targetgrid,
                             save_result=save_result, average_area=average_area,
                             gridscheme=gridscheme, template=template,
                             mp=(nproc > 1), tempdir=tempdir,
                             ignore_common_warnings=ignore_common_warnings)
    if nproc == 1:
        data = list(itertools.chain.from_iterable(map(func, paths)))
    else:  # Need maxtaskperchild, to avoid multiprocessing getting stuck
        with multiprocessing.Pool(nproc, maxtasksperchild=1) as pool:
            data = list(itertools.chain.from_iterable(pool.map(func, paths)))
    return data


def run(paths, areas, regrid=False, save_result=True, average_area=True, nproc=1,
        template=TEMPLATE, tempdir=None, subdir_per_realization=False,
        ignore_common_warnings=False):
    paths = [pathlib.Path(str(path)) if not isinstance(path, pathlib.Path) else path
             for path in paths]
    # No need to regrid the EC-EARTH data: all on the same grid
    data = process(paths, areas, regrid=regrid, save_result=save_result, average_area=average_area,
                   nproc=nproc, template=template, tempdir=tempdir,
                   subdir_per_realization=subdir_per_realization,
                   ignore_common_warnings=ignore_common_warnings)

    # Handle data post-processing, so we can return the data to the caller
    # Data files were not passed when using multiprocessing: files may be
    # too large for, or incompatible with, the pickling protocol used
    # by multiprocessing. We'll have to reload the data from disk instead.
    if nproc > 1:
        logger.debug("Reloading files in main thread")
        data = [Data(item.path, item.realization, item.area, iris.load_cube(str(item.path)))
                for item in data]
        if not save_result:
            # Data files were saved to temporary files. Force Iris to load
            # all data (foregoing lazy loading), and remove the temporary files
            for item in data:
                if item.cube.has_lazy_data:
                    item.cube.data    # pylint: disable=pointless-statement
                os.remove(item.path)
            data = [Data(None, item.realization, item.area, item.cube) for item in data]

    logger.info("Finished processing %s", pformat(data))

    return data
