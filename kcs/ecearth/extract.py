"""Module to extract data from EC-EARTH NetCDF data files.

This module wraps the area extraction funcionality from
`kcs.utils.coord`. It can run multiple processes in
parallel. Extracted datasets can be saved (by default) to disk, in
subdirectoriees named after the variable and area (given by a template
that follows Python formatted strings with variable names; the default
is given in the `TEMPLATE` constant).

The module can also be used as a executable module, with the `-m
kcs.ecearth.extract` option to the `python` executable.

"""

import sys
import os
import glob
import pathlib
import argparse
from pprint import pformat
from tempfile import NamedTemporaryFile
from collections import namedtuple
import itertools
import functools
import re
import multiprocessing
import warnings
import logging
import iris
import ksc.utils.coord
import ksc.config


Data = namedtuple('Data', ['path', 'realization', 'area', 'cube'])

# Allowed template substitution variable names: var, area, filename
# (filename refers to the filename part of the input file, not the full path)
TEMPLATE = "{var}-ecearth-{area}-area-averaged/{filename}"

logger = logging.getLogger(__name__)


def process_single(path, areas, targetgrid=None, save_result=True,
                   average_area=True, gridscheme='area', outputdir=".", template=TEMPLATE,
                   mp=False, tempdir=None):
    regex = re.search(r'\_(?P<realization>\d+)\.nc$', path.name)
    if regex:
        realization = int(regex.group('realization'))
    else:
        raise ValueError(f'realization not found: {path.name}')
    logger.info('Reading %s', path)
    cube = iris.load_cube(str(path))
    logger.info('Fixing coordinates')
    cube = kcs.utils.coord.fixcoords(cube, realization)
    logger.info('Extracting areas')
    cubes = kcs.utils.coord.extract_areas(cube, areas=areas, targetgrid=targetgrid,
                                          average_area=average_area, gridscheme=gridscheme)
    assert len(cubes) == len(areas)

    data = []
    if save_result:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            for area, cube in cubes.items():
                var = cube.var_name
                filename = path.name
                outpath = pathlib.Path(outputdir) / template
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
            gridscheme='area', nproc=1, outputdir=".", template=TEMPLATE, tempdir=None):
    targetgrid = None
    if regrid:
        targetgrid = ksc.utils.coord.create_grid()

    func = functools.partial(process_single, areas=areas, targetgrid=targetgrid,
                             save_result=save_result, average_area=average_area,
                             gridscheme=gridscheme, outputdir=outputdir, template=template,
                             mp=(nproc > 1), tempdir=tempdir)
    if nproc == 1:
        data = list(itertools.chain.from_iterable(map(func, paths)))
    else:  # Need maxtaskperchild, to avoid multiprocessing getting stuck
        with multiprocessing.Pool(nproc, maxtasksperchild=1) as pool:
            data = list(itertools.chain.from_iterable(pool.map(func, paths)))
    return data


def parse_args():
    """Parse the command line arguments"""

    areas = list(ksc.config.AREAS.keys())

    class ListAreas(argparse.Action):
        """Helper class for argparse to list available areas and exit"""
        def __call__(self, parser, namespace, values, option_string):
            print("\n".join(areas))
            parser.exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+', help="Input files. "
                        "Globbing patterns (including recursive globbing with '**') allowed.")
    parser.add_argument('--area', action='append', required=True,
                        choices=areas, help="One or more area names")
    parser.add_argument('--outputdir', default=".", help="Main output directory")
    parser.add_argument('--template', default=TEMPLATE,
                        help="Output path template, including subdirectory")
    parser.add_argument('-v', '--verbosity', action='count',
                        default=0, help="Verbosity level")
    parser.add_argument('-P', '--nproc', type=int, default=1,
                        help="Number of simultaneous processes")
    parser.add_argument('--list-areas', action=ListAreas, nargs=0,
                        help="List availabe areas and quit")
    parser.add_argument('--regrid', action='store_true',
                        help="Regrid the data (to a 1x1 deg. grid)")
    parser.add_argument('--no-save-results', action='store_true',
                        help="Store the resulting extracted datasets on disk")
    parser.add_argument('--no-average-area', action='store_true',
                        help="Don't average the extracted areas")
    parser.add_argument('--tempdir')
    args = parser.parse_args()
    # Expand any glob patterns in args.files
    args.files = list(itertools.chain.from_iterable(glob.glob(pattern) for pattern in args.files))
    args.save_result = not args.no_save_results
    args.average_area = not args.no_average_area
    args.area = {name: kcs.config.AREAS[name] for name in args.area}
    return args


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

    ksc_logger = logging.getLogger('ksc')
    ksc_logger.setLevel(level)
    ksc_logger.addHandler(handler)


def run(paths, areas, regrid=False, save_result=True, average_area=True, nproc=1,
        outputdir=".", template=TEMPLATE, tempdir=None):
    paths = [pathlib.Path(str(path)) if not isinstance(path, pathlib.Path) else path
             for path in paths]
    # No need to regrid the EC-EARTH data: all on the same grid
    data = process(paths, areas, regrid=regrid, save_result=save_result, average_area=average_area,
                   nproc=nproc, outputdir=outputdir, template=template, tempdir=tempdir)

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
    kcs_logger = logging.getLogger('kcs')
    kcs_logger.setLevel(level)
    kcs_logger.addHandler(handler)


def main():
    args = parse_args()
    setup_logging(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)
    run(args.files, args.area, regrid=args.regrid,
        save_result=args.save_result, average_area=args.average_area,
        nproc=args.nproc, outputdir=args.outputdir, template=args.template,
        tempdir=args.tempdir)
    logger.debug("%s finished", sys.argv[0])


if __name__ == "__main__":
    main()
