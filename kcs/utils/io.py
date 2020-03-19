from collections.abc import Iterable
import pathlib
import logging
import iris
import iris.cube
import iris.coord_categorisation
import iris.analysis
import iris.util
try:
    from iris.util import equalise_attributes
except ImportError:   # Iris 2
    from iris.experimental.equalise_cubes import equalise_attributes
import iris.exceptions
from .date import make_year_constraint_all_calendars
from .constraints import CoordConstraint


logger = logging.getLogger(__name__)


def load_cube(paths, variable_name=None): #, timespan, areas=None, average_area=True, targetgrid=None):
    """Read datasets from paths into Iris cubes.

    Combines cubes if there are more than one dataset in the same file.

    Returns a list of lists. Inner lists corresponds to the areas (in
    order), outer lists corresponds to the paths

    """

    if isinstance(paths, (str, pathlib.Path)):
        if variable_name:
            cubes = iris.load_cubes(str(paths), constraints=variable_name)
        else:
            cubes = iris.load_cubes(str(paths))
    else:
        if variable_name:
            cubes = iris.load([str(path) for path in paths], constraints=variable_name)
        else:
            cubes = iris.load([str(path) for path in paths])
    # Select only the cubes with 3/4D data (time, lat, long, height)
    cubes = iris.cube.CubeList([cube for cube in cubes if len(cube.coords()) >= 3])

    if len(cubes) == 0:
        return None
    iris.experimental.equalise_cubes.equalise_attributes(cubes)
    iris.util.unify_time_units(cubes)

    try:
        cube = cubes.concatenate_cube()
    except iris.exceptions.ConcatenateError as exc:
        logger.warning("%s for %s", exc, str(paths))
        logger.warning("Using only the first cube of [%s]", cubes)
        #print(type(cubes))
        #print(paths)
        #print(cubes)
        #print([cube.coord('time')[[0, -1]] for cube in cubes])
        #from collections import OrderedDict
        #names = [cube.metadata.name() for cube in cubes]
        #unique_names = list(OrderedDict.fromkeys(names))
        #print(names, unique_names)
        #for cube in cubes:
        #    print(len(cube.coord('latitude').points), len(cube.coord('latitude').bounds))
        ##print(cubes[0].coord('longitude').bounds == cubes[1].coord('longitude').bounds)
        #print(cubes[0].coord('latitude').points - cubes[1].coord('latitude').points)
        #import sys; sys.exit()
        cube = cubes[0]  # iris.load always returns a cubelist, so just take the first element
        #raise
    return cube
#    if timespan is not None:
#        if isinstance(timespan, (tuple, list)) and len(timespan) == 2:
#            timespan = make_year_constraint_all_calendars(timespan[0], timespan[1])
#        calendar = cube.coord('time').units.calendar
#        time_constraint = timespan.get(calendar, timespan['default'])
#        timecube = cube.extract(time_constraint)
#    else:
#        timecube = cube
#    if timecube is None:
#        return [None] * len(areas)


def extract_areas(cube, var, areas=None, targetgrid=None, average_area=True, gridscheme='area'):
    """Regrid, extract and average multiple areas from a cube for a given variable"""

    if areas is None:
        areas = ['global']
    if isinstance(areas, str):
        areas = [areas]

    if targetgrid is not None:
        if gridscheme == 'area':
            scheme = iris.analysis.AreaWeighted()
        elif gridscheme == 'linear':
            scheme = iris.analysis.Linear()
        else:
            scheme = iris.analysis.Linear()
        gridcube = timecube.regrid(targetgrid, scheme)
    else:
        gridcube = timecube

    results = []
    for area in areas:
        excube = gridcube.copy()
        if area is not None:
            if isinstance(area, iris.Constraint):
                excube = excube.extract(area)
            elif isinstance(area, dict) and 'latitude' in area and 'longitude' in area:
                if (isinstance(area['latitude'], (list, tuple)) and len(area['latitude']) == 2 and
                    isinstance(area['longitude'], (list, tuple)) and len(area['longitude']) == 2):
                    long_constraint = CoordConstraint(area['longitude'][0], area['longitude'][1])
                    lat_constraint = CoordConstraint(area['latitude'][0], area['latitude'][1])
                    constraint = iris.Constraint(longitude=long_constraint, latitude=lat_constraint)
                    excube = excube.extract(constraint)
                else:
                    coords = [('latitude', area['latitude']), ('longitude', area['longitude'])]
                    excube = excube.interpolate(coords, iris.analysis.Linear())
        if excube is None:
            results.append(None)
            continue

        iris.coord_categorisation.add_season(excube, 'time')
        iris.coord_categorisation.add_season_year(excube, 'time')
        iris.coord_categorisation.add_year(excube, 'time')

        if average_area and (len(excube.coord('latitude').points) > 1 or
                             len(excube.coord('longitude').points) > 1):
            weights = iris.analysis.cartography.area_weights(excube)
            excube_meanarea = excube.collapsed(['latitude', 'longitude'],
                                               iris.analysis.MEAN, weights=weights)
        else:
            excube_meanarea = excube
        results.append(excube_meanarea)

    return results
