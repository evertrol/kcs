import sys
import os
from pathlib import Path
from collections import namedtuple
import functools
import multiprocessing as mp
import warnings
import logging
import numpy as np
import iris
from ksc.utils.io import load_cube
from ksc.utils.coord import CoordConstraint
from ksc.utils.date import make_year_constraint_all_calendars


RCP_HISTORICAL = -1
NPROC = 20
PathList = namedtuple('PathList', ['pathlist', 'rcp', 'model', 'realization', 'areas'])


def get_paths(cmip5dir, rcps, var, models=None):
    data = []
    for rcp in rcps:
        subdir = cmip5dir
        subdir /= f"historical/Amon/{var}" if rcp == RCP_HISTORICAL else f"rcp{rcp}/Amon/{var}"
        if models is None:
            modelnames = [path.stem for path in subdir.glob("*")]
        else:
            modelnames = models
        for model in modelnames:
            modeldir = subdir / model
            for rdir in modeldir.glob("r*"):
                pathlist = list(rdir.glob("[a-z]*.nc"))
                data.append(PathList(pathlist, rcp, model, rdir.stem, {}))
    return data


def get_areas():
    coords = {
        'rhinebasin': {'w': 6, 'e': 9, 'n': 52, 's': 47},
        'nl': dict(w=3.3, e=7.1, s=50.0, n=53.6),
        'nlbox': dict(w=4.5, e=8, s=50.5, n=53),
        'weurbox': dict(w=4, e=14, s=47, n=53),
    }

    coord = coords['nlbox']
    long_constraint = CoordConstraint(coord['w'], coord['e'])
    lat_constraint = CoordConstraint(coord['s'], coord['n'])
    nlbox = iris.Constraint(longitude=long_constraint, latitude=lat_constraint)
            #longitude=lambda c: coord['w'] <= c <= coord['e'],
            #latitude=lambda c: coord['s'] <= c <= coord['n'])
    coord = coords['weurbox']
    weurbox = iris.Constraint(longitude=long_constraint, latitude=lat_constraint)
            #longitude=lambda c: coord['w'] <= c <= coord['e'],
            #latitude=lambda c: coord['s'] <= c <= coord['n'])
    areas = dict(names=['nlpoint', 'global', 'nlbox', 'weurbox'],
                 constraints=[{'latitude': 51.25, 'longitude': 6.25}, None, nlbox, weurbox])
    return areas


def get_timespan():
    timespan = {}
    for exp, start, end in [['historical', 1950, 2005], ['rcp', 2006, 2099]]:
        timespan[exp] = make_year_constraint_all_calendars(start, end)
    return timespan


def load_cube_timespan(item, variable_name, timespan, areas):
    print(item.model, item.rcp)
    timekey = 'historical' if item.rcp == RCP_HISTORICAL else 'rcp'
    results = load_cube(item.pathlist, variable_name=variable_name,
                        timespan=timespan[timekey], areas=areas['constraints'])
    results = dict(zip(areas['names'], results))
    return results


def read_item(item, areas, timespan, var, var_name, targetgrid=None):
    timekey = 'historical' if item.rcp == RCP_HISTORICAL else 'rcp'
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning,
                                message=f"Missing CF-netCDF measure variable 'areacella', "
                                "referenced by netCDF variable '{var}'")
        warnings.filterwarnings("ignore", category=UserWarning,
                                message="Using DEFAULT_SPHERICAL_EARTH_RADIUS.")
        results = load_cube(item.pathlist, variable_name=var_name,
                            timespan=timespan[timekey], areas=areas['constraints'],
                            targetgrid=targetgrid)
    results = dict(zip(areas['names'], results))

    item = PathList(item.pathlist, item.rcp, item.model, item.realization, results)
    return item


def save_extracted_item(item, dirname):
    model = item.model
    rcp = item.rcp
    realization = item.realization
    for area, cube in item.areas.items():
        if cube is None:
            continue
        os.makedirs(dirname, exist_ok=True)
        if rcp == RCP_HISTORICAL:
            fname = f"{model}-historical-{realization}.nc"
        else:
            fname = f"{model}-rcp{rcp}-{realization}.nc"
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            iris.save(cube, os.path.join(dirname, fname))


def create_grid(nlong=360, nlat=180):

    # Start and end half a step size away from the edges
    offset = 360 / nlong / 2
    start = 0 + offset
    end = 360 - offset
    longitude = np.linspace(start, end, nlong)
    offset = 180 / nlat / 2
    start = -90 + offset
    end = 90 - offset
    latitude = np.linspace(start, end, nlat)
    data = np.zeros((len(latitude), len(longitude)), dtype=np.float)
    longitude = iris.coords.DimCoord(longitude, standard_name='longitude', units='degrees')
    longitude.guess_bounds()
    latitude = iris.coords.DimCoord(latitude, standard_name='latitude', units='degrees')
    latitude.guess_bounds()

    targetgrid = iris.cube.Cube(data, dim_coords_and_dims=[(latitude, 0), (longitude, 1)])
    return targetgrid


def process(item, areas, timespan, var, var_name, targetgrid=None, save_extracted=False):
    item = read_item(item, areas, timespan, var, var_name, targetgrid)
    if save_extracted:
        for area in item.areas.keys():
            dirname = (f"extracted-{var}-{area}" if not isinstance(save_extracted, str)
                       else save_extracted.format(**locals()))
            save_extracted_item(item, dirname)
    return item


def run(models, cmip5dir, var, rcps, var_name, options, nproc=1):
    data = get_paths(cmip5dir, rcps, var, models=models)
    if not data:
        logging.warning("No files found")
        return
    areas = get_areas()
    timespan = get_timespan()
    targetgrid = None
    if options['regrid']:
        targetgrid = create_grid()

    process_partial = functools.partial(
        process, areas=areas, timespan=timespan,
        var=var, var_name=var_name, targetgrid=targetgrid,
        save_extracted=options['save-extracted']
    )
    if nproc == 1:
        data = map(process_partial, data)
    else:
        with mp.Pool(nproc) as pool:
            data = pool.map(process_partial, data)
    for item in data:
        if None in item.areas.values():
            print(f"Removing item {item.model}-{item.rcp}-{item.realization}",
                  list(item.areas.values()))
    data = [item for item in data if None not in item.areas.values()]
    print(len(data))


def main():
    nproc = 1
    options = {'save-extracted': "area-averaged-{var}-{area}",
               'regrid': True,}
    try:
        cmip5dir = Path(sys.argv[1])
    except IndexError:
        cmip5dir = Path("/data/cmip5")
    var = "tas"
    #var_name = "precipitation_flux"
    var_name = "air_temperature"
    # -1 indicates historical model runs
    rcps = [RCP_HISTORICAL, 45, 60, 85]
    # Use None for all models
    bomodels = ['bcc-csm1-1']
    models = None
    run(models, cmip5dir, var, rcps, var_name, options, nproc=nproc)


if __name__ == "__main__":
    main()
