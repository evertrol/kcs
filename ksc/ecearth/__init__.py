import pathlib
import glob
import numpy as np
import iris


ALLSEASONS = ['djf', 'mam', 'jja', 'son']


def load_datasets(var, area, datadir=".", nmax=None,
                  template=("{var}-ecearth-{area}-area-averaged",
                            "{var}_Amon_ECEARTH23_rcp85_186001-210012_[0-9][0-9].nc")):
    datadir = pathlib.Path(datadir)
    cubes = {}
    subdir = template[0].format(**locals())
    filename = template[1].format(**locals())
    path = datadir / subdir / filename
    # glob.glob is a tad more shell-like than pathlib.Path.glob
    paths = [str(path) for path in glob.glob(str(path))]
    if nmax:
        paths = paths[:nmax]
    cubes = iris.load(paths)

    return cubes


def segment_data(cubes, period, control_period, nsections, seasons=None):
    """Given a list of cubes (or CubeList), return a dict with periods and seasons extracted"""

    if seasons is None:
        seasons = ALLSEASONS

    data = {season: {} for season in seasons}
    for key, years in zip(['control', 'future'], [control_period, period]):
        # Chop each cube into n-year segments
        span = (years[1] - years[0] + 1) // nsections
        constraint = iris.Constraint(year=lambda point: years[0] <= point <= years[1])
        excubes = [constraint.extract(cube) for cube in cubes]
        for season in seasons:
            constraint = iris.Constraint(season=lambda point: point == season)
            season_cubes = [constraint.extract(cube) for cube in excubes]
            data[season][key] = np.array([
                [iris.Constraint(year=lambda point: year <= point < year+span).extract(cube)
                 for year in range(years[0], years[1], span)]
                for cube in season_cubes], dtype=np.object)

    return data


def prepare_data(cubes, period, control_period, nsections, seasons=None):
    """Given a list of cubes (or CubeList), return a dict with periods and seasons extracted"""

    if seasons is None:
        seasons = ALLSEASONS

    data = {season: {} for season in seasons}
    for key, years in zip(['control', 'future'], [control_period, period]):
        # Chop each cube into n-year segments
        span = (years[1] - years[0] + 1) // nsections
        constraint = iris.Constraint(year=lambda point: years[0] <= point <= years[1])
        excubes = [constraint.extract(cube) for cube in cubes]
        for season in seasons:
            constraint = iris.Constraint(season=lambda point: point == season)
            season_cubes = [constraint.extract(cube) for cube in excubes]
            data[season][key] = np.array([
                [iris.Constraint(year=lambda point: year <= point < year+span).extract(cube)
                 for year in range(years[0], years[1], span)]
                for cube in season_cubes], dtype=np.object)

    return data
