"""
#%#
# Where to get the information for the matching from
# Valid values: 'attributes', 'filename'.
#%#
# Use 'attributes' to match the datasets by information from
# attributes, specifically 'realization', 'physics_version',
# 'initaliazation_method', 'experiment', 'model_id', and
# 'parent_experiment_rip'. For filename, relevant parts of
# the filenames are used (using regular expressions for
# e.g. the ensemble part of the filename). If information is
# missing, an exception is raised.
# Both options can be given in a list if the first option
# fails, the next one is tried. If both fail, an exception
# is raised.
#%#
# The default is [attributes, filename]: first by
# attributes, and use the filename as a fallback.
info_from: [attributes, filename]
"""
import logging
import re
import pandas as pd


ATTRIBUTES = {
    'experiment': "experiment",
    'model': "model_id",
    'realization': "realization",
    'initialization': "initialization_method",
    'physics': "physics_version",
    'prip': "parent_experiment_rip"
}
ATTRIBUTES_DEFAULT = {
    'experiment': "",
    'model': "",
    'realization': 0,
    'initialization': 0,
    'physics': 0,
    'prip': None,
}
# Note that there is no use of \w+, since this includes the underscore
# Hence, sets like [-A-Za-z0-9]+ are used
FILENAME_PATTERN = {
    'var': r'^(?P<var>[a-z]+)_',
    'mip': r'^[a-z]+_(?P<mip>[A-Za-z]+)_',
    'model': r'^[a-z]+_[A-Za-z]+_(?P<model>[-A-Za-z0-9]+)_',
    'experiment': r'^[a-z]+_[A-Za-z]+_[-A-Za-z0-9]+_(?P<experiment>[A-Za-z0-9]+)_',
    'realization': r'_r(?P<realization>\d+)i\d+p\d+_',
    'initialization': r'_r\d+i(?P<initialization>\d+)p\d+_',
    'physics': r'_r\d+i\d+p(?P<physics>\d+)_',
}

logger = logging.getLogger(__name__)


def _get_attributes_from_cube(attrs, attributes):
    """DUMMY"""
    data, found = {}, True

    try:
        data['experiment'] = attrs[attributes['experiment']]
    except KeyError:
        found = False

    try:
        data['model'] = attrs[attributes['model']]
    except KeyError:
        found = False

    try:
        prip = attrs[attributes['prip']]
        match = re.search(r'r(\d+)i(\d+)p(\d+)', prip)
        if match:
            data['prip'] = int(match.group(1)), int(match.group(2)), int(match.group(3))
        else:
            data['prip'] = None
    except KeyError:
        data['prip'] = None

    for key in ['realization', 'physics', 'initialization']:
        try:
            data[key] = int(attrs[attributes[key]])
        except KeyError:
            found = False
        except ValueError:
            raise ValueError(f"{key} is not an integer: {attrs[attributes[key]]}")

    return data, found


def _get_attributes_from_filename(filename, pattern):
    """DUMMY"""
    data, found = {}, True

    match = re.search(pattern['experiment'], filename)
    if match:
        data['experiment'] = match.group('experiment')
    else:
        found = False

    match = re.search(pattern['model'], filename)
    if match:
        data['model'] = match.group('model')
    else:
        found = False

    for key in ['realization', 'physics', 'initialization']:
        match = re.search(pattern[key], filename)
        if match:
            data[key] = int(match.group(key))
        else:
            found = False

    return data, found


def get_single(cube, path, info_from, attributes, pattern):
    """Obtain attributes for a single cube"""
    attrs = cube.attributes
    filename = path.name
    data = {}
    # Run in reversed, so the information from the most valuable asset
    # is added last (in `data.update(result)`)
    for info in reversed(info_from):
        if info == 'attributes':
            result, _ = _get_attributes_from_cube(attrs, attributes)
        elif info == 'filename':
            result, _ = _get_attributes_from_filename(filename, pattern)
        else:
            raise ValueError("incorrect argument for 'info_from': "
                             "should be 'attributes' or 'filename'")
        data.update(result)

    if 'experiment' not in data:
        raise KeyError("missing experiment info")
    if 'model' not in data:
        raise KeyError("missing model info")
    if not ('realization' in data and 'physics' in data and 'initialization' in data):
        raise KeyError("missing ensemble RIP info")

    return data


def get(cubes, paths, info_from=('attributes', 'filename'),
        attributes=None, filename_pattern=None):
    """Match dataset experiments, historical with future experiments

    Anything where the experiment does not match 'historical' is
    assumed to be a future experiment (rcp, ssp).

    Parameters
    ----------

    - match_by: 'ensemble' or 'model'

    - info_from: 'attributes' or 'filename', or both in a list or tuple

    - attributes: dict, or None

        The attribute names for the following info. This information
        is given as a dict, with the keys below. The default values
        are given after each key, and is used when attributes=None.
        - experiment: "experiment"
        - model: "model_id"
        - realization: "realization"
        - initialization: "initialization"
        - physics: "physics_version"

    """

    if attributes is None:
        attributes = ATTRIBUTES
    if filename_pattern is None:
        filename_pattern = FILENAME_PATTERN
    if isinstance(info_from, str):
        info_from = [info_from]

    dataset = []
    for cube, path in zip(cubes, paths):
        default = ATTRIBUTES_DEFAULT.copy()
        attrs = get_single(cube, path, info_from=info_from,
                           attributes=attributes, pattern=filename_pattern)
        default.update(attrs)
        default.update({'cube': cube, 'path': path})
        dataset.append(default)

    dataset = pd.DataFrame(dataset)
    dataset['experiment'] = dataset['experiment'].str.lower()

    return dataset
