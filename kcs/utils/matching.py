"""
# Match experiment by ensemble ID
# Matches historical and future experiments
match_experiments:
#%#
# What attribute(s) to match the historical and future experiments by
# Valid values: 'model', 'ensemble'
# The default is by 'ensemble'
match_by: 'ensemble'
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
#%#
# What to do with future scenarions that don't have a
# matching historical run?
# Valid values: 'remove', 'random', 'randomrun', 'exception'
# - 'remove': if historical data is missing, remove the
# future experiment.
# - 'random': pick an entirely random historical run from the model
# - 'randomrun': keep i # & p the same, find a random
#   historical run with a (different) r. If there's still no
#   match to be found, an exception is raised.
# - 'error' the script raises an exception if no matching
#   historical run is found.
# The default is 'error': to raise an exception
fix_nonmatching_historical: randomrun
"""

import re
from collections import defaultdict
import pandas as pd
from ..types import DataAttrs


ATTRIBUTES = {
    'experiment': "experiment",
    'model': "model_id",
    'realizaton': "realization",
    'initialization': "initialization_method",
    'physics': "physics_version",
}
ATTRIBUTES_EMPTY = {
    'experiment': "",
    'model': "",
    'realizaton': 0,
    'initialization': 0,
    'physics': 0,
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


def _get_attributes_from_cube(attrs, match_by, on_no_match, attributes):
    """DUMMY"""
    data, found = {}, True

    try:
        data['experiment'] = attrs[attributes['experiment']]
    except KeyError:
        found = False

    # We always need the model, even if match_by == 'realization'
    try:
        data['model'] = attrs[attributes['model']]
    except KeyError:
        found = False

    if match_by != 'model':
        for key in ['realization', 'physics', 'initialization']:
            try:
                data[key] = int(attrs[attributes[key]])
            except KeyError:
                found = False
            except ValueError:
                raise ValueError(f"{key} is not an integer: {attrs[attributes[key]]}")

    return data, found


def _get_attributes_from_filename(filename, match_by, on_no_match, pattern):
    """DUMMY"""
    data, found = {}, True

    match = re.search(pattern['experiment'], filename)
    if match:
        data['experiment'] = match.group('experiment')
    else:
        found = False

    # We always need the model, even if match_by == 'ensemble'
    match = re.search(pattern['model'], filename)
    if match:
        data['model'] = match.group('model')
    else:
        found = False

    if match_by != 'model':
        for key in ['realization', 'physics', 'initialization']:
            match = re.search(pattern[key], filename)
            if match:
                data[key] = int(match.group(key))
            else:
                found = False

    return data, found


def get_attributes(cube, path, match_by, info_from, on_no_match, attributes, pattern):
    """DUMMY"""
    attrs = cube.attributes
    filename = path.name
    data= {}
    for info in info_from:
        found = True
        if info == 'attributes':
            result, found = _get_attributes_from_cube(attrs, match_by, on_no_match, attributes)
        elif info == 'filename':
            result, found = _get_attributes_from_filename(filename, match_by, on_no_match, pattern)
        else:
            raise ValueError("incorrect argument for 'info_from': should be 'attributes' or 'filename'")
        data.update(result)
        if found:
            break
    else:
        # We may still have all the necessary info
        if 'experiment' not in data:
            raise KeyError("missing experiment info")
        if match_by == 'model':
            if 'model' not in data:
                raise KeyError("missing model info")
        else:
            if not ('realization' in data and 'physics' in data and 'initialization' in data):
                raise KeyError("missing ensemble info")

    return data


def match(dataset, match_by='ensemble', on_no_match='error', historical_key='historical'):
    """DUMMY DOC-STRING"""

    exp_selection = dataset['experiment'] != 'historical'
    hist_selection = dataset['experiment'] == 'historical'

    dataset['match_historical_run'] = None

    print(dataset.head())

    if match_by == 'ensemble':
        keys = ['model', 'realization', 'initialization', 'physics']
    else:
        keys = ['model']

    groups = []
    for index, group in dataset.groupby(keys):
        hist_sel = group['experiment'] == 'historical'
        exp_sel = group['experiment'] != 'historical'
        if match_by == 'ensemble' and sum(hist_sel) > 1:
            raise ValueError(f"Found multiple historical runs in "
                             f"{index[0]}, r{index[1]}i{index[2]}p{index[3]}")
        if sum(hist_sel) == 0:
            if on_no_match == 'error':
                msg = f"no historical data for {index[0]}"
                if match_by == 'ensemble':
                    msg += f", r{index[1]}i{index[2]}p{index[3]}"
                raise ValueError(msg)
            elif on_no_match == 'remove' or on_no_match == 'ignore':
                continue

        # Grab the historical cube
        cube = group.loc[hist_sel, 'cube']
        # Pick the first (if multiple, e.g. for match_by=='model') historical cubes
        if isinstance(cube, pd.Series):
            cube = cube.iloc[0]
        group.loc[exp_sel, 'match_historical_run'] = [cube] * sum(exp_sel)
        groups.append(group)

    dataset = pd.concat(groups)

    return dataset


def run(cubes, paths, match_by='ensemble', info_from=('attributes', 'filename'),
        on_no_match='error', historical_key='historical',
        attributes=None, filename_pattern=None):
    """Match dataset experiments, historical with future experiments

    Anything where the experiment does not match 'historical' is
    assumed to be a future experiment (rcp, ssp).

    Parameters
    ----------

    - match_by: 'ensemble' or 'model'

    - info_from: 'attributes' or 'filename', or both in a list or tuple

    - on_no_match: 'remove', 'random', 'randomrun', or 'error'

          What to do with a future run that has no matching historical run.

          - 'remove' the dataset

          - 'random': pick an entirely random historical run from the model

          - 'randomrun': keep i # & p the same, find a random
            historical run with a (different) r. If there's still no
            match to be found, an exception is raised.

          - 'error' the script raises an exception if no matching
            historical run is found.

    - historical_key: string

        What value for the 'experiment' attribute indicates a historical run?

    - attributes: dict, or None

        The attribute names for the following info. This information
        is given as a dict, with the keys below. The default values
        are given after each key, and is used when attributes=None.
        - experiment: "experiment"
        - model: "model_id"
        - realization: "realization"
        - initialization: "initialization"
        - physics: "physics_version"

      Note that the dictionary does not have to be complete, depending
      on the choice of `match_by` and `on_no_match`.

    """

    if attributes is None:
        attributes = ATTRIBUTES
    if filename_pattern is None:
        filename_pattern = FILENAME_PATTERN
    if isinstance(info_from, str):
        info_from = [info_from]

    dataset = []
    for cube, path in zip(cubes, paths):
        default = ATTRIBUTES_EMPTY.copy()
        attrs = get_attributes(cube, path, match_by=match_by, info_from=info_from,
                               on_no_match=on_no_match, attributes=attributes,
                               pattern=filename_pattern)
        default.update(attrs)
        dataset.append(
            dict(path=path, cube=cube, realization=default['realization'],
                 initialization=default['initialization'],
                 experiment=default['experiment'],
                 physics=default['physics'],
                 model=default['model'])
            )

    dataset = pd.DataFrame(dataset)

    dataset = match(dataset, match_by=match_by, on_no_match=on_no_match)

    return dataset
