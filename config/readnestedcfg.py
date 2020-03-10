#! /usr/bin/env python

import sys
import os
from pprint import pprint
import toml


try:
    fname = sys.argv[1]
except IndexError:
    fname = 'config/config-split.toml'
dirname = os.path.dirname(fname)
with open(fname) as fh:
    config = toml.load(fh)


def nested_update(current, new):
    """Nested update of dicts

    Inner dicts of `current` are not overwritten: only immutable values (strings,
    numbers, booleans) are changed.

    Lists are extended with values from `new`.

    """

    for key, value in new.items():
        if isinstance(value, dict) and isinstance(current[key], dict):
            nested_update(current[key], value)
        elif isinstance(value, list) and isinstance(current[key], list):
            current[key].extend(value)
        else:
            current[key] = value


for name, section  in config.items():
    if 'include' in section:
        fname = section['include']
        if fname.startswith('./'):  # relative with respect to the main config file
            fname = os.path.normpath(os.path.join(dirname, fname))
        with open(fname) as fh:
            section_config = toml.load(fh)
        if name not in section_config:
            raise KeyError(f"{fname} is missing the [{name}] heading")
        nested_update(section_config[name], section)
        config[name] = section_config[name]
        del config[name]['include']


pprint(config)

# try:
#     fname = sys.argv[2]
# except IndexError:
#     fname = 'config/config.toml'
# with open(fname) as fh:
#     config2 = toml.load(fh)
# for key in config:
#     assert key, config[key] == config2[key]
