"""DUMMY DOCSTRING"""


import os
import logging
import toml
from .default import CONFIG_TOML_STRING


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def adjust_config(config):
    """DUMMY DOCSTRING"""

    # Small adjustment: TOML has only string keys in dicts, but here,
    # integer keys are needed
    if 'resampling' in config:
        penalties = {int(key): value for key, value in config['resampling']['penalties'].items()}
        config['resampling']['penalties'] = penalties

    # Fill in empty definitions if not available
    if 'data' in config:
        subsections = ['extra', 'cmip']
        for sub in subsections:
            for key, value in config['data']['matching'].items():
                if key in subsections:
                    continue
                if key not in config['data'][sub]['matching']:
                    config['data'][sub]['matching'][key] = value


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


def read_config(filename):
    """DUMMY DOCSTRING"""

    if not filename:
        filename = os.path.join(os.getcwd(), 'kcs-config.toml')
        if not os.path.exists(filename):
            logger.debug("using built-in config")
            return default_config
        logger.debug("using kcs-config.toml in local directory")
    else:
        logger.debug("reading config file %s", filename)

    with open(filename) as fh:  # pylint: disable=invalid-name
        config = toml.load(fh)

    for name, section in config.items():
        if 'include' in section:
            fname = section['include']
            if fname.startswith('./'):  # relative with respect to the main config file
                dirname = os.path.dirname(filename)
                fname = os.path.normpath(os.path.join(dirname, fname))
            with open(fname) as fh:  # pylint: disable=invalid-name
                section_config = toml.load(fh)
            if name not in section_config:
                raise KeyError(f"{fname} is missing the [{name}] heading")
            nested_update(section_config[name], section)
            config[name] = section_config[name]
            del config[name]['include']

    adjust_config(config)

    # Replace the default configuration per section/table
    for name, section in config.items():
        default_config[name] = section

    return default_config


default_config = toml.loads(CONFIG_TOML_STRING)  # pylint: disable=invalid-name
adjust_config(default_config)
