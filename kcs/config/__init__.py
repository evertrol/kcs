import toml
from .default import config_toml_string


default_config = toml.loads(config_toml_string)
# Small adjustment: TOML has only string keys in dicts, but here,
# integer keys are needed
penalties = {int(key): value for key, value in default_config['resampling']['penalties'].items()}
default_config['resampling']['penalties'] = penalties


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
    with open(filename) as fh:
        config = toml.load(fh)

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

    penalties = {int(key): value for key, value in config['resampling']['penalties'].items()}
    config['resampling']['penalties'] = penalties

    return config



SCENARIO_HISTORICAL = -1

AREAS = {
    'rhinebasin': {'w': 6, 'e': 9, 'n': 52, 's': 47},
    'nl': dict(w=3.3, e=7.1, s=50.0, n=53.6),
    'nlbox': dict(w=4.5, e=8, s=50.5, n=53),
    'weurbox': dict(w=4, e=14, s=47, n=53),
    'global': None,
    'nlpoint': dict(latitude=51.25, longitude=6.25),
}
