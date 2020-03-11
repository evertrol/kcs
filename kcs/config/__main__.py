from .default import config_toml_string


if __name__ == '__main__':
    print(config_toml_string.strip())
    from pprint import pprint
    from . import default_config
    pprint(default_config)
