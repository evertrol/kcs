import sys
from .default import config_toml_string


def main():
    try:
        section = sys.argv[1]
    except IndexError:
        print(config_toml_string.strip())
        return

    found = False
    for line in config_toml_string.split('\n'):
        if line == f"[{section}]":
            found = True
        elif line.startswith('[') and not line.startswith(f"[{section}"):  # new section
            found = False
        if found:
            print(line)


if __name__ == '__main__':
    main()
