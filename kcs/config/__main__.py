"""DUMMY DOCSTRING"""

import sys
from .default import CONFIG_TOML_STRING


def main():
    """DUMMY DOCSTRING"""

    try:
        section = sys.argv[1]
    except IndexError:
        print(CONFIG_TOML_STRING.strip())
        return

    found = False
    for line in CONFIG_TOML_STRING.split('\n'):
        if line == f"[{section}]":
            found = True
        elif line.startswith('[') and not line.startswith(f"[{section}"):  # new section
            if found:
                break
            found = False
        if found:
            print(line)


if __name__ == '__main__':
    main()
