"""Module to parse listings of files

This module provides a single function to parse a list of path names, possibly recursively.

The name of the module and function derive from the author's
familiarity with the identially-named files, with similar
functionality, used by the astronomical data processing tools IRAF and
F-tools.

"""

from pathlib import Path
import shlex


def atlist(path, maxnesting=10):
    """Return an iterator over the list of paths given in ``path``

    ``path`` can be a string or a ``pathlib.Path`` object. If the name
    is preceeded by an '@' character, the file will be opened and read
    line by line, interpreting each line as a path name.

    Nesting at-lists is allowed, up to a maximum level of
    ``maxnesting`` (which defaults to 10). This also prevents infinite
    recursion. At the maximum level, this function yields ``None``,
    for convenience: this ``None`` can be easily filtered out (e.g.,
    with ``filter(None, atlist(input_path))``); it does not raise a
    ``StopIteration``.

    At-lists inside an at-list are relative with respect to their
    parent at-list. Thus, only the 'root' level at-list will need the
    proper path with respect to the calling script.

    Path names should be one name per line.

    Path names inside an at-list are shell-escaped using the ``shlex``
    module. Whitespace, either leading, trailing or in the middle, is
    all removed, unless escaped with a backslash, or the full path
    name is quoted. That is, path names are treated as in a POSIX
    shell. With one path per line, this means path names can't be
    separated by just spaces or tabs.

    Comment lines are allowed, starting with a '#'. You can escape or
    quote the character if your filename starts with a '#'. Trailing
    comments are not allowed, and will be considered part of the path
    name.

    Empty lines result in an empty string, or the current directory
    when using ``pathlib.Path``; that is, ``PosixPath('.')``. The
    latter is the default behaviour of ``pathlib.Path('')``. Note that
    this can make a difference when filtering on falsy values:
    ``PosixPath('.')`` is a truthy value, while ``''`` is falsy.

    Supplying a non at-list as input ``path`` simply returns ``path``
    itself: no processing is done.

    The output type of the iterator is the same as that of the input
    ``path``: either a ``pathlib.Path`` or a ``str``.

    """

    if maxnesting < 0:
        yield
        return  # quit iterator

    _aspath = isinstance(path, Path)

    s_path = str(path)
    if s_path.startswith('@'):
        s_path = s_path[1:]
    else:
        yield path
        return  # quit iterator

    with open(s_path) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            if line.startswith('@'):
                newpath = ''.join(shlex.split(line[1:]))
                newpath = Path(s_path).parent / newpath
                newpath = f'@{newpath}'
                if _aspath:
                    newpath = Path(newpath)
                yield from atlist(newpath, maxnesting=maxnesting-1)
                continue
            outpath = ''.join(shlex.split(line))
            yield Path(outpath) if _aspath else outpath
