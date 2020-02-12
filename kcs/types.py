from collections import namedtuple


Data = namedtuple('Data', ['path', 'realization', 'area', 'cube'])
DataAttrs = namedtuple('DataAttrs', ['path', 'realization', 'area', 'cube',
                                     'initialization', 'experiment', 'physics',
                                     'model'])
