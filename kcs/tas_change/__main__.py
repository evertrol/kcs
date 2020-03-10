"""Calculate the (CMIP) percentile distribution for the given input
datasets.

The input datasets should correspond to a single variable, e.g., the
air temperature ('tas').

Example usage:

$ python -m kcs.tas_change  @cmip6-tas-global-averaged.list  \
    --on-no-match=randomrun -vvv  --norm-by=run  --reference-period 1991 2020

"""

import sys
import argparse
import logging
import pathlib
from itertools import chain
import iris
import kcs.utils.date
import kcs.utils.logging
import kcs.utils.argparse
import kcs.utils.attributes
import kcs.utils.matching
from kcs.utils.atlist import atlist
from . import run


HISTORICAL_KEY = 'historical'
REFERENCE_PERIOD = (1981, 2010)
MINDATA = {'historical': 20, 'future': 4}
PERC_PERIOD = (1950, 2100)


logger = logging.getLogger('kcs.tas_change')  # pylint: disable=invalid-name


def read_data(paths, info_from=('attributes', 'filename'),
              attributes=None, filename_pattern=None):
    """DUMMY DOC-STRING"""
    cubes = [iris.load_cube(str(path)) for path in paths]

    # Get the attributes, and create a dataframe with cubes & attributes
    dataset = kcs.utils.attributes.get(
        cubes, paths, info_from=info_from,
        attributes=attributes, filename_pattern=filename_pattern)

    return dataset


def parse_args():
    """DUMMY DOC-STRING"""
    parser = argparse.ArgumentParser(parents=[kcs.utils.argparse.parser],
                                     conflict_handler='resolve')
    parser.add_argument('files', nargs='+', help="Input file paths")
    parser.add_argument('--outfile', required=True,
                        help="Output CSV file with the distribution "
                        "percentiles-versus-year table. "
                        "The default is 'distribution-percentiles.csv'")
    parser.add_argument('--historical-key', default=HISTORICAL_KEY,
                        help="Identifier for the historical experiment "
                        f"(default '{HISTORICAL_KEY}').")
    parser.add_argument('--period', nargs=2, type=int, default=PERC_PERIOD,
                        help="Period (years, inclusive) for which to calculate "
                        "the percentile distribution. Default is 1950-2100.")
    parser.add_argument('--relative', action='store_true',
                        help="Calculate relative change (values will be "
                        "a percentage change between future and reference period")
    parser.add_argument('--season', choices=['djf', 'mam', 'jja', 'son'],
                        help="Season to extract / use. Leave blank to "
                        "use full years")
    parser.add_argument('--no-year-average', action='store_true',
                        help="Do not use yearly/seasonal averages")
    parser.add_argument('--reference-period', nargs=2, type=int,
                        default=list(REFERENCE_PERIOD),
                        help="Reference period (to normalize EC-EARTH data): start and end year. "
                        "Years are inclusive (i.e., Jan 1 of 'start' up to and "
                        f"including Dec 31 of 'end'). Default {REFERENCE_PERIOD}.")
    parser.add_argument('--norm-by', choices=['model', 'experiment', 'run'],
                        default='run',
                        help="Normalize data (to --reference-period) per 'model', "
                        "'experiment' or 'run'. These normalization choices are from "
                        "'wide' (large spread) to 'narrow' (little spread).")
    parser.add_argument('--match-by', choices=['model', 'ensemble'], default='ensemble',
                        help="Match future and historical runs by model (very generic) "
                        "or ensemble (very specific). Default is 'ensemble'.")
    parser.add_argument('--match-info-from', choices=['attributes', 'filename'], nargs='+',
                        default=['attributes', 'filename'],
                        help="Where to get the information from to match runs. 'attributes' "
                        "will use the NetCDF attributes, 'filename' will attempt to deduce "
                        "them from the filename. Both can be given, in order of importance, "
                        "that is, the second then serves as a fallback. "
                        "Default is 'attributes filename'. "
                        "Important attributes are 'parent-experiment-rip', 'realization', "
                        "'physics_version', 'initialization_method', 'experiment', 'model_id'")
    parser.add_argument('--on-no-match', choices=['error', 'remove', 'randomrun', 'random'],
                        default='error', help="What to do with a (future) experiment run that "
                        "has no matching run. 'error' raise an exception, 'remove' removes "
                        "(ignores) the run. 'randomrun' picks a random history run with the "
                        "same 'physics' and 'initialization' values (but a different "
                        "'realization' value), while 'random' picks a random history run from "
                        "all ensembles for that model.")
    parser.add_argument('--average-experiments', action='store_true', help="Average ensemble "
                        "runs over their model-experiment, before calculating percentiles.")
    args = parser.parse_args()
    args.paths = [pathlib.Path(filename) for filename in args.files]
    args.average_years = not args.no_year_average
    return args


def main():
    """DUMMY DOC-STRING"""
    args = parse_args()
    kcs.utils.logging.setup(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    paths = list(chain.from_iterable(atlist(path) for path in args.paths))
    dataset = read_data(paths)
    dataset = kcs.utils.matching.match(
        dataset, match_by=args.match_by, on_no_match=args.on_no_match,
        historical_key=args.historical_key)

    result, _ = run(dataset, historical_key=args.historical_key,
                    season=args.season, average_years=args.average_years,
                    relative=args.relative, reference_period=args.reference_period,
                    period=args.period, normby=args.norm_by,
                    average_experiments=args.average_experiments)
    result.to_csv(args.outfile, index_label="date")
    logger.info("Done processing: percentiles = %s", result)


if __name__ == '__main__':
    main()
