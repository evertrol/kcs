"""

Example usage:

$ python -m kcs.change_perc @cmip5-pr-nlpoint-averaged.list --season djf \
    --cmip-period 2036  2065 --relative --csvfile pr_change_2050_djf_nlpoint.csv -v

For runs that are already concatenated:

$ python -m kcs.change_perc @ecearth-pr-nlpoint-averaged.list --season jja \
    --period 2049 2078 --relative --run-changes pr_change_2050W_jja_nlpoint_ecearth.csv \
    --no-matching

"""

import sys
import argparse
import logging
import pathlib
import itertools
import pandas as pd
import iris
import iris.cube
from iris.util import unify_time_units
try:
    from iris.util import equalise_attributes
except ImportError:   # Iris 2
    from iris.experimental.equalise_cubes import equalise_attributes
import kcs.utils.argparse
import kcs.utils.logging
import kcs.utils.attributes
import kcs.utils.matching
from kcs.utils.atlist import atlist
from . import run


HISTORICAL_KEY = 'historical'
REFERENCE_PERIOD = (1981, 2010)

logger = logging.getLogger('kcs.change_perc')


def read_data(paths, attributes_from=('attributes', 'filename'),
              attributes=None, filename_pattern=None):
    """Read the dataset from nc files and get attribute information

    Returns a dataset in the form of a Pandas DataFrame.

    """
    cubes = [iris.load_cube(str(path)) for path in paths]

    # Get the attributes, and create a dataframe with cubes & attributes
    dataset = kcs.utils.attributes.get(
        cubes, paths, info_from=attributes_from,
        attributes=attributes, filename_pattern=filename_pattern)

    return dataset


def concat_cubes(dataset, historical_key=HISTORICAL_KEY):
    """Concatenate cubes into a dataset spanning the full time frame"""

    concatenated = pd.DataFrame(columns=dataset.columns)
    for model, group in dataset.groupby('model'):
        future_sel = group['experiment'] != historical_key
        for _, row in group.loc[future_sel, :].iterrows():
            cube = row['cube']
            matchid = row['index_match_run']
            histcube = dataset.loc[matchid, 'cube']
            time = cube.coord('time')
            start = time.units.num2date(time.points)[0]
            time = histcube.coord('time')
            end = time.units.num2date(time.points)[-1]
            if end > start:
                logger.warning("Historical experiment ends past the start of future experiment: "
                               "trimming historical dataset to match %s - %s r%di%dp%d",
                               model, row['experiment'],
                               row['realization'], row['initialization'], row['physics'])
                logger.debug("Historical end: %s. Future start: %s", end, start)
                logger.info("Constraining historical run to end before %s", start)
                # pylint: disable=cell-var-from-loop
                constraint = iris.Constraint(time=lambda cell: cell.point < start)
                histcube = constraint.extract(histcube)
                time = histcube.coord('time')
                end = time.units.num2date(time.points)[-1]
            # Since we may have matched on different realizations, set them to
            # equal, otherwise the concatenation procedure will fail
            histcube.replace_coord(cube.coord('realization'))
            cubes = iris.cube.CubeList([histcube, cube])
            equalise_attributes(cubes)
            unify_time_units(cubes)
            try:
                row['cube'] = cubes.concatenate_cube()
            except iris.exceptions.ConcatenateError as exc:
                logger.warning("DATA SKIPPED: Error concatenating cubes: %s - %s r%di%dp%d: %s",
                               model, row['experiment'], row['realization'],
                               row['initialization'], row['physics'], exc)
                continue
            logger.info("Concatenated %s - %s r%di%dp%d", model, row['experiment'],
                        row['realization'], row['initialization'], row['physics'])
            # By adding the rows with concatenated cubes,
            # we can construct a new DataFrame with only concatenated cubes,
            # but with all the essential attributes from the input dataset.
            concatenated = concatenated.append(row)
    concatenated.reset_index(inplace=True)
    logger.info("concatenated a total of %d realizations", len(concatenated))

    return concatenated


def parse_args():
    """DUMMY DOCSTRING"""
    parser = argparse.ArgumentParser(parents=[kcs.utils.argparse.parser],
                                     conflict_handler='resolve')
    parser.add_argument('files', nargs='+', help="Input data. This should be area-averaged data, "
                        "and already be filtered on relevant variable and area.")
    parser.add_argument('--season', required=True, choices=['year', 'djf', 'mam', 'jja', 'son'],
                        help="season of interest.")
    parser.add_argument('--period', required=True, type=int, nargs=2, default=[2036, 2065],
                        help="Period  of interest: start and end year, inclusive.")
    parser.add_argument('--relative', action='store_true', help="Variable is "
                        "relative (percentual) change")
    parser.add_argument('--csvfile', nargs='?', const=True, help="Write "
                        "(CMIP) percentile distributions to CSV. "
                        "This file is for the distribution calculated across runs. "
                        "Takes an optional filename argument; if not given, the "
                        "default filename is 'varchange_{season}.csv'.")
    parser.add_argument('--run-changes', nargs='?', const=True, help="Write the "
                        "calculated values and changes for each individual run to a "
                        "CSV file. Takes an optional filename argument; if not given, "
                        "the default filename is 'individual_run_percentiles_{season}.csv'.")
    parser.add_argument('--pr-changes', type=float, nargs='+', default=[4, 8],
                        help="One or more values for the change in precipitation per degree "
                        "temperature change, in percents. Default is [4, 8].")
    parser.add_argument('--reference-period', type=int, nargs=2,
                        default=list(REFERENCE_PERIOD),
                        help="Reference period in years (inclusive)")

    parser.add_argument('--no-matching', action='store_true', help="Perform no matching "
                        "and concatenation. This assumes the input files are already "
                        "concatenated, and contain both the --period and --reference-period "
                        "ranges. Useful for particular datasaets of interest, not for CMIP "
                        "data.")
    parser.add_argument('--match-by', choices=['model', 'ensemble'], default='ensemble',
                        help="Match future and historical runs by model (very generic) "
                        "or ensemble (very specific). Default is 'ensemble'.")
    parser.add_argument('--attributes-from', choices=['attributes', 'filename'], nargs='+',
                        default=['attributes', 'filename'],
                        help="Where to get the attribute information from. 'attributes' "
                        "will use the NetCDF attributes, 'filename' will attempt to deduce "
                        "them from the filename. Both can be given, in order of importance, "
                        "that is, the second then serves as a fallback. "
                        "Default is 'attributes filename'. ")
    parser.add_argument('--on-no-match', choices=['error', 'remove', 'randomrun', 'random'],
                        default='randomrun', help="What to do with a (future) experiment run that "
                        "has no matching run. 'error' raise an exception, 'remove' removes the "
                        "run. 'randomrun' picks a random history run with the same 'physics' and "
                        "'initialization' values (but a different 'realization' value), while "
                        "'random' picks a random history run from all ensembles for that model.")
    parser.add_argument('--historical-key', default=HISTORICAL_KEY, help="Attribute/filename value "
                        "to indicate a historical run.")

    args = parser.parse_args()
    args.paths = [pathlib.Path(filename) for filename in args.files]

    return args


def setup_logging(verbosity=0):
    """DUMMY DOCSTRING"""
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[max(0, min(verbosity, len(levels)))]
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt="%Y-%m-%dT%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def main():
    """DUMMY DOCSTRING"""
    args = parse_args()
    kcs.utils.logging.setup(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    paths = list(itertools.chain.from_iterable(atlist(path) for path in args.paths))
    dataset = read_data(paths, attributes_from=args.attributes_from)
    if not args.no_matching:
        dataset = kcs.utils.matching.match(
            dataset, match_by=args.match_by, on_no_match=args.on_no_match,
            historical_key=args.historical_key)

        dataset = concat_cubes(dataset, historical_key=args.historical_key)

    percentiles, run_changes = run(dataset, args.season, args.period,
                                   reference_period=args.reference_period, relative=args.relative)

    if args.csvfile:
        csvfile = f"varchange_{args.season}.csv" if args.csvfile is True else args.csvfile
        percentiles.to_csv(csvfile)
    if args.run_changes:
        csvfile = (f"individual_run_percentiles_{args.season}.csv" if args.run_changes is True
                   else args.run_changes)
        run_changes.to_csv(csvfile, index=False)

    logger.info("Done processing. Distribution of changes = %s", percentiles)


if __name__ == "__main__":
    main()
