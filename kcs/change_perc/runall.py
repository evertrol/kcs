"""Calculate and plot percentile distributions for all input data,
scenarios from the steering table, for a given set of seasons and
variables

Example usage:

$ python -m kcs.change_perc.runall @cmip-all-nlpoint-averaged.list --season djf jja  \
    --steering steering.csv --runs @ecearth-all-nlpoint-averaged.list --relative pr \
    --no-matching -v --plottype pdf --write-csv

Note the above runs all single core. It is possible to speed things up
by using e.g. four cores, by simply splitting the above up into four
individual runs, separated by variable and season:

$ python -m kcs.change_perc.runall @cmip-tas-nlpoint-averaged.list --season djf  \
    --steering steering.csv --runs @ecearth-tas-nlpoint-averaged.list --relative pr \
    --no-matching -v --plottype pdf --write-csv  &
$ python -m kcs.change_perc.runall @cmip-tas-nlpoint-averaged.list --season jja  \
    --steering steering.csv --runs @ecearth-tas-nlpoint-averaged.list --relative pr \
    --no-matching -v --plottype pdf --write-csv  &
$ python -m kcs.change_perc.runall @cmip-pr-nlpoint-averaged.list --season djf  \
    --steering steering.csv --runs @ecearth-pr-nlpoint-averaged.list --relative pr \
    --no-matching -v --plottype pdf --write-csv  &
$ python -m kcs.change_perc.runall @cmip-pr-nlpoint-averaged.list --season jja  \
    --steering steering.csv --runs @ecearth-pr-nlpoint-averaged.list --relative pr \
    --no-matching -v --plottype pdf --write-csv  &

Provided the list files exists, of course (the `--relative pr` is left
even for the tas runs, just to keep things simple).

One could even cut the steering.csv file in two parts, a
steering2050.csv and a steering2085.csv file, each have just two lines
with the respective epochs. A simple nested bash loop would then let
things run on eight cores:

for epoch in 2050 2085
do
    fname="steering${epoch}.csv"
    for var in tas pr
    do
        listname1="@cmip-${var}-nlpoint-averaged.list"
        listname2="@ecearth-${var}-nlpoint-averaged.list"
        for season in djf jja
        do
            python -m kcs.change_perc.runall $listname1 --season $season  \
              --steering $fname --runs $listname2 --relative pr --no-matching -v \
              --plottype pdf --write-csv  &
        done
    done
done

This may be I/O limited, and an eight times increase in running time
is not to be expected, but it should certainly be faster than a single
core run.

Note that parallel processing is not implemented in the script(s)
itself: there is a problem trying to run (Iris) data processing in a
Python multiprocessing environment. Potentially, the underlying Dask
engine that Iris uses should be able to speed up things itself, but it
is unclear how.

"""

import sys
import argparse
import logging
import pathlib
import itertools
import matplotlib.pyplot as plt
import pandas as pd
import iris
import iris.cube
from iris.util import unify_time_units
try:
    from iris.util import equalise_attributes
except ImportError:   # Iris 2
    from iris.experimental.equalise_cubes import equalise_attributes
from ..config import default_config, read_config
from ..utils.argparse import parser as kcs_parser
from ..utils.logging import setup as setup_logging
from ..utils.attributes import get as get_attrs
from ..utils.matching import match
from ..utils.atlist import atlist
from . import run as calculate
from . import plot


VARNAME = {
    'pr': "precip",
    'tas': "t2m",
}
YTITLE = {
    'pr': "Change (%)",
    'tas': r"Change (${}^{\circ}$C)",
}

logger = logging.getLogger('runall')  # pylint: disable=invalid-name


def read_data(paths, attributes_from=('attributes', 'filename'),
              attributes=None, filename_pattern=None):
    """Read the dataset from nc files and get attribute information

    Returns a dataset in the form of a Pandas DataFrame.

    """
    cubes = [iris.load_cube(str(path)) for path in paths]

    # Get the attributes, and create a dataframe with cubes & attributes
    dataset = get_attrs(
        cubes, paths, info_from=attributes_from,
        attributes=attributes, filename_pattern=filename_pattern)

    return dataset


def concat_cubes(dataset, historical_key=None):
    """Concatenate cubes into a dataset spanning the full time frame"""

    if historical_key is None:
        historical_key = default_config['data']['attributes']['historical_experiment']

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


def run(dataset, runs, seasons, steering, relative=None, reference_span=30,
        reference_period=None, plottype='png', writecsv=False):
    """DUMMY DOCSTRING"""

    if reference_period is None:
        reference_period = default_config['data']['cmip']['control_period']

    if relative is None:
        relative = []

    for epoch, steering_group in steering.groupby('epoch'):
        period = epoch - reference_span // 2 + 1, epoch + reference_span // 2
        for var, vargroup in dataset.groupby('var'):
            rel = var in relative
            for season in seasons:
                logger.info("Calculating CMIP percentiles for %s, season %s, epoch %s",
                            var, season, epoch)
                perc_distr, perc = calculate(vargroup.copy(), season, period, relative=rel,
                                             reference_period=reference_period)
                if writecsv:
                    filename = f"{var}_{epoch}_{season}_perc_distr.csv"
                    perc_distr.to_csv(filename, index=True)
                    filename = f"{var}_{epoch}_{season}_perc.csv"
                    perc.to_csv(filename, index=True)

                scenarios = {}
                for _, row in steering_group.iterrows():
                    data = runs.loc[runs['var'] == var, :].copy()
                    name = row['name'].rstrip('0123456789')  # remove the year part
                    period = [int(year) for year in row['period'].strip('()').split(',')]
                    logger.info("Calculating runs percentiles for %s, season %s, epoch %s, "
                                "scenario %s", var, season, epoch, name)
                    _, scenarios[name] = calculate(data, season, period, relative=rel,
                                                   reference_period=reference_period)
                    if writecsv:
                        filename = f"{var}_{epoch}_{season}_{name}_perc.csv"
                        scenarios[name].to_csv(filename, index=False)

                labels = {
                    'title': '',
                    'text': f"{VARNAME[var]}, {season.upper()}",
                    'y': YTITLE[var],
                    'x': '',
                    'epoch': epoch,
                }
                columns = ['mean', '5', '10', '50', '90', '95']
                xlabels = ['ave', 'P05', 'P10', 'P50', 'P90', 'P95']
                logger.info("Creating plot for variable %s, season %s, epoch %s")
                plot.run(perc_distr, labels, limits=None, columns=columns, xlabels=xlabels,
                         scenarios=scenarios)
                plt.tight_layout()
                filename = f"{var}_{epoch}_{season}.{plottype.lower()}"
                plt.savefig(filename, bbox_inches='tight')


def parse_args():
    """DUMMY DOCSTRING"""
    parser = argparse.ArgumentParser(parents=[kcs_parser],
                                     conflict_handler='resolve')
    parser.add_argument('files', nargs='+', help="Input data. This should be area-averaged data, "
                        "and already be filtered on relevant area. Variables are determined from "
                        "attributes or filenames, according to --attributes-from.")
    parser.add_argument('--season', nargs='+', required=True,
                        choices=['year', 'djf', 'mam', 'jja', 'son'],
                        help="One or more seasons of interest.")
    parser.add_argument('--steering', required=True, help="Steering CSV table.")
    parser.add_argument('--runs', nargs='+', required=True, help="Special individual runs, "
                        "all variables. ")
    parser.add_argument('--relative', nargs='+', default=['pr'], help="List of short variable "
                        "names for which the relative (percentual) change is to be calculated.")
    parser.add_argument('--reference-period', type=int, nargs=2,
                        help="Reference period (used both on CMIP data and special runs).")
    parser.add_argument('--reference-span', type=int, default=30,
                        help="Timespan around epoch of interest. Default is 30 years.")

    parser.add_argument('--plottype', default='png', help="Type of plot, e.g., 'png' (the default) "
                        "or 'pdf'.")
    parser.add_argument('--write-csv', action='store_true', help="Write CSV files for CMIP "
                        "percentile distributions, and individual runs percentiles. Files are "
                        "automatically named after variable, epoch, season and scenario.")
    parser.add_argument('--no-matching', action='store_true', help="Perform no matching "
                        "and concatenation for the --runs data. This assumes the input files "
                        "are already concatenated, and contain both the period and "
                        "--reference-period ranges.")
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
    parser.add_argument('--historical-key', help="Attribute/filename value "
                        "to indicate a historical run.")

    args = parser.parse_args()
    setup_logging(args.verbosity)
    read_config(args.config)

    args.paths = [pathlib.Path(filename) for filename in args.files]
    args.runs = [pathlib.Path(filename) for filename in args.runs]
    if args.reference_period is None:
        args.reference_period = default_config['data']['extra']['control_period']
    if args.historical_key is None:
        args.historical_key = default_config['data']['attributes']['historical_experiment']

    return args


def main():
    """DUMMY DOCSTRING"""

    args = parse_args()
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    paths = list(itertools.chain.from_iterable(atlist(path) for path in args.paths))
    dataset = read_data(paths, attributes_from=args.attributes_from)
    # Handle matching concatenation of cubes separately for each
    # variable; otherwise, matches will be across a variable
    dslist = []
    for _, group in dataset.groupby('var'):
        tmpds = group.copy()
        tmpds = match(
            tmpds, match_by=args.match_by, on_no_match=args.on_no_match,
            historical_key=args.historical_key)
        tmpds = concat_cubes(tmpds, historical_key=args.historical_key)
        dslist.append(tmpds)
    dataset = pd.concat(dslist)
    logger.debug("CMIP dataset: %s", dataset)

    paths = list(itertools.chain.from_iterable(atlist(path) for path in args.runs))
    runs = read_data(paths, attributes_from=args.attributes_from)
    if not args.no_matching:
        dslist = []
        for _, group in runs.groupby('var'):
            tmpds = group.copy()
            tmpds = match(
                tmpds, match_by=args.match_by, on_no_match=args.on_no_match,
                historical_key=args.historical_key)
            tmpds = concat_cubes(tmpds, historical_key=args.historical_key)
            dslist.append(tmpds)
        runs = dataset.pd.concat(dslist)
    logger.debug("Individual runs: %s", runs)

    steering = pd.read_csv(args.steering, index_col=False)
    logger.debug("Steeering table: %s", steering)
    run(dataset, runs, args.season, steering,
        reference_period=args.reference_period, relative=args.relative,
        plottype=args.plottype, writecsv=args.write_csv)

    logger.info("Done processing.")


if __name__ == "__main__":
    main()
