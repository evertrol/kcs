"""Usage example:

# Calculate resamples for two scenarios, one epoch: G_L and G_H,
# 2050. Set the required precipitation changes to 4% per global
# temperature increase ('L') and 8% ('H')

$ python -m kcs.resample @ecearth-all-nlpoint-short.list --steering steering.csv \
    --ranges ranges.toml --precip-scenario L 4 --precip-scenario H 8 --relative pr \
    --scenario G 2050 L --scenario G 2050 H -vv

# Calculate resamples for all scenarios and epochs (eight
# total). Naming follows the name and epoch columns from the steering
# table, plus the names for the precipitation scenarios.

$ python -m kcs.resample @ecearth-all-nlpoint-short.list --steering steering.csv \
    --ranges ranges.toml --precip-scenario L 4 --precip-scenario H 8 --relative pr

"""

import sys
import math
import argparse
import logging
import pathlib
import itertools
import toml
import numpy as np
import pandas as pd
import iris
import kcs.utils.argparse
import kcs.utils.logging
from kcs.utils.atlist import atlist
import kcs.utils.attributes
from . import run


NPROC = 1
CONTROL_PERIOD = (1981, 2010)
N1 = 1000
N2 = 50
N3 = 8
NSAMPLE = 10_000
NSECTIONS = 6


logger = logging.getLogger('kcs.resample')   # pylint: disable=invalid-name


def read_data(paths, attributes_from=('attributes', 'filename'),
              attributes=None, filename_pattern=None):
    """Read the dataset from nc files and get attribute information

    Returns a dataset in the form of a Pandas DataFrame.

    """
    cubes = [iris.load_cube(str(path)) for path in paths]

    if attributes_from is False:
        return pd.DataFrame({'cube': cubes, 'path': paths})

    # Get the attributes, and create a dataframe with cubes & attributes
    dataset = kcs.utils.attributes.get(
        cubes, paths, info_from=attributes_from,
        attributes=attributes, filename_pattern=filename_pattern)

    return dataset


def read_steering_target(filename, scenarios, scenario_selection=None):
    """Read and amend the steering table

    Reads the steering table from CSV, adds the precip scenarios, and
    selects any given scenarios.

    """

    table = pd.read_csv(filename)
    table = table.rename(columns={'target year': 'epoch', 'resampling period': 'period',
                                  'name': 'scenario'})
    table['epoch'] = table['epoch'].astype(str)
    # Convert the "(start, stop)" string to a proper tuple of integer years.
    table['period'] = table['period'].apply(lambda x: tuple(map(int, x[1:-1].split(','))))

    # Add the precip scenarios.
    table2 = pd.DataFrame({'subscenario': list(scenarios.keys()),
                           'precip_change_per_t': list(scenarios.values())})
    table['key'] = table2['key'] = 0
    table = pd.merge(table, table2, on='key', how='outer').drop(columns='key')
    table['precip_change'] = table['model_delta_t'] * table['precip_change_per_t']

    # If there is any scenarion selection, remove the not-selected ones
    if scenario_selection:
        selection = np.zeros(len(table), dtype=np.bool)
        for scen, epoch, subscen in scenario_selection:
            selection = selection | ((table['scenario'] == scen) &
                                     (table['subscenario'] == subscen) &
                                     (table['epoch'] == epoch))
        table = table[selection]

    logger.debug("Scenario's and target precipiation for S1: %s",
                 table[['epoch', 'scenario', 'subscenario', 'period', 'model_delta_t',
                        'precip_change_per_t', 'precip_change']].sort_values(
                            ['subscenario', 'scenario', 'epoch'], ascending=[True, False, False]))

    return table


def str2bool(string):
    """Convert a command line string value to a Python boolean"""
    if isinstance(string, bool):
        return string
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if string.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean string expected.')


def parse_args():
    """Parse the command line arguments"""
    parser = argparse.ArgumentParser(parents=[kcs.utils.argparse.parser],
                                     conflict_handler='resolve')

    parser.add_argument('files', nargs='+', help="NetCDF4 files to resample.")
    parser.add_argument('--steering', required=True, help="CSV file with steering information")
    parser.add_argument('--ranges', required=True,
                        help="TOML file with lower and upper percentile bounds for all scenarios, "
                        "for future and control periods.")

    parser.add_argument('--scenario', nargs=3, action='append', help="Scenario(s) to "
                        "resample data for. Should be three values: `G/W 2050/2085 H/L`. "
                        "The option can be repeated. The default is run all (eight) scenarios.")
    parser.add_argument('--precip-scenario', nargs=2, action='append',
                        help="Label and percentage of winter precipitation increase per degree. "
                        "For example, L 4, or H 8. The label should correspond to the third value "
                        "in the --scenarion option, if given.")
    parser.add_argument('--n1', type=int, default=N1,
                        help="number of S1 resamples to keep")
    parser.add_argument('--n3', type=int, default=N3,
                        help="number of S3 resamples to keep")
    parser.add_argument('--nsample', type=int, default=NSAMPLE,
                        help="Monte Carlo sampling number")
    parser.add_argument('--control-period', nargs=2, type=int, default=CONTROL_PERIOD,
                        help="Control period given by start and end year (inclusive)")
    parser.add_argument('--nsections', type=int, default=NSECTIONS,
                        help="Number of sections to slice the periods into. This is done by "
                        "integer division, and any remains are discarded. Thus, a period of "
                        "30 years (inclusive) and `--nsections=6` (the default) results in six "
                        "five-year periods")
    parser.add_argument('--penalties', help="TOML file with penalties for multiple occurences. "
                        "See example file for its format.")
    parser.add_argument('--relative', nargs='+', default=['pr'], help="List of short variable "
                        "names for which the relative (percentual) change is to be calculated.")

    parser.add_argument('-N', '--nproc', type=int, default=NPROC,
                        help="Number of simultaneous processes")

    args = parser.parse_args()

    args.paths = [pathlib.Path(filename) for filename in args.files]

    args.pr_scenarios = {}
    for label, perc in args.precip_scenario:
        args.pr_scenarios[label] = float(perc)

    with open(args.penalties) as fh:  # pylint: disable=invalid-name
        penalties = toml.load(fh)
    penalties = {int(key): value for key, value in penalties['penalties'].items()}
    # Fill up the penalty dict
    for i in range(max(penalties.keys())+1, args.n3+1):
        penalties[i] = math.inf
    args.penalties = penalties

    return args



def main():
    """DUMMY DOCSTRING"""
    args = parse_args()
    kcs.utils.logging.setup(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    paths = list(itertools.chain.from_iterable(atlist(path) for path in args.paths))
    dataset = read_data(paths)

    steering_table = read_steering_target(args.steering, args.pr_scenarios, args.scenario)

    with open(args.ranges) as fh:  # pylint: disable=invalid-name
        ranges = toml.load(fh)

    run(dataset, steering_table, ranges, args.penalties,
        args.n1, args.n3, args.nsample, args.nsections, args.control_period,
        relative=args.relative, nproc=args.nproc)


if __name__ == '__main__':
    main()
