"""Usage:

# Calculate resamples for two scenarios, one epoch: G_L and G_H,
# 2050. Set the required precipitation changes to 4% per global
# temperature increase ('L') and 8% ('H')

$ python -m kcs.resample @ecearth-all-nlpoint.list --steering steering.csv \
    --ranges ranges.toml --penalties penalties.toml --relative pr \
    --precip-scenario L 4 --precip-scenario H 8 \
    --scenario G 2050 L --scenario G 2050 H -vv

# Calculate resamples for all scenarios and epochs (eight
# total). Naming follows the name and epoch columns from the steering
# table, plus the names for the precipitation scenarios.

$ python -m kcs.resample @ecearth-all-nlpoint.list --steering steering.csv \
    --ranges ranges.toml --penalties penalties.toml --relative pr \
    --precip-scenario L 4 --precip-scenario H 8

"""

import sys
import math
import argparse
import logging
import pathlib
import itertools
import json
import toml
import numpy as np
import pandas as pd
import h5py
import iris
from ..utils.argparse import parser as kcs_parser
from ..utils.logging import setup as setup_logging
from ..utils.atlist import atlist
from ..utils.attributes import get as get_attrs
from ..config import read_config, default_config
from . import run


STATS = ['mean', '5', '10', '25', '50', '75', '90', '95']


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
    dataset = get_attrs(
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


def save_indices_h5(filename, indices):
    """Save the (resampled) array of indices in a HDF5 file"""
    h5file = h5py.File(filename, 'a')
    for key, value in indices.items():
        name = "/".join(key)
        try:
            group = h5file[name]
        except KeyError:
            group = h5file.create_group(name)
        for k, v in value['meta'].items():  # pylint: disable=invalid-name
            # Transforms a dict to str so it can be saved in HDF 5
            group.attrs[k] = json.dumps(v) if isinstance(v, (dict, list)) else v
        if 'control' in group:
            del group['control']
        group.create_dataset('control', data=value['data']['control'])
        if 'future' in group:
            del group['future']
        group.create_dataset('future', data=value['data']['future'])


def save_resamples(filename, diffs, as_csv=True):
    """Save the resampled data, that is, the changes (differences) between
    epochs, to a HDF5 file

    With `as_csv` set to `True`, save each individual
    scenario-variable-season combination as a separate CSV file, named
    after the combination (and `resampled_` prepended). These CSV
    files can be used with kcs.change_perc.plot, with the --scenario
    option.

    Files are always overwritten if they exist.

    """

    h5file = h5py.File(filename, 'w')
    for key, value in diffs.items():
        keyname = "/".join(key)
        for var, value2 in value.items():
            for season, diff in value2.items():
                name = f"{keyname}/{var}/{season}"
                if name not in h5file:
                    h5file.create_group(name)
                group = h5file[name]

                # Remove existing datasets, to avoid problems
                # (we probably could overwrite them; this'll work just as easily)
                for k in {'diff', 'mean', 'std', 'keys'}:
                    if k in group:
                        del group[k]

                group.create_dataset('diff', data=diff.values)
                group.create_dataset('mean', data=diff.mean(axis=0))
                group.create_dataset('std', data=diff.std(axis=0))
                # pylint: disable=no-member
                dataset = group.create_dataset('keys', (len(STATS),), dtype=h5py.string_dtype())
                dataset[:] = STATS

                assert len(STATS) == len(diff.mean(axis=0))

                if as_csv:
                    csvfile = "_".join(key)
                    csvfile = f"resampled_{csvfile}_{var}_{season}.csv"
                    diff.to_csv(csvfile, index=False)


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
    parser = argparse.ArgumentParser(parents=[kcs_parser],
                                     conflict_handler='resolve')

    parser.add_argument('files', nargs='+', help="NetCDF4 files to resample.")
    parser.add_argument('--steering', required=True, help="CSV file with steering information")
    parser.add_argument('--ranges',
                        help="TOML file with lower and upper percentile bounds for all scenarios, "
                        "for future and control periods.")

    parser.add_argument('--scenario', nargs=3, action='append', help="Scenario(s) to "
                        "resample data for. Should be three values: `G/W 2050/2085 H/L`. "
                        "The option can be repeated. The default is run all (eight) scenarios.")
    parser.add_argument('--precip-scenario', nargs=2, action='append',
                        help="Label and percentage of winter precipitation increase per degree. "
                        "For example, L 4, or H 8. The label should correspond to the third value "
                        "in the --scenarion option, if given.")
    parser.add_argument('--nstep1', type=int,
                        help="number of S1 resamples to keep")
    parser.add_argument('--nstep3', type=int,
                        help="number of S3 resamples to keep")
    parser.add_argument('--nsample', type=int,
                        help="Monte Carlo sampling number")
    parser.add_argument('--reference-period', nargs=2, type=int,
                        help="Reference period given by start and end year (inclusive)")
    parser.add_argument('--nsections', type=int,
                        help="Number of sections to slice the periods into. This is done by "
                        "integer division, and any remains are discarded. Thus, a period of "
                        "30 years (inclusive) and `--nsections=6` (the default) results in six "
                        "five-year periods")
    parser.add_argument('--penalties',
                        help="TOML file with penalties for multiple occurences.")
    parser.add_argument('--relative', nargs='+', default=['pr'], help="List of short variable "
                        "names for which the relative (percentual) change is to be calculated.")

    parser.add_argument('--indices-out', default="indices.h5", help="HDF 5 output file "
                        "for the indices.")
    parser.add_argument('--resamples-out', default="resamples.h5", help="HDF 5 output file "
                        "for the resampled data.")

    parser.add_argument('-N', '--nproc', type=int, help="Number of simultaneous processes.")

    args = parser.parse_args()
    setup_logging(args.verbosity)
    # Read and set defaults
    read_config(args.config)

    if args.reference_period is None:
        args.reference_period = default_config['data']['extra']['control_period']
    if args.nproc is None:
        args.nproc = default_config['resampling']['nproc']
    if args.nstep1 is None:
        args.nstep1 = default_config['resampling']['nstep1']
    if args.nstep3 is None:
        args.nstep3 = default_config['resampling']['nstep3']
    if args.nsample is None:
        args.nsample = default_config['resampling']['nsample']
    if args.nsections is None:
        args.nsections = default_config['resampling']['nsections']

    args.paths = [pathlib.Path(filename) for filename in args.files]

    args.pr_scenarios = {}
    for label, perc in args.precip_scenario:
        args.pr_scenarios[label] = float(perc)

    if args.ranges:
        filename = args.ranges
    else:
        filename = default_config['resampling']['step2_conditions']
        with open(filename) as fh:  # pylint: disable=invalid-name
            args.ranges = toml.load(fh)

    if args.penalties:
        with open(args.penalties) as fh:  # pylint: disable=invalid-name
            penalties = toml.load(fh)
            penalties = penalties['penalties']
    else:
        penalties = default_config['resampling']['penalties']
    penalties = {int(key): value for key, value in penalties.items()}
    # Fill up the penalty dict
    for i in range(max(penalties.keys())+1, args.nstep3+1):
        penalties[i] = math.inf
    args.penalties = penalties

    return args



def main():
    """DUMMY DOCSTRING"""
    args = parse_args()
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    paths = list(itertools.chain.from_iterable(atlist(path) for path in args.paths))
    dataset = read_data(paths)

    steering_table = read_steering_target(args.steering, args.pr_scenarios, args.scenario)

    indices, diffs = run(dataset, steering_table, args.ranges, args.penalties,
                         args.nstep1, args.nstep3, args.nsample, args.nsections,
                         args.reference_period, relative=args.relative, nproc=args.nproc)

    save_indices_h5(args.indices_out, indices)
    save_resamples(args.resamples_out, diffs)


if __name__ == '__main__':
    main()
