#! /usr/bin/env python

import sys
import math
import json
import argparse
import logging
from pprint import pformat
import numpy as np
import pandas as pd
import h5py
import ecearth
import resampling


NPROC = 1
CONTROL_PERIOD = (1981, 2010)
N1 = 1000
N2 = 50
N3 = 8
NSAMPLE = 10_000
NSECTIONS = 6


logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('steering', help="CSV file with steering information")
    parser.add_argument('ranges', help="JSON file with lower and upper percentile bounds "
                        "for all scenarios")

    parser.add_argument('--scenario', nargs=3, action='append', help="Scenario(s) to "
                        "resample data for. Should be three values: `2050/2085 G/W H/L`. "
                        "The option can be repeated. The default is all (eight) scenarios.")
    parser.add_argument('--precip-scenario', nargs=2, action='append',
                        help="Label and percentage of winter precipitation increase per degree")
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
    parser.add_argument('--ensemble-penalty', type=float, nargs=2, action='append',
                        default=[[3, 1], [4, 5]], help="Penalty for multiple occurrence of "
                        "same ensemble member in a resample. Two values: number of occurences "
                        "and penalty. The option can be used multiple times. "
                        "If a number of occurences below the lowest given number is not given, "
                        "its penalty is assumed to be 0. Penalties for number of occurrences "
                        "above a given number, are assumed to be infinite.")

    parser.add_argument('--variable', nargs='+', default=['tas', 'pr'],
                        help="Short name for variable(s) of interest (as used in the file names). "
                        "Multiple values allowed. E.g., `--variable tas pr`.")
    parser.add_argument('--relative', nargs='+', type=str2bool, default=[False, True],
                        help="Indicate if the --variables should be computed relatively or not")
    parser.add_argument('--area', default='nlpoint', help="Name of area of interest.")
    parser.add_argument('--datadir', default='.', help="EC-EARTH main data directory.")
    parser.add_argument('--maxdata', type=int, help="Maximum number of data EC-EARTH "
                        "data sets to read")

    parser.add_argument('-N', '--nproc', type=int, default=NPROC,
                        help="Number of simultaneous processes")
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbosity level")

    args = parser.parse_args()

    args.pr_scenarios = {}
    for label, perc in args.precip_scenario:
        args.pr_scenarios[label] = float(perc)

    penalties = dict(args.ensemble_penalty)
    lowest = True
    for i in range(1, args.n3+1):
        if i in penalties:
            lowest = False
            continue
        penalties[i] = 0 if lowest else math.inf
    args.penalties = penalties

    assert len(args.variable) == len(args.relative)
    relative = {var: value for var, value in zip(args.variable, args.relative)}
    args.meta = {
        'vars': args.variable,
        'relative': relative,
        'area': args.area,
        'dir': args.datadir,
        'nmax': args.maxdata,
    }

    return args


def setup_logging(verbosity=0):
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[max(0, min(verbosity, len(levels)))]
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt="%Y-%m-%dT%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Set the import modules to the same logging level & handler
    for module in {'resampling', 'ecearth'}:
        l = logging.getLogger(module)
        l.setLevel(level)
        l.addHandler(handler)


def verify(ranges, penalties):
    #logger.debug("Percentile ranges: %s", pformat(ranges))
    #for var in ranges:
    #    for season in ranges[var]:
    #        for period, range_ in ranges[var][season].items():
    #            name = f"{var}:{season}:{period}"
    #            assert 0 <= range_[0] <= 100, \
    #                f"lower bound of range for {name} is outside 0 -- 100"
    #            assert 0 <= range_[1] <= 100, \
    #                f"upper bound of range for {name} is outside 0 -- 100"
    #            assert range_[0] < range_[1], \
    #                f"lower bound of range for {name} is not below upper bound"

    logger.debug("Penalties: %s", penalties)


def read_steering_target(filename, scenarios, scenario_selection=None):
    df = pd.read_csv(filename)
    df = df.rename(columns={'target year': 'epoch', 'resampling period': 'period'})
    df['epoch'] = df['epoch'].astype(str)
    df['period'] = df['period'].apply(lambda x: tuple(map(float, x[1:-1].split(','))))

    df2 = pd.DataFrame({'subscenario': list(scenarios.keys()),
                        'precip_change_per_t': list(scenarios.values())})
    df['key'] = df2['key'] = 0
    df = pd.merge(df, df2, on='key', how='outer').drop(columns='key')
    df['precip_change'] = df['steering delta-t'] * df['precip_change_per_t']

    if scenario_selection:
        selection = np.zeros(len(df), dtype=np.bool)
        for epoch, scen, subscen in scenario_selection:
            selection = selection | ((df['scenario'] == scen) & (df['subscenario'] == subscen) & (df['epoch'] == epoch))
        df = df[selection]

    logger.debug("Scenario's and target precipiation for S1: %s",
                 df[['epoch', 'scenario', 'subscenario', 'period', 'steering delta-t',
                     'precip_change_per_t', 'precip_change']].sort_values(
                         ['subscenario', 'scenario', 'epoch'], ascending=[True, False, False]))

    return df


def read_ranges(filename):
    with open(filename) as fh:
        data = json.load(fh)
    #from pprint import pprint; pprint(data)
    scenarios = list(data.keys())
    subscenarios = list(data[scenarios[0]].keys())
    epochs = list(data[scenarios[0]][subscenarios[0]].keys())
    index, _ = pd.MultiIndex.from_product([scenarios, subscenarios, epochs]).sortlevel()
    index.names = ['scenario', 'subscenario', 'epoch']
    keys = []
    for key in index:
        ranges = data[key[0]][key[1]][key[2]]
        for r in ranges:
            for var in r:
                for season in r[var]:
                    keys.append((var, season))
    keys = set(keys)
    colindex, _ = pd.MultiIndex.from_product([keys, ['control', 'future'], ['start', 'stop']]).sortlevel()

    df = pd.DataFrame(index=index, columns=colindex)
    startstop = {'start': 0, 'stop': 1}
    for key in df.index:
        ranges = data[key[0]][key[1]][key[2]]
        for r in ranges:
            for col in df.columns:
                r2 = r.get(col[0][0], {}).get(col[0][1])
                if not r2:
                    continue
                i = startstop.get(col[2])
                df.loc[key, col] = r2[col[1]][i]


    logger.debug("Percentile range table: %s",
                 df.sort_values(['subscenario', 'scenario', 'epoch'],
                                ascending=[True, False, False]))
    df.to_csv('blah.csv')

    return df, data


def run(steering_csv, ranges_csv, pr_scenarios, penalties, meta,
        n1=N1, n3=N3, nsample=NSAMPLE,
        nsections=NSECTIONS, control_period=CONTROL_PERIOD,
        scenarios=None, nproc=NPROC):

    df = read_steering_target(steering_csv, pr_scenarios, scenarios)

    ranges, ranges_json = read_ranges(ranges_csv)

    verify(ranges, penalties)

    columns = np.arange(nsections)

    all_indices = {}
    data = {}
    for _, row in df.iterrows():
        period = tuple(map(int, row['period']))
        scenario = row['scenario']
        subscenario = row['subscenario']
        epoch = row['epoch']
        basekey = (str(epoch), scenario, subscenario)
        precip_change = row['precip_change']
        logger.debug("Processing %s_%s - %s (%s)", scenario, subscenario, epoch, period)
        data[basekey] = {}
        ndata = set()
        for var in meta['vars']:
            logger.debug("Extracting EC-EARTH data, variable %s", var)
            cubes = ecearth.load_datasets(var, meta['area'], meta['dir'], meta['nmax'])
            data[basekey][var] = ecearth.prepare_data(cubes, period, control_period, nsections)
            season = list(data[basekey][var].keys())[0]
            segment = data[basekey][var][season]['future']
            ndata.add(len(segment))

        assert len(ndata) == 1, "Datasets are not the same length for different variables"
        ndata = ndata.pop()

        indices = resampling.create_indices(ndata, nsections)

        means = resampling.calc_means(data[basekey])

        logger.debug("Calculating S1")
        logger.debug("Precipitation change: %.1f", precip_change)
        s1_indices = resampling.calculate_s1(means, indices, precip_change, nproc=nproc)
        s1_indices['control'] = s1_indices['control'][:n1]
        s1_indices['future'] = s1_indices['future'][:n1]

        print('========')
        controlmean = []
        for cube in data[basekey]['pr']['djf']['control'][...].flatten():
            controlmean.append(cube.data)
        controlmean = np.array(controlmean).mean()
        print('Control mean =', controlmean)
        aves = []
        for i in s1_indices['future'][:5]:
            for cube in data[basekey]['pr']['djf']['future'][i, columns].flatten():
                aves.append(cube.data)
            ave = np.array(aves).mean()
            print(i, 100 * (ave - controlmean) / controlmean)
        print('=========')

        r = ranges_json[scenario][subscenario][str(epoch)]
        s2_indices = resampling.calculate_s2(
            means, s1_indices, ranges=r)
        logger.debug("The S2 subset has %d & %d indices for the control & future periods, resp.",
                     len(s2_indices['control']), len(s2_indices['future']))
        print('~~~~~~~~~~')
        print('Control mean =', controlmean)
        aves = []
        for i in s2_indices['future'][:5]:
            for cube in data[basekey]['pr']['djf']['future'][i, columns].flatten():
                aves.append(cube.data)
            ave = np.array(aves).mean()
            print(i, 100 * (ave - controlmean) / controlmean)
        print('~~~~~~~~~')
        s3_indices = resampling.calculate_s3(means, s2_indices, penalties, n3=n3, nsample=nsample)


        attrs = {
            'scenario': scenario, 'subscenario': subscenario, 'epoch': epoch,
            'period': period, 'control-period': control_period,
            'area': meta['area'],
            'ranges': r, 'winter-precip-change': precip_change}
        all_indices[basekey] = {'data': s3_indices, 'meta': attrs}
        #s2_indices = {key: indices[:8, ...] for key, indices in s2_indices.items()}
        #all_indices[basekey] = {'data': s2_indices, 'meta': attrs}

        filename = f"{scenario}_{subscenario}-{epoch}.json"
        #resampling.save_indices(filename, s2_indices, meta=attrs)

        cms = [controlmean]
        print(scenario, subscenario, epoch, period)
        print('-------')
        print('Control mean =', controlmean)
        aves = []
        for i in s3_indices['future']:
            for cube in data[basekey]['pr']['djf']['future'][i, columns].flatten():
                aves.append(cube.data)
            ave = np.array(aves).mean()
            print(i, 100 * (ave - controlmean) / controlmean)
        print('-------')
        print(all_indices[basekey]['data']['future'])
        print('-------')
        ave = []
        aves = []
        controlmean = [cube.data.mean() for cube in data[basekey]['pr']['djf']['control'][s3_indices['control'], columns].flatten()]
        #print(controlmean)
        cms.append(np.mean(controlmean))
        for i in s3_indices['control']:
            for cube in data[basekey]['pr']['djf']['control'][i, columns]:
                aves.append(cube.data)
        controlmean = np.mean(aves)
        cms.append(controlmean)
        print('control means =', cms)
        aves = []
        for i in s3_indices['future']:
            for cube in data[basekey]['pr']['djf']['future'][i, columns]:
                aves.append(cube.data)
            ave = np.array(aves).mean()
            print(i, 100 * (ave - controlmean) / controlmean)
        print('-------')

    for key, indices in all_indices.items():
        print(key)
        for i, j in zip(indices['data']['control'], indices['data']['future']):
            cmean = np.mean([cube.data for cube in data[key]['pr']['djf']['control'][i, columns]])
            fmean = np.mean([cube.data for cube in data[key]['pr']['djf']['future'][j, columns]])
            print(i, j, (fmean - cmean) / cmean * 100)

    diffs = resampling.resample(all_indices, data, meta['vars'], meta['relative'])

    resampling.save_indices_h5(filename, all_indices)
    resampling.save_resamples("resamples.h5", diffs)



#            for season, diff in diffs.items():
#                filename = f"{variable}-{meta['scenario']}-{meta['epoch']}-{season}.json"
#                print(variable, meta['scenario'], meta['epoch'], season)
#                print(diff.mean(axis=0))
#                print(diff.std(axis=0))
#
#                with open(filename, 'w') as fh:
#                    json.dump({'data': diff.mean(axis=0).to_dict(),
#                               'std': diff.std(axis=0).to_dict()},
#                              fh)

def main():
    args = parse_args()
    setup_logging(args.verbose)
    logger.debug("Running ./%s", " ".join(sys.argv))
    logger.debug("Parsed arguments: %s", pformat(args))
    run(args.steering, args.ranges, args.pr_scenarios, args.penalties, args.meta,
        args.n1, args.n3, args.nsample, args.nsections, args.control_period,
        scenarios=args.scenario,
        nproc=args.nproc)


if __name__ == '__main__':
    main()
