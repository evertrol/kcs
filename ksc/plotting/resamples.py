#! /usr/bin/env python

import logging
import pathlib
import json
import pandas as pd
import h5py
import matplotlib.pyplot as plt
import plotting


logger = logging.getLogger(__name__)


def run(var, season, area, epoch, cmipcsv, scenarios, labels, resamples_file, options):
    oplot = options.get('plot', {})
    df = pd.read_csv(cmipcsv, index_col=0)

    figure = plt.figure(figsize=(6, 6))
    perc_ranges = [(10, 90), (25, 75)]
    plotting.plot_cmip5_percentiles(df, perc_ranges=perc_ranges, zorder=2)

    percs2 = {}
    with h5py.File(resamples_file, 'r') as h5file:
        group = h5file[str(epoch)]
        for key1, key2 in scenarios:
            try:
                ds = group[key1][key2][var][season]
            except KeyError:
                logger.warning("Scenario %s %s not found in resamples file", key1, key2)
                continue
            keys = ds['keys']
            percs2[f'{key1}_{key2}'] = {'data': {
                'mean': dict(zip(keys, ds['mean'][...])),
                'std': dict(zip(keys, ds['std'][...]))
            }}
            percs2[f'{key1}_{key2}mean'] = percs2[f'{key1}_{key2}']['data']['mean']


    plotting.plot_ecearth_percentiles(percs2, labels)

    plotting.plot_finish_percentiles(var, season, epoch, labels,
                                     text=oplot.get('text'),
                                     title=oplot.get('title'),
                                     xlabel=oplot.get('xlabel'),
                                     ylabel=oplot.get('ylabel'),
                                     ylimits=oplot.get('ylimits'))

    plt.tight_layout()
    filename = f"change-resampled-{var}-{area}-{season}-{epoch}.pdf"
    plt.savefig(filename, bbox_inches='tight')



# # # #  Use below as script / with the -m option # # # #

import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='count', default=0,
                        help="Verbosity level")
    parser.add_argument('var', help="Variable of interest (short name)")
    parser.add_argument('season', choices=['djf', 'mam', 'jja', 'son'],
                        help="Season of interest (three-letter abbrevation)")
    parser.add_argument('cmip_csv',
                        help="CSV file with CMIP distribution percentiles")
    parser.add_argument('resamples_h5file')
    parser.add_argument('--epoch', type=int, default=0, help="Epoch (year) of interest")
    parser.add_argument('--area', default='', help="Area of interest")
    parser.add_argument('--scenario', nargs=2, action='append',
                        help="Scenario(s) to plot; specify as two letters: `G/W L/H`. Option can be repeated.")
    parser.add_argument('--label', action='append', help="Label(s) for scenarios")
    parser.add_argument('--ecearth-dir', default=".", help="EC-EARTH base data directory")
    parser.add_argument('--plot-title',)
    parser.add_argument('--plot-text',)
    parser.add_argument('--plot-xlabel')
    parser.add_argument('--plot-ylabel')
    parser.add_argument('--plot-ylimits', nargs=2, type=float)

    args = parser.parse_args()

    if args.scenario is None:
        args.scenario = [['G', 'H'], ['G', 'L'], ['W', 'H'], ['W', 'L']]
    if args.label is None:
        args.label = [f'{key1}_{key2}' for key1, key2 in args.scenarios]
    assert len(args.label) == len(args.scenario)

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


def main():
    args = parse_args()
    setup_logging(args.verbose)
    options = {
        'labels': args.label,
        'plot': {
            'title': args.plot_title,
            'text': args.plot_text,
            'xlabel': args.plot_xlabel,
            'ylabel': args.plot_ylabel,
            'ylimits': args.plot_ylimits,
        },
    }
    run(args.var, args.season, args.area, args.epoch, args.cmip_csv, args.scenario, args.label,
        args.resamples_h5file, options)


if __name__ == '__main__':
    main()
