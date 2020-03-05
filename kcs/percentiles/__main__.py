from pathlib import Path
import re
import argparse
from collections import defaultdict
from enum import IntEnum
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd
import iris
from iris.cube import CubeList
from iris.util import unify_time_units
try:
    from iris.util import equalise_attributes
except ImportError:
    from iris.experimental.equalise_cubes import equalise_attributes
import kcs.plotting


pd.set_option('display.max_rows', None)

REFPERIOD = [1981, 2010]
class RCP(IntEnum):
    HISTORICAL = -1
    RCP26 = 26
    RCP45 = 45
    RCP6 = 60
    RCP85 = 85
STATS = ['mean', '5', '10', '25', '50', '75', '90', '95']

logger = logging.getLogger(__name__)


def plot(df, var, area, season, year, ecpercs=None,
         title=None, xlabel=None, ylabel=None, ylimits=None):
    figure = plt.figure(figsize=(6, 6))
    plt.rcParams.update({
        'legend.fontsize': 14,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'font.size': 14,
    })

    return figure

    if colors is None:
        colors = COLORS

    x = [0.5, 1.5]
    #y1 = [df.loc['mean', '5']] * 2
    #y2 = [df.loc['mean', '95']] * 2
    #plt.fill_between(x, y1, y2, color='#bbbbbb', alpha=0.8, zorder=2)
    y1 = [df.loc['mean', '10']] * 2
    y2 = [df.loc['mean', '90']] * 2
    plt.fill_between(x, y1, y2, color='#888888', alpha=0.4, zorder=3)
    y1 = [df.loc['mean', '25']] * 2
    y2 = [df.loc['mean', '75']] * 2
    plt.fill_between(x, y1, y2, color='#333333', alpha=0.4, zorder=4)

    ps = ['5', '10', '50', '90', '95']
    x = [2, 3, 4, 5, 6]
    #y1 = [df.loc[p, '5'] for p in ps]
    #y2 = [df.loc[p, '95'] for p in ps]
    #plt.fill_between(x, y1, y2, color='#bbbbbb', alpha=0.8, zorder=2)
    y1 = [df.loc[p, '10'] for p in ps]
    y2 = [df.loc[p, '90'] for p in ps]
    plt.fill_between(x, y1, y2, color='#888888', alpha=0.4, zorder=3)
    y1 = [df.loc[p, '25'] for p in ps]
    y2 = [df.loc[p, '75'] for p in ps]
    plt.fill_between(x, y1, y2, color='#333333', alpha=0.4, zorder=4)

    colors = {'G': '#aaee88', 'W': '#225511'}
    if ecpercs is not None:
        x = [1, 2, 3, 4, 5, 6]
        y = ['mean', '5', '10', '50', '90', '95']
        colnames = ['shift-' + name for name in y]
        if isinstance(ecpercs, dict):
            for key, color in colors.items():
                df = ecpercs.get(key)
                if df is None:
                    continue
                for _, row in df.loc[:, colnames].iterrows():
                    plt.plot(x[1:], row[1:], '-o', lw=0.5, color=color, zorder=5)
                    plt.plot(x[:1], row[:1], 'o', lw=0.5, color=color, zorder=5)
                mean = [ecpercs[key + 'mean'][col] for col in y]
                plt.plot(x[1:], mean[1:], '-', lw=6, color=color, zorder=6)
                plt.plot([0.5, 1.5], [mean[0], mean[0]], '-', lw=6, color=color, zorder=6)
        else:
            for _, row in ecpercs.loc[:, colnames].iterrows():
                plt.plot(x[1:], row[1:], '-o', lw=0.5, color='green', zorder=5)
                plt.plot(x[:1], row[:1], 'o', lw=0.5, color='green', zorder=5)

    ax = plt.gca()
    fig.text(0.65, 0.90, "2050", ha='right')
    if var == 'tas':
        text = f"t2m, {season.upper()}"
        ylabel = r"Change (${}^{\circ}$C)"
        if ylimits:
            ticks = np.arange(ylimits[0], ylimits[1]+0.01, 0.5)
            labels = [str(x) if abs(x % 1) < 0.01 else '' for x in ticks]
            plt.yticks(ticks, labels)
    else:
        text = f"precip, {season.upper()}"
        ylabel = r"Change (%)"
    fig.text(0.20, 0.15, text, ha='left')
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    handles = [Patch(facecolor='#888888', edgecolor='#333333', lw=0, label='CMIP5')]
    for key, color in colors.items():
        if key in ecpercs:
            handles.append(Line2D([0], [0], color=color, lw=6, label=key))
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)

    plt.xticks([1, 2, 3, 4, 5, 6], ['ave', 'P05', 'P10', 'P50', 'P90', 'P95'])
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(direction='in', length=10, which='both')
    ax.yaxis.set_tick_params(direction='in', length=10, which='both')
    if ylimits:
        plt.ylim(*ylimits)

    plt.tight_layout()

    filename = f"change-{var}-{area}-{season}-{year}.pdf"
    plt.savefig(filename, bbox_inches='tight')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('var', help="Short variable name (e.g. 'pr' or 'tas')")
    parser.add_argument('area', help="area name of interest")
    parser.add_argument('season', choices=['year', 'djf', 'mam', 'jja', 'son'],
                        help="season of interest")
    parser.add_argument('--relative', action='store_true', help="Variable is "
                        "relative (percentuel) change")
    parser.add_argument('--write-csv', action='store_true', help="Write "
                        "CMIP5 percentiles and percentile distributions to CSV")
    parser.add_argument('--scenario', action='append', default=[],
                        help="EC-EARTH scenario name(s), corresponding to "
                        "given period. Option can be repeated, and should "
                        "correspond to the --period option. "
                        "Leaving this option out will not (over)plot EC-EARTH data.")
    parser.add_argument('--period', nargs=2, action='append', type=int, default=[],
                        help="EC-EARTH period of interest, "
                        "in start and end year (inclusive). "
                        "The number of periods must match a corresponding scenario.")
    parser.add_argument('--ecearth-dir', help="EC-EARTH main data directory, "
                        "with extracted area averages in subdirectories.")
    parser.add_argument('-v', '--verbose', action='count',
                        default=0, help="Verbosity level")
    parser.add_argument('--ref-period', type=int, nargs=2,
                        default=[REFPERIOD[0], REFPERIOD[1]],
                        help="Reference period in years (inclusive)")
    parser.add_argument('--cmip5-period', type=int, nargs=2, default=[2036, 2065],
                        help="CMIP5 year range of interest (inclusive)")
    parser.add_argument('--basedir', default=".", help="Base input directory")
    parser.add_argument('--plot-title',)
    parser.add_argument('--plot-text',
                        help="Text in bottom-left corner. Defaults to variable & season")
    parser.add_argument('--plot-xlabel')
    parser.add_argument('--plot-ylabel')
    parser.add_argument('--plot-ylimits', nargs=2, type=float)
    parser.add_argument('--no-plot', action='store_true', help="Don't create a plot")

    args = parser.parse_args()
    if args.ecearth_dir is None:
        args.ecearth_dir = args.basedir
    args.ecearth = len(args.scenario) > 0
    if len(args.scenario) != len(args.period):
        raise ValueError("Number of EC-EARTH scenarios does not match "
                         "the number of EC-EARTH periods")
    args.basedir = Path(args.basedir)
    args.ecearth_dir = Path(args.ecearth_dir)

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


def load_cubes(basedir, var, area):
    logger.info("Reading data")
    dirname = basedir / f"{var}-{area}-area-averaged"
    paths = sorted(dirname.glob("*.nc"))
    filenames = [path.name for path in paths]
    cubes = [iris.load_cube(str(path)) for path in paths]
    df = pd.DataFrame(
        {'cube': cubes,
         'frealization': [str(path.stem).split('-')[-1] for path in paths],
         'filename': filenames,
        })

    attrs = pd.DataFrame([row['cube'].attributes for _, row in df.iterrows()])
    df = pd.concat([df, attrs], axis=1)

    pattern = re.compile(r'(r\d+i\d+p\d+)')
    # fn-real-id: filename-realization-identification
    df['fn-real-id'] = df['filename'].apply(lambda string: pattern.search(string).group(1))

    # A few datasets have lacking metadata; set the metadata deduced from the filename
    expid = df['experiment_id'].astype('str')
    indices = np.where((expid == 'NaN') | (expid == '') | (expid == 'nan'))[0]
    pattern = r'(historical|rcp\d\d)'
    regex = re.compile(pattern)
    df.loc[indices, 'experiment_id'] = [regex.search(filename).group(1)
                                        for filename in df['filename'][indices]]

    df['model'] = df['model_id']

    df['rcpname'] = [cube.attributes.get('experiment', '').lower() for cube in cubes]
    df['rcp'] = RCP.HISTORICAL
    df.loc[df['rcpname'] == 'rcp4.5', 'rcp'] = RCP.RCP45
    df.loc[df['rcpname'] == 'rcp6', 'rcp'] = RCP.RCP6
    df.loc[df['rcpname'] == 'rcp8.5', 'rcp'] = RCP.RCP85
    del df['rcpname']

    return df


def concat_cubes(df):
    """Match and concatenate cubes into a dataset spanning the full time frame.

    Cubes are concatenated by filename matching: run IDs (the "r*i*p*"
    part in the filenames, with '*' being one or more digits 0-9)
    between historical and future scenario (RCP or SSP) should match
    for datasets to be concatenated into one long dataset, spanning
    1850/60/80 to 2100.

    Matching by metadata would be better, but the essential metadata
    is not always there, correct, or sometimes insufficient (when
    there are changings in e.g. the physics part of a realization,
    where one would have r1i1p1, r1i1p2, r1i1p3).

    """

    concatenated = pd.DataFrame(columns=df.columns)
    for model, group in df.groupby('model_id'):
        nuse = defaultdict(int)
        hist_rows = group['experiment_id'].str.lower() == 'historical'
        nonhist_rows = group['experiment_id'].str.lower() != 'historical'
        for _, row in group.loc[nonhist_rows, :].iterrows():
            rcp = row['experiment_id']
            pid = row['fn-real-id']
            match = hist_rows & (group['fn-real-id'] == pid)
            try:
                assert match.sum() > 0
            except AssertionError:
                logger.warning("No match for %s", row['filename'])
                continue
            if match.sum() > 1:
                logger.warning("Multiple historical matches found for RCP %s - %s: %s "
                               "Using the first item.", rcp, pid, group.loc[match, 'filename'])
            matchrow = group.loc[match, :].iloc[0, :]
            logger.debug("%s %s: matching %s with historical %s",
                         model, rcp, row['fn-real-id'], matchrow['fn-real-id'])
            cubes = CubeList([matchrow['cube'], row['cube']])
            equalise_attributes(cubes)
            unify_time_units(cubes)
            try:
                row['cube'] = cubes.concatenate_cube()
            except iris.exceptions.ConcatenateError:
                logger.warning("DATA SKIPPED: Error concatenating %s with %s",
                               matchrow['filename'], row['filename'])
                continue
            nuse[pid] += 1
            concatenated = concatenated.append(row)
    concatenated.reset_index(inplace=True)
    grouped = concatenated.groupby(['model', 'fn-real-id'])
    concatenated['parent_nuse'] = grouped['fn-real-id'].transform('count')

    return concatenated


def load_ecearth(datadir, var, area):
    """Load EC-EARTH data into Iris cubes"""

    logger.info('Reading EC-EARTH data')
    dirname = datadir / f"{var}-ecearth-{area}-area-averaged/"
    paths = [str(path) for path in dirname.glob(f"{var}*.nc")]
    cubes = iris.load(paths)

    return cubes


def process_ecearth(cubes, season, constraint=None, rcp=RCP.RCP85, model='EC-EARTH'):
    """Move cubes into a dataframe and extract a season (if applicable)"""

    df = pd.DataFrame({'cube': cubes, 'rcp': rcp, 'model': model})
    if season == 'year':
        df['extracted'] = df['cube']
    elif constraint:
        logger.info("EC-EARTH: extracting season")
        df['extracted'] = df['cube'].apply(constraint.extract)
    else:
        logger.warning("Season given to extract, but the constraint passed is empty")
        df['extracted'] = df['cube']

    return df


def load_data(datadir, var, area, season, ecdir=None):
    """Read datasets, concatenate them and extract seasons if needed.

    Additionally reads EC-EARTH data if requested (by passing a non-Falsey value for `ecdir`).

    """

    df = load_cubes(datadir, var, area)
    df = concat_cubes(df)

    constraint = iris.Constraint(season=lambda point: point == season)
    if season == 'year':
        df['extracted'] = df['cube']
    else:
        logger.info("Extracting season")
        df['extracted'] = df['cube'].apply(constraint.extract)

    ecdf = None
    if ecdir:
        cubes = load_ecearth(ecdir, var, area)
        ecdf = process_ecearth(cubes, season, constraint)

    return df, ecdf


def calc_percentiles(df, period, column='extracted',
                     relative=False, refperiod=None):
    logger.info("Calculating percentiles")
    if refperiod is None:
        refperiod = REFPERIOD
    stats = defaultdict(list)
    refconstraint = iris.Constraint(year=lambda year: refperiod[0] <= year <= refperiod[1])
    constraint = iris.Constraint(year=lambda year: period[0] <= year <= period[1])

    stats = []
    # This loop could easily be done in parallel However, attempts
    # with multiprocessing.Pool thus far failed: quite often, the
    # process got stuck on a lock. I have not been able to deduce the
    # cause; it could be caused by memory issues due to the large
    # memory overhead and copying of data, but memory should all be
    # well within system limits. It might even be a bug in Python's
    # multiprocessing.
    # Possibly, underlying routines are not thread-safe (e.g.,
    # Iris extract), though I'd expect Python's multiprocessing
    # to not suffer from this, since these would be individual
    # processes.
    for _, row in df.iterrows():
        cube = row[column]
        ref_cube = refconstraint.extract(cube)
        future_cube = constraint.extract(cube)
        if ref_cube is None:
            logger.warning("Reference data not found for %s", row['model'])
            continue
        if future_cube is None:
            logger.warning("Future data not found for %s", row['model'])
            continue

        stat = {
            'model': row['model'],
            'future': row['rcp'],
            'ref-mean': ref_cube.data.mean(),
            'future-mean': future_cube.data.mean()
        }
        basepercs = np.percentile(ref_cube.data, list(map(int, STATS[1:])))
        percs = np.percentile(future_cube.data, list(map(int, STATS[1:])))
        for baseperc, perc, col in zip(basepercs, percs, STATS[1:]):
            stat[f'ref-{col}'] = baseperc
            stat[f'future-{col}'] = perc
        stats.append(stat)
    stats = pd.DataFrame(stats)

    # Calculate the difference between reference and main periods
    for col in STATS:
        stats[f'shift-{col}'] = stats[f'future-{col}'] - stats[f'ref-{col}']
        if relative:
            stats[f'shift-{col}'] = (stats[f'shift-{col}'] / stats[f'ref-{col}']) * 100
        stats[col] = stats[f'shift-{col}']

    return stats


def calc_percentile_distribution(df):
    percs = pd.DataFrame({'mean': 0, '5': 0, '10': 0, '25': 0, '50': 0,
                          '75': 0, '90': 0, '95': 0}, index=STATS)
    percentiles = list(map(int, STATS[1:]))
    for key in percs.index:
        p = np.percentile(df.loc[:, key], percentiles)
        percs.loc[key, STATS[1:]] = p
        percs.loc[key, 'mean'] = df.loc[:, key].mean()
    return percs


def run(basedir, var, area, season, options=None):
    if options is None:
        options = {}
    oplot = options.get('plot', {})
    relative = options.get('relative', False)
    ecopt = options.get('ecearth', {})
    scenarios = ecopt.get('scenarios', [])
    year = options['period']
    year = (year[0] + year[1] - 1) // 2

    ecdir = ecopt['datadir'] if ecopt.get('calc') else None
    df, ecdf = load_data(basedir, var, area, season, ecdir)

    l = len(df)
    df = df[['model', 'realization', 'rcp', 'filename', 'extracted']].dropna()
    if len(df) < l:
        logging.warning("Removed %d rows with unknown or bad data", len(df) - l)

    percs = calc_percentiles(df, period=options['period'], refperiod=options['ref-period'],
                             column='extracted', relative=relative)
    perc_distr = calc_percentile_distribution(percs)
    if options.get('csv'):
        percs.to_csv(f'change-{var}-{season}-{area}-{year}.csv', index=False)
        perc_distr.to_csv(f'change-distr-{var}-{season}-{area}-{year}.csv', index=True)

    if oplot.get('plot'):
        figure = plt.figure(figsize=(6, 6))
        perc_ranges = [(10, 90), (25, 75)]
        kcs.plotting.plot_cmip5_percentiles(perc_distr, var, area, season, year,
                                        perc_ranges=perc_ranges, zorder=2)

    ecpercs = None
    if ecdf is not None:
        ecpercs = {
            scenario: calc_percentiles(ecdf, period=period, refperiod=options['ref-period'],
                                       column='extracted', relative=relative)
            for scenario, period in zip(scenarios, ecopt.get('periods', []))
        }
        for key in list(ecpercs.keys()):
            ecpercs[key+'mean'] = {col: ecpercs[key][col].mean() for col in STATS}

        if oplot.get('plot'):
            kcs.plotting.plot_ecearth_percentiles(ecpercs, scenarios, zorder=5)

    if oplot.get('plot'):
        kcs.plotting.plot_finish_percentiles(var, season, year, scenarios,
                                             text=oplot.get('text'),
                                             title=oplot.get('title'),
                                             xlabel=oplot.get('xlabel'),
                                             ylabel=oplot.get('ylabel'),
                                             ylimits=oplot.get('ylimits'))

        plt.tight_layout()
        filename = f"change-{var}-{area}-{season}-{year}.pdf"
        plt.savefig(filename, bbox_inches='tight')


def main():
    args = parse_args()
    setup_logging(args.verbose)

    options = {
        'ref-period': args.ref_period,
        'period': args.cmip5_period,
        'relative': args.relative,
        'csv': args.write_csv,
        'ecearth': {
            'calc': args.ecearth,
            'datadir': args.ecearth_dir,
            'scenarios': args.scenario,
            'periods': args.period,
        },
        'plot': {
            'plot': not args.no_plot,
            'title': args.plot_title,
            'text': args.plot_text,
            'xlabel': args.plot_xlabel,
            'ylabel': args.plot_ylabel,
            'ylimits': args.plot_ylimits,
        }
    }

    run(args.basedir, args.var, args.area, args.season, options=options)


if __name__ == "__main__":
    main()
