"""Example usage:

$ python -m kcs.change_perc.plot pr_change_2050_jja_nlpoint.csv pr_change_2050_jja_nlpoint.png \
    --epoch 2050 --text 'precip, DJF', --ytitle 'Change (%)' --ylimits -60 45


To overplot a series of single runs for a model of interest, use the
--scenario-run option, which takes a name and a CSV file (can be used
multiple times, for multiple scenarios). The CSV files are the output
of the --run-changes option of the `kcs.change_perc` module.

$ python -m kcs.change_perc.plot pr_change_2050_jja_nlpoint.csv pr_change_2050_jja_nlpoint.png \
    --epoch 2050 --text 'precip, DJF', --ytitle 'Change (%)' --ylimits -60 45 \
    --scenario-run G pr_change_G2050_jja_nlpoint_ecearth.csv \
    --scenario-run W pr_change_W2050_jja_nlpoint_ecearth.csv

"""

import sys
import logging
import argparse
from collections import OrderedDict
import pandas as pd
import matplotlib.pyplot as plt
# from matplotlib.lines import Line2D
# from matplotlib.patches import Patch
import kcs.utils.logging
import kcs.utils.argparse


COLORS = {'G': '#AAEE88', 'W': '#225511',
          (5, 95): "#BBBBBB", (10, 90): "#888888", (25, 75): "#333333",
          'W_H': '#CA181A', 'G_L': '#84F1F1',
          'G_H': '#DD9825', 'W_L': '#2501F6'}
ALPHAS = {(5, 95): 0.8, (10, 90): 0.4, (25, 75): 0.4}
# PERC_RANGES = [(5, 95), (10, 90), (25, 75)]
PERC_RANGES = [(10, 90), (25, 75)]


# pylint: disable=invalid-name
logname = 'kcs.change_perc.plot' if __name__ == '__main__' else __name__
logger = logging.getLogger(logname)   # pylint: disable=invalid-name


def plot_cmip(data, colors=None, alphas=None, perc_ranges=None,
              zorder=2, columns=None, figure=None):
    """DUMMY DOCSTRING"""
    if figure is None:
        figure = plt.gcf()

    if colors is None:
        colors = COLORS
    if alphas is None:
        alphas = ALPHAS
    if perc_ranges is None:
        perc_ranges = PERC_RANGES

    # pragma pylint: disable=invalid-name
    for perc in perc_ranges:
        # Plot the mean
        if (columns and columns[0] == 'mean') or not columns:
            x = [0.5, 1.5]
            y1 = [data.loc['mean', str(perc[0])]] * 2
            y2 = [data.loc['mean', str(perc[1])]] * 2
        plt.fill_between(x, y1, y2, color=colors[perc], alpha=alphas[perc], zorder=zorder)
        # Plot the percentiles
        cols = [column for column in columns if column != 'mean']
        x = list(range(2, 2+len(cols)))
        y1 = [data.loc[column, str(perc[0])] for column in cols]
        y2 = [data.loc[column, str(perc[1])] for column in cols]
        plt.fill_between(x, y1, y2, color=colors[perc], alpha=alphas[perc], zorder=zorder)
        zorder += 1
    # pragma pylint: enable=invalid-name

    return figure


def plot_runs(percs, columns=None, only_mean=False, color='black', zorder=5, figure=None):
    """Plot individual runs, and their mean"""
    if figure is None:
        figure = plt.gcf()

    if (columns and columns[0] == 'mean') or not columns:
        if not only_mean:
            for mean in percs['mean']:
                plt.plot([1], [mean], 'o', lw=0.5, ms=4, color=color, zorder=zorder)
        # Plot the mean of the means
        mean = percs['mean'].mean()
        plt.plot([0.5, 1.5], [mean, mean], '-', lw=6, color=color, zorder=zorder)
    columns = [column for column in columns if column != 'mean']
    # pragma pylint: disable=invalid-name
    if not only_mean:
        for _, row in percs.iterrows():
            data = row[columns]
            x = list(range(2, 2+len(columns)))
            plt.plot(x, data, '-o', lw=0.5, ms=4, color=color, zorder=zorder+1)
    mean = percs[columns].mean(axis=0)
    x = list(range(2, 2+len(columns)))
    plt.plot(x, mean, '-', lw=6, color=color, zorder=zorder+1)
    # pragma pylint: enable=invalid-name

    return figure


def plot_finish(colors=None,
                text='', title='', xlabel='', ylabel='', epoch='',
                xlabels=None, ylimits=None, figure=None):
    """Add titles, limits, legends etc to finish the percentile plot"""

    if not figure:
        figure = plt.gcf()
    axes = plt.gca()

    if colors is None:
        colors = COLORS

    if epoch:
        figure.text(0.90, 0.90, f"{epoch}", ha='right')
    figure.text(0.20, 0.15, text, ha='left')
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    # handles = [Patch(facecolor='#888888', edgecolor='#333333', lw=0, label='CMIP5')]
    # for key in scenarios:
    #     color = colors.get(key, '#000000')
    #     handles.append(Line2D([0], [0], color=color, lw=6, label=key))
    # plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)

    if not xlabels:
        xlabels = ['ave', 'P05', 'P10', 'P50', 'P90', 'P95']
    xticks = range(1, len(xlabels)+1)
    plt.xticks(xticks, xlabels)
    axes.xaxis.set_ticks_position('both')
    axes.yaxis.set_ticks_position('both')
    axes.xaxis.set_tick_params(direction='in', length=10, which='both')
    axes.yaxis.set_tick_params(direction='in', length=10, which='both')
    if ylimits:
        plt.ylim(*ylimits)

    return figure


def run(data, labels, limits, columns, xlabels, scenarios=None, only_scenario_mean=False):
    """Do all the work"""

    if scenarios is None:
        scenarios = {}

    figure = plt.figure(figsize=(6, 6))
    figure = plot_cmip(data, zorder=2, columns=columns)

    zorder = 5
    for name, runs in scenarios.items():
        color = COLORS.get(name, 'black')
        zorder += 2
        figure = plot_runs(runs, columns, only_mean=only_scenario_mean,
                           color=color, zorder=zorder, figure=figure)

    plot_finish(colors=None, text=labels['text'], epoch=labels['epoch'],
                xlabel=labels['x'], ylabel=labels['y'], title=labels['title'],
                ylimits=limits, xlabels=xlabels)

    if labels['title']:
        plt.title(labels['title'])
    return figure


def parse_args():
    """DUMMY DOCSTRING"""
    parser = argparse.ArgumentParser(parents=[kcs.utils.argparse.parser],
                                     conflict_handler='resolve')
    parser.add_argument('csvfile', help="Input CSV file with percentile change distribution.")
    parser.add_argument('output', help="Output figure file. The extension automatically "
                        "specifies the file type.")
    parser.add_argument('--scenario-run', nargs=2, action='append',
                        help="Optional individual runs for scenarios, to overplot. "
                        "Can be used multiple times, once for each scenario. Takes "
                        "two argument: a scenario name (e.g., 'G', or 'W_L') and a "
                        "CSV file that contains a list of runs: it should "
                        "at least contain columns corresponding to --columns.")
    parser.add_argument('--only-scenario-mean', action='store_true', help="Plot only the "
                        "mean of the individual scenario runs (if given).")
    parser.add_argument('--title', help="Plot title.")
    parser.add_argument('--text', help="Text in bottom-left corner. Can be used as "
                        "alternative to a plot title.")
    parser.add_argument('--epoch', help="Plot given epoch value in top-left corner.")
    parser.add_argument('--xtitle')
    parser.add_argument('--ytitle')
    parser.add_argument('--ylimits', nargs=2, type=float)
    parser.add_argument('--columns', nargs='+', default=['mean', '5', '10', '50', '90', '95'],
                        help="Data columns to plot. These should all be present in the input "
                        "file. If the first column is 'mean', this will be plotted separately.")
    parser.add_argument('--xlabels', nargs='+', default=['ave', 'P05', 'P10', 'P50', 'P90', 'P95'],
                        help="X-axis tick mark labels. Should have the same number of columns "
                        "as --columns.")

    args = parser.parse_args()
    if len(args.columns) != len(args.xlabels):
        raise ValueError("'--columns' and '--xlabels' don't have the same number of values")

    return args


def main():
    """DUMMY DOCSTRING"""
    args = parse_args()
    kcs.utils.logging.setup(args.verbosity)
    logger.debug("%s", " ".join(sys.argv))
    logger.debug("Args: %s", args)

    data = pd.read_csv(args.csvfile, index_col=0)
    labels = dict(title=args.title, text=args.text, x=args.xtitle, y=args.ytitle,
                  epoch=args.epoch)

    scenarios = OrderedDict()
    for name, csvfile in args.scenario_run:
        scenarios[name] = pd.read_csv(csvfile, index_col=False)

    run(data, labels, limits=args.ylimits,
        columns=args.columns, xlabels=args.xlabels,
        scenarios=scenarios, only_scenario_mean=args.only_scenario_mean)

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')

    logger.info("Done processing. Save figured to %s", args.output)


if __name__ == "__main__":
    main()
