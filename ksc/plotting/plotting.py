import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


COLORS = {'G': '#AAEE88', 'W': '#225511',
          (5, 95): "#BBBBBB", (10, 90): "#888888", (25, 75): "#333333",
          'W_H': '#CA181A', 'G_L': '#84F1F1',
          'G_H': '#DD9825', 'W_L': '#2501F6'}
ALPHAS = {(5, 95): 0.8, (10, 90): 0.4, (25, 75): 0.4}
PERC_RANGES = [(5, 95), (10, 90), (25, 75)]


def plot_cmip5_percentiles(df, colors=None, alphas=None, perc_ranges=None,
                           zorder=2, figure=None):
    if figure is None:
        figure = plt.gcf()

    if colors is None:
        colors = COLORS
    if alphas is None:
        alphas = ALPHAS
    if perc_ranges is None:
        perc_ranges = PERC_RANGES

    for perc in perc_ranges:
        # Plot the mean
        x = [0.5, 1.5]
        y1 = [df.loc['mean', str(perc[0])]] * 2
        y2 = [df.loc['mean', str(perc[1])]] * 2
        plt.fill_between(x, y1, y2, color=colors[perc], alpha=alphas[perc], zorder=zorder)
        # Plot the percentiles
        ps = ['5', '10', '50', '90', '95']
        x = [2, 3, 4, 5, 6]
        y1 = [df.loc[p, str(perc[0])] for p in ps]
        y2 = [df.loc[p, str(perc[1])] for p in ps]
        plt.fill_between(x, y1, y2, color=colors[perc], alpha=alphas[perc], zorder=zorder)
        zorder += 1

    figure = plt.gcf()

    return figure, zorder


def plot_ecearth_percentiles(percs, scenarios, colors=None, zorder=5, figure=None):
    if figure is None:
        figure = plt.gcf()
    if colors is None:
        colors = COLORS

    x = [1, 2, 3, 4, 5, 6]
    y = ['mean', '5', '10', '50', '90', '95']
    colnames = ['shift-' + name for name in y]
    if isinstance(percs, dict):
        for key in scenarios:
            color = colors.get(key, '#000000')
            df = percs.get(key)
            if hasattr(df, 'columns') and set(colnames) <= set(df.columns):
                for _, row in df.loc[:, colnames].iterrows():
                    plt.plot(x[1:], row[1:], '-o', lw=0.5, color=color, zorder=zorder)
                    plt.plot(x[:1], row[:1], 'o', lw=0.5, color=color, zorder=zorder)
            mean = [percs[key + 'mean'][col] for col in y]
            plt.plot(x[1:], mean[1:], '-', lw=6, color=color, zorder=zorder+1)
            plt.plot([0.5, 1.5], [mean[0], mean[0]], '-', lw=6, color=color, zorder=zorder+1)
    else:
        for _, row in percs.loc[:, colnames].iterrows():
            plt.plot(x[1:], row[1:], '-o', lw=0.5, color='green', zorder=5)
            plt.plot(x[:1], row[:1], 'o', lw=0.5, color='green', zorder=5)


def plot_finish_percentiles(var, season, year, scenarios=None, colors=None,
                            text='', title='', xlabel='', ylabel='', ylimits=None):
    """Add titles, limits, legends etc to finish the percentile plot"""

    figure = plt.gcf()
    ax = plt.gca()

    if scenarios is None:
        scenarios = []
    if colors is None:
        colors = COLORS

    if year:
        figure.text(0.65, 0.90, f"{year}", ha='right')
    if var == 'tas':
        text = text or f"t2m, {season.upper()}"
        ylabel = ylabel or r"Change (${}^{\circ}$C)"
        if ylimits:
            ticks = np.arange(ylimits[0], ylimits[1]+0.01, 0.5)
            labels = [str(x) if abs(x % 1) < 0.01 else '' for x in ticks]
            plt.yticks(ticks, labels)
    else:
        text = text or f"precip, {season.upper()}"
        ylabel = ylabel or r"Change (%)"
    figure.text(0.20, 0.15, text, ha='left')
    if xlabel:
        plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

    handles = [Patch(facecolor='#888888', edgecolor='#333333', lw=0, label='CMIP5')]
    for key in scenarios:
        color = colors.get(key, '#000000')
        handles.append(Line2D([0], [0], color=color, lw=6, label=key))
    plt.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False)

    plt.xticks([1, 2, 3, 4, 5, 6], ['ave', 'P05', 'P10', 'P50', 'P90', 'P95'])
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_tick_params(direction='in', length=10, which='both')
    ax.yaxis.set_tick_params(direction='in', length=10, which='both')
    if ylimits:
        plt.ylim(*ylimits)

    return figure


def run():
    plt.tight_layout()
    filename = f"change-{var}-{area}-{season}-{year}.pdf"
    plt.savefig(filename, bbox_inches='tight')
