def plot():
    if oplot.get('plot'):
        figure = plt.figure(figsize=(6, 6))
        perc_ranges = [(10, 90), (25, 75)]
        plotting.plot_cmip5_percentiles(perc_distr, var, area, season, year,
                                        perc_ranges=perc_ranges, zorder=2)

    if oplot.get('ecearth-plot'):
        plotting.plot_ecearth_percentiles(ecpercs, scenarios, zorder=5)

    if oplot.get('plot'):
        plotting.plot_finish_percentiles(var, season, year, scenarios,
                                         text=oplot.get('text'),
                                         title=oplot.get('title'),
                                         xlabel=oplot.get('xlabel'),
                                         ylabel=oplot.get('ylabel'),
                                         ylimits=oplot.get('ylimits'))

        plt.tight_layout()
        filename = f"change-{var}-{area}-{season}-{year}.pdf"
        plt.savefig(filename, bbox_inches='tight')
