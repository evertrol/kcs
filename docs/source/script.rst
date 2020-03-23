.. _example-script:

Script example
==============

There is also a script that runs through all the steps in the usage
guide, but then as a Python program. This may be easier to edit, and
for finding relevant functions.

While the script is rather lengthy, a lot of it are comments explaning
the steps, plus a lot of bookkeeping, and some plotting
functionality. The actual steps aren't that many.

The full script is included below, but can also be found as
``extras/single-script.py``.


Full code
---------

.. code-block:: python

    import sys
    from pathlib import Path
    import math
    from copy import deepcopy
    import toml
    import pandas as pd
    import matplotlib.pyplot as plt
    from kcs.config import read_config, default_config
    from kcs.utils.atlist import atlist
    from kcs.extraction import calc as extract
    from kcs.utils.io import read_averaged_data, concat_cubes
    from kcs.utils.attributes import get as get_attrs
    from kcs.utils.matching import match
    from kcs.tas_change import calc as calc_tas_change
    from kcs.tas_change.plot import plot as plot_tas_change
    from kcs.steering import calc as calc_steering
    from kcs.steering.core import normalize_average_dataset
    from kcs.steering.plot import plot as plot_steering
    from kcs.change_perc import calc as calc_change_perc
    from kcs.change_perc.plot import plot as plot_change_perc
    from kcs.resample import calc as resample


    # Read any command-line supplied configuration file
    # If not supplied, read the default user one (kcs-config.toml in working directory)
    try:
        read_config(sys.argv[1])
    except IndexError:
        read_config(None)
    # default_config is now updated


    # Read the dataset files from our lists
    cmip_lists = dict(tas="@cmip6-tas.list", pr="@cmip6-pr.list")
    cmip_files = {key: list(atlist(fname)) for key, fname in cmip_lists.items()}
    ecearth_lists = dict(tas="@ecearth-tas.list", pr="@ecearth-pr.list")
    ecearth_files = {key: list(atlist(fname)) for key, fname in ecearth_lists.items()}

    # Alternatively, we could use e.g. glob:
    #     import glob
    #     files_pr = glob.glob("/data/cmip6/*/Amon/pr/*/*/*.nc")
    # etcetera.


    # Step 0: extract and average the areas

    data = {'cmip': {'global': {}, 'nlpoint': {}},
            'ecearth': {'global': {}, 'nlpoint': {}}}
    # Areas are given as a dictionary: {name: definition}
    # The definition can be taken from the default_config
    # Conveniently, None is the definition for the global area
    area = {'global': None}
    # Almost all parameters have a proper default
    # The template parameter is just made explicit
    template = "data/cmip6/{var}-{area}-averaged/{filename}.nc"
    data['cmip']['global']['tas'] = extract(cmip_files['tas'], area, ignore_common_warnings=True,
                                         template=template)
    area = {'nlpoint': default_config['areas']['nlpoint']}
    for var in {'tas', 'pr'}:
        data['cmip']['nlpoint'][var] = extract(cmip_files[var], area, ignore_common_warnings=True,
                                            template=template)

    area = {'global': None}
    template = "data/ecearth/{var}-{area}-averaged/{filename}.nc"
    data['ecearth']['global']['tas'] = extract(ecearth_files['tas'], area, ignore_common_warnings=True,
                                            template=template)
    area = {'nlpoint': default_config['areas']['nlpoint']}
    for var in {'tas', 'pr'}:
        data['ecearth']['nlpoint'][var] = extract(ecearth_files[var], area, ignore_common_warnings=True,
                                               template=template)



    # Step 1-pre: read previously created data
    datasets = {'cmip': {'global': {}, 'nlpoint': {}},
                'ecearth': {'global': {}, 'nlpoint': {}}}

    skipped_above = False
    if skipped_above:
        # Use pre-defined @-lists
        cmip_lists = {'global-tas': "@cmip6-tas-global-averaged.list",
                      'nlpoint-pr': "@cmip6-pr-nlpoint-averaged.list",
                      'nlpoint-tas': "@cmip6-tas-nlpoint-averaged.list"}
        ecearth_lists = {'global-tas': "@ecearth-tas-global-averaged-short.list",
                         'nlpoint-pr': "@ecearth-pr-nlpoint-averaged-short.list",
                         'nlpoint-tas': "@ecearth-tas-nlpoint-averaged-short.list"}
        cmip_files = {key: list(atlist(Path(fname))) for key, fname in cmip_lists.items()}
        ecearth_files = {key: list(atlist(Path(fname))) for key, fname in ecearth_lists.items()}

        for area in {'global', 'nlpoint'}:
            for var in {'tas', 'pr'}:
                if area == 'global' and var == 'pr':
                    continue
                key = f"{area}-{var}"
                datasets['cmip'][area][var] = read_averaged_data(cmip_files[key])
                # Limit to eight datasets for testing purposes
                datasets['ecearth'][area][var] = read_averaged_data(ecearth_files[key][:8])

    else:
        for name in data:
            for area in data[name]:
                for var in data[name][area]:
                    paths = [Path(item.path) for item in data[name][area][var]]
                    cubes = [item.cube for item in data[name][area][var]]
                    datasets[name][area][var] = get_attrs(cubes, paths)

    # Step 1a: calculate the tas annual change
    dataset = datasets['cmip']['global']['tas']
    dataset = match(dataset)
    cols = ['model', 'experiment', 'realization', 'index_match_run']
    #print(dataset[cols])
    print(dataset.columns)
    tas_change_percentiles, dataset = \
        calc_tas_change(dataset, reference_period=[1990, 2019], relative=False)
    datasets['cmip']['global']['tas'] = dataset
    print(tas_change_percentiles)
    print(dataset.columns)
    print(tas_change_percentiles.index.dtype)
    plot_tas_change(tas_change_percentiles, "tas_change_cmip.png",
                    xlabel="Year", ylabel="Temperature increase [${}^{\circ}$C]",
                    title="Yearly annual temperature change",
                    legend=True, smooth=10)

    #tas_change_percentiles.to_csv("tas_change_cmip.csv", index_label="date")
    #tas_change_percentiles = pd.read_csv("tas_change_cmip.csv")


    # Step 1b: calculate steering table for the scenarios

    scenarios = []
    for epoch in {2050, 2085}:
        # Note: percentiles need to be strings
        scenarios.extend([{'name': 'G', 'percentile': '10', 'epoch': epoch},
                          {'name': 'W', 'percentile': '90', 'epoch': epoch}])
    dataset = datasets['ecearth']['global']['tas']
    steering = calc_steering(dataset, tas_change_percentiles, scenarios,
                             rolling_mean=10, reference_period=[1990, 2019])
    print(steering)
    steering = pd.DataFrame(steering)
    print(steering)
    # Normalize EC-EARTH data for the plot
    ecearth_data = normalize_average_dataset(dataset['cube'], relative=False,
                                           reference_period=[1990, 2019])

    plot_steering(tas_change_percentiles, steering, "tas_change_cmip_steering.png",
                  extra_data=ecearth_data, reference_epoch=2005,
                  xlabel="Year", ylabel="Temperature increase [${}^{\circ}$C]",
                  title="Yearly annual temperature change",
                  legend=True, smooth=10)


    #steering.to_csv("steering.csv", index=False)
    #steering = pd.read_csv("steering.csv")
    #steering_table['period'] = steering_table['period'].apply(
    #    lambda x: tuple(map(int, x.strip('()').split(','))))



    # Step 2

    # The nested loops below are essentially the same as the
    # kcs.change_perc.runall helper module
    area = 'nlpoint'
    seasons = ['djf', 'mam', 'jja', 'son']
    distrs = {}
    writecsv = False
    columns = ['mean', '5', '10', '50', '90', '95']
    xlabels = ['ave', 'P05', 'P10', 'P50', 'P90', 'P95']

    for var in ['tas', 'pr']:
        relative = var == 'pr'
        if var == 'pr':
            text = 'prec'
            ylabel = "Change (%)"
        elif var == 'tas':
            text = 't2m'
            ylabel = r"Change (${}^{\circ}$C)"

        distrs[var] = {}
        dataset = datasets['cmip'][area][var]
        dataset = match(dataset)
        cmip_dataset = concat_cubes(dataset)

        ecearth_dataset = datasets['ecearth'][area][var]

        for epoch, steering_group in steering.groupby('epoch'):
            distrs[var][epoch] = {}
            period = epoch - 15 + 1, epoch + 15

            for season in seasons:
                perc_distr, perc = calc_change_perc(dataset.copy(), season, period,
                                                    relative=relative, reference_period=[1990, 2019])
                # Save the CMIP percentile distributions for later
                distrs[var][epoch][season] = perc_distr
                if writecsv:
                    filename = f"{var}_{epoch}_{season}_perc_distr.csv"
                    perc_distr.to_csv(filename, index=True)
                    filename = f"{var}_{epoch}_{season}_perc.csv"
                    perc.to_csv(filename, index=True)

                scenarios = {}
                for _, row in steering_group.iterrows():
                    data = ecearth_dataset.copy()
                    name = row['name'].rstrip('0123456789')  # remove the year part
                    period = row['period']  # Matched EC-EARTH period
                    print(period)
                    _, scenarios[name] = calc_change_perc(data, season, period, relative=relative,
                                                          reference_period=[1990, 2019])
                    if writecsv:
                        filename = f"{var}_{epoch}_{season}_{name}_perc.csv"
                        scenarios[name].to_csv(filename, index=False)


                labels = {
                    'title': '',
                    'text': text,
                    'y': ylabel,
                    'x': '',
                    'epoch': epoch,
                }
                plot_change_perc(perc_distr, labels, limits=None, columns=columns, xlabels=xlabels,
                                 scenarios=scenarios)
                plt.tight_layout()
                filename = f"{var}_{epoch}_{season}.png"
                plt.savefig(filename, bbox_inches='tight')


    from pprint import pprint
    pprint(distrs)


    # Step 3: resample the EC-EARTH data

    # Concatenate tas & pr, since we'll need both in resampling step 2
    dataset = pd.concat([datasets['ecearth']['nlpoint'][var] for var in ['tas', 'pr']])
    print(dataset.columns)
    print(dataset[['var', 'experiment', 'model']])

    # Update steering table with precipitation conditions for resampling 1
    steering = steering.rename(columns={'target year': 'epoch', 'resampling period': 'period',
                                        'name': 'scenario'})
    precip_table = pd.DataFrame({'subscenario': ['L', 'H'],
                                 'precip_change_per_t': [4, 8]})
    # must be a better way to create an outer product without inserting a random key
    steering['key'] = precip_table['key'] = 0
    table = pd.merge(steering, precip_table, on='key', how='outer').drop(columns='key')
    # Update precipiation values with delta-t
    table['precip_change'] = table['model_delta_t'] * table['precip_change_per_t']
    # We should end up with nstep1 samples after this step
    nstep1 = 1000

    # Load the conditions for resampling step 2 from a config file
    # We could define them in code, but that would become quite lengthy
    with open('step2_conditions.toml') as fh:
        conditions = toml.load(fh)
    # transform the epoch keys to integer, since TOML only uses string keys
    conditions2 = deepcopy(conditions)
    for key in conditions:
        for prkey in conditions[key]:
            for epoch, value in conditions[key][prkey].items():
                conditions2[key][prkey][int(epoch)] = value
                del conditions2[key][prkey][epoch]
    conditions = conditions2
    from pprint import pprint
    pprint(conditions)

    # Number of final resampled datasets and penalties for resampling step 3
    nstep3 = 8
    penalties = {1: 0.0, 2: 0.0, 3: 1.0, 4: 3.0}
    # fill up the penalties with inf to the number of resampled datasets
    for i in range(max(penalties.keys())+1, nstep3+1):
        penalties[i] = math.inf

    print(table)
    indices, diffs = resample(dataset, table, conditions, penalties,
                              nstep1=nstep1, nstep3=nstep3, nsections=6,
                              reference_period=[1990, 2019], relative=['pr'])

    for key in diffs:
        print(key)
        for var in diffs[key]:
            for season in diffs[key][var]:
                print(var, season)
                print(diffs[key][var][season])


    columns = ['mean', '5', '10', '50', '90', '95']
    xlabels = ['ave', 'P05', 'P10', 'P50', 'P90', 'P95']
    scenarios = {}
    for epoch in {2050, 2085}:
        scenarios[epoch] = {}
        for var in {'pr', 'tas'}:
            scenarios[epoch][var] = {}
            for season in seasons:
                scenarios[epoch][var][season] = {}
    for key, diff in diffs.items():
        epoch, scenario, pr_scenario = key
        epoch = int(epoch)
        for var in {'pr', 'tas'}:
            for season in seasons:
                scenarios[epoch][var][season][f"{scenario}_{pr_scenario}"] = diff[var][season]

    for key, value in diffs.items():
        epoch, scenario, pr_scenario = key
        epoch = int(epoch)
        for var, value2 in value.items():
            relative = var == 'pr'
            if var == 'pr':
                text = 'precip'
                ylabel = "Change (%)"
                ylimits = [-50, 50]
            elif var == 'tas':
                text = 't2m'
                ylabel = r"Change (${}^{\circ}$C)"
                ylimits = [-1, 5]
            for season, diff in value2.items():
                distr = distrs[var][epoch][season]
                labels = {
                    'title': '',
                    'text': f"{text}, {season}",
                    'y': ylabel,
                    'x': '',
                    'epoch': epoch,
                }
                plot_change_perc(distr, labels, limits=ylimits,
                                 columns=columns, xlabels=xlabels,
                                 scenarios=scenarios[epoch][var][season],
                                 only_scenario_mean=True)
                plt.tight_layout()
                filename = f"resampled_{var}_change_{epoch}_{season}_nlpoint.png"
                plt.savefig(filename, bbox_inches='tight')
                plt.close()
