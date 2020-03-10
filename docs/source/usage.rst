==========================
A brief usage guide to kcs
==========================


Preliminaries
=============


This suite (package) of Python modules contains functionality to
perform the first steps for calculating the KNMI Climate
Scenarios. The KNMI Climate Scenarios provide a set of scenarios for
the Netherlands in 2050 and 2085 (middle and end of the 21st
century). These are calculated using a regional model, and the latter
is adjusted to match the global climate change, using global
models. This last step is provided in this package. The package is
called ``kcs``, abbreviated for KNMI Climate Scenarios.


Some of the example commands given are quite long, because of the use
of several options. For readability, these lines are broken across
multiple lines, with a backslash at the end of the lines that need to
be continued. In most terminals and shells, you can safely copy-paste
such commands with the newlines and backslashes and run them, but you
may have to remove the backslash and newlines first. So the following
two commands are the same::

    conda create --name kcs iris pandas h5py toml python=3.7 --channel conda-forge

    conda create --name kcs --channel conda-forge \
        iris pandas h5py toml python=3.7


Installation
------------

For a more detailed installation guide, see the ``INSTALL`` document
in the root directory of the package.

Installation requires a few Python packages: ``SciTools-Iris``,
``Pandas`` and ``h5py`` and ``toml``. The easiest way is to use Conda
to install these. If you don't have Conda, you can obtain the
Miniconda installer from
https://docs.conda.io/en/latest/miniconda.html ; follow the
installation for Miniconda, for Python 3.

With Conda, create a new environment with the required packages::

    conda create --name kcs iris pandas h5py toml python=3.7 --channel conda-forge

You can pick another name, and you may want to change to a newer
Python version in the future, provided it's supported by the various
packages.

Once set up, activate the new environment, and install KCS manually
from its repository::

    conda activate kcs

    pip install git+https://github.com/evertrol/kcs.git


Running modules
---------------

KCS is set up as a set of runnable modules. These take the place of
scripts, and should be thought of as such (in fact, they could be
installed as scripts, but for clarity, they are kept inside the KCS
package as runnable modules). The way to run such a module is as follows::

    python -m kcs.<some_module> [<arguments> ...]

Note the ``-m`` flag after ``python``, which indicates that the
following argument is a module. This also keeps the Python executable
together with the package.

All runnable modules have a `-h` or `--help` option, to get a quick
help; and a`-v` option for verbosity: used once, it will log warnings
(fairly useful). ``-vv`` will produce some "info" log messages as
well, and ``-vvv`` (the maximum level) will produce quite a number of
debug messages along the way.


@-lists
-------

Several of the modules take a list of files as input. These can be
given as one or more standard shell globbing patterns (for example,
``/mnt/cmip5/rcp85/Amon/tas/*/*/tas*.nc``), but another option is to
use a so-called at-list (@-list): this is a file, containing a list of
filenames, one per line. When this file is given as input, but
prepended with a ``@`` sign, the file will be interpreted by the KCS
package as a list of files given inside. E.g., a file ``files.list``
that contains the following::

    tas_Amon_GISS-E2-R_rcp85_r1i1p1_200601-202512.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_202601-205012.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_205101-207512.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_207601-210012.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_210101-212512.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_212601-215012.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_215101-217512.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_217601-220012.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_220101-222512.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_222601-225012.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_225101-227512.nc
    tas_Amon_GISS-E2-R_rcp85_r1i1p1_227601-230012.nc

and it's used as follows::

    python -m kcs.read_files @files.list

this is interpreted as:

.. code-block:: bash

    python -m kcs.read_files tas_Amon_GISS-E2-R_rcp85_r1i1p1_200601-202512.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_202601-205012.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_205101-207512.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_207601-210012.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_210101-212512.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_212601-215012.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_215101-217512.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_217601-220012.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_220101-222512.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_222601-225012.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_225101-227512.nc \
        tas_Amon_GISS-E2-R_rcp85_r1i1p1_227601-230012.nc

Lines in an @-list can be commented-out using a ``#`` at the start of
a line (but trailing comments at the end of a line are not allowed;
these are interpreted as part of the file name).

@-lists can also be nested. This is convenient if you have a set of
sub-lists, and want to group these together. For example, a
``tas-all.list`` could contain::

    @tas-historical.list
    #@tas-rcp26.list
    @tas-rcp45.list
    @tas-rcp60.list
    @tas-rcp85.list

where the RCP 2.6 files are (temporarily) commented out.

Note that file names inside an @-list are relative to the current
working directory, if they are relative; using absolute paths may be
safest option, although this may of course cause lenghty lines.


Adjusting code
--------------

Sometimes, you may want to change some code in the ``kcs``
package. There are a few ways to do this. The most standard way is to
clone the repository, and perform an "editable" installation. Do this
before you install the KCS package with ``pip`` as shown above, or
uninstall it first (``pip uninstall kcs``).

Clone the repository (in a directory where you want the actual code to
stay):

.. code-block:: bash

    git clone https::/github.com/evertrol/kcs.git

    cd kcs

Then install an "editable" version (make sure you are in the activated
Conda environment)::

    pip install -e .

Now, any change you make to the files in the cloned repository, will
be automatically reflected in the installed Python package.


Steps
=====


The following steps perform all the routines for producing the initial
climate scenario setup. The details can be found in Lenderink et al,
2014, and the corresponding appendix.


Step 0: area extraction and averaging
-------------------------------------

Step 0 involves the extraction and averaging of areas. The areas for
the specific KNMI Climate Scenarios are a global average (for the air
temperature, ``tas``), and a point "average" for the air temperature
``tas`` and precipitation ``pr``. This is to be done for both the
available CMIP data, and the model of interest that will be used to
downscale the regional module.

This step is numbered 0, since the user may have other ways to obtain
the same result. The end result should be a set of one-dimensional
datasets containing area-averaged time-series for relevant variable,
in a NetCDF file following the CF conventions, grouped in separate
directories by area and variable. These can then be input to step 1.


For the global CMIP ``tas`` average, run the following command:

.. code-block:: bash

    python -m kcs.extraction --area global @cmip-tas.list --ignore-common-warnings -v

Iris can be quite chatty regarding potential problems: the
`--ignore-common-warnings` option turns this chattiness down, but it
may be worth to leave this option out a first time, to see that the
notices are indeed not really a problem.

Note that optional arguments can be put before or after the mandatory
arguments (``@cmip-tas.list`` is the only required argument here). The
``-m`` is an option to ``python``, not to ``kcs.extraction``.

The ``global`` area is predefined; there are also ``nlpoint``,
``nlbox``, ``rhinebasin`` and ``weurbox`` areas. For their
definitions, see the ``kcs/config/__init__.py`` file; you can also add
more definitions here.

All areas are averaged using a weighted-average, except for the single
point area (``nlpoint``): this uses a standard linear interpolation
(as used in ``iris.cube.Cube.interpolate``).


For the extraction and averaging, all datasets are handled separately:
there is no matching between historical and future data (as is done in
later steps below), since this is not needed.

The output is written to a set of files in a subdirectory that is
named following the variable and area:
``data/<var>-<area>-averaged/<filename>.nc`` (``<filename>`` is the
input filename of an individal dataset file, without its
extension). You can change this using the ``--template`` option, with
a Python-formatting like string. In the example below, we extract the
``tas`` data for our model of interest (EC-EARTH), and save these
results into a separate directory:

.. code-block:: bash

    python -m kcs.extraction --area global  @ecearth-tas.list \
        --template "data/ecearth/{var}-{area}-averaged/{filename}.nc" \
        --ignore-common-warnings -v

Another example, would be if you want separate directories for
e.g. CMIP5 and CMIP6 data:

.. code-block:: bash

    python -m kcs.extraction --area global @cmip5-tas.list \
        --template "data/cmip5/{var}-{area}-averaged/{filename}.nc" \
        --ignore-common-warnings -v

    python -m kcs.extraction --area global @cmip6-tas.list \
        --template "data/cmip6/{var}-{area}-averaged/{filename}.nc" \
        --ignore-common-warnings -v


The examples below perform the extraction for the ``nlpoint`` area,
for both ``tas`` and ``pr``, and for both the CMIP data and the
EC-EARTH ("model of interest") data:

.. code-block:: bash

    python -m kcs.extraction --area nlpoint @cmip-tas.list \
        --ignore-common-warnings -v

    python -m kcs.extraction --area nlpoint @cmip-pr.list \
        --ignore-common-warnings -v

    python -m kcs.extraction --area nlpoint @ecearth-tas.list \
        --template "data/ecearth/{var}-{area}-averaged/{filename}.nc" \
        --ignore-common-warnings -v

    python -m kcs.extraction --area nlpoint @ecearth-pr.list \
        --template "data/ecearth/{var}-{area}-averaged/{filename}.nc" \
        --ignore-common-warnings -v


For non-global and non-point areas, there is a ``--regrid`` option,
which will regrid the data to a common one by one grid before
extraction; this should ensure the same area is extracted, since Iris
does not interpolate grid points when performing area extraction.  If
you want to change the grid to regrid to, you can change the function
``create_grid`` in ``kcs/utils/coord.py``.


The end result of step 0 should be six subdirectories: three for
extracted CMIP data, and three for te model of interest. These three
directories are a global ``tas`` directory, an nlpoint ``tas``
directory and an nlpoint ``pr`` directory.


Step 1a: global tas change
--------------------------

This step simply calculates the global temperature change (historical
and future scenarios), averaging all available model runs, normalised
to a reference (control) period.

Again, the examples use an @-list. These list contain the
area-averaged data from the previous step; the filename indicates the
datasets involved.

.. code-block:: bash

   python -m kcs.tas_change  @cmip-tas-global-averaged.list \
       --outfile=tas_change.csv --on-no-match=randomrun -v  \
       --norm-by=run  --reference-period 1991 2020

Notes on the options:

* ``--outfile``: the output CSV file. This contains the percentiles
  and mean of the normalised ``tas`` value for each year. The
  statistics are calculated across all individual model runs.

* ``--on-no-match``: if a future experiment run can't be matched with
  a historical experiment run, an attempt is made to pick another,
  random, historical run from the same model. The matches are
  usually made using the attributes of the dataset, in particular the
  ``parent_experiment_rip`` attribute, and otherwise an attempt is
  made to match the ``rip`` parameters themselves.

  Other values are ``error`` (the script will exit with an error),
  ``remove`` will remove the experiment run, ``random`` is more broad
  than ``randomrun`` and will ignore the initialization and physics
  parameters when picking a random match.

* ``--norm-by``: normalise the runs per run, or per experiment, or
  even per model. These options change from a "tight" normalisation to
  a very "broad" normalisation.

* ``--reference-period``: which period to normalise the (matched) runs
  to. The default is 1981 -- 2010, which is the reference period used
  with CMIP5 data in Lenderink et al, 2014. The example above has a
  reference period used for CMIP6 data. Note that years are inclusive,
  and run from January 1 to December 12, thus each reference period is
  exactly 30 years.


The output is a CSV file, which looks somewhat as follows::

    date,mean,5,10,25,50,75,90,95
    1950-01-01,-0.740,-0.902,-0.893,-0.861,-0.816,-0.657,-0.432,-0.429
    1951-01-01,-0.754,-1.089,-1.080,-0.911,-0.820,-0.633,-0.259,-0.256
    1952-01-01,-0.806,-1.098,-1.089,-0.928,-0.839,-0.624,-0.437,-0.434
    1953-01-01,-0.806,-1.099,-1.094,-1.045,-0.723,-0.706,-0.403,-0.400
    1954-01-01,-0.765,-1.099,-1.094,-1.069,-0.640,-0.569,-0.364,-0.361
    ....

(Numbers are truncated to just three decimal digits for display
purposes.)


This CSV file is input for the plot below, and for step 1b.


Plot the tas change
~~~~~~~~~~~~~~~~~~~

To create a plot of the temperature change, use the following
command:

.. code-block:: bash

    python -m kcs.tas_change.plot  tas_change.csv cmip6.png \
        --xrange 1950 2100 --ylabel 'Temperature change [${}^{\circ}$]' \
        --title 'Global year temperature change'  --smooth 7 --yrange -1 6

The module has two required arguments: the CSV file calculated above,
and an output figure file name (its extension will determine the file
type automatically). The meaning of most options will be evident. It
is possible to use some LaTeX in the various label and title arguments
(see the Matplotlib documentation for details).

The ``--smooth`` parameters calculates a rolling window average over
the data, and should be an integer. In the above example, a rolling
average is calculated with a seven-year window.


Step 1b: matching the model of interest with the CMIP tas change
----------------------------------------------------------------

This step takes the result of step 1a, and matches the global CMIP
``tas`` change with the global ``tas`` change of our model of
interest, for relevant epochs. The user picks one or more epochs, and
percentiles, and the procedure will match the CMIP change in ``tas``
with an identical change in ``tas`` for the model of interest, which
results in a specific year, calculated over a 30-year period. These
define the scenarios: high and low temperature change (90 and 10
percentile CMIP change) for middle and end of centeury (2050 and 2085;
2085, because the 30-year period average ranges from 2070 to 2099, the
end of the CMIP data).

The output is a CSV file, which contains the so-called steering
table. This table contains the matching period in our model of
interest, the actual temperature change (with respect to the reference
period), and a possible correction factor, in case the model of
interest can't match the CMIP temperature change in the CMIP time
range (for example, the EC-EARTH model can't match the 90 percentile
temperature change for the 2070-2099 period: it is too cool for that,
and doesn't run beyond 2100 to allow it to increase its temperature).

Note that individual runs in the model of interest are averaged. These
should, therefore, be runs of the same experiment, and preferably just
be different realizations of the same model-experiment.

.. code-block:: bash

   $ python -m kcs.steering  tas_change.csv  @ecearth-tas-global-averaged.list \
       --scenario G 2050 10 --scenario W 2050 90 --scenario G 2085 10 --scenario W 2085 90 \
       --rolling-mean 10 --outfile steering.csv

The module takes two mandatory input files: the CMIP CSV file with the
tas change computed previously, and a list of globally-averaged tas
data of the model of interest, EC-EARTH.

The ``--scenario`` options set the various scenarios of interest. The
option can be repeated, and takes three values: a name, an epoch and a
percentile. Be aware that the percentile to match should also be
present in the ``tas_change.csv``.

Here, we also calculated a rolling mean over the CMIP input data
before the EC-EARTH data are matched, to smooth out any bumps in the
distribution.

Finally, the output steering table is written to ``steering.csv`` with
the ``--outfile`` option (otherwise it will only be printed to the
standard output), and looks as follows::

    name,epoch,percentile,cmip_delta_t,period,model_delta_t,factor
    G,2050,10,1.078,"(2021, 2050)",1.104,0.976
    W,2050,90,2.276,"(2049, 2078)",2.289,0.994
    G,2085,10,1.327,"(2027, 2056)",1.330,0.998
    W,2085,90,4.662,"(2070, 2099)",3.290,1.417

(Numbers are truncated to just three decimal digits for display
purposes.)

There are ``cmip_delta_t`` and and ``model_delta_t`` columns. The
first gives the change in global ``tas`` between the reference period
and the epoch period (the 30-year period around the epoch), the second
gives the change in global ``tas`` between the reference period and
the ``period`` for the model of interest. These are usually
near-identical (they will not be exactly the same, since dates are
rounded to years), which shows in the ``factor`` column being close to
one. Notice, however, how they are quite different in the last row:
this is because the model of interest reached the end of the valid
period for future experiments, the year 2099. As a result, the
correction factor is significantly different from ``1.0``, ``1.42``
here.


Plot the model of interest over the CMIP data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To plot the CMIP data (as above) with the model of interest, and
indicate the values for which the scenarios are calculated (that is,
the epoch-percentile points), you can use the following command:

.. code-block:: bash

    python -m kcs.steering.plot tas_change.csv steering.csv cmip-ecearth-scenarios.png \
        --ecearth-data @ecearth-tas-global.list --reference-epoch 1995 \
        --ylabel 'Temperature increase [${}^{\circ}$]'  --smooth 10

This command has three mandatory arguments:

* The table with the tas change percentiles versus years (as before),
  in CSV format.

* The steering table, in CSV format.

* An output figure file name (the extension will automatically
  determines the file type).

The ``--smooth`` option is as before for other commands: it applies a
rolling average, in this case with a window of 10 (years).

The ``-reference-epoch <year>`` will plot a marker for the reference
epoch, which is taken to be the centre of the reference period used
before.


Step 2: calculating the variable changes for CMIP
-------------------------------------------------

This step calculates the changes in the CMIP data between a reference
period and period of interest (the usual 2050 and 2085 epochs, or 2035
-- 2064 and 2070 - 2099 periods). This is done for several seasons and
variables: the examples below calculate it for winter and summer, and
for ``tas`` and ``pr``. The area under consideration is a local area,
in this particular case a single point in the south-west of the
Netherlands (``nlpoint``): this step looks at the local changes, in
contrast to the global temperature change above.

The following four commands calculate the changes:

.. code-block:: bash

    python -m kcs.change_perc @cmip-pr-nlpoint-averaged.list --season djf \
        --period 2035 2064 --relative --csvfile pr_change_2050_djf_nlpoint.csv -v

    python -m kcs.change_perc @cmip-pr-nlpoint-averaged.list --season jja \
        --period 2035 2064 --relative --csvfile pr_change_2050_jja_nlpoint.csv -v

    python -m kcs.change_perc @cmip-tas-nlpoint-averaged.list --season djf \
        --period 2035 2064 --csvfile tas_change_2050_djf_nlpoint.csv -v

    python -m kcs.change_perc @cmip-tas-nlpoint-averaged.list --season jja \
        --period 2035 2064 --csvfile tas_change_2050_jja_nlpoint.csv -v


(The 2085 scenarios are identical except for the ``--period 2035 2064`` option.)

Notes:

* The commands can only function one season, and one variable, at a
  time. To speed things up, one can run these commands in parallel,
  simply putting them in the background in the shell when run.

* There is a mandatory input list of files, but the ``--season`` and
  ``--period`` options are also required.

* The ``pr`` variable requires a ``--relatve`` flag, to indicate that
  we want to calculate the *relative* change in precipitation (for
  ``tas``, we calculate the absolute change).

* The ``--csvfile`` option will write a CSV file, which can be used as
  input for plotting the changes later on. An example file is given
  below.

* As with the global ``tas`` calculation, historical and future
  experiments are matched first. The default settings should be fine,
  but there are a few options that allow changing how this is done, as
  for ``kcs.tas_change``. Please use the ``--help`` to examine those
  if felt necessary.


The output of the calculation is a table that contains the
distribution of the changes in the CMIP distribution. As a result, the
CSV file contains both a mean and percentiles along both axes::

    ,mean,5,10,25,50,75,90,95
    mean,-5.672,-25.525,-21.824,-13.071,-5.191,2.562,8.720,13.544
    5,-15.337,-66.890,-50.293,-31.471,-13.756,0.804,15.039,23.959
    10,-14.055,-54.061,-44.544,-28.308,-11.878,1.369,12.420,18.332
    25,-10.790,-44.817,-36.995,-22.955,-7.886,0.213,12.677,16.739
    50,-7.552,-35.312,-28.169,-16.606,-7.088,3.231,9.817,14.888
    75,-3.874,-21.315,-18.101,-11.165,-3.317,2.962,8.665,15.999
    90,-1.079,-17.224,-12.760,-7.146,-1.547,5.953,11.574,14.656
    95,-0.421,-17.170,-14.149,-7.606,-0.380,7.175,14.238,18.228

(Numbers are truncated to just three decimal digits for display
purposes.)


Plotting the CMIP changes
~~~~~~~~~~~~~~~~~~~~~~~~~

The above output files can be plotted with the following command:

.. code-block:: bash

    python -m kcs.change_perc.plot pr_change_2050_jja_nlpoint.csv pr_change_2050_jja_nlpoint.png \
      --epoch 2050 --text 'precip, JJA', --ytitle 'Change (%)' --ylimits -60 45

There are two required arguments: the CSV input file, and the figure
output file. The other options are nearly all for annotations
(``--epoch`` prints an epoch indicator on the graph) or the axes
limits.


Calculating and overplotting the model of interest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The same calculation can be applied for the model of interest. In the example case, the data were already matched and concatenated between historical and future experiments; this is why there is a ``--no-matching`` option given.

    python -m kcs.change_perc @ecearth-pr-nlpoint-averaged.list --season jja \
        --period 2049 2078 --relative --run-changes pr_change_2050W_jja_nlpoint_ecearth.csv \
        --no-matching

There are a few notable differences here compared to the calculation for the CMIP data:

* The ``--no-matching`` option has been explained above

* The ``--period`` is taken from the steering table, since we are
  dealing with the matched EC-EARTH data in this case.

* The ``--csvfile`` option is missing: we are not writing the
  distribution of the (CMIP) data distribution.

* Instead, we write the (percentile) changes for each individual run
  to a file, with the ``--run-changes`` option.

The resulting ``pr_change_2050W_jja_nlpoint_ecearth.csv`` looks as follows::

    ensemble,ref-mean,fut-mean,ref-10,fut-10,ref-50,fut-50,ref-90,fut-90,mean,10,50,90
    r1i1p1,3.679-05,3.209-05,1.966-05,1.498-05,3.472-05,3.097-05,5.298-05,5.019-05,-12.794,-23.809,-10.812,-5.279
    r2i1p1,3.541-05,3.271-05,1.976-05,1.786-05,3.491-05,3.132-05,5.316-05,4.794-05,-7.623,-9.616,-10.269,-9.816
    r3i1p1,3.611-05,3.282-05,2.162-05,1.297-05,3.526-05,3.011-05,5.397-05,5.524-05,-9.102,-40.012,-14.606,2.354
    ...

(Numbers are truncated as usual for display purposes. The number of displayed rows and columns is also limited: not all default percentiles are shown, just 10, 50 and 90.)

For each realization, there is a row. The rows contains the
percentiles for the reference (control) period, the future period
(2049 -- 2078 here), and their differences (the latter columns are
simply called ``mean``, ``5``, ``10``, ...). It is the latter we are
interested in.

Given the above output, we can replot the CMIP distribution, but now
overplot the individual EC-EARTH runs:

.. code-block:: bash

    python -m kcs.change_perc.plot pr_change_2050_jja_nlpoint.csv pr_change_2050_jja_nlpoint.png \
        --epoch 2050 --text 'precip, DJF', --ytitle 'Change (%)' --ylimits -60 45 \
        --scenario-run G pr_change_G2050_jja_nlpoint_ecearth.csv \
        --scenario-run W pr_change_W2050_jja_nlpoint_ecearth.csv


Step 3: resample the runs of the model of interest
--------------------------------------------------

This step forms the core of the kcs module: it resamples the input
runs (realizations) of the model of interest (EC-EARTH in our case),
and resamples these in three stages, to try and match them with the
CMIP distribution for ``tas`` and ``pr`` calculated above. Note that
the CMIP data is not actually input: the matching is eye-balled by
overplotting the resampled data.

At the moment, the three stages are all calculated in one function,
and are unfortunately not separable. There are, of course, options are
available for each individual stage to set parameters.

Below, the individual stages are described. Below that, the command is
given to run one resample.


Stage 1: match prescribed precipitation scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The four scenarios in the steering table (a warm "W" and moderate "G"
(gematigd) scenario, for two epochs) is extended by adding
precipitation requirements, in the form of percentage increase per
degree of global temperature increase. In this particular case, we
chose 4% and 8% increase per degree. This double the steering table,
which has now obtained an extra column with the actual (temperature
times percentage) precipitation.

The actual input runs are resampled by dividing the relevant period
(from the steering table, for a specific scenario) into five-year
intervals, which gives 6 segments per run, across 16 runs in our
specific case. This yields :math:`16^6` possible combinations. This is
done for both the future period of interest, and the control
(reference) period.

The list of combinations is limited to a 1000 (configurable)
combinations which have a precipitation change closest to the
target. The changes here are calculated using the means of the
individual five-year segments. For the control period, the target is
the mean of the individual runs. That is, all realizations are
averaged over the 30-year control period, and this mean is targeted by
the resampled control period.


Stage 2: limit the allowed percentile ranges
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Given the distribution of the resampled combinations, calculated using
the means of the five-year segments, the combinations are limited
further by selecting resamples that have values within a certain
percentile range.

There are three settings that are applied one after the other: one for
the precipitation range in summer, one for the range of temperature
change in winter and one ofr the range of temperature change in
summer.

The resulting number of valid combinations is about 50, which are
passed on to stage 3.

All the percentile ranges are configurable for each scenario. The input is supplied through a configuration file in the TOML format, and looks similar to this::

    [[W.H.2085]]
    var = "pr"
    season = "jja"
    control = [60.0, 100.0]
    future = [0.0, 40.0]
    [[W.H.2085]]
    var = "tas"
    season = "djf"
    control = [20.0, 50.0]
    future = [50.0, 80.0]
    [[W.H.2085]]
    var = "tas"
    season = "jja"
    control = [10.0, 50.0]
    future = [60.0, 100.0]


    [[W.H.2050]]
    var = "pr"
    season = "jja"
    control = [70.0, 100.0]
    future = [0.0, 40.0]
    [[W.H.2050]]
    var = "tas"
    season = "djf"
    control = [10.0, 40.0]
    future = [60.0, 90.0]
    [[W.H.2050]]
    var = "tas"
    season = "jja"
    control = [10.0, 50.0]
    future = [60.0, 100.0]

    ...

The syntax is hopefully obvious: this uses a double bracket (an array
in TOML), with the name following the tempearture scenario G/W, then
the precipitation scenario H/L, then the epoch (all
dot-separated). The array has 3 items, since there are three "sub"
scenarios where the percentile restrictions are specified. The
specific values required are the variable of interest, the season, the
control percentile range, and the future percentile range. The latter
two are 2-element list of floating point numbers.

Comments and empty lines are optional.

In total, there would be eight scenarios: ``W.H.2085``, ``W.H.2050``,
``W.L.2085``, ``W.L.2050``, ``G.H.2085``, ``G.H.2050``, ``G.L.2085``,
``G.L.2050``. Each scenario has three items, and each item requires
four key-value pairs to be defined.


Stage 3: random selection of a set of resamples, with limited re-use of input segments
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Out of the 50 combinations, we select 8 (or any reasonable number)
resamples. These are selected randomly, and this is done 10.000
(configurable) times. Out of these 10.000 sets of resamples, we
provide penalties for re-use of the same segment in the resample runs:
triple use yields a penalty of 1, four times a penalty of 4, anything
more is thrown out automatically. Penalties are additive, since
multiple segments may occur multiple times. These penalties are all
configurable, with a small TOML configuration file. This file looks as
follows::

    [penalties]
    # Penalties for number of (multiple) occurrences of run-segment in resamples.
    # Starts from 1 occurrence, that is, no duplicate.
    # Only give the number of occurrences that have a penalty less than
    # infinity, including a 0.0 penalty (for e.g. a single, `1`,
    # occurrence).
    # All penalties should be floating point numbers.
    # Has the form `n-occurrences` = `penalty-value`.
    1 = 0.0
    2 = 0.0
    3 = 1.0
    4 = 5.0
    # 5 and more yield an infinite penalty value

The comments are not required, but this makes the configuration file
hopefully self-documenting.



Running the resampling module
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The actual command is:

.. code-block:: bash

    python -m kcs.resample @ecearth-all-nlpoint.list --steering steering.csv \
        --ranges ranges.toml --precip-scenario L 4 --precip-scenario H 8 --relative pr \
        --penalties penalties.toml -v

Points to notice:

* The precipitation scenarios need to be given explicitly, since the
  steering stable does not contain that information.

* The ``--relative`` option takes a *list* of short variable names
  that should be calculated as a relative change; in this case only
  ``pr``.

* The variables are deduced from the input, which here contains both
  ``pr`` and ``tas`` datasets. In fact, the
  ``ecearth-all-nlpoint.list`` file looks like::

    @ecearth-tas-nlpoint.list
    @ecearth-pr-nlpoint.list

  That is, it is a nested @-list.

* The list of files (in this case an @-list), the ``--steering``,
  ``--ranges`` and ``--penalties`` options are required. And you'll
  need to specify at least one ``--precip-scenario``. The last option
  takes two arguments: a name, and a value, indicating the percentage
  change in precipitation per degree temperature change, as before.

* It is also possible to run just one (or a few) scenario(s) from the
  steering table. In that case, use the ``--scenario`` option, which
  takes three arguments: the global scenario name (G/W), the epoch
  (2050/2085) and the precipitation scenario (which you give with the
  ``--precip-scenario``). This option can also be used multiple times.

  Without this option, all scenarios are calculated in succession.

The last option makes it possible to run all eight variants in
parallel. For example, one can create a bash script that contains
eight copies of the command at the start of this section, each put in
the background, and each with a different ``--scenario [G,W]
[2050,2085] [L,H]`` option added.

A single scenario calculation takes up to fifteen minutes, depending
on the number of input runs (sixteen runs in the fifteen minute case);
while the actual calculation doesn't take too long, reading the
datasets, extracting seasons and calculating the means adds to the
total running time.


Output
~~~~~~

The output of the resampling is written to a HDF 5 file. Two, in fact.

One contains the data for the resampled data: the aveages and standard
deviations, and the percentile changes between (resampled) control and
(resampled) future. It is the latter we are normally interested in,
since this fits with the previous (CMIP) calculations.

The structure of this file (by default named ``resamples.h5``) looks
as follows::

    /2050/G/H/pr/djf/diff    Dataset {12, 8}
    /2050/G/H/pr/djf/keys    Dataset {8}
    /2050/G/H/pr/djf/mean    Dataset {8}
    /2050/G/H/pr/djf/std     Dataset {8}
    /2050/G/H/pr/jja         Group
    /2050/G/H/pr/jja/diff    Dataset {12, 8}
    /2050/G/H/pr/jja/keys    Dataset {8}
    /2050/G/H/pr/jja/mean    Dataset {8}
    /2050/G/H/pr/jja/std     Dataset {8}
    /2050/G/H/pr/mam         Group
    /2050/G/H/pr/mam/diff    Dataset {12, 8}
    ...

Thus, for each epoch, temperature and precipitation scenario, each
variable, and each season (there is also "son") there are four
datasets: ``diff``, ``keys``, ``mean``, ``std``. The size of the
datasets in this example is 8, because there were eight statistics
calculated: the mean and the 5, 10, 25, 50, 75, 90 and 95
percentiles. The latter can be found in the ``keys`` dataset.

The ``diff`` datasets is 12 by 8, where the first dimension equals the
number of requested resampled runs (the ``--nstep3`` option, here
12). The second dimension are the statistics again.

The file structure can be examined with the command ``h5ls --recursive
resamples.h5`` (which yields the above listing), while a quick look at
the data can be obtained with the ``h5dump`` command, for example:

.. code-block:: bash

   h5dump --dataset /2050/G/H/pr/mam/diff resamples.h5

   (0,0): -4.86969, 2.98059, -8.86912, -12.488, -5.37732, -5.38094, -6.44242,
   (0,7): -1.30814,
   (1,0): 3.95456, 10.5316, 7.51921, 7.81239, 8.34815, -2.34305, 6.44583,
   (1,7): -4.05653,
   (2,0): 7.42924, 13.0411, 7.01548, 15.2165, -1.90975, 8.9375, 14.6143,
   (2,7): 14.7941,
   (3,0): 4.56155, 3.41901, 10.5502, 15.5352, 8.15393, 4.33509, -5.11064,
   (3,7): -5.99621,
   (4,0): 2.9189, -26.567, -3.01042, -5.91511, 5.42413, 2.54435, 8.93456,
   (4,7): 3.50783,
   (5,0): -1.60492, -8.12653, 2.79788, -9.45805, 2.84853, -2.63126,
   (5,6): -0.277906, 4.41441,
   (6,0): -0.216957, -32.8458, -13.7598, -2.39312, 5.25471, 4.29372, 4.76788,
   (6,7): 5.07935,
   (7,0): 6.55494, 42.2289, 6.38604, 4.6321, 1.54924, 6.47195, 2.73307,
   (7,7): 10.9296,
   (8,0): 6.70868, 26.3777, 5.348, 17.1805, 18.6534, 0.712798, 0.348545,
   (8,7): -5.1629,
   (9,0): 5.97353, 30.6496, 50.8518, 18.0891, 0.961236, 4.07857, 2.55165,
   (9,7): 2.51619,
   (10,0): 1.59531, 4.25298, 2.49238, -0.913777, -6.77163, 3.00944, 6.46021,
   (10,7): 2.06491,
   (11,0): 5.87735, 24.6462, 18.1432, 13.357, 4.82466, 5.44672, 1.35844,
   (11,7): -1.77472

The ``(x, y)`` are part of the ``h5dump`` output, and indicate the
dataset coordinates (indices).  For each of the twelve resampled runs
(row-wise), there are eight statistics (column-wise), the ones
mentioned above. The values are the (relative, since ``pr`` was used)
differences between the control and future period.


The other HDF 5 file is called ``indices.h5``, and it specifies the
*indices* for the original runs and segments to obtain the resampled
runs. This file looks as follows::

    /                        Group
    /2050                    Group
    /2050/G                  Group
    /2050/G/H                Group
    /2050/G/H/control        Dataset {8, 6}
    /2050/G/H/future         Dataset {8, 6}
    /2050/G/L                Group
    /2050/G/L/control        Dataset {8, 6}
    /2050/G/L/future         Dataset {8, 6}
    ...

It has a control and future dataset for each scenario. Each dataset is
two dimensional: the first axis is for the number of output runs,
while the second is for the number of segments (30 years / 5 years = 6
segments, in our case). The values in the dataset are all positive
integers, varying between 0 and the number of *input* runs minus 1
(minus 1, because Python indexes from 0 to n-1). Thus, for a set of
indices as follows::

     (0,0): 0, 4, 1, 6, 1, 3,
     (1,0): 2, 2, 2, 4, 3, 0,
     (2,0): 6, 1, 7, 0, 2, 4,
     (3,0): 7, 6, 7, 3, 0, 3,
     (4,0): 2, 4, 5, 7, 6, 6,
     (5,0): 6, 5, 6, 5, 7, 6,
     (6,0): 7, 1, 2, 6, 7, 4,
     (7,0): 0, 6, 3, 4, 3, 3

Resampled run number 0 would use the first five-year segment of input
run number 0, its second five-year segment from input run number 4,
its third five-year segment from input run number 1, and so on. Note
that the five-year segments match one-on-one: the third segment in the
input is also always a third segment in any output.

This also shows which runs have double (or triple) uses in a paricular
5-year segment, since this would result in the same index repeated in
a column. Input runs 6 and 7 seem to be popular, with several repeats
in several columns. Thanks to the penalty system, no triple or more
repeats occur. Note that repeats across a row do not matter: even if a
row contains all the same indices, that just means that the original
input run was a perfect match for all the conditions we threw at
it.

This indices file can be particularly helpful for the downsampling: it
indicates which parts of our input runs have been used to create our
resampled runs, but that also means it indicates which part of our
*regional* model runs we have to use (and resample accordingly);
provided our regional model input runs match one-to-one with our
global model-of-interest runs.

Finally, there will also be numerous CSV output files, named something like ``resampled_<epoch>_<G/W>_<H/L>_<var>_<season>.csv``. These are similar to the ``pr_change_W2050_jja_nlpoint_ecearth.csv`` files mentioned further above: they contain, for each resampled run, the necessary statistics, and are in fact identical to the ``diff`` datasets in the HDF 5 file. For example, ``resampled_2050_G_H_pr_mam.csv`` looks as follows::

    mean,5,10,25,50,75,90,95
    -4.869,2.980,-8.869,-12.487,-5.377,-5.380,-6.442,-1.308
    3.954,10.531,7.519,7.812,8.348,-2.343,6.445,-4.056
    7.429,13.041,7.015,15.216,-1.909,8.937,14.614,14.794
    4.561,3.419,10.550,15.535,8.153,4.335,-5.110,-5.996
    2.918,-26.567,-3.010,-5.915,5.424,2.544,8.934,3.507
    -1.604,-8.126,2.797,-9.458,2.848,-2.631,-0.277,4.414
    -0.216,-32.845,-13.759,-2.393,5.254,4.293,4.767,5.079
    6.554,42.228,6.386,4.632,1.549,6.471,2.733,10.929
    6.708,26.377,5.348,17.180,18.653,0.712,0.348,-5.162
    5.973,30.649,50.851,18.089,0.961,4.078,2.551,2.516
    1.595,4.252,2.492,-0.913,-6.771,3.009,6.460,2.064
    5.877,24.646,18.143,13.357,4.824,5.446,1.358,-1.774

(Again truncated to three decimal digits.) Compare this to the
``h5dump`` output.

These CSV files can be used when plotting the distribution of a
variable change, exactly the same as before.


Plotting the resampled runs
~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can overplot the resampled runs on top of the CMIP data as follows,
using the same plotting command from before:

.. code-block:: bash

    python -m kcs.change_perc.plot pr_change_2050_jja_nlpoint.csv pr_change_2050_jja_nlpoint.png \
        --epoch 2050 --text 'precip, JJA', --ytitle 'Change (%)' --ylimits -60 45 \
        --scenario-run G_H resampled_2050_G_H_pr_jja.csv \
        --scenario-run W_H resampled_2050_W_H_pr_jja.csv \
        --scenario-run G_L resampled_2050_G_L_pr_jja.csv \
        --scenario-run W_L resampled_2050_W_L_pr_jja.csv

Note that the input file names have changed: we now have precipitation
scenarios, but we have lost information of the area we used. So if
different areas are to be considered, a simple solution would be to
use different subdirectories.

The resulting plot (``pr_change_2050_jja_nlpoint.png``) will show all
the individual resampled runs. That can be used as a measure to see
how well the resampled runs cover the CMIP data, and how close they
are to their average. If you don't want to plot the individual runs,
use the ``--only-scenario-mean`` option:

.. code-block:: bash

    python -m kcs.change_perc.plot pr_change_2050_jja_nlpoint.csv pr_change_2050_jja_nlpoint.png \
        --epoch 2050 --text 'precip, JJA', --ytitle 'Change (%)' --ylimits -60 45 \
        --scenario-run G_H resampled_2050_G_H_pr_jja.csv \
        --scenario-run W_H resampled_2050_W_H_pr_jja.csv \
        --scenario-run G_L resampled_2050_G_L_pr_jja.csv \
        --scenario-run W_L resampled_2050_W_L_pr_jja.csv
        --only-scenario-mean
