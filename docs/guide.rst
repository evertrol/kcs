==================================================
A brief guide to the running the kcs functionality
==================================================


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

You pick another name, and you may want to change to a newer Python
version in the future, provided it's supported by the various
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

this is interpreted as::

    python -m kcs.read_files tas_Amon_GISS-E2-R_rcp85_r1i1p1_200601-202512.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_202601-205012.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_205101-207512.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_207601-210012.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_210101-212512.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_212601-215012.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_215101-217512.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_217601-220012.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_220101-222512.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_222601-225012.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_225101-227512.nc tas_Amon_GISS-E2-R_rcp85_r1i1p1_227601-230012.nc

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
stay)::

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

For the global CMIP ``tas`` average, run it as follows::

    python -m kcs.extraction --area global @cmip-tas.list --ignore-common-warnings -v

Iris can be quite chatty regarding potential problems: the
`--ignore-common-warnings` option turns this chattiness down, but it
may be worth to leave this option out a first time, to see that the
notices are indeed not really a problem. The `-v` option is a
verbosity option: used once, it will log warnings (fairly
useful). ``-vv`` will produce some "info" log messages as well, and
``-vvv`` (the maximum level) will produce quite a number of debug
messages along the way.

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
results into a separate directory::

    python -m kcs.extraction --area global  @ecearth-tas.list --template "data/ecearth/{var}-{area}-averaged/{filename}.nc"--ignore-common-warnings -v

Another example, would be if you want separate directories for
e.g. CMIP5 and CMIP6 data::

    python -m kcs.extraction --area global @cmip5-tas.list --template "data/cmip5/{var}-{area}-averaged/{filename}.nc"--ignore-common-warnings -v
    python -m kcs.extraction --area global @cmip6-tas.list --template "data/cmip6/{var}-{area}-averaged/{filename}.nc"--ignore-common-warnings -v


The examples below perform the extraction for the ``nlpoint`` area,
for both ``tas`` and ``pr``, and for both the CMIP data and the
EC-EARTH ("model of interest") data::

    python -m kcs.extraction --area nlpoint @cmip-tas.list --ignore-common-warnings -v
    python -m kcs.extraction --area nlpoint @cmip-pr.list --ignore-common-warnings -v

    python -m kcs.extraction --area nlpoint @ecearth-tas.list --template "data/ecearth/{var}-{area}-averaged/{filename}.nc"--ignore-common-warnings -v
    python -m kcs.extraction --area nlpoint @ecearth-pr.list --template "data/ecearth/{var}-{area}-averaged/{filename}.nc"--ignore-common-warnings -v


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

::

   python -m kcs.tas_change  @cmip-tas-global-averaged.list --outfile=tas_change.csv --on-no-match=randomrun -v  --norm-by=run  --reference-period 1991 2020

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
^^^^^^^^^^^^^^^^^^^

To create a plot of the temperature change, use the following
command::

    python -m kcs.tas_change.plot  tas_change.csv cmip6.png --xrange 1950 2100 --ylabel 'Temperature change [${}^{\circ}$]' --title 'Global year temperature change'  --smooth 7 --yrange -1 6

The module has two required arguments: the CSV file calculated above,
and an output figure file name. The meaning of most options will be
evident. It is possible to use some LaTeX in the various label and
title arguments (see the Matplotlib documentation for details).

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
2085, because the 30-year period average ranges from 2071 to 2100, the
end of the CMIP data).

The output is a CSV file, which contains the so-called steering
table. This table contains the matching period in our model of
interest, the actual temperature change (with respect to the reference
period), and a possible correction factor, in case the model of
interest can't match the CMIP temperature change in the CMIP time
range (for example, the EC-EARTH model can't match the 90 percentile
temperature change for the 2071-2100 period: it is too cool for that,
and doesn't run beyond 2100 to allow it to increase its temperature).

Note that individual runs in the model of interest are averaged. These
should, therefore, be runs of the same experiment, and preferably just
be different realizations of the same model-experiment.

::

   $ python -m kcs.steering  tas_change.csv  @ecearth-tas-global-averaged.list --scenario G 2050 10 --scenario W 2050 90 --scenario G 2085 10 --scenario W 2085 90  --rolling-mean 10 --outfile steering.csv

The module takes two mandatory input files: the CMIP CSV file with the
tas change computed previously, and a list of globally-averaged tas
data of the model of interest, EC-EARTH.

The ``--scenario`` options set the various scenarios of interest. The
option can be repeated, and takes three values: a name, an epoch and a
percentile. Be aware that the percentile to match should also be
present in the ``tas_change.csv``.

Here, we also calculated a rolling mean over the CMIP input data before the EC-EARTH data are matched, to smooth out any bumps in the distribution.

Finally, the output steering table is written to ``steering.csv`` with the ``--outfile`` option (otherwise it will only be printed to the standard output), and looks as follows::

    name,epoch,percentile,cmip_delta_t,period,model_delta_t,factor
    G,2050,10,1.0785846264403356,"(2021, 2050)",1.1043185980838182,0.9766969679871955
    W,2050,90,2.2764075664683068,"(2049, 2078)",2.2897636822278784,0.9941670331033565
    G,2085,10,1.327832066651713,"(2027, 2056)",1.3304594858793557,0.9980251790787104
    W,2085,90,4.662798733237607,"(2070, 2099)",3.290130807845358,1.4172077055778771

There is a cmip_delta and
