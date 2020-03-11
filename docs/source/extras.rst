=====================
Extra example scripts
=====================

In the ``extras/`` directory, there are a few bash example scripts
that show how one might run several processes in parallel (the
``*-all.bash`` files). The scripts contains nested loops over the
epochs, the temperature and precipitation scenarios. The ``wait``
command waits for the child processes (that is, all the backgrounded
processes) to finish before it continues, and putting in one of the
loops limits the number of simultaneous processes. At the end of the
script, it prevents the bash script from exiting until all
backgrounded processes have finished (this way, one could, for
example, time the bash script for an estimate of the total duration:
``time percentiles-all.bash``).


The ``percentiles-all.bash`` also uses a small extra module,
``kcs.change_perc.runall``, which is convenient when calculating all
scenarios from a steering table. The default ``kcs.change_perc``
functionality requires two runs, one for CMIP data and one for the
model of interest, each with different options set, such as
``--period``, ``--run-changes`` and ``--no-matching`` in our case. The
``runall`` module takes care of that, and deduce the relevant periods
from the steering table. Variables are deduced from the input
files. It assumes that the data for the model of interest is already
concatenated. The script does split things up with a nested loop into
epochs, variables and seasons, so that processes can be run
simultaneously.

The ``resample-all.bash`` is very similar to the
``percentiles-all.bash`` script, in that it loops through scenarios
and epochs, running processes in parallel. The
``resample-and-plot.bash`` script goes one step further, as it will
take the outputs from the resampling processes, and creates figures
for the CMIP distribution with the resampled runs overplotted, for
variations of epoch, variable and season.

