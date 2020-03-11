#! /bin/bash

# This uses the `kcs.change_perc.runall` module, which is a small
# utility that uses a steering table as input to deduce the correct
# periods for calculating the statistics for the EC-EARTH data.

# It will also creates plots of the CMIP distribution with the
# individual EC-EARTH runs overplotted. Output file names are
# generated automatically, following the pattern
# "{var}_{epoch}_{season}.{ext}".


for epoch in 2050 2085
do
		# Create a separate steering table for this epoch from the main steering table file
		fname="steering${epoch}.csv"
		head -n1 steering.csv > $fname
		grep ",${epoch}," steering.csv >> $fname

        for var in tas pr
        do
                listname1="@cmip5-${var}-nlpoint-averaged.list"
                listname2="@ecearth-${var}-nlpoint-averaged.list"
                for season in djf jja mam son
                do
                        echo "python -m kcs.change_perc.runall $listname1 --season $season  --steering $fname --runs $listname2 --relative pr --no-matching -v --plottype pdf --write-csv  &"
                        python -m kcs.change_perc.runall $listname1 --season $season  --steering $fname --runs $listname2 --relative pr --no-matching -v --plottype pdf --write-csv  &
                done
				# Run four processes at a time.
				wait
        done
done
# Finish the bash script only when all child processes have finished.
wait
