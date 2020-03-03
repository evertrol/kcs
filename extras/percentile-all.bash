#! /bin/bash

for epoch in 2050 2085
do
        fname="steering${epoch}.csv"
        for var in tas pr
        do
                listname1="@cmip5-${var}-nlpoint-averaged.list"
                listname2="@ecearth-${var}-nlpoint-averaged.list"
                for season in djf jja
                do
                        echo "python -m kcs.change_perc.runall $listname1 --season $season  --steering $fname --runs $listname2 --relative pr --no-matching -v --plottype pdf --write-csv  &"
                        python -m kcs.change_perc.runall $listname1 --season $season  --steering $fname --runs $listname2 --relative pr --no-matching -v --plottype pdf --write-csv  &
                done
        done
		# Run four processes at a time, if there are only four cores.
		#wait
done
# Finish the bash script only when all child processes have finished.
wait
