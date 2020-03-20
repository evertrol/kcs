#! /bin/bash -v

set -e  # Exit this script if a child process fails



# STEP 0: extraction

# Output directory of extracted files
# Note: CSV files, plots and HDF 5 output files are not put here, but
# in the current working directory.
WORK_DATA_DIR=data

# Make sure the input files (cmip6-tas.list, cmip6-pr.list,
# ecearth-tas.list and ecearth-pr.list here) are created properly in
# advance.

# Three extraction steps are run simultaneous, to speed things up a
# bit, but without taxing the CPUs, system memory and file I/O too
# much (note the wait statement).
python -m kcs.extraction --area global @cmip6-tas.list \
    --template "$WORK_DATA_DIR/cmip6/{var}-{area}-averaged/{filename}.nc" \
    --ignore-common-warnings -v  &
python -m kcs.extraction --area nlpoint @cmip6-tas.list \
    --template "$WORK_DATA_DIR/cmip6/{var}-{area}-averaged/{filename}.nc" \
    --ignore-common-warnings -v  &
python -m kcs.extraction --area nlpoint @cmip6-pr.list \
    --template "$WORK_DATA_DIR/cmip6/{var}-{area}-averaged/{filename}.nc" \
    --ignore-common-warnings -v  &
wait

python -m kcs.extraction --area global @ecearth-tas.list \
    --template "$WORK_DATA_DIR/ecearth/{var}-{area}-averaged/{filename}.nc" \
    --ignore-common-warnings -v  &
python -m kcs.extraction --area nlpoint @ecearth-tas.list \
    --template "$WORK_DATA_DIR/ecearth/{var}-{area}-averaged/{filename}.nc" \
    --ignore-common-warnings -v  &
python -m kcs.extraction --area nlpoint @ecearth-pr.list \
    --template "$WORK_DATA_DIR/ecearth/{var}-{area}-averaged/{filename}.nc" \
    --ignore-common-warnings -v  &

# We'll always need to wait at the end; otherwise, we'll be running our analysis on incomplete data
wait



# STEP 1a: calculate the change in annual temperature

# Using find instead of ls to obtain full (relative) paths
find $WORK_DATA_DIR/cmip6/tas-global-averaged -name "*.nc" > cmip6-tas-global-averaged.list

python -m kcs.tas_change  @cmip6-tas-global-averaged.list \
       --outfile=tas_change_cmip6.csv  --reference-period 1991 2020 -v


# STEP 1b: calculate the steering table

find $WORK_DATA_DIR/ecearth/tas-global-averaged -name "*.nc" > ecearth-tas-global-averaged.list

python  -m kcs.steering  tas_change_cmip6.csv  @ecearth-tas-global-averaged.list \
    --scenario G 2050 10 --scenario W 2050 90 --scenario G 2085 10 --scenario W 2085 90 \
    --rolling-mean 10 --outfile steering.csv

# Plot the CMIP data, with indicators of the various scenarios and the
# (averaged) EC-EARTH data overplotted
python -m kcs.steering.plot tas_change_cmip6.csv steering.csv \
        --outfile cmip6-ecearth-scenarios.png \
		--extra-data @ecearth-tas-global-averaged.list --reference-epoch 2005 \
		--ylabel 'Temperature increase [${}^{\circ}C$]'  --smooth 10 \
        --grid --legend



for epoch in 2050 2085
do
		# Create a separate steering table for this epoch from the main steering table file
		fname="steering${epoch}.csv"
		head -n1 steering.csv > $fname
		grep ",${epoch}," steering.csv >> $fname

        for var in tas pr
        do
                listname1="@cmip6-${var}-nlpoint-averaged.list"
                listname2="@ecearth-${var}-nlpoint-averaged.list"
                for season in djf jja mam son
                do
                        python -m kcs.change_perc.runall $listname1 --season $season  --steering $fname --runs $listname2 --relative pr --no-matching -v --plottype png --write-csv  &
                done
				# Run four processes at a time.
				wait
        done
done


for epoch in 2050 2085
do
		for scenario in G W
		do
				for precip in L H
				do
						python -m kcs.resample  @ecearth-all-nlpoint-averaged.list  --steering steering.csv --relative pr -vvv  --precip-scenario L 4 --precip-scenario H 8 --scenario $scenario $epoch $precip --indices-out indices_${scenario}_${epoch}_${precip}.h5 --resamples-out resamples_${scenario}_${epoch}_${precip}.h5 &
				done
		done
		wait  # run a maximum of four calculations at the same time
done


for epoch in 2050 2085
do
		for var in pr tas
		do
				relative=""
				ylimits="--ylimits 0 7"
				if [ "$epoch" == "2050" ]
				then
						ylimits="--ylimits 0 4.5"
				fi
				ytitle="--ytitle 'Change [K]'"
				if [ "$var" == "pr" ]
				then
						relative="--relative"
						ylimits="--ylimits -60 45"
						ytitle="--ytitle 'Change (%)'"
				fi
				for season in djf mam jja son
				do
						# ${string^^} changes the string to upper case
						# This requires Bash 4 (which is not default on MacOS)!
						python -m kcs.change_perc.plot ${var}_${epoch}_${season}_perc_distr.csv \
							   ${var}_change_${epoch}_${season}_nlpoint.png \
							   --epoch ${epoch} --text "${var}, ${season^^}" \
							   --ytitle 'Change (%)' $ylimits \
							   --scenario-run G_L resampled_${epoch}_G_L_${var}_${season}.csv \
							   --scenario-run W_L resampled_${epoch}_W_L_${var}_${season}.csv \
							   --scenario-run G_H resampled_${epoch}_G_H_${var}_${season}.csv \
							   --scenario-run W_H resampled_${epoch}_W_H_${var}_${season}.csv \
							   --only-scenario-mean
				done
		done

done




# Finish the bash script only when all child processes have finished.
wait
