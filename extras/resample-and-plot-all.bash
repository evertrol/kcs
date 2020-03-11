#! /bin/bash

# Calculate the resampling for the input runs.
# Also create plots if the four G/W_H/L scenarios have been calculated for an epoch.
# Change option values accordingly (e.g., nstep3) before running.

for epoch in 2050 2085
do
		for scenario in G W
		do
				for precip in L H
				do
						echo "python -m kcs.resample  @ecearth-all-nlpoint-short.list --ranges step2ranges.toml --steering steering.csv --penalties penalties.toml --relative pr -vv --nstep3   --precip-scenario L 4 --precip-scenario H 8 --scenario $scenario $epoch $precip --indices-out indices_${scenario}_${epoch}_${precip}.h5 --resamples-out resamples_${scenario}_${epoch}_${precip}.h5 &"
						python -m kcs.resample  @ecearth-all-nlpoint-short.list --ranges step2ranges.toml --steering steering.csv --penalties penalties.toml --relative pr -vv --nstep3 12  --precip-scenario L 4 --precip-scenario H 8 --scenario $scenario $epoch $precip --indices-out indices_${scenario}_${epoch}_${precip}.h5 --resamples-out resamples_${scenario}_${epoch}_${precip}.h5 &
				done
		done
		# Run only four processes at a time.
		wait

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
						# ${string^^} change the string to upper case
						# This requires Bash 4!
						python -m kcs.change_perc.plot ${var}_${epoch}_${season}_perc_distr.csv \
							   ${var}_change_${epoch}_${season}_nlpoint.png \
							   --epoch ${epoch} --text "${var}, ${season^^}" \
							   --ytitle 'Change (%)' $ylimits \
							   --scenario-run G_H resampled_${epoch}_G_H_${var}_${season}.csv \
							   --scenario-run W_H resampled_${epoch}_W_H_${var}_${season}.csv \
							   --scenario-run G_L resampled_${epoch}_G_L_${var}_${season}.csv \
							   --scenario-run W_L resampled_${epoch}_W_L_${var}_${season}.csv \
							   --only-scenario-mean
				done
		done
done

# Finish the bash script only when all child processes have finished.
wait
