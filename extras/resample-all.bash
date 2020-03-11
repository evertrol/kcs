#! /bin/bash

# Calculate the resampling for the input runs
# Change option values accordingly (e.g., nstep3) before running.

for scenario in G W
do
		for epoch in 2050 2085
		do
				for precip in L H
				do
						echo "python -m kcs.resample  @ecearth-all-nlpoint.list --ranges step2ranges.toml --steering steering.csv --penalties penalties.toml --relative pr -vv --nstep3 12  --precip-scenario L 4 --precip-scenario H 8 --scenario $scenario $epoch $precip --indices-out indices_${scenario}_${epoch}_${precip}.h5 --resamples-out resamples_${scenario}_${epoch}_${precip}.h5 &"
						python -m kcs.resample  @ecearth-all-nlpoint.list --ranges step2ranges.toml --steering steering.csv --penalties penalties.toml --relative pr -vv --nstep3 12  --precip-scenario L 4 --precip-scenario H 8 --scenario $scenario $epoch $precip --indices-out indices_${scenario}_${epoch}_${precip}.h5 --resamples-out resamples_${scenario}_${epoch}_${precip}.h5 &
				done
				# Run only two processes at a time.
				# wait
		done
		# Run only four processes at a time.
		wait
done
# Finish the bash script only when all child processes have finished.
wait

