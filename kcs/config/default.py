config_toml_string = r'''

# Notes about values in the TOML config file:
# - strings should always be quoted. Multi-line strings are possible
#   using triple-quotes, but should not be necessary in this config file
# - floating point values require a decimal dot, or an exponent (or
#   both). Otherwise, they are interpreted as integers.
# More information about the TOML config format: https://github.com/toml-lang/toml



[scenario]

[scenario.defs]
# Note: extra definitions for a name or precip do not have to be commented out below, even if the corresponding
# scenarios are commented-out above

# percentile of CMIP tas for a given epoch that the 'W' and 'G' scenarios correspond to
W = 90.0
G = 10.0
# percent change in precipitation (times global tas increase) for 'L' and 'H' scenarios
L = 4.0
H = 8.0


# Scenarios to use.
# This defines only the names and epoch.
# Values (what G/W and L/H correspond to) are filled in below.
# Comment-out scenarios of no-interest.
# Note the use of plural, 'scenarios', here
[[scenario.list]]
name = "G"
epoch = 2050
precip = "L"
[[scenario.list]]
name = "W"
epoch = 2050
precip = "L"
[[scenario.list]]
name = "G"
epoch = 2050
precip = "H"
[[scenario.list]]
name = "W"
epoch = 2050
precip = "H"
[[scenario.list]]
name = "G"
epoch = 2085
precip = "L"
[[scenario.list]]
name = "W"
epoch = 2085
precip = "L"
[[scenario.list]]
name = "G"
epoch = 2085
precip = "H"
[[scenario.list]]
name = "W"
epoch = 2085
precip = "H"


[areas]
# Define areas of interest
# Use w, e, s, n identifiers in a inline-table/map/dict for a rectangular area
# Or lat, lon in a inline-table/map/dict for a single point
# Or the special value "global"
# Values for w/e/s/n/lat/lon should all be floating point.
# Shapefiles and masks are not yet supported.
global = "global"
nlpoint = {lat = 51.25, lon = 6.25}
nlbox = {w = 4.5, e = 8.0, s = 50.5, n = 53.0}
weurbox = {w = 4.0, e = 14.0, s= 47.0, n = 53.0}
rhinebasin = {w = 6, e = 9, n = 52, s = 47}

[variables]
# Some generic configuration regarding variables

# For which variables to calculate a relative change,
# instead of an absolute change (compare 'pr' versus 'tas')
# A list of short variable names.
relative = ["pr"]


[data]
# Some generic configuration for the input data
# This assumes NetCDF files with proper (CF-conventions) attributes

# Define the attribute names for meta information.
# Each definition should be a list: this allows to handle different
# conventions (e.g., between CMIP5 and CMIP6) if one is not available.
experiment = ["experiment_id"]
model = ["model_id", "source_id"]
realization = ["realization"]
initialization = ["initialization_method"]
physics = ["physics_version"]
prip = ["parent_experiment_rip", "parent_variant_label"]
var =  ["variable_id"]

# What is the attribute value that indicates historical experiments?
# Everything else is assumed to be a future experiment.
# This value is case-insensitive.
historical_experiment = "historical"

[data.filenames]
# Definitions of filename patterns, to obtain attribute information from.
# Several are given, for various conventions.
# All are tried, until a match is found.

# Regexes can be notably hard to read, especially in this case, since every
# blackslash needs to be escaped, resulting in lots of double backslashes.

[data.filenames.esmvaltool]
pattern = """^CMIP\\d_\
(?P<model>[-A-Za-z0-9]+)_\
(?P<mip>[A-Za-z]+)_\
(?P<experiment>[A-Za-z0-9]+)_\
r(?P<realization>\\d+)\
i(?P<initialization>\\d+)\
p(?P<physics>\\d+)_\
(?P<var>[a-z]+)_\
.*\\.nc$\
"""

[data.filenames.cmip5]
pattern = """^\
(?P<var>[a-z]+)_\
(?P<mip>[A-Za-z]+)_\
(?P<model>[-A-Za-z0-9]+)_\
(?P<experiment>[A-Za-z0-9]+)_\
r(?P<realization>\\d+)\
i(?P<initialization>\\d+)\
p(?P<physics>\\d+)_\
.*\\.nc$\
"""

[data.filenames.cmip6]
pattern = """^\
(?P<var>[a-z]+)_\
(?P<mip>[A-Za-z]+)_\
(?P<model>[-A-Za-z0-9]+)_\
(?P<experiment>[A-Za-z0-9]+)_\
r(?P<realization>\\d+)\
i(?P<initialization>\\d+)\
p(?P<physics>\\d+)\
f\\d+_\
gn_\
.*\\.nc$\
"""

[data.filenames.ecearth]
pattern = """^\
(?P<var>[a-z]+)_\
(?P<mip>[A-Za-z]+)_\
(?P<model>[-A-Za-z0-9]+)_\
(?P<experiment>[A-Za-z0-9]+)_\
.*\\.nc$\
"""


[cmip]
# Configuration for everything that considers CMIP data


# Actual timespan a given scenario epoch corresponds to
periods = {2050 = [2036, 2065], 2085 = [2071, 2100]}

# Control period defines the reference period to which to compare (and
# possibly normalize) to.
# CMIP5 would be [1981, 2010], CMIP6 would be [1991, 2020]
control_period = [1981, 2010]

[cmip.data]
# List the data files for the different data types.
# This should e.g. filter out bad data files and unwanted experiments.
# Can be globbing patterns or @-lists.
tas_global = "@cmip_tas_global.list"
tas_nlpoint = "@cmip_tas_nlpoint.list"
pr_nlpoint = "@cmip_pr_nlpoint.list"

# Normalize the CMIP data to the control period.
# Choices are "model", "experiment" or "run". These options vary from
# the most to the least spread of normalized data around the control period.
# Leave blank to not normalize (usually a bad idea).
norm_by = "run"

# Calculate the tas change for a specific seasonal average, or a yearly average
# Choices are "year", "djf", "mam", "jja", "son".
season = "year"


[cmip.data.matching]
# Configuration how to match and concatenate CMIP historical and future experiments

# Match future and historical runs by model (very generic) or ensemble (very specific).
match_by = "ensemble"

# Where to get the match info from. Either (NetCDF) 'attributes' or the 'filename' pattern
# Should always be a list: the later options in the list serve as a fallback in case earlier
# options don't succeed
match_info_from = ["attributes", "filename"]

# What to do when a future ensemble can't be matched:
# - "error": raise an error, and stop the program
# - "remove": remove (ignore) the future ensemble
# - "randomrun": pick a random historical run that matches all attributes, except the realization
# - "random": pick a random historical run from all ensembles of that model
on_no_match = "randomrun"


[extra_data]
# Configuration for additional data
# This is the data of interest, for which a steering table will be
# calculated, and whose runs will be resampled.
# This assumes the datasets are already concatenated datasets: historical + future.

data = "@ecearth-tas_global.list"
control_period = [1981, 2010]


[statistics]
# Which statistics (mean and percentiles) to calculate.
# There are two subsets:
# 1. the statistics for the tas change (which leads to the steering table)
# 2. the statistics for the seasonal-regional change plots

[statistics.tas_change]
# A list of floating point numbers (matching between these floating
# point numbers for finding the right percentiles, is done with an
# accuracy of 0.0001 tolerance).
percentiles = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
# Calculate the mean as well
mean = true

[statistics.regional_changes]
percentiles = [5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0]
mean = true


[plotting]
figsize = [8.0, 8.0]

[plotting.tas_increase]
# Configuration for global temperature increase plot

# CMIP percentiles levels to plot
# Each level needs a name, to be re-used in the styline configuration
levels = {extreme = [5.0, 95.0], middle = [10.0, 90.0], narrow = [25.0, 75.0]}

# Smooth the plot across time; value should be an integer. Leave blank for no smoothing.
rolling_window = 10

[plotting.tas_increase.extra_data]
# Overplot extra datasets?
overplot = true
# Plot averaged data, or individual runs
average_data = true
# Smooth with rolling window; same as for CMIP data
rolling_window = 10


[plotting.tas_increase.labels]
x = "Year"
y = "Increase [${}^{\\circ}$]"

[plotting.tas_increase.range]
# Use a 2-element list. Leave blank to let Matplotlib figure things out.
x = [1950, 2100]  # years, in integers
y = [-1.0, 6.0]   # always float

[plotting.tas_increase.styles]
# Re-use the level names above.
# Use Matplotlib color and alpha codes.
# Anything not given is 'black' (color) and 1.0 (alpha; opaque).
colors = {extreme = "#bbbbbb", middle = "#888888", narrow = "#555555", extra_data = "#669955"}
alpha = {extreme = 0.8, middle = 0.4, narrow = 0.2}



[resampling]

nsections = 6
nstep1 = 1000
nstep3 = 8
# Monte-Carlo number of samples
nsample = 10_000

# TOML file that defines the percentiles ranges used in step 2
step2table = "step2.toml"


# Penalties for number of (multiple) occurrences of segment in resamples, in step 3.
# Starts from 1 occurrence, that is, no duplicate.
# Only give the number of occurrences that have a penalty less than
# infinity, including a 0.0 penalty (for e.g. a single, `1`, occurrence).
# All penalties should be floating point numbers.
penalties = {1 = 0.0, 2 = 0.0, 3 = 1.0, 4 = 5.0}

'''

