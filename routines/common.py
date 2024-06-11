###################################################################################################
#
# ASTR 620 - Galaxies
#
# common.py: This file should be included at the beginning of all class Jupyter notebooks
#
# (c) Benedikt Diemer, University of Maryland
#
###################################################################################################

import matplotlib as mpl
import matplotlib.pyplot as plt

from colossus.cosmology import cosmology

###################################################################################################
# LOCAL VARIABLES -- SHOULD BE EDITED BY USER
###################################################################################################

# Set data dir. This is where all data files used in the notebooks need to be stored (and where
# temporary files will be created).
data_dir = '/Users/benedito/University/data/teaching/data_astro_620/data_files/'

###################################################################################################
# PHYSICS
###################################################################################################

# Set the default cosmology
cosmo = cosmology.setCosmology('planck18')

# Solar metallicity and Log10(O/H) + 12
Z_SOLAR = 0.0134
LOG_OH_SOLAR = 8.69

###################################################################################################
# PLOTTING
###################################################################################################

# Make the plots prettier. We start from a clean (default) set of settings to avoid mixed 
# user/script settings.
mpl.rcParams.update(mpl.rcParamsDefault)

plt.rcParams['figure.figsize'] = (4.5, 4.5)
plt.rcParams['figure.dpi'] = 100.0
plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['figure.subplot.left'] = 0.2
plt.rcParams['figure.subplot.right'] = 0.95
plt.rcParams['figure.subplot.bottom'] = 0.2
plt.rcParams['figure.subplot.top'] = 0.95
plt.rcParams['figure.subplot.wspace'] = 0.2
plt.rcParams['figure.subplot.hspace'] = 0.2

plt.rcParams['axes.labelpad'] = 10

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12.0

plt.rcParams['xtick.top'] = True
plt.rcParams['xtick.bottom'] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['xtick.major.pad'] = 7
plt.rcParams['xtick.minor.pad'] = 7
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams['ytick.major.pad'] = 7
plt.rcParams['ytick.minor.pad'] = 7

plt.rcParams['legend.numpoints'] = 1
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['legend.markerscale'] = 1.0
plt.rcParams['legend.borderpad'] = 0.5
plt.rcParams['legend.labelspacing'] = 0.4
plt.rcParams['legend.borderaxespad'] = 0.5
plt.rcParams['legend.handlelength'] = 2.0
plt.rcParams['legend.handletextpad'] = 0.8
plt.rcParams['legend.columnspacing'] = 2.0
plt.rcParams['legend.handleheight'] = 0.7
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.shadow'] = False
plt.rcParams['legend.fancybox'] = False

# Save the default color cycle in a variable so we can set colors explicitly
prop_cycle = plt.rcParams['axes.prop_cycle']
color_cycle = prop_cycle.by_key()['color']

###################################################################################################
