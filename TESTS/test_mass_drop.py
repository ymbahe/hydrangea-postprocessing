"""
Investigate highly suspicious 'mass drop' in massive galaxies shortly
after accretion

Started 9-Apr-2018
"""

import numpy as np
import yb_utils as yb
import sim_tools as st
from pdb import set_trace

rundir = '/virgo/simulations/Hydrangea/10r200/HaloF0/HYDRO/'
spiderloc = rundir + '/highlev/SpiderwebTables.hdf5'
linkloc = rundir + '/highlev/SpiderwebLinks.hdf5'

gal = 1729

shi = yb.read_hdf5(spiderloc, 'SubHaloIndex')

set_trace()

print("Done!")
