"""
Test subhalo quantities from Cantor ouptut

Started 3-Jun-2019
"""

import sim_tools as st
import yb_utils as yb
import numpy as np
from pdb import set_trace

rundir = '/virgo/simulations/Hydrangea/10r200/CE-0/HYDRO/'
posloc = rundir + 'highlev/GalaxyPositionsSnap.hdf5'
fgtloc = rundir + 'highlev/FullGalaxyTables.hdf5'
spiderloc = rundir + 'highlev/SpiderwebTables.hdf5'

cantorloc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/ANALYSIS/10r200/CE-0/HYDRO/highlev/Cantor_Catalogues_27May19_TestCantor_MostBoundCen.hdf5.old'
 
shi = yb.read_hdf5(fgtloc, 'SHI')
gal_s0 = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_000/Galaxy')

pos_sf = yb.read_hdf5(posloc, 'Centre')[gal_s0]
pos_ca = yb.read_hdf5(cantorloc, 'Snapshot_000/Subhalo/CentreOfPotential')

vmax_sf = yb.read_hdf5(fgtloc, 'Vmax')[gal_s0]
vmax_ca = yb.read_hdf5(cantorloc, 'Snapshot_000/Subhalo/Vmax')

shmr_sf = yb.read_hdf5(fgtloc, 'StellarHalfMassRad')[gal_s0]
shmr_ca = yb.read_hdf5(cantorloc, 'Snapshot_000/Subhalo/StellarHalfMassRadius')

set_trace()

print("Done!")
