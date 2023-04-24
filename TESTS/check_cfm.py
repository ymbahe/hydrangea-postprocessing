"""
Check that all snapshots have ContaminationFlagMulti...
"""

import h5py as h5
import numpy as np
import sim_tools as st
import yb_utils as yb
import os 

rundir_base = '/virgo/simulations/Hydrangea/10r200/'
outname = 'BoundaryFlag.hdf5'

n_halo = 30
n_snap = 30

for ihalo in range(n_halo):
    
    print("")
    print("**************************")
    print("Now processing halo F{:d}" .format(ihalo))
    print("**************************")
    print("")
     
    rundir = rundir_base + '/HaloF' + str(ihalo) + '/DM'
    if not os.path.exists(rundir):
        continue

    halo_is_fine = True
    for isnap in range(n_snap):

        subdir = st.form_files(rundir, isnap)
        f = h5.File(yb.dir(subdir) + outname, 'r')
        if not "ContaminationFlagMulti" in f:
            print("No CFM found for snapshot {:d}.{:d}!!"
                  .format(ihalo, isnap))
            halo_is_fine = False

    if halo_is_fine:
        print("Simulation {:d} is fine." .format(ihalo))


print("Done!")
