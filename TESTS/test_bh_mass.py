"""
Check whether recomputed BH mass differs from original Subfind values...
"""

import numpy as np
import sim_tools as st
import yb_utils as yb
import time
import os

rundir_base = '/virgo/simulations/Hydrangea/10r200/'

for isim in range(30):

    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo F{:d}" .format(isim))
    print("**************************")
    print("")
     
    rundir = rundir_base + '/HaloF' + str(isim) + '/HYDRO'
    if not os.path.exists(rundir):
        continue

    for isnap in range(30):

        subdir = st.form_files(rundir, isnap)
        bh_file = yb.dir(subdir) + '/BlackHoleMass.hdf5'

        if not os.path.exists(bh_file):
            continue
        
        flag = yb.read_hdf5_attribute(bh_file, "BlackHoleMass", "ConsistentWithSubfind")

        if flag == 0:
            print("Simulation F{:d}, snapshot {:d} has inconsistencies!"
                  .format(isim, isnap))


print("Done!")
