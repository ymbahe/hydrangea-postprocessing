"""
Re-compute the black hole (subgrid) mass in the subfind table. 
This is necessary because the 'BlackHoleMass' entry is demonstrably incorrect
in at least a few cases (as found by Lieke).

Started 23-02-18
"""

import time
import os
import numpy as np
import scipy.spatial
from pdb import set_trace
import sim_tools as st
import h5py as h5
import yb_utils as yb
from mpi4py import MPI

rundir_base = '/virgo/simulations/Hydrangea/C-EAGLE/'
outname = 'BlackHoleMass.hdf5'

n_halo = 30
n_snap = 30

flag_redo = True    # Set to true to force re-computation of already existing files

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

for ihalo in [17, 19, 20, 23, 26, 27]:#range(30):
    
    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo F{:d}" .format(ihalo))
    print("**************************")
    print("")
     
    rundir = rundir_base + '/HaloF' + str(ihalo) + '/HYDRO'
    if not os.path.exists(rundir):
        continue

    for isnap in range(30):

        # Skip this one if we are multi-threading and it's not for this task to worry about
        if not isnap % numtasks == rank:
            continue

        snapdir, subdir = st.form_files(rundir, isnap=isnap, types = 'snap sub', stype = 'snap')
        
        print(" --- Snapshot {:d} ---" .format(isnap))

        if snapdir is None or subdir is None:
            continue
                
        if not os.path.exists(snapdir):
            print("   ... Snapshot not found, skipping S{:d}... " .format(isnap))
            continue

        if not os.path.exists(subdir):
            print("   ... Subfind output not found, skipping S{:d}... " .format(isnap))
            continue

        outloc = yb.dir(subdir) + outname

        if not flag_redo:
            if os.path.exists(outloc):
                continue

        nbh = yb.read_hdf5_attribute(snapdir, 'Header', 'NumPart_Total')[5]
        if nbh == 0:
            print("No black hole particles, skipping...")
            continue

        bh_sgmass = st.eagleread(snapdir, 'PartType5/BH_Mass', astro = False)
 
        # Find subhalo index of each particle

        len_ptype = st.eagleread(subdir, 'Subhalo/SubLengthType', astro = False)
        sh_len = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)
        sh_off = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False)
        sh_ids = st.eagleread(subdir, 'IDs/ParticleID', astro = False)

        fof_fsh = st.eagleread(subdir, 'FOF/FirstSubhaloID', astro = False)
        sh_mbh = st.eagleread(subdir, 'Subhalo/BlackHoleMass', astro = False)
        
        bh_id = st.eagleread(snapdir, 'PartType5/ParticleIDs', astro = False)
        revid_bh = st.create_reverse_list(bh_id, maxval = sh_ids.max()+1)

        nsh = len_ptype.shape[0]
        
        mbh = np.zeros(nsh, dtype = np.float32)
        
        for ish in range(nsh):

            if ish % 100000 == 0:
                print("Reached subhalo {:d}/{:d}..." .format(ish, nsh))

            ids_this = sh_ids[sh_off[ish]:sh_off[ish]+sh_len[ish]]
            inds_bh = revid_bh[ids_this]

            subind_match = np.nonzero(inds_bh >= 0)[0]

            if len(subind_match) == 0:
                continue

            ind_this = inds_bh[subind_match]
            mbh[ish] = np.sum(bh_sgmass[ind_this])
            

        # ends loop over individual subhaloes

        diff = mbh-sh_mbh

        flag = 1
        print("")
        
        ind_diff = np.nonzero((mbh > 0) & (np.abs(diff)/mbh > 1e-5))[0]
        n_diff = len(ind_diff)
        if n_diff > 0:
            max_diff = np.max(np.abs(diff[ind_diff]/mbh[ind_diff]))
            print("OFFSETS FOUND!")
            print("max diff = {:.3f}%, and {:d} are above 1e-5"
                  .format(max_diff*100, n_diff))
            flag = 0

        yb.write_hdf5(mbh, outloc, "BlackHoleMass", new = True, comment = "Sum of subgrid BH masses of all BH particles in this subhalo. Should be identical to Subhalo/BlackHoleMass in subfind tables, but is not always.")
        yb.write_hdf5_attribute(outloc, "BlackHoleMass", "aexp-scale-exponent", 0)
        yb.write_hdf5_attribute(outloc, "BlackHoleMass", "h-scale-exponent", -1.0)
        yb.write_hdf5_attribute(outloc, "BlackHoleMass", "ConsistentWithSubfind", flag)

    # ends loop through snapshots

# ends loop through simulations

print("Done!")
