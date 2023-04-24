"""
Test for mistracing occurences similar to BCG in F25
    (i.e., jumping to wrong galaxy when subfind mis-assigns particles)

Started 24-Apr-2018
"""

import numpy as np
import sim_tools as st
import yb_utils as yb
from pdb import set_trace

rundir = '/virgo/simulations/Hydrangea/10r200/HaloF0/HYDRO/'
spiderloc = rundir + '/highlev/SpiderwebTables.hdf5'
fgtloc = rundir + '/highlev/FullGalaxyTable.hdf5'
linkloc = rundir + '/highlev/SpiderwebLinks.hdf5'

mpeak = yb.read_hdf5(fgtloc, 'Full/Msub')
satFlag = yb.read_hdf5(fgtloc, 'SatFlag')
SHI = yb.read_hdf5(fgtloc, 'SHI')
mdm = yb.read_hdf5(fgtloc, 'MDM')
msub = yb.read_hdf5(fgtloc, 'Msub')
mstar = yb.read_hdf5(fgtloc, 'Mstar')


for isnap in range(0, 29):  # No tracing from last snap!

    pre = 'Level1/Snapshot_' + str(isnap).zfill(3) + '/'
    sender = yb.read_hdf5(linkloc, pre + 'Sender')
    receiver = yb.read_hdf5(linkloc, pre + 'Receiver')
    choice = yb.read_hdf5(linkloc, pre + 'Choice')
    coreRank = yb.read_hdf5(linkloc, pre + 'CoreRank')
    galaxy = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' + str(isnap).zfill(3) + '/Galaxy')
    length = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' + str(isnap).zfill(3) + '/Forward/Length')
    coreNumPart = yb.read_hdf5(linkloc, pre + 'CoreNumPart')
    numPart = yb.read_hdf5(linkloc, pre + 'NumPart')

    ind = np.nonzero((mpeak[galaxy] >= 13.0) &
                     (mpeak[galaxy] < 14.0) &
                     (length == 1) &
                     (satFlag[galaxy, isnap] == 0) & 
                     (satFlag[galaxy, isnap+1] == 0)
                     )[0]
                     
    print("Snapshot {:d}, found {:d} galaxies to test..." 
          .format(isnap, len(ind)))

    for ish in ind:
        ish_next = SHI[galaxy[ish], isnap+1]
        
        ind_links_from = np.nonzero((sender == ish) & 
                                    (choice == 0) &
                                    (receiver != ish_next) &
                                    (mdm[receiver, isnap+1] > mstar[receiver, isnap+1]-2) &
                                    (coreNumPart > 0.5*numPart) &
                                    (mstar[receiver, isnap+1] > mstar[ish, isnap]-1))[0]
        
        
        n_links_from = len(ind_links_from)
        if n_links_from == 0:
            continue

        ind_links_to = np.nonzero((receiver == ish_next) &
                                  (coreRank == 0) &
                                  (sender != ish) &
                                  (mpeak[galaxy[sender]] > mpeak[galaxy[ish]]-1)
                                  )[0]
        
        n_links_to = len(ind_links_to)
        if n_links_to > 0:
            print("Suspect subhalo S{:d}.{:d} - galaxy {:d}..."
                  .format(isnap, ish, galaxy[ish]))

            set_trace()

print("Done!") 
