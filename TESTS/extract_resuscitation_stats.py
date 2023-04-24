"""
Simple program to extract data about resuscitated galaxies.
"""

import numpy as np
import sim_tools as st
import yb_utils as yb
import hydrangea_tools as ht
from pdb import set_trace

rundir = '/virgo/simulations/Hydrangea/10r200/CE-0/HYDRO/'
hldir = rundir + 'highlev/'
cantorloc = ht.clone_dir(hldir) + 'CantorCatalogue_7Jun19.hdf5'
fgtloc = hldir + 'FullGalaxyTables.hdf5'
posloc = hldir + 'GalaxyPositionsSnap.hdf5'
pathloc = hldir + 'GalaxyPaths.hdf5'
spiderloc = hldir + 'SpiderwebTables.hdf5'

outloc = '/virgo/scratch/ybahe/TESTS/ResuscitatedInfo_CE0-S20.hdf5'

isnap = 20
shi = yb.read_hdf5(fgtloc, 'SHI')
shiX = yb.read_hdf5(cantorloc, 'SubHaloIndexExtended')

contFlag = yb.read_hdf5(fgtloc, 'Full/ContFlag')[:, 1]

mergeList = yb.read_hdf5(spiderloc, 'MergeList')
carriers = mergeList[:, isnap]
ind_resusc = np.nonzero((shi[:, isnap] < 0) & (shiX[:, isnap] >= 0) &
                        (shi[:, isnap] != -19) & (carriers >= 0) &
                        (contFlag == 0))[0]

carriers_resusc = carriers[ind_resusc]
# Find proper carriers for -9:

dirDesc = []   # List for direct descendant SHs per snap
galSH = []     # List for subhalo galaxy IDs per snap
for iback in range(0, 6):
    isnap_curr = isnap - iback
    if isnap_curr < 0: break # Don't go beyond S0.
    
    directDescendantSH = yb.read_hdf5(
        spiderloc, 'Subhalo/Snapshot_{:03d}/Forward/SubHaloIndexCR0'
        .format(isnap_curr))
    dirDesc.append(directDescendantSH)
    galaxySH = yb.read_hdf5(
        spiderloc, 'Subhalo/Snapshot_{:03d}/Galaxy' 
        .format(isnap_curr))
    galSH.append(galaxySH)

nsnap_back_resusc = np.zeros(len(ind_resusc), dtype = int) + np.nan

ind_m9 = np.nonzero(shi[ind_resusc, isnap] == -9)[0]

for iigal, igal in enumerate(ind_resusc):
    snap_last_alive = np.nonzero(shi[igal, :isnap] >= 0)[0][-1]
    shi_last_alive = shi[igal, snap_last_alive]
    nsnap_back = isnap - snap_last_alive

    nsnap_back_resusc[iigal] = nsnap_back
    if shi[igal, isnap] != -9: continue

    if len(dirDesc) < 6: set_trace()
    if nsnap_back > 5: set_trace()

    print(nsnap_back)
    if shi_last_alive >= len(dirDesc[nsnap_back]): set_trace()
    directDescendant = dirDesc[nsnap_back][shi_last_alive]

    if directDescendant < 0:
        # Galaxy does not merge (send any links) -- ignore here.
        carriers_resusc[iigal] = np.nan
        continue
                
    # Find the galaxy with which it has merged, in target snap:
    # (NB: need to look up the SHI in the snap AFTER last alive)
    galDirDesc = galSH[nsnap_back - 1][directDescendant]
    mergeGal = mergeList[galDirDesc, isnap]
    if shi[mergeGal, isnap] < 0:
        print("Weird -- hiding galaxy ({:d}), "
              "or even its merger target ({:d}), "
              "do not exist in target snap. I'm giving up."
              .format(galDirDesc, mergeGal))
        continue
                
    carriers_resusc[iigal] = mergeGal

snepInd = yb.read_hdf5(pathloc, 'SnapshotIndex')[isnap]
snepPre = 'Snepshot_' + str(snepInd).zfill(4) + '/'

pos_cand = yb.read_hdf5(pathloc, snepPre + 'Coordinates')[ind_resusc, :]

astro_conv = yb.read_hdf5_attribute(pathloc, snepPre + 'Coordinates', 
                                    'aexp-factor')
astro_conv *= yb.read_hdf5_attribute(pathloc, snepPre + 'Coordinates', 
                                     'h-factor')
pos_carr = yb.read_hdf5(posloc, 'Centre')[carriers_resusc, isnap, :]

vel_cand = yb.read_hdf5(pathloc, snepPre + 'Velocity')[ind_resusc, :]
velDisp_cand = yb.read_hdf5(pathloc, snepPre + 'VelocityDispersion')[ind_resusc]
vel_carr = yb.read_hdf5(posloc, 'Velocity')[carriers_resusc, isnap, :]

pos_cand *= astro_conv
#pos_carr *= astro_conv

mResusc = yb.read_hdf5(cantorloc, 'Snapshot_' + str(isnap).zfill(3) + '/Subhalo/Mass')[shiX[ind_resusc, isnap]]

disp_cand = yb.read_hdf5(pathloc, snepPre + 'CoordinateDispersion')[ind_resusc]*astro_conv

rad = np.linalg.norm(pos_cand-pos_carr, axis = 1)
velRad = np.linalg.norm(vel_cand-vel_carr, axis = 1)

yb.write_hdf5(ind_resusc, outloc, 'GalID', new = True)
yb.write_hdf5(disp_cand, outloc, 'Dispersion')
yb.write_hdf5(rad, outloc, 'Radius')
yb.write_hdf5(pos_cand, outloc, 'PosCand')
yb.write_hdf5(pos_carr, outloc, 'PosCarr')
yb.write_hdf5(shi[ind_resusc, isnap], outloc, 'SubSHI')
yb.write_hdf5(mResusc, outloc, 'MassCand')
yb.write_hdf5(vel_cand, outloc, 'VelCand')
yb.write_hdf5(velDisp_cand, outloc, 'VelDispCand')
yb.write_hdf5(velRad, outloc, 'VelRad')
yb.write_hdf5(nsnap_back_resusc, outloc, 'NsnapBack')
yb.write_hdf5(carriers_resusc, outloc, 'CarrierID')

set_trace()
