"""
Extract masses of galaxies in subfind and Monk

Started 05-Apr-2019
"""

import sim_tools as st
import yb_utils as yb
import numpy as np
from pdb import set_trace
import monk
import time
import hydrangea_tools as ht
from scipy.spatial import cKDTree
import os

rootdir = '/virgo/simulations/Hydrangea/10r200/'
simtype = 'HYDRO'
snap_z0 = 29

sim_all = np.zeros(0, dtype = np.int8)
galaxy_all = np.zeros(0, dtype = np.int32)
msub_sf_all = np.zeros(0, dtype = np.float32)
mstar_sf_all = np.zeros(0, dtype = np.float32)
rrel_next_all = np.zeros(0, dtype = np.float32)
msub_monk_all = np.zeros(0, dtype = np.float32)
mstar_monk_all = np.zeros(0, dtype = np.float32)
msub_cantor_all = np.zeros(0, dtype = np.float32)
mstar_cantor_all = np.zeros(0, dtype = np.float32)

satflag_all = np.zeros(0, dtype = np.int8)

outloc = '/virgo/scratch/ybahe/HYDRANGEA/RESULTS/CantorComparison_10Apr19_FOF-CenUnbindWithSats_vs_Monk.hdf5'

for isim in range(0, 1):

    simstime = time.time()

    print("")
    print("======================")
    print("Processing CE-{:d}    " .format(isim))
    print("======================")    
    print("")

    rundir = rootdir + 'CE-' + str(isim) + '/' + simtype + '/'

    if not os.path.isdir(rundir): continue

    hldir = rundir + '/highlev/'
    fgtloc = hldir + 'FullGalaxyTables.hdf5'
    spiderloc = hldir + 'SpiderwebTables.hdf5'
    monkloc = ht.clone_dir(hldir) + 'RecomputedSubhaloMembership_04Apr19_gc5p0.hdf5'
    cantorloc = ht.clone_dir(hldir) + 'CANTOR_Catalogues_06Apr19_PTest_fromSnap_FOF_WithCenUnbind.hdf5'

    galaxy = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_029/Galaxy')

    msub_sf_sim = yb.read_hdf5(fgtloc, 'Msub')[galaxy, -1]
    mstar_sf_sim = yb.read_hdf5(fgtloc, 'Mstar')[galaxy, -1]
    rrel_next_sim = yb.read_hdf5(fgtloc, 'RelRadius')[galaxy, -1]
    satflag_sim = yb.read_hdf5(fgtloc, 'SatFlag')[galaxy, -1]


    massType_monk_sim = yb.read_hdf5(monkloc, 'Subhalo/MassType')
    mstar_monk_sim = np.log10(massType_monk_sim[:, 4])+10.0
    msub_monk_sim = np.log10(np.sum(massType_monk_sim, axis = 1))+10.0

    massType_cantor_sim = yb.read_hdf5(cantorloc, 'Snapshot_029/MassType')
    mstar_cantor_sim = np.log10(massType_cantor_sim[:, 4])+10.0
    msub_cantor_sim = np.log10(np.sum(massType_cantor_sim, axis = 1))+10.0


    sim_all = np.concatenate((sim_all, np.zeros(len(galaxy), dtype = np.int8)+isim))
    galaxy_all = np.concatenate((galaxy_all, galaxy))
    msub_sf_all = np.concatenate((msub_sf_all, msub_sf_sim))
    mstar_sf_all = np.concatenate((mstar_sf_all, mstar_sf_sim))
    rrel_next_all = np.concatenate((rrel_next_all, rrel_next_sim))
    msub_monk_all = np.concatenate((msub_monk_all, msub_monk_sim))
    mstar_monk_all = np.concatenate((mstar_monk_all, mstar_monk_sim))
    msub_cantor_all = np.concatenate((msub_cantor_all, msub_cantor_sim))
    mstar_cantor_all = np.concatenate((mstar_cantor_all, mstar_cantor_sim))

    satflag_all = np.concatenate((satflag_all, satflag_sim))
    
    yb.write_hdf5(sim_all, outloc, 'Sim', new = True)
    yb.write_hdf5(galaxy_all, outloc, 'Galaxy')
    yb.write_hdf5(msub_sf_all, outloc, 'Msub_Subfind')
    yb.write_hdf5(mstar_sf_all, outloc, 'Mstar_Subfind')
    yb.write_hdf5(msub_monk_all, outloc, 'Msub_Monk')
    yb.write_hdf5(mstar_monk_all, outloc, 'Mstar_Monk')
    yb.write_hdf5(msub_cantor_all, outloc, 'Msub_Cantor')
    yb.write_hdf5(mstar_cantor_all, outloc, 'Mstar_Cantor')
    yb.write_hdf5(rrel_next_all, outloc, 'RRelNext')
    yb.write_hdf5(satflag_all, outloc, 'SatFlag')


print("Done!")
