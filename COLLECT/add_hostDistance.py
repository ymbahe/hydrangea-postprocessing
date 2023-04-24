# Add distance from (nearest) host galaxy to FGT.
# For centrals, this refers to the nearest FoF group with greater M200c.
# Started 25-Oct-2018

import hydrangea_tools as ht
import sim_tools as st
import eagle_routines as er
import numpy as np
from pdb import set_trace
from scipy.optimize import curve_fit
import scipy.spatial
from astropy.io import fits
import glob
import image_routines as im
import time
import os.path
import yb_utils as yb
from mpi4py import MPI
import copy

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

simname = 'Hydrangea'
simtype = 'HYDRO'
tractype = "Spiderweb"

if simname == 'Hydrangea':
    basedir = '/virgo/simulations/Hydrangea/C-EAGLE/'
    nsnap = 30
    boxsize = None
    nsim = 30

elif simname == 'EAGLE':
    basedir = '/virgo/simulations/Eagle/S15_AGNdT9/'
    outdir = '/freya/ptmp/mpa/ybahe/EAGLE/L0050N0752/S15_AGNdT9/'
    boxsize = 50.0
    nsnap = 29
    nsim = 1
    
snaplist = np.arange(nsnap, dtype=int)

for isim in range(1, nsim):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    if simname == 'Hydrangea':
        rundir = basedir + 'CE-' + str(isim) + '/' + simtype
        outloc = rundir + '/highlev/FullGalaxyTables.hdf5'
        posloc = rundir + '/highlev/GalaxyPositionsSnap.hdf5'
        tracdir = rundir 

    elif simname == 'EAGLE':
        rundir = basedir
        outloc = outdir + '/highlev/FullGalaxyTables.hdf5'
        posloc = outdir + '/highlev/GalaxyPositionsSnap.hdf5'
        tracdir = outdir

    if not os.path.exists(rundir):
        continue

    print("")
    print("=============================")
    print("Processing simulation CE-" + str(isim))
    print("=============================")
    print("", flush = True)


    if tractype == "Spiderweb":
        tracfile = tracdir + '/highlev/SpiderwebTables.hdf5'
    else:
        print("Wrong tracing file.")
        set_trace()

    if not os.path.exists(tracfile):
        print("Tracing file not found!")
        continue

    if tractype == "Protea":
        shia_29 = yb.read_hdf5(tracfile, 'Snapshot_' + str(nsnap-1).zfill(3) + '/SubHaloIndexAll')
    elif tractype == "Spiderweb":
        shi_table = yb.read_hdf5(tracfile, 'SubHaloIndex') 
        shia_29 = shi_table[:, -1]

    
    ngal_max = shia_29.shape[0]
    sim_rrel = np.zeros((ngal_max, nsnap), dtype = np.float32)+1000

    # Loop through snapshots:

    for isnap in range(nsnap):

        print("")
        print("-----------------------------")
        print("Snapshot " + str(isnap))
        print("-----------------------------")
        print("", flush = True)

        ind_alive = np.nonzero(shi_table[:, isnap] >= 0)[0]
        if len(ind_alive) == 0: continue

        # Load positions of alive galaxies
        pos_all = yb.read_hdf5(posloc, "Centre")[:, isnap, :]
        pos_alive = pos_all[ind_alive, :]
        satFlag_alive = yb.read_hdf5(outloc, "SatFlag")[ind_alive, isnap]
        cenGal_alive = yb.read_hdf5(outloc, "CenGal")[ind_alive, isnap]
        m200_alive = yb.read_hdf5(outloc, "M200")[ind_alive, isnap]
        r200_all = yb.read_hdf5(outloc, "R200")[:, isnap]

        # Part 1: Satellites
        ind_sat = np.nonzero(satFlag_alive == 1)[0]
        print("Processing {:d} satellites..." .format(len(ind_sat)))

        cen_sat = cenGal_alive[ind_sat]
        pos_cen_sat = pos_all[cen_sat, :]
        pos_sat = pos_alive[ind_sat, :]
        r200_cen_sat = r200_all[cen_sat]
        rrel_sat = np.linalg.norm((pos_cen_sat - pos_sat), axis = 1)/r200_cen_sat
        sim_rrel[ind_alive[ind_sat], isnap] = rrel_sat
        
        # Part 2: Centrals
        ind_cen = np.nonzero(satFlag_alive == 0)[0]
        print("Processing {:d} centrals..." .format(len(ind_cen)))

        sorter_cen = np.flip(np.argsort(m200_alive[ind_cen]), axis = 0)

        tree = scipy.spatial.cKDTree(pos_alive[ind_cen,:], boxsize = boxsize) 
        cen_pos = pos_alive[ind_cen,:]
            
        maxrad = 10*r200_all[ind_alive].max()

        for iicen, icen in enumerate(ind_cen):

            if iicen % 1000 == 0:
                print("Reached {:d}/{:d}..."  .format(iicen, len(ind_cen)))
            
            #ind_moreMass = sorter_cen[:icen]
            #if len(ind_moreMass) == 0: continue

            pos_curr = pos_alive[icen, :]

            subgals = np.array(tree.query_ball_point(pos_curr, maxrad))
            subgals_relpos = pos_alive[ind_cen[subgals],:]-pos_curr[None,:]
            subgals_rrel = np.linalg.norm(subgals_relpos, axis = 1)/r200_all[ind_alive[icen]]

            ind_tothis = np.nonzero((subgals_rrel > 0) &   # Not the current one itself 
                                    (m200_alive[ind_cen[subgals]] < m200_alive[icen]) & # Less massive
                                    (subgals_rrel < sim_rrel[ind_alive[ind_cen[subgals]], isnap]))[0]  # Current is closest

            sim_rrel[ind_alive[ind_cen[subgals[ind_tothis]]], isnap] = subgals_rrel[ind_tothis] 

    yb.write_hdf5(sim_rrel, outloc, 'RelRadius', update = True)
 
    
print("Done!")
