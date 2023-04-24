# Program to extract the mass (m200) growth history of each cluster 
# Started 21 Sep 2016

# Add-on to extract spectre flag and store it in 'Full' folder of output
# Started 19-03-2018


import hydrangea_tools as ht
import sim_tools as st
import eagle_routines as er
import numpy as np
from pdb import set_trace
from scipy.optimize import curve_fit
from astropy.io import fits
import glob
import image_routines as im
import time
import os.path
import yb_utils as yb
from mpi4py import MPI
import copy

write_specFlag = True

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

simname = 'EAGLE'
simtype = 'HYDRO'
tractype = "Spiderweb"

if simname == 'Hydrangea':
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    nsnap = 30
    nsim = 30

elif simname == 'EAGLE':
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    outdir = '/freya/ptmp/mpa/ybahe/EAGLE/L0100N1504/REFERENCE/'
    nsnap = 29
    nsim = 1
    
snaplist = np.arange(1, nsnap, dtype=int)  # Can't get spectres in first snap!

for isim in range(nsim):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    if simname == 'Hydrangea':
        rundir = basedir + 'HaloF' + str(isim) + '/' + simtype
        outloc = ht.clone_dir(rundir) + '/highlev/FullGalaxyTablesMay18.hdf5'
        tracdir = ht.clone_dir(rundir) 

    elif simname == 'EAGLE':
        rundir = basedir
        outloc = outdir + '/highlev/FullGalaxyTables.hdf5'
        tracdir = outdir

    if not os.path.exists(rundir):
        continue

    print("")
    print("=============================")
    print("Processing simulation F" + str(isim))
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

    sim_specFlag = np.zeros(ngal_max, dtype = np.int8)   # Is spectre (1) or not (0)
    sim_specParent = np.arange(ngal_max, dtype = np.int32)  # ParentGalaxy (itself if not spectre, =default)  

    sim_firstSnap = yb.read_hdf5(tracfile, 'FirstSnap')
    if len(sim_firstSnap) != ngal_max:
        print("Inconsistent galaxy numbers.")
        set_trace()

    # --------------------------------------------------
    # ------ Now loop through all snaps... -------
    # --------------------------------------------------

    for isnap, snap in enumerate(snaplist):
        
        snapname = 'Snapshot_' + str(snap).zfill(3)

        print("")
        print("-----------------------------")
        print("Snapshot F" + str(isim) + "." + str(snap))
        print("-----------------------------")
        print("", flush = True)

        ind_thissnap = np.nonzero(sim_firstSnap == snap)[0]

        if len(ind_thissnap) == 0:
            continue

        flags_born = yb.read_hdf5(tracfile, 'Subhalo/' + snapname + '/Flags')
        sh_spectre = np.nonzero((flags_born & 2**27) > 0)[0]
        gal_thissnap = yb.read_hdf5(tracfile, 'Subhalo/'+ snapname + '/Galaxy')
        gal_prevsnap = yb.read_hdf5(tracfile, 'Subhalo/Snapshot_' + str(snap-1).zfill(3) + '/Galaxy') 
        sh_prog_all = yb.read_hdf5(tracfile, 'Subhalo/' + snapname + '/Reverse/SubHaloIndex')
        
        if len(sh_spectre) > 0:
            sim_specFlag[gal_thissnap[sh_spectre]] = 1
            sim_specParent[gal_thissnap[sh_spectre]] = gal_prevsnap[sh_prog_all[sh_spectre]]

    if write_specFlag:
        yb.write_hdf5(sim_specFlag, outloc, 'Full/SpectreFlag') 
    yb.write_hdf5(sim_specParent, outloc, 'Full/SpectreParents')
    

print("Done!")
