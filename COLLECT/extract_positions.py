"""
Helper program to extract the positions of all galaxies.
Also fills positions for hidden and dead galaxies from SNL-output, 
and interpolates hidden positions where SNL could not locate anything.

Started 09-MAR-2018
"""

import hydrangea_tools as ht
import sim_tools as st
import eagle_routines as er
import numpy as np
from pdb import set_trace
from astropy.io import ascii
import time
import os.path
import yb_utils as yb
from mpi4py import MPI
import copy
from astropy.cosmology import Planck13
import scipy.interpolate

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

simname = 'EAGLE'
simtype = 'HYDRO'

flag_redo = True

if simname == 'Hydrangea':
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    nsnap = 30
    nsim = 30
    snapAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/hydrangea_snepshots_allsnaps.dat'

elif simname == 'EAGLE':
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    outdir = '/freya/ptmp/mpa/ybahe/EAGLE/L0100N1504/REFERENCE/'
    nsnap = 29
    nsim = 1
    snapAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/eagle_outputs_new.txt'

snap_aexp = np.array(ascii.read(snapAexpLoc, format = 'no_header', guess = False)['col1'])
snap_zred = 1/snap_aexp - 1
snap_time = Planck13.age(snap_zred).value
    
for isim in range(nsim):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    if simname == 'Hydrangea':
        rundir = basedir + 'HaloF' + str(isim) + '/' + simtype
        outloc = ht.clone_dir(rundir) + '/highlev/GalaxyPositionsSnap.hdf5'
        tracdir = ht.clone_dir(rundir) + '/highlev/'  

    elif simname == 'EAGLE':
        rundir = basedir
        outloc = outdir + '/highlev/GalaxyPositionsSnap.hdf5'
        tracdir = outdir + '/highlev/'

    if not os.path.exists(rundir):
        print("Rundir '" + rundir + "' not found, skipping...")
        continue

    if not flag_redo:
        if os.path.exists(outloc): continue

    print("")
    print("=============================")
    print("Processing simulation F" + str(isim))
    print("=============================")
    print("", flush = True)


    tracfile = tracdir + '/SpiderwebTables.hdf5'
    pathfile = tracdir + '/GalaxyPaths.hdf5'

    if not os.path.exists(tracfile):
        print("Tracing file not found!")
        set_trace()

    shi_all = yb.read_hdf5(tracfile, 'SubHaloIndex') 
    shi_z0 = shi_all[:, -1]
    ngal = shi_all.shape[0]

    sim_pos = np.zeros((ngal,nsnap,3))-1000

    if os.path.exists(pathfile):
        rootIndexSnap = yb.read_hdf5(pathfile, 'RootIndex/Allsnaps')

    # --------------------------------------------------
    # ------ Now loop through all snaps... -------
    # --------------------------------------------------
    

    for isnap in range(nsnap):
        
        subdir = st.form_files(rundir, isnap, 'sub')
        snapname = 'Snapshot_' + str(isnap).zfill(3)

        if os.path.exists(pathfile):
            snepname = 'Snepshot_' + str(rootIndexSnap[isnap]).zfill(4)

        print("")
        print("-----------------------------")
        print("Snapshot F" + str(isim) + "." + str(isnap))
        print("-----------------------------")
        print("", flush = True)

        ind_alive_snap = np.nonzero(shi_all[:, isnap] >= 0)[0]
        if len(ind_alive_snap) == 0: continue
            
        sh_pos, astro_conv, aexp = st.eagleread(subdir, 'Subhalo/CentreOfPotential', astro = True)
        sim_pos[ind_alive_snap, isnap, :] = sh_pos[shi_all[ind_alive_snap, isnap], :]
    
        if os.path.exists(pathfile):

            # Now fill in gaps using SNL...
            gal_skipped = np.nonzero((shi_all[:, isnap] < 0))[0]
            
            if len(gal_skipped) > 0:
                pos_snl = yb.read_hdf5(pathfile, snepname + '/Coordinates')*astro_conv
                sim_pos[gal_skipped, isnap, :] = pos_snl[gal_skipped, :]


    # ----------------------------------------------------------------
    # ------- Final bit: interpolate over (still) missing snaps ------
    # ----------------------------------------------------------------

    for igal in range(ngal):
        
        ind_bad = np.nonzero(sim_pos[igal, :, 0] < 0)[0]
        
        if len(ind_bad) == 0:
            continue

        ind_good = np.nonzero(sim_pos[igal, :, 0] >= 0)[0]
        
        if len(ind_good) < 2:
            continue

        if len(ind_good) < 4:
            kind = 'linear'
        else:
            kind = 'cubic'

        csi = scipy.interpolate.interp1d(snap_time[ind_good], sim_pos[igal, ind_good, :], kind = kind, axis = 0, assume_sorted = True, fill_value = "extrapolate")

        pos_interp = csi(snap_time)
        sim_pos[igal, ind_bad, :] = pos_interp[ind_bad, :]


    yb.write_hdf5(sim_pos, outloc, "Centre", new = True, comment = "Centre of potential of galaxy [i] (first index) in snapshot [j] (second index), along dimension [k] (third index), in pMpc. When a galaxy is not alive in a given snapshot, the estimated value from the GalaxyPaths catalogue is inserted instead. Where this is undefined, gaps are filled using cubic spline (linear) interpolation, provided a position could be determined in at least 4 (2) snapshots.")
    

print("Done!")
