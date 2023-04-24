# Add on forward-projected maximum subhalo and stellar masses to FGT
# Started 05-Apr-2018

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

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

simname = 'Hydrangea'
simtype = 'HYDRO'
tractype = "Spiderweb"

if simname == 'Hydrangea':
    basedir = '/virgo/simulations/Hydrangea/C-EAGLE/'
    evolloc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/RESULTS/GalaxyEvolCatalogue_8Jun18_HYDRO.hdf5' 
    nsnap = 30
    nsim = 30

elif simname == 'EAGLE':
    basedir = '/virgo/simulations/Eagle/S15_AGNdT9/'
    outdir = '/freya/ptmp/mpa/ybahe/EAGLE/L0050N0752/S15_AGNdT9/'
    nsnap = 29
    nsim = 1
    
snaplist = np.arange(nsnap, dtype=int)

evol_satflag = yb.read_hdf5(evolloc, 'SatFlag')
evol_satflag[np.nonzero(evol_satflag == 255)] = 0
evol_msub = yb.read_hdf5(evolloc, 'Msub')
evol_mstar = yb.read_hdf5(evolloc, 'Mstar')
evol_mstarInit = yb.read_hdf5(evolloc, 'MstarInit')

ind_cen = np.nonzero(np.max(evol_satflag, axis = 1) == 0)[0]
n_cen = len(ind_cen)

print("Found {:d} comparison centrals..." .format(n_cen))

for isim in range(24, 30):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    if simname == 'Hydrangea':
        rundir = basedir + 'CE-' + str(isim) + '/' + simtype
        outloc = ht.clone_dir(rundir) + '/highlev/FullGalaxyTablesMay18.hdf5'
        tracdir = rundir 

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

    sim_msubMax_fp = np.zeros(ngal_max) -1
    sim_mstarMax_fp = np.zeros(ngal_max) -1
    sim_mstarInitMax_fp = np.zeros(ngal_max) -1

    sim_msub = yb.read_hdf5(outloc, 'Msub')
    sim_msubPeak = yb.read_hdf5(outloc, 'Full/Msub')
    sim_mstar = yb.read_hdf5(outloc, 'Mstar')
    sim_mstarPeak = yb.read_hdf5(outloc, 'Full/Mstar')
    sim_mstarInit = yb.read_hdf5(outloc, 'MstarInit')
    sim_mstarInitPeak = yb.read_hdf5(outloc, 'Full/MstarInit')

    sim_maxSnapMsub = np.argmax(sim_msub, axis = 1)
    sim_maxSnapMstar = np.argmax(sim_mstar, axis = 1)
    sim_maxSnapMstarInit = np.argmax(sim_mstarInit, axis = 1)

    # --------------------------------------------------
    # ------ Now loop through all galaxies... ----------
    # --------------------------------------------------

    next_dot = ngal_max/100

    for igal in range(ngal_max):
        
        if igal > next_dot:
            print(".", end = '', flush = True)
            next_dot += ngal_max/100

        if sim_msubPeak[igal] < 10.0 and sim_mstarPeak[igal] < 9.0:
            continue

        ind_comp = np.nonzero(np.abs(evol_msub[:, sim_maxSnapMsub[igal]]-sim_msubPeak[igal]) <= 0.05)[0]
        if len(ind_comp) > 0:
            sim_msubMax_fp[igal] = np.median(evol_msub[ind_comp, -1])

        ind_comp = np.nonzero(np.abs(evol_mstar[:, sim_maxSnapMstar[igal]]-sim_mstarPeak[igal]) <= 0.05)[0]
        if len(ind_comp) > 0:
            sim_mstarMax_fp[igal] = np.median(evol_mstar[ind_comp, -1])

        ind_comp = np.nonzero(np.abs(evol_mstarInit[:, sim_maxSnapMstar[igal]]-sim_mstarInitPeak[igal]) <= 0.05)[0]
        if len(ind_comp) > 0:
            sim_mstarInitMax_fp[igal] = np.median(evol_mstarInit[ind_comp, -1])
            

    print("")
    yb.write_hdf5(sim_msubMax_fp, outloc, 'Full/ForwardProjected/Msub', update = True)
    yb.write_hdf5(sim_mstarMax_fp, outloc, 'Full/ForwardProjected/Mstar', update = True)
    yb.write_hdf5(sim_mstarInitMax_fp, outloc, 'Full/ForwardProjected/MstarInit', update = True)
 
    
print("Done!")
