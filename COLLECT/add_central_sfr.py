"""
Add star formation rate within 30 pkpc to FullGalaxyTables.

 -- 17-May-2019: Adapted from add_shmr.py
"""

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
    nsnap = 30
    nsim = 30

elif simname == 'EAGLE':
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    outdir = '/freya/ptmp/mpa/ybahe/EAGLE/L0100N1504/REFERENCE/'
    nsnap = 29
    nsim = 1
    
snaplist = np.arange(nsnap, dtype=int)

for isim in range(nsim):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    if simname == 'Hydrangea':
        rundir = basedir + 'CE-' + str(isim) + '/' + simtype
        outloc = rundir + '/highlev/FullGalaxyTables.hdf5'
        tracdir = rundir 

    elif simname == 'EAGLE':
        rundir = basedir
        outloc = outdir + '/highlev/FullGalaxyTables.hdf5'
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
    sim_sfr_ap = np.zeros((ngal_max, nsnap), dtype = np.float32) + np.nan

    # Loop through snapshots:

    for isnap in range(nsnap):

        print("")
        print("-----------------------------")
        print("Snapshot " + str(isim) + "." + str(isnap))
        print("-----------------------------")
        print("", flush = True)

        ind_alive = np.nonzero(shi_table[:, isnap] >= 0)[0]
        if len(ind_alive) == 0: continue
        shi_alive = shi_table[ind_alive, isnap]
                
        subdir = st.form_files(rundir, isnap)
        sfr_ap = st.eagleread(
            subdir, 'Subhalo/ApertureMeasurements/SFR/030kpc', astro = True)[0]
        sim_sfr_ap[ind_alive, isnap] = np.log10(sfr_ap[shi_alive])
        
    yb.write_hdf5(sim_sfr_ap, outloc, 'SFR30kpc', update = True, 
                  comment = "Star formation rate within 30 pkpc of galaxy i "
                  "(first index) in snapshot j (second index), "
                  "in log_10(Mdot/(M_sun/yr)). -Inf indicates a star "
                  "formation rate of zero, and NaN that the galaxy was "
                  "not found.")
 
    
print("Done!")
