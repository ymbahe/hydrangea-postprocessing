"""
Helper program to extract the BH data of all galaxies

Started 20-MAY-2018
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

simname = 'Hydrangea'
simtype = 'HYDRO'

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
    
for isim in range(30):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    sim_stime = time.time()

    if simname == 'Hydrangea':
        rundir = basedir + 'HaloF' + str(isim) + '/' + simtype
        outloc = ht.clone_dir(rundir, loc = 'freya') + '/highlev/GalaxyBHTablesMay18.hdf5'
        tracdir = ht.clone_dir(rundir, loc = 'freya') + '/highlev/'  

    elif simname == 'EAGLE':
        rundir = basedir
        set_trace()

    if not os.path.exists(rundir):
        continue

    print("")
    print("=============================")
    print("Processing simulation F" + str(isim))
    print("=============================")
    print("", flush = True)


    tracfile = tracdir + '/SpiderwebTablesMay18.hdf5'
    bh_name_simple = 'BlackHoleMass.hdf5' 
    bh_name = 'BlackHoleMasses.hdf5'

    if not os.path.exists(tracfile):
        print("Tracing file not found!")
        set_trace()
    

    shi_all = yb.read_hdf5(tracfile, 'SubHaloIndex') 
    ngal = shi_all.shape[0]

    sim_mbh_part = np.zeros((ngal,nsnap), dtype = np.float32)-np.inf  # Particle mass
    sim_mbh_sg = np.zeros((ngal,nsnap), dtype = np.float32)-np.inf    # Subgrid mass
    
    sim_mbh_reass = np.zeros((ngal,nsnap), dtype = np.float32)-np.inf  # Reassigned particle mass
    sim_mbh_sg_reass = np.zeros((ngal,nsnap), dtype = np.float32)-np.inf  # Reassigned subgrid mass

    sim_mbh_orig_cen = np.zeros((ngal, nsnap), dtype = np.float32)-np.inf   # Total orig mass within centre
    sim_mbh_sg_orig_cen = np.zeros((ngal, nsnap), dtype = np.float32)-np.inf   # Total orig SG mass within centre

    sim_mbh_reass_cen = np.zeros((ngal, nsnap), dtype = np.float32)-np.inf   # Total reassigned mass within centre
    sim_mbh_sg_reass_cen = np.zeros((ngal, nsnap), dtype = np.float32)-np.inf   # Total reassigned SG mass within centre


    # --------------------------------------------------
    # ------ Now loop through all snaps... -------
    # --------------------------------------------------

    for isnap in range(nsnap):
        
        subdir = st.form_files(rundir, isnap, 'sub')
        snapname = 'Snapshot_' + str(isnap).zfill(3)

        print("")
        print("-----------------------------")
        print("Snapshot F" + str(isim) + "." + str(isnap))
        print("-----------------------------")
        print("", flush = True)

        ind_alive_snap = np.nonzero(shi_all[:, isnap] >= 0)[0]
        shi_alive_snap = shi_all[ind_alive_snap, isnap]

        massloc = yb.dir(subdir) + '/' + bh_name 
        massloc_simple = yb.dir(subdir) + '/' + bh_name_simple
        if not os.path.exists(massloc):
            print("No BH mass file found...")
            continue

            if not os.path.exists(massloc_simple):
                print("No simple BH mass file found, skipping...")
                continue

        if os.path.exists(massloc):
            sim_mbh_part[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'BHMass'))+10.0)[shi_alive_snap]
            sim_mbh_sg[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'BHSubgridMass'))+10.0)[shi_alive_snap]
        
            sim_mbh_reass[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'ReassignedBHMass'))+10.0)[shi_alive_snap]
            sim_mbh_sg_reass[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'ReassignedBHSubgridMass'))+10.0)[shi_alive_snap]
        
            sim_mbh_orig_cen[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'BHCentralMass'))+10.0)[shi_alive_snap]
            sim_mbh_sg_orig_cen[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'BHCentralSubgridMass'))+10.0)[shi_alive_snap]

            sim_mbh_reass_cen[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'ReassignedBHCentralMass'))+10.0)[shi_alive_snap]
            sim_mbh_sg_reass_cen[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'ReassignedBHCentralSubgridMass'))+10.0)[shi_alive_snap]


            sim_mbh_sg[ind_alive_snap, isnap] = (np.log10(yb.read_hdf5(massloc, 'BHMass'))+10.0)[shi_alive_snap]


    yb.write_hdf5(sim_mbh_sg, outloc, "BHSubgridMass", comment = "Sum of BH *subgrid* masses of all BH particles in this subhalo, in astro units", new = True)

    if os.path.exists(massloc):
        yb.write_hdf5(sim_mbh_part, outloc, "BHMass", comment = "Sum of BH *particle* masses of all BH particles in this subhalo, in astro units")
        
        yb.write_hdf5(sim_mbh_reass, outloc, "ReassignedBHMass", comment = "Sum of BH *particle* masses of all BH particles in this subhalo *after reassignment*, in astro units.")
        yb.write_hdf5(sim_mbh_sg_reass, outloc, "ReassignedBHSubgridMass", comment = "Sum of BH *subgrid* masses of all BH particles in this subhalo *after reassignment*, in astro units.")

        yb.write_hdf5(sim_mbh_orig_cen, outloc, "BHCentralMass", comment = "Sum of BH *particle* masses of central BH particles in this subhalo, in astro units.")
        yb.write_hdf5(sim_mbh_sg_orig_cen, outloc, "BHCentralSubgridMass", comment = "Sum of BH *subgrid* masses of central BH particles in this subhalo, in astro units.")

        yb.write_hdf5(sim_mbh_reass_cen, outloc, "ReassignedBHCentralMass", comment = "Sum of BH *particle* masses of central BH particles in this subhalo *after reassignment*, in astro units.")
        yb.write_hdf5(sim_mbh_sg_reass_cen, outloc, "ReassignedBHCentralSubgridMass", comment = "Sum of BH *subgrid* masses of central BH particles in this subhalo *after reassignment*, in astro units.")


    print("")
    print("Finished processing simulation CE-{:d} in {:.3f} sec."  .format(isim, time.time()-sim_stime))

print("Done!")
