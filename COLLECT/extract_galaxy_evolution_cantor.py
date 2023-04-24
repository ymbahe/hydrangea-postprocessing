"""
Extract frequently-used subhalo properties from all snapshots and store them 
in a single HDF5 file, aligned by galaxy ID.

-- Begun 5 June 2017 (on train to Leiden)
-- Cleaned up 08-May-2019: removed non-Spiderweb tracing support, debugged 
      Full/SatFlag counter, added comments to all data sets. 
-- Adapted 3-Aug-2019: uses Cantor instead of Subfind for input.

Used to be called 'galaxy_extraction_jun17.py'
"""

import hydrangea_tools as ht
import sim_tools as st
import yb_utils as yb
import numpy as np
from pdb import set_trace
import time
import os.path
from mpi4py import MPI

# ==================================================================

simname = 'EAGLE'        # 'EAGLE' or 'Hydrangea'
simtype = 'HYDRO'        # 'HYDRO' or 'DM'
overwrite = True         # Re-generate existing outputs
datestamp = ''           # Date stamp (may be empty)

if simname == 'Hydrangea':
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    hldir = ht.clone_dir(rundir) + '/highlev/'
    outdir = hldir
    nsnap = 30
    nsim = 1

elif simname == 'EAGLE':
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    hldir = er.clone_dir(basedir) + '/highlev/'
    outdir = hldir
    nsnap = 29
    nsim = 1


# ==================================================================


# Set up MPI interface, to process different simulations in parallel
comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()
    
for isim in range(nsim):
    
    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue

    outloc = outdir + 'CantorGalaxyTables' + datestamp + '.hdf5'
        
    if simname == 'Hydrangea':
        rundir = basedir + 'CE-' + str(isim) + '/' + simtype
    elif simname == 'EAGLE':
        rundir = basedir
    else:
        print("Do not understand simname='" + simname + "'.")
        set_trace()
    
    if not os.path.exists(rundir):
        continue

    print("")
    print("=============================")
    print("Processing simulation " + str(isim))
    print("=============================")
    print("", flush = True)

    if os.path.exists(outloc) and overwrite is False:
        print("Output already exists.")
        continue

    sim_stime = time.time()

    cantorloc = hldir + '/highlev/CantorCatalogue.hdf5'
    if not os.path.exists(cantorloc):
        print("Cantor catalogue file '" + cantorloc + "' not found!")
        continue

    sim_shi = yb.read_hdf5(cantorloc, 'SubhaloIndex') 
    sim_cenGal = yb.read_hdf5(cantorloc, 'CenGalExtended')
    ngal_max = shi_table.shape[0]

    # Set up arrays for all quantities to extract
    sim_satflag = np.zeros((ngal_max,nsnap),dtype = np.byte) + 255
    sim_msub = np.zeros((ngal_max,nsnap), dtype = np.float32) + np.nan

    sim_mstar = np.zeros((ngal_max,nsnap), dtype = np.float32) + np.nan
    sim_mdm = np.zeros((ngal_max,nsnap), dtype = np.float32) + np.nan
    sim_mgas = np.zeros((ngal_max,nsnap), dtype = np.float32) + np.nan
    sim_mbh = np.zeros((ngal_max,nsnap), dtype = np.float32) + np.nan
    
    
    # --------------------------------------------------
    # ------ Now loop through all snaps... -------
    # --------------------------------------------------

    for isnap in range(nsnap):
        
        subdir = st.form_files(rundir, isnap, 'sub')
        
        if subdir is None:
            continue
        if not os.path.exists(subdir):
            continue
        
        snapname = 'Snapshot_' + str(isnap).zfill(3)

        print("")
        print("-----------------------------")
        print("Snapshot " + str(isim) + "." + str(isnap))
        print("-----------------------------")
        print("", flush = True)

        snap_pre = 'Snapshot_{:03d}/' .format(isnap)
        sub_pre = snap_pre + 'Subhalo/'

        # 1.) Record SHI of all galaxies in current snapshot

        snap_shi = sim_shi[:, isnap]
        galaxy_id = yb.read_hdf5(cantorloc, sub_pre + 'Galaxy')

        ind_alive = np.nonzero(snap_shi >= 0)[0]
        ind_alive_out = np.nonzero(sim_shi[:,isnap] >= 0)[0]

        if len(ind_alive) == 0: continue

        # 2.) Record total and by-particle masses

        sh_msub = yb.read_hdf5(cantorloc, sub_pre + 'Mass')
        sim_msub[ind_alive_out, isnap] = np.log10(
            sh_msub[snap_shi[ind_alive]])+10.0
        
        if simtype == "HYDRO":
            sh_masstype = np.log10(
                yb.read_hdf5(cantorloc, sub_pre + 'MassType'))+10.0

            sim_mstar[ind_alive_out, isnap] = (
                sh_masstype[snap_shi[ind_alive], 4])
            sim_mdm[ind_alive_out, isnap] = (
                sh_masstype[snap_shi[ind_alive], 1])
            sim_mgas[ind_alive_out, isnap] = (
                sh_masstype[snap_shi[ind_alive], 0])
            sim_mbh[ind_alive_out, isnap] = (
                sh_masstype[snap_shi[ind_alive], 5])

        else:
            sim_mdm[ind_alive_out, isnap] = np.log10(
                sh_msub[snap_shi[ind_alive]])+10.0

        # 3.) [ Vmax -- removed ]
        # 4.) Record satellite flag

        sim_satflag[ind_alive, isnap] = 1
        ind_cen = np.nonzero(
            sim_cenGal[ind_alive, isnap] == np.arange(len(ind_alive)))[0]
        sim_satflag[ind_alive[ind_cen], isnap] = 0

        # 5.) Record contamination flag [currently skipped -- use SF version]

        """
        if simname == 'Hydrangea':
            file_cont_flag = hldir + '/SubhaloExtra.hdf5'
            if not os.path.exists(file_cont_flag):
                print("WARNING: no contamination flag found for sim {:d}, snapshot {:d}!" .format(isim, isnap))
                set_trace()
            
            sh_cont_flag = yb.read_hdf5(file_cont_flag, "/Snapshot_" + str(isnap).zfill(3) + "/BoundaryFlag")
            sim_snap_cont_flag = sh_cont_flag[snap_shi[ind_alive]]
            sim_contflag[ind_alive_out, isnap] = sim_snap_cont_flag
        """                          


    # -----------------------------------------------------------    
    # --------- Ends loop through individual snapshots ----------
    # -----------------------------------------------------------


    yb.write_hdf5(sim_shi, outloc, 'SHI', new = True, comment = "Subhalo index of galaxy i (first index) in snapshot j (second index). This is an exact duplicate of the data set 'SubhaloIndex' in the Cantor catalogue, included here for convenience and cross-checking.")
    #yb.write_hdf5(sim_vmax, outloc, 'Vmax', comment = "Maximum circular velocity of galaxy i (first index) in snapshot j (second index), in proper km/s. -1 indicates that the galaxy is not found.")
    yb.write_hdf5(sim_satflag, outloc, 'SatFlag', comment = "Flag indicating if galaxy i (first index) is, in snapshot j (second index), a satellite (1) or central (0). 255 indicates that the galaxy is not found.")
    yb.write_hdf5(sim_msub, outloc, 'Msub', comment = "Total mass of the subhalo of galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). NaN indicates that the galaxy was not found.")
    yb.write_hdf5(sim_contflag, outloc, 'ContFlag', comment = "Flag to indicate that galaxy i (first index) is, in snapshot j (second index), close to the the zoom-in edge. Higher values correspond to smaller distances to the nearest boundary particle. 255 indicates that the galaxy was not found.")
    yb.write_hdf5(sim_r200, outloc, 'R200', comment = "R200(c) radius of the halo hosting galaxy i (first index) in snapshot j (second index), in pMpc. -1 indicates that the galaxy was not found.")
    yb.write_hdf5(sim_rmax, outloc, 'VmaxRadius', comment = "Radius at which the circular velocity of galaxy i (first index) in snapshot j (second index) is maximum, in pMpc. -1 indicates that the galaxy was not found.")
    yb.write_hdf5(sim_m200, outloc, 'M200', comment = "M200(c) mass of the halo hosting galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). -1 indicates that the galaxy was not found.")
    yb.write_hdf5(sim_cenGal, outloc, 'CenGal', comment = "Galaxy ID of the central galaxy of galaxy i (first index) in snapshot j (second index). For centrals, this points to themselves. -1 if the galaxy does not exist in a snapshot.")
    

    yb.write_hdf5(sim_sfr, outloc, 'SFR', comment = "Total star formation rate of galaxy i (first index) in snapshot j (second index), in log_10(Mdot/(M_sun/yr)). -Inf indicates a star formation rate of zero, and -1 that the galaxy was not found.")
    yb.write_hdf5(sim_mstar, outloc, 'Mstar', comment = "Total stellar mass of galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). -Inf indicates that the galaxy has no stars, and -1 that it was not found.")
    yb.write_hdf5(sim_mstar_ap, outloc, 'Mstar30kpc', comment = "Stellar mass within 30pkpc of galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). -Inf indicates that the galaxy has no stars within this radius, and -1 that it was not found.")
    yb.write_hdf5(sim_mstar_init, outloc, 'MstarInit', comment = "Total initial stellar mass of galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). -Inf indicates that the galaxy has no stars, and -1 that it was not found.")
    yb.write_hdf5(sim_mgas_ap, outloc, 'Mgas30kpc', commment = "Gas mass within 30pkpc of galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). -Inf indicates that the galaxy has no gas within this radius, and -1 that it was not found.")
    yb.write_hdf5(sim_mdm, outloc, 'MDM', comment = "Total DM mass of galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). -Inf indicates that the galaxy has no DM, and -1 that it was not found.")
    yb.write_hdf5(sim_mgas, outloc, 'MGas', comment = "Total gas mass of galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). -Inf indicates that the galaxy has no gas, and -1 that it was not found.")
    yb.write_hdf5(sim_mbh, outloc, 'MBH', comment = "Total BH (particle) mass of galaxy i (first index) in snapshot j (second index), in log_10(M/M_sun). -Inf indicates that the galaxy has no BHs, and -1 that it was not found.")
    

    # Final bit: calculate 'global/max' values for each galaxy:
    
    # 1.) Maxima of vmax/mstar/msub/m200
    sim_vmax_top = np.max(sim_vmax, axis = 1)
    sim_msub_top = np.max(sim_msub, axis = 1)
    sim_m200_top = np.max(sim_m200, axis = 1)

    
    sim_mstar_top = np.max(sim_mstar, axis = 1)
    sim_mstarAp_top = np.max(sim_mstar_ap, axis = 1)
    sim_mstarInit_top = np.max(sim_mstar_init, axis = 1)
    sim_mgas_top = np.max(sim_mgas, axis = 1)
    sim_mdm_top = np.max(sim_mdm, axis = 1)
    sim_mbh_top = np.max(sim_mbh, axis = 1)
    sim_mgasAp_top = np.max(sim_mgas_ap, axis = 1)
    

    # 2.) *Sum* of (cleaned) sat flags = num of snaps in which the galaxy was a sat
    #     Changed 08-May-19, previously non-existing snaps were erroneously counted as sats...
    sim_satflag_clean = np.copy(sim_satflag)
    ind_nonex = np.nonzero(sim_satflag_clean == 255)
    sim_satflag_clean[ind_nonex] = 0
    sim_satflag_all = np.sum(sim_satflag_clean, axis = 1)

    # 3.) Combined contamination flag
    sim_contflag_all = np.zeros((sim_shi.shape[0], 4), dtype = np.int8)+127

    sim_contflag[np.nonzero(sim_shi < 0)] = 0
    
    for istrict in range(4):

        # Erase all flag marks stricter than current level:
        sim_contflag[np.nonzero(sim_contflag <= istrict)] = 0

        # Make temporary copy and set all non-zero entries to 1:
        contflags_temp = np.copy(sim_contflag)
        contflags_temp[np.nonzero(contflags_temp)] = 1
        sim_contflag_all[:, istrict] = np.sum(contflags_temp, axis = 1)

    yb.write_hdf5(sim_vmax_top, outloc, 'Full/Vmax', comment = 'Peak Vmax of galaxy i along its entire evolution, in proper km/s.')
    yb.write_hdf5(sim_satflag_all, outloc, 'Full/SatFlag', comment = 'Number of snapshots in which galaxy i is a satellite.')
    yb.write_hdf5(sim_contflag_all, outloc, 'Full/ContFlag', comment = 'Number of snapshots in which galaxy i (first index) is contaminated at level j (second index).')
    yb.write_hdf5(sim_m200_top, outloc, 'Full/M200', comment = 'Peak M200 of galaxy i along its entire evolution, in log_10(M/M_sun).')
    yb.write_hdf5(sim_msub_top, outloc, 'Full/Msub', comment = 'Peak total (subhalo) mass of galaxy i along its entire evolution, in log_10(M/M_sun).')

    yb.write_hdf5(sim_mstar_top, outloc, 'Full/Mstar', comment = 'Peak stellar mass of galaxy i along its entire evolution, in log_10(M/M_sun). -1 (or -inf) indicates that the galaxy never contained any stars.')
    yb.write_hdf5(sim_mdm_top, outloc, 'Full/MDM', comment = 'Peak DM mass of galaxy i along its entire evolution, in log_10(M/M_sun). -1 (or -inf) indicates that the galaxy never contained any DM.')
    yb.write_hdf5(sim_mgas_top, outloc, 'Full/Mgas', comment = 'Peak gas mass of galaxy i along its entire evolution, in log_10(M/M_sun). -1 (or -inf) indicates that the galaxy never contained any gas.')
    yb.write_hdf5(sim_mbh_top, outloc, 'Full/MBH', comment = 'Peak BH (particle) mass of galaxy i along its entire evolution, in log_10(M/M_sun). -1 (or -inf) indicates that the galaxy never contained any BHs.')
    yb.write_hdf5(sim_mgasAp_top, outloc, 'Full/Mgas30kpc', comment = 'Peak gas mass within 30 pkpc of galaxy i along its entire evolution, in log_10(M/M_sun). -1 (or -inf) indicates that the galaxy never contained any gas within this radius.')
    yb.write_hdf5(sim_mstarAp_top, outloc, 'Full/Mstar30kpc', comment = 'Peak stellar mass within 30 pkpc of galaxy i along its entire evolution, in log_10(M/M_sun). -1 (or -inf) indicates that the galaxy never contained any stars within this radius.')
    yb.write_hdf5(sim_mstarInit_top, outloc, 'Full/MstarInit', comment = 'Peak initial stellar mass of galaxy i along its entire evolution, in log_10(M/M_sun). -1 (or -inf) indicates that the galaxy never contained any stars.')

    sim_etime = time.time()
    print("Processing simulation {:d} took {:.3f} min."
          .format(isim, (sim_etime - sim_stime)/60))
    
print("Done!")
