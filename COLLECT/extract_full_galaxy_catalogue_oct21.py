"""
Extract combined galaxy evolution catalogues from all simulations.

Galaxies are selected based on adjustable selection criteria, and
all the full (snapshot-based) evolution of the selected galaxies
from all simulations is combined and written to an (HDF5) file.

In its current version, this script can be run on both Hydrangea and
EAGLE. Mstar, Mstar30kpc, MstarInit, Msub, and Vmax are interpolated
over skipped snapshots.

-- Started 21 Sep 2016
-- Modified 8-May-2018 to interpolate (properly) key quantities in skipped
     snapshots, and include clean galaxies outside 10 r200

This code used to be called 'extract_galaxy_growth_[date].py'.

!! 08-May-2019: check through code, possibly clean up inclusion criteria. !!

20-OCT-2021: started updating
"""

import hydrangea as hy
from pythontools import TimeStamp
import numpy as np
import scipy.interpolate
from pdb import set_trace
from astropy.cosmology import Planck13
from astropy.io import ascii
import os.path

datestamp = '20Oct21'

simtype = 'HYDRO'
simname = "Hydrangea"
include_ceo = False

outloc = (f'/virgo/scratch/ybahe/HYDRANGEA/RESULTS/FULL_CATALOGUES/'
          f'FullGalaxyCatalogue_{simname}_{datestamp}_{simtype}.hdf5')

if simname == "Eagle":
    basedir_hl = '/virgo/scratch/ybahe/EAGLE/'
    snapAexpLoc = ('/virgo/scratch/ybahe/HYDRANGEA/OutputLists/'
                   'eagle_outputs_new.txt')
    nsnap = 29
    isnap_z0 = 28
    n_sim = 6
    boxvec = ['L0025N0376', 'L0025N0752', 'L0025N0752', 'L0050N0752',
              'L0050N0752', 'L0100N1504']
    modelvec = ['REFERENCE', 'REFERENCE', 'RECALIBRATED', 'REFERENCE',
                'S15_AGNdT9', 'REFERENCE']
    resvec = [0, 1, 1, 0, 0, 0]
    sizevec = [25.0, 25.0, 25.0, 50.0, 50.0, 100.0]

else:
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    snapAexpLoc = ('/virgo/scratch/ybahe/HYDRANGEA/OutputLists/'
                   'hydrangea_snepshots_allsnaps.dat')
    nsnap = 30
    n_sim = 30
    isnap_z0 = 29
    
# Calculate snapshot times
snap_aexp = np.array(ascii.read(snapAexpLoc, format = 'no_header',
                                guess = False)
                     ['col1'])
snap_zred = 1/snap_aexp - 1
snap_time = Planck13.age(snap_zred).value

# Set up (empty) output arrays for to-be-collected quantities
full_sim = np.array([],dtype=np.int32)
full_shi = np.zeros((0,nsnap),dtype=np.int32)
full_gal = np.array([], dtype = np.int32)

full_vmax = np.zeros((0,nsnap), dtype = np.float32)
full_satflag = np.zeros((0,nsnap),dtype=np.int8)
full_mstar = np.zeros((0,nsnap), dtype = np.float32)
full_mstarInit = np.zeros((0, nsnap), dtype = np.float32)
full_msub = np.zeros((0,nsnap), dtype = np.float32)
full_mstar_30kpc = np.zeros((0,nsnap), dtype = np.float32)
full_msub_30kpc = np.zeros((0,nsnap), dtype = np.float32)
full_mgas_30kpc = np.zeros((0,nsnap), dtype = np.float32)
full_sfr = np.zeros((0,nsnap), dtype = np.float32)
full_sfr_30kpc = np.zeros((0,nsnap), dtype = np.float32)
full_mgas = np.zeros((0,nsnap), dtype = np.float32)
full_m200 = np.zeros((0,nsnap), dtype = np.float32)
full_shmr = np.zeros((0, nsnap), dtype = np.float32)
full_pos = np.zeros((0, nsnap, 3), dtype = np.float32)
full_vel = np.zeros((0, nsnap, 3), dtype = np.float32)
full_cenGal = np.zeros((0, nsnap), dtype = np.int32)
full_everSatFlag = np.zeros(0, dtype = np.int8)
full_rrelAll = np.zeros((0, nsnap), dtype = np.float32)
full_contFlag = np.zeros(0, dtype = np.int8)
full_spectreFlag = np.zeros(0, dtype = np.int8)
full_spectreParents = np.zeros(0, dtype = np.int32)

# Set up output arrays for quantities that only exist in Hydrangea
# or EAGLE, but not in both
if simname == 'Hydrangea':
    full_rrel = np.zeros(0, dtype = np.float32)
    full_rrelMean = np.zeros(0, dtype = np.float32)
    full_mhneutral = np.zeros((0, nsnap), dtype = np.float32)
    full_mhi = np.zeros((0, nsnap), dtype = np.float32)

else:
    full_volume = np.chararray(0, itemsize = 10)
    full_model = np.chararray(0, itemsize = 12)
    full_resolution = np.zeros(0, dtype = np.int8)
    full_boxsize = np.zeros(0, dtype = np.float32)

# Now loop through individual simulations
for isim in range(n_sim):

    if simname == 'Hydrangea':
        hldir = f'{basedir}CE-{isim}/{simtype}/highlev/'
    else:
        hldir = f'{basedir_hl}{boxvec[isim]}/{modelvec[isim}/highlev/'
        
    if not os.path.exists(hldir):
        continue

    if simname == 'Hydrangea': 
        print("")
        print("=============================")
        print(f"Processing simulation CE-{isim}"
        print("=============================")
        print("", flush = True)

        if isim in [17, 19, 20, 23, 26, 27] and include_ceo is false:
              continue
    else:
        print("")
        print("=============================")
        print(f"Processing simulation EAGLE-{boxvec[isim]}/{modelvec[isim]}")
        print("=============================")
        print("", flush = True)

    # First, identify galaxies based on z = 0 properties
    fgtloc = hldir + '/FullGalaxyTables.hdf5'
    posloc = hldir + '/GalaxyPositionsSnap.hdf5'
    rundir = basedir + 'CE-' + str(isim) + '/' + simtype 
    subdir = st.form_files(rundir, 29)
    
    sw = st.Spiderweb(hldir, highlev = True)
    spiderloc = hldir + '/SpiderwebTables.hdf5'

    if simname == 'Hydrangea':
        galaxy_z0 = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_029/Galaxy')        
        sh_pos = yb.read_hdf5(posloc, 'Centre')[galaxy_z0, -1, :]

        # Position and r200 of central cluster (SHI=0)
        pos_cl_z0 = sh_pos[0, :]
        r200_cl = yb.read_hdf5(fgtloc, 'R200')[galaxy_z0[0], -1]

        r200Mean_cl = st.eagleread(subdir, 'FOF/Group_R_Mean200', astro = True)[0][0]

        sh_relpos = sh_pos-pos_cl_z0[None,:]
        sh_rrel = np.linalg.norm(sh_relpos, axis=1)/r200_cl
        sh_rrelMean = np.linalg.norm(sh_relpos, axis=1)/r200Mean_cl

    fgt_mmax = yb.read_hdf5(fgtloc, 'Full/Msub')
    fgt_mmaxStar = yb.read_hdf5(fgtloc, 'Full/Mstar')
    fgt_contFlag = yb.read_hdf5(fgtloc, 'Full/ContFlag')[:, 2]
    fgt_spectreFlag = yb.read_hdf5(fgtloc, 'Full/SpectreFlag')
    fgt_shi = yb.read_hdf5(fgtloc, 'SHI')
    fgt_cenGal = yb.read_hdf5(fgtloc, 'CenGal')
    fgt_msub = yb.read_hdf5(fgtloc, 'Msub')
    fgt_m200 = yb.read_hdf5(fgtloc, 'M200')
    fgt_mstar = yb.read_hdf5(fgtloc, 'Mstar')
    fgt_mstarInit = yb.read_hdf5(fgtloc, 'MstarInit')
    fgt_mstar30kpc = yb.read_hdf5(fgtloc, 'Mstar30kpc')
    fgt_vmax = yb.read_hdf5(fgtloc, 'Vmax')
    fgt_satFlag = yb.read_hdf5(fgtloc, 'SatFlag')
    fgt_everSatFlag = yb.read_hdf5(fgtloc, 'Full/SatFlag')
    fgt_mgas30kpc = yb.read_hdf5(fgtloc, "Mgas30kpc")
    fgt_mgas = yb.read_hdf5(fgtloc, "MGas")
    fgt_sfr = yb.read_hdf5(fgtloc, "SFR")
    fgt_sfr30kpc = yb.read_hdf5(fgtloc, 'SFR30kpc')
    co_sfr30kpc = yb.read_hdf5_attribute(fgtloc, 'SFR30kpc', 'Comment')
    fgt_shmr = yb.read_hdf5(fgtloc, "StellarHalfMassRad")
    fgt_pos = yb.read_hdf5(posloc, "Centre")

    if simname == "Hydrangea":
        fgt_mhneutral = yb.read_hdf5(fgtloc, 'MHneutral')
        fgt_mhi = yb.read_hdf5(fgtloc, 'MHI')

    if isim < 50:
        fgt_rrelAll = yb.read_hdf5(fgtloc, "RelRadius")
    else:
        fgt_rrelAll = np.zeros_like(fgt_shmr)

    # Index into galaxies alive at z = 0, not currently used
    gal_z0 = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' + str(isnap_z0).zfill(3) + '/Galaxy')
    
    if simname == "Hydrangea":
        gal_sel = np.nonzero((fgt_contFlag == 0) & ((fgt_mmax >= 1000.0) | (fgt_mmaxStar >= 8.0)))[0]
    else:
        gal_sel = np.nonzero((fgt_mmax >= 1000.0) | (fgt_mstar[:, -1] >= 8.0))[0]

    #gal_sel = sw.sh_to_gal(ind_sel, isnap_z0)
    n_sel = len(gal_sel)

    print("In simulation {:d}, there are {:d} selected galaxies." .format(isim, n_sel))

    sim_sim = np.zeros(n_sel,dtype=np.int8)+isim
    
    if simname == 'Eagle':
        sim_volume = np.chararray(n_sel, itemsize = 10)
        sim_volume[:] = boxvec[isim]
        sim_model = np.chararray(n_sel, itemsize = 12)
        sim_model[:] = modelvec[isim]
        sim_resolution = np.zeros(n_sel, dtype = np.int8)
        sim_resolution[:] = resvec[isim] 
        sim_boxsize = np.zeros(n_sel, dtype = np.float32)
        sim_boxsize[:] = sizevec[isim]

    # Fill in properties from FGT directly
    sim_shi = fgt_shi[gal_sel, :]
    sim_cenGal = fgt_cenGal[gal_sel, :]
    sim_mstar = fgt_mstar[gal_sel, :]
    sim_mstarInit = fgt_mstarInit[gal_sel, :]
    sim_msub = fgt_msub[gal_sel, :]
    sim_m200 = fgt_m200[gal_sel, :]
    sim_mstar_30kpc = fgt_mstar30kpc[gal_sel, :]
    sim_vmax = fgt_vmax[gal_sel, :]
    sim_satflag = fgt_satFlag[gal_sel, :]
    sim_mgas = fgt_mgas[gal_sel, :] 
    sim_mgas_30kpc = fgt_mgas30kpc[gal_sel, :]
    sim_sfr = fgt_sfr[gal_sel, :]
    sim_sfr_30kpc = fgt_sfr30kpc[gal_sel, :]
    sim_shmr = fgt_shmr[gal_sel, :]
    sim_pos = fgt_pos[gal_sel, :, :]
    sim_spectreFlag = fgt_spectreFlag[gal_sel]
    sim_everSatFlag = fgt_everSatFlag[gal_sel]
    

    if simname == 'Hydrangea':
        sim_rrel = np.zeros(n_sel, dtype = np.float32)-100
        ind_alive_z0 = np.nonzero(sim_shi[:, -1] >= 0)[0]
        sim_rrel[ind_alive_z0] = sh_rrel[sim_shi[ind_alive_z0, -1]]
        sim_rrelMean = np.zeros(n_sel, dtype = np.float32)-100
        sim_rrelMean[ind_alive_z0] = sh_rrelMean[sim_shi[ind_alive_z0, -1]]
        sim_mhneutral = fgt_mhneutral[gal_sel, :]
        sim_mhi = fgt_mhi[gal_sel, :]

    sim_rrelAll = fgt_rrelAll[gal_sel, :]
    sim_contFlag = fgt_contFlag[gal_sel]


    # New bit: interpolate over missing entries

    for igal in range(len(gal_sel)):
    
        ind_hidden = np.nonzero(sim_shi[igal, :] == -9)[0]
        ind_ok = np.nonzero(sim_shi[igal, :] >= 0)[0]

        if len(ind_hidden) == 0:
            continue

        if len(ind_ok) < 2:
            continue

        if len(ind_ok) < 4:
            kind = 'linear'
        else:
            kind = 'cubic'

        csi_mstar_30kpc = scipy.interpolate.interp1d(snap_time[ind_ok], sim_mstar_30kpc[igal, ind_ok], kind = kind, assume_sorted = True, fill_value = 'extrapolate')    

        csi_mstar = scipy.interpolate.interp1d(snap_time[ind_ok], sim_mstar[igal, ind_ok], kind = kind, assume_sorted = True, fill_value = 'extrapolate')    
        csi_mstarInit = scipy.interpolate.interp1d(snap_time[ind_ok], sim_mstarInit[igal, ind_ok], kind = kind, assume_sorted = True, fill_value = 'extrapolate')    
        csi_msub = scipy.interpolate.interp1d(snap_time[ind_ok], sim_msub[igal, ind_ok], kind = kind, assume_sorted = True, fill_value = 'extrapolate')    
        csi_vmax = scipy.interpolate.interp1d(snap_time[ind_ok], sim_vmax[igal, ind_ok], kind = kind, assume_sorted = True, fill_value = 'extrapolate')    

        mstar_30kpc_interp = csi_mstar_30kpc(snap_time)
        mstar_interp = csi_mstar(snap_time)
        mstarInit_interp = csi_mstarInit(snap_time)
        msub_interp = csi_msub(snap_time)
        vmax_interp = csi_vmax(snap_time)
    
        sim_satflag[igal, ind_hidden] = 1
        sim_everSatFlag[igal] = 1
    
        sim_mstar_30kpc[igal, ind_hidden] = mstar_30kpc_interp[ind_hidden]
        sim_mstarInit[igal, ind_hidden] = mstarInit_interp[ind_hidden] 
        sim_msub[igal, ind_hidden] = msub_interp[ind_hidden]
        sim_mstar[igal, ind_hidden] = mstar_interp[ind_hidden]
        sim_vmax[igal, ind_hidden] = vmax_interp[ind_hidden]


    full_sim = np.concatenate((full_sim, sim_sim), axis = 0)
    full_shi = np.concatenate((full_shi, sim_shi), axis = 0)
    full_cenGal = np.concatenate((full_cenGal, sim_cenGal), axis = 0)
    full_gal = np.concatenate((full_gal, gal_sel), axis = 0)
    full_vmax = np.concatenate((full_vmax, sim_vmax), axis = 0)
    full_satflag = np.concatenate((full_satflag, sim_satflag), axis = 0)
    full_msub = np.concatenate((full_msub, sim_msub))
    full_m200 = np.concatenate((full_m200, sim_m200))
    full_mstar = np.concatenate((full_mstar, sim_mstar))
    full_mstarInit = np.concatenate((full_mstarInit, sim_mstarInit))
    #full_msub_30kpc = np.concatenate((full_msub_30kpc, sim_msub_30kpc))
    full_mstar_30kpc = np.concatenate((full_mstar_30kpc, sim_mstar_30kpc))
    full_mgas = np.concatenate((full_mgas, sim_mgas))
    full_mgas_30kpc = np.concatenate((full_mgas_30kpc, sim_mgas_30kpc))
    full_sfr = np.concatenate((full_sfr, sim_sfr))
    full_sfr_30kpc = np.concatenate((full_sfr_30kpc, sim_sfr_30kpc))
    full_shmr = np.concatenate((full_shmr, sim_shmr))
    full_pos = np.concatenate((full_pos, sim_pos))
    full_rrelAll = np.concatenate((full_rrelAll, sim_rrelAll))
    full_spectreFlag = np.concatenate((full_spectreFlag, sim_spectreFlag))

    if simname == 'Hydrangea':
        full_rrel = np.concatenate((full_rrel, sim_rrel))
        full_rrelMean = np.concatenate((full_rrelMean, sim_rrelMean))
        full_mhi = np.concatenate((full_mhi, sim_mhi))
        full_mhneutral = np.concatenate((full_mhneutral, sim_mhneutral))
    else:
        full_volume = np.concatenate((full_volume, sim_volume))
        full_model = np.concatenate((full_model, sim_model))
        full_resolution = np.concatenate((full_resolution, sim_resolution))
        full_boxsize = np.concatenate((full_boxsize, sim_boxsize))
    
    full_everSatFlag = np.concatenate((full_everSatFlag, sim_everSatFlag))
    full_contFlag = np.concatenate((full_contFlag, sim_contFlag))

    yb.write_hdf5(full_sim, outloc, 'Sim', new = True)
    yb.write_hdf5(full_shi, outloc, 'SHI')
    yb.write_hdf5(full_cenGal, outloc, 'CenGal')
    yb.write_hdf5(full_vmax, outloc, 'Vmax')
    yb.write_hdf5(full_satflag, outloc, 'SatFlag')
    yb.write_hdf5(full_msub, outloc, 'Msub')
    yb.write_hdf5(full_m200, outloc, 'M200')
    yb.write_hdf5(full_mstar, outloc, 'Mstar')
    yb.write_hdf5(full_mstarInit, outloc, 'MstarInit')
    yb.write_hdf5(full_mstar_30kpc, outloc, 'Mstar30kpc')
    yb.write_hdf5(full_mgas, outloc, "Mgas")
    yb.write_hdf5(full_mgas_30kpc, outloc, "Mgas30kpc")
    yb.write_hdf5(full_sfr, outloc, "SFR")
    yb.write_hdf5(full_sfr_30kpc, outloc, "SFR30kpc", comment = co_sfr30kpc)
    yb.write_hdf5(full_shmr, outloc, "StellarHalfMassRad")
    yb.write_hdf5(full_pos, outloc, "Position")
    yb.write_hdf5(full_gal, outloc, "Galaxy")
 
    if simname == 'Hydrangea':
        yb.write_hdf5(full_rrel, outloc, 'rrel_z0')
        yb.write_hdf5(full_rrelMean, outloc, 'rrelMean_z0')
        yb.write_hdf5(full_mhi, outloc, 'MHI')
        yb.write_hdf5(full_mhneutral, outloc, 'MHNeutral')
    else:
        yb.write_hdf5(full_volume, outloc, 'EAGLE_Volume')
        yb.write_hdf5(full_model, outloc, 'EAGLE_Model')
        yb.write_hdf5(full_resolution, outloc, 'EAGLE_Resolution')
        yb.write_hdf5(full_boxsize, outloc, 'EAGLE_Boxsize')

    yb.write_hdf5(full_rrelAll, outloc, 'RRelNearest')
    yb.write_hdf5(full_everSatFlag, outloc, 'EverSatFlag')
    yb.write_hdf5(full_contFlag, outloc, 'ContFlag')
    yb.write_hdf5(full_spectreFlag, outloc, 'SpectreFlag')


print("Done!")
