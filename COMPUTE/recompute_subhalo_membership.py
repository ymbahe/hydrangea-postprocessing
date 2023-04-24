"""
Program to re-compute subhalo membership of particles to satellites,
based on particles that belonged to it the last time it was a central.
"""

"""
Still to do: convert output back to useful alignment -- either with 
subpart, or snapshot (for individual particle types).

Or, convert to ID-offset list in analogy to current data in Subhalo.
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

include_merging = False  # Include particles in to-be-merged galaxies?
include_subfind = False  # Include particles in z = 0 Subfind subhalo?
guaranteed_centre = None # If not None, the multiple of r_1/2* that is fully included

for isim in range(0, 1):

    simstime = time.time()

    rundir = rootdir + 'CE-' + str(isim) + '/' + simtype + '/'

    if not os.path.isdir(rundir): continue

    hldir = rundir + '/highlev/'
    fgtloc = hldir + 'FullGalaxyTables.hdf5'
    spiderloc = hldir + 'SpiderwebTables.hdf5'
    posloc = hldir + 'GalaxyPositionsSnap.hdf5'

    outloc_sim = ht.clone_dir(hldir) + '/RecomputedSubhaloMembership_07Apr19_NoMerg-NoSF-NoCen.hdf5'

    # 0.) Identify satellite galaxies alive at z = 0 and their last central-snapshot
    
    satFlag = yb.read_hdf5(fgtloc, 'SatFlag')
    cenGal = yb.read_hdf5(fgtloc, 'CenGal')
    shi = yb.read_hdf5(fgtloc, 'SHI')

    numGal = satFlag.shape[0]

    gal_sat_z0 = np.nonzero((satFlag[:, -1] == 1) & (shi[:, -1] >= 0))[0]
    gal_alive_z0 = np.nonzero(shi[:, -1] >= 0)[0]
    n_sat_z0 = len(gal_sat_z0)
    n_sh_z0 = len(gal_alive_z0)
    print("Found {:d} satellites at z = 0..." .format(n_sat_z0))

    # Go backwards through snaps and find last-central for each target galaxy
    snapLastCen = np.zeros(n_sat_z0, dtype = np.int8)-1
    for isnap in range(29, -1, -1):
        
        ind_this = np.nonzero((snapLastCen < 0) & (satFlag[gal_sat_z0, isnap] == 0) & (cenGal[cenGal[gal_sat_z0, snap_z0], isnap] != gal_sat_z0))[0]
        snapLastCen[ind_this] = isnap
        

    # Check whether any galaxies have never been centrals.
    # For these, set reference snap to point of peak total mass.

    subind_neverCen = np.nonzero(snapLastCen < 0)[0]
    print("Could not find a last-central snapshot for {:d} galaxies..." 
          .format(len(subind_neverCen)))

    if len(subind_neverCen) > 0:
        msub = yb.read_hdf5(fgtloc, 'Msub')
        snapLastCen[subind_neverCen] = np.argmax(msub[gal_sat_z0[subind_neverCen], :], axis = 1)

    # Sanity check to make sure all have a reference snap now:
    n_notmatched = np.count_nonzero(snapLastCen < 0)
    if n_notmatched > 0:
        print("WTF? {:d} galaxies still have no reference snap? Investigate NOW!"
              .format(n_notmatched))
        set_trace()

    galPos_z0 = yb.read_hdf5(posloc, 'Centre')[:, snap_z0, :]
    galVel_z0 = yb.read_hdf5(posloc, 'Velocity')[:, snap_z0, :]
    
    # 1.) Load particles (ID, mass, pos, vel, int. energy, type) at z = 0, invert ID list. 
    
    subdir_z0, partdir_z0 = st.form_files(rundir, snap_z0, 'sub subpart')
    npTotalType = yb.read_hdf5_attribute(partdir_z0, 'Header', 'NumPart_Total')
    npTotalType[2] = 0
    npTotalType[3] = 0
    npTotal = np.sum(npTotalType)

    print("There are {:d} particles (at z = 0) in total..."
          .format(npTotal))

    # Set up full-particle arrays
    all_ids = np.zeros(npTotal, dtype = np.int)-1
    all_pos = np.zeros((npTotal, 3), dtype = np.float64)-1
    all_vel = np.zeros((npTotal, 3), dtype = np.float32)-1
    all_mass = np.zeros(npTotal, dtype = np.float32)-1
    all_energy = np.zeros(npTotal, dtype = np.float32)-1
    all_type = np.zeros(npTotal, dtype = np.int8)-1
    all_galaxy = np.zeros(npTotal, dtype = np.int32)-1

    fof_fsh = st.eagleread(subdir_z0, 'FOF/FirstSubhaloID', astro = False)
    sh_galaxy = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' + str(snap_z0).zfill(3) + '/Galaxy')

    mergelist_z0 = yb.read_hdf5(spiderloc, 'MergeList')[:, snap_z0]

    ids_z0 = st.eagleread(subdir_z0, 'IDs/ParticleID', astro = False)
    off_z0 = st.eagleread(subdir_z0, 'Subhalo/SubOffset', astro = False)
    len_z0 = st.eagleread(subdir_z0, 'Subhalo/SubLength', astro = False)

    loadstime = time.time()
    print("")
    print(" --- Done with first stage of setup ({:.2f} min.) --- "
          .format((loadstime-simstime)/60))
    print("")
    

    for iptype in [0, 1, 4, 5]:  #range(6):
        print("Loading particles of type {:d}..." .format(iptype))

        npStart = np.sum(npTotalType[:iptype])
        npEnd = np.sum(npTotalType[:iptype+1])
        
        pt = 'PartType{:d}/' .format(iptype)
        ptype_ids = st.eagleread(partdir_z0, pt + 'ParticleIDs', astro = False)

        if ptype_ids.shape[0] != npTotalType[iptype]:
            print("Read unexpected number of particles: {:d} instead of {:d}!"
                  .format(ptype_ids.shape[0], npTotalType))
            set_trace()
    
        all_ids[npStart:npEnd] = ptype_ids
        del ptype_ids

        ptype_pos = st.eagleread(partdir_z0, pt + 'Coordinates', astro = True)[0]
        all_pos[npStart:npEnd, :] = ptype_pos
        del ptype_pos

        ptype_vel = st.eagleread(partdir_z0, pt + 'Velocity', astro = True)[0]
        all_vel[npStart:npEnd, :] = ptype_vel
        del ptype_vel

        ptype_gn = st.eagleread(partdir_z0, pt + 'GroupNumber', astro = False)
        ptype_sgn = st.eagleread(partdir_z0, pt + 'SubGroupNumber', astro = False)
        ptype_shi = np.zeros(npTotalType[iptype], dtype = np.int32)-1
        
        ind_in_sg = np.nonzero(ptype_sgn < 2**30)[0]
        ptype_shi[ind_in_sg] = ptype_sgn[ind_in_sg] + fof_fsh[np.abs(ptype_gn[ind_in_sg])-1]
        ptype_galaxy = np.zeros(npTotalType[iptype], dtype = np.int32)-1
        ptype_galaxy[ind_in_sg] = sh_galaxy[ptype_shi[ind_in_sg]]

        # To begin with, all satellites are assigned to their central. 
        # Below, particles that are still bound to the satellite are 
        # then re-assigned

        all_galaxy[npStart+ind_in_sg] = cenGal[ptype_galaxy[ind_in_sg], snap_z0]
                
        del ptype_shi
        del ptype_galaxy
        del ind_in_sg
        del ptype_sgn
        del ptype_gn

        if iptype != 1:
            ptype_mass = st.eagleread(partdir_z0, pt + 'Mass', astro = True)[0]
        else:
            ptype_mass = np.zeros(npTotalType[1], dtype = np.float32)+st.m_dm(rundir)
        all_mass[npStart:npEnd] = ptype_mass
        del ptype_mass

        if iptype == 0:
            ptype_energy = st.eagleread(partdir_z0, pt + 'InternalEnergy', astro = True)[0]
        else:
            ptype_energy = np.zeros(npTotalType[iptype])
        all_energy[npStart:npEnd] = ptype_energy
        del ptype_energy

        all_type[npStart:npEnd] = iptype

        print("...done!")

    # Sanity check to make sure all outputs are filled correctly
    if np.count_nonzero(all_type < 0):
        print("Why are some output fields not filled?!")
        set_trace()

    # Now prune the list to retain only those associated with a subhalo 
    # (i.e. exclude `fuzz' that's only in FOF or aperture)

    ind_in_sh = np.nonzero(all_galaxy >= 0)[0]
    n_in_gal = len(ind_in_sh)
    print("Out of {:d} particles in full list, {:d} are in a subhalo (={:.2f} \%)"
          .format(npTotal, n_in_gal, n_in_gal/npTotal*100))

    all_ids = all_ids[ind_in_sh]
    all_pos = all_pos[ind_in_sh, :]
    all_vel = all_vel[ind_in_sh, :]
    all_mass = all_mass[ind_in_sh]
    all_energy = all_energy[ind_in_sh]
    all_type = all_type[ind_in_sh]
    all_galaxy = all_galaxy[ind_in_sh]

    # Invert ID list

    print("Inverting z=0 ID list...")
    snapdir = st.form_files(rundir, snap_z0, 'snap')
    numDMTotal = yb.read_hdf5_attribute(snapdir, 'Header', 'NumPart_Total')[1]
    maxID = 2*(numDMTotal+1)
    
    all_revID = yb.create_reverse_list(all_ids, maxval = maxID, delete_ids = False)
    print("...done!")

    # 1b.) Set up output: subhalo masses (by type) [galaxy list already set up and initialized]
    sh_MassType = np.zeros((n_sh_z0, 6), dtype = np.float32)

    # Extra bit (if guaranteed_centre): set up tree of all particles
    if guaranteed_centre is not None:
        print("Setting up cKDTree of all particles...")
        tstime = time.time()
        particleTree = cKDTree(all_pos)
        print("... done ({:.2f} sec.)." .format(time.time()-tstime)) 
        #print("... done ({:.2f} sec.), now tree-ing subhaloes..." .format(time.time()-tstime)) 
        #tstime = time.time()
        #subhaloTree = cKDTree(galPos_z0[gal_sat_z0, :])
        #print("... done ({:.2f} sec.)." .format(time.time()-tstime))

        shmr_z0 = yb.read_hdf5(fgtloc, 'StellarHalfMassRad')[:, snap_z0]

    snapstime = time.time()
    print("")
    print(" --- Done loading particles ({:.2f} min.) --- "
          .format((snapstime - loadstime)/60))
    print("")


    # -------------------------------------
    # 2.) Loop through snapshots, backwards
    # -------------------------------------

    counterA_snapSetup = 0.0
    counterA_search = 0.0
    counterA_actualLoadInfall = 0.0
    counterA_loadInfall = 0.0
    counterA_loadSubfind = 0.0
    counterA_loadCentral = 0.0
    counterA_prepParticles = 0.0
    counterA_monk = 0.0
    

    for isnap in range(snap_z0, -1, -1):
        
        sstime = time.time()

        print("Now processing galaxies from snapshot {:d}..."
              .format(isnap))

        ind_thisSnap = np.nonzero(snapLastCen == isnap)[0]
        n_thisSnap = len(ind_thisSnap)

        print("... found {:d} galaxies..." .format(n_thisSnap))
        if n_thisSnap == 0: continue

        # - a) Load subhalo particle IDs and division information
        
        subdir = st.form_files(rundir, isnap)
        snap_ids = st.eagleread(subdir, 'IDs/ParticleID', astro = False)
        snap_off = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False)
        snap_len = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)
        
        # - b) Loop through relevant subhaloes (last-central in current snap)

        counter_loadInfall = 0.0
        counter_search = 0.0
        counter_actualLoadInfall = 0.0

        counter_loadSubfind = 0.0
        counter_loadCentral = 0.0
        counter_prepParticles = 0.0
        counter_monk = 0.0

        # Search for to-be-merging galaxies in a more clever way than brute force...
        if include_merging:
            ind_alive_snap = np.nonzero(shi[:, isnap] >= 0)[0]
            mergeTarg_z0 = mergelist_z0[ind_alive_snap]
            argsort_mergeTarg = np.argsort(mergeTarg_z0)
            galLims = np.arange(numGal+1, dtype = np.int) 
            splits_gal = np.searchsorted(mergeTarg_z0, galLims, sorter = argsort_mergeTarg)
                
        counter_snapSetup = time.time()-sstime

        for iigal, igal in enumerate(gal_sat_z0[ind_thisSnap]):
            
            gstime = time.time()

            # New bit: Find *all* galaxies alive in isnap that will merge with igal
            #          by z = 0

            gal_sh = shi[igal, isnap]
            if gal_sh < 0:
                print("Galaxy {:d} has SHI={:d} in its reference snapshot {:d}!"
                      .format(igal, gal_sh, isnap))
                set_trace()

            infall_stime = time.time()

            searchStime = time.time()
            if include_merging:
                subind_thisgal = argsort_mergeTarg[splits_gal[igal]:splits_gal[igal+1]] 
                ind_contrib_gals = ind_alive_snap[subind_thisgal]
                #ind_contrib_gals_bf = np.nonzero((mergelist_z0 == igal) & (shi[:, isnap] >= 0))[0]
                #if len(ind_contrib_gals) != len(ind_contrib_gals_bf):
                #    set_trace()
                #if np.max(np.abs(ind_contrib_gals-ind_contrib_gals_bf)) > 0:
                #    set_trace()
            else:
                ind_contrib_gals = np.array([igal])
            counter_search += time.time()-searchStime

            n_contrib_gals = len(ind_contrib_gals)

            print("Galaxy {:d} has {:d} contributors in total..." 
                  .format(igal, n_contrib_gals))

            # For debugging ease, make sure that the galaxy itself is always the first entry
            if ind_contrib_gals[0] != igal:
                loc_igal = np.nonzero(ind_contrib_gals == igal)[0]
                if len(loc_igal) != 1:
                    print("Why is there not exactly one match to galaxy itself in contrib list??")
                    set_trace()

                ind_contrib_gals[loc_igal] = ind_contrib_gals[0]
                ind_contrib_gals[0] = igal

            if n_contrib_gals <= 0:
                print("Why are there zero contributing galaxies??????")
                set_trace()

            # Find total number of particles in contributing galaxies, and set
            # up array to hold all their IDs.

            shi_contrib = shi[ind_contrib_gals, isnap]
            num_part_contrib = int(np.sum(snap_len[shi_contrib]))  # Originally uint
            offset_contrib = np.cumsum(snap_len[shi_contrib])
            offset_contrib = np.concatenate((np.array([0], dtype = snap_len.dtype), offset_contrib))

            gal_ids = np.zeros(num_part_contrib, dtype = int)-1

            # Load IDs of all contributing galaxies

            stime_actualLoad = time.time()
            for iicont, icont_sh in enumerate(shi_contrib):
                icont_ids = snap_ids[snap_off[icont_sh]:snap_off[icont_sh]+snap_len[icont_sh]]
                gal_ids[offset_contrib[iicont]:offset_contrib[iicont+1]] = icont_ids
                
            print("Finished loading {:d} IDs from {:d} contributing galaxies."
                  .format(num_part_contrib, len(shi_contrib)))

            tsA = time.time()
            counter_loadInfall += (tsA - infall_stime)
            counter_actualLoadInfall += (tsA - stime_actualLoad)
            
            # New new bit: also load all particles that are in the subhalo at z = 0 already.

            if include_subfind:
                print("Now adding particles in SF-subhalo at z = 0...")
                shi_z0_gal = shi[igal, snap_z0]
                ids_z0_gal = ids_z0[off_z0[shi_z0_gal]:off_z0[shi_z0_gal]+len_z0[shi_z0_gal]]
                ids_z0_gal = ids_z0_gal.astype(int)
                gal_ids = np.unique(np.concatenate((gal_ids, ids_z0_gal)))
                print("... done, added {:d} particles (total now {:d})."
                      .format(len(gal_ids)-num_part_contrib, len(gal_ids)))

            tsB = time.time()
            counter_loadSubfind += (tsB - tsA)
            
            # New new new bit: Also load all particles that are, at z = 0,
            #                  within x times the stellar half-mass radius.

            halopos = galPos_z0[igal, :]

            if guaranteed_centre is not None:
                print("Now adding particles near the centre of the subfind subhalo...")
                
                if shmr_z0[igal] == 0:
                    print("No stars --> no SHMR --> no additions.")
                else:
                    num_ids_orig = len(gal_ids)
                    ind_guarCen_first = particleTree.query_ball_point(halopos, shmr_z0[igal] * guaranteed_centre)
                    ind_guarCen_first = np.array(ind_guarCen_first, dtype = np.int32) # List --> array
                    deltapos_gC = all_pos[ind_guarCen_first, :] - halopos[None, :]
                    deltarad_gC = np.linalg.norm(deltapos_gC, axis = 1)
                    subind_realCen = np.nonzero(deltarad_gC <= shmr_z0[igal]*guaranteed_centre)[0]
                    
                    galIDs_gC = all_ids[ind_guarCen_first[subind_realCen]]
                    
                    gal_ids = np.unique(np.concatenate((gal_ids, galIDs_gC)))
                    print("... done, added {:d} particles (total now {:d})."
                          .format(len(gal_ids)-num_ids_orig, len(gal_ids)))


            tsC = time.time()
            counter_loadCentral += (tsC-tsB)

            # --- i) Infer z=0 indices of their particles

            gal_inds_all = all_revID[gal_ids]

            # Discard any particles that do not exist any more at z = 0
            # (this can happen if they got swallowed by a BH)
            
            # New: also discard particles that do not belong to the galaxy's z = 0 FOF
            #      (i.e. which are currently assigned to a galaxy whose central is
            #       not the central of this galaxy, at z = 0)
            #      This should not be common.

            ind_found = np.nonzero((gal_inds_all >= 0) & 
                                   (cenGal[all_galaxy[gal_inds_all], snap_z0] == cenGal[igal, snap_z0]))[0]

            n_ind_found = len(ind_found)
            print("Galaxy {:d}: found {:.2f} per cent of particles at z = 0."
                  .format(igal, n_ind_found/num_part_contrib*100))

            gal_inds = gal_inds_all[ind_found]


            # --- ii) Find bound particles (key part) ----
            
            gal_pos = all_pos[gal_inds, :]
            gal_vel = all_vel[gal_inds, :]
            gal_mass = all_mass[gal_inds].astype(np.float32)
            gal_energy = all_energy[gal_inds].astype(np.float32)
            
            # Now prepare to call MONK to unbind these particles.
            # In this version, we fix the galaxy position/velocity to that of the 
            # remnant at z = 0
            
            halopos_init = galPos_z0[igal, :]
            halovel_init = galVel_z0[igal, :]

            pos6d_halo = np.concatenate((gal_pos, gal_vel), axis = 1)

            tsD = time.time()
            counter_prepParticles += (tsD - tsC)

            #if igal == 5765: 
            #    set_trace()
            
            # Call MONK to find bound particles:
            ind_bound = monk.monk(pos6d_halo, gal_mass, gal_energy, halopos_init, halovel_init, 
                                  1, 0.005, -1, 1, 1, 0.1)  # mode, tol, maxGap, fixCentre, centreMode, centreFrac. 
            # Disable maxGap until fully finished
            
            # ind_bound contains the (sub-)indices that remain bound at the end
            
            n_bound = len(ind_bound)
            if n_bound > 0:
                print("Determined that {:d}/{:d} (={:.2f} per cent) of particles remain bound."
                      .format(n_bound, len(gal_inds), n_bound/len(gal_inds)*100))

                # --- iii) Mark bound (!!!) particles on full list
                all_galaxy[gal_inds[ind_bound]] = igal

            tsE = time.time()
            counter_monk += (tsE - tsD)

            print("Finished galaxy {:d} in {:.2f} sec."
                  .format(igal, time.time()-gstime))

        # Ends loop through galaxies in current snapshot
        print("")
        print("-----------------------------------------")
        print("Finished snapshot {:d} in {:.2f} min."
              .format(isnap, (time.time()-sstime)/60))
        print("")

        print("Load infall particles:  {:5.2f} sec."  .format(counter_loadInfall))
        print("  -- search:            {:5.2f} sec."  .format(counter_search))
        print("  -- actual load:       {:5.2f} sec."  .format(counter_actualLoadInfall))
        print("Load subfind particles: {:5.2f} sec."  .format(counter_loadSubfind))
        print("Load central particles: {:5.2f} sec."  .format(counter_loadCentral))
        print("Prepare particles:      {:5.2f} sec."  .format(counter_prepParticles))
        print("Unbinding with MONK:    {:5.2f} sec."  .format(counter_monk))
        print("-----------------------------------------")
        print("")

        counterA_snapSetup += counter_snapSetup
        counterA_loadInfall += counter_loadInfall
        counterA_search += counter_search
        counterA_actualLoadInfall += counter_actualLoadInfall
        counterA_loadSubfind += counter_loadSubfind
        counterA_loadCentral += counter_loadCentral
        counterA_prepParticles += counter_prepParticles
        counterA_monk += counter_monk
    

    # Ends loop through snapshots
    # The final bit is to re-compute the mass of all galaxies after the reassignment.
    
    argsort_gal = np.argsort(all_galaxy)
    gal_lims = np.arange(numGal+1, dtype = np.int) 
    splits_gal = np.searchsorted(all_galaxy, gal_lims, sorter = argsort_gal)
    
    for iigal, igal in enumerate(gal_alive_z0):
        subind_thisgal = argsort_gal[splits_gal[igal]:splits_gal[igal+1]] 
        type_thisgal = all_type[subind_thisgal]

        argsort_ptype = np.argsort(all_type[subind_thisgal])
        type_lims = np.arange(7, dtype = np.int)
        splits_ptype = np.searchsorted(type_thisgal, type_lims, sorter = argsort_ptype)

        for iptype in [0, 1, 4, 5]:
            ind_thisType = subind_thisgal[argsort_ptype[splits_ptype[iptype]:splits_ptype[iptype+1]]]
            mass_thisType = np.sum(all_mass[ind_thisType])
            sh_MassType[shi[igal, snap_z0], iptype] = mass_thisType

            
    # 3.) Write output
    # Output catalogues:
    # i) Galaxy-sorted offset list

    # This should now be simpler, because we don't keep non-galaxied particles any more
    yb.write_hdf5(all_ids[argsort_gal], outloc_sim, 'Galaxy/IDs', new = True)
    yb.write_hdf5(splits_gal, outloc_sim, 'Galaxy/Offset')
    
    # Old version below:
    #yb.write_hdf5(all_ids[argsort_gal[splits_gal[0]:splits_gal[-1]]], 
    #              outloc_sim, 'Galaxy/IDs', new = True)
    #yb.write_hdf5(splits_gal-splits_gal[0], outloc_sim, 'Galaxy/Offset')
    

    # ii) Subhalo-sorted offset list

    all_sh = shi[all_galaxy, snap_z0]
    if np.min(all_sh) < 0:
        print("Problem with conversion of galaxyID --> subhalo index")
        set_trace()

    argsort_sh = np.argsort(all_sh)
    sh_lims = np.arange(n_sh_z0+1, dtype = np.int) 
    splits_sh = np.searchsorted(all_sh, sh_lims, sorter = argsort_sh)
    
    yb.write_hdf5(all_sh[argsort_sh], outloc_sim, 'Subhalo/IDs')
    yb.write_hdf5(splits_sh, outloc_sim, 'Subhalo/Offset')
    yb.write_hdf5(sh_MassType, outloc_sim, 'Subhalo/MassType')

    # iii) ESP-aligned subhalo-index list
    # [Not yet implemented]
 
    print("")
    print("-----------------------------------------")
    
    print("Finished processing simulation CE-{:d} in {:.2f} min." 
          .format(isim, (time.time()-simstime)/60))

    print("Initial setup:          {:5.2f} sec."  .format(loadstime-simstime))
    print("Load z = 0 particles:   {:5.2f} sec."  .format(snapstime-loadstime))
    print("Load infall particles:  {:5.2f} sec."  .format(counterA_loadInfall))
    print("  -- search:            {:5.2f} sec."  .format(counterA_search))
    print("  -- actual load:       {:5.2f} sec."  .format(counterA_actualLoadInfall))

    print("Load subfind particles: {:5.2f} sec."  .format(counterA_loadSubfind))
    print("Load central particles: {:5.2f} sec."  .format(counterA_loadCentral))
    print("Prepare particles:      {:5.2f} sec."  .format(counterA_prepParticles))
    print("Unbinding with MONK:    {:5.2f} sec."  .format(counterA_monk))
    print("-----------------------------------------")
    print("")

print("Done!")
