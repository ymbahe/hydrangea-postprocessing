"""
Determine whether each particle at z = 0 is in-situ, accreted, or stolen

Started 21-02-2018

Version adapted 18-Sep-2018, which also splits particles by galaxy root mass
"""

import numpy as np
import sim_tools as st
import yb_utils as yb
from astropy.io import ascii
from pdb import set_trace
import calendar
import time
from mpi4py import MPI
import os
import hydrangea_tools as ht
import eagle_routines as er

simname = "Hydrangea"

if simname == "Hydrangea":
    n_sim = 30
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    snap0 = 29
else:
    n_sim = 1
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    snap0 = 28

# Input catalogues from stellar_birth_props.py 
# (have ID, root galaxy, and root snapshot for all particles)
#catname = 'highlev/ParticleRootInfo_MinRatio1p20Curr_withRootSnap.hdf5'
catname = 'highlev/ParticleRootInfo_MinRatioNone_withRootSnap.hdf5'

# Special catalogue without min-ratio, for *some purpose*
catname_None = 'highlev/ParticleRootInfo_MinRatioNone_withRootSnap.hdf5'

# Output catalogue: total masses in individual origin categories and masses,
# for multiple radii in each galaxy
#outname = 'highlev/ParticleOriginFlags_RadMassBins_MinRatioNone_MinMstar9p0_MassAtPeak_parentsOnlyForIS_18Mar19_Mstar11p0_withDM.hdf5'
#outname = 'highlev/ParticleOriginFlags_RadMassBins_MinRatio1p20_MinMstar9p0_MassAtRoot_parentsOnlyForIS-MRNone_14Mar19_Mstar11p0_withDM.hdf5'
outname = 'highlev/ParticleOriginFlags_RadMassBins_MinRatioNone_MinMstar9p0_MassAtRoot_parentsOnlyForIS_10Apr19_Mstar11p0_withDM_stellarMass_Cantor.hdf5'

runtype = "HYDRO"

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

min_mstar = 9.0   # If not None, threshold between merged and quasi-stripped
min_contrastFrac = None   # Does nothing
min_remainFrac = None     # Does nothing
min_mstar_include = None  # Minimum stellar mass to include galaxy

subhaloType = 'Cantor'  # Does not yet do anything useful...

initialMassForStars = False  # True --> Use initial mass for stars (and initial baryon mass for gas)
massBinningByStars = False # True --> Use stellar (not total) mass to characterize root galaxy.
massBinningAtRoot = True # True --> Use mass at root snapshot (not peak)
fullParentAnalysis = False # True --> compute parent-origin masses for all particles. False --> do this only for stars that are (themselves) in-situ.
metalMassOnly = False  # True --> Count only metal mass (False: total stellar mass)

# Specify radial and mass bins
radType = 'abs'   # 'rel' --> r/r200 or 'abs' --> r/Mpc (currently ignored)
radlims = np.logspace(-3, 0.5, num=36, endpoint = True)
radlims = np.concatenate((np.array([0]), radlims))
masslims = np.linspace(10.0, 15.4, num = 28, endpoint = True)
masslims = np.concatenate((np.array([0]), masslims))

nrad = len(radlims)-1
nmass = len(masslims)-1

# Main loop through individual simulations:
for isim in range(n_sim):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo CE-{:d}" .format(isim))
    print("**************************")
    print("")
    
    # Set up locations of input catalogues
 
    if simname == "Hydrangea":
        rundir = basedir + '/CE-' + str(isim) + '/' + runtype
        catloc_None = ht.clone_dir(rundir) + '/' + catname_None
        catloc = ht.clone_dir(rundir) + '/' + catname
        hldir = rundir + '/highlev/' 
        outloc = ht.clone_dir(rundir) + '/' + outname
    else:
        rundir = basedir
        catloc = er.clone_dir(rundir) + '/' + catname
        outloc = er.clone_dir(rundir) + '/' + outname
        hldir = er.clone_dir(rundir) + '/highlev/'

    if not os.path.exists(rundir):
        continue

    spiderloc = hldir + '/SpiderwebTables.hdf5'
    fgtloc = hldir + '/FullGalaxyTables.hdf5'
    posloc = hldir + '/GalaxyPositionsSnap.hdf5'

    # Write meta-data output already now:
    yb.write_hdf5_attribute(outloc, "Header", "CatalogueID", yb.read_hdf5_attribute(catloc, "Header", "CatalogueID"), new = True)
    
    yb.write_hdf5(radlims, outloc, "BinEdges", comment = "Radii of bin edges")
    #yb.write_hdf5(np.array([radType]), outloc, "RadType", comment = "Absolute or r/r200")

    yb.write_hdf5(masslims, outloc, "MassBinEdges", comment = "Masses of bin edges")

    if massBinningByStars:
        massBinningType = b'Stars'
    else:
        massBinningType = b'Total' 
    yb.write_hdf5_attribute(outloc, "Header", "MassBinningType", massBinningType)

    if massBinningAtRoot:
        massBinningTime = b'Root'
    else:
        massBinningTime = b'Peak'
    yb.write_hdf5_attribute(outloc, "Header", "MassBinningTime", massBinningTime)

    if metalMassOnly:
        massType = b'Metals'
    else:
        massType = b'All'
    yb.write_hdf5_attribute(outloc, 'Header', 'MassType', massType)

    subdir, snapdir, partdir = st.form_files(rundir, snap0, 'sub snap subpart')

    # Load info about subhaloes and glaxies
    fof_fsh = st.eagleread(subdir, 'FOF/FirstSubhaloID', astro = False)
    len_ptype = st.eagleread(subdir, 'Subhalo/SubLengthType', astro = False)
    nsh = len_ptype.shape[0]
    
    gal_mstar_full = yb.read_hdf5(fgtloc, 'Mstar')
    gal_mstar = gal_mstar_full[:,-1]
    gal_mstar_peak = yb.read_hdf5(fgtloc, 'Full/Mstar')
    gal_mtot_full = yb.read_hdf5(fgtloc, 'Msub')
    gal_mtot_peak = yb.read_hdf5(fgtloc, 'Full/Msub')

    gal_z0 = yb.read_hdf5(spiderloc, "Subhalo/Snapshot_" + str(snap0).zfill(3) + "/Galaxy")
    mergelist = yb.read_hdf5(spiderloc, "MergeList")
    shilist = yb.read_hdf5(spiderloc, 'SubHaloIndex')[:, -1]
    cenGal = yb.read_hdf5(fgtloc, 'CenGal')
    
    sh_mstar = gal_mstar[gal_z0]
    sh_pos = yb.read_hdf5(posloc, 'Centre')[gal_z0, snap0, :]

    # Now loop through particle types to process:
    for iptype in [1, 4]:

        print("")
        print("---- Processing type {:d} ----" .format(iptype))
        print("")

        # Read informatin from particle-parent catalogue
        # (Particle ID, root galaxy, and root snapshot)
        cat_id = yb.read_hdf5(catloc, 'PartType{:d}/ParticleIDs' .format(iptype))
        cat_rootGal = yb.read_hdf5(catloc, 'PartType{:d}/RootGalaxy' .format(iptype))
        cat_rootSnap = yb.read_hdf5(catloc, 'PartType{:d}/RootSnapshot' .format(iptype))
        
        # For stars, there are two additional datasets to read in:
        # (i)  the root info for the particles' parent gas particle
        # (ii) the root info without mass threshold

        if iptype == 4:
            
            cat_parentRootGal = yb.read_hdf5(catloc, 'PartType{:d}/ParentRootGalaxy' .format(iptype))
            cat_parentRootSnap = yb.read_hdf5(catloc, 'PartType{:d}/ParentRootSnapshot' .format(iptype))

            cat_RootGal_None = yb.read_hdf5(catloc_None, 'PartType{:d}/RootGalaxy' .format(iptype))
            cat_RootSnap_None = yb.read_hdf5(catloc_None, 'PartType{:d}/RootSnapshot' .format(iptype))
            cat_None_id = yb.read_hdf5(catloc_None, 'PartType{:d}/ParticleIDs' .format(iptype))

        # Load z = 0 particle IDs
        #snap_id = st.eagleread(snapdir, 'PartType{:d}/ParticleIDs' .format(iptype), astro = False)
        part_id = st.eagleread(partdir, 'PartType{:d}/ParticleIDs' .format(iptype), astro = False)

        # Load z = 0 particle masses
        if iptype == 1:
            part_mass = np.zeros(len(part_id), dtype = np.float32)+st.m_dm(rundir)
        elif iptype == 4:
            if initialMassForStars:
                part_mass = st.eagleread(partdir, 'PartType{:d}/InitialMass' .format(iptype), astro = True)[0]
            else:
                part_mass = st.eagleread(partdir, 'PartType{:d}/Mass' .format(iptype), astro = True)[0]

            if metalMassOnly:
                part_zmet = st.eagleread(partdir, 'PartType{:d}/Metallicity' .format(iptype), astro = True)[0]
                part_mass *= part_zmet

        elif iptype == 0:
            if initialMassForStars:
                part_mass = np.zeros(len(part_id), dtype = np.float32)+st.m_bar(rundir)
            else:
                part_mass = st.eagleread(partdir, 'PartType{:d}/Mass' .format(iptype), astro = True)[0]

        else:
            part_mass = st.eagleread(partdir, 'PartType{:d}/Mass' .format(iptype), astro = True)[0]

        
        # In this version, we only match to PART
        # In principle, one could also match to SNAP directly (based on 
        # e.g. distance from galaxy), but that would require dealing with
        # multiple galaxies claiming the same particle)

        # cat_ind_snap, matched_snap = yb.find_id_indices(snap_id, cat_id)
        
        cat_ind_part, matched_part = yb.find_id_indices(part_id, cat_id)
       
        if iptype == 4:
            cat_None_ind_part, matched_part_None = yb.find_id_indices(part_id, cat_None_id)
        
        # To port back to SNAP, match SNAP <-> PART
        #snap_ind_part, matched_sp = yb.find_id_indices(part_id, snap_id)

        # Sanity checks that all particles have been located
        if cat_ind_part.min() < 0:
            print("Why can some particles not be matched?")
            set_trace()

        if iptype == 4:
            if cat_None_ind_part.min() < 0:
                print("Why can some particles not be matched in None-catalogue?")
                set_trace()

        #if snap_ind_part.min() < 0:
        #    print("Why can some ESP particles not be matched to snap?")
        #    set_trace()

        # Reconstruct the SHI of each particle from PART
        part_shi = np.zeros(len(part_id), dtype = np.int32)-1
        part_fof = st.eagleread(partdir, 'PartType{:d}/GroupNumber' .format(iptype), astro = False)-1
        part_sgn = st.eagleread(partdir, 'PartType{:d}/SubGroupNumber' .format(iptype), astro = False)

        ind_in_sh = np.nonzero((part_fof >= 0) & (part_sgn < 2**30))[0]
        part_shi[ind_in_sh] = fof_fsh[part_fof[ind_in_sh]]+part_sgn[ind_in_sh]

        # Load positions of all PART particles
        part_pos = st.eagleread(partdir, 'PartType{:d}/Coordinates' .format(iptype), astro = True)[0]

        # Translate SHI (at z = 0) --> galID
        part_gal = np.zeros(len(part_id), dtype = np.int32)-1
        part_gal[ind_in_sh] = gal_z0[part_shi[ind_in_sh]]

        # Port SHI and galID to SNAP
        #snap_shi = np.zeros(len(snap_id), dtype = np.int32)-1
        #snap_gal = np.zeros(len(snap_id), dtype = np.int32)-1

        #snap_shi[snap_ind_part] = part_shi
        #ind_in_sh_snap = np.nonzero(snap_shi >= 0)[0]
        #snap_gal[ind_in_sh_snap] = gal_z0[snap_shi[ind_in_sh_snap]]
        
        # Now loop through 'standard' / 'parent' mode
        for itype in range(2):   # 0=standard, 1=parent

            if iptype != 4 and itype == 1:
                continue

            print(" -- Processing subtype {:d} -- " .format(itype))

            if itype == 0:
                #snap_rootgal = cat_rootGal[cat_ind_snap]
                part_rootgal = cat_rootGal[cat_ind_part]
                part_rootsnap = cat_rootSnap[cat_ind_part]
            elif itype == 1:
                #snap_rootgal = cat_parentRootGal[cat_ind_snap]
                part_rootgal = cat_parentRootGal[cat_ind_part]
                part_rootsnap = cat_parentRootSnap[cat_ind_part]
                part_rootgal_star_None = cat_RootGal_None[cat_None_ind_part]
                part_rootsnap_star_None = cat_RootSnap_None[cat_None_ind_part]
            else:
                print("Unexpected itype")
                set_trace()
                
                
            # New bit added 25-Sep-2018: load root mass for each galaxy
            part_rootMass = np.zeros(part_rootgal.shape[0], dtype = np.float32)-np.inf
            ind_inRootGal = np.nonzero(part_rootgal >= 0)[0]

            if not massBinningByStars:
                if massBinningAtRoot:
                    part_rootMass[ind_inRootGal] = gal_mtot_full[part_rootgal[ind_inRootGal], part_rootsnap[ind_inRootGal]]
                else:
                    part_rootMass[ind_inRootGal] = gal_mtot_peak[part_rootgal[ind_inRootGal]]
            else:
                if massBinningAtRoot:
                    part_rootMass[ind_inRootGal] = gal_mstar_full[part_rootgal[ind_inRootGal], part_rootsnap[ind_inRootGal]]
                else:
                    part_rootMass[ind_inRootGal] = gal_mstar_peak[part_rootgal[ind_inRootGal]]
                    

            #ind_insitu_snap = np.nonzero((snap_rootgal == snap_gal) & (snap_gal >= 0))[0]
            ind_insitu_part = np.nonzero((part_rootgal == part_gal) & (part_gal >= 0))[0]

            if itype == 1:
                ind_insitu_part_star = np.nonzero((part_rootgal_star_None == part_gal) & (part_gal >= 0))[0]
    
            ind_tracefail = np.nonzero(part_rootgal < 0)[0]

            #ind_acc_snap = np.nonzero((snap_gal >= 0) & (snap_rootgal != snap_gal) & (mergelist[snap_rootgal, -1] == snap_gal))[0]
            ind_acc_part = np.nonzero((part_gal >= 0) & (part_rootgal >= 0) & (part_rootgal != part_gal) & (mergelist[part_rootgal, -1] == part_gal))[0]

            # The "new fun" (Sat 24-Feb-18) will happen here.
            # Break down "the rest" into following categories:

            # - "quasi-merged": stolen from something in same FOF below min criterion
            # - "stripped": stolen from something in same FOF ABOVE min crit.
            # - "stolen": stolen from something in OTHER FOF
            # - "adopted": stolen from something that is no longer alive

            #ind_other_snap = np.nonzero((snap_gal >= 0) & (mergelist[snap_rootgal, -1] != snap_gal))[0]
            ind_other_part = np.nonzero((part_gal >= 0) & (part_rootgal >= 0) & (mergelist[part_rootgal, -1] != part_gal))[0]
            
            #subind_dead_snap = np.nonzero(shilist[snap_rootgal[ind_other_snap]] < 0)[0]
            #subind_alive_snap = np.nonzero(shilist[snap_rootgal[ind_other_snap]] >= 0)[0]
            #subind_otherfof_snap = np.nonzero(cenGal[snap_rootgal[ind_other_snap[subind_alive_snap]], -1] != cenGal[snap_gal[ind_other_snap[subind_alive_snap]], -1])[0]
            #subind_samefof_snap = np.nonzero(cenGal[snap_rootgal[ind_other_snap[subind_alive_snap]], -1] == cenGal[snap_gal[ind_other_snap[subind_alive_snap]], -1])[0]
            #if min_mstar is not None:
            #    subind_stripped_snap = np.nonzero(sh_mstar[shilist[snap_gal[subind_alive_snap[subind_samefof_snap]]]] >= min_mstar)[0]
            #    subind_qm_snap = np.nonzero(sh_mstar[shilist[snap_gal[subind_alive_snap[subind_samefof_snap]]]] < min_mstar)[0]

            # And again, this time for part...

            # - "quasi-merged": stolen from something in same FOF below min criterion
            # - "stripped": stolen from something in same FOF ABOVE min crit.
            # - "stolen": stolen from something in OTHER FOF
            # - "adopted": stolen from something that is no longer alive

            subind_dead_part = np.nonzero(shilist[part_rootgal[ind_other_part]] < 0)[0]
            subind_alive_part = np.nonzero(shilist[part_rootgal[ind_other_part]] >= 0)[0]
            subind_otherfof_part = np.nonzero(cenGal[part_rootgal[ind_other_part[subind_alive_part]], -1] != cenGal[part_gal[ind_other_part[subind_alive_part]], -1])[0]
            subind_samefof_part = np.nonzero(cenGal[part_rootgal[ind_other_part[subind_alive_part]], -1] == cenGal[part_gal[ind_other_part[subind_alive_part]], -1])[0]
            if min_mstar is not None:
                subind_stripped_part = np.nonzero(sh_mstar[shilist[part_gal[subind_alive_part[subind_samefof_part]]]] >= min_mstar)[0]
                subind_qm_part = np.nonzero(sh_mstar[shilist[part_gal[subind_alive_part[subind_samefof_part]]]] < min_mstar)[0]


            # --- END OF COPIED BIT ---- 

            #code_snap = np.zeros(len(snap_id), dtype = np.int8)-1
            code_part = np.zeros(len(part_id), dtype = np.int8)-1
            
            if itype == 1:
                code_part_star = np.zeros(len(part_id), dtype = np.int8)-1
                code_part_star[ind_insitu_part_star] = 0

            # 'Snap' only left as stub for now
            #code_snap[ind_insitu_snap] = 0
            #code_snap[ind_acc_snap] = 1
            #code_snap[ind_stolen_snap] = 2
            
            code_part[ind_insitu_part] = 0
            code_part[ind_acc_part] = 1
            code_part[ind_other_part[subind_alive_part[subind_samefof_part[subind_qm_part]]]] = 2
            code_part[ind_other_part[subind_alive_part[subind_samefof_part[subind_stripped_part]]]] = 3
            code_part[ind_other_part[subind_alive_part[subind_otherfof_part]]] = 4
            code_part[ind_other_part[subind_dead_part]] = 5
            code_part[ind_tracefail] = 6
            
            print("Preparing to compute masses for subhaloes...")

            # Prepare index lists to quickly find all particles
            # in each subhalo
            argsort_part = np.argsort(part_shi[ind_in_sh])
            offset_part = np.zeros(nsh+1, dtype = int)
            offset_part[1:] = np.cumsum(len_ptype[:, iptype])
        
            # Set up output lists
            sh_m_insitu = np.zeros((nsh, nrad, nmass))
            sh_m_acc = np.zeros((nsh, nrad, nmass))
            sh_m_qm = np.zeros((nsh, nrad, nmass))
            sh_m_stripped = np.zeros((nsh, nrad, nmass))
            sh_m_otherfof = np.zeros((nsh, nrad, nmass))
            sh_m_dead = np.zeros((nsh, nrad, nmass))
            sh_m_tracefail = np.zeros((nsh, nrad, nmass))

            # Now go through all subhaloes...
            for ish in range(nsh):

                #if ish == 0 and iptype == 4: set_trace()

                if ish % 1000 == 0:
                    print("Reached subhalo {:d}/{:d}..." .format(ish, nsh))

                # Temporary shortcut: only compute values for massive gals
                if min_mstar_include is not None:
                    if sh_mstar[ish] < min_mstar_include: continue

                # Particles belonging to current SH
                ind_this = ind_in_sh[argsort_part[offset_part[ish]:offset_part[ish+1]]]
                if len(ind_this) == 0:
                    continue

                # In parent-mode, only consider particles that are classed
                # as in-situ (without threshold) from stars-only
                if itype == 1:  
                    subind_this = np.nonzero(code_part_star[ind_this] == 0)[0]
                    if len(subind_this) == 0:
                        continue
                    ind_this = ind_this[subind_this]

                if np.count_nonzero(part_shi[ind_this] != ish) > 0:
                    print("Unexpected SHI in particle list")
                    set_trace()
                    
                if np.count_nonzero(code_part[ind_this] < 0) > 0:
                    print("Particle has not been assigned a code?!")
                    set_trace()

                # Loop through mass bins
                this_mass = part_rootMass[ind_this]
                argsort_mass = np.argsort(this_mass)
                splits_mass = np.searchsorted(this_mass, masslims, sorter = argsort_mass)
                for imass in range(nmass):
                    subind_thismass = argsort_mass[splits_mass[imass]:splits_mass[imass+1]] 
                
                    # Loop through radial bins

                    cen_this = sh_pos[ish, :]
                    this_relpos = part_pos[ind_this[subind_thismass], :] - cen_this[None, :]
                    this_rad = np.linalg.norm(this_relpos, axis = 1)
                    
                    argsort_rad = np.argsort(this_rad)
                    splits_rad = np.searchsorted(this_rad, radlims, sorter = argsort_rad)
                    for irad in range(nrad):
                    
                        subind_thisrad = argsort_rad[splits_rad[irad]:splits_rad[irad+1]] 

                        ind_insitu = np.nonzero(code_part[ind_this[subind_thismass[subind_thisrad]]] == 0)[0]
                        ind_acc = np.nonzero(code_part[ind_this[subind_thismass[subind_thisrad]]] == 1)[0]
                        ind_qm = np.nonzero(code_part[ind_this[subind_thismass[subind_thisrad]]] == 2)[0]
                        ind_stripped = np.nonzero(code_part[ind_this[subind_thismass[subind_thisrad]]] == 3)[0]
                        ind_otherfof = np.nonzero(code_part[ind_this[subind_thismass[subind_thisrad]]] == 4)[0]
                        ind_dead = np.nonzero(code_part[ind_this[subind_thismass[subind_thisrad]]] == 5)[0]
                        ind_tracefail = np.nonzero(code_part[ind_this[subind_thismass[subind_thisrad]]] == 6)[0]

                        sh_m_insitu[ish, irad, imass] = np.sum(part_mass[ind_this[subind_thismass[subind_thisrad[ind_insitu]]]])
                        sh_m_acc[ish, irad, imass] = np.sum(part_mass[ind_this[subind_thismass[subind_thisrad[ind_acc]]]])
                        sh_m_qm[ish, irad, imass] = np.sum(part_mass[ind_this[subind_thismass[subind_thisrad[ind_qm]]]])
                        sh_m_stripped[ish, irad, imass] = np.sum(part_mass[ind_this[subind_thismass[subind_thisrad[ind_stripped]]]])
                        sh_m_otherfof[ish, irad, imass] = np.sum(part_mass[ind_this[subind_thismass[subind_thisrad[ind_otherfof]]]])
                        sh_m_dead[ish, irad, imass] = np.sum(part_mass[ind_this[subind_thismass[subind_thisrad[ind_dead]]]])
                        sh_m_tracefail[ish, irad, imass] = np.sum(part_mass[ind_this[subind_thismass[subind_thisrad[ind_tracefail]]]])

            if itype == 1:
                coda = "ParentGas"
            else:
                coda = ""

            pt = "PartType{:d}" .format(iptype)

            #yb.write_hdf5(code_snap, outloc, pt + "/OriginSnap" + coda, comment = "Origin code of each particle from snapshot file. -1: not in subhalo, 0: in-situ, 1: accreted, 2: stolen")

            yb.write_hdf5(code_part, outloc, pt + "/OriginESP" + coda, comment = "Origin code of each particle from eagle_subfind_particles. -1: not in subhalo, 0: in-situ, 1: accreted, 2: stolen")
    
            yb.write_hdf5(sh_m_insitu, outloc, pt + "/MassInSitu" + coda, comment = "Mass formed in-situ")
    
            yb.write_hdf5(sh_m_acc, outloc, pt + "/MassAccreted" + coda, comment = "Mass accreted through mergers")

            yb.write_hdf5(sh_m_qm, outloc, pt + "/MassQuasiMerged" + coda, comment = "Mass from galaxies that survive, but below mass threshold")

            yb.write_hdf5(sh_m_stripped, outloc, pt + "/MassStripped" + coda, comment = "Mass from galaxies that survive, and are above mass threshold")

            yb.write_hdf5(sh_m_otherfof, outloc, pt + "/MassOtherFOF" + coda, comment = "Mass from galaxies in other FOF group")

            yb.write_hdf5(sh_m_dead, outloc, pt + "/MassDead" + coda, comment = "Mass from galaxies that are disrupted")

            yb.write_hdf5(sh_m_tracefail, outloc, pt + "/MassTracefail" + coda, comment = "Mass from untraceable particles")

    
        
print("Done!")
