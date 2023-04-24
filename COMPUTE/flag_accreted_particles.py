"""
Determine whether each particle at z = 0 is in-situ, accreted, or stolen

Output is stored accumulated for individual subhaloes. Current development
work is on the sibling code 'flag_accreted_particles_massbin.py'.

Started 21-02-2018
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

catname_None = 'highlev/ParticleRootInfo.hdf5'  #_MinRatioNone.hdf5'#_MinRatio1-20.hdf5'
catname = 'highlev/ParticleRootInfo_MinRatio1b10_Curr.hdf5'#_MinRatio1-20.hdf5'
outname = 'highlev/ParticleOriginFlags_NoRadBins_MinRatio1b10current_MinMstar9p0_PGForInsitu.hdf5'

runtype = "HYDRO"

comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

min_mstar = 9.0
min_contrastFrac = None
min_remainFrac = None

#radlims = np.logspace(np.log10(0.03), np.log10(10.0), num=20, endpoint = True)
#radlims = np.concatenate((np.array([0]), radlims))
radlims = np.array((0, 10.0))

nrad = len(radlims)-1

for isim in range(n_sim):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    hstime = time.time()

    print("")
    print("**************************")
    print("Now processing halo F{:d}" .format(isim))
    print("**************************")
    print("")
     
    if simname == "Hydrangea":
        rundir = basedir + '/HaloF' + str(isim) + '/' + runtype
        outloc = ht.clone_dir(rundir) + '/' + outname
        catloc_None = rundir + '/' + catname_None
        catloc = ht.clone_dir(rundir) + '/' + catname
        hldir = rundir + '/highlev/' 
    else:
        rundir = basedir
        outloc = er.clone_dir(rundir) + '/' + outname
        catloc = er.clone_dir(rundir) + '/' + catname
        hldir = er.clone_dir(rundir) + '/highlev/'

    if not os.path.exists(rundir):
        continue

    spiderloc = hldir + '/SpiderwebTables.hdf5'
    fgtloc = hldir + '/FullGalaxyTable.hdf5'

    subdir, snapdir, partdir = st.form_files(rundir, snap0, 'sub snap subpart')

    fof_fsh = st.eagleread(subdir, 'FOF/FirstSubhaloID', astro = False)
    len_ptype = st.eagleread(subdir, 'Subhalo/SubLengthType', astro = False)

    sh_pos = st.eagleread(subdir, 'Subhalo/CentreOfPotential', astro = True)[0] 

    nsh = len_ptype.shape[0]
    gal_mstar = yb.read_hdf5(fgtloc, 'Mstar')[:,-1]
    
    yb.write_hdf5_attribute(outloc, "Header", "CatalogueID", yb.read_hdf5_attribute(catloc, "Header", "CatalogueID"), new = True)
    
    yb.write_hdf5(radlims, outloc, "BinEdges", comment = "Radii of bin edges, in Mpc")

    gal_z0 = yb.read_hdf5(spiderloc, "Subhalo/Snapshot_" + str(snap0).zfill(3) + "/Galaxy")
    mergelist = yb.read_hdf5(spiderloc, "MergeList")
    shilist = yb.read_hdf5(spiderloc, 'SubHaloIndex')[:, -1]
    cenGal = yb.read_hdf5(fgtloc, 'CenGal')

    sh_mstar = gal_mstar[gal_z0]

    for iptype in [4]:


        print("")
        print("---- Processing type {:d} ----" .format(iptype))
        print("")

        cat_id = yb.read_hdf5(catloc, 'PartType{:d}/ParticleIDs' .format(iptype))
        cat_rootGal = yb.read_hdf5(catloc, 'PartType{:d}/RootGalaxy' .format(iptype))
    
        if iptype == 4:
            cat_parentRootGal = yb.read_hdf5(catloc, 'PartType{:d}/ParentRootGalaxy' .format(iptype))
            cat_RootGal_None = yb.read_hdf5(catloc_None, 'PartType{:d}/RootGalaxy' .format(iptype))
            cat_None_id = yb.read_hdf5(catloc_None, 'PartType{:d}/ParticleIDs' .format(iptype))

            #if np.max(np.abs(cat_RootGalId - cat_id)) > 0:
            #    print("Inconsistent particle IDs in catalogues...")
            #    set_trace()

        snap_id = st.eagleread(snapdir, 'PartType{:d}/ParticleIDs' .format(iptype), astro = False)
        part_id = st.eagleread(partdir, 'PartType{:d}/ParticleIDs' .format(iptype), astro = False)
        
        if iptype == 1:
            part_mass = np.zeros(len(snap_id), dtype = np.float32)+st.m_dm(rundir)
        elif iptype == 4:
            part_mass = st.eagleread(partdir, 'PartType{:d}/InitialMass' .format(iptype), astro = True)[0]
        elif iptype == 0:
            part_mass = np.zeros(len(snap_id), dtype = np.float32)+st.m_bar(rundir)
        else:
            part_mass = st.eagleread(partdir, 'PartType{:d}/Mass' .format(iptype), astro = True)[0]

        
        cat_ind_snap, matched_snap = yb.find_id_indices(snap_id, cat_id)
        cat_ind_part, matched_part = yb.find_id_indices(part_id, cat_id)
        cat_None_ind_part, matched_part_None = yb.find_id_indices(part_id, cat_None_id)

        #cat_id_rev = st.create_reverse_list(cat_id, maxval = np.max(snap_id)+1)
        #snap_id_rev = st.create_reverse_list(snap_id, maxval = np.max(part_id)+1)
        snap_ind_part, matched_sp = yb.find_id_indices(part_id, snap_id)

        #cat_ind_snap = cat_id_rev[snap_id]
        #cat_ind_part = cat_id_rev[part_id]

        if cat_ind_part.min() < 0:
            print("Why can some particles not be matched?")
            set_trace()

        if cat_None_ind_part.min() < 0:
            print("Why can some particles not be matched in None-catalogue?")
            set_trace()

        if snap_ind_part.min() < 0:
            print("Why can some ESP particles not be matched to snap?")
            set_trace()

        part_shi = np.zeros(len(part_id), dtype = np.int32)-1
        part_fof = np.abs(st.eagleread(partdir, 'PartType{:d}/GroupNumber' .format(iptype), astro = False))-1
        part_sgn = st.eagleread(partdir, 'PartType{:d}/SubGroupNumber' .format(iptype), astro = False)
        ind_in_sh = np.nonzero(part_sgn < 2**30)[0]
        part_shi[ind_in_sh] = fof_fsh[part_fof[ind_in_sh]]+part_sgn[ind_in_sh]

        part_pos = st.eagleread(partdir, 'PartType{:d}/Coordinates' .format(iptype), astro = True)[0]


        snap_shi = np.zeros(len(snap_id), dtype = np.int32)-1
        snap_shi[snap_ind_part] = part_shi
    
        part_gal = np.zeros(len(part_id), dtype = np.int32)-1
        part_gal[ind_in_sh] = gal_z0[fof_fsh[part_fof[ind_in_sh]]+part_sgn[ind_in_sh]]

        snap_gal = np.zeros(len(snap_id), dtype = np.int32)-1
        ind_in_sh_snap = np.nonzero(snap_shi >= 0)[0]
        snap_gal[ind_in_sh_snap] = gal_z0[snap_shi[ind_in_sh_snap]]
 
        for itype in range(2):

            if iptype != 4 and itype == 1:
                continue

            print(" -- Processing subtype {:d} -- " .format(itype))

            if itype == 0:
                snap_rootgal = cat_rootGal[cat_ind_snap]
                part_rootgal = cat_rootGal[cat_ind_part]
            elif itype == 1:
                snap_rootgal = cat_parentRootGal[cat_ind_snap]
                part_rootgal = cat_parentRootGal[cat_ind_part]
                part_rootgal_star_None = cat_RootGal_None[cat_None_ind_part]
            else:
                print("Unexpected itype")
                set_trace()

            ind_insitu_snap = np.nonzero((snap_rootgal == snap_gal) & (snap_gal >= 0))[0]
            ind_insitu_part = np.nonzero((part_rootgal == part_gal) & (part_gal >= 0))[0]

            if itype == 1:
                ind_insitu_part_star = np.nonzero((part_rootgal_star_None == part_gal) & (part_gal >= 0))[0]
    
            ind_acc_snap = np.nonzero((snap_gal >= 0) & (snap_rootgal != snap_gal) & (mergelist[snap_rootgal, -1] == snap_gal))[0]
            ind_acc_part = np.nonzero((part_gal >= 0) & (part_rootgal != part_gal) & (mergelist[part_rootgal, -1] == part_gal))[0]

            # The "new fun" (Sat 24-Feb-18) will happen here.
            # Break down "the rest" into following categories:

            # - "quasi-merged": stolen from something in same FOF below min criterion
            # - "stripped": stolen from something in same FOF ABOVE min crit.
            # - "stolen": stolen from something in OTHER FOF
            # - "adopted": stolen from something that is no longer alive

            ind_other_snap = np.nonzero((snap_gal >= 0) & (mergelist[snap_rootgal, -1] != snap_gal))[0]
            ind_other_part = np.nonzero((part_gal >= 0) & (mergelist[part_rootgal, -1] != part_gal))[0]
            
            subind_dead_snap = np.nonzero(shilist[snap_rootgal[ind_other_snap]] < 0)[0]
            subind_alive_snap = np.nonzero(shilist[snap_rootgal[ind_other_snap]] >= 0)[0]
            subind_otherfof_snap = np.nonzero(cenGal[snap_rootgal[ind_other_snap[subind_alive_snap]], -1] != cenGal[snap_gal[ind_other_snap[subind_alive_snap]], -1])[0]
            subind_samefof_snap = np.nonzero(cenGal[snap_rootgal[ind_other_snap[subind_alive_snap]], -1] == cenGal[snap_gal[ind_other_snap[subind_alive_snap]], -1])[0]
            if min_mstar is not None:
                subind_stripped_snap = np.nonzero(sh_mstar[shilist[snap_gal[subind_alive_snap[subind_samefof_snap]]]] >= min_mstar)[0]
                subind_qm_snap = np.nonzero(sh_mstar[shilist[snap_gal[subind_alive_snap[subind_samefof_snap]]]] < min_mstar)[0]

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


            code_snap = np.zeros(len(snap_id), dtype = np.int8)-1
            code_part = np.zeros(len(part_id), dtype = np.int8)-1
            
            if itype == 1:
                code_part_star = np.zeros(len(part_id), dtype = np.int8)-1
                code_part_star[ind_insitu_part_star] = 0

            code_snap[ind_insitu_snap] = 0
            code_snap[ind_acc_snap] = 1
            #code_snap[ind_stolen_snap] = 2
            
            code_part[ind_insitu_part] = 0
            code_part[ind_acc_part] = 1
            code_part[ind_other_part[subind_alive_part[subind_samefof_part[subind_qm_part]]]] = 2
            code_part[ind_other_part[subind_alive_part[subind_samefof_part[subind_stripped_part]]]] = 3
            code_part[ind_other_part[subind_alive_part[subind_otherfof_part]]] = 4
            code_part[ind_other_part[subind_dead_part]] = 5
            
            print("Preparing to compute masses for subhaloes...")

        
            argsort_part = np.argsort(part_shi[ind_in_sh])
            offset_part = np.zeros(nsh+1, dtype = int)
            offset_part[1:] = np.cumsum(len_ptype[:, iptype])
        
            sh_m_insitu = np.zeros((nsh, nrad))
            sh_m_acc = np.zeros((nsh, nrad))
            sh_m_qm = np.zeros((nsh, nrad))
            sh_m_stripped = np.zeros((nsh, nrad))
            sh_m_otherfof = np.zeros((nsh, nrad))
            sh_m_dead = np.zeros((nsh, nrad))

            for ish in range(nsh):

                if ish % 1000 == 0:
                    print("Reached subhalo {:d}/{:d}..." .format(ish, nsh))

                ind_this = ind_in_sh[argsort_part[offset_part[ish]:offset_part[ish+1]]]
                if len(ind_this) == 0:
                    continue

                if itype == 1:
                    subind_this = np.nonzero(code_part_star[ind_this] == 0)[0]
                    if len(subind_this) == 0:
                        continue
                    ind_this = ind_this[subind_this]

                if np.count_nonzero(part_shi[ind_this] != ish) > 0:
                    print("Unexpected SHI in particle list")
                    set_trace()
                    
                if np.count_nonzero(code_part[ind_this] < 0) > 0:
                    print("Particle in subhalo has no subhalo ID?")
                    set_trace()

                # Loop through radial bins

                cen_this = sh_pos[ish, :]
                this_relpos = part_pos[ind_this, :] - cen_this[None, :]
                this_rad = np.linalg.norm(this_relpos, axis = 1)

                argsort_rad = np.argsort(this_rad)
                splits_rad = np.searchsorted(this_rad, radlims, sorter = argsort_rad)
                for irad in range(nrad):
                    
                    subind_thisrad = argsort_rad[splits_rad[irad]:splits_rad[irad+1]] 

                    ind_insitu = np.nonzero(code_part[ind_this[subind_thisrad]] == 0)[0]
                    ind_acc = np.nonzero(code_part[ind_this[subind_thisrad]] == 1)[0]
                    ind_qm = np.nonzero(code_part[ind_this[subind_thisrad]] == 2)[0]
                    ind_stripped = np.nonzero(code_part[ind_this[subind_thisrad]] == 3)[0]
                    ind_otherfof = np.nonzero(code_part[ind_this[subind_thisrad]] == 4)[0]
                    ind_dead = np.nonzero(code_part[ind_this[subind_thisrad]] == 5)[0]

                    sh_m_insitu[ish, irad] = np.sum(part_mass[ind_this[subind_thisrad[ind_insitu]]])
                    sh_m_acc[ish, irad] = np.sum(part_mass[ind_this[subind_thisrad[ind_acc]]])
                    sh_m_qm[ish, irad] = np.sum(part_mass[ind_this[subind_thisrad[ind_qm]]])
                    sh_m_stripped[ish, irad] = np.sum(part_mass[ind_this[subind_thisrad[ind_stripped]]])
                    sh_m_otherfof[ish, irad] = np.sum(part_mass[ind_this[subind_thisrad[ind_otherfof]]])
                    sh_m_dead[ish, irad] = np.sum(part_mass[ind_this[subind_thisrad[ind_dead]]])

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

    
        
print("Done!")
