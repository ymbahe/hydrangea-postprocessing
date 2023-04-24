"""
Extract "birth galaxy" of all stellar (and optionally other) particles.

Starting with the first snapshot in which the particle exists (in its
type, for stars/BHs), the script tests whether it belongs to a galaxy, which is
then its 'root'. Particles that are not associated to a galaxy are re-tested
in the subsequent snapshot, until z = 0 if necessary.

Optionally, a galaxy hosting the particle can be discarded if its mass is
below an adjustable threshold in units of the particle's z = 0 galaxy mass,
either at z = 0 or in the snapshot under consideration.

For stars/BHs, also the root galaxy of the parent gas particle is determined,
in an analogous way but starting from snapshot 0.

Output:

PartType[x]/ParticleIDs  --> Particle IDs, for matching to other data sets
PartType[x]/RootSnapshot --> First snapshot in which particle is in subhalo
PartType[x]/RootGalaxy   --> Galaxy of particle in root snapshot

For stars and BHs only:
PartType[x]/ParentRootSnapshot --> As RootSnapshot, but for gas parent particle
PartType[x]/ParentRootGalaxy   --> As RootGalaxy, but for gas parent particle

 -- Started 2-Feb-2018
 -- Updated 9-May-2019: improving documentation and removing any involvement
                        of eagle_subfind_particles tables... 

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
import sys

simname = "Eagle"     # 'Eagle' or 'Hydrangea'
runtype = "HYDRO"     # 'HYDRO' or 'DM'
ptypeList = [4]      # List of particle types to process

# Next line specifies the minimum mass of a subhalo to be considered as 
# 'containing' a particle, in units of the particle's z = 0 host.
# If it is None, no limit is applied.
min_ratio = None

# Specify the point at which subhalo masses are compared (only relevant if 
# min_ratio != None). 'current' (at snapshot itself) or 'z0' (at z = 0).
comp_type = 'current'

# This currently does nothing. In principle, it would be useful to make this
# work as it may provide a better 'progenitor mass'.
#rootType = 'last'  # 'first' (first SH association) or 'last' (before final) 

if simname == "Eagle":
    n_halo = 1
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    snap_z0 = 28
    nsnap = 29
    snapAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/eagle_outputs_new.txt'
else:    
    n_halo = 30
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    snap_z0 = 29
    nsnap = 30
    snapAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/hydrangea_snapshots_plus.dat'

outname = 'highlev/ParticleRootInfo_MinRatioNone_withRootSnap.hdf5'

# =============================================================================

def process_ptype(ptype):
    """Process one particle type for one simulation"""

    


# Set up MPI to enable processing different sims in parallel
comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

snap_aexp = np.array(ascii.read(snapAexpLoc, format = 'no_header', guess = False)['col1'])

for isim in range(n_halo):

    # Skip this one if we are multi-threading and it's not for this task to 
    # worry about
    if not isim % numtasks == rank:
        continue
        
    sim_stime = time.time()

    print("")
    print("**************************")
    print("Now processing halo CE-{:d}" .format(isim))
    print("**************************")
    print("")

    sys.stdout.flush()

    # Set up standard path and file names:
    if simname == "Eagle":
        rundir = basedir
        outloc = er.clone_dir(rundir) + '/' + outname
        hldir = er.clone_dir(rundir) + '/highlev/'
        fgtloc = hldir + '/FullGalaxyTables.hdf5'
    else:
        rundir = basedir + '/CE-' + str(isim) + '/' + runtype
        outloc = ht.clone_dir(rundir) + '/' + outname
        hldir = rundir + '/highlev'
        fgtloc = hldir + '/FullGalaxyTables.hdf5' 

    if not os.path.exists(rundir):
        continue

    if not os.path.exists(yb.dir(outloc)):
        os.makedirs(yb.dir(outloc))

    # Set up files to load particles from.
    # !!!! NEEDS UPDATING TO REMOVE SUBPART INVOLVEMENT !!!!
    if min_ratio is None:
        snapdir_z0, subdir_z0 = st.form_files(rundir, snap_z0, 'snap sub')
        snapdir_0, subdir_0 = st.form_files(rundir, 0, 'snap sub')
    else:
        snapdir_z0, subdir_z0 = st.form_files(rundir, snap_z0, 'subpart sub')
        snapdir_0, subdir_0 = st.form_files(rundir, snap_z0, 'subpart sub')
        
    catID = calendar.timegm(time.gmtime())
    yb.write_hdf5_attribute(outloc, "Header", "CatalogueID", catID, new = True)

    sw = st.Spiderweb(hldir, highlev = True)

    # Load required information from galaxy evolution tables:
    spectreFlag = yb.read_hdf5(fgtloc, 'Full/SpectreFlag')
    spectreParents = yb.read_hdf5(fgtloc, 'Full/SpectreParents')
    shi = yb.read_hdf5(fgtloc, 'SHI')
    satFlag = yb.read_hdf5(fgtloc, 'SatFlag')

    print("Done with setup, now looping through particle types...")
    sys.stdout.flush()

    for iiptype in ptypeList:
        process_ptype(iiptype)

        print("")
        print("---------- Starting ptype = {:d} ----------- " 
              .format(iiptype))
    
        sys.stdout.flush()


        # Load all IDs that have ever existed 
        # (note: for BHs, only covers surviving ones to z = 0)
        if iiptype == 4 or iiptype == 5:
            ptype_ids = st.eagleread(snapdir_z0, 'PartType{:d}/ParticleIDs' .format(iiptype), astro = False, silent = True)
        
        else:
            ptype_ids = st.eagleread(snapdir_0, 'PartType{:d}/ParticleIDs' .format(iiptype), astro = False, silent = True)

        maxid = np.max(ptype_ids)
        numPart = len(ptype_ids)
    
        if min_ratio is not None:

            print("Loading data for min_ratio determination...", flush = True)

            # Determine subhalo mass of particles at z = 0
            sh_mass_z0 = st.eagleread(subdir_z0, 'Subhalo/Mass', astro = True, silent = True)[0]
            fof_fsh = st.eagleread(subdir_z0, 'FOF/FirstSubhaloID', astro = False, silent = True)
            
            ptype_fof = np.abs(st.eagleread(snapdir_z0, 'PartType{:d}/GroupNumber' .format(iiptype), astro = False, silent = True))-1
            ptype_sgn = st.eagleread(snapdir_z0, 'PartType{:d}/SubGroupNumber' .format(iiptype), astro = False, silent = True)
            ptype_shi = np.zeros(len(ptype_ids), dtype = np.int32)-1
            ptype_gal = np.zeros(len(ptype_ids), dtype = np.int32)-1

            ind_in_sh_z0 = np.nonzero(ptype_sgn < 2**30)[0]
            ptype_shi[ind_in_sh_z0] = fof_fsh[ptype_fof[ind_in_sh_z0]]+ptype_sgn[ind_in_sh_z0]
            ptype_gal[ind_in_sh_z0] = sw.sh_to_gal(ptype_shi[ind_in_sh_z0], snap_z0)
            ptype_mz0 = np.zeros(len(ptype_ids), dtype = np.float32)
            if comp_type == 'z0':
                ptype_mz0[ind_in_sh_z0] = sh_mass_z0[ptype_shi[ind_in_sh_z0]]

        if iiptype == 4:
            formAexp = st.eagleread(snapdir_z0, 'PartType4/StellarFormationTime', astro = False, silent = True)
        elif iiptype == 5:
            formAexp = st.eagleread(snapdir_z0, 'PartType5/BH_FormationTime', astro = False, silent = True)
        else:
            formAexp = 0
        

        # Find first snapshot AFTER birth. In the event that there is one
        # exactly at the birth time, pick that one (hence side = 'left')

        print("Find birth snapshot...", flush = True)

        if iiptype == 4 or iiptype == 5:
            ptype_birthSnap = np.searchsorted(snap_aexp, formAexp, side = 'left').astype(np.int8)
            ptype_rootSnap = np.zeros(numPart, dtype = np.int8)
            ptype_rootSnap[:] = ptype_birthSnap[:]
            ptype_parentRootSnap = np.zeros(numPart, dtype = np.int8)
    
        else:
            ptype_rootSnap = np.zeros(numPart, dtype = np.int8)

        # Now the fun part - have to work out the SHI of each particle at its 
        # root snap

        ptype_rootSHI = np.zeros(numPart, dtype = np.int32)-1
        ptype_rootGal = np.zeros(numPart, dtype = np.int32)-1

        if iiptype == 4 or iiptype == 5:
            ptype_parentRootSHI = np.zeros(numPart, dtype = np.int32)-1
            ptype_parentRootGal = np.zeros(numPart, dtype = np.int32)-1        

        for isnap in range(nsnap):
    
            print("")
            print("------------------------------")
            print("     Sim {:d}, snapshot {:d}     " .format(isim, isnap))
            print("------------------------------", flush = True)
        
            
            ind_root_this = np.nonzero(ptype_rootSnap == isnap)[0]

            if iiptype == 4 or iiptype == 5:
                ind_gas_root_this = np.nonzero(ptype_parentRootSnap == isnap)[0]

            if len(ind_root_this) == 0 and len(ind_gas_root_this) == 0:
                continue

            print("")
            print("Processing {:d} particles of ptype {:d}..." 
                  .format(len(ind_root_this), iiptype))
    
            if iiptype == 4 or iiptype == 5:
                print("and {:d} particles including gas..."
                      .format(len(ind_gas_root_this)))

                print("   ---> including {:d} re-trials"
                      .format(len(ind_root_this)-np.count_nonzero(ptype_birthSnap == isnap)))


            print("")
            print("Reading subfind output...", end = "", flush = True)
            partdir, subdir = st.form_files(rundir, isnap, 'subpart sub')

            fof_fsh = st.eagleread(subdir, 'FOF/FirstSubhaloID', astro = False, silent = True)

            part_ptype_id = st.eagleread(partdir, 'PartType{:d}/ParticleIDs' .format(iiptype), astro = False, silent = True)
            part_ptype_gn = np.abs(st.eagleread(partdir, 'PartType{:d}/GroupNumber' .format(iiptype), astro = False, silent = True))-1

            part_ptype_sgn = st.eagleread(partdir, 'PartType{:d}/SubGroupNumber' .format(iiptype), astro = False, silent = True)

            part_ptype_shi = np.zeros(len(part_ptype_sgn),dtype=np.int32)-1
            ind_in_sh = np.nonzero(part_ptype_sgn < 2**30)[0]
            part_ptype_shi[ind_in_sh] = fof_fsh[part_ptype_gn[ind_in_sh]]+part_ptype_sgn[ind_in_sh]

            
            if min_ratio is not None:

                print("Loading data for min_ratio determination...", flush = True)
                sh_mass = st.eagleread(subdir, 'Subhalo/Mass', astro = True, silent = True)[0]

                

                # Need to update comparison mass (of descendant)
                if comp_type == 'current':

                    # Determine main progenitor of final SH at this snap
                    ptype_shi_curr = np.zeros(len(ptype_ids), dtype = np.int32)-1
                    ptype_shi_curr[ind_in_sh_z0] = sw.gal_to_sh(ptype_gal[ind_in_sh_z0], isnap)
                    ind_prog_alive = np.nonzero(ptype_shi_curr >= 0)[0]
                    refmass_alive = sh_mass[ptype_shi_curr[ind_prog_alive]]
                    ind_newmax = np.nonzero(refmass_alive > ptype_mz0[ind_prog_alive])[0]
                    
                    ptype_mz0[ind_prog_alive[ind_newmax]] = refmass_alive[ind_newmax] 


            if iiptype == 4 or iiptype == 5:
                part_gas_id = st.eagleread(partdir, 'PartType0/ParticleIDs', astro = False, silent = True)
                part_gas_gn = np.abs(st.eagleread(partdir, 'PartType0/GroupNumber', astro = False, silent = True))-1
                part_gas_sgn = st.eagleread(partdir, 'PartType0/SubGroupNumber' .format(iiptype), astro = False, silent = True)

                # Need to concatenate gas and star/BH lists to capture
                # both particles that are already in their final state,
                # and which are still gas

                part_gas_id = np.concatenate((part_gas_id, part_ptype_id))
                part_gas_gn = np.concatenate((part_gas_gn, part_ptype_gn))
                part_gas_sgn = np.concatenate((part_gas_sgn, part_ptype_sgn))
                
                ind_gas_in_sh = np.nonzero(part_gas_sgn < 2**30)[0]
                part_gas_shi = np.zeros(len(part_gas_sgn),dtype=np.int32)-1
                part_gas_shi[ind_gas_in_sh] = fof_fsh[part_gas_gn[ind_gas_in_sh]]+part_gas_sgn[ind_gas_in_sh]

            # Need to match IDs...
            
            ind_match, ind_matched = yb.find_id_indices(ptype_ids[ind_root_this], part_ptype_id)
            shi_match = part_ptype_shi[ind_match[ind_matched]]
            
            # Extra step added 25-Sep-2018:
            # need to check if this points to a spectre galaxy and if yes,
            # update to its parent
            # Translate the result back to SHI so that the rest of the code
            # works as before, even though the ultimate result is galaxy ID...

            # Changed 08-Mar-19: Only modify if spectre is also a satellite,
            # and its parent is still alive. Otherwise consider it 'proper'
            # (=don't touch)

            gal_match = sw.sh_to_gal(shi_match, isnap, dealWithOOR = True)
            ind_goodGal = np.nonzero(gal_match >= 0)[0]
            subind_spectre = np.nonzero((spectreFlag[gal_match[ind_goodGal]] == 1) & (satFlag[gal_match[ind_goodGal], isnap] == 1) & (shi[spectreParents[gal_match[ind_goodGal]], isnap] >= 0))[0]
            
            if len(subind_spectre) > 0:
                gal_match[ind_goodGal[subind_spectre]] = spectreParents[gal_match[ind_goodGal[subind_spectre]]]
                shi_match[ind_goodGal[subind_spectre]] = sw.gal_to_sh(gal_match[ind_goodGal[subind_spectre]], isnap) 
            print("   Updated {:d} spectre identifications..." .format(len(subind_spectre)))

            """
            if simname == "Hydrangea":
                partid_rev = st.create_reverse_list(part_ptype_id, maxval = maxid+1)
                ind_matched = np.nonzero(partid_rev[ptype_ids[ind_root_this]] >= 0)[0]
                shi_match = part_ptype_shi[partid_rev[ptype_ids[ind_root_this[ind_matched]]]]
            else:

                # Need to identify matching IDs in sh_ids by brute force
                sorter_in = np.argsort(ptype_ids[ind_root_this])
                sorter_part = np.argsort(part_ptype_id)

                shi_match_all = np.zeros(len(ind_root_this), dtype = np.int32)-1
                ind_in_sorted_part = katamaran_search(ptype_ids[ind_root_this[sorter_in]], part_ptype_id[sorter_part])
                ind_prematched_sorted = np.nonzero(ind_in_sorted_part >= 0)[0]
                ind_matched = sorter_in[ind_prematched_sorted]
                ind_matched_ptype = sorter_part[ind_in_sorted_part[ind_prematched_sorted]]

                shi_match = part_ptype_shi[ind_matched_ptype] 
            """

            if min_ratio is None:
                ptype_rootSHI[ind_root_this[ind_matched]] = shi_match
            else:
                subind_massive = np.nonzero(sh_mass[shi_match] >= min_ratio * ptype_mz0[ind_root_this[ind_matched]])[0]
                ptype_rootSHI[ind_root_this[ind_matched[subind_massive]]] = shi_match[subind_massive]

            ind_root_in_sh = np.nonzero(ptype_rootSHI[ind_root_this] >= 0)[0]

            if iiptype == 4 or iiptype == 5:
                ind_match_gas, ind_gas_matched = yb.find_id_indices(ptype_ids[ind_gas_root_this], part_gas_id)
                shi_match_gas = part_gas_shi[ind_match_gas[ind_gas_matched]]


                """
                gasid_rev = st.create_reverse_list(part_gas_id, maxval = maxid+1)
                ind_gas_matched = np.nonzero(gasid_rev[ptype_ids[ind_gas_root_this]] >= 0)[0]

                shi_match_gas = part_gas_shi[gasid_rev[ptype_ids[ind_gas_root_this[ind_gas_matched]]]]
                """

                if min_ratio is None:
                    ptype_parentRootSHI[ind_gas_root_this[ind_gas_matched]] = shi_match_gas
                else:
                    subind_massive_gas = np.nonzero(sh_mass[shi_match_gas] >= min_ratio * ptype_mz0[ind_gas_root_this[ind_gas_matched]])[0]
                    ptype_parentRootSHI[ind_gas_root_this[ind_gas_matched[subind_massive_gas]]] = shi_match_gas[subind_massive_gas]


                ind_gas_root_in_sh = np.nonzero(ptype_parentRootSHI[ind_gas_root_this] >= 0)[0]


            print("")
            print("---> could match {:d} new particles to subhaloes..."
                  .format(len(ind_root_in_sh)))
            if iiptype == 4 or iiptype == 5:
                print("---> and could match {:d} new parent gas particles to subhaloes..."
                      .format(len(ind_gas_root_in_sh)))
    
            
            sys.stdout.flush()

            if len(ind_root_in_sh) > 0:
                ptype_rootGal[ind_root_this[ind_root_in_sh]] = sw.sh_to_gal(ptype_rootSHI[ind_root_this[ind_root_in_sh]], isnap)
    
            if iiptype == 4 or iiptype == 5:
                if len(ind_gas_root_in_sh) > 0:
                    ptype_parentRootGal[ind_gas_root_this[ind_gas_root_in_sh]] = sw.sh_to_gal(ptype_parentRootSHI[ind_gas_root_this[ind_gas_root_in_sh]], isnap)

            # Those that were NOT in a SH at birth, give them another chance
            # in next snapshot...

            ind_out_sh = np.nonzero(ptype_rootSHI[ind_root_this] < 0)[0]
            ptype_rootSnap[ind_root_this[ind_out_sh]] += 1

            if iiptype == 4 or iiptype == 5:
                ind_gas_out_sh = np.nonzero(ptype_parentRootSHI[ind_gas_root_this] < 0)[0]
                ptype_parentRootSnap[ind_gas_root_this[ind_gas_out_sh]] += 1
    

            print("---> could not match {:d} particles - deferring to next snapshot"
                  .format(len(ind_out_sh)))
        
            if iiptype == 4 or iiptype == 5:
                print("---> and could not match {:d} parent gas particles - deferring to next snapshot"
                      .format(len(ind_gas_out_sh)))
        
            sys.stdout.flush()


        # ----- ends loop through snapshots -------


        pre = "PartType{:d}" .format(iiptype)

        #if iiptype == 4 or iiptype == 5:    
        #    yb.write_hdf5(ptype_birthSnap, outloc, pre+"/BirthSnapshot", comment = "First snapshot in which a particle has been formed")

        yb.write_hdf5(ptype_rootSnap, outloc, pre+"/RootSnapshot", comment = "First snapshot in which a particle has been in a subhalo")

        if iiptype == 4 or iiptype == 5:
            yb.write_hdf5(ptype_parentRootSnap, outloc, pre+"/ParentRootSnapshot", comment = "First snapshot in which the particle *or its gas parent particle* was in a subhalo.")
            #yb.write_hdf5(ptype_parentRootSHI, outloc, pre+"/ParentRootSHI", comment = "Subhalo of parent gas particle in its root snapshot")
            yb.write_hdf5(ptype_parentRootGal, outloc, pre+"/ParentRootGalaxy", comment = "Galaxy of parent gas particle in its root snapshot")

        #yb.write_hdf5(ptype_rootSHI, outloc, pre+"/RootSHI", comment = "Subhalo of particle in its root snapshot")

        yb.write_hdf5(ptype_rootGal, outloc, pre+"/RootGalaxy", comment = "Galaxy of particle in its root snapshot")

        yb.write_hdf5(ptype_ids, outloc, pre+"/ParticleIDs", comment = "Particle IDs")


    # ---- ends loop through particle types



print("Done!")
