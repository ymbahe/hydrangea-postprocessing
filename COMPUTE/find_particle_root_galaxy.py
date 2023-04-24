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
                        of eagle_subfind_particles tables. Also large-scale
                        re-structuring.

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

include_parents = False 

# This currently does nothing. In principle, it would be useful to make this
# work as it may provide a better 'progenitor mass'.
rootType = 'first'  # 'first' (first SH association) or 'last' (before final) 

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


def find_root_snap(iiptype):
    """Find the root snapshot for particles of a given type"""
    
    if iiptype not in [4, 5]: return

    if iiptype == 4:
        formAexp = st.eagleread(snapdir_z0, 'PartType4/StellarFormationTime', astro = False, silent = True)
    else:
        formAexp = st.eagleread(snapdir_z0, 'PartType5/BH_FormationTime', astro = False, silent = True)
        
    # Find first snapshot AFTER birth. In the event that there is one
    # exactly at the birth time, pick that one (hence side = 'left')

    print("Find birth snapshot...", flush = True)
    
    ptype_birthSnap = np.searchsorted(snap_aexp, formAexp, side = 'left').astype(np.int8)
    return ptype_birthSnap.astype(np.int8)


def find_galaxy_for_ids(ids, isnap):
    """Identify the galaxy hosting input particle IDs in a given snapshot."""

    subdir = st.form_files(rundir, isnap)
    sf_ids = st.eagleread(subdir, 'IDs/ParticleID', astro = False)
    sf_offset = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False)
    sf_length = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)

    # The next three lines are robust to the presence of particles 
    # outside the subfind ID list or subhaloes
    sf_inds = yb.find_id_indices(ids, sf_ids)
    sf_shi = st.ind_to_sh(sf_inds, sf_offset, sf_length)
    sf_gal = sw.sh_to_gal(sf_shi, isnap, dealWithOOR = True)

    ind_in_gal = np.nonzero(sf_gal >= 0)[0]

    return sf_gal, ind_in_gal


def process_snapshot(isnap):
    """Process one snapshot for root-galaxy memberships..."""

    print("")
    print("---------- Starting snapshot = {:d} (sim = {:d}) ----------- " 
          .format(isnap, isim))
    print("")
    sys.stdout.flush()

    # Need to consider all particles that have current snap
    # as their root snap (increased if not found)
    
    if rootType == 'first':
        ind_root_this = np.nonzero(all_rootSnap == isnap)[0]
    else:
        #Still needs some thinking.
        pass
        
    if len(ind_root_this) == 0 and len(ind_gas_root_this) == 0:
        return

    print("Processing {:d} particles in snapshot {:d}..." 
          .format(len(ind_root_this), isnap))

    # Set up a mask to record which particles could be matched
    mask_root_this = np.zeros(len(ind_root_this), dtype = np.int8)

    # Identify the galaxy (if any) of all currently-considered particles
    all_gal, ind_in_gal = find_galaxy_for_ids(all_ids[ind_root_this], isnap)

    print("... of which {:d} are in a galaxy..." .format(len(ind_in_gal)))
    if len(ind_in_gal) == 0: return
         
    # Extra step added 25-Sep-2018:
    # need to check if this points to a spectre galaxy and if yes,
    # update to its parent
    # Translate the result back to SHI so that the rest of the code
    # works as before, even though the ultimate result is galaxy ID...
    
    # Changed 08-Mar-19: Only modify if spectre is also a satellite,
    # and its parent is still alive. Otherwise consider it 'proper'
    # (=don't touch)

    subind_spectre = np.nonzero((spectreFlag[all_gal[ind_in_gal]] == 1) & (satFlag[all_gal[ind_in_gal], isnap] == 1) & (shi[spectreParents[all_gal[ind_in_gal]], isnap] >= 0))[0]
    
    if len(subind_spectre) > 0:
        all_gal[ind_in_gal[subind_spectre]] = spectreParents[all_gal[ind_in_gal[subind_spectre]]]
        print("   Updated {:d} spectre identifications..." .format(len(subind_spectre)))

    if min_ratio is None:
        # Every galaxy will do:
        subind_massive = np.arange(len(ind_in_gal))
    else:
        # In this case, we need to check whether the identified host galaxies
        # are actually massive enough to qualify.
        
        if comp_type == 'current':
            # Need to first update comparison mass (of z = 0 galaxy)

            # Determine main progenitor of final SH at this snap
            all_z0Gal_currSHI = np.zeros(numPart, dtype = np.int32)-1
            all_z0Gal_currSHI[ind_in_gal_z0] = sw.gal_to_sh(all_gal_z0[ind_in_gal_z0], isnap)
            
            # Check which are alive in current snap and update their
            # maximum progenitor mass, if greater than previous value.
            ind_prog_alive = np.nonzero(all_z0Gal_currSHI >= 0)[0]
        
            refmass_alive = msub_all[all_z0Gal_currSHI[ind_prog_alive], isnap]
            ind_newmax = np.nonzero(refmass_alive > all_compMass[ind_prog_alive])[0]
            all_compMass[ind_prog_alive[ind_newmax]] = refmass_alive[ind_newmax] 
            isnap_ref = isnap
        else:
            isnap_ref = snap_z0

        # Now test which galaxies are massive enough:
        subind_massive = np.nonzero(msub_all[all_gal[ind_in_gal], isnap_ref] >= min_ratio * all_compMass[ind_root_this[ind_in_gal]])[0]

    mask_root_this[ind_in_gal[subind_massive]] = 1
    ind_unmatched = np.nonzero(mask_root_this == 0)[0]

    print("Could match {:d} particles to galaxies ({:d} unmatched)."
          .format(len(subind_massive), len(ind_unmatched)), flush = True)

    all_rootGal[ind_root_this[ind_in_gal[subind_massive]]] = all_gal[ind_in_gal[subind_massive]]
    all_rootSnap[ind_root_this[ind_unmatched]] += 1
    
    return    # ----- ends processing of current snapshot ----


def write_output(iiptype):
    """Write output for particle type iiptype"""

    pre = "PartType{:d}" .format(iiptype)

    pt_inds = np.arange(ptype_offsets[iiptype], ptype_offsets[iiptype+1])
    
    yb.write_hdf5(all_ids[pt_inds], outloc, pre+"/ParticleIDs", comment = "Particle IDs")
    yb.write_hdf5(all_rootSnap[pt_inds], outloc, pre+"/RootSnapshot", comment = "First snapshot in which a particle has been in a subhalo")
    yb.write_hdf5(ptype_rootGal[pt_inds], outloc, pre+"/RootGalaxy", comment = "Galaxy of particle in its root snapshot")

    if iiptype in [4, 5] and include_parents:
        pt_inds_parents = np.arange(ptype_offsets[iiptype+2], ptype_offsets[iiptype+3])
        yb.write_hdf5(all_rootSnap[pt_inds_parents], outloc, pre+"/ParentRootSnapshot", comment = "First snapshot in which the particle *or its gas parent particle* was in a subhalo.")
        yb.write_hdf5(all_rootGal[pt_inds_parents], outloc, pre+"/ParentRootGalaxy", comment = "Galaxy of particle or its gas parent parent in its root snapshot.")

    return


# =======================================================================
# Actual program starts here
# =======================================================================
        
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
    snapdir_z0 = st.form_files(rundir, snap_z0, 'snap')
    snapdir_0 = st.form_files(rundir, 0, 'snap')

    catID = calendar.timegm(time.gmtime())
    yb.write_hdf5_attribute(outloc, "Header", "CatalogueID", catID, new = True)

    sw = st.Spiderweb(hldir, highlev = True)

    # Load required information from galaxy evolution tables:
    spectreFlag = yb.read_hdf5(fgtloc, 'Full/SpectreFlag')
    spectreParents = yb.read_hdf5(fgtloc, 'Full/SpectreParents')
    shi = yb.read_hdf5(fgtloc, 'SHI')
    satFlag = yb.read_hdf5(fgtloc, 'SatFlag')
    
    if min_ratio is not None:
        msub_all = yb.read_hdf5(fgtloc, 'Msub')
        gal_sh_z0 = yb.read_hdf5(spiderloc, 'Subhalo/Snapshot_' + str(snap_z0).zfill(3) + '/Galaxy')

    print("Done with setup, now looping through particle types...")
    sys.stdout.flush()

    # Form one unified list of particles (all types), with offsets to 
    # indicate blocks of particle types

    all_ids = np.zeros(0, dtype = int)
    ptype_offsets = np.zeros(9, dtype = int)

    for iiptype in ptypeList:
    
        # Load all IDs that have ever existed 
        # (note: for BHs, only covers surviving ones to z = 0)
        if iiptype == 4 or iiptype == 5:
            ptype_ids = st.eagleread(snapdir_z0, 'PartType{:d}/ParticleIDs' .format(iiptype), astro = False, silent = True)
        else:
            ptype_ids = st.eagleread(snapdir_0, 'PartType{:d}/ParticleIDs' .format(iiptype), astro = False, silent = True)

        all_ids = np.concatenate((all_ids, ptype_ids))
        ptype_offset[iiptype+1:] += numPart

    # If parent-finding is included, we simply duplicate the required IDs.
    # That way, we don't have to include parents as separate category in the 
    # main parg of the program.
    if include_parents:
        for iiParType in [4, 5]:
            if iiParType in ptypeList: 
                all_ids = np.concatenate((all_ids, all_ids[ptype_offset[iiParType]:ptype_offset[iiParType]]))
            ptype_offset[iiParType+3:] += (ptype_offset[iiParType+1]-ptype_offset[iiParType])

    maxid = np.max(all_ids)
    numPart = len(all_ids)

    # Extra section necessary if mass threshold is switched on:
    if min_ratio is not None or rootType == 'last':

        # Identify the z = 0 host galaxy of all particles
        all_gal_z0, ind_in_gal_z0 = find_galaxy_for_ids(all_ids, snap_z0)

    if min_ratio is not None:
        # Set up the comparison mass array
        all_compMass = np.zeros(numPart, dtype = np.float32)

        # If we compare at z = 0, can already fill comparison array now:
        if comp_type == 'z0':
            all_compMass[ind_in_gal_z0] = msub_all[all_gal_z0[ind_in_gal_z0], snap_z0]
        # if comp_type == 'current', masses will be loaded in each snapshot.


    # Back in 'general' mode. Find the root snapshot for each particle
    # (first to search)

    all_rootSnap = np.zeros(numPart, dtype = np.int8)
    all_rootGal = np.zeros(numPart, dtype = np.int32)-1

    for iiptype in ptypeList:
        if iiptype in [4, 5]:
            all_rootSnap[ptype_offset[iiptype]:ptype_offset[iiptype+1]] = find_root_snap(iiptype)

    for isnap in range(nsnap):
        process_snapshot(isnap)

    # Write ouptut, sequentially for different types:
    for iiptype in ptypeList:
        write_output(iiptype)


print("Done!")
