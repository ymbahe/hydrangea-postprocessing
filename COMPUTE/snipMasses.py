"""
Highly experimental program to determine galaxy masses in snipshots.

-- Started 20-May-2019
"""

import numpy as np
import yb_utils as yb
import sim_tools as st
import time
from astropy.cosmology import Planck13
import os
import hydrangea_tools as ht
from astropy.io import ascii
import monk

from cantor_dev import SplitList, eprint, TimeStamp
from pdb import set_trace

rootdir = '/virgo/simulations/Hydrangea/10r200/'
simType = 'HYDRO'
simRange = [0, 1]  # First and beyond-last sim to process
verbose = 0
snepList = 'basic'

# -------------- Options to modify the behaviour of the code ------------------

# Include 'out-of-sync' snaps for before/after snipshots?
par_incl_extra_snaps = True                  # Default: True

# Include boundary particles in analysis? Only useful for debugging,
# since these are not included in any SF particle lists.
par_include_boundary_particles = False       # Default: False

# Should input be read from CANTOR or Subfind?
par_load_from_cantor = True                  # Default: True

# How verbose should MONK be?
par_monk_verb = 0

# If only `massive' galaxies should be processed, set here threshold in
# total and stellar mass: [m_sub, m_star]. Otherwise: None
par_galaxy_thresholds = [10.0, 8.0]

par_write_binding_energy = 0

# Report CentreOfPotential as min. at unbinding (True), or at finish (False)?
# (Note: False only works with par_write_binding_energy, otherwise changed)
par_cop_at_unbinding = True             # Default: False

# Cadence of full reporting for galaxy unbinding (every Nth):
par_report_frequency = 1000

# Use monotonic unbinding mode (1, instead of 'vdBO-like', 0) for sats?
par_monk_monotonic = 1                   # Default: 1

# Center on ZMF for satellites (0), or 10% most-bound particles (1)?
par_monk_centering = 1                   # Default: 0

# Allow varying centre for satellites (0) or keep fixed (1)?
par_monk_fixCentre = 0                   # Default: 0

# Bypass Monk completely (testing only!! --> keeps all particles as bound)
par_bypass_monk = False

# Tree opening criterion within MONK -- larger means less accurate.
par_monk_potErrTol = 1.0                 # Default: 1.0

# Set minimum number of particles (at unbinding) for a subhalo:
par_min_subhalo_members = 10


def setup():
    """Set derived parameters and do consistency checks."""

    # Globalize internally-set global parameters:
    global par_typeList         # Array of particle types to consider.
    
    # Set derived parameters:
    if par_include_boundary_particles:
        par_typeList = np.arange(6, dtype = np.int8)
    else:
        par_typeList = np.array([0, 1, 4, 5])

    if par_bypass_monk:
        eprint("Warning: Monk bypassed!", linestyle = '+')
        if par_write_binding_energy:
            eprint("Disabled BE writing...", linestyle = '+')
            par_write_binding_energy = 0
    

def process_simulation(isim):
    """
    Process all snepshots in one simulation.

    Fairly thin wrapper around process_snepshot().
    """
    

    # Set up a 'Simulation' class instance that holds its basic properties:
    sim = Simulation(isim, rootdir = rootdir, simType = simType)
    if not sim.exists:
        print("Simulation {:d} does not exist, skipping..." .format(isim))
        return

    # Find snepshots of this simulation:
    sim.load_snepshot_list(snepList)
    print("There are {:d} snepshots." .format(sim.nSnep))

    sim.timeStamp.set_time('Load snepshot list')

    # Loop (forwards) over target snepshots. If it is a snapshot,
    # nothing is done. If it is a snipshot, particles are assigned to 
    # galaxies, their masses determined, and all derived data written out.
    for isnep_proc in range(0, sim.nSnep):
        process_sim_snepshot(isnep_proc, sim)
        sim.timeStamp.set_time('Snepshot {:d}' .format(isnep_proc))
    
    sim.timeStamp.print_time_usage('Finished simulation {:d}'
                                       .format(isim), print_sub = False)
    sim.timeStamp.print_time_usage('Finished simulation {:d}' .format(isim))

def process_sim_snepshot(isnep, sim):
    """
    Core routine, to find masses in one snepshot.
    
    Parameters:
    -----------
    isnep : int
        The snepshot index to process.
    sim : Simulation instance
        The simulation to which this snepshot belongs.
    """
 
    stime = time.time()
    snep = Snepshot(sim, isnep)
    if snep.type == 'snap':
        print("Snepshot {:d} is a snapshot, don't need to process."
              .format(isnep))
        return

    # If we get here, we hit a snipshot, which we *do* need to process...
    # First, find previous and next snapshots:
    snep.find_boundary_snaps()
    snep.load_coordinates()
    snep.timeStamp.set_time('Snepshot setup')

    # Load particle-level data for this snipshot:
    targetParticles = TargetParticles(snep)
    targetParticles.load_particle_data()
    targetParticles.create_origin_arrays()
    targetParticles.create_id_lookup_list()
    snep.timeStamp.set_time('Load particle data')

    # Find target galaxies (alive either in previous or next snap):
    targetGalaxies = Galaxies(snep)
    targetGalaxies.find_galaxies()
    snep.timeStamp.set_time('Find target galaxies')

    galTS = TimeStamp()
    gts_inds = galTS.add_counters(("Load particles", "Lookup particles",
                                   "Unbinding"))
    # Loop through individual galaxies and process them:
    for iigal, igal in enumerate(targetGalaxies.gal):
        galTS.start_time()

        

        gal = Galaxy(igal, targetParticles, galVerbose)
        gal.load_particles()
        galTS.increase_time(index=gts_inds[0])
        gal.lookup_particles()
        gal.find_coordinates()
        galTS.increase_time(index=gts_inds[1])

        if verbose or iigal % 1000 == 0:
            print("")
            print("Unbinding galaxy {:d}/{:d} (ID={:d})..."
                  .format(iigal, len(targetGalaxies.gal), igal))
            print("")
            
        # It is advisable to actually process all galaxies...:
        targetParticles.unbind_galaxy(gal)
        galTS.increase_time(index=gts_inds[2])

    snep.timeStamp.import_times(galTS)
    snep.timeStamp.set_time('Unbind galaxies')
    # Determine masses of all galaxies:
    output = Output(snep)
    output.compute_masses()
    output.write()
    snep.timeStamp.set_time('Compute and write output')

    # Print time usage statistics:
    snep.timeStamp.print_time_usage('Completed snepshot {:d} of sim {:d}' 
                                    .format(isnep, sim.isim))
    sim.timeStamp.import_times(snep.timeStamp)
    

class Output:
    """Class for computing, storing, and writing output."""

    def __init__(self, snep):
        """
        Class constructor.

        Parameters:
        -----------
        snep : Snepshot instance
            The snepshot for which output should be computed.
        """

        self.snep = snep
        self.galaxies = snep.target_galaxies
        self.particles = snep.particles
        self.sim = snep.sim

        # Set up a 'subhalo' index for the snepshot:
        self.shi = np.zeros(self.sim.numGal, dtype = np.int32) - 1

        # Set up 'result' array for masses by type:
        self.sh_MassType = np.zeros((self.galaxies.gal.shape[0], 6),
                                    dtype = np.float32)
        self.sh_MassType[...] = None  # Initialize
        
        # Set up output file name:
        self.outloc = (ht.clone_dir(self.sim.hldir) 
                       + '/SnipshotCatalogues/Snepshot_' 
                       + str(snep.isnep).zfill(4) + '.hdf5')
        if not os.path.isdir(yb.dir(self.outloc)):
            os.makedirs(yb.dir(self.outloc))

        if os.path.exists(self.outloc):
            os.rename(self.outloc, self.outloc + '.old')

    def compute_masses(self):                        # Class: Output
        """
        Compute the masses (by particle type) of galaxies after unbinding.
        """

        part = self.particles    # For convenience

        # Set up look-up table for particles by galaxyID:
        self.part_lut_galaxy = SplitList(part.galaxy, self.sim.gal_lims) 

        n_foundGal = 0   # Initialize

        # Go through all considered galaxies and find their masses:
        for igal in self.galaxies.gal:
            
            # Find galaxy's particles in full particle lists:
            partInd = self.part_lut_galaxy(igal)
            if len(partInd) == 0: continue
            
            n_foundGal += 1
            # Give the galaxy a 'subhalo' index:
            self.shi[igal] = n_foundGal-1

            # Now break the galaxy's particles down by type:
            partGal_lut_type = SplitList(part.type[partInd], np.arange(7))

            for iptype in par_typeList:
                ind_thisType = partGal_lut_type(iptype)

                # NB: sh_MassType is already set up large enough to 
                #     accommodate all possibly re-discovered galaxies.
                self.sh_MassType[self.shi[igal], iptype] = (
                    np.sum(part.mass[partInd[ind_thisType]]))

        print("In snepshot {:d}, there are {:d} assigned subhaloes, "
              "number tested: {:d}."
              .format(self.snep.isnep, n_foundGal, len(self.galaxies.gal)))

        self.numSH = n_foundGal

    def write(self):                                   # Class: Output
        """Write internally-stored data for this snapshot."""

        isnep = self.snep.isnep      # For convenience
        galaxies = self.galaxies
        particles = self.particles

        # Invert the SHI list to get a SHI --> galaxy table (ignoring lost):
        self.revGal = yb.create_reverse_list(self.shi, cut = True) 

        # -----------------------------------------
        # i) Subhalo-sorted particle + offset list
        # -----------------------------------------
 
        # Set up by-SH lookup table:
        part_lut_sh = SplitList(self.shi[particles.galaxy],
                                np.arange(self.numSH+1, dtype = np.int))
        
        # First, we need the 'boundaries' of the lookup table directly:
        splits_sh = part_lut_sh.splits
        length_sh = splits_sh[1:]-splits_sh[:-1]

        # Remove offset from having unbound particles:
        splits_sh -= splits_sh[0]  

        yb.write_hdf5(splits_sh, self.outloc, 'Subhalo/Offset', 
                      comment = "Offset of subhalo index i in ID list. "
                      "The particles belonging to this SH are stored in "
                      "indices [offset]:[offset]+[length].")
        yb.write_hdf5(length_sh, self.outloc, 'Subhalo/Length', 
                      comment = 'Number of particles in ID list that belong '
                      'to subhalo index i. The particles belonging to this '
                      'SH are stored in indices [offset]:[offset]+[length].')
        
        # Now write the (sorted) ID list (excluding unbound particles):
        sortedIDs = particles.galaxy[part_lut_sh.argsort]
        sortedIDs = sortedIDs[splits_sh[0] : splits_sh[-1]]
        yb.write_hdf5(sortedIDs, self.outloc, 'IDs', 
                      comment = "IDs of all particles associated with a "
                      "subhalo.")

        # Write masses:
        yb.write_hdf5(self.sh_MassType[:self.numSH, :], self.outloc, 
                      'Subhalo/MassType', 
                      comment = "Mass per particle type of each subhalo "
                      "(units: 10^10 M_sun).")

        # Write galaxy <--> subhalo lists:
        yb.write_hdf5(self.shi, self.outloc, 'SubHaloIndex',
                      comment = 'Subhalo index of galaxy ID [i].')
        yb.write_hdf5(self.revGal, self.outloc, 'Subhalo/Galaxy',
                      comment = 'Galaxy ID of subhalo [i].')

        # Write potential minimum of each subhalo:
        yb.write_hdf5(self.snep.potMin_ID[self.revGal], self.outloc, 
                      'Subhalo/PotentialMinimumID',
                      comment = 'ID of the particles at the potential '
                      'minimum of subhalo [i].')
        yb.write_hdf5(self.snep.potMin_pos[self.revGal, :], self.outloc,
                      'Subhalo/CentreOfPotential',
                      comment = 'Coordinates of the particle with the '
                      'lowest potential energy in subhalo [i], in pMpc.')

class Galaxy:
    """Class for holding properties about one specific galaxy."""
    
    def __init__(self, igal, targetParticles, verbose=False):
        """
        Class constructor. Load particles from ref_snap.

        Parameters:
        -----------
        igal : int
            ID of the galaxy.
        target_particles : TargetParticles instance
            Particles from which these galaxy is to be identified.
        verbose : bool, optional
            If True, galaxy will produce verbose output (default: False).
        """

        self.igal = igal
        self.verbose = verbose
        self.particles = targetParticles
        self.target_snep = targetParticles.snep
        self.sim = self.target_snep.sim
        
        # Set up (initially empty) particle ID list:
        self.gal_IDs = np.zeros(0, dtype = int)

        # Set up 'association list', how particles were found.
        self.origins = np.zeros(0, dtype = np.int8)

    def load_particles(self):                             # Class: Galaxy
        """Load all particles belonging to the galaxy."""
        self.load_galaxy_particles_snap(self.target_snep.snap_prev, 0)
        self.load_galaxy_particles_snap(self.target_snep.snap_next, 1)
        
        self.gal_IDs, source_index = np.unique(self.gal_IDs,
                                               return_index = True)
        self.origins = self.origins[source_index]

    def lookup_particles(self):                           # Class: Galaxy
        """Unicate and look up galaxy's particles."""
        
        # Unicate the particle list to remove duplicates:
        self.unicate_particles()

        # Translate IDs --> indices, reject any particles that are
        # not in the target snapshot's list (should be very uncommon):
        self.part_inds, subIndFound = self.particles.get_indices(self) 
            
        # Need to explicitly reduce origins array to account
        # for possibility of excluded particles:
        self.origins = self.origins[subIndFound]

        # Print out some statistics about identified particles:
        self.print_particle_statistics()
        
    def unicate_particles(self):                          # Class: Galaxy
        """Remove duplicates from the internally-loaded particle list."""
 
        # Store original number of particles for reference:
        self.full_particle_number = len(self.gal_IDs)

        # source_index contains the indices of the first occurrence of
        # each particle
        self.gal_IDs, source_index = np.unique(self.gal_IDs, 
                                               return_index = True)
        self.origins = self.origins[source_index] 

        # Do a consistency check to make sure we've got some left:
        if len(self.gal_IDs) <= 0:
            print("Galaxy {:d} has no particles to consider at all?!"
                  .format(self.igal))
            set_trace()
        if verbose:
            print("Galaxy {:d} -- {:d} unique particles."
                  .format(self.igal, len(self.gal_IDs)))

    def print_particle_statistics(self):                  # Class: Galaxy
        """Print information about target particles for one galaxy."""

        # First a sanity check that there are not zero particles:
        if len(self.part_inds) == 0:
            print("Galaxy {:d} has no target particles at all?!"
                  .format(self.igal))
            set_trace()
        
        # Count the number by origin code (N.B.: during unication, the 
        # highest priority (lowest number) origin route is chosen).
        numOriginTypes, bE = np.histogram(self.origins, bins = 2, 
                                          range = [0, 2])
        if verbose:
            print("Found {:d} particles in target list, (out of {:d}, "
                  "{:.2f} per cent)..." 
                  .format(len(self.part_inds), self.full_particle_number, 
                          len(self.part_inds)/self.full_particle_number*100))
            print("Origins: {:d} -- {:d}"
                  .format(*numOriginTypes))

        # Finally, increment target-snap-wide origin counts:
        self.target_snep.target_galaxies.add_origin_counts(numOriginTypes)

                                                          # Class: Galaxy
    def print_bound_particle_statistics(self, ind_bound): 
        """Print information about bound particles for one galaxy."""
        
        # Count the number of bound particles by origin:
        numOriginTypes, bE = np.histogram(self.origins[ind_bound], bins = 2, 
                                          range = [0, 2])
        if verbose:
            print("Bound origins: {:d} -- {:d}"
                  .format(*numOriginTypes))

        # Increment target-snap-wide origin counts:
        self.target_snep.target_galaxies.add_bound_origin_counts(
            numOriginTypes)


                                                          # Class: Galaxy
    def load_galaxy_particles_snap(self, snap, originCode):           
        """Load and add particles belonging to the galaxy in a given snap."""
        
        # Find subhalo index in reference snap:
        shi = snap.shi[self.igal]

        gal_IDs = (snap.ids[snap.off[shi] : snap.off[shi] + 
                            snap.len[shi]]).astype(np.int)
        
        self.gal_IDs = np.concatenate((self.gal_IDs, gal_IDs))
        self.origins = np.concatenate((self.origins, np.zeros(len(gal_IDs), dtype = np.int8) + originCode))

        if verbose:
            print("Loaded {:d} particles in snap {:d} for galaxy {:d}..."
                  .format(len(gal_IDs), snap.isnap, self.igal))

    def find_coordinates(self):                           # Class: Galaxy
        """Load the (approximate) coordinates of a galaxy."""
        self.pos = self.target_snep.galPos[self.igal, :]
        self.vel = self.target_snep.galVel[self.igal, :]
                

class Simulation:
    """Class for holding basic properties about a simulation."""

    def __init__(self, isim, rootdir='/virgo/simulations/Hydrangea/10r200/',
                 simType='HYDRO', simClass='Hydrangea'):
        """
        Constructor for the class.

        Parameters
        ----------
        isim : int
            Index of the simulation.
        rootdir : string, optional
            The base directory of the simulation family. Defaults to
            '/virgo/simulations/Hydrangea/10r200/'.
        simType : string, optional
            The type of simulation, 'HYDRO' (default) or 'DM'.
        """

        self.timeStamp = TimeStamp()
        
        self.isim = isim
        self.rundir = rootdir + 'CE-' + str(isim) + '/' + simType + '/'    
        
        if not os.path.isdir(self.rundir):
            self.exists = False
            return
        self.exists = True

        if simClass == 'Hydrangea':
            self.hldir = self.rundir + '/highlev/'
        else:
            self.hldir = er.clone_dir(self.rundir) + '/highlev/'

        self.fgtloc = self.hldir + 'FullGalaxyTables.hdf5'
        self.spiderloc = self.hldir + 'SpiderwebTables.hdf5'
        self.pathloc = self.hldir + 'GalaxyPaths.hdf5'
        self.cantorloc = (ht.clone_dir(self.hldir) 
                          + 'CantorCatalogue_7Jun19.hdf5')

        # Load precompiled data tables now (for simplicity):
        self.load_precompiled_data()

    def load_precompiled_data(self):                  # Class: Simulation
        """Load simulation-wide precompiled data"""
        
        if not par_load_from_cantor:
            self.shi = yb.read_hdf5(self.fgtloc, 'SHI')
        else:
            self.shi = yb.read_hdf5(self.cantorloc, 'SubHaloIndexExtended')

        self.numGal = self.shi.shape[0]
        
        # Set up an array of 'galaxy limits', for lookup (later)
        self.gal_lims = np.arange(self.numGal+1, dtype = np.int)  

    def load_snepshot_list(self, snepList='root'):    # Class: Simulation
        """
        Load a snepshot list for the simulation.

        Parameters:
        -----------
        snepList : string, optional
            The name of the snepshot list to load, defaults to 'root'.
            If snepList + '.dat' does not exist, an error is raised.
        """
        
        snepListFile = self.rundir + 'sneplists/' + snepList + '.dat'
        if not os.path.exists(snepListFile):
            raise Exception('Requested snepshot list "{:s}" does not exist.'
                            .format(snepList))

        snepData = ascii.read(snepListFile)
        self.snep_index = np.array(snepData['index'])
        self.snep_rootIndex = np.array(snepData['rootIndex'])
        self.snep_aexp = np.array(snepData['aexp'])
        self.snep_type = np.array(snepData['sourceType'])
        self.snep_num = np.array(snepData['sourceNum'])
        
        self.nSnep = len(self.snep_index)
    

class Galaxies:
    """
    Class for info about galaxies to be processed in one target snep.
    """

    def __init__(self, snep):
        """
        Constructor for the class.

        Sets up basic info.

        Parameters
        ----------
        snep : Snepshot class instance
            The target snepshot for assigning particles to galaxies.
        """
        
        self.snep = snep
        self.sim = self.snep.sim
        
        # Add pointer to underlying snepshot:
        snep.set_galaxies(self)

        # Set up counter of particle origins:
        self.numOrigins = np.zeros(2, dtype = int)
        self.numBoundOrigins = np.zeros(2, dtype = int)

        # Set up (initially) empty lists of lost galaxies:
        self.lost_galaxies = []
        self.snep = snep

    def find_galaxies(self):                            # Class: Galaxies
        """Find all galaxies to consider for current snepshot."""

        snep = self.snep   # For convenience
        self.gal = np.nonzero((self.sim.shi[:, snep.prev_snap] >= 0) | 
                              (self.sim.shi[:, snep.next_snap] >= 0))[0] 

        # If desired, limit galaxy selection to reasonably massive ones:
        if par_galaxy_thresholds is not None:
            msubMin = par_galaxy_thresholds[0]
            mstarMin = par_galaxy_thresholds[1]
            ind_massive = np.nonzero(
                (self.sim.msubPeak[self.gal] >= msub_min) | 
                (self.sim.mstarPeak[self.gal] > mstar_min))[0]
            self.gal = self.gal[ind_massive]
                             
        print("Found {:d} galaxies for snepshot {:d}..."
              .format(len(self.gal), self.snep.isnep))

    def add_origin_counts(self, numOrigins):            # Class: Galaxies
        """Increment the total counts of particles by origin."""
        self.numOrigins += numOrigins

    def add_bound_origin_counts(self, numOrigins):      # Class: Galaxies
        """Increment the total counts of bound particles by origin."""
        self.numBoundOrigins += numOrigins

    def add_lost_gal(self, igal):                       # Class: Galaxies
        """Add a galaxy to the internally-kept list of lost galaxies."""
        self.lost_galaxies.append(igal)
        

class Snepshot:
    """Class for general information/data about an individual snepshot."""
    
    def __init__(self, sim, isnep):
        """
        Constructor for the class.
        
        Parameters
        ----------
        sim : Simulation class instance
            The simulation to which this output belongs.
        isnep : int
            The index of the snepshot (in loaded snepshot list, may 
            or may not be root).
        """

        self.timeStamp = TimeStamp()
        self.sim = sim
        self.isnep = isnep
        
        self.aexp = sim.snep_aexp[isnep]
        self.rootIndex = sim.snep_rootIndex[isnep]
        self.type = sim.snep_type[isnep]
        self.num = sim.snep_num[isnep]

        self.zred = 1/self.aexp - 1
        self.hubble_z = Planck13.H(self.zred)#.value

        # Determine Plummer-equivalent softening length
        self.epsilon = min(1/(1+self.zred)*2.66*1e-3, 7e-4)

        print("Determined H(z) = {:.2f}, epsilon(z) = {:.2f} kpc." 
              .format(self.hubble_z, self.epsilon*1e3))
        if not self.hubble_z.unit == 'km / (s Mpc)':
            print("Hubble constant seems in non-standard units...")
            set_trace()

        self.partdir = st.form_files(self.sim.rundir, self.num, 'snap', 
                                     stype = self.type)
        
        # Set up arrays to hold galaxies' potential minimum:
        self.potMin_ID = np.zeros(sim.numGal, dtype = int) - 1
        self.potMin_pos = np.zeros((sim.numGal, 3), dtype = float) + np.nan
        
    def find_boundary_snaps(self):                      # Class: Snepshot
        """Find previous and next snapshot."""
        self.prev_snap = st.find_next_snap(
            self.aexp, dir = 'previous', include_extra = par_incl_extra_snaps)
        self.next_snap = st.find_next_snap(
            self.aexp, dir = 'next', include_extra = par_incl_extra_snaps)

        self.snap_prev = Snapshot(self.sim, self.prev_snap)
        self.snap_next = Snapshot(self.sim, self.next_snap)

        print("Determined bracketing snapshots as {:d} and {:d}."
              .format(self.prev_snap, self.next_snap))
        
    def load_coordinates(self):                         # Class: Snepshot
        """Load coordinates from GalaxyPaths."""
        snepPre = 'Snepshot_' + str(self.rootIndex).zfill(4)
        pathLoc = self.sim.pathloc

        # Consistency check:
        aexp_path = (
            yb.read_hdf5_attribute(pathLoc, snepPre, 'ExpansionFactor'))
        if np.abs(aexp_path-self.aexp) > 1e-4:
            print("Inconsistent expansion factor from GalaxyPaths...")
            set_trace()

        self.galPos = yb.read_hdf5(pathLoc, snepPre + '/Coordinates')
        self.galVel = yb.read_hdf5(pathLoc, snepPre + '/Velocity')

        # Convert position to proper units:
        pos_conv_aexp = yb.read_hdf5_attribute(
            pathLoc, snepPre + '/Coordinates', 'aexp-factor')
        pos_conv_h = yb.read_hdf5_attribute(
            pathLoc, snepPre + '/Coordinates', 'h-factor')
        self.galPos *= (pos_conv_aexp * pos_conv_h)

    def set_particles(self, particles):                 # Class: Snepshot
        """Set a reference to the snepshot's particle structure."""
        self.particles = particles

    def set_galaxies(self, galaxies):                   # Class: Snepshot
        """Set a reference to the snepshot's (target) galaxies."""
        self.target_galaxies = galaxies

    def print_time_usage(self, stime):                  # Class: Snepshot
        """
        Report time spent processing this (target) snapshot.
        
        Parameters:
        -----------
        stime : float
            The start time of processing this snapshot.
        """
        eprint("Place-holder for time-usage reporting, snepshot {:d}."
                  .format(self.isnep), linestyle = ':')

    def set_potMin(self, igal, index):                  # Class: Snepshot
        """
        Set the potential minimum particle for a given galaxy.

        Parameters:
        ----------
        igal : int
            The galaxyID whose potential minimum should be set.
        index : int
            The particle index (in self.particles) to set as pot. min.
        """

        self.potMin_ID[igal] = self.particles.ids[index]
        self.potMin_pos[igal, :] = self.particles.pos[index, :]


class Snapshot:
    """Class for general information/data about an individual snapshot."""
    
    def __init__(self, sim, isnap):
        """
        Constructor for the class.
        
        Parameters
        ----------
        sim : Simulation class instance
            The simulation to which this output belongs.
        isnap : int
            The index of the snapshot.
        """

        self.sim = sim
        self.isnap = isnap

        self.subdir = st.form_files(self.sim.rundir, isnap, 'sub')
        self.shi = sim.shi[:, isnap]

        if par_load_from_cantor:
            self.load_cantor_sh_data()
        else:
            self.load_sf_sh_data()

    def load_sf_sh_data(self):
        """Load subhalo--galaxy associations from Subfind."""
        
        self.off = st.eagleread(self.subdir, 'Subhalo/SubOffset', 
                                astro = False)
        self.len = st.eagleread(self.subdir, 'Subhalo/SubLength',
                                astro = False)
        self.ids = st.eagleread(self.subdir, 'IDs/ParticleID',
                                astro = False)

    def load_cantor_sh_data(self):
        """Load subhalo--galaxy associations from Cantor."""
 
        cantorloc = self.sim.cantorloc
       
        snapPre = 'Snapshot_' + str(self.isnap).zfill(3) + '/'
        self.off = yb.read_hdf5(cantorloc, snapPre + 'Subhalo/Offset')
        self.len = yb.read_hdf5(cantorloc, snapPre + 'Subhalo/Length')
        self.ids = yb.read_hdf5(cantorloc, snapPre + 'IDs')
        

class TargetParticles:
    """Class to load and hold particles in target snepshot."""

    def __init__(self, snep):
        """
        Constructor for this class.

        Determine total number of particles (by type) in the input snepshot,
        and set up internal storage for their required properties.
        The particle data are not actually read in here.

        Parameters:
        -----------
        snep : snepshot class instance
            The (target) snepshot in which to load particles.
        """

        self.snep = snep
        self.sim = self.snep.sim
        self.partdir = snep.partdir

        # Find the (relevant) total number of particles per type:
        self.npTotalType = yb.read_hdf5_attribute(self.partdir, 
                                                  'Header', 'NumPart_Total')
        for iptype in range(6):
            if iptype not in par_typeList: 
                self.npTotalType[iptype] = 0
        npTotal = np.sum(self.npTotalType)
        print("There are {:d} particles (in snep {:d}) in total..."
              .format(npTotal, self.snep.isnep))
        self.npTotal = npTotal

        # Set up full-particle arrays
        self.ids = np.zeros(npTotal, dtype = np.int)-1
        self.pos = np.zeros((npTotal, 3), dtype = np.float64)-1
        self.vel = np.zeros((npTotal, 3), dtype = np.float32)-1
        self.mass = np.zeros(npTotal, dtype = np.float32)-1
        self.energy = np.zeros(npTotal, dtype = np.float32)-1
        self.type = np.zeros(npTotal, dtype = np.int8)-1
        self.galaxy = np.zeros(npTotal, dtype = np.int32)-1
    
        # Add reference to target snapshot:
        snep.set_particles(self)

    def load_particle_data(self):                # Class: TargetParticles
        """Load actual particle data, for all particle types."""
                
        for iptype in par_typeList:
            if self.npTotalType[iptype] == 0: continue
            print("Loading {:d} particles of type {:d}..." 
                  .format(self.npTotalType[iptype], iptype))
            sTimeType = time.time()

            # NB: the following works because types 2 and 3 have been 
            #     explicitly zeroed in totals (if discarded)
            npStart = np.sum(self.npTotalType[:iptype])
            npEnd = np.sum(self.npTotalType[:iptype+1])
            pt = 'PartType{:d}/' .format(iptype)

            # 1.) Particle IDs
            ptype_ids = st.eagleread(self.partdir, pt + 'ParticleIDs', 
                                     astro = False)
            if ptype_ids.shape[0] != self.npTotalType[iptype]:
                print("Read unexpected number of particles: "
                      "{:d} instead of {:d}!"
                      .format(ptype_ids.shape[0], npTotalType[iptype]))
                set_trace()
            self.ids[npStart:npEnd] = ptype_ids

            # 2.) Particle positions and velocities:
            self.pos[npStart:npEnd, :] = st.eagleread(
                self.partdir, pt + 'Coordinates', astro = True)[0] 
            self.vel[npStart:npEnd, :] = st.eagleread(
                self.partdir, pt + 'Velocity', astro = True)[0]

            # 3.) Mass and internal energy -- depends on particle type:
            if iptype != 1:
                self.mass[npStart:npEnd] = st.eagleread(
                    self.partdir, pt + 'Mass', astro = True)[0]
            else:
                self.mass[npStart:npEnd] = (np.zeros(self.npTotalType[1], 
                                                     dtype = np.float32) 
                                            + st.m_dm(self.sim.rundir))

            if iptype == 0:
                self.energy[npStart:npEnd] = st.eagleread(
                    self.partdir, pt + 'InternalEnergy', astro = True)[0]

            # 4.) Record the particle type (for simplicity)
            self.type[npStart:npEnd] = iptype

            print("")
            print("- - - - - - - - - - - - - - - - - - - - - - ")
            print("Loading type {:d} took {:.2f} sec." 
                  .format(iptype, time.time() - sTimeType))
            print("- - - - - - - - - - - - - - - - - - - - - - ")
            print("")

        # Have now ended the loop over particle types to load
        # Sanity check to make sure all outputs are filled correctly
        if np.count_nonzero(self.type < 0):
            print("Why are some output fields not filled?!")
            set_trace()

    def get_indices(self, galaxy):               # Class: TargetParticles
        """
        Look up indices of a galaxy's IDs, and match to galaxy's FOF.

        Parameters:
        -----------
        galaxy : Galaxy instance
            The galaxy whose currently loaded IDs (galaxy.ids) are to be 
            looked up and checked for membership to its central.

        Returns:
        --------
        indices : ndarray (int)
            Indices into the internal particle arrays of the found particles.
        ind_found : ndarray (int)
            The index into the *input* (ID) array of particles that could
            be found (i.e. belong to the same FOF as the target galaxy).
        """

        ids = galaxy.gal_IDs              # For convenience
        
        # Find locations of particles using pre-computed reverse list:
        all_indices = self.revID[ids]

        # Identify particles that could be matched:
        ind_found = np.nonzero(all_indices >= 0)[0] 
        return all_indices[ind_found], ind_found

    def unbind_galaxy(self, gal):               # Class: TargetParticles
        """Perform gravitational unbinding for one specific galaxy."""

        stime = time.time()
        igal = gal.igal
        
        # Find bound particles:
        monk_res = self.monk(gal.part_inds, gal.pos, gal.vel, 
                             gal.target_snep.hubble_z.value,
                             fixCentre = par_monk_fixCentre, 
                             centreMode = par_monk_centering,
                             monotonic = par_monk_monotonic,
                             resLimit=galaxies.targetSnap.epsilon)
    
        if par_write_binding_energy:
            ind_bound = monk_res[0]
            binding_energy = monk_res[1]
        else:
            ind_bound = monk_res

        n_bound = len(ind_bound)

        if gal.verbose:
            print("Determined that {:d}/{:d} (={:.2f} per cent) of "
                  "particles remain bound."
                  .format(n_bound, len(sat.part_inds), 
                          n_bound / len(sat.part_inds) * 100))

        if n_bound:
            gal.print_bound_particle_statistics(ind_bound)
            self.update_galaxy_particles(gal, ind_bound)
            # Set the potential minimum particle:
            self.snep.set_potMin(igal, gal.part_inds[ind_bound[0]])
        else:
            print("Did not find any bound remnant of galaxy {:d}..."
                  .format(igal))
            self.snep.target_galaxies.add_lost_gal(igal)

        if verbose:
            print("Finished galaxy {:d} in {:.2f} sec."
                  .format(igal, time.time()-stime))

                                                 # Class: TargetParticles
    def update_galaxy_particles(self, galaxy, ind_bound):
        """
        Update particles' galaxy tag where permissible.

        Any particles that already have a galaxy tag with a stronger 
        association than for the current galaxy are not updated.

        Parameters:
        -----------
        galaxy : Galaxy instance
            The galaxy whose  bound particles should be updated.
        ind_bound : ndarray (int)
            The indices of the galaxy's particles that have been identified 
            as bound to the galaxy and that should now be updated.
        """

        # Construct full (internal) indices of to-be-tested particles:
        indices = galaxy.part_inds[ind_bound]

        # Determine current 'origin status' of particles, i.e. how they 
        # were associated to their currently tagged galaxy (if any):
        curr_origin = self.sourceType[indices]

        # Determine origin status w.r.t. new galaxy:
        new_origin = galaxy.origins[ind_bound]

        # Test for which particles re-association is allowed:
        ind_reassociate = np.nonzero(new_origin < curr_origin)[0]
        
        # Do the update for eligible particles:
        self.galaxy[indices[ind_reassociate]] = galaxy.igal
        self.sourceType[indices[ind_reassociate]] = (
            new_origin[ind_reassociate])

    def create_origin_arrays(self):              # Class: TargetParticles
        """
        Set up arrays to record association status of particles.

        These will hold information about the way in which each 
        particle is (potentially) claimed by a satellite galaxy.
        """

        nPart = len(self.ids)

        # A numerical code to record the association type 
        # (higher ==  weaker, so 255 -> none at all):
        self.sourceType = np.zeros(nPart, dtype = np.int8) + 255   
        
    def create_id_lookup_list(self):             # Class: TargetParticles
        """Set up a reverse ID list for rapid particle lookup."""
        
        stime = time.time()
        print("Inverting all-particle ID list...", end = '')
        maxID = 2 * (self.npTotalType[1] + 1)
        self.revID = yb.create_reverse_list(
            self.ids, maxval = maxID, delete_ids = False)
        print("done ({:.2f} sec.)!" .format(time.time() - stime))


                                                 # Class: TargetParticles
    def monk(self, indices, halopos_init, halovel_init, hubble_z, fixCentre,
             centreMode=0, centreFrac=0.1, status=None, monotonic=1,
             resLimit=0.0007, potErrTol=par_monk_potErrTol, 
             returnBE=par_write_binding_energy):

        """
        Run MONK to determine self-bound particles from input set.
 
        Parameters:
        -----------
        indices : ndarray (int), [N]
            The indices (into the list of currently loaded particles) 
            from which the self-bound subset should be determined.
        halopos_init : ndarray [3]
            The initial estimate of the halo position. This will be updated
            to the final position of the subhalo.
        halovel_init : ndarray [3]
            The initial estimate of the halo velocity. This will be updated 
            to the final velocity of the subhalo.
        hubble_z : float
            The Hubble constant in units of |vel|/|pos|, i.e. usually
            km/s/Mpc. Note that this must be the appropriate value for the 
            target snapshot redshift, *not* H_0!
        fixCentre : int
            Should the halo centre be fixed (1) or free (0)?
        centreMode : int, optional
            Should the halo be centred on the ZMF (0, default) or on the 
            subset of most-bound particles (1)? This is only relevant
            if fixCentre == 0.
        centreFrac : float, optional
            If fixCentre == 0 and centreMode == 1, specifies the fraction
            of most-bound particles to use for determining the halo
            centre (in position and velocity). Otherwise, this is ignored.
            Default value is 0.1 (i.e., use most-bound 10%).
        status : ndarray (int), optional
            If not None, an N-element array can be supplied that is 0 or 1
            depending on whether the i-th input particle is initially 
            unbound or bound, respectively. Unbound particles are treated 
            as mass-less, i.e. do not affect the potential. If None, all
            particles are assumed to be initially bound.
        monotonic : int, optional
            Set to 0 to allow 're-binding' of unbound particles. Note that
            this can, in principle, lead to runaway loops, so re-binding
            stops after 20 iterations.
        resLimit : float, optional
            The softening resolution limit, which sets the minimum
            distance between two particles or nodes within MONK. The default
            value is 0.0007 (0.7 kpc).
        potErrTol : float, optional
            The 'node opening criterion' within MONK. Defaut: 1.0
        returnBE : int, optional
            If 1 (default), the internal energy of bound particles will
            be updated to the binding energy. If 0, no binding energy
            information is passed back from MONK.

        Returns:
        --------
        ind_bound : ndarray (int), [<=N]
            The indices (into the input 'indices') that are bound.
        """

        gal_pos = self.pos[indices, :]
        gal_vel = self.vel[indices, :]
        pos6d = np.concatenate((gal_pos, gal_vel), axis = 1)

        gal_mass = self.mass[indices].astype(np.float32)
        gal_energy = self.energy[indices].astype(np.float32)

        # Set particles to initially bound, unless specified otherwise:
        if status is None:
            gal_status = np.zeros(len(indices), dtype = np.int32) + 1
        else:
            gal_status = status

        # Debugging/testing option that completely bypasses MONK:
        if par_bypass_monk:
            ind_bound = np.arange(len(indices), dtype = int)
            return ind_bound
    
        # Call MONK to find bound particles:
        # (Disable maxGap as removed from code)
        ind_bound = monk.monk(pos6d,        # 6D pos 
                              gal_mass,     # Mass
                              gal_energy,   # Int. energy
                              gal_status,   # Initial binding status
                              halopos_init, # Centre (pos)
                              halovel_init, # Centre (vel)
                              1,            # Mode (exact [0] or tree [1])
                              monotonic,    # Monotonic unbinding [1]?
                              0.005,        # Tolerance
                              -1,           # maxGap
                              fixCentre,    # fixCentre
                              centreMode,   # centreMode
                              centreFrac,   # centreFrac
                              hubble_z,     # Hubble (H(z))
                              resLimit,     # Softening limit
                              potErrTol,     # Potential error tolerance
                              par_monk_verb, # verbosity
                              returnBE)    # Return (Binding) Energy

        if par_write_binding_energy:
            return ind_bound, gal_energy
        else:
            return ind_bound

def main():

    # Set up derived parameters:
    setup()

    # Loop through simulations and process each individually:
    for isim in range(simRange[0], simRange[1]):
        simstime = time.time()
        process_simulation(isim)
        eprint("Finished processing all snepshots of simulation\n"
               "CE-{:d} in {:.2f} min."
               .format(isim, (time.time()-simstime)/60), linestyle = '=')

    print("Done!")
    

if __name__ == '__main__':
    main()


    
