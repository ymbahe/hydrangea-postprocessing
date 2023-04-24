"""
Program to re-compute subhalo membership of particles to satellites,
based on particles that belonged to it the last time it was a central.

Nomenclature: 

'reference snap': the snapshot at which each galaxy's starting set of particles
                  is determined.

'target snap': the snapshot that is currently processed.

Adapted to 'CANTOR' on 6-Apr-19
Continuous, severe, re-writing and extending until 22-Jun-2019
"""

import sim_tools as st
import yb_utils as yb
import numpy as np
from pdb import set_trace
#import monk
import time
import hydrangea_tools as ht
from scipy.spatial import cKDTree
import scipy.stats
import os
from astropy.cosmology import Planck13
from astropy.io import ascii
import gc
import pandas as pd
#import galquant
import ctypes as c
import yaml
import sys
import subprocess

GNewton = 6.67e-11  # Used in SI-units!

# ============================================================================
# ============================================================================

def main():

    # Parse input parameter file:
    par = parse_parameter_file(sys.argv[1])

    # Check if simulation was specified on the command line:
    if len(sys.argv) > 2:
        par['Sim']['Start'] = int(sys.argv[2])
        par['Sim']['End'] = par['Sim']['Start'] + 1

    if len(sys.argv) > 3:
        par['Snaps']['Start'] = int(sys.argv[3])
    if len(sys.argv) > 4:
        par['Snaps']['End'] = int(sys.argv[4])
        

    # Set derived parameters and perform consistency checks:
    setup_cantor(par)

    # Loop through simulations and process each individually:
    for isim in range(par['Sim']['Start'], par['Sim']['End']):
        simstime = time.time()
        process_simulation(isim)
        eprint("Finished processing all snapshots of simulation\n"
               "CE-{:d} in {:.2f} min."
               .format(isim, (time.time()-simstime)/60), linestyle = '=')

    print("Done!")


def dict2att(dictIn, outloc, container='Header', pre='',
             bool_as_int=True):
    """
    Write all elements of a dictionary as attributes to an HDF5 file.

    Typically, this is used to write the paramter structure of a 
    program to its output. If keys are themselves dictionaries, these 
    are recursively output with an underscore between them 
    (e.g. dictIn['Sim']['Input']['Flag'] --> 'Sim_Input_Flag').

    Parameters:
    -----------
    dictIn : dict
        The dictionary to output.
    outloc : string
        The HDF5 file name to write the dictionary to.
    container : string, optional
        The container to which the dict's elements will be written as 
        attributes (can be group or dataset). The default is a group 
        'Header'. If the container does not exist, a group with the specified
        name will be implicitly created.
    pre : string, optional:
        A prefix given to all keys from this dictionary. This is mostly
        used to output nested dictionaries (see description at top), but 
        may also be used to append a 'global' prefix to all keys.
    bool_as_int : bool, optional:
        If True (default), boolean keys will be written as 0 or 1, instead
        of as True and False.
    """

    if len(pre):
        preOut = pre + '_'
    else:
        preOut = pre

    for key in dictIn.keys():

        value = dictIn[key]
        if isinstance(value, dict):
            # Nested dict: call function again to iterate
            dict2att(value, outloc, container = container, 
                     pre = preOut + key, bool_as_int=bool_as_int)
        else:
            # Single value: write to HDF5 file

            if value is None:
                value = 0

            if bool_as_int and isinstance(value, bool):
                value = int(value)

            if isinstance(value, str):
                value = np.string_(value)

            yb.write_hdf5_attribute(outloc, container, preOut + key, 
                                    value)

def dict2out(dictIn, bool_as_int=True, pre=''):
    """
    Write all elements of a dictionary as attributes to the output.

    If any keys are themselves dictionaries, these 
    are recursively output with an underscore between them 
    (e.g. dictIn['Sim']['Input']['Flag'] --> 'Sim_Input_Flag').

    Parameters:
    -----------
    dictIn : dict
        The dictionary to output.
    bool_as_int : bool, optional:
        If True (default), boolean keys will be written as 0 or 1, instead
        of as True and False.
    """

    if len(pre):
        preOut = pre + '_'
    else:
        preOut = pre

    for key in dictIn.keys():

        value = dictIn[key]
        if isinstance(value, dict):
            # Nested dict: call function again to iterate
            dict2out(value, pre = preOut + key, bool_as_int=bool_as_int)
        else:
            # Single value: write to HDF5 file

            if value is None:
                value = 0

            if bool_as_int and isinstance(value, bool):
                value = int(value)

            print(preOut + key, ': ', value) 


def eprint(string, linestyle = '-', padWidth = 1, lineWidth = 1, 
           textPad = None):
    """
    Print a string with padding.

    Parameters:
    -----------

    string : string
        The text string to print in the middle of the padding.
    linestyle : string, optional
        The pattern to use for framing the string, default: '-'.
    padWidth : int, optional
        The number of blank lines to print either side of the frame,
        defaults to 1.
    lineWidth : int, optional
        The number of lines to draw either side of the string,
        defaults to 1.
    textPad : int, optional
        If not None, the width to which the string is padded with 
        the framing linestyle (centrally aligned).
    """
    
    stringLength = len(string)
    lineElementLength = len(linestyle)

    if textPad is not None:
        padLength = textPad//2 - (stringLength//2 + 1)
        if padLength < 0:
            padLength = 0
        numPad = padLength//lineElementLength + 1
        padString = (linestyle * numPad)[:padLength]

        string = padString + ' ' + string + ' ' + padString
        string = string[:textPad]
        stringLength = len(string)

    # Work out how often the line element must be repeated:
    numLines = stringLength // lineElementLength
    line = (linestyle * numLines)[:stringLength]
        
    for ipad in range(padWidth):
        print("")
    for iline in range(lineWidth):
        print(line)
    print(string)
    for iline in range(lineWidth):
        print(line)
    for ipad in range(padWidth):
        print("")

def print_memory_usage(pre=''):
    """
    Print (total) current memory usage of the program, as seen by the OS.

    Parameters:
    -----------
    pre : str, optional
        A string that precedes the memory usage message (typically used
        to indicate where in the program the measurement is done).
    """

    # Get program's PID
    pid = os.getpid()

    # Call pmap to get memory footprint
    ret = subprocess.check_output(['pmap', str(pid)])

    # Extract the one number we care about from the output: total usage
    rets = str(ret)
    ind = rets.find("Total:")
    part = rets[ind:]
    colInd = part.find(":")
    kInd = part.find("K")
    mem = int(part[colInd+1 : kInd])

    print(pre + "Using {:d} KB (={:.1f} MB, {:.1f} GB)." 
          .format(mem, mem/1024, mem/1024/1024))

def parse_parameter_file(par_file):
    """Parse the YAML input parameter file to `par' dict."""

    global par

    with open(par_file, 'r') as stream:
        try:
            par = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return par

def setup_cantor(par):
    """Set derived parameters and do consistency checks."""

    # Set derived parameters:
    if (par['Lost']['FindTemporarilyLost'] or 
        par['Lost']['FindPermanentlyLost']):
        par['Lost']['Recover'] = True
    else:
        par['Lost']['Recover'] = False

    if par['Input']['IncludeBoundary']:
        par['Input']['TypeList'] = np.arange(6, dtype = np.int8)
    else:
        par['Input']['TypeList'] = np.array([0, 1, 4, 5])

    # Consistency checks and warnings:
    if par['Lost']['FindPermanentlyLost'] and not par['Input']['FromCantor']:
        raise Exception("Unbinding lost galaxies requires "
                        "loading CANTOR input.")
    
    if par['Sources']['Sats'] or par['Sources']['FOF']:
        if not par['Input']['RegularizedCens']:
            eprint("WARNING", textPad = 60, linestyle = '=')
            print("Including root sat/fof particles without using regularized")
            print("cen-sat input may lead to nonsensical results.")
            print("")
            
        if not par['Input']['FromCantor']:
            eprint("WARNING", textPad = 60, linestyle = '=')
            print("Including root sat/fof particles without using CANTOR")
            print("input may lead to nonsensical results.")
            print("")

    if (not par['Output']['COPAtUnbinding'] 
        or par['Output']['WriteBindingEnergy']):
        par['Monk']['ReturnBindingEnergy'] = 1
    else:
        par['Monk']['ReturnBindingEnergy'] = 0

    if par['Sources']['Prior'] and not par['Input']['FromCantor']:
        eprint("INCONSISTENCY", textPad = 60, linestyle = '@')
        raise Exception("Cannot use previous snapshot without "
                        "loading Cantor output.")

    if par['Lost']['MaxLostSnaps'] is None:
        par['Lost']['MaxLostSnaps'] = np.inf

    if (par['Lost']['Recover'] and par['Sources']['Subfind'] 
        and not par['Sources']['Prior']):
        eprint("WARNING", textPad = 60, linestyle = '=')
        print("Included subfind particles and included lost")
        print("galaxies, but not loaded previous Cantor particles.")

    # Write parameter structure to output:
    print("\n----------------------------------------------------------")
    print("Cantor configuration:")
    print("-----------------------------------------------------------")
    dict2out(par)
    print("-----------------------------------------------------------")
    print("")

    return


def process_simulation(isim):
    """Run Cantor on one simulation."""

    print_memory_usage("At start of simulation {:d}: " .format(isim))

    # Set up a 'Simulation' class instance that holds its basic properties:
    sim = Simulation(isim, rootdir = par['Sim']['Rootdir'], 
                     simType = par['Sim']['Type'])

    if not sim.exists:
        print("Simulation {:d} does not exist, skipping..." .format(isim))
        return

    # Set up time-stamp associated to simulation:
    sim.timeStamp = TimeStamp()

    # Set up a 'CantorOutput' class instance for its output:
    sim_output = CantorOutput(sim)
    sim.timeStamp.set_time('Simulation setup')

    # Loop (forwards) over target snapshots. In each, subhalo membership 
    # of particles is re-established, with separate unbinding steps for all 
    # previous snapshots (for subhaloes whose root snapshots these are).
    for isnap_proc in range(par['Snaps']['Start'], par['Snaps']['End']):
        print_memory_usage("At start of target snap {:d}: " 
                           .format(isnap_proc))
        process_sim_snapshot(isnap_proc, sim, sim_output)
        sim.timeStamp.set_time('Snapshot {:d}' .format(isnap_proc))
    
    sim.timeStamp.print_time_usage("Finished processing simulation {:d}"
                                   .format(isim), mode='top', minutes=True)
    sim.timeStamp.print_time_usage("Finished processing simulation {:d}"
                                   .format(isim), mode='sub', minutes=True)

    print_memory_usage("End of simulation {:d}, before cleaning: "
                       .format(isim))
    #set_trace()
    del sim_output
    del sim
    gc.collect()
    print_memory_usage("After cleaning: ")


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

        self.isim = isim
        self.rundir = rootdir + 'CE-' + str(isim) + '/' + simType + '/'    
        self.stime = time.time()  # Time of starting to process it

        if not os.path.isdir(self.rundir):
            self.exists = False
            return
        self.exists = True

        if simClass == 'Hydrangea':
            self.hldir = self.rundir + '/highlev/'
        else:
            self.hldir = er.clone_dir(self.rundir) + '/highlev/'

        self.pathloc = self.hldir + 'GalaxyPaths.hdf5'
        self.fgtloc = self.hldir + 'FullGalaxyTables.hdf5'
        self.spiderloc = self.hldir + 'SpiderwebTables.hdf5'
        self.posloc = self.hldir + 'GalaxyPositionsSnap.hdf5'
        self.extraloc = self.hldir + 'SubhaloExtra.hdf5'
        self.regCenLoc = (ht.clone_dir(self.hldir) 
                          + '/RegularizedCentrals.hdf5')
        if not os.path.exists(self.fgtloc):
            self.fgtloc = None
        if not os.path.exists(self.spiderloc):
            self.spiderloc = None
        if not os.path.exists(self.posloc):
            self.posloc = None
        if not os.path.exists(self.extraloc):
            self.extraloc = None
            
        # Load precompiled data tables now (for simplicity):
        self.load_precompiled_data()

    def __del__(self):
        """Destructor to minimize memory leaks"""
        gc.collect()

    def load_precompiled_data(self):                  # Class: Simulation
        """Load simulation-wide precompiled data"""
        
        if par['Input']['RegularizedCens']:
            self.cenGal = yb.read_hdf5(self.regCenLoc, 'CenGal_Regularized')
            self.satFlag = yb.read_hdf5(self.regCenLoc, 'SatFlag_Regularized')
        else:
            self.cenGal = yb.read_hdf5(self.fgtloc, 'CenGal')        
            self.satFlag = yb.read_hdf5(self.fgtloc, 'SatFlag')

        if (not par['Sources']['SubfindInSwaps'] 
            and par['Input']['RegularizedCens']):
            self.SFsatFlag = yb.read_hdf5(self.fgtloc, 'SatFlag')

        self.shi = yb.read_hdf5(self.fgtloc, 'SHI')
        self.msub = yb.read_hdf5(self.fgtloc, 'Msub')

        self.mergeList = yb.read_hdf5(self.spiderloc, 'MergeList')

        if par['Galaxies']['DiscardSpectres']:
            self.spectreFlag = yb.read_hdf5(self.fgtloc, 'Full/SpectreFlag')
            self.spectreParents = yb.read_hdf5(self.fgtloc, 
                                               'Full/SpectreParents')

        self.lastsnap = yb.read_hdf5(self.spiderloc, 'LastSnap')

        self.numGal = self.satFlag.shape[0]
        self.galPos = yb.read_hdf5(self.posloc, 'Centre')
        self.galVel = yb.read_hdf5(self.posloc, 'Velocity')
        
        if par['Sources']['Centre'] is not None:
            self.shmr = yb.read_hdf5(self.fgtloc, 'StellarHalfMassRad')

        # Set up an array of 'galaxy limits', for lookup (later)
        self.gal_lims = np.arange(self.numGal+1, dtype = np.int)  

    def set_output(self, output):                     # Class: Simulation
        """Set a reference to this simulation's output instance."""
        self.output = output

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

    def print_time_statistics(self):                  # Class: Simulation
        """Print final time consumption statistics for simulation."""
        runtime = time.time() - self.stime
        eprint("Finished processing simulation {:d} in {:.3f} min."
               .format(self.isim, runtime/60.0), linestyle = '=')
        

class CantorOutput:
    """Class for storing and writing the output from Cantor."""

    def __init__(self, sim):
        """
        Constructor for the class.

        Parameters
        ----------
        sim : Simulation class instance
            The simulation to which this output belongs.
        """

        self.sim = sim
        snap_start = par['Snaps']['Start']

        # Set up the Cantor output file for this simulation:
        self.outloc = ht.clone_dir(sim.hldir) + par['Output']['File']
        if not (par['Input']['FromCantor'] and snap_start > 0):
            if os.path.exists(self.outloc):
                os.rename(self.outloc, self.outloc + '.old')
        
        # Set up cross-snapshot output array(s):
        if par['Input']['FromCantor'] and snap_start > 0:
            self.shiExtended = yb.read_hdf5(self.outloc,
                                                'SubhaloIndex')
        else:
            self.shiExtended = np.zeros_like(self.sim.shi)-1            
            ind_sf_dead = np.nonzero(sim.shi < 0)

            # Copy different codes for non-detection over from SF:
            self.shiExtended[ind_sf_dead] = sim.shi[ind_sf_dead]

        if par['Lost']['Recover']:
            if par['Input']['FromCantor'] and snap_start > 0:
                # Need to load these from prior (partial) output
                # file if continuing from snap > 0:
                self.cenGalExtended = yb.read_hdf5(self.outloc, 
                                                   'CenGalExtended')
            else:
                self.cenGalExtended = np.zeros((self.sim.numGal, 
                                                par['Snaps']['Num']), 
                                               dtype = np.int32)-1

        # Initialize last-written snapshot 
        if par['Input']['FromCantor'] and snap_start > 0:
            # Last-written is the one before the start!
            self.lastSnap = snap_start - 1
        else:
            #  -1 --> not even 0 written
            self.lastSnap = -1

        self.write_header()
        sim.set_output(self)

    def __del__(self):
        """Destructor, to minimize memory leaks"""
        #del self.shiExtended
        #del self.cenGalExtended
        #set_trace()
        #sim.set_output(None)
        gc.collect()

    def write_header(self):
        """Write header with run parameters."""

        dict2att(par, self.outloc, 'Header', bool_as_int = True)
        yb.write_hdf5_attribute(self.outloc, 'Header', 'Simulation', 
                                self.sim.isim)

    def write(self, snap):                        # Class: CantorOutput
        """Write internally-stored cross-snapshot data for snap."""
        
        isnap = snap.isnap
        galaxies = snap.target_galaxies

        # New central-list including resuscitated galaxies
        if par['Lost']['Recover']:
            yb.write_hdf5(self.cenGalExtended, self.outloc, 'CenGalExtended', 
                          comment = "Central galaxy ID for galaxy i "
                          "(first index) in snapshot j (second index). "
                          "This includes as far as possible galaxies that "
                          "were not identified by Subfind.")
        
        # New Galaxy --> subhalo list (using cantor IDs)
        # Write new galID <--> SHI lists:
        yb.write_hdf5(self.shiExtended, self.outloc, 
                      'SubhaloIndex',
                      comment = "Subhalo index of galaxy i (first index) "
                      "in snapshot j (second index). Note that this is "
                      "in general different from the corresponding "
                      "Subfind index, even for subhaloes that were "
                      "detected by both Subfind and Cantor. In each "
                      "snapshot, the 'Subhalo/SubfindIndex' and "
                      "'Subhalo/IndexBySubfindID' arrays can be used "
                      "to translate between Cantor and Subfind indices.")

        # Update last-processed snapshot:
        self.lastSnap = isnap

    def update_shiX(self, snap_out):              # Class: CantorOutput
        """Update cross-snapshot arrays from snapshot output class"""
        
        isnap = snap_out.snap.isnap

        self.shiExtended[snap_out.sh_galID, isnap] = (
            np.arange(snap_out.nSH))
        
        if par['Lost']['Recover']:
            self.cenGalExtended[:, isnap] = (
                snap_out.snap.target_galaxies.cenGalAssoc)
            

class SnapshotOutput:
    """Class for storing and writing one snapshot's output from Cantor."""

    def __init__(self, snap, order_cens_first, full_snapshot=True,
                 outloc=None):
        """
        Constructor for the class.

        Parameters
        ----------
        snap : Snapshot class instance
            The snapshot to which the output belongs.
        order_cens_first : bool
            If True, the subhaloes will be sorted so that the cen is 
            always first. Requires that central information is available.
        full_snapshot : bool, optional
            If True, some extra output will be computed and written that 
            is only relevant for snapshots (such as SF-cross-indices).
            Default is True.
        outloc : str, optional
            The output file to write the results to. If None (default),
            this is the same as the simulation-wide output file.
        """

        self.snap = snap
        self.sim = snap.sim
        self.order_cens_first = order_cens_first
        self.full_snapshot = full_snapshot

        if full_snapshot:
            self.sim_output = self.sim.output

        # Initialize internal time-stamp 
        self.timeStamp = TimeStamp()

        # Retrieve the Cantor output file for this simulation:
        if outloc is None:
            self.outloc = ht.clone_dir(self.sim.hldir) + par['Output']['File']
        else:
            self.outloc = outloc

        if self.full_snapshot:
            # Initialize last-written snapshot 
            if par['Input']['FromCantor'] and par['Snaps']['Start'] > 0:
                # Last-written is the one before the start!
                self.lastSnap = par['Snaps']['Start'] - 1
            else:
                #  -1 --> not even 0 written
                self.lastSnap = -1

        snap.output = self
        self.timeStamp.set_time('Setup')

    def __del__(self):
        """Destructor, to minimize memory leaks"""
        gc.collect()
                                                    # Class: SnapshotOutput
    def setup_snapshot_outputs(self):
        """Set up snapshot-specific output arrays."""

        # Find number of subhaloes in this snapshot
        nSH = self.nSH

        # Particle offset and length information
        self.sh_offset = np.zeros(nSH, dtype = np.int64) - 1
        self.sh_length = np.zeros(nSH, dtype = np.int32) - 1
        self.sh_lengthType = np.zeros((nSH, 6), dtype = np.int32)
        self.sh_offsetType = np.zeros((nSH, 7), dtype = np.int64)
        self.sh_offsetTypeAp = np.zeros((nSH, 6, 4), dtype = np.int64) - 1
        
        # Mass by particle type and apertures
        self.sh_massTypeAp = np.zeros((nSH, 6, 5), dtype = np.float32)
        self.sh_mass = np.zeros(nSH, dtype = np.float32)

        # Centre-of-mass position and velocity:
        self.sh_comPos = np.zeros((nSH, 3), dtype = np.float32) + np.nan
        self.sh_zmfVel = np.zeros((nSH, 3), dtype = np.float32) + np.nan
        self.sh_comPosType = np.zeros((nSH, 6, 3), dtype = np.float32) + np.nan
        self.sh_zmfVelType = np.zeros((nSH, 9, 3), dtype = np.float32) + np.nan

        # Velocity dispersions: 
        self.sh_velDisp = np.zeros((nSH, 6), dtype = np.float32) + np.nan
        
        # Maximum circular velocity and its radius:
        self.sh_vmax = np.zeros(nSH, dtype = np.float32) + np.nan
        self.sh_radOfVmax = np.zeros(nSH, dtype = np.float32) + np.nan

        # Maximum radii:
        self.sh_rMax = np.zeros(nSH, dtype = np.float32) + np.nan
        self.sh_rMaxType = np.zeros((nSH, 6), dtype = np.float32) + np.nan

        # Angular momentum vectors:
        self.sh_angMom = np.zeros((nSH, 9, 3), dtype = np.float32) + np.nan

        # MIT axes and ratios:
        self.sh_axes = np.zeros((nSH, 6, 3, 3), dtype = np.float32) + np.nan
        self.sh_axRat = np.zeros((nSH, 6, 2), dtype = np.float32) + np.nan

        # Stellar kinematic morphology parameters (Correa+17):
        self.sh_kappaCo = np.zeros((nSH, 2), dtype = np.float32) + np.nan
        
        # Stellar mass percentile radii:
        self.sh_smr = np.zeros((nSH, 2, 3), dtype = np.float32) + np.nan
        
        self.timeStamp.set_time('Set up outputs')


    def compute_cantorID(self):                   # Class: SnapshotOutput
        """
        Establish the subhalo order of all galaxies in current snap.

        For each galaxy, the total mass is computed, and they are then 
        sorted first by FOF-index and second by mass (in descending order).
        However, the central galaxy is always output first (provided its
        mass is > 0, of course).
        
        Writes:
        --------
            self.(nSH, sh_galID, sh_fof, fof_fsh, fof_nsh, fof_cenSH)
        """

        print("Computing Cantor subhalo IDs ", end='', flush = True)

        snap = self.snap
        particles = snap.particles
        galaxies = snap.target_galaxies

        if self.full_snapshot:
            cenGal = galaxies.cenGalAssoc        

        # Compute total mass of each galaxy and find those with m > 0
        df = pd.DataFrame({'mass':particles.mass, 'galaxy':particles.galaxy})
        gal_mass = df.groupby('galaxy')['mass'].sum()

        gal_found_mass = gal_mass.values
        ind_found = gal_mass.index.values

        # We need to make sure we exclude fake '-1' galaxies!
        ind_real = np.nonzero(ind_found >= 0)[0]
        gal_found_mass = gal_found_mass[ind_real]
        ind_found = ind_found[ind_real]

        self.nSH = len(ind_found)

        print("(N_SH={:d})... " .format(self.nSH), end = '', flush = True)

        if self.full_snapshot:
            # Establish FOF-index of each identified galaxy
            fof = snap.fof_index[snap.shi[cenGal[ind_found]]]

            if self.order_cens_first:
                # Set mass of centrals to infinity, to make sure they are 
                # listed first in each FOF group
                ind_cen = np.nonzero(
                    snap.sim.satFlag[ind_found, snap.isnap] == 0)[0]
                gal_found_mass[ind_cen] = np.inf

            # Create double-sorter. This is essentially the translation array
            # CantorID --> GalID, which we'll invert then.
            sorter = np.lexsort((-gal_found_mass, fof))
            
        else:
            sorter = np.argsort(-gal_found_mass)

        self.sh_galID = ind_found[sorter]
        self.gal_cantorID = yb.create_reverse_list(self.sh_galID, 
                                                   maxval = snap.sim.numGal-1)

        # Set up look-up table for particles by cantorID
        self.part_lut_galaxy = SplitList(particles.galaxy, self.sim.gal_lims) 

        if self.full_snapshot:
            # Get FOF-subhalo offsets and lengths
            self.sh_fof = snap.fof_index[snap.shi[cenGal[self.sh_galID]]]
            sh_lut_fof = SplitList(self.sh_fof, 
                                   np.arange(snap.nFOF+1, dtype = np.int))

            self.fof_fsh = sh_lut_fof.splits
            self.fof_nsh = self.fof_fsh[1:] - self.fof_fsh[:-1]

            # For convenience, also create a duplicate of FSH that is set to 
            # -1 for FOFs without any galaxies:
            self.fof_cenSH = np.copy(self.fof_fsh[:-1])
            ind_fof_noSH = np.nonzero(self.fof_nsh == 0)[0]
            self.fof_cenSH[ind_fof_noSH] = -1

        self.timeStamp.set_time('Sort subhaloes')
        print("done (took {:.3f} sec.)."
              .format(self.timeStamp.get_time()))
        
    def validate_subhaloes(self):                 # Class: SnapshotOutput
        """Validate that number of subhaloes matches expectations."""
      
        snap = self.snap
        particles = snap.particles
        galaxies = snap.target_galaxies

        # Extract resuscitated/lost galaxies from their sets
        resuscitated = np.array(list(galaxies.resuscitated))
        lost_cens = galaxies.lost_cens
        lost_sats = galaxies.lost_sats
        eliminated = particles.lost_gals

        # Find total number of galaxies tried for resuscitation
        n_resusc_cand = np.count_nonzero(
            snap.shi[galaxies.candidate_galaxies] < 0)

        print("Snap {:d}: lost {:d} cens and {:d} sats, {:d} eliminated."
              .format(snap.isnap, len(lost_cens), len(lost_sats), 
                      len(eliminated)))
        if n_resusc_cand:
            print("   Resuscitated {:d} galaxies (out of {:d} candidates, "
                  "={:.2f}%)"
            .format(len(resuscitated), n_resusc_cand, 
                    len(resuscitated)/n_resusc_cand*100))
        
        # Now verify that numbers match up...
        diff = (self.nSH + len(lost_cens) + len(lost_sats) + len(eliminated)
                + len(galaxies.spectres)
                - len(resuscitated) - snap.nSH)

        if diff:
            print("Inconsistent numbers of subhaloes (diff={:d})."
                  "Please investigate." .format(diff))
            set_trace()

    def compute_centre_and_radii(self):           # Class: SnapshotOutput
        """
        Compute/extract centre-of-potential of each galaxy, and the radii
        of all particles from these centres.
        """
        
        print("Get subhalo centres and particle radii... ", end = '',
              flush = True)

        particles = self.snap.particles 
        galaxies = self.snap.target_galaxies

        # Find centre-of-potential, depending on settings, and particle 
        # distances from this centre:
        if par['Output']['COPAtUnbinding']:
            galCen = galaxies.get_potMin_pos_multi(self.sh_galID)
        else:
            # Re-compute COP now, based on particles NOW in galaxy:
            galCen = particles.get_potMin_pos_multi(self.sh_galID)
        self.sh_centreOfPotential = galCen

        # Calculate particle radii
        particles.calculate_all_radii()

        # Also record final frame coordinates from MONK:
        self.sh_monkPos = galaxies.get_pos_multi(self.sh_galID)
        self.sh_monkVel = galaxies.get_vel_multi(self.sh_galID)

        self.timeStamp.set_time('Compute centre and radii...')
        print("done (took {:.3f} sec.)."
              .format(self.timeStamp.get_time()))

    def arrange_particles(self):                  # Class: SnapshotOutput
        """
        Establish the near- ordering of particles.

        Particles are sorted by FOF --> subhalo --> radius.
        (Particles not in a subhalo are put at the end of each FOF block,
        and are not sorted by radius). The particle list is finally
        re-ordered at a later step, where it is split by type.
        """

        print("Re-arrange particle sequence... ", end = '', flush = True)

        snap = self.snap
        particles = snap.particles

        if self.full_snapshot:
            # Establish FOF-index of each particle
            part_fof = snap.fof_index[snap.shi[particles.fof_cenGal]]
        
            # Set up by-FOF lookup table 
            # (go via fof_cenGal in case of SF-dead galaxies)
            part_lut_fof = SplitList(part_fof, 
                                     np.arange(snap.nFOF+1, dtype = np.int))
        
            # First, we need the 'boundaries' of the lookup table directly:
            self.fof_offset = part_lut_fof.splits
            self.fof_length = self.fof_offset[1:] - self.fof_offset[:-1]
        
            if (self.fof_offset[0] != 0 or 
                self.fof_offset[-1] != len(particles.fof_cenGal)):
                print("Unexpected FOF split beginning/end values!")
                set_trace()

        # Set up a list to sort IDs into output order: sort by
        # [FOF] --> subhalo --> type --> radius
        particleSortList = np.zeros_like(particles.ids)-1

        # Set up array of SHI per particle. Those not in SH are assigned
        # a SHI one above the maximum, so they will be sorted to the end 
        # of their FOF group.
        part_shi = self.gal_cantorID[particles.galaxy]
        ind_not_in_sh = np.nonzero(part_shi < 0)[0]

        if self.full_snapshot:
            if len(ind_not_in_sh) and not par['Input']['LoadFullFOF']:
                print("We have particles outside galaxies, but did not "
                      "load full FOF. Should not happen.")
                set_trace()

        part_shi[ind_not_in_sh] = self.nSH+1
        particles.galaxy = part_shi

        # Sort all particles in one go! Do *NOT* split
        # by type yet, because we first need them combined-by-radius for
        # vmax computation. Split by types will be done by 
        # external property calculation routine.

        if self.full_snapshot:
            particleSortList = np.lexsort(
                (particles.rad, particles.galaxy, part_fof))
        else:
            particleSortList = np.lexsort(
                (particles.rad, particles.galaxy))
            
        particles.ids = particles.ids[particleSortList]
        particles.mass = particles.mass[particleSortList]
        particles.pos = particles.pos[particleSortList, :]
        particles.vel = particles.vel[particleSortList, :]
        particles.type = particles.type[particleSortList]
        particles.rad = particles.rad[particleSortList]
        particles.galaxy = particles.galaxy[particleSortList]

        self.timeStamp.set_time('Sort particles')
        print("done (took {:.3f} sec.)." 
              .format(self.timeStamp.get_time()))

    def compute_galaxy_properties(self):          # Class: SnapshotOutput
        """Compute galaxy properties through external C code."""

        print("Computing galaxy properties... ", end = '', flush = True)

        particles = self.snap.particles

        # *********** IMPORTANT ********************************
        # This next line needs to be modified to point
        # to the full path of where the library has been copied.
        # *******************************************************

        ObjectFile = "/u/ybahe/ANALYSIS/PACKAGES/lib/galquant.so"

        numSH = self.sh_centreOfPotential.shape[0]
        numPart = particles.mass.shape[0]
        verbose = 0

        c_numPart = c.c_long(numPart)
        c_numSH = c.c_int(numSH)
        c_verbose = c.c_int(verbose)
        c_epsilon = c.c_float(self.snap.epsilon)

        mass_p = particles.mass.ctypes.data_as(c.c_void_p)
        pos_p = particles.pos.ctypes.data_as(c.c_void_p)
        vel_p = particles.vel.ctypes.data_as(c.c_void_p)
        type_p = particles.type.ctypes.data_as(c.c_void_p)
        shi_p = particles.galaxy.ctypes.data_as(c.c_void_p)
        rad_p = particles.rad.ctypes.data_as(c.c_void_p)
        ids_p = particles.ids.ctypes.data_as(c.c_void_p)

        cop_p = self.sh_centreOfPotential.ctypes.data_as(c.c_void_p)
        off_p = self.sh_offset.ctypes.data_as(c.c_void_p)
        len_p = self.sh_length.ctypes.data_as(c.c_void_p)
        lenType_p = self.sh_lengthType.ctypes.data_as(c.c_void_p)
        offType_p = self.sh_offsetType.ctypes.data_as(c.c_void_p)
        offTypeAp_p = self.sh_offsetTypeAp.ctypes.data_as(c.c_void_p)
        massTypeAp_p = self.sh_massTypeAp.ctypes.data_as(c.c_void_p)
        vmax_p = self.sh_vmax.ctypes.data_as(c.c_void_p)
        rvmax_p = self.sh_radOfVmax.ctypes.data_as(c.c_void_p)
        mtot_p = self.sh_mass.ctypes.data_as(c.c_void_p)
        comPos_p = self.sh_comPos.ctypes.data_as(c.c_void_p)
        zmfVel_p = self.sh_zmfVel.ctypes.data_as(c.c_void_p)
        rMax_p = self.sh_rMax.ctypes.data_as(c.c_void_p)
        rMaxType_p = self.sh_rMaxType.ctypes.data_as(c.c_void_p)
        comPosType_p = self.sh_comPosType.ctypes.data_as(c.c_void_p)
        zmfVelType_p = self.sh_zmfVelType.ctypes.data_as(c.c_void_p)
        velDisp_p = self.sh_velDisp.ctypes.data_as(c.c_void_p)
        angMom_p = self.sh_angMom.ctypes.data_as(c.c_void_p)
        axes_p = self.sh_axes.ctypes.data_as(c.c_void_p)
        axRat_p = self.sh_axRat.ctypes.data_as(c.c_void_p)
        kappaCo_p = self.sh_kappaCo.ctypes.data_as(c.c_void_p)
        smr_p = self.sh_smr.ctypes.data_as(c.c_void_p)
        
        nargs = 33
        myargv = c.c_void_p * 33
        argv = myargv(c.addressof(c_numPart), 
                      c.addressof(c_numSH),
                      mass_p, pos_p, vel_p, type_p, shi_p, rad_p, ids_p,
                      cop_p, off_p, len_p, lenType_p, offType_p,
                      offTypeAp_p,
                      massTypeAp_p, vmax_p, rvmax_p, mtot_p, 
                      comPos_p, zmfVel_p, rMax_p, rMaxType_p, 
                      comPosType_p, zmfVelType_p, velDisp_p, 
                      angMom_p, axes_p, axRat_p, kappaCo_p, smr_p,
                      c.addressof(c_verbose), c.addressof(c_epsilon))

        lib = c.cdll.LoadLibrary(ObjectFile)
        succ = lib.galquant(nargs, argv)

        self.timeStamp.set_time('Compute galaxy properties')               
        print("done (took {:.3f} sec.)." 
              .format(self.timeStamp.get_time()))

    def write(self, prefix=None):                  # Class: SnapshotOutput
        """
        Write internally-stored data to file.

        Parameters:
        -----------
        prefix : string, optional
            The HDF5 group name to which output will be written. If None
            (default), 'Snapshot_0xx/' will be used.
        """
        
        print("Writing output to HDF5 file... ", end = '')
        
        snap = self.snap
        isnap = snap.isnap      # For convenience
        galaxies = snap.target_galaxies
        particles = snap.particles

        # Set up snapshot-dependent prefix in output file:
        if prefix is None:
            snapPre = 'Snapshot_' + str(isnap).zfill(3) + '/'
        elif len(prefix) == 0:
            snapPre = ''
        elif prefix[-1] != '/':
            snapPre = prefix + '/'
        else:
            snapPre = prefix

        if len(snapPre) == 0:
            attDir = 'Header'
        else:
            attDir = snapPre

        # Write snapshot-specific header information:
        yb.write_hdf5_attribute(self.outloc, attDir, 'Redshift', snap.zred)
        yb.write_hdf5_attribute(self.outloc, attDir, 'aExp', 1/(1+snap.zred))
        yb.write_hdf5_attribute(self.outloc, attDir, 'NumSubhalo',
                                self.nSH)
        if self.full_snapshot:
            yb.write_hdf5_attribute(self.outloc, attDir, 'NumFOF', snap.nFOF)

        # Construct Cantor <--> subfind cross-indices: 
        if self.full_snapshot:
            sf_shi = self.gal_cantorID[snap.sh_galaxy]
            sh_sf = snap.shi[self.sh_galID] 

        # ------------------------------
        # i) Particle ID and radius list
        # ------------------------------

        yb.write_hdf5(particles.ids, self.outloc, 
                      snapPre + 'IDs', 
                      comment = "IDs of all particles associated with a "
                      "FOF group or subhalo (depending on settings). "
                      "Particles are sorted by FOF, then subhalo, then "
                      "type, and then radial distance from the subhalo "
                      "centre.")

        # If desired, write binding energies:
        if par['Output']['WriteBindingEnergy']:
            yb.write_hdf5(particles.binding_energy,
                          self.outloc, snapPre + 'BindingEnergy', 
                          comment = "Specific binding energy of all "
                          "particles in the "
                          "ID list. For particles not associated with a "
                          "subhalo, this is NaN, for others it is the "
                          "difference between KE and PE relative to their "
                          "subhalo (always negative). Note that it is also "
                          "NaN for centrals if central unbinding was "
                          "disabled. Units: km^2/s^2.")

        # If desired, write particle radii:
        if par['Output']['WriteRadii']:
            yb.write_hdf5(particles.rad, self.outloc,
                          snapPre + 'Radius',
                          comment = "Radial distance of each particle "
                          "from the centre-of-potential of its subhalo "
                          "(units: pMpc). Note that for subhaloes whose "
                          "centre-of-potential was taken from Subfind, "
                          "rounding errors may imply that no particle is "
                          "at a radius of exactly zero.")
        
        # --------------------------------------------------
        # ii) FOF offset/length list, and FOF --> SH indices
        # --------------------------------------------------
        
        if self.full_snapshot:
            yb.write_hdf5(self.fof_offset, self.outloc, snapPre+'FOF/Offset', 
                          comment = "Offset of FOF *index* i in ID list. "
                          "The particles belonging to this FOF are stored in "
                          "indices [offset]:[offset]+[length]. "
                          "Note that this includes a 'coda', so it has one "
                          "more elements than the number of FOFs, "
                          "and IDs may also be retrieved as "
                          "[offset[i]]:[offset[i+1]].")
            yb.write_hdf5(self.fof_length, self.outloc, snapPre+'FOF/Length', 
                          comment = 
                          'Number of particles in ID list that belong '
                          'to FOF *index* i. The particles belonging to this '
                          'FOF are stored in indices '
                          '[offset]:[offset]+[length]. '
                          'Note that subhaloes of each FOF are stored in '
                          'descending mass order, except for the central '
                          'subhalo, which always comes first.')

            # Write FOF --> subhalo links:
            yb.write_hdf5(self.fof_nsh, self.outloc, 
                          snapPre + 'FOF/NumOfSubhaloes',
                          comment = 'Number of subhaloes belonging to this '
                          'FOF.')
            yb.write_hdf5(self.fof_fsh, self.outloc, snapPre + 
                          'FOF/FirstSubhalo',
                          comment = 'Index of first subhalo belonging to this '
                          'FOF. Note that this will be >= 0 even if there is '
                          'not a single subhalo in the FOF group, to keep the '
                          'list monotonic. See CenSubhalo for a safe pointer '
                          'to the central subhalo, if it exists.')
            yb.write_hdf5(self.fof_cenSH, self.outloc, snapPre + 
                          'FOF/CenSubhalo', 
                          comment = 'Index of first subhalo belonging to this '
                          'FOF. -1 if the FOF has not a single subhalo.')
        
        # -----------------------------
        # iii) Subhalo index properties
        # -----------------------------

        grp = snapPre + 'Subhalo/'

        # Subhalo --> galaxy index
        yb.write_hdf5(self.sh_galID, self.outloc, grp + 'Galaxy', 
                      comment = "Galaxy ID for each subhalo index.")

        
        if self.full_snapshot:
            # Subhalo --> FOF
            yb.write_hdf5(self.sh_fof, self.outloc, grp + 'FOF_Index', 
                          comment = "Index of FOF group that this subhalo "
                          "belongs to.")
        
            # Cantor <--> Subfind subhalo cross indices
            yb.write_hdf5(sh_sf, self.outloc, 
                          grp + 'SubfindIndex', 
                          comment = "Index of the corresponding subhalo in "
                          "the Subfind catalogue. If this is < 0, it means "
                          "that the subhalo was lost by Subfind but recovered "
                          "by Cantor.")
            yb.write_hdf5(sf_shi, self.outloc, 
                          grp + 'IndexBySubfindID', 
                          comment = "Reverse index to identify the subhalo "
                          "corresponding to a given Subfind subhalo. To find "
                          "the (Cantor) subhalo corresponding to "
                          "Subfind subhalo index i, look at Cantor index "
                          "IndexBySubfindID[i].")

        # Subhalo --> particle indices
        yb.write_hdf5(self.sh_offset, self.outloc, 
                      grp + 'Offset', 
                      comment = "First index in ID list belonging to "
                      "subhalo i. The particles belonging to this subhalo "
                      "are stored in indices [offset]:[offset]+[length].")
        yb.write_hdf5(self.sh_length, self.outloc, 
                      grp + 'Length', 
                      comment = "Number of particles in ID list belonging "
                      "to subhalo i. The particles belonging to this subhalo "
                      "are stored in indices [offset]:[offset]+[length].")
        yb.write_hdf5(self.sh_offsetType, self.outloc, 
                      grp + 'OffsetType', 
                      comment = "First index in ID list belonging to "
                      "subhalo i (first index) and type j (second index). "
                      "The particles of this type belonging to this subhalo "
                      "are stored in indices [offset_i]:[offset_i+1]. "
                      "The last element (j = 6) serves as a coda for this.")
        yb.write_hdf5(self.sh_lengthType, self.outloc, 
                      grp + 'LengthType', 
                      comment = "Number of particles in ID list that belong "
                      "to subhalo i (first index) and have type j (second "
                      "index). These particles are stored at indices "
                      "[offset_i]:[offset_i+1].")
        
        if par['Output']['GalaxyProperties']:
            self.write_galaxy_properties(grp)


        self.timeStamp.set_time('Write Subhalo data')
        print("done (took {:.3f} sec.)." 
              .format(self.timeStamp.get_time()))
                
    def write_galaxy_properties(self, grp):
        """
        Write out galaxy properties, beyond segmentation map info.

        Parameters:
        -----------
        grp : str
            The HDF5 group to write data to.
        """

        # Maximum radius of subhalo particles
        yb.write_hdf5(self.sh_rMax, self.outloc, grp + 'MaxRadius',
                      comment = "Distance of furthest particle from subhalo "
                      "centre of potential (units: pMpc).")
        yb.write_hdf5(self.sh_rMaxType, self.outloc, grp + 'MaxRadiusType',
                      comment = "Distance of furthest particle of "
                      "a given type from subhalo centre of potential "
                      "(units: pMpc).")
        

        # ----------------------------------------------
        # iv) Main physical subhalo properties (for all)
        # ----------------------------------------------
        
        # Mass: total, and by type
        yb.write_hdf5(self.sh_mass, self.outloc, grp + 'Mass',
                      comment = "Total mass of each subhalo "
                      "(units: 10^10 M_Sun).")

        yb.write_hdf5(self.sh_massTypeAp[:, :, 4], self.outloc, 
                      grp + 'MassType', 
                      comment = "Mass per particle type of each subhalo "
                      "(units: 10^10 M_Sun).")

        # Subhalo coordinates
        yb.write_hdf5(self.sh_centreOfPotential, self.outloc,
                      grp + 'CentreOfPotential', 
                      comment = "Coordinates of particle with the lowest "
                      "gravitational potential (units: pMpc). For "
                      "galaxies that were not unbound directly "
                      "(i.e. centrals, if their unbinding was disabled), "
                      "the value is taken from the Subfind catalogue.")
        yb.write_hdf5(self.sh_monkPos, self.outloc,
                      grp + 'Position',
                      comment = "Coordinates of subhalo in final "
                      "unbinding iteration. Units: pMpc. NaN if particles "
                      "from this subhalo were not processed "
                      "(in particular possible for centrals).")
        yb.write_hdf5(self.sh_monkVel, self.outloc,
                      grp + 'Velocity',
                      comment = "Velocity of subhalo in final "
                      "unbinding iteration. Units: pMpc. NaN if particles "
                      "from this subhalo were not processed "
                      "(in particular possible for centrals).")
        yb.write_hdf5(self.sh_comPos, self.outloc,
                      grp + 'CentreOfMass', 
                      comment = "Coordinates of subhalo centre of mass "
                      "(units: pMpc).")
        yb.write_hdf5(self.sh_zmfVel, self.outloc,
                      grp + 'ZMF_Velocity', 
                      comment = "Velocity of the subhalo's zero-momentum "
                      "frame (i.e. mass-weighted velocity of all its "
                      "particles). Units: km/s.")

        # DM and stellar (total) velocity dispersion (index 2/5 => @infty)
        yb.write_hdf5(self.sh_velDisp[:, 2], self.outloc,
                      grp + 'VelocityDispersion_DM', 
                      comment = "Dark matter velocity dispersion of the "
                      "subhalo (units: km/s). A value of NaN indicates "
                      "that the subhalo has no DM, and 0 "
                      "(typically) means that it only has one DM "
                      "particle.")
        yb.write_hdf5(self.sh_velDisp[:, 5], self.outloc,
                      grp + 'VelocityDispersion_Stars', 
                      comment = "Stellar velocity dispersion of the "
                      "subhalo (units: km/s). A value of NaN indicates "
                      "that the subhalo has no stars, and 0 "
                      "(typically) means that it only has one stellar "
                      "particle.")
        
        # Gas, DM, stellar (total) angular momentum (index 2/5/8 => @infty)
        yb.write_hdf5(self.sh_angMom[:, 2, :], self.outloc,
                      grp + 'AngularMomentum_Gas', 
                      comment = "Angular momentum vector of gas particles "
                      "in the subhalo (units: 10^10 M_sun * pMpc * km/s). "
                      "The angular momentum is computed relative to the "
                      "subhalo centre of potential and the (total) gas "
                      "ZMF velocity.")
        yb.write_hdf5(self.sh_angMom[:, 5, :], self.outloc,
                      grp + 'AngularMomentum_DM', 
                      comment = "Angular momentum vector of DM particles "
                      "in the subhalo (units: 10^10 M_sun * pMpc * km/s). "
                      "The angular momentum is computed relative to the "
                      "subhalo centre of potential and the (total) DM "
                      "ZMF velocity.")
        yb.write_hdf5(self.sh_angMom[:, 8, :], self.outloc,
                      grp + 'AngularMomentum_Stars', 
                      comment = "Angular momentum vector of star particles "
                      "in the subhalo (units: 10^10 M_sun * pMpc * km/s). "
                      "The angular momentum is computed relative to the "
                      "subhalo centre of potential and the (total) stellar "
                      "ZMF velocity.")

        # Maximum circular velocity and its radius
        yb.write_hdf5(self.sh_vmax, self.outloc,
                      grp + 'Vmax', 
                      comment = "Maximum circular velocity of the "
                      "subhalo, calculated as max(sqrt(GM(<r)/r)). "
                      "Units: km/s.")
        yb.write_hdf5(self.sh_radOfVmax, self.outloc,
                      grp + 'RadiusOfVmax', 
                      comment = "Radius at which the circular velocity of "
                      "the subhalo is maximum, calculated as "
                      "argmax(sqrt(GM(<r)/r)). Units: pMpc.")

        # For clarity, 'extra' output (for massive galaxies) is outsourced:
        self.write_extra_output(grp)

                     
    def write_extra_output(self, snapPre):         #Class: SnapshotOutput
        """
        Write out extra physical properties of massive subhaloes.
            [Helper function of write()]

        These are all those that either have log_10 M_star/M_sun >= 8.5,
        or log_10 M_tot/M_sun >= 10.5. The rationale is that this excludes
        a gzillion tiny objects for which these values are pretty much
        meaningless anyway due to resolution limits.
        
        Parameters:
        -----------
        snapPre : string
            The snapshot-specific HDF5 group to which output is written.
            The extra output is written to [snapPre]/Extra/.
        """

        # Set up HDF5 directory to write to:
        grp = snapPre + 'Extra/'
        
        # --- Find galaxies worth the effort: ---------------
        # --- M_star or M_tot above specified threshold -----

        msub_min = par['Output']['Extra']['Mtot']
        mstar_min = par['Output']['Extra']['Mstar']
        if msub_min is None: 
            msub_min = 0
        if mstar_min is None:
            mstar_min = 0

        shi_extra = np.nonzero(
            (self.sh_mass >= 10.0**(msub_min-10)) |
            (self.sh_massTypeAp[:, 4, -1] >= 10.0**(mstar_min-10)))[0]

        # Also create reverse list, to find extraID from Subhalo ID
        extra_ids = yb.create_reverse_list(shi_extra, maxval = self.nSH-1)

        n_extra = len(shi_extra)
        print("Out of {:d} galaxies, {:d} are massive enough for extra "
              "output (={:.2f}%)." 
              .format(self.nSH, n_extra, n_extra/self.nSH*100))
        
        # ------------------------------------------
        # i) Indexing information extra <--> subhalo
        # ------------------------------------------

        yb.write_hdf5_attribute(self.outloc, grp, 'NumExtra', n_extra)

        yb.write_hdf5(shi_extra,
                      self.outloc, grp + 'SubhaloIndex',
                      comment = "Index into the `main' subhalo catalogue "
                      "for subhalo with extraID i (i.e. whose properties are "
                      "stored in index i in this extended catalogue).")
        yb.write_hdf5(extra_ids,
                      self.outloc, grp + 'ExtraIDs',
                      comment = "Index into this extended catalogue by "
                      "`main' subhalo index. Subhaloes that are not massive "
                      "enough to be included in the extended catalogue "
                      "have a value of -1.")

        # --------------------------------
        # ii) Particle indices by aperture
        # --------------------------------
        
        yb.write_hdf5(self.sh_offsetTypeAp[shi_extra, :, :],
                      self.outloc, grp + 'OffsetTypeApertures',
                      comment = "Index of first particle of subhalo i "
                      "(first index) and type j (second index) that is "
                      "more than 3/10/30/100 pkpc (third index) from "
                      "the centre of potential of its subhalo. To find "
                      "all particles of this type that are within one "
                      "of these apertures, load ids[offsetType:"
                      "offsetTypeApertures]. Note that this array has "
                      "no coda, so to load particles up to the outermost "
                      "one, the final particle index must be retrieved "
                      "from /Subhalo/OffsetType.")
        
        # --------------------------------------------------
        # iii) Properties that are the same for gas/DM/stars
        # --------------------------------------------------
        
        for ptype in [0, 1, 4, 5]:
            self.write_extra_type_properties(grp, ptype, shi_extra)

        # ----------------------------
        # iv) Star-specific properties
        # ----------------------------
        
        # Stellar mass percentiles:
        yb.write_hdf5(self.sh_smr[shi_extra, :, :], self.outloc,
                      grp + 'Stars/QuantileRadii', 
                      comment = "Radii containing {20, 50, 80} per cent "
                      "(3rd index) of the stellar mass within "
                      "{30 pkpc, infty} (2nd index) from the subhalo centre "
                      "(units: pMpc). "
                      "The radii are interpolated between that "
                      "of the outermost particle enclosing less than "
                      "the target mass and the one "
                      "immediately beyond it. If there is only one "
                      "star particle, the result is taken "
                      "as half its radius.")
 
        # Kinematic morphology parameter:
        yb.write_hdf5(self.sh_kappaCo[shi_extra], self.outloc,
                      grp + 'Stars/KappaCo',
                      comment = "Stellar kinematic morphology parameter "
                      "as in Correa+17 (based on Sales+10); defined as "
                      "K_rot/K_tot with "
                      "K_rot = sum(1/2 * m_i * v'_i**2) and "
                      "K_tot = sum(1/2 * m_i * v_i**2), where "
                      "v'_i = L_z_i / (m_i * R_i), with L_z_i the (ith "
                      "particle's) component of angular momentum along the "
                      "total angular momentum axis and "
                      "R_i its perpendicular distance from the axis. " 
                      "Only particles with positive-definite L_z are "
                      "considered. Particles are selected within "
                      "{10, 30} pkpc (2nd index), and both the reference "
                      "velocity and angular momentum are computed over "
                      "all (star) particles in the same aperture. The "
                      "reference position is the subhalo "
                      "centre of potential. The corresponding "
                      "angular momentum axes are stored in "
                      "'Extra/AngularMomentum'.")

                                                  # Class: SnapshotOutput
    def write_extra_type_properties(self, grp, ptype, shi_extra):
        """
        Write extra properties that are the same for several types.
           [Helper function of write_extra_output()]

        Parameters:
        -----------
        grp : string
            The HDF5 group to which the extra output is written 
            (within current snapshot group).
        ptype : int
            The particle type code for which to write output
            (0=gas, 1=DM, 4=stars, 5=BHs)
        shi_extra : ndarray (int)
            The indices of subhaloes that are massive enough to warrant
            writing the extended output.
        """ 

        typeNames = ['Gas', 'DM', '', '', 'Stars', 'BHs']
        typeNamesLCS = ['gas', 'DM', '', '', 'star', 'BH']
        typeNamesLCP = ['gas', 'DM', '', '', 'stars', 'BHs']

        # Offsets into the ptype+aperture output arrays 
        zmfOff = np.array([0, 3, 6, 6, 6, 9, 9])  # ZMF-vel & ang.mom.
        axOff = np.array([0, 0, 3, 3, 3, 6, 6])   # Vel. disp. & MIT

        grp_type = grp + typeNames[ptype] + '/'
        tns = typeNamesLCS[ptype]
        tnp = typeNamesLCP[ptype]

        # Mass by aperture:
        yb.write_hdf5(self.sh_massTypeAp[shi_extra, ptype, :4], self.outloc,
                      grp_type + 'ApertureMasses', 
                      comment = "Sum of " + tns + " particles masses within "
                      "{3, 10, 30, 100} pkpc from the subhalo centre of "
                      "potential. Units: 10^10 M_Sun.")

        # Centre of mass:
        yb.write_hdf5(self.sh_comPosType[shi_extra, ptype, :], self.outloc,
                      grp_type + 'CentreOfMass', 
                      comment = "Centre of mass of " + tns + " particles " 
                      "in this subhalo (units: pMpc).")

        if ptype == 5: return

        # ZMF velocity (only write 30pkpc -- offset+1)
        yb.write_hdf5(self.sh_zmfVelType[shi_extra, zmfOff[ptype]+1, :], 
                      self.outloc, grp_type + 'ZMF_Velocity_30kpc', 
                      comment = "Zero-momentum-frame velocity of " + tns + 
                      "particles within 30 pkpc from the subhalo centre "
                      "of potential (units: km/s).")

        # Angular momentum vector:
        # (note that infty is already written as standard output)
        yb.write_hdf5(
            self.sh_angMom[shi_extra, zmfOff[ptype]:zmfOff[ptype+1]-1, :], 
            self.outloc, grp_type + 'AngularMomentum', 
            comment = "Angular momentum vector of " + tns + " particles "
            "within {10, 30} pkpc from the subhalo centre "
            "of potential (units: 10^10 M_sun * pMpc * km/s). "
            "The angular momentum is computed relative to the "
            "subhalo centre of potential and the respective "
            "particles' ZMF velocity.")
        
        if (ptype == 1 or ptype == 4):
            # Moment-of-inertia tensor axes and ratios:
            yb.write_hdf5(
                self.sh_axes[shi_extra, axOff[ptype]:axOff[ptype+1], :, :], 
                self.outloc, grp_type + 'Axes', 
                comment = "Principal axes of the " + tns + 
                " moment-of-inertia tensor, within {10, 30, infty} pkpc "
                "(2nd index) from the subhalo centre of potential. "
                "The 3rd index specifies the minor (0), intermediate (1), " 
                "and major (2) axis; the 4th index specifies the x/y/z "
                "component of the (unit) vector along the respective axis."
                "A value of NaN indicates that there are no particles "
                "within the respective aperture. The corresponding axis "
                "ratios are stored in 'Extra/AxisRatios'.")
            
            yb.write_hdf5(
                self.sh_axRat[shi_extra, axOff[ptype]:axOff[ptype+1], :],
                self.outloc, grp_type + 'AxisRatios', 
                comment = "Ratio of the minor and intermediate "
                "axis to the major axis, respectively (0 and 1 along 3rd "
                "index) of the " + tns + " moment-of-inertia tensor "
                "within {10, 30, infty} pkpc (2nd index)."
                "A value of NaN indicates that there are no particles "
                "within the respective aperture. The corresponding axis "
                "(unit) vectors are stored in 'Extra/Axes'.");

            # Velocity dispersions
            # (infty value already written as main output)
            yb.write_hdf5(
                self.sh_velDisp[shi_extra, axOff[ptype]:axOff[ptype+1]-1], 
                self.outloc, grp_type + 'VelocityDispersion', 
                comment = "Velocity dispersion of " + tns + " particles "
                "within {10, 30} pkpc from the subhalo centre "
                "of potential (units: km/s). The reference velocity is "
                "taken as the ZMF velocity of " + tns + " particles within "
                "the same aperture, and particles are weighted by mass.")


class Snapshot:
    """Class for general information/data about an individual snapshot."""
    
    def __init__(self, sim, isnap, is_target_snap=False):
        """
        Constructor for the class.
        
        Parameters
        ----------
        sim : Simulation class instance
            The simulation to which this output belongs.
        isnap : int
            The index of the snapshot.
        is_target_snap : bool, optional
            If True, indicates that this is the target snapshot.
            Else (default), it is assumed that it points to a prior
            snapshot, that has already been processed by Cantor.
        """

        self.sim = sim
        self.isnap = isnap
        self.timeStamp = TimeStamp()
        self.is_target_snap = is_target_snap

        self.subdir, self.espdir, self.snapdir = st.form_files(
            self.sim.rundir, isnap, 'sub subpart snap')
    
        self.zred = st.snap_age(self.snapdir, type = 'zred')
        self.hubble_z = Planck13.H(self.zred)#.value

        # Determine Plummer-equivalent softening length
        self.epsilon = min(1/(1+self.zred)*2.66*1e-3, 7e-4)

        print("Determined H(z) = {:.2f}, epsilon(z) = {:.2f} kpc." 
              .format(self.hubble_z, self.epsilon*1e3))
        if not self.hubble_z.unit == 'km / (s Mpc)':
            print("Hubble constant seems in non-standard units...")
            set_trace()

        # Artificially suppress Hubble flow if desired:
        if par['Check']['NoHubble']:
            self.hubble_z = 0*self.hubble_z.unit
            print("Artificially set H(z) = {:.2f}."
                  .format(self.hubble_z))

        self.ids = None   # Initialize as 'not yet loaded'.
        self.sf_sh_is_loaded = False  # Same as above.

    def set_particles(self, particles):                 # Class: Snapshot
        """Set a reference to the snapshot's particle structure."""
        self.particles = particles

    def set_priorCantorData(self, priorCantorData):     # Class: Snapshot
        """Set a reference to the snapshot's prior cantor data structure."""
        self.priorCantorData = priorCantorData

    def set_galaxies(self, galaxies):                   # Class: Snapshot
        """Set a reference to the snapshot's (target) galaxies."""
        self.target_galaxies = galaxies

    def load_catalogue_data(self):                      # Class: Snapshot
        """Load snapshot info from high-level catalogues."""

        self.shi = self.sim.shi[:, self.isnap]
        self.nSH = np.count_nonzero(self.shi >= 0)
        self.cenGal = self.sim.cenGal[:, self.isnap]
        self.sh_galaxy = yb.read_hdf5(
            self.sim.spiderloc, 'Subhalo/Snapshot_{:03d}/Galaxy' 
            .format(self.isnap))

        if par['Input']['FromCantor'] and not self.is_target_snap:
            print("Setting up SHI as Cantor...")
            self.shiLoad = self.sim.output.shiExtended[:, self.isnap]
        else:
            self.shiLoad = self.shi
        
    def load_sh_data_from_snapshot(self):               # Class: Snapshot
        """
        Load data for assigning SH/FOF membership to snapshot particles.
        """

        subdir = self.subdir   # For convenience

        # Load SF ID list
        self.ids = st.eagleread(subdir, 'IDs/ParticleID', astro = False)

        # Load appropriate sub-divisions, depending on whether we want to
        # load all particles in FOFs, or only those in SF-subhaloes
        if par['Input']['LoadFullFOF']:
            self.off = st.eagleread(subdir, 'FOF/GroupOffset', 
                                       astro = False)
            self.len = st.eagleread(subdir, 'FOF/GroupLength', 
                                       astro = False)
            self.fof_fsh = st.eagleread(subdir, 'FOF/FirstSubhaloID', 
                                        astro = False)
            self.fof_nsh = st.eagleread(subdir, 'FOF/NumOfSubhalos', 
                                        astro = False)
            
            self.off_sh = st.eagleread(
                subdir, 'Subhalo/SubOffset', astro = False)
            self.len_sh = st.eagleread(
                subdir, 'Subhalo/SubLength', astro = False)

        else:
            self.load_sf_sh_particle_data()
            
            # For consistency with FOF-section, need to assign separate
            # variables to the same objects for SH lookup:
            self.off_sh = self.off
            self.len_sh = self.len
                                                        # Class: Snapshot
    def load_sf_sh_particle_data(self, includeFOF=False):  
        """Load Subfind data to associate particles to subhaloes."""

        subdir = self.subdir
        if self.sf_sh_is_loaded: return

        self.off = st.eagleread(subdir, 'Subhalo/SubOffset', astro = False)
        self.len = st.eagleread(subdir, 'Subhalo/SubLength', astro = False)
        if self.ids is None:
            self.ids = st.eagleread(subdir, 'IDs/ParticleID', astro = False)

        if includeFOF:
            self.off_fof = st.eagleread(subdir, 'FOF/GroupOffset', 
                                        astro = False)
            self.len_fof = st.eagleread(subdir, 'FOF/GroupLength',
                                        astro = False)
        self.sf_sh_is_loaded = True

        # For consistency with FOF-section, need to assign separate
        # variables to the same objects for SH lookup:
        self.off_sh = self.off
        self.len_sh = self.len
                                                 # Class: Snapshot
    def load_cantor_sh_particle_data(self, cantorOutput, includeFOF=False):
        """Load Cantor data to associate particles to subhaloes."""
        
        outloc = cantorOutput.outloc
        snapPre = 'Snapshot_{:03d}/' .format(self.isnap)

        self.ids = yb.read_hdf5(outloc, snapPre + 'IDs')
        self.off_sh = yb.read_hdf5(outloc, snapPre + 'Subhalo/Offset')
        self.len_sh = yb.read_hdf5(outloc, snapPre + 'Subhalo/Length')   

        if includeFOF:
            self.off_fof = yb.read_hdf5(outloc, snapPre + 'FOF/Offset')
            self.len_fof = yb.read_hdf5(outloc, snapPre + 'FOF/Length')

    def load_sh_data_from_esp(self):                    # Class: Snapshot
        """Load data for converting ESP information to subhalo indices."""

        subdir = self.subdir    # For convenience

        # Don't need to load ID list + divisions, but must set up conversion
        # from GroupNumber + SubGroupNumber --> SHI
        self.fof_fsh = st.eagleread(
            subdir, 'FOF/FirstSubhaloID', astro = False)

        if par['Input']['LoadFullFOF']:
            self.fof_nsh = st.eagleread(
                subdir, 'FOF/NumOfSubhalos', astro = False)

        # If we want to explicicly load SF-associated particles, we need
        # the particle IDs/offsets/lengths also with ESP:
        if par['Sources']['Subfind'] or par['Sources']['RefSnap']:
            self.load_sf_sh_particle_data()

    def load_sh_coordinates(self):                      # Class: Snapshot
        """
        Load the phase-space coordinates of all subhaloes.

        This is also includes the (approximate) coordinates for 
        SF-dead galaxies.
        
        Velocities can optionally be read from the SnipLocate output,
        which may be more accurate than the raw Subfind velocities.
        (controlled by par['Galaxies']['VelFromTracers'] flag).
        """
        
        self.galPos = self.sim.galPos[:, self.isnap, :]
        
        if par['Galaxies']['VelFromTracers']:
            snepInd = yb.read_hdf5(
                self.sim.pathloc, 'SnapshotIndex')[self.isnap]
            self.galVel = yb.read_hdf5(
                self.sim.pathloc, 'Snepshot_' + 
                str(snepInd).zfill(4) + '/Velocity')
        else:
            self.galVel = self.sim.galVel[:, self.isnap, :]            

                                                        # Class: Snapshot
    def shi_to_galaxy(self, shi, return_central = False):
        """
        Convert SHI array into (optionally: central) galaxyID.
        
        Parameters:
        -----------
        shi : ndarray (int)
            The subhalo indices (in current snapshot) to look up. Negative
            values are ignored, yielding a 'placeholder' galaxyID of -1.
        return_central: bool, optional
            If True, the central galaxy of the subhalo's FOF group is found,
            via the internally-loaded cenGal array. Default is False.
        """
        
        galaxy = np.zeros_like(shi) - 1   # -1 is default value (non-existing)
        ind_in = np.nonzero(shi >= 0)[0]
        galaxy[ind_in] = self.sh_galaxy[shi[ind_in]]

        if return_central:
            # Convert (existing) galaxy IDs to that of their central:
            # (with try statement to ensure cenGal has actually been loaded):
            try:
                galaxy[ind_in] = self.cenGal[galaxy[ind_in]]
            except NameError:
                print("Looks like cenGal table is not loaded for snapshot "
                      "{:d}... -- investigate." .format(self.isnap))
                set_trace()

        return galaxy
                                                        # Class: Snapshot
    def load_fof_index(self, cantorOutput, from_cantor=False):
        """
        Load FOF index for all subhaloes.

        If from_cantor is False (default), load from SF, otherwise from 
        cantor output itself.
        """
        if from_cantor:
            outloc = cantorOutput.outloc
            snapPre = 'Snapshot_{:03d}/' .format(self.isnap)
            self.fof_index = yb.read_hdf5(
                outloc, snapPre + 'Subhalo/FOF_Index')
        else:
            self.fof_index = st.eagleread(
                self.subdir, 'Subhalo/GroupNumber', astro = False) - 1

            self.nFOF = yb.read_hdf5_attribute(self.subdir, 
                                               'Header', 'TotNgroups') 

                                                        # Class: Snapshot
    def find_alive_galaxies(self, cantorOutput, from_cantor=False):        
        """
        Find all galaxies that are alive in this snapshot.
        
        If from_cantor is False (default), load from SF, otherwise from
        cantor output itself.
        """
        if from_cantor:
            if self.isnap > cantorOutput.lastSnap:
                print("Oops! We haven't processed snap {:d} yet..."
                      .format(self.isnap))
                set_trace()

            shi = cantorOutput.shiExtended
            self.ind_alive = np.nonzero(shi[:, self.isnap] >= 0)[0]
        else:
            self.ind_alive = np.nonzero(self.shi  >= 0)[0]
        
    def build_satellite_list(self, cantorOutput):       # Class: Snapshot
        """Set up a way to quickly find a galaxy's satellites."""

        # If we include lost galaxies, we have to load the 'extended'
        # central galaxy table from the output (to include SF-dead ones):

        if par['Lost']['Recover']:
            cenGalsSnap = (
                cantorOutput.cenGalExtended[self.ind_alive, self.isnap])
        else:
            cenGalsSnap = self.cenGal[self.ind_alive]

        # Set up a look-up table to find galaxies by their central:
        self.lut_cenGal = SplitList(cenGalsSnap, self.sim.gal_lims)    

    def build_merger_list(self, target_snap):           # Class: Snapshot
        """
        Set up a way to quickly find a galaxy's (alive) mergees:

        All those galaxies that are alive in this (=ref) snapshot, and 
        will merge with a given galaxy by target_snap.
        """
        
        # Simple case first: all galaxies alive in target snap:
        if not par['Lost']['FindPermanentlyLost']:
            # Merger targets of all (now) alive galaxies in target_snap:
            mergeTarg = self.sim.mergeList[self.ind_alive, target_snap.isnap] 
            
            # Set up a look-up table to find galaxies by their 
            # merger target:
            self.lut_mergeTarg = SplitList(mergeTarg, self.sim.gal_lims)
            return

        # If we unbind permanently lost galaxies, then this needs
        # to explicitly find the last-alive snapshot of all...
        self.lut_mergeTarg = []
        for isnap in range(target_snap.isnap+1):  #par['Snaps']['End']):
            if isnap in target_snap.target_galaxies.last_sf_alive:
                # Merger targets of all (now) alive galaxies in isnap:
                mergeTarg = self.sim.mergeList[self.ind_alive, isnap] 
                
                # Set up a look-up table to find galaxies by their 
                # merger target:
                self.lut_mergeTarg.append(
                    SplitList(mergeTarg, self.sim.gal_lims))
            else:
                self.lut_mergeTarg.append([])

    def get_gal_ids(self, igal):                        # Class: Snapshot
        """Get the particle IDs for one specific galaxy."""

        shi = self.shi[igal]
        offset = self.off_sh[shi]
        length = self.len_sh[shi]
        return (self.ids[offset : offset+length]).astype(int)


class TimeStamp:
    """Class for recording the runtime of program sections."""
    
    def __init__(self, verbose=False):
        """Constructor, set up (empty) list and initialize starting time."""
        self.timeList = []
        self.markList = []
        self.prevTime = time.time()

        self.otherTime = []   # Delta_t from other
        self.otherMark = []   # Mark from other
        self.otherInd = []    # Internal index of which times are part
        self.verbose = verbose

    def set_time(self, mark):                          # Class: TimeStamp
        """
        Save a 'timestamp' at a milestone in processing this snapshot.
        
        What is stored internally is the time elapsed since the last
        'stamping' call. The mark is stored in a separate list.
        
        Parameters:
        -----------
        mark : string
            A text string describing the block that has ended at the time of
            calling this function (e.g. 'Reading in particles').
        """
        self.timeList.append(time.time() - self.prevTime)
        self.markList.append(mark)
        self.prevTime = time.time()
        
    def get_time(self, mark=None):                      # Class: TimeStamp
        """
        Retrieve the length of a previously recorded time-interval.

        Parameters:
        -----------
        mark : string, optional
            The string description that was saved with the timestamp. The
            time interval of this mark is returned. If none is specified, 
            the last interval is returned.
                    
            If there are multiple instances of 'mark', the first is retrieved.
            If 'mark' does not match any of the recorded interval labels, 
            the program aborts into pdb. 

        Returns:
        --------
        delta_t : float
            The time, in seconds, of the desired time interval.
        """
        
        if not self.timeList:
            print("No time events have been recorded!")
            set_trace()

        if mark is None:
            return self.timeList[-1]
        else:
            ind_mark = np.nonzero(self.markList == mark)[0]
            if len(ind_mark) == 0:
                print("The mark '" + mark + "' was not found in the time "
                      "list. Please investigate.")
                set_trace()
            return self.timeList[ind_mark[0]]

    def add_counters(self, marks):                     # Class: TimeStamp
        """
        Add a number of (empty) time counters to the lists.

        These counters can be filled later with increase_time().

        Parameters:
        -----------
        marks : list of str
            The marks of the newly created counters.

        Returns:
        --------
        indices : list of int
            The indices of the newly created counters.
        """
        
        indices = []

        for imark in marks:
            self.timeList.append(0)
            self.markList.append(imark)
            indices.append(len(self.timeList)-1)

        return indices

    def start_time(self):                             # Class: TimeStamp
        """Re-set the internal clock, i.e. start a new time interval."""
        self.prevTime = time.time()

    def copy_times(self, other):                      # Class: TimeStamp
        """
        Copy timings from other structure, increasing existing counters.
        
        Note that, if there are multiple instances of a given mark, only
        the first one will be incremented.
  
        This function differs from import_times in that it does not create
        a new sub-level of time-counters. Also (like import_times), it 
        does not save sub-counters of the other structure.
        """

        if not other.timeList:
            if self.verbose:
                print("No time events have been recorded in other list.")
            return

        # Import time marks one-by-one:
        for iitime in range(len(other.timeList)):
            itime = other.timeList[iitime]
            imark = other.markList[iitime]
            
            # Search for already-started counters:
            ind = np.nonzero(np.array(self.markList) == imark)[0]
            if len(ind) == 0:
                self.timeList.append(itime)
                self.markList.append(imark)
            else:
                # If counter already exists, just add to it:
                self.timeList[ind[0]] += itime
        
    def import_times(self, other):                    # Class: TimeStamp
        """
        Increment internal counters with (partial) timings from other list.

        This is used to transfer timings from processing ref snaps to the 
        target snap. A separate time list is created and 'linked' 
        to the currently running time-keeping interval in this list.

        Any 'other' (sub-)lists already saved in the other list are not
        imported. If that's a problem, change the code.

        Parameters:
        -----------
        other : TimeList instance
            The list from which time stamps should be imported.
        """

        if not other.timeList:
            if self.verbose:
                print("No time events have been recorded in other list.")
            return


        # Determine internal index to which imported times are `attached':
        currIndex = len(self.timeList)

        # Import time marks one-by-one:
        for iitime in range(len(other.timeList)):
            itime = other.timeList[iitime]
            imark = other.markList[iitime]
            
            # Search for already-started counters:
        
            makeNewEntry = True
            if len(self.otherMark) > 0:
                ind = np.nonzero((np.array(self.otherMark) == imark) & 
                                 (np.array(self.otherInd) == currIndex))[0]
                if len(ind) > 0:
                    # If counter already exists, just add to it:
                    self.otherTime[ind[0]] += itime
                    makeNewEntry = False
            if makeNewEntry:
                self.otherTime.append(itime)
                self.otherMark.append(imark)
                self.otherInd.append(currIndex)

                                                       # Class: TimeStamp
    def increase_time(self, mark=None, index=None):
        """Increase an existing time counter."""
    
        if index is not None:
            self.timeList[index] += (time.time()-self.prevTime)
        elif mark is not None:
            index = np.nonzero(self.markList == mark)[0][0]
            self.timeList[index] += (time.time()-self.prevTime)
        else:
            print("Cannot increase time without knowing where!")
            set_trace()

        self.prevTime = time.time()

                                                        # Class: TimeStamp
    def print_time_usage(self, caption=None, mode='detailed',
                         minutes=False, percent=True):          
        """
        Print a report on the internally stored times.

        Parameters:
        -----------
        caption : str, optional
            A string to display at the beginning of the report. If None,
            no caption string is printed.

        mode : str, optional
            Defines how output should be structured. Options are:
            -- 'detailed' (default):
               All sub-counters are printed separately for their
               respective top-level counter.
            -- 'top':
               Only top-level counters are printed.
            -- 'sub':
               All sub-counters with the same mark are combined and
               printed. No top-level information is printed.
        """

        if mode not in ['detailed', 'top', 'sub']:
            print("Wrong mode option '" + mode + "' for "
                  "print_time_usage(). Please investigate.")
            set_trace()
        
        if caption is None:
            caption = "Finished "

        self.minutes = minutes
        self.percent = percent

        self.fullTime = np.sum(np.array(self.timeList))

        print("")
        print("-" * 70)
        print(caption + " ({:.2f} min.)"
              .format(self.fullTime/60))
        print("-" * 70)

        if mode in ['detailed', 'top']:
            markLength = len(max(self.markList, key=len))+3
            if self.otherMark:
                otherMarkLength = len(max(self.otherMark, key=len))+3+5
                if otherMarkLength < markLength+3:
                    otherMarkLength = markLength+3

            for iimark, imark in enumerate(self.markList):
                print((imark+':').ljust(markLength) + 
                      self._tstr(self.timeList[iimark]))
                
                if mode == 'detailed':
                    ind_other = np.nonzero(
                        np.array(self.otherInd) == iimark)[0]            
                    for iiother in ind_other:
                        print(("  -- "+self.otherMark[iiother]+':').ljust(
                            otherMarkLength) + 
                              self._tstr(self.otherTime[iiother]))
        else:
            # Print combine sub-counters
            sub_unique = set(self.otherMark)
            markLength = len(max(sub_unique, key=len))+3
            for iisub, isub in enumerate(sub_unique):
                ind_this = np.nonzero(np.array(self.otherMark) == isub)[0]
                if len(ind_this) == 0: set_trace()

                sumTime = np.sum(np.array(self.otherTime)[ind_this])
                print((isub+':').ljust(markLength) + self._tstr(sumTime))
            
        print("-" * 70)
        print("")

    def _tstr(self, time):
        """Produce formatted time string"""
        
        tstr = "{:5.2f} sec." .format(time)
        if not self.minutes and not self.percent:
            return tstr

        if self.minutes and self.percent:
            return (tstr + " ({:.2f} min., {:.1f}%)" 
                    .format(time/60, time/self.fullTime*100))

        if self.minutes:
            return tstr + " ({:.2f} min.)" .format(time/60)

        if self.percent:
            return tstr + " ({:.1f}%)" .format(time/self.fullTime*100)


class Galaxies:
    """
    Class for info about galaxies to be processed in one target snap.
    """

    def __init__(self, snap):
        """
        Constructor for the class.

        Sets up basic info and finds all SF-identified galaxies.

        Parameters
        ----------
        snap : Snapshot class instance
            The target snapshot for assigning particles to galaxies.
        """
        
        self.targetSnap = snap
        self.tSnap = snap.isnap          # For convenience
        self.sim = self.targetSnap.sim
        self.out = self.sim.output

        # Find which galaxies are found by SF:
        self.candidate_galaxies = np.nonzero(self.targetSnap.shi >= 0)[0]
        self.numCandidate_galaxies = len(self.candidate_galaxies)

        print("Found {:d} Subfind-alive galaxies in snapshot {:d}..."
              .format(self.numCandidate_galaxies, self.targetSnap.isnap))

        # Start set of all last-SF-alive snapshots of candidate galaxies:
        self.last_sf_alive = set()
        if self.numCandidate_galaxies:
            self.last_sf_alive.add(snap.isnap)

        # Start a (to-be-augmented/modified) array of pointers to centrals:
        # If appropriate, this will be modified below for dead galaxies.
        self.cenGalAssoc = np.copy(self.targetSnap.cenGal)
        
        # Add pointer to underlying snapshot:
        snap.set_galaxies(self)

        # Set up counter of particle origins:
        self.numOrigins = np.zeros(6, dtype = int)
        self.numBoundOrigins = np.zeros(6, dtype = int)

        # Set up (initially) empty lists of lost cens and sats:
        self.lost_cens = set([])
        self.lost_sats = set([])
        self.resuscitated = set([])

        # Set up lookup list to find galaxies by ID:
        self.revGal = np.zeros(self.sim.numGal, dtype = int) - 1
        self.revGal[self.candidate_galaxies] = np.arange(
            self.numCandidate_galaxies)

    def add_temporarily_lost(self):                     # Class: Galaxies
        """
        Add temporarily lost galaxies to the list (code -9).
        
        The rule is that these galaxies must have merged (i.e. have
        a direct descendant in the snapshot immediately after their
        disappearance) and have an identifiable carrier in the target
        snapshot, the central of which is taken as their central.
        """
        
        spiderloc = self.sim.spiderloc    # Pull out for convenience
        tSnap = self.targetSnap.isnap     # Ditto

        # Load info about direct descendants from the possible snapshots
        # before any temporarily missing ones could have disappeared.
        # It is more efficient to do this once, rather than (possibly) 
        # re-loading it for every single galaxy.
        # Note that we start at the target snap -- this is not strictly 
        # necessary for the descendant, but it is for galaxySH 
        # since some (many!) galaxies will be first missing in this snap.

        dirDesc = []   # List for direct descendant SHs per snap
        galSH = []     # List for subhalo galaxy IDs per snap
        for iback in range(0, 6):
            isnap_curr = tSnap - iback
            if isnap_curr < 0: break # Don't go beyond S0.

            directDescendantSH = yb.read_hdf5(
                spiderloc, 'Subhalo/Snapshot_{:03d}/Forward/SubHaloIndexCR0'
                .format(isnap_curr))
            dirDesc.append(directDescendantSH)
            galaxySH = yb.read_hdf5(
                spiderloc, 'Subhalo/Snapshot_{:03d}/Galaxy' 
                .format(isnap_curr))
            galSH.append(galaxySH)

        # Identify a 'longlist' of candidates to be added: all SHI == -9.
        # Set up a 'marklist' for marking galaxies to be included.

        gal_temp_lost_base = np.nonzero(self.targetSnap.shi == -9)[0]
        n_temp_lost_base = len(gal_temp_lost_base)
        marklist = np.zeros(n_temp_lost_base, dtype = np.int8)
        print("Considering {:d} temporarily lost galaxies..."
              .format(n_temp_lost_base))

        # Now go through candidates one-by-one and inspect:
        for iilost, ilost in enumerate(gal_temp_lost_base):

            # Find the direct descendant just after the last-SF-alive snap:
            snap_last_alive = np.nonzero(
                self.sim.shi[ilost, :tSnap] >= 0)[0][-1]
            nsnap_back = tSnap - snap_last_alive

            shi_last_alive = self.sim.shi[ilost, snap_last_alive]
            directDescendant = dirDesc[nsnap_back][shi_last_alive]
            
            if directDescendant < 0:
                # Galaxy does not merge (send any links) -- ignore here.
                continue
                
            # At this point, there *is* a direct descendant 
            # --> the galaxy has merged, and we consider it.
            marklist[iilost] = 1  

            # Find the galaxy with which it has merged, in target snap:
            # (NB: need to look up the SHI in the snap AFTER last alive)
            galDirDesc = galSH[nsnap_back - 1][directDescendant]
            mergeGal = self.sim.mergeList[galDirDesc, tSnap]
            if self.targetSnap.shi[mergeGal] < 0:
                if par['Verbose']:
                    print("Weird -- hiding galaxy ({:d}), "
                          "or even its merger target ({:d}), "
                          "do not exist in target snap. I'm giving up."
                          .format(galDirDesc, mergeGal))
                continue
                
            # Record the central of the object with which it has merged
            # as the central from which to unbind this satellite:
            self.cenGalAssoc[ilost] = self.targetSnap.cenGal[mergeGal]

        # Done going through individual galaxies, store result.
        subind_tempLost = np.nonzero(marklist == 1)[0]
        gal_temp_lost = gal_temp_lost_base[subind_tempLost]
        self.candidate_galaxies = np.unique(
            np.concatenate((self.candidate_galaxies, gal_temp_lost)))
        print("Added {:d} galaxies that were temporarily lost by Subfind..."
              .format(len(gal_temp_lost)))
        self.numCandidate_galaxies = len(self.candidate_galaxies)

        # Update reverse-lookup-list:
        self.revGal[self.candidate_galaxies] = np.arange(
            self.numCandidate_galaxies)

    def add_permanently_lost(self):                     # Class: Galaxies
        """
        Add permanently lost galaxies to the list.

        The rule is that they must have merged (code -5/-15), not just
        disappeared into nothing (as may happen when a FOF group 
        fluctuates below the detection threshold). We do not have to check 
        this explicitly, because their SF status code tells us 
        (-5/-15 vs. -10/-20).

        Unless the experimental flag `par['Lost']['FindCantorLost']'
        is set, the galaxy must have been identified by Cantor in the 
        previous snapshot. In addition, the galaxy must have had its last 
        SF-alive snapshot within `par['Lost']['MaxLostSnaps']' snapshots 
        (otherwise the assumption is that it should have re-surfaced by now).

        To prevent `ghost' galaxies that have physically become part
        of their merger host (but may still be self-bound), the offset
        from their merger host has to satisfy
        (delta_r/sigma_r)^2 + (delta_v/sigma_v)^2 > par['Lost']['MinOffset']^2
        where delta_r(v) is the position (velocity) offset from its carrier,
        and sigma_r(v) the position (velocity) dispersion of its core 
        particles (as recorded in GalaxyPaths.hdf5).

        Note that this is function is not called for target snapshot 0,
        since there can be no lost galaxies there already.
        """

        isnap_targ = self.targetSnap.isnap

        # Most general long-list: galaxies merged < N_max snaps back:
        gal_perm_lost = np.nonzero(
            ((self.targetSnap.shi == -5) | (self.targetSnap.shi == -15)) 
            & (self.sim.lastsnap >= isnap_targ-par['Lost']['MaxLostSnaps']))[0]

        # Limit to galaxies found by Cantor in previous snap?
        if not par['Lost']['FindCantorLost']:
            subInd = np.nonzero(
                self.sim.output.shiExtended[
                    gal_perm_lost, isnap_targ-1] >= 0)[0]
            gal_perm_lost = gal_perm_lost[subInd]

        # Limit to galaxies that are sufficiently offset from carrier:
        if par['Lost']['MinOffset'] is not None:
            subInd = np.nonzero(self.phase_space_offset(gal_perm_lost)
                                >= par['Lost']['MinOffset'])
            gal_perm_lost = gal_perm_lost[subInd]

        # Find the central galaxies to attach these to in target snap:
        # the central of their carrier (ignore those where it does not exist)
        
        mergeGal = self.sim.mergeList[gal_perm_lost, isnap_targ]
        subind_carried = np.nonzero(self.targetSnap.shi[mergeGal] >= 0)[0]
        gal_perm_lost = gal_perm_lost[subind_carried]

        self.cenGalAssoc[gal_perm_lost] = (
            self.targetSnap.cenGal[mergeGal[subind_carried]]) 

        # Add all the last-alive snapshots of these to the internal set:
        self.last_sf_alive.update(self.sim.lastsnap[gal_perm_lost])
        
        # Done processing the galaxies, store result.
        self.candidate_galaxies = np.unique(
            np.concatenate((self.candidate_galaxies, gal_perm_lost)))
        print("Added {:d} galaxies that were permanently lost by Subfind..."
              .format(len(gal_perm_lost)))
        print("Unique last-alive snapshots of candidate galaxies:")
        print(self.last_sf_alive)
        self.numCandidate_galaxies = len(self.candidate_galaxies)

        # Update reverse-lookup-list:
        self.revGal[self.candidate_galaxies] = np.arange(
            self.numCandidate_galaxies)

    def phase_space_offset(self, galIDs):               # Class: Galaxies
        """
        Determine the phase space offset of galaxies from their carriers.

        This is defined as 
        offset^2 = (delta_r/sigma_r)^2 + (delta_v/sigma_v)^2,
        where delta_r(v) is the coordinate (velocity) offset from the
        carrier, and sigma_r(v) the coordinate (velocity) dispersion of
        the galaxies' core particles, all as recorded in the 
        GalaxyPaths.hdf5 catalogues and all at the target snap.

        Parameters:
        -----------
        galIDs : ndarray (int)
            The IDs of the galaxies whose offset is to be measured.

        Returns:
        --------
        offset : ndarray (float)
            The phase-space offset for each galaxy (which may be zero).
        """
        
        # Set up name aliases for convenience:
        pathLoc = self.sim.pathloc
        snap = self.targetSnap
        isnap = snap.isnap
        mergeList = self.sim.mergeList[:, isnap]

        # Work out the snepshot index of the current target snapshot 
        # (required because info in GalaxyPaths.hdf5 is arranged by these):
        isnep = yb.read_hdf5(pathLoc, 'SnapshotIndex')[isnap]
        snepPre = 'Snepshot_' + str(isnep).zfill(4) + '/'

        # Load coordinate/velocity centroids and dispersions:
        # (convert to proper Mpc):
        conv_astro_pos = yb.read_hdf5_attribute(
            pathLoc, snepPre + 'Coordinates', 'aexp-factor')
        conv_astro_pos *= yb.read_hdf5_attribute(
            pathLoc, snepPre + 'Coordinates', 'h-factor')
        conv_astro_vel = yb.read_hdf5_attribute(
            pathLoc, snepPre + 'Velocity', 'aexp-factor')
        conv_astro_vel *= yb.read_hdf5_attribute(
            pathLoc, snepPre + 'Velocity', 'h-factor')

        pos = yb.read_hdf5(pathLoc, snepPre + 'Coordinates')*conv_astro_pos
        posDisp = yb.read_hdf5(
            pathLoc, snepPre + 'CoordinateDispersion')*conv_astro_pos
        vel = yb.read_hdf5(pathLoc, snepPre + 'Velocity')*conv_astro_vel
        velDisp = yb.read_hdf5(
            pathLoc, snepPre + 'VelocityDispersion')*conv_astro_vel

        # Get carriers of galaxies:
        carrierIDs = mergeList[galIDs]
        
        # Form coordinate and velocity offsets galaxy <--> carrier:
        posOffset = np.linalg.norm(
            pos[galIDs, :] - snap.galPos[carrierIDs, :], axis = 1)
        velOffset = np.linalg.norm(
            vel[galIDs, :] - snap.galVel[carrierIDs, :], axis = 1)

        # Limit position dispersion and offsets to a multiple of the 
        # softening length:
        if par['Lost']['MinSoftenings'] is not None:
            minVal = par['Lost']['MinSoftenings'] * snap.epsilon
            posDisp = np.clip(posDisp, minVal, None)
            posOffset = np.clip(posOffset, minVal, None)

        return np.sqrt((posOffset/posDisp[galIDs])**2 
                       + (velOffset/velDisp[galIDs])**2)

    def assign_cen_sat(self):                           # Class: Galaxies
        """
        Decide which of the candidate galaxies are cen, which sat.

        For SF-alive galaxies, this is taken from the cenGal table 
        (regularized if desired). All SF-dead galaxies are treated as sats.
        """
        
        satflag_cand = (
            self.sim.satFlag[self.candidate_galaxies, self.targetSnap.isnap])

        subind_cen = np.nonzero(satflag_cand == 0)[0]
        subind_sat = np.nonzero(satflag_cand != 0)[0]  # Includes SF-dead
        
        self.cen_candidates = self.candidate_galaxies[subind_cen]
        self.sat_candidates = self.candidate_galaxies[subind_sat]
    
        print("Out of {:d} galaxies to be tested in snap {:d},\n"
              "{:d} are centrals and {:d} satellites." 
              .format(self.numCandidate_galaxies, self.targetSnap.isnap,
                      len(self.cen_candidates), len(self.sat_candidates)))

    def remove_spectres(self):                          # Class: Galaxies
        """
        Remove spectres from the list of candidate galaxies.

        Criteria for removal are:
        - galaxy is flagged as a spectre
        - EITHER: par['Galaxies']['DiscardAllSpectres']
          OR:
          - galaxy is a satellite in target snapshot
          - galaxy is in same FOF group as its (alive) spectre parent
        """
        
        sats = self.sat_candidates
        spectreFlag = self.sim.spectreFlag
        spectreParents = self.sim.spectreParents

        ind_spectre = np.nonzero(
            (spectreFlag[sats] == 1) &
            ((par['Galaxies']['DiscardAllSpectres']) | (
            (self.sim.shi[spectreParents[sats], self.targetSnap.isnap] >= 0) &
            (self.targetSnap.cenGal[spectreParents[sats]] ==
             self.targetSnap.cenGal[sats]))))[0]
        self.spectres = set(self.sat_candidates[ind_spectre])
        print("Identified {:d} spectre galaxies in target snap {:d}..."
              .format(len(ind_spectre), self.targetSnap.isnap))

        # If no spectres were found, we are done:
        if len(self.spectres) == 0: return

        # Remove galaxies from *total* candidate list (first)...
        mask = np.zeros(len(self.candidate_galaxies))
        full_ind = self.revGal[self.sat_candidates[ind_spectre]]
        if (np.max(np.abs(np.sort(self.candidate_galaxies[full_ind]) -
                          np.sort(self.sat_candidates[ind_spectre]))) > 0):
            print("Inconsistency in spectre removal -- investigate.")
            set_trace()
        mask[full_ind] = 1
        ind_good = np.nonzero(mask == 0)[0]
        self.candidate_galaxies = self.candidate_galaxies[ind_good]
        self.numCandidate_galaxies -= len(ind_spectre)

        # Now remove spectres from sat_candidates array:
        mask = np.zeros(len(self.sat_candidates))
        mask[ind_spectre] = 1
        ind_good = np.nonzero(mask == 0)[0]
        self.sat_candidates = self.sat_candidates[ind_good]

        # Sanity check for number consistency:
        if len(self.candidate_galaxies) != self.numCandidate_galaxies:
            print("Inconsistent galaxy number. Argh!")
            set_trace()

        # Update reverse-lookup-list:
        self.revGal[self.candidate_galaxies] = np.arange(
            self.numCandidate_galaxies)
        
    def find_reference_snapshots(self):                 # Class: Galaxies
        """
        Determine the 'reference snapshot' of each satellite.

        Normally, this is the last snap in which it was a central,
        and the (current) central not one of its satellites (which should
        not happen with regularization anyway). If no such snapshot can
        be found, the one in which its mass is maximum is taken instead.
        """

        cenGal = self.sim.cenGal
        satFlag = self.sim.satFlag

        # Set up and initialize a list of reference snaps for all sats.
        refSnap = np.zeros(len(self.sat_candidates), dtype = np.int8)-1

        # Now go through snapshots in reverse and match their galaxies:
        for isnap in range(self.tSnap, -1, -1):
            # Three clauses below: galaxy must (i) not have a refSnap, 
            # (ii) be a central, (iii) not be cen-sat swapped with its
            # central in target snapshot:
            ind_this = np.nonzero(
                (refSnap < 0)     
                & (satFlag[self.sat_candidates, isnap] == 0)  
                & (cenGal[cenGal[self.sat_candidates, self.tSnap], isnap]
                   != self.sat_candidates)  
            )[0]
            refSnap[ind_this] = isnap
            
        # Check whether any galaxies have never been centrals.
        # For these, set reference snap to point of peak total mass.

        subind_neverCen = np.nonzero(refSnap < 0)[0]
        print("Could not find a reference snapshot for {:d} satellites..." 
              .format(len(subind_neverCen)))
        if len(subind_neverCen) > 0:
            refSnap[subind_neverCen] = (np.argmax(self.sim.msub[
                self.sat_candidates[subind_neverCen], :(self.tSnap+1)], 
                                                  axis = 1))
                                        
        # Sanity check to make sure all have a reference snap now:
        n_notmatched = np.count_nonzero(refSnap < 0)
        if n_notmatched > 0:
            print("WTF? {:d} galaxies still have no reference snap? "
                  "Investigate NOW!" .format(n_notmatched))
            set_trace()

        # Sanity check to make sure all galaxies exist in their ref snap:
        if np.min(self.sim.shi[self.sat_candidates, refSnap]) < 0:
            print("Galaxies not existing in reference snap!")
            set_trace()

        self.sat_refSnap = refSnap
              
    def setup_centre_arrays(self):                      # Class: Galaxies
        """Set up arrays to hold potential minimum ID/position."""
        self.potMin_ID = np.zeros(self.numCandidate_galaxies, dtype = int)-1
        self.potMin_pos = np.zeros((self.numCandidate_galaxies, 3), 
                                   dtype = float) + np.nan
        self.monk_pos = np.copy(self.potMin_pos)
        self.monk_vel = np.copy(self.potMin_pos)

    def add_origin_counts(self, numOrigins):            # Class: Galaxies
        """Increment the total counts of particles by origin."""
        self.numOrigins += numOrigins

    def add_bound_origin_counts(self, numOrigins):      # Class: Galaxies
        """Increment the total counts of bound particles by origin."""
        self.numBoundOrigins += numOrigins

    def add_lost_sat(self, igal):                       # Class: Galaxies
        """Add a galaxy to the internally-kept list of lost satellites."""
        self.lost_sats.add(igal)

    def add_lost_cen(self, igal):                       # Class: Galaxies
        """Add a galaxy to the internally-kept list of lost centrals."""
        self.lost_cens.add(igal)

    def add_resuscitated(self, igal):                   # Class: Galaxies
        """
        Add a galaxy to the internal list of `resuscitated' galaxies.
        
        This means galaxies that have been lost by Subfind, but were
        (re-)discovered by Cantor.
        """
        self.resuscitated.add(igal)

    def set_final_coordinates(self, igal, pos, vel):    # Class: Galaxies
        """
        Record the position and velocity at the end of MONK unbinding.

        Parameters:
        -----------
        igal : int
            The galaxyID whose coordinates should be set.
        pos : ndarray (float) [3]
            The coordinates of the final reference frame of the galaxy.
        vel : ndarray (float) [3]
            The velocity coordinates of the final reference frame.
        """

        # Look up internal index of galaxy:
        galIndex = self.revGal[igal]
        if galIndex < 0:
            print("Uh oh. Trying to set coordinates of non-existing "
                  "galaxy (ID={:d}). Please investigate."
                  .format(igal))
            set_trace()
        
        self.monk_pos[galIndex, :] = pos
        self.monk_vel[galIndex, :] = vel

    def get_pos(self, igal):                            # Class: Galaxies
        """Retrieve the position of a galaxy by its ID."""
        galIndex = self.revGal[igal]
        if galIndex < 0:
            print("Wrong galaxy ID.")
            set_trace()
        return self.monk_pos[galIndex, :]

    def get_vel(self, igal):                            # Class: Galaxies
        """Retrieve the velocity of a galaxy by its ID."""
        galIndex = self.revGal[igal]
        if galIndex < 0:
            print("Wrong galaxy ID.")
            set_trace()
        return self.monk_vel[galIndex, :]
         
    def get_pos_multi(self, galIDs):                      # Class: Galaxies
        """Retrieve position of many galaxies by their ID."""
        galIndex = self.revGal[galIDs]
        if np.min(galIndex) < 0:
            print("Illegal galaxy IDs.")
            set_trace()
        return self.monk_pos[galIndex, :]

    def get_vel_multi(self, galIDs):                      # Class: Galaxies
        """Retrieve the velocity of many galaxies by their ID."""
        galIndex = self.revGal[galIDs]
        if np.min(galIndex) < 0:
            print("Illegal galaxy IDs.")
            set_trace()
        return self.monk_vel[galIndex, :]
        

    def set_potMin(self, igal, index):                  # Class: Galaxies
        """
        Set the potential minimum particle for a given galaxy.

        Parameters:
        ----------
        igal : int
            The galaxyID whose potential minimum should be set.
        index : int
            The particle index (in self.particles) to set as pot. min.
        """

        # Look up internal index of galaxy:
        galIndex = self.revGal[igal]
        if galIndex < 0:
            print("Uh oh. Trying to set potential minimum of non-existing "
                  "galaxy (ID={:d}). Please investigate."
                  .format(igal))
            set_trace()
        
        particles = self.targetSnap.particles
        self.potMin_ID[galIndex] = particles.ids[index]
        self.potMin_pos[galIndex, :] = particles.pos[index, :]

    def get_potMin_pos(self, igal):                     # Class: Galaxies
        """
        Return the potential minimum position for a galaxy ID.
        
        If no position was recorded (because the galaxy is a central
        and we didn't unbind them), the input position from SF is used.
        """

        galIndex = self.revGal[igal]
        if galIndex < 0:
            print("Uh oh. Trying to get potential minimum of non-existing "
                  "galaxy (ID={:d}). Please investigate."
                  .format(igal))
            set_trace()
        
        if not np.isnan(self.potMin_pos[galIndex, 0]):
            return self.potMin_pos[galIndex, :]
        else:
            return self.targetSnap.galPos[igal, :]

    def get_potMin_pos_multi(self, galID):              # Class: Galaxies
        """
        Return the potential minimum position of the specified galaxies.

        If no position was recorded (because the galaxy is a central
        and we didn't unbind them), the input position from SF is used.

        Parameters:
        -----------
        galID : ndarray (int) [N]
            The galaxy IDs for which the potential centre should be found.

        Returns:
        --------
        cen : ndarray (float) [N, 3]
            The position of each galaxy's potential minimum.
        """
        
        galIndex = self.revGal[galID]
        if np.min(galIndex) < 0:
            print("Uh oh. Trying to access non-existing galaxies...")
            set_trace()

        cen = self.potMin_pos[galIndex, :]

        # Fill in position of skipped galaxies with values from Subfind:
        ind_skipped = np.nonzero(np.isnan(cen[:, 0]))[0]
        cen[ind_skipped, :] = self.targetSnap.galPos[galID[ind_skipped], :]

        return cen


class SplitList:
    """Class to simplify particle lookup by a given property."""

    def __init__(self, quant, lims):
        """
        Class constructur.

        Parameters:
        -----------
        quant : ndarray
            The quantity by which elements should be retrievable.
        lims : ndarray
            The boundaries (in the same quantity and units as quant) of 
            each retrievable 'element'.
        """
        
        self.argsort = np.argsort(quant)
        self.splits = np.searchsorted(quant, lims, sorter = self.argsort)

    def __call__(self, index):
        """
        Return all input elements that fall in a given bin.

        Parameters:
        -----------
        index : int
            The index corresponding to the supplied 'lims' array for which
            elements should be retrieved. All elements with 'quant' between
            lims[index] and lims[index+1] will be returned.
        
        Returns:
        --------
        elements : ndarray
            The elements that lie in the desired quantity range.
        """

        return self.argsort[self.splits[index]:self.splits[index+1]]  
        
class Galaxy:
    """Class to hold information about one galaxy for unbinding."""

    def __init__(self, igal, ref_snap, target_particles, verbose):
        """
        Class constructor. Load particles from ref_snap.

        Parameters:
        -----------
        igal : int
            ID of the galaxy.
        ref_snap : Snapshot instance
            Reference snapshot of the galaxy, where its own particles are
            to be looked up. The galaxy is always SF-alive here.
        """

        self.verbose = verbose
        self.igal = igal
        self.ref_snap = ref_snap
        self.particles = target_particles
        self.target_snap = target_particles.snap
        self.ref_shi = self.ref_snap.shiLoad[igal]
        self.sim = self.ref_snap.sim
        self.target_shi = self.target_snap.shi[igal]

        if self.ref_shi  < 0:
            if not par['Input']['FromCantor']:
                print("Galaxy {:d} has SHI={:d} in its reference snapshot "
                      "{:d}!" .format(igal, self.ref_shi, ref_snap.isnap))
                set_trace()
            
            # In load_from_cantor mode, we can actually have a galaxy
            # being non-existent in its reference snapshot...
            self.gal_IDs = np.zeros(0, dtype = np.int)

        elif par['Sources']['RefSnap']:
            # Find IDs of galaxy itself
            self.gal_IDs = (
                ref_snap.ids[ref_snap.off_sh[self.ref_shi]:
                             ref_snap.off_sh[self.ref_shi] 
                             + ref_snap.len_sh[self.ref_shi]]).astype(np.int)

        else:
            self.gal_IDs = np.zeros(0, dtype = np.int)

        # Set up 'origins' flag to record how particles were found:
        self.origins = np.zeros(len(self.gal_IDs), dtype = np.int8)

        # Set up distances array if we load central particles
        # (will be filled properly in central-loading step):
        if par['Sources']['Centre']:
            self.distances = np.zeros(0, dtype = np.float32)

        if self.verbose:
            print("Loaded {:d} particles in ref-subhalo for galaxy {:d}..."
                  .format(len(self.gal_IDs), self.igal))

    def add_external_particles(self):                     # Class: Galaxy      
        """Add particles that are not in the galaxy in the ref snap."""

        # --------- Now add particles from other sources ----------

        # If desired, add particles in galaxy in previous snapshot:
        if par['Sources']['Prior']:
            self.add_prior_particles()

        # If desired, add particles in merging galaxies:
        if par['Sources']['Mergers']:
            self.add_merging_particles()

        # If desired, add particles in (optionally: subhaloes of)
        # the galaxy's FOF (internally checks if it's a central):
        if par['Sources']['Sats'] or par['Sources']['FOF']:
            self.add_fof_particles()

        # If desired, add particles in the galaxy's SF subhalo in 
        # the target snapshot (or last Cantor detection if dead):
        if par['Sources']['Subfind']:
            self.add_original_particles(self.target_snap)

        # If desired, add particles near the centre of the galaxy
        # in the target snapshot:
        if par['Sources']['Centre'] is not None:
            self.add_central_particles(self.particles)

    def lookup_particles(self):                           # Class: Galaxy
        """Unicate and look up galaxy's particles."""
        
        # Unicate the particle list to remove duplicates:
        self.unicate_particles()

        # Translate IDs --> indices, reject any particles that are
        # not in the galaxy's FOF (should not be common):
        self.part_inds, subIndFound = self.particles.get_indices(self) 
            
        # Need to explicitly reduce origins/distances arrays to account
        # for exclusion or particles in 'wrong' FOFs:
        self.origins = self.origins[subIndFound]
        if par['Sources']['Centre']:
            self.distances = self.distances[subIndFound]

        # Print out some statistics about identified particles:
        self.print_particle_statistics()

    def print_particle_statistics(self):                  # Class: Galaxy
        """Print information about target particles for one galaxy."""

        # First a sanity check that there are not zero particles:
        if len(self.part_inds) == 0:
            if self.verbose:
                print("Galaxy {:d} has no target particles at all?!"
                      .format(self.igal))
            if not par['Input']['FromCantor']:
                set_trace()
            return

        # Count the number by origin code (N.B.: during unication, the 
        # highest priority (lowest number) origin route is chosen).
        numOriginTypes, bE = np.histogram(self.origins, bins = 6, 
                                          range = [-1, 5])

        if self.verbose: 
            print("Found {:d} particles in target list, (out of {:d}, "
                  "{:.2f} per cent; {:d} including duplicates)..." 
                  .format(len(self.part_inds), self.unique_particle_number, 
                          len(self.part_inds)/self.unique_particle_number*100,
                          self.full_particle_number))
            print("Origins: {:d} -- {:d} -- {:d} -- {:d} -- {:d} -- {:d}"
                  .format(*numOriginTypes))

        # Finally, increment target-snap-wide origin counts:
        self.target_snap.target_galaxies.add_origin_counts(numOriginTypes)
    
                                                          # Class: Galaxy
    def print_bound_particle_statistics(self, ind_bound): 
        """Print information about bound particles for one galaxy."""
        
        # Count the number of bound particles by origin:
        numOriginTypes, bE = np.histogram(self.origins[ind_bound], bins = 6, 
                                          range = [-1, 5])
        if self.verbose:
            print("Bound origins: {:d} -- {:d} -- {:d} -- {:d} -- {:d} -- {:d}"
                  .format(*numOriginTypes))

        # Increment target-snap-wide origin counts:
        self.target_snap.target_galaxies.add_bound_origin_counts(
            numOriginTypes)

    def append_origins(self, num, code):                  # Class: Galaxy
        """
        Extend the internal 'origins' array by [num] elements of [code].
        """
        self.origins = np.concatenate((
            self.origins, np.zeros(num, dtype = np.int8) + code))

    def prepend_origins(self, num, code):                  # Class: Galaxy
        """
        Extend the internal 'origins' array by [num] elements of [code].

        In contrast to append_origins, the new elements are added in front.
        """
        self.origins = np.concatenate((
            np.zeros(num, dtype = np.int8) + code, self.origins))

    def add_prior_particles(self):                        # Class: Galaxy
        """
        Add particles from the snapshot prior to the target one.

        If the galaxy was not found by Cantor in the previous snapshot,
        then we search for a maximum of par['Sources']['MaxPriorCantor']
        snapshots backwards and pick the closest-alive snapshot. If the 
        galaxy is not found in any of them, no particles are added here.

        Note that getting here requires loading from Cantor, which is 
        nevertheless checked explicitly.

        Nothing happens if the target snapshot is the first one (S0).
        """

        if not par['Input']['FromCantor']:
            return

        if self.target_snap.isnap == 0: 
            self.isnap_prior = -1
            return

        isnap = self.target_snap.isnap
        igal = self.igal
        output = self.target_snap.sim.output
        priorData = self.target_snap.priorCantorData

        # First test whether the galaxy could be found in the last snap:
        shi_prior = self.sim.output.shiExtended[igal, isnap-1]
        if shi_prior >= 0: 
            isnap_prior = isnap - 1
        else:
            # Check further-back snaps:
            snaps_found = np.nonzero(
                output.shiExtended[
                    igal, isnap-par['Sources']['MaxPriorCantor'] : isnap] 
                >= 0)[0]
            if len(snaps_found):
                isnap_prior = (snaps_found[-1] 
                               + (isnap - par['Sources']['MaxPriorCantor']))
                shi_prior = output.shiExtended[igal, isnap_prior]
            else:
                self.isnap_prior = -1
                return  # Galaxy never found by Cantor -- can't do anything.

        ids_gal = priorData.get_sh_ids(isnap-isnap_prior, shi_prior)
        self.isnap_prior = isnap_prior
        self.gal_IDs = np.concatenate((ids_gal, self.gal_IDs))
        
        # Record origin of particles (prior subhalo: code -1)
        self.prepend_origins(len(ids_gal), -1)

        if self.verbose:
            print("   ... added {:d} particles from galaxy in its "
                  "previous snap ({:d})." 
                  .format(len(ids_gal), self.isnap_prior))
        
    def add_merging_particles(self):                      # Class: Galaxy
        """Add particles from galaxies that merge between ref and target."""

        igal = self.igal
        lastSnap = min(self.target_snap.isnap, self.sim.lastsnap[igal])
        
        # The way we find merging galaxies differs depending on whether
        # we unbind permanently dead galaxies (in this case, we must look
        # for mergers at the point when it was last alive):
        if par['Lost']['FindPermanentlyLost']:
            if self.ref_snap.lut_mergeTarg[lastSnap] == []:
                set_trace()
            gal_merging = (self.ref_snap.ind_alive[
                self.ref_snap.lut_mergeTarg[lastSnap](igal)])
        else:
            # Safety check to make sure lastSnap is not in past:
            if lastSnap < target_snap.isnap:
                print("Why are we unbinding a dead galaxy?")
                set_trace()
            gal_merging = (
                self.ref_snap.ind_alive[self.ref_snap.lut_mergeTarg(igal)])

        gal_IDs_merging = self.find_multiGal_IDs(gal_merging, self.ref_snap)
        
        self.gal_IDs = np.concatenate((self.gal_IDs, gal_IDs_merging))

        # Record origin of particles (merging: code 1)
        self.append_origins(len(gal_IDs_merging), 1)
    
        if self.verbose:
            print("   ... added {:d} particles from {:d} merger contributors."
                  .format(len(gal_IDs_merging), len(gal_merging)))

    def add_fof_particles(self):                          # Class: Galaxy
        """
        Add particles from (optionally: SHs within) the galaxy's FOF.

        If the galaxy is not a central in ref_snap, nothing happens.
        """

        snap = self.ref_snap
        if snap.sim.satFlag[self.igal, snap.isnap] == 1: return
        
        if par['Sources']['FOF']:
            # Include *all* particles in FOF group
            gal_fof = self.ref_snap.fof_index[self.ref_shi]
            fof_IDs = snap.ids[snap.off_fof[gal_fof] :
                               snap.off_fof[gal_fof]+snap.len_fof[gal_fof]]
        else:
            # "Only" include subhaloes in FOF group
            # First, find all satellites, using pre-computed look-up-table:
            gals_in_fof = self.ref_snap.ind_alive[
                self.ref_snap.lut_cenGal(self.igal)]
            fof_IDs = self.find_multiGal_IDs(gals_in_fof, self.ref_snap)

        self.gal_IDs = np.concatenate((self.gal_IDs, fof_IDs))

        # Record origin of particles (fof: code 2)
        self.append_origins(len(fof_IDs), 2)

        if self.verbose:
            print("   ... added {:d} particles from ref FOF group."
                  .format(len(fof_IDs)))

    def add_original_particles(self, target_snap):        # Class: Galaxy
        """
        Add particles identified by SF in target snap.

        For SF-dead galaxies, this is skipped (no SF counterpart).
        """

        # Abort for SF-dead galaxies:
        if target_snap.shi[self.igal] < 0: return

        # If desired, abort for galaxies that are currently a SF-central:
        if (par['Input']['RegularizedCens'] and 
            not par['Sources']['SubfindInSwaps']): 
            if not target_snap.sim.SFsatFlag[self.igal, target_snap.isnap]:
                return

        ids_gal = target_snap.get_gal_ids(self.igal)
        self.gal_IDs = np.concatenate((self.gal_IDs, ids_gal))
        
        # Record origin of particles (target subhalo: code 3)
        self.append_origins(len(ids_gal), 3)

        if self.verbose:
            print("   ... added {:d} particles from galaxy in snap {:d}." 
                  .format(len(ids_gal), target_snap.isnap))
    
    def add_central_particles(self, target_particles):    # Class: Galaxy
        """
        Add particles near the galaxy centre in the target snap.

        There are two inclusion radii:
        (i) A multiple of the galaxy's stellar half-mass radius. This is
            set by the par['Sources']['Centre'] variable (if None, we don't
            load anything in this function).
        (ii) A fixed radius around the galaxy centre, in kpc. This is
             set by the par['Sources']['CentreKpc'] variable (if not None).
        """
        
        if par['Sources']['Centre'] is None: return
        isnap = target_particles.snap.isnap   # For convenience

        # Need to set up a fake 'distance' array for all previously-added
        # particles, so that we can unicate particles later:
        dummy_distances = np.zeros(len(self.gal_IDs), dtype = np.float32)

        # Work out how far out we need to include particles:
        maxRad = self.sim.shmr[self.igal, isnap]
        if par['Sources']['CentreKpc'] is not None:
            maxRad = max(maxRad, par['Sources']['CentreKpc']/1e3) # kpc -> Mpc
        
        # If this is negative (non-existing galaxy) or zero (no stars),
        # we can stop right here:
        # (still need to append distances array, to keep it aligned 
        # with the IDs).
        if maxRad <= 0: 
            self.distances = np.concatenate((self.distances, dummy_distances)) 
            return

        # Get IDs of neighbours, and their distance from the galaxy:
        ids_cen, deltarad = target_particles.find_ngb_particles(self.igal, 
                                                                maxRad)
        # If there are no neighbours, we can return:
        if ids_cen is None:
            self.distances = np.concatenate((self.distances, dummy_distances)) 
            return

        # Append new IDs to previously found ones:
        self.gal_IDs = np.concatenate((self.gal_IDs, ids_cen))
        
        # Also need to update 'distance' array:
        self.distances = np.concatenate((dummy_distances, deltarad))

        # Record origin of particles (central: code 4)
        self.append_origins(len(ids_cen), 4)

        if self.verbose:
            print("   ... added {:d} central particles." .format(len(ids_cen)))

    def find_multiGal_IDs(self, galIDs, snap):    # Class: Galaxy
        """
        Find the particle IDs for all input galaxies.

        Parameters:
        -----------
        galIDs : ndarray (int)
            The galaxy IDs whose particles should be identified.
        snap : Snapshot instance
            The snapshot in which the particles should be found.
        
        Returns:
        --------
        part_ids : ndarray (int)
            The IDs of all galaxies.
        """

        # Identify galaxy subhalo indices 
        # (snap.shiLoad is set up to point to SF/Cantor SHIs appropriately)
        shi_gals = snap.shiLoad[galIDs]

        # Get total number of to-be-loaded particles:
        num_part_gals = int(np.sum(snap.len_sh[shi_gals]))  # Originally uint

        # Make `occupation plan' for output (works for CANTOR and SF input):
        offset_gals = np.cumsum(snap.len_sh[shi_gals])
        offset_gals = np.insert(offset_gals, 0, 0)

        # Load IDs of all contributing galaxies, directly into final place:
        part_ids = np.zeros(num_part_gals, dtype = int) - 1
        for iish, ish in enumerate(shi_gals):
            ish_ids = (
                snap.ids[snap.off_sh[ish] : snap.off_sh[ish]+snap.len_sh[ish]])
            part_ids[offset_gals[iish] : offset_gals[iish+1]] = ish_ids

        if par['Verbose'] >= 2:
            print("Finished loading {:d} IDs from {:d} contributing galaxies."
                  .format(num_part_gals, len(shi_gals)))
    
        return part_ids

    def unicate_particles(self):                          # Class: Galaxy
        """Remove duplicates from the internally-loaded particle list."""
 
        # Store original number of particles for reference:
        self.full_particle_number = len(self.gal_IDs)

        # source_index contains the indices of the first occurrence of
        # each particle
        self.gal_IDs, source_index = np.unique(self.gal_IDs, 
                                               return_index = True)
        if par['Sources']['Centre']:
            self.distances = self.distances[source_index]
        self.origins = self.origins[source_index] 

        # Do a consistency check to make sure we've got some left:
        if len(self.gal_IDs) <= 0:
            if self.verbose:
                print("Galaxy {:d} has no particles to consider at all?!"
                      .format(self.igal))
            
            # This situation can actually arise when loading back from
            # Cantor, since galaxies may then have zero particles in their 
            # reference snapshot. Otherwise, this indicates a problem:
            if not par['Input']['FromCantor']:
                set_trace()
        if self.verbose:
            print("Galaxy {:d} -- {:d} unique particles."
                  .format(self.igal, len(self.gal_IDs)))
        
        # Store unique particle number for future use:
        self.unique_particle_number = len(self.gal_IDs)

    def get_central(self, snap = None):                   # Class: Galaxy
        """
        Return this galaxy's central galaxy ID. 

        Parameters:
        -----------
        snap : Snapshot instance, optional
            The snapshot in which the central should be found. If this 
            is None (default), query the central in the target snapshot.
        """
        if snap is None:
            snap = self.target_snap
        return snap.target_galaxies.cenGalAssoc[self.igal]

    def find_target_coordinates(self):                    # Class: Galaxy
        """Find the coordinates of the galaxy in the target snapshot."""

        # Need to make sure these are float64, for compatibility with
        # MONK C routine.
        self.pos = self.target_snap.galPos[self.igal, :].astype(np.float64)
        self.vel = self.target_snap.galVel[self.igal, :].astype(np.float64)

        if np.min(self.pos) < 0:
            print("Looks like galaxy {:d} could not be located in "
                  "snap {:d}..." .format(self.igal, self.target_snap.isnap))
            # This can happen legitimately in strange cases...

class TargetParticles:
    """Class to load and hold particles in target snapshot."""
    
    def __init__(self, snap):
        """
        Constructor for this class.

        Determine total number of particles (by type) in the input snapshot,
        and set up internal storage for their required properties.
        The particle data are not actually read in here.

        Parameters:
        -----------
        snap : snapshot class instance
            The (target) snapshot in which to load particles.
        """

        self.snap = snap
        self.sim = self.snap.sim
        if par['Input']['FromSnapshot']:
            self.partdir = self.snap.snapdir
        else:
            self.partdir = self.snap.espdir 

        # Find the (relevant) total number of particles per type:
        self.npTotalType = yb.read_hdf5_attribute(self.partdir, 
                                                  'Header', 'NumPart_Total')
        for iptype in range(6):
            if iptype not in par['Input']['TypeList']: 
                self.npTotalType[iptype] = 0
        npTotal = np.sum(self.npTotalType)
        print("There are {:d} particles (in snap {:d}) in total..."
              .format(npTotal, self.snap.isnap))
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
        snap.set_particles(self)
    
    # Class: TargetParticles
    def cen_galaxy_from_ids(self, ptype_ids):
        """Derive (optionally: central) galaxy ID for an input ID array."""
        
        snap = self.snap    # For convenience

        gateSFID = st.Gate(ptype_ids, snap.ids)
        inds_SFID = gateSFID.in2()

        # Below, 'ptype_shi' refers to FOF or SH index (par.-dependent)
        ptype_hi = st.ind_to_sh(inds_SFID, snap.off, snap.len)
        ind_in = np.nonzero(ptype_hi >= 0)[0]

        del gateSFID
        del inds_SFID

        if par['Input']['LoadFullFOF']:
            # Extra step: FOF --> central SHI
            # First, reject particles in `empty' FOFs (with 0 SHs)
            subind_emptyFOF = np.nonzero(
                snap.fof_nsh[ptype_hi[ind_in]] == 0)[0]                
            ptype_hi[ind_in[subind_emptyFOF]] = -1
            
            # Now need to re-build list of in-group particles
            ind_in = np.nonzero(ptype_hi >= 0)[0]
            
            # Finally, convert FOF --> (cen) SHI
            ptype_hi[ind_in] = snap.fof_fsh[ptype_hi[ind_in]]

        # Final step: convert to (optionally: central) galaxyID
        return snap.shi_to_galaxy(
            ptype_hi, return_central = par['Input']['InitializeSatsToCen'])

    # Class: TargetParticles
    def cen_galaxy_from_esp(self, iptype):
        """
        Find (optionally: central) galaxy ID from ESP data.
        
        Only requires the particle type to process, the actual data 
        (GrouNumber and SubGroupNumber) are loaded internally.
        """
        
        pt = 'PartType{:d}/' .format(iptype)
        partDir = self.partdir
        snap = self.snap

        # Load group number (always required):
        ptype_gn = st.eagleread(partDir, pt + 'GroupNumber', astro = False)

        ptype_shi = np.zeros(self.npTotalType[iptype], dtype = np.int32)-1
        if par['Input']['LoadFullFOF']:
            # In this case, `SHI' is the GROUP CENTRAL
            # NB: We only consider FOFs that have at least one subhalo
            ind_in = np.nonzero((ptype_gn > 0) 
                                & (snap.fof_nsh[ptype_gn - 1] > 0))[0]
            ptype_shi[ind_in] = snap.fof_fsh[ptype_gn[ind_in_sg] - 1]
        else:
            # Load only particles in subhaloes --> load SubGroupNumber:
            ptype_sgn = st.eagleread(partDir, pt + 'SubGroupNumber', 
                                     astro = False)
            ind_in = np.nonzero(ptype_sgn < 2**30)[0]
            ptype_shi[ind_in] = (ptype_sgn[ind_in] 
                                 + snap.fof_fsh[np.abs(ptype_gn[ind_in]) - 1])
            del ptype_sgn
        
        del ptype_gn
        
        # Final step: convert to (optionally: central) galaxyID
        return snap.shi_to_galaxy(
            ptype_shi, return_central = par['Input']['InitializeSatsToCen'])
                    
    def load_particle_data(self):                # Class: TargetParticles
        """Load actual particle data, for all particle types."""
                
        for iptype in par['Input']['TypeList']:
            if self.npTotalType[iptype] == 0: continue
            print("Loading {:d} particles of type {:d}..." 
                  .format(self.npTotalType[iptype], iptype))
            sTimeType = time.time()

            # NB: the following works because types 2 and 3 have been 
            #     explicitly zeroed in totals (if discarded)
            npStart = np.sum(self.npTotalType[:iptype])
            npEnd = np.sum(self.npTotalType[:iptype+1])
            pt = 'PartType{:d}/' .format(iptype)

            # 1.) Particle IDs, and conversion to (central) galaxy
            ptype_ids = st.eagleread(self.partdir, pt + 'ParticleIDs', 
                                     astro = False)
            if ptype_ids.shape[0] != self.npTotalType[iptype]:
                print("Read unexpected number of particles: "
                      "{:d} instead of {:d}!"
                      .format(ptype_ids.shape[0], npTotalType[iptype]))
                set_trace()
            self.ids[npStart:npEnd] = ptype_ids

            # Getting to central galaxy depends on snapshot/ESP loading:
            if par['Input']['FromSnapshot']:
                self.galaxy[npStart:npEnd] = (
                    self.cen_galaxy_from_ids(ptype_ids))
            else:
                self.galaxy[npStart:npEnd] = (
                    self.cen_galaxy_from_esp(iptype))
                
            del ptype_ids

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

    def reject_unassociated(self):               # Class: TargetParticles
        """
        Prune the particle list -- remove those not in FOF/SH.

        Such 'outside' particles include non-FOF particles (including 
        'aperture' particles even when loading from ESP), and particles
        within FOFs that are not in any SF-subhalo (when only considering
        SF-subhalo particles).
        """
        
        stime = time.time()

        ind_assigned = np.nonzero(self.galaxy >= 0)[0]
        n_assigned = len(ind_assigned)
        print("Out of {:d} loaded particles, {:d} have been associated "
              "with a subhalo\n(={:.2f}%). Eliminating the rest..."
              .format(self.npTotal, n_assigned, n_assigned/self.npTotal*100))

        self.ids = self.ids[ind_assigned]
        self.pos = self.pos[ind_assigned, :]
        self.vel = self.vel[ind_assigned, :]
        self.mass = self.mass[ind_assigned]
        self.energy = self.energy[ind_assigned]
        self.type = self.type[ind_assigned]
        self.galaxy = self.galaxy[ind_assigned]
        
        print("   done ({:.2f} sec.)." .format(time.time() - stime))
    
    def create_origin_arrays(self):              # Class: TargetParticles
        """
        Set up arrays to record association status of particles.

        These will hold information about the way in which each 
        particle is (potentially) claimed by a satellite galaxy.
        """

        nPart = len(self.ids)

        # A numerical code to record the association type 
        # (higher ==  weaker, so 100 -> none at all):
        self.sourceType = np.zeros(nPart, dtype = np.int8) + 100   

        # Distance from the galaxy centre, for 'central-included' particles:
        # (further == weaker, so initialize to infinity):
        self.distance = np.zeros(nPart, dtype = np.float32) + np.inf   

        # Snapshot in which 'prior Cantor' membership was tested:
        # (later wins, so anything will beat -1)
        self.lastSeen = np.zeros(nPart, dtype = np.int8) - 1
    
        # Also create binding energy array, if desired:
        if par['Monk']['ReturnBindingEnergy']:
            self.binding_energy = np.zeros(nPart, dtype = np.float32)+np.nan

    def create_id_lookup_list(self):             # Class: TargetParticles
        """Set up a reverse ID list for rapid particle lookup."""
        
        stime = time.time()
        print("Inverting all-particle ID list...", end = '', flush = True)
        nTotalDM = yb.read_hdf5_attribute(self.snap.snapdir, 
                                                  'Header', 'NumPart_Total')[1]
        maxID = 2 * (nTotalDM + 1)
        self.revID = yb.create_reverse_list(
            self.ids, maxval = maxID, delete_ids = False)
        print("done ({:.2f} sec.)!" .format(time.time() - stime))

                                                 # Class: TargetParticles
    def monk(self, indices, halopos_init, halovel_init, hubble_z, fixCentre,
             centreMode=0, centreFrac=0.1, status=None, monotonic=1,
             resLimit=0.0007, potErrTol=1.0, returnBE=1, verbose=0):

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
        if par['Monk']['Bypass']:
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
                              potErrTol,    # Potential error tolerance
                              verbose,      # Monk verbosity
                              returnBE)     # Return (Binding) Energy

        if returnBE:
            return ind_bound, gal_energy
        else:
            return ind_bound
    
    def save_full_cens(self):                    # Class: TargetParticles
        """
        Make a copy of the pre-unbinding particle-->cenGal list.

        When centrals are unbound before satellites, this is required to
        (later) associate potentially central-unbound particles in sats to
        their FOF. Even if not, it simplifies the lookup because it avoids
        having to filter through cenGal explicitly.
        """
        self.fof_cenGal = np.copy(self.galaxy)
        
    
    def unbind_cens(self, galaxies):             # Class: TargetParticles
        """
        Remove unbound particles from central galaxies.

        Parameters:
        -----------
        
        galaxies : Galaxies class instance
            The galaxies whose centrals should be tested.
        """

        unbind_startTime = time.time()

        cens = galaxies.cen_candidates
        numCens = len(galaxies.cen_candidates)
        eprint("Beginning to unbind {:d} centrals..."
               .format(numCens), linestyle = '=')
        
        # Counter for how many particles are in centrals (in total)
        nPartCenTotal = 0

        # Set up a list saying whether each particle is bound
        statList = np.zeros(len(self.galaxy), dtype = np.int8)+1

        # List saying what fraction of each central was found bound:
        boundFracs = np.zeros(numCens, dtype = np.float32)

        # List saying how many particles each central had originally:
        numOrigParts = np.zeros(numCens, dtype = np.int32)

        # Prepare a split list to efficiently find particles that 
        # (currently) belong to each galaxy:
        lookUpTable = SplitList(self.galaxy, self.sim.gal_lims)

        for iicen, igal in enumerate(cens):

            if iicen % par['Check']['ReportFrequency'] == 0 or par['Verbose']:
                print("")
                print("Unbinding central {:d}/{:d} -- galID {:d}..."
                      .format(iicen, numCens, igal))
            
            # Load particle indices of current galaxy: 
            gal_inds = lookUpTable(igal)
            nPartCenTotal += len(gal_inds)

            # Mark central's particles as unbound first:
            statList[gal_inds] = 0

            # Do the actual unbinding -- keeping the centre fixed here.
            monk_res = self.monk(
                gal_inds, self.snap.galPos[igal, :], self.snap.galVel[igal, :],
                self.snap.hubble_z.value, fixCentre=1, 
                resLimit = self.snap.epsilon, 
                potErrTol = par['Monk']['PotErrTol'],
                returnBE = par['Monk']['ReturnBindingEnergy'],
                verbose = par['Monk']['Verbose'])

            if par['Verbose']:
                print("GalaxyID {:d} has {:d} bound particles "
                      "(original: {:d}, kept {:.2f}%)"
                      .format(igal, len(ind_bound), len(gal_inds),
                              len(ind_bound)/len(gal_inds)*100))
                
            if par['Monk']['ReturnBindingEnergy']:
                ind_bound = monk_res[0]
                binding_energy = monk_res[1]
            else:
                ind_bound = monk_res

            # Record result of unbinding for this galaxy:
            statList[gal_inds[ind_bound]] = 1
            boundFracs[iicen] = len(ind_bound)/len(gal_inds)
            numOrigParts[iicen] = len(gal_inds)

            if len(ind_bound) > 0:
                # Set the potential minimum particle:
                self.snap.target_galaxies.set_potMin(
                    igal, gal_inds[ind_bound[0]])

                # Record final subhalo position and velocity as used by MONK:
                self.snap.target_galaxies.set_final_coordinates(
                    igal, sat.pos, sat.vel)
            
                # Set binding energy, if desired (also works for post-sat
                # unbinding, since we only consider particles in the cen)
                if par['Monk']['ReturnBindingEnergy']:
                    self.binding_energy[gal_inds[ind_bound]] = (
                        binding_energy[ind_bound])

            else:
                self.snap.target_galaxies.add_lost_cen(igal)
                if par['Verbose']:
                    print("Added galaxyID {:d} as lost central."
                          .format(igal))

        # Ends loop over individual centrals to unbind. Split (full)
        # particle set into bound and unbound:
        all_n_bound = np.count_nonzero(statList == 1)
        all_ind_unbound = np.nonzero(statList == 0)[0]
        
        # We do NOT remove unbound particles from the list: they may still 
        # be bound to a satellite (the potential is always lower, but 
        # the peculiar velocity may be lower too).
        # --> Simply set to 'unassigned' for now.
        self.galaxy[all_ind_unbound] = -1
            
        unbind_endTime = time.time()
        print("Out of {:d} particles, {:d} were initially marked\n"
              "as in centrals (= {:.2f}%)" 
              .format(len(self.galaxy), nPartCenTotal, 
                      nPartCenTotal/len(self.galaxy)*100))
        print("After unbinding, {:d}/{:d} particles (= {:.2f}%) remain bound."
              .format(all_n_bound, len(self.galaxy), 
                      all_n_bound/len(self.galaxy)*100))
        print("Unbinding centrals took {:.2f} min."
              .format((unbind_endTime - unbind_startTime)/60.0))
    
    def build_tree(self):                        # Class: TargetParticles
        """
        Construct a cKDTree from all internally loaded particles.

        This will be used later to identify particles that lie close
        to the centre of each galaxy.

        Returns:
        --------
        build_time : float
            The time (in seconds) taken to build the tree (for logging).
        """
        
        print("Building a cKDTree of all {:d} particles..."
              .format(self.pos.shape[0]), end = '')
        stime = time.time()
        self.particleTree = cKDTree(self.pos)
        build_time = time.time() - stime
        print(" done ({:.2f} sec.)." .format(build_time))
        return build_time
    
                                                 # Class: TargetParticles
    def unbind_sat_from_central(self, igal, galaxies, ref_snap, galVerbose):
        """
        Unbind one specific satellite galaxy from its central.

        Parameters:
        -----------
        igal : int
            The galaxyID of the satellite to unbind.
        galaxies : Galaxies instance
            The collection of galaxies in the target snapshot.
        ref_snap : Snapshot instance
            The reference snapshot in which igal is unbound.
        """

        ref_snap.timeStamp.start_time()
        stime = time.time()
        
        # Set up Galaxy instance and load all possible source particles: 
        sat = Galaxy(igal, ref_snap, self, galVerbose)
        sat.add_external_particles()
        ref_snap.timeStamp.increase_time(index=ref_snap.tsInds[0])

        sat.lookup_particles()
        ref_snap.timeStamp.increase_time(index=ref_snap.tsInds[1])        
        
        # Don't need to go on if there are no particles loaded:
        if len(sat.part_inds) == 0:
            if self.snap.shi[igal] >= 0:
                self.snap.target_galaxies.add_lost_sat(igal)
            if sat.verbose:
                print("Galaxy {:d} has no particles to start with -- done."
                      .format(igal))
            return

        sat.find_target_coordinates()
        if np.min(sat.pos) < 0:
            print("Abandoning galaxy {:d}, since its position could "
                  "not be determined..." .format(igal))
            if self.sim.shi[igal, self.snap.isnap] >= 0:
                print("Should not happen for SF-identified galaxies!")
                set_trace()
            # Do not need to add to any tracking lists, since the galaxy
            # must have been SF-dead and so would only be counted as 
            # resuscitated if it had been recovered.
            return

        ref_snap.timeStamp.increase_time(index=ref_snap.tsInds[2])        

        # Test particles for gravitational binding status with MONK:
    
        monk_res = self.monk(sat.part_inds, sat.pos, sat.vel, 
                             galaxies.targetSnap.hubble_z.value,
                             fixCentre = par['Monk']['FixCentre'], 
                             centreMode = par['Monk']['Centering'],
                             monotonic = par['Monk']['Monotonic'],
                             resLimit=galaxies.targetSnap.epsilon,
                             potErrTol = par['Monk']['PotErrTol'], 
                             returnBE = par['Monk']['ReturnBindingEnergy'],
                             verbose = par['Monk']['Verbose'])

        if par['Monk']['ReturnBindingEnergy']:
            ind_bound = monk_res[0]
            binding_energy = monk_res[1]
        else:
            ind_bound = monk_res

        ref_snap.timeStamp.increase_time(index=ref_snap.tsInds[3])        

        n_bound = len(ind_bound)
        if sat.verbose and len(sat.part_inds):
            print("Determined that {:d}/{:d} (={:.2f} per cent) of "
                  "particles remain bound."
                  .format(n_bound, len(sat.part_inds), 
                          n_bound / len(sat.part_inds) * 100))

        if ((sat.target_shi >= 0 and 
             n_bound > par['Galaxies']['Threshold']['All'])
            or (sat.target_shi < 0 and 
                n_bound > par['Galaxies']['Threshold']['Recovered'])):

            sat.print_bound_particle_statistics(ind_bound)
            n_updated = self.update_satellite_particles(sat, monk_res)

            # Set the potential minimum particle:
            self.snap.target_galaxies.set_potMin(
                igal, sat.part_inds[ind_bound[0]])
            
            # Record final subhalo position and velocity as used by MONK:
            self.snap.target_galaxies.set_final_coordinates(
                igal, sat.pos, sat.vel)

            if sat.verbose:
                print("Updated {:d} particles to {:d}."
                      .format(n_updated, sat.igal))
            # If the galaxy was lost by SF, add to 'resuscitated':
            # (but only if some particles were actually updated!)
            if self.snap.shi[igal] < 0 and n_updated:
                self.snap.target_galaxies.add_resuscitated(igal)

        else:
            if sat.verbose:
                print("Did not find any bound remnant of galaxy {:d}..."
                      .format(sat.igal))
            n_updated = 0
        if n_updated == 0:
            if sat.verbose:
                print("Did not update any particles to galaxy {:d}..."
                      .format(sat.igal))
            
            # If the galaxy is in the SF catalogue, add it as 'lost':
            if self.snap.shi[igal] >= 0:
                self.snap.target_galaxies.add_lost_sat(igal)
            
        if sat.verbose:
            print("Finished satellite {:d} in {:.2f} sec."
              .format(igal, time.time()-stime))

        ref_snap.timeStamp.increase_time(index=ref_snap.tsInds[4])        
        del sat

                                                 # Class: TargetParticles
    def update_satellite_particles(self, galaxy, monk_res):
        """
        Update particles' galaxy tag where permissible.

        Any particles that already have a galaxy tag with a stronger 
        association than for the current galaxy are not updated.

        Parameters:
        -----------
        galaxy : Galaxy instance
            The galaxy whose  bound particles should be updated.
        monk_res : ndarray (int) or list
            If binding energies are to be written, this is a list of two
            arrays: ind_bound (the indices of the galaxy's particles that 
            have been identified as bound to the galaxy and that should 
            now be updated), and binding_energy (the BE of all particles of
            the galaxy, which is only meaningful for bound particles).
            If no binding energy is to be written, this is just ind_bound.

        Returns:
        --------
        len_reassociate : int
            Number of particles that were reassociated to the galaxy.
        """

        # Decompose input depending on parameter setting:
        if par['Monk']['ReturnBindingEnergy']:
            ind_bound = monk_res[0]
            binding_energy = monk_res[1]
        else:
            ind_bound = monk_res

        # Construct full (internal) indices of to-be-tested particles:
        indices = galaxy.part_inds[ind_bound]

        # Determine current 'origin status' of particles, i.e. how they 
        # were associated to their currently tagged galaxy (if any):
        curr_origin = self.sourceType[indices]
        curr_distance = self.distance[indices]
        curr_lastSeen = self.lastSeen[indices]
        curr_gal = self.galaxy[indices]

        # Determine origin status w.r.t. current galaxies:
        new_origin = galaxy.origins[ind_bound]
        if par['Sources']['Centre']:
            new_distance = galaxy.distances[ind_bound]
        else:
            new_distance = np.inf   # Dummy

        # Snapshot in which 'prior Cantor' membership was evaluated:
        if par['Sources']['Prior']:
            if not 'isnap_prior' in dir(galaxy): set_trace()
            isnap_lookup = galaxy.isnap_prior
        else:
            isnap_lookup = -1

        # Test for which bound (!) particles re-association is allowed:
        ind_reassociate = np.nonzero(
            # i) Better origin code
            (new_origin < curr_origin) |   

            # ii) Same origin code and...
            ((new_origin == curr_origin) & (
            # ii.a) ... closer distance if by centrality (5):
                ((new_origin == 5) & (new_distance < curr_distance))  
                                              
            # ii.b) ... *later* association if by prior Cantor membership
                | ((new_origin == -1) & (isnap_lookup > curr_lastSeen))
            
            # ii.c) ... earlier-deceased galaxy if by mergers (1)
                | ((new_origin == 1) & (self.sim.lastsnap[curr_gal] >
                                        self.sim.lastsnap[galaxy.igal]))
            )))[0]
        
        # Test which galaxies the to-be-updated particles are currently in, 
        # and subtract them to keep their total up to date.
        old_gal_ids, num_in_gal = np.unique(
            self.galaxy[indices[ind_reassociate]], return_counts = True) 
        self.update_particle_numbers(old_gal_ids, num_in_gal, galaxy.igal)

        # Do the update for eligible particles:
        self.galaxy[indices[ind_reassociate]] = galaxy.igal
        self.sourceType[indices[ind_reassociate]] = (
            new_origin[ind_reassociate])
        if par['Sources']['Centre']:
            self.distance[indices[ind_reassociate]] = (
                new_distance[ind_reassociate])
        self.lastSeen[indices[ind_reassociate]] = isnap_lookup
        
        # If desired, update binding energies:
        if par['Monk']['ReturnBindingEnergy']:
            self.binding_energy[indices[ind_reassociate]] = binding_energy[
                ind_bound[ind_reassociate]]

        return len(ind_reassociate)
                                                 # Class: TargetParticles
    def unbind_sats_in_refSnap(self, galaxies, ref_snap):
        """
        Unbind satellite galaxies in a particular reference snapshot.

        Parameters:
        ----------
        galaxies : Galaxies instance
            The galaxies to be considered in current target snapshot.
        ref_snap : Snapshot instance
            The current reference snapshot.
        """

        stime = time.time()
        ind_ref = np.nonzero(galaxies.sat_refSnap == ref_snap.isnap)[0]
        print("Now processing {:d} galaxies from snapshot {:d}..."
              .format(len(ind_ref), ref_snap.isnap))

        ref_snap.timeStamp.set_time('Finding ref-galaxies')

        # Set up time cumulative time counters for per-galaxy steps:
        ref_snap.tsInds = ref_snap.timeStamp.add_counters(
            ("Load particles", "Lookup particles", "Find coordinates", 
             "Monk", "Update particles"))

        # Loop through galaxies to be processed in this step:
        for iigal, igal in enumerate(galaxies.sat_candidates[ind_ref]):
            if par['Verbose'] or iigal % par['Check']['ReportFrequency'] == 0:
                eprint("Processing galaxy {:d}/{:d} (ID {:d})"
                       .format(iigal, len(ind_ref), igal), linestyle = '* * ')
            # Determine whether galaxy should produce verbose output:    
            if par['Verbose'] or iigal % par['Check']['ReportFrequency'] == 0:
                galVerbose = True
            else:
                galVerbose = False
            self.unbind_sat_from_central(igal, galaxies, ref_snap, galVerbose)

    def find_ngb_particles(self, igal, radius):  # Class: TargetParticles
        """
        Find all particles that lie near a given galaxy.

        Parameters:
        -----------
        igal : int
            The galaxyID around which to search.
        radius : float
            The maximum distance of a neighbour from the galaxy.
        
        Returns:
        --------
        ngb_ids : ndarray (int)
            The IDs of the neighbour particles.
        ngb_rad : ndarray (float)
            The distance of each neighbour from the galaxy centre.
        """
        
        # Coordinates of galaxy centre:
        galPos = self.snap.galPos[igal, :]
        
        # Find particles within this radius using the pre-built tree.
        # (must convert to ndarray to use as array slice later):
        preInd = np.array(self.particleTree.query_ball_point(galPos, radius))
        if len(preInd) == 0:
            return None, None    # Must be handled appropriately.

        # Safety check to isolate those particles really within target rad:
        deltaRad = np.linalg.norm((self.pos[preInd, :] 
                                   - galPos[None, :]), axis = 1)
        subInd = np.nonzero(deltaRad <= radius)[0]
        cenInd = preInd[subInd]

        # Now find the IDs of the 'central' particles. 
        # (in practice, this is technically unneccessary -- in the end
        # we need the indices. But going temporarily back to IDs allows 
        # direct combination with particles selected through other routes).
        ngb_ids = self.ids[cenInd].astype(int)
        
        return ngb_ids, deltaRad[subInd]
    
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
        cenGal = self.snap.cenGal     # "   "
        
        # Find locations of particles using pre-computed reverse list:
        all_indices = self.revID[ids]

        # Find the FOF-central galaxy for all these particles:
        # (explicitly use cenGal in case sats were not initialized to cen):
        all_fofCentral = cenGal[self.fof_cenGal[all_indices]]

        # Identify particles that could be matched (i.e. are in *any* FOF/SH)
        # and lie in same FOF as the galaxy (in the target snap):
        ind_found = np.nonzero((all_indices >= 0) 
                               & (all_fofCentral == galaxy.get_central()))[0]

        return all_indices[ind_found], ind_found

    def print_final_statistics(self):            # Class: TargetParticles
        """Print summary of bound particles after all unbinding."""

        # Find particles in galaxies (general), and satellites:
        self.ind_in_gal = np.nonzero(self.galaxy >= 0)[0]
        n_in_gal = len(self.ind_in_gal)
        if n_in_gal == 0:
            print("At the end of unbinding, no particles remain in a galaxy.")
            print("This is suspicious, you may want to investigate.")
            return

        cenGal = self.snap.target_galaxies.cenGalAssoc
        subInd_in_sat = np.nonzero(
            self.galaxy[self.ind_in_gal] != cenGal[
                self.galaxy[self.ind_in_gal]])[0]
        ind_in_sat = self.ind_in_gal[subInd_in_sat]
        n_in_sat = len(ind_in_sat)
 
        print("At the end of unbinding, {:d} particles are in galaxies,\n"
              "of which {:d} ({:.2f} per cent) are in satellites."
              .format(n_in_gal, n_in_sat, n_in_sat/n_in_gal*100))

        # Sanity check to make sure all satellite particles have an origin:
        if np.max(self.sourceType[ind_in_sat]) > 4:
            print("Why are there satellite particles without a valid "
                  "source type?")
            set_trace()
            
        # Print the distribution of origin codes for satellite particles:
        numBySource, bE = np.histogram(self.sourceType[ind_in_sat], 
                                       bins = 6, range = [-1, 5]) 

        print("")
        print("Origin of satellite particles:")
        names = ['Prior', 'Ref-snap', 'Mergers', 'Sats', 'Subfind', 'Centre']
        sl = [max(len(str(numBySource[_i])), len(names[_i])) 
              for _i in range(6)]
        topLine = u"\u250C" + "-" * (np.sum(np.array(sl)) + 17) + u"\u2510" 
        divLine = u"\u251C" + "-" * (np.sum(np.array(sl)) + 17) + u"\u2524" 
        botLine = u"\u2514" + "-" * (np.sum(np.array(sl)) + 17) + u"\u2518" 
        print(topLine)
        print("| {:>{w0}} | {:>{w1}} | {:>{w2}} | {:>{w3}} | {:>{w4}} "
              "| {:>{w5}} |"
              .format(*names,
                      w0=sl[0],w1=sl[1],w2=sl[2],w3=sl[3],w4=sl[4],w5=sl[5]))
        print(divLine)
        print("| {:>{w0}} | {:>{w1}} | {:>{w2}} | {:>{w3}} | {:>{w4}} "
              "| {:>{w5}} |"
              .format(*numBySource,
                      w0=sl[0],w1=sl[1],w2=sl[2],w3=sl[3],w4=sl[4],w5=sl[5]))
        print(botLine)
        print("")

    def find_particle_numbers(self):             # Class: TargetParticles
        """
        Find the number of particles currently associated to each galaxy.
        """
        print("Measuring number of particles per galaxy...", end = '')
        self.numPartByGal, bE = np.histogram(
            self.galaxy, bins = self.sim.numGal, range = [0, self.sim.numGal])

        # Initialize a list for galaxies that are lost later on:
        self.lost_gals = set([])
        print(" done!")
                                                 # Class: TargetParticles
    def update_particle_numbers(self, galID, num, igal):
        """
        Reduce the stored particle numbers for specified galaxies.

        The counter for galaxy 'igal' is incremented by the total lost
        from the others. If any galaxy is reduced to zero particles,
        it is added to an internal list of 'lost' galaxies.

        Parameters:
        -----------
        galID : ndarray (int)
            The galaxy IDs whose particle numbers should be reduced.
        num : ndarray (int)
            The number of particles to subtract from each galaxy.
        igal : int
            The galaxy ID to which particles should be added.
        """

        # Find particles in actual galaxies:
        subInd = np.nonzero(galID >= 0)[0]

        if len(subInd) > 0:
            # Adjust stored particle numbers
            self.numPartByGal[galID[subInd]] -= num[subInd]
        
            # Check if any galaxy has been reduced to zero:
            if np.min(self.numPartByGal[galID[subInd]] < 0):
                print("Problem: we seem to have reduced a galaxy to less than "
                      "zero particles. Should not really happen.")
                set_trace()

            ind_num_zero = np.nonzero((self.numPartByGal[galID[subInd]] == 0) 
                                      & (num[subInd] > 0))[0]
            
            # Add eliminated galaxies to their set for bookkeeping:
            if len(ind_num_zero) > 0:
                eprint("{:d} galaxies were reduced to zero particles during "
                       "evaluation of galID {:d}." 
                       .format(len(ind_num_zero), igal), linestyle = '#')
                lostSet = set(galID[subInd[ind_num_zero]])
                self.lost_gals = self.lost_gals.union(lostSet)

                # Test whether any of these have been resuscitated:
                # if so, remove from BOTH sets (`dead again').
                lostResusc = lostSet & self.snap.target_galaxies.resuscitated
                if lostResusc:
                    print("{:d} resuscitated galaxies were eliminated:"
                          .format(len(lostResusc)))
                    print(lostResusc)
                    self.snap.target_galaxies.resuscitated = (
                        self.snap.target_galaxies.resuscitated 
                        - lostResusc)
                    self.lost_gals = (self.lost_gals - lostResusc)

        # And increase number of particles in 'gaining' galaxy:
        self.numPartByGal[igal] += np.sum(num)

    def get_potMin_pos_multi(self, galID):          # Class: TargetParticles
        """
        Find the potential minimum of the specified galaxies.

        This is found as the position of the particle with the lowest
        gravitational potential that is bound to the galaxy at the end of
        all unbinding.

        If no binding energies are recorded (because the galaxy is a central
        and we didn't unbind them), the input position from SF is used.

        Parameters:
        -----------
        galID : ndarray (int) [N]
            The galaxy IDs for which the potential centre should be found.

        Returns:
        --------
        cen : ndarray (float) [N, 3]
            The position of each galaxy's potential minimum.

        For the moment, this is implemented in a 'brute-force' way, 
        by looping through the galaxies.
        """

        cen = np.zeros((len(galID), 3)) + np.nan

        for iigal, igal in enumerate(galID):
            partInd = self.sim.output.part_lut_galaxy(igal)
            
            if len(partInd) == 0:
                print("Trying to look up potential minimum for "
                      "non-existing galaxy. Should not happen.")
                set_trace()

            ind_potMin = partInd[np.argmin(self.binding_energy[partInd])]
            
            if np.isnan(self.binding_energy[partInd[ind_potMin]]):
                cen[iigal, :] = self.snap.galPos[igal, :]
            else:
                cen[iigal, :] = self.pos[ind_potMin, :]

    def calculate_radii(self, partInd, galCen):     # Class: TargetParticles
        """ 
        Calculate the distance of specified particles from a given centre.

        Parameters:
        -----------
        partInd : ndarray (int)
            The indices into the internal particle array for which radii
            should be calculated.
        galCen : ndarray (float) [3]
            The centre relative to which distances should be calculated.
        """
        
        deltaPos = self.pos[partInd, :] - galCen[None, :]
        self.rad[partInd] = np.linalg.norm(deltaPos, axis = 1)
    
    def calculate_all_radii(self):               # Class: TargetParticles
        """
        Calculate radii of all particles from their galaxy centres.
        
        The centres are taken from self.sim.output.sh_centreOfPotential,
        which is already computed by the time this is called.

        Updates:
        --------
        self.rad --> Set to radius (-1 for particles not in a galaxy).
        """

        # Initialize radius array -- NaN indicates 'not-in-a-galaxy'.
        # Note that this works in sorting as of numpy v.1.4.0.
        self.rad = np.zeros_like(self.mass) + np.nan

        # Extract galaxy centres
        cenGal = self.snap.output.sh_centreOfPotential[
            self.sim.output.shiExtended[self.galaxy[self.ind_in_gal], 
                                        self.snap.isnap], :]
        self.rad[self.ind_in_gal] = np.linalg.norm(
            self.pos[self.ind_in_gal, :] - cenGal, axis = 1)
                
    
class PriorCantorData:
    """Subhalo--particle tables for last N snapshots from Cantor.""" 

    def __init__(self, targSnap):
        """Load the tables for a particular target snapshot."""
        
        # Initialize as None, to hold dummy element for current snap.
        self.ids_sh_back = [None]
        self.off_sh_back = [None]
        self.len_sh_back = [None]
        cantorFile = targSnap.sim.output.outloc

        for iback in range(1, par['Sources']['MaxPriorCantor'] + 1):
            isnap_load = targSnap.isnap - iback
            if isnap_load < 0: break
            if not par['Lost']['Recover'] and iback > 1: break

            self.ids_sh_back.append(yb.read_hdf5(
                cantorFile, 'Snapshot_{:03d}/IDs' .format(isnap_load)))
            self.off_sh_back.append(yb.read_hdf5(
                cantorFile, 'Snapshot_{:03d}/Subhalo/Offset' 
                .format(isnap_load)))
            self.len_sh_back.append(yb.read_hdf5(
                cantorFile, 'Snapshot_{:03d}/Subhalo/Length'
                .format(isnap_load)))

        targSnap.set_priorCantorData(self)

    def get_sh_ids(self, iback, shi):            # Class: PriorCantorData 
        """
        Retrieve the particle IDs for a given subhalo in a prior snapshot.
        
        Parameters:
        -----------
        iback : int
            Number of snapshot steps back from target_snap that IDs 
            should be extracted (e.g.: if target_snap = 10 and we want 
            particles from snap 8, this is 2).
        shi : int
            (Cantor) subhalo index of the galaxy to extract particles for,
            in the target snapshot specified by iback.
        
        Returns: 
        --------
        ids : ndarray (int) [N]
            The N particle IDs of the subhalo, as an int64 array.
        """

        # Consistency checks 
        if iback > par['Sources']['MaxPriorCantor']:
            print("Inconsistent back-snapshot numbers -- investigate.")
            set_trace()
        if shi >= len(self.off_sh_back[iback]): 
            set_trace()

        ids = self.ids_sh_back[iback]
        offset = self.off_sh_back[iback][shi]
        length = self.len_sh_back[iback][shi]
        return (ids[offset : offset+length]).astype(int)


def process_sim_snapshot(isnap_proc, sim, sim_output):
    """
    Process one snapshot of a simulation through Cantor.

    This function is the `fundamental unit' of the code. It establishes
    which satellites can (attempt to) be unbound and finds their
    respective reference snapshots, loads the target snapshot data,
    and then iterates over individual ref snaps for unbinding.
    
    Parameters:
    -----------
    isnap_proc : int
        The index of the `target' snapshot to process.
    sim : class Simulation instance
        The simulation to which this snapshot belongs.
    sim_output : class CantorOutput instance
        The output class for writing the results.
    """

    # Initialize time stamp for this snapshot:
    timeStamp = TimeStamp()

    eprint("Processing snapshot {:d} of sim {:d}"
           .format(isnap_proc, sim.isim))
    
    target_snap = Snapshot(sim, isnap_proc, is_target_snap = True)
    target_snap.load_catalogue_data()
    target_snap.load_fof_index(None)
    target_snap.load_sh_coordinates()

    # Load data for converting particle IDs to subhalo indices:
    if par['Input']['FromSnapshot']:
        target_snap.load_sh_data_from_snapshot()
    else:
        target_snap.load_sh_data_from_esp()

    # Pre-load data from prior cantor outputs (may get moved):
    if par['Lost']['Recover'] or par['Sources']['Prior']:
        priorCantorData = PriorCantorData(target_snap)

    timeStamp.set_time('Setup')
    setupTime = timeStamp.get_time()

    # ===============================================================
    # 0.) Identify satellite galaxies to be unbound, and their 
    #     order of unbinding (generally their last-central snapshot)
    # ===============================================================
    
    # Basic setup includes all SF-identifications in target snap:
    candidate_galaxies = Galaxies(target_snap)

    # If desired, add temporarily lost galaxies (code -9 in SHI):
    if par['Lost']['FindTemporarilyLost']:
        candidate_galaxies.add_temporarily_lost()

    # If desired, add galaxies that were recently fully lost.
    # The idea here is that, close to the end of the simulation, we do         
    # not know whether these will re-surface, so assume they will.
    # For testing, this can be extended to also be applied to target
    # snapshots that are not close to the simulation end -- there should
    # not be many galaxies recovered there.
    # In snapshot 0, there cannot be any lost galaxies.
    if par['Lost']['FindPermanentlyLost'] and isnap_proc > 0:
        if (par['Snaps']['Num'] - isnap_proc <= 5 
            or par['Lost']['FindInAllSnaps']):
            candidate_galaxies.add_permanently_lost()
 
    # Now decide which galaxies are treated as centrals and satellites.
    # All non-SF-alive galaxies are considered as satellites.
    candidate_galaxies.assign_cen_sat()

    # If desired, remove spectres from the galaxy list:
    if par['Galaxies']['DiscardSpectres']:
        candidate_galaxies.remove_spectres()
    
    # Determine the reference snapshot for each satellite:
    candidate_galaxies.find_reference_snapshots()

    # Set up snapshot-specific output (pt. 1)
    candidate_galaxies.setup_centre_arrays()


    timeStamp.set_time('Finding target galaxies')
    setupTime += timeStamp.get_time()

    print_memory_usage()
    eprint(" --- Done with first stage of setup ({:.2f} min.) --- "
           .format(setupTime/60.0))
        
    # ============================================================
    # 1.) Load particles (ID, mass, pos, vel, int. energy, type) 
    #     and SH information in target snapshot, invert ID list. 
    # ============================================================

    target_particles = TargetParticles(target_snap)
              
    # Now load actual particle data:    
    target_particles.load_particle_data()
    timeStamp.set_time('Load particle data')

    print_memory_usage()
    eprint("Loading particle data from files took {:.2f} min."
           .format(timeStamp.get_time()/60.0), linestyle = '-')

    # Reject particles that are not part of a FOF or subhalo:
    target_particles.reject_unassociated()

    # Now set up arrays to hold particle origin/association information:
    target_particles.create_origin_arrays()

    # ... and set up ID lookup list for later
    target_particles.create_id_lookup_list()

    # If we include particles near the centre, set up a tree to find them:
    if par['Sources']['Centre'] is not None:
        counter_treeBuild = target_particles.build_tree()

    # If desired, unbind NOW particles from centrals. Also need to make
    # a copy of (pre-unbinding) cens for all particles, to (later) 
    # associate particles to FOF groups (including cen-unbound ones).
    target_particles.save_full_cens()

    timeStamp.set_time('Processing particle data')

    if par['Galaxies']['UnbindCensWithSats']:
        target_particles.unbind_cens(candidate_galaxies)
        timeStamp.set_time('Unbinding centrals (prior)')
        
    # Need to make a list of number of particles per galaxy:
    target_particles.find_particle_numbers()
    timeStamp.set_time('Finding initial particle numbers')

    # Now loop backwards over snapshots to unbind satellites:    
    # If we don't want to do that, end before we've begun
    if par['Galaxies']['UnbindSats']:
        endSnap = -1
    else:
        endSnap = isnap_proc + 1

    unbind_stime = time.time()
    for isnap in range(isnap_proc, endSnap, -1):

        print_memory_usage("At start of ref-snap {:d}: " 
                           .format(isnap))

        # Start ref-snap time-stamp
        ref_timeStamp = TimeStamp()

        # Set up snapshot instance for current (reference) snapshot:
        # This differs depending on whether we're looking at the target
        # snapshot itself, or a prior one.
        if isnap == isnap_proc:
            ref_snap = target_snap
            # Cannot load from Cantor in this case -- snap not done yet!
            ref_snap.find_alive_galaxies(sim_output)

        else:
            # Ref is NOT target snapshot -- need to set up from scratch
            ref_snap = Snapshot(sim, isnap)
            ref_snap.load_catalogue_data()
            ref_snap.find_alive_galaxies(
                sim_output, from_cantor = par['Input']['FromCantor'])

            # Load SH (&FOF) -- particle associations:
            if par['Input']['FromCantor']:
                ref_snap.load_cantor_sh_particle_data(
                    sim_output, includeFOF=par['Sources']['FOF'])
            else:
                ref_snap.load_sf_sh_particle_data(
                    includeFOF=par['Sources']['FOF'])
            # If needed, load FOF index of subhaloes 
            # (already loaded for target_snap):
            if par['Sources']['FOF']:
                ref_snap.load_fof_index(
                    sim_output, from_cantor = par['Input']['FromCantor'])

        ref_timeStamp.set_time('Load data')

        # If needed, set up a way to find satellites in ref:
        if par['Sources']['Sats']:
            ref_snap.build_satellite_list(sim_output)

        # If needed, set up a way to find merging galaxies in ref:
        if par['Sources']['Mergers']:
            ref_snap.build_merger_list(target_snap)

        ref_timeStamp.set_time('Galaxy setup')
        ref_snap.timeStamp = TimeStamp()

        # Do satellite unbinding for reference snapshot:
        print_memory_usage("Immediately before sat unbinding: ")
        target_particles.unbind_sats_in_refSnap(candidate_galaxies, ref_snap)
        print_memory_usage("Immediately after sat unbinding: ")

        ref_timeStamp.import_times(ref_snap.timeStamp)
        ref_timeStamp.set_time('Unbinding')
        ref_timeStamp.print_time_usage("Finished ref snapshot {:d}"
                                       .format(ref_snap.isnap),
                                       minutes=False,percent=False)

        # Transfer internal time counters to target_snap-wide counter:
        timeStamp.import_times(ref_timeStamp)
        
        del ref_snap
        gc.collect()
            
    # Done processing individual reference snapshots!
    print_memory_usage("At end of satellite unbinding: ")
    timeStamp.set_time('Unbinding satellites')

    # If desired, unbind NOW particles from centrals.
    if par['Galaxies']['UnbindCensAfterSats']:
        target_particles.unbind_cens(candidate_galaxies)
        timeStamp.set_time('Unbinding centrals (post)')

    # Print statistics about post-unbinding particles & source types:
    target_particles.print_final_statistics()

    # Set up snapshot-specific output instance:
    snap_output = SnapshotOutput(target_snap, order_cens_first=True)

    # Compute 'cantorID' index for all galaxies that have at least one
    # particle belonging to it after all the unbinding.
    snap_output.compute_cantorID()
    snap_output.validate_subhaloes()
    sim_output.update_shiX(snap_output)

    # Compute centre-of-potential for each galaxy, and particles' radii:
    snap_output.compute_centre_and_radii()

    # Establish the final output sequence of particles, and the 
    # various offset lists
    snap_output.arrange_particles()

    # Compute quantities of galaxies after particle re-assignment:
    print_memory_usage("Before galaxy property calculation: ")
    snap_output.setup_snapshot_outputs()
    snap_output.compute_galaxy_properties()
    
    # Write results to disk, including (partial) simulation-wide arrays:
    snap_output.write()
    sim_output.write(target_snap)
    print_memory_usage("After galaxy property calculation: ")
    
    timeStamp.import_times(snap_output.timeStamp)
    timeStamp.set_time('Preparing and writing output')

    # Print time usage statistics for this (target) snapshot:
    timeStamp.print_time_usage(
        "Finished processing snapshot {:d} of sim {:d}"
        .format(isnap_proc, sim.isim), minutes = True)

    # Last step: transfer snap's time markers to simulation-wide counters.
    sim.timeStamp.import_times(timeStamp)

    # Explicitly clean up snapshot variables, to avoid memory leaks
    print_memory_usage("End of target-snap {:d}, before cleaning: "
                       .format(isnap_proc))

    target_snap.set_galaxies(None)
    target_snap.set_priorCantorData(None)
    target_snap.set_particles(None)
    del target_snap
    del target_particles
    del candidate_galaxies
    del priorCantorData
    del snap_output
    gc.collect()

    print_memory_usage("After clean-up: ")

if __name__ == '__main__':
    main()
