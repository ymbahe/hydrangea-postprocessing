"""
Estimate the positions of galaxies at high cadence through interpolation.

The starting point is the GalaxyPaths catalogue, which includes positions
in the snipshots; the exact input sneplist can be specified. Because of the 
high number of outputs per galaxy, this is only done for reasonably massive
galaxies (exact definition of which is adjustable).

The output is stored in one HDF5 file per simulation, in [freya]/highlev/, 
and includes reference pointers to/from the galaxy IDs:

Galaxy         --> galaxy IDs of the followed galaxies
GalaxyRevIndex --> index in this catalogue of galaxy ID i. -1 if not followed. 
[InterpolatedMsub] --> Msub of followed galaxies [N_gal, N_time], if enabled.
InterpolatedPositions --> Positions of followed galaxies [N_gal, 3, N_time]
InterpolationTimes --> Age of the Universe at the N_time interpolation times.

Started 28-Feb-2018
"""


import sim_tools as st
import yb_utils as yb
import hydrangea_tools as ht
import eagle_routines as er

import numpy as np
from astropy.io import ascii
from pdb import set_trace
import time
from mpi4py import MPI
import os
from astropy.cosmology import Planck13
import scipy.interpolate
import sys

# ==============================================================

simname = "Hydrangea"     # 'Hydrangea' or 'EAGLE'
runtype = "HYDRO"         # 'HYDRO' or 'DM'
baselist = "default_long" # input sneplist, lower-case first letter
Baselist = "Default_long" # input sneplist, upper-case first letter

delta_t = 0.01  # Interpolation cadence in Gyr

# Select which galaxies to include. The two mass thresholds are alternative
# (a galaxy satisfying either is included), the last is obligatory.

mmax_threshold = 10.0    # Minimum peak mass for inclusion [log10(M/M_sun)]
mmaxStar_threshold = 9.0 # Minimum peak stellar mass [log10(M/M_sun)] 
contFlag_threshold = 0   # Maximum contamination flag level for inclusion

include_msub = False # Interpolate not just position, but also Msub?

if simname == "Eagle":
    n_sim = 1
    basedir = '/virgo/simulations/Eagle/L0100N1504/REFERENCE/'
    snap_z0 = 28
    nsnap = 29
    snepAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/eagle_outputs_new.txt'
else:  
    n_sim = 30
    basedir = '/virgo/simulations/Hydrangea/10r200/'
    snap_z0 = 29
    nsnap = 30
    snepAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/hydrangea_snepshots_' + baselist + '.dat'
    snapAexpLoc = '/freya/ptmp/mpa/ybahe/HYDRANGEA/OutputLists/hydrangea_snepshots_allsnaps.dat'

outname = 'highlev/GalaxyCoordinates10Myr_May18.hdf5'

# ============================================================

# Set up MPI, to exploit embarrasing parallelism
comm = MPI.COMM_WORLD
numtasks = comm.Get_size()
rank = comm.Get_rank()

# Read 
snep_aexp = np.array(ascii.read(snepAexpLoc, format = 'no_header', guess = False)['col1'])
snap_aexp = np.array(ascii.read(snapAexpLoc, format = 'no_header', guess = False)['col1'])

snep_zred = 1/snep_aexp - 1
snep_time = Planck13.age(snep_zred).value
time_s0 = snep_time[0]
time_z0 = Planck13.age(0).value

snap_zred = 1/snap_aexp - 1
snap_time = Planck13.age(snap_zred).value

for isim in range(0, 30):

    # Skip this one if we are multi-threading and it's not for this task to worry about
    if not isim % numtasks == rank:
        continue
        
    hstime = time.time()

    print("")
    print("*****************************")
    print("Now processing halo CE-{:d}" .format(isim))
    print("*****************************")
    print("")

    sys.stdout.flush()

    if simname == "Eagle":
        rundir = basedir
        outloc = er.clone_dir(rundir) + '/' + outname
        hldir = er.clone_dir(rundir, loc = 'virgo') + '/highlev/'
    else:
        rundir = basedir + '/HaloF' + str(isim) + '/' + runtype
        outloc = ht.clone_dir(rundir, loc = 'freya') + '/' + outname
        hldir = ht.clone_dir(rundir, loc = 'freya') + '/highlev'

    if not os.path.exists(rundir):
        continue

    if not os.path.exists(yb.dir(outloc)):
        os.makedirs(yb.dir(outloc))

    pathloc = hldir + '/GalaxyPathsMay18.hdf5'
    fgtloc = hldir + '/FullGalaxyTablesMay18.hdf5'

    mmax = yb.read_hdf5(fgtloc, 'Full/Msub')
    mmaxStar = yb.read_hdf5(fgtloc, 'Full/Mstar')
    contFlag = yb.read_hdf5(fgtloc, 'Full/ContFlag')[:, 2]

    if include_msub:
        msub = yb.read_hdf5(fgtloc, 'Msub')

    gal_sel = np.nonzero(((mmax >= mmax_threshold) | (mmaxStar >= mmaxStar_threshold)) & (contFlag <= contFlag_threshold))[0]
    ngal = len(gal_sel)

    gal_revInd = np.zeros(contFlag.shape[0], dtype = np.int32)-1
    gal_revInd[gal_sel] = np.arange(ngal, dtype = np.int32)

    print("There are {:d} galaxies with mmax >= 10.0 or mmaxStar >= 9.0 in sim {:d}..."
          .format(ngal, isim))

    snep_rootind = yb.read_hdf5(pathloc, 'RootIndex/' + Baselist)
    nsnep = snep_rootind.shape[0]

    full_pos_snep = np.zeros((ngal, 3, nsnep))

    print("Now loading positions from individual snepshots...")

    for iisnep, isnep in enumerate(snep_rootind):
        pos = yb.read_hdf5(pathloc, 'Snepshot_' + str(isnep).zfill(4) + '/Coordinates')[gal_sel, :]
        aexp_factor = yb.read_hdf5_attribute(pathloc, 'Snepshot_' + str(isnep).zfill(4) + '/Coordinates', 'aexp-factor')
        h_factor = yb.read_hdf5_attribute(pathloc, 'Snepshot_' + str(isnep).zfill(4) + '/Coordinates', 'h-factor')
        pos *= (aexp_factor*h_factor)

        full_pos_snep[:, :, iisnep] = pos

    print("Finished loading snepshot positions...")
    
    # Perform actual interpolation:
    time_fine = np.arange(time_s0, time_z0, delta_t) 
    print("There are {:d} interpolated outputs between {:.3f} and {:.3f} Gyr"
          .format(len(time_fine), time_fine[0], time_fine[-1]))

    csi = scipy.interpolate.interp1d(snep_time, full_pos_snep, kind = 'cubic', axis = 2, assume_sorted = True)
    full_pos_fine = csi(time_fine)

    yb.write_hdf5(full_pos_fine, outloc, "InterpolatedPositions", new = True)
    yb.write_hdf5(time_fine, outloc, "InterpolationTimes")
    yb.write_hdf5(gal_sel, outloc, "Galaxy")
    yb.write_hdf5(gal_revInd, outloc, 'GalaxyRevIndex')
    
    if include_msub:
        csi_mass = scipy.interpolate.interp1d(snap_time, msub[gal_sel, :], kind = 'cubic', axis = 1, assume_sorted = True)
        full_mass_fine = csi_mass(time_fine)
        yb.write_hdf5(full_mass_fine, outloc, "InterpolatedMsub")


print("Done!")
