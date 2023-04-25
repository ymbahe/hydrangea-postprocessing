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

25-APR-2023: further updates

"""

import hydrangea as hy
from pythontools import TimeStamp
import numpy as np
import scipy.interpolate
from pdb import set_trace
from astropy.cosmology import Planck13
from astropy.io import ascii
import os.path

datestamp = '25Apr23'

simtype = 'HYDRO'
simname = "Hydrangea"
include_ceo = False

min_logmsub_peak = 1000.0
min_logmstar_peak = 9.0

outloc = (f'/net/quasar/data3/Hydrangea/'
          f'FullGalaxyCatalogue_{simname}_{datestamp}_{simtype}.hdf5')

if simname == "Eagle":

    raise NameError("EAGLE is currently not fully supported.")

    basedir_hl = '/net/amsteldiep/data2/bahe/FROM_VIRGO_SCRATCH/EAGLE/'
    snapAexpLoc = ('/net/amsteldiep/data2/bahe/FROM_VIRGO_SCRATCH/HYDRANGEA/'
                   'OutputLists/eagle_outputs_new.txt')
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
    basedir = '/net/quasar/data3/Hydrangea/10r200/'
    snapAexpLoc = ('/net/amsteldiep/data2/bahe/FROM_VIRGO_SCRATCH/HYDRANGEA/'
                   'OutputLists/hydrangea_snepshots_allsnaps.dat')
    nsnap = 30
    n_sim = 30
    isnap_z0 = 29
    
# Calculate snapshot times
snap_aexp = np.array(
    ascii.read(snapAexpLoc, format='no_header', guess=False)['col1']
)
snap_zred = 1/snap_aexp - 1
snap_time = Planck13.age(snap_zred).value

# Specify quantities to go into the catalogue...
# (i) From FullGalaxyTables:
quants_fgt = [
    ('SubhaloIndices', 'SHI', np.int32, 'Indices of the corresponding subhalo '
     'in the Subfind catalogue. Negative values mean that this galaxy does not '
     'exist in this snapshot.'),
    ('VMax', 'Vmax', np.float32, 'Maximum circular velocity [km/s], '
     'calculated as the maximum of sqrt{GM(<r)/r}.'),
]
quants_gps = [
    ('Coordinates', 'Centre', np.float32, 'Coordinates of each galaxy [pMpc].'),
    ('Velocities', 'Velocity', np.float32, 'Velocity of each galaxy [km/s].'),
]

# (ii-a) From Subfind FOF catalogues:
quants_subfind_fof = [
    ('R200c', 'Group_R_Crit200', np.float32, 'Radius enclosing a density of '
     '200 times the critical density. For satellites, this refers to their '
     'central halo.'),
]

# (ii-b) From Subfind Subhalo catalogues:
quants_subfind_sub = [
]

# (iii) Derived quantities (not already in a catalogue):
quants_snapshot = [
]

quants_hydrangea_only = ['RRelNearest', 'RRelMean', 'MHNeutral', 'MHI']
quants_eagle_only = ['Volume', 'Model', 'Resolution', 'Boxsize']

quants_to_interpolate = [
    'MStar30kpc', 'MStar', 'MStarInitial', 'MSubhalo', 'VMax']

# ===========================================================================

quant_lists = [
    quants_fgt, quants_gps, quants_subfind_fof, quants_subfind_sub,
    quants_snapshot]

# Remove quantities that are not applicable to the selected simulation type
for quant_list in quant_lists:
    for iquant, quant in enumerate(quant_list):
        if simname == 'Hydrangea' and quant[0] in quants_eagle_only:
            del quant_list[iquant]
        if simname == 'EAGLE' and quant[0] in quants_hydrangea_only:
            del quant_list[iquant]

# Prepare the full output lists. 'Simulation' and 'GalaxyID' are special...
full_data = {
    'Simulation': np.zeros(0, dtype=np.int8),
    'GalaxyID': np.zeros(0, dtype=np.int32)
}
comments = {
    'Simulation': 'Simulation index for each galaxy.',
    'GalaxyID': 'ID of each galaxy in its simulation.',
}

for quant_list in quant_lists:
    for quant in quant_list:
        name = quant[0]
        if name in full_data:
            raise KeyError(f"Data set {name} appears to be requested twice!")
        full_data[name] = np.zeros(0, dtype=quant[2])
        comments[name] = quant[3]

# Now loop through individual simulations
for isim in range(n_sim):

    if simname == 'Hydrangea':
        sim = hy.Simulation(isim)
        hldir = sim.high_level.dir
    else:
        sim = None
        hldir = f'{basedir_hl}{boxvec[isim]}/{modelvec[isim]}/highlev/'
        
    if not os.path.exists(hldir):
        continue

    if simname == 'Hydrangea': 
        print("")
        print("=============================")
        print(f"Processing simulation CE-{isim}"
        print("=============================")
        print("", flush=True)

        if isim in [17, 19, 20, 23, 26, 27] and include_ceo is False:
            continue
        if isim in [10, 27]:
            continue

    else:
        print("")
        print("=============================")
        print(f"Processing simulation EAGLE-{boxvec[isim]}/{modelvec[isim]}")
        print("=============================")
        print("", flush=True)

    fgtloc = hldir + '/FullGalaxyTables.hdf5'
    posloc = hldir + '/GalaxyPositionsSnap.hdf5'

    # First, identify galaxies
    logMsubMax = hy.hdf5.read_data(fgtloc, 'Full/Msub')
    logMstarMax = hy.hdf5.read_data(fgtloc, 'Full/Mstar')
    contFlag = hy.hdf5.read_data(fgtloc, 'Full/ContFlag')[:, 2]
    spectreFlag = hy.hdf5.read_data(fgtloc, 'Full/SpectreFlag')

    gal_sel = np.nonzero(
        (contFlag == 0) & (spectreFlag == 0) &
        (
            (logMsubMax >= min_logmsub_peak) |
            (logMstarMax >= min_logmstar_peak)
        )
    )[0]
    ngal_sel = len(gal_sel)
    print(f"In simulation {isim}, there are {ngal_sel} selected galaxies.")

    # Now extract data... FGT/GPS first:
    sim_data = {
        'GalaxyID': gal_sel,
        'Simulation': np.zeros(ngal_sel, dtype=np.int8) + isim,
    }    
    for quant in quants_fgt:
        name = quant[0]
        source = quant[1]
        sim_data[name] = hy.hdf5.read_data(fgtloc, source, read_index=gal_sel)
    for quant in quants_gps:
        name = quant[0]
        source = quant[1]
        sim_name[name] = hy.hdf5.read_data(gpsloc, source, read_index=gal_sel)

    # Interpolate over missing entries
    for iigal, igal in enumerate(gal_sel):        

        # Find snapshots in which the galaxy is hidden. If there are none: easy
        shi = sim_data['SubhaloIndices'][iigal, :]
        ind_hidden = np.nonzero(shi == -9)[0]
        if len(ind_hidden) == 0:
            continue

        # If we know one thing about hidden galaxies, then that they must be
        # satellites at these snapshots -- so mark them as such.        
        if 'SatelliteFlag' in sim_data:
            sim_data['SatelliteFlag'][igal, ind_hidden] = 1
        if 'EverSatelliteFlag' in sim_data:
            sim_data['EverSatelliteFlag'][igal] = 1

        # Check whether there are enough OK snapshots for a good interpolation
        ind_ok = np.nonzero(shi >= 0)[0]
        if len(ind_ok) < 2:
            continue
        if len(ind_ok) < 4:
            kind = 'linear'
        else:
            kind = 'cubic'

        for quant in quants_to_interpolate:
            try:
                csi = scipy.interpolate.interp1d(
                    snap_time[ind_ok], sim_data[quant][igal, ind_ok],
                    kind=kind, assume_sorted=True, fill_value='extrapolate'
                )
                sim_data[quant][igal, ind_hidden] = csi(snap_time[ind_hidden])
            # If the data set is not found, just ignore it
            except KeyError:
                pass
    
    # Append this simulation's data to the full output list
    for quant in full_data:
        full_data[quant] = np.concatenate((full_data[quant], sim_data[quant]))

# Done processing simulations, we now have the full output list. Write it.
for quant in full_data.keys():
    hy.hdf5.write_data(
        outloc, quant, full_data[quant], comment=comments[quant])

print("Done!")