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
import h5py as h5
from pythontools import TimeStamp
import numpy as np
import scipy.interpolate
from pdb import set_trace
from astropy.cosmology import Planck13
from astropy.io import ascii
import os

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
                   'OutputLists/hydrangea_snapshots_plus.dat')
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
    ('SubhaloIndices', 'SHI', np.int32, 2, 'Indices of the corresponding '
     'subhalo in the Subfind catalogue. Negative values mean that this galaxy '
     'does not exist in this snapshot.'),
    ('VMax', 'Vmax', np.float32, 2, 'Maximum circular velocity [km/s], '
     'calculated as the maximum of sqrt{GM(<r)/r}. It is interpolated over '
     'snapshots in which the galaxy is not detected (SubfindIndices < 0).'),
    ('SatelliteFlags', 'SatFlag', np.int8, 2, 'Flag indicating whether the '
     'galaxy is a central (0) or satellite (1) in each snapshot.'),
    ('EverSatelliteFlags', 'Full/SatFlag', np.int8, 1, 'Flag indicating '
     'whether the galaxy was always a central (0) or was a satellite in at '
     'least one snapshot (> 0).'),
    ('SubhaloMasses', 'Msub', np.float32, 2, 'Total mass of the subhalo in '
     'each snapshot, as calculated by Subfind [M_Sun]. This is interpolated '
     'over snapshots in which the galaxy is not detected '
     '(SubfindIndices < 0).'),
    ('M200c', 'M200', np.float32, 2, 'Mass of the host halo within R200c '
     '[M_Sun]. For satellites, this refers to their host halo.'),
    ('R200c', 'R200', np.float32, 2, 'Radius enclosing a density of 200 times '
     'the critical density [pMpc]. For satellites, this refers to their host '
     'halo.'),
    ('StellarMasses', 'Mstar', np.float32, 2, 'Total stellar mass [M_Sun]. '
     'This is interpolated over snapshots in which the galaxy is not '
     'detected (SubhaloIndices < 0).'),
    ('StellarMasses30kpc', 'Mstar30kpc', np.float32, 2,
     'Stellar mass within a spherical aperture of 30 pkpc [M_Sun]. This is '
     'interpolated over snapshots in which the galaxy is not detected '
     '(SubhaloIndices < 0).'),
    ('InitialStellarMasses', 'MstarInit', np.float32, 2,
     'Total initial stellar mass [M_Sun], i.e. not accounting for mass loss '
     'due to stellar evolution. This is interpolated over snapshots in '
     'which the galaxy is not detected (SubhaloIndices < 0).'),
    ('GasMasses', 'MGas', np.float32, 2, 'Total gas mass [M_Sun]. Note that '
     'this includes all phases, including diffuse hot gas.'),
    ('GasMasses30kpc', 'Mgas30kpc', np.float32, 2, 'Gas mass within a '
     'spherical aperture of 30 pkpc [M_Sun].'),
    ('StarFormationRates', 'SFR', np.float32, 2, 'Instantaneous total star '
     'formation rate [M_Sun/yr].'),
    ('StarFormationRates30kpc', 'SFR30kpc', np.float32, 2, 'Instantaneous '
     'star formation rate within a 30 pkpc radial aperture [M_Sun/yr].'),
    ('StellarHalfMassRadii', 'StellarHalfMassRad', np.float32, 2,
     'Radius enclosing half the total stellar mass of the galaxy, in 3D '
     '[Mpc].'),
    ('AtomicHydrogenMasses', 'MHI', np.float32, 2, 'Total mass of atomic '
     'Hydrogen (HI) in the galaxy [M_Sun].'),
    ('NeutralHydrogenMasses', 'MHneutral', np.float32, 2, 'Total mass of '
     'neutral Hydrogen in the galaxy [M_Sun].'),
    ('RelativeRadii', 'RelRadius', np.float32, 2, 'Distance to the nearest '
     '"significant" neighbour, in units of the R200c of this neighbour. '
     'For satellites, this is the their central galaxy. For centrals, it is '
     'the nearest other central with a higher M200c than itself. A value of '
     '1000 means that there are no other centrals within at least 10 r200.'),
]
quants_gps = [
    ('Coordinates', 'Centre', np.float32, 3,
     'Coordinates of each galaxy [pMpc]. The first index specifies the '
     'galaxy, the second index the snapshot, and the third the x/y/z axis.'),
    ('Velocities', 'Velocity', np.float32, 3,
     'Velocity of each galaxy [km/s]. The first index specifies the galaxy, '
     'the second index the snapshot, and the third the x/y/z axis.'),
]

# (ii-a) From Subfind FOF catalogues:
quants_subfind_fof = [ 
    ('M200m', 'Group_M_Mean200', np.float32, 2, 'Mass within the radius '
     'enclosing 200 times the mean density [M_Sun]. For satellites, this '
     'refers to their central halo.'),
    ('R200m', 'Group_R_Mean200', np.float32, 2, 'Radius enclosing a density '
     'of 200 times the mean density [pMpc]. For satellites, this refers to '
     'their central halo.'),
]

quants_subfind_fof = []

# (ii-b) From Subfind Subhalo catalogues:
quants_subfind_sub = [
]

# (iii) Derived quantities (not already in a catalogue):
quants_snapshot = [
    ('CentralGalaxies', 'find_centrals', np.int32, 2,
     'Pointer to the central galaxy in this catalogue. For centrals, this '
     'points to itself.'),
    ('CarrierGalaxies', 'find_carriers', np.int32, 2,
     'Pointer to the galaxy of which this galaxy has become part through a '
     'merger. For galaxies that are still alive, this points to themselves.'),
]

quants_hydrangea_only = ['RRelNearest', 'RRelMean', 'MHNeutral', 'MHI']
quants_eagle_only = ['Volume', 'Model', 'Resolution', 'Boxsize']

quants_to_interpolate = [
    'MStar30kpc', 'MStar', 'MStarInitial', 'MSubhalo', 'VMax']
quants_to_delog = [
    'SubhaloMasses', 'M200c', 'StellarMasses', 'StellarMasses30kpc',
    'InitialStellarMasses', 'GasMasses', 'GasMasses30kpc',
    'StarFormationRates', 'StarFormationRates30kpc'
]

add_interpolated_positions = True

# ===========================================================================

def load_fof_quantities(fof_quants, sim_data):
    if len(fof_quants) == 0:
        return
        
    n_gal = sim_data['SubhaloIndices'].shape[0]
    n_snaps = sim_data['SubhaloIndices'].shape[1]

    for quant in fof_quants:
        name = quant[0]
        sim_data[name] = np.zeros((n_gal, n_snaps), dtype=quant[2])
    
    for isnap in range(n_snaps):
        sub_file = sim.get_subfind_file(isnap)
        sub = hy.SplitFile(sub_file, 'Subhalo')
        fof = hy.SplitFile(sub_file, 'FOF')
        
        sub_sel = sim_data['SubhaloIndices'][:, isnap]
        fof_sel = sub.GroupNumber[sub_sel] - 1

        ind_bad = np.nonzero(sub_sel < 0)[0]
        for quant in fof_quants:
            name = quant[0]
            source = quant[1]
            sim_data[name][:, isnap] = fof.read_data(source)[fof_sel]
            sim_data[name][ind_bad, isnap] = -100.0
                     
def find_centrals(sim_data):
    fgt_loc = hldir + 'FullGalaxyTables.hdf5'
    cengals = hy.hdf5.read_data(fgt_loc, 'CenGal')[sim_data['GalaxyIDs'], :]

    max_id = np.max(sim_data['GalaxyIDs'])
    n_gal = len(sim_data['GalaxyIDs'])
    rev_list = np.zeros(max_id+1) - 1
    rev_list[sim_data['GalaxyIDs']] = np.arange(n_gal)
    sim_data['CentralGalaxies'] = rev_list[cengals] + gal_offset
    ind_bad = np.nonzero(cengals < 0)
    sim_data['CentralGalaxies'][ind_bad] = cengals[ind_bad]


def find_carriers(sim_data):
    spider_loc = hldir + 'SpiderwebTables.hdf5'
    mergelist = hy.hdf5.read_data(spider_loc, 'MergeList')[
        sim_data['GalaxyIDs'], :]
    gal_ids = sim_data['GalaxyIDs']
    
    max_id = np.max(mergelist)
    #max_id2 = np.max(gal_ids)
    
    n_gal = len(sim_data['GalaxyIDs'])
    rev_list = np.zeros(max_id + 1) - 1
    rev_list[sim_data['GalaxyIDs']] = np.arange(n_gal)
    sim_data['CarrierGalaxies'] = rev_list[mergelist] + gal_offset
    ind_bad = np.nonzero(mergelist < 0)
    sim_data['CarrierGalaxies'][ind_bad] = mergelist[ind_bad]

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
    'Simulations': np.zeros(0, dtype=np.int8),
    'GalaxyIDs': np.zeros(0, dtype=np.int32)
}
comments = {
    'Simulations': 'Simulation index for each galaxy.',
    'GalaxyIDs': 'ID of each galaxy in its simulation.',
}

for quant_list in quant_lists:
    for quant in quant_list:
        name = quant[0]
        if name in full_data:
            raise KeyError(f"Data set {name} appears to be requested twice!")

        if quant[3] == 3:
            shape = (0, 30, 3)
        elif quant[3] == 2:
            shape = (0, 30)
        elif quant[3] == 1:
            shape = (0)

        full_data[name] = np.zeros(shape, dtype=quant[2])
        comments[name] = quant[4]

if add_interpolated_positions:
    full_data['InterpolatedCoordinates'] = np.zeros((0, 1350, 3), np.float32)
    comments['InterpolatedCoordinates'] = (
        'Coordinates of the galaxy interpolated to 10 Myr time steps [pMpc].')

gal_offset = 0

# Now loop through individual simulations
for isim in range(n_sim):

    if simname == 'Hydrangea':
        sim = hy.Simulation(isim)
        hldir = sim.high_level_dir
    else:
        sim = None
        hldir = f'{basedir_hl}{boxvec[isim]}/{modelvec[isim]}/highlev/'
        
    if not os.path.exists(hldir):
        continue

    if simname == 'Hydrangea': 
        print("")
        print("=============================")
        print(f"Processing simulation CE-{isim}")
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


    gal_sel_a = np.nonzero(
        (contFlag == 0) & (spectreFlag == 0) &
        (
            (logMsubMax >= min_logmsub_peak) |
            (logMstarMax >= min_logmstar_peak)
        )
    )[0]

    spider_loc = hldir + 'SpiderwebTables.hdf5'
    mergelist = hy.hdf5.read_data(spider_loc, 'MergeList')
    gal_sel = np.unique(mergelist[gal_sel_a, :])
    gal_sel = gal_sel[gal_sel >= 0]

    ngal_sel = len(gal_sel)
    print(f"In simulation {isim}, there are {ngal_sel} selected galaxies.")

    # Now extract data... FGT/GPS first:
    sim_data = {
        'GalaxyIDs': gal_sel,
        'Simulations': np.zeros(ngal_sel, dtype=np.int8) + isim,
    }    
    for quant in quants_fgt:
        name = quant[0]
        source = quant[1]
        sim_data[name] = hy.hdf5.read_data(fgtloc, source, read_index=gal_sel)
    for quant in quants_gps:
        name = quant[0]
        source = quant[1]
        sim_data[name] = hy.hdf5.read_data(posloc, source, read_index=gal_sel)

    # Then Subfind FOF:
    load_fof_quantities(quants_subfind_fof, sim_data)

    # Special quantities
    for quant in quants_snapshot:
        func_name = quant[1]
        globals()[func_name](sim_data)

    # Interpolate over missing entries
    for iigal, igal in enumerate(gal_sel):        

        # Find snapshots in which the galaxy is hidden. If there are none: easy
        shi = sim_data['SubhaloIndices'][iigal, :]
        ind_hidden = np.nonzero(shi == -9)[0]
        if len(ind_hidden) == 0:
            continue

        # If we know one thing about hidden galaxies, then that they must be
        # satellites at these snapshots -- so mark them as such.        
        if 'SatelliteFlags' in sim_data:
            sim_data['SatelliteFlags'][iigal, ind_hidden] = 1
        if 'EverSatelliteFlags' in sim_data:
            sim_data['EverSatelliteFlags'][iigal] = 1

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
                    snap_time[ind_ok], sim_data[quant][iigal, ind_ok],
                    kind=kind, assume_sorted=True, fill_value='extrapolate'
                )
                sim_data[quant][iigal, ind_hidden] = csi(snap_time[ind_hidden])
            # If the data set is not found, just ignore it
            except KeyError:
                pass

    # De-log quantities
    for quant in quants_to_delog:
        if quant not in sim_data:
            continue

        sim_data[quant] = 10.0**sim_data[quant]
        ind_neg = np.nonzero(sim_data['SubhaloIndices'] < 0)
        sim_data[quant][ind_neg] = -999

    # Add interpolated positions, if desired
    if add_interpolated_positions:
        inter_loc = hldir + 'GalaxyCoordinates10Myr.hdf5'
        inter_inds = hy.hdf5.read_data(inter_loc, 'GalaxyRevIndex')[gal_sel]
        ind_bad = np.nonzero(inter_inds < 0)[0]
        inter_pos = hy.hdf5.read_data(inter_loc, 'InterpolatedPositions')[
            inter_inds, ...]
        inter_pos[ind_bad, ...] = -999
        sim_data['InterpolatedCoordinates'] = np.transpose(
            inter_pos, (0, 2, 1))
        sim_data['InterpolationTimes'] = hy.hdf5.read_data(
            inter_loc, 'InterpolationTimes')
        
    # Append this simulation's data to the full output list
    gal_offset += ngal_sel
    for quant in full_data:
        full_data[quant] = np.concatenate((full_data[quant], sim_data[quant]))
        if full_data[quant].shape[0] != gal_offset:
            raise ValueError("Inconsistent data set lengths!")

# Done processing simulations, we now have the full output list. Write it.

if 'InterpolationTimes' in sim_data:
    full_data['InterpolationTimes'] = sim_data['InterpolationTimes']
    comments['InterpolationTimes'] = (
        'Cosmic time corresponding to the interpolation points in '
        'InterpolatedCoordinates [Gyr].'
    )
full_data['SnapshotRedshifts'] = snap_zred
comments['SnapshotRedshifts'] = 'Redshift corresponding to each snapshot.'

full_data['SnapshotTimes'] = snap_time
comments['SnapshotTimes'] = 'Cosmic age corresponding to each snapshot [Gyr].'

if os.path.isfile(outloc):
    os.rename(outloc, outloc + '.old')
with h5.File(outloc, 'w') as f:
    for quant in full_data.keys():
        dset = f.create_dataset(
            quant, full_data[quant].shape, dtype=full_data[quant].dtype,
            compression='gzip'
        )
        dset[...] = full_data[quant]
        dset.attrs.create('Description', np.string_(comments[quant]))

print("Done!")
