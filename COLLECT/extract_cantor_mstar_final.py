"""
Small program to extract final CANTOR masses for galaxies in 
accretion catalogue.

Started 19-Sep-2019
"""

import numpy as np
import yb_utils as yb

catloc = ('/freya/ptmp/mpa/ybahe/HYDRANGEA/DISRUPTION/'
          'AccretionCatalogue_1Oct18_HYDRO.hdf5')

outloc = ('/virgo/scratch/ybahe/HYDRANGEA/RESULTS/ICL/'
          'StellarMassLoss_19Sep19.hdf5')

cat_sim = yb.read_hdf5(catloc, 'Sim')
cat_gal = yb.read_hdf5(catloc, 'Galaxy')
ncat = len(cat_sim)

basedir = '/virgo/simulations/Hydrangea/10r200/'

mstar_init_peak = np.zeros(ncat, dtype = np.float32)-1
mstar_init_z0 = np.zeros(ncat, dtype = np.float32)-1
mstar_final_cantor = np.zeros(ncat, dtype = np.float32)-1
flag_cantor_alive = np.zeros(ncat, dtype = np.int8)

for isim in range(30):

    print("Processing simulation {:d}..." .format(isim))
    ind_thissim = np.nonzero(cat_sim == isim)[0]
    n_thissim = len(ind_thissim)
    if n_thissim == 0: continue

    gal_thissim = cat_gal[ind_thissim]

    fgtloc = basedir + 'CE-{:d}/HYDRO/highlev/FullGalaxyTables.hdf5' .format(isim)
    cantorloc = basedir + 'CE-{:d}/HYDRO/highlev/CantorCatalogue.hdf5' .format(isim)

    # Load peak and final INITIAL stellar mass from FGT:
    mstar_init_peak[ind_thissim] = yb.read_hdf5(
        fgtloc, 'Full/MstarInit')[gal_thissim]
    mstar_init_z0[ind_thissim] = yb.read_hdf5(
        fgtloc, 'MstarInit')[gal_thissim, -1]
    
    # Load final REAL stellar mass from Cantor:
    cshi = yb.read_hdf5(cantorloc, 'SubhaloIndex')[gal_thissim, -1]
    ind_cantor_alive = np.nonzero(cshi >= 0)[0]

    flag_cantor_alive[ind_thissim[ind_cantor_alive]] = 1
    mstar_final_cantor[ind_thissim[ind_cantor_alive]] = np.log10(yb.read_hdf5(
        cantorloc, 'Snapshot_029/Subhalo/MassType')[cshi[ind_cantor_alive], 4])+10.0

yb.write_hdf5(cat_sim, outloc, 'Sim', new = True)
yb.write_hdf5(cat_gal, outloc, 'Galaxy')
yb.write_hdf5(mstar_init_peak, outloc, 'MstarInit_Peak')
yb.write_hdf5(mstar_init_z0, outloc, 'MstarInit_z0')
yb.write_hdf5(mstar_final_cantor, outloc, 'MstarCantor_z0')
yb.write_hdf5(flag_cantor_alive, outloc, 'FlagCantorAlive_z0')

print("Done!")
