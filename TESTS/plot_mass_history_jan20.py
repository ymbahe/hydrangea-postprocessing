"""
Simple program to extract the mass evolution of one galaxy,
from Subfind and Cantor.
"""

import numpy as np
import sim_tools as st
import yb_utils as yb
import hydrangea_tools as ht
from pdb import set_trace

rundir = '/virgo/simulations/Hydrangea/10r200/CE-28/HYDRO/'
hldir = rundir + 'highlev/'

fgtloc = hldir + 'FullGalaxyTables.hdf5'
posloc = hldir + 'GalaxyPositionsSnap.hdf5'
fineloc = ht.clone_dir(hldir) + 'GalaxyCoordinates10Myr_May18.hdf5'

spiderloc = hldir + 'SpiderwebTables.hdf5'

cantorloc = hldir + '/Cantor/GalaxyTables.hdf5'
outloc = '/virgo/scratch/ybahe/TESTS/GalaxyMassEvol_CE-28G-G3272.hdf5'

igal = 3272
cenGalID = 474#None#10589#39902

mass_sf = yb.read_hdf5(fgtloc, 'Msub')[igal, :]
mstar_sf = yb.read_hdf5(fgtloc, 'Mstar')[igal, :]
mstar30_sf = yb.read_hdf5(fgtloc, 'Mstar30kpc')[igal, :]
mgas_sf = yb.read_hdf5(fgtloc, 'MGas')[igal, :]
mdm_sf = yb.read_hdf5(fgtloc, 'MDM')[igal, :]

fineID = yb.read_hdf5(fineloc, 'GalaxyRevIndex')
timeFine = yb.read_hdf5(fineloc, 'InterpolationTimes')
finePos = yb.read_hdf5(fineloc, 'InterpolatedPositions')

shi_cantor = yb.read_hdf5(cantorloc, 'SubhaloIndex')[igal, :]

mass_cantor = np.zeros(30)
mdm_cantor = np.zeros(30)
mstar_cantor = np.zeros(30)
mgas_cantor = np.zeros(30)
vd_stars_cantor = np.zeros(30)
vd_dm_cantor = np.zeros(30)

cenGal = yb.read_hdf5(fgtloc, 'CenGal')
cenGalX = yb.read_hdf5(ht.clone_dir(hldir) + 'RegularizedCentrals.hdf5', 'CenGal_Regularized')


if cenGalID is None:
    cenGalID = cenGalX[igal, 29]

if cenGalID < 0:
    cenGalID = yb.read_hdf5(spiderloc, 'MergeList')[igal, 29]

pos_all = yb.read_hdf5(posloc, 'Centre')

clPos = pos_all[cenGalID, :, :]
rad = np.linalg.norm(pos_all[igal, ...] - clPos, axis = 1)


clPosFine = finePos[fineID[cenGalID], :, :]
galPosFine = finePos[fineID[igal], :, :]

radFine = np.linalg.norm(galPosFine - clPosFine, axis = 0)

time = ht.snap_times(conv = 'age')

kappa_co10 = np.zeros(30)
kappa_co30 = np.zeros(30)
phi = np.zeros(30)


for isnap in range(30):
    print(shi_cantor[isnap])
    cantorsnap = hldir + '/Cantor/Cantor_{:03d}.hdf5' .format(isnap)
    if shi_cantor[isnap] >= 0:
        mass_cantor[isnap] = np.log10(yb.read_hdf5(cantorsnap, 'Subhalo/Mass')[shi_cantor[isnap]]) + 10.0
        mstar_cantor[isnap] = np.log10(yb.read_hdf5(cantorsnap, 'Subhalo/MassType')[shi_cantor[isnap], 4]) + 10.0
        mdm_cantor[isnap] = np.log10(yb.read_hdf5(cantorsnap, 'Subhalo/MassType')[shi_cantor[isnap], 1]) + 10.0
        mgas_cantor[isnap] = np.log10(yb.read_hdf5(cantorsnap, 'Subhalo/MassType')[shi_cantor[isnap], 0]) + 10.0
        vd_stars_cantor[isnap] = yb.read_hdf5(cantorsnap, 'Subhalo/VelocityDispersion_Stars')[shi_cantor[isnap]]

        vd_dm_cantor[isnap] = yb.read_hdf5(cantorsnap, 'Subhalo/VelocityDispersion_DM')[shi_cantor[isnap]]

        extra_id = yb.read_hdf5(cantorsnap, 'Subhalo/Extra/ExtraIDs')[shi_cantor[isnap]]
        if extra_id >= 0:
            kappa_co = yb.read_hdf5(cantorsnap, 'Subhalo/Extra/Stars/KappaCo')[extra_id, :]
            kappa_co10[isnap] = kappa_co[0]
            kappa_co30[isnap] = kappa_co[1]

            #phi[isnap] = yb.read_hdf5(cantorloc, 'Snapshot_' + str(isnap).zfill(3) + '/Subhalo/Centre/StellarShape')[shi_cantor[isnap]]
        

yb.write_hdf5(mass_cantor, outloc, 'MassCantor', new = True)
yb.write_hdf5(mstar_cantor, outloc, 'MstarCantor')
yb.write_hdf5(mdm_cantor, outloc, 'MDMCantor')
yb.write_hdf5(mass_sf, outloc, 'MassSF')
yb.write_hdf5(mstar_sf, outloc, 'MstarSF')
yb.write_hdf5(mstar30_sf, outloc, 'Mstar30SF')
yb.write_hdf5(mdm_sf, outloc, 'MDMSF')
yb.write_hdf5(rad, outloc, 'Radius')
yb.write_hdf5(time, outloc, 'Time')
yb.write_hdf5(radFine, outloc, 'RadiusFine')
yb.write_hdf5(timeFine, outloc, 'TimeFine')
yb.write_hdf5(mgas_sf, outloc, 'MgasSF')
yb.write_hdf5(mgas_cantor, outloc, 'MgasCantor')
yb.write_hdf5(kappa_co10, outloc, 'KappaCo10')
yb.write_hdf5(kappa_co30, outloc, 'KappaCo30')
yb.write_hdf5(phi, outloc, 'Phi')
yb.write_hdf5(vd_stars_cantor, outloc, 'VelDisp_Stars')
yb.write_hdf5(vd_dm_cantor, outloc, 'VelDisp_DM')


